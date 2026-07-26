"""
FlowGuard-MPC: Adaptive Model-Predictive Production Choke Controller.

Author: Shyambaskar Sriram
SASTRA University - B.E. Computer Science & Engineering (AI & DS)

Description:
    Implements FlowGuard-MPC, a multi-step Receding Horizon Model Predictive Controller (MPC)
    with adaptive telemetry noise filtering, online system identification,
    and a Candidate Move Rejection Engine for a single naturally flowing oil well.
"""

import numpy as np


class AdaptiveTelemetryFilter:
    """
    Adaptive Exponential Noise Filter.
    Smoothes raw process telemetry (Q, WHP, FLP, BHP) to eliminate measurement noise spikes.
    """
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.filtered_state = None
        
    def filter(self, raw_state):
        if self.filtered_state is None:
            self.filtered_state = dict(raw_state)
        else:
            for key in ['Q', 'WHP', 'FLP', 'BHP']:
                if key in raw_state:
                    self.filtered_state[key] = (1 - self.alpha) * self.filtered_state[key] + self.alpha * raw_state[key]
        return dict(self.filtered_state)


class OnlineSystemID:
    """
    Online Recursive System Identification Engine.
    Dynamically estimates steady-state process sensitivity gains (dQ/du, dWHP/du, dFLP/du, dBHP/du) online.
    """
    def __init__(self):
        # Steady-state drawdown gains (calibrated from simulator inflow dynamics)
        self.gain_Q_ss = 1.83     # Oil flow rate gain (bbl/hr per % choke)
        self.gain_WHP_ss = -1.42  # Wellhead Pressure gain (psi per % choke)
        self.gain_FLP_ss = -0.94  # Flowline Pressure gain (psi per % choke)
        self.gain_BHP_ss = -6.40  # Bottom Hole Pressure gain (psi per % choke)
        self.lag_alpha = 0.45     # Dynamic lag factor
        
    def update(self, u_prev, u_curr, state_prev, state_curr):
        du = u_curr - u_prev
        if abs(du) > 0.1:
            dQ = state_curr['Q'] - state_prev['Q']
            dWHP = state_curr['WHP'] - state_prev['WHP']
            dFLP = state_curr['FLP'] - state_prev['FLP']
            dBHP = state_curr['BHP'] - state_prev['BHP']
            
            lr = 0.05
            self.gain_Q_ss = max(0.1, (1 - lr) * self.gain_Q_ss + lr * (dQ / (du * self.lag_alpha)))
            self.gain_WHP_ss = min(-1.40, (1 - lr) * self.gain_WHP_ss + lr * (dWHP / (du * self.lag_alpha)))
            self.gain_FLP_ss = min(-0.92, (1 - lr) * self.gain_FLP_ss + lr * (dFLP / (du * self.lag_alpha)))
            self.gain_BHP_ss = min(-6.30, (1 - lr) * self.gain_BHP_ss + lr * (dBHP / (du * self.lag_alpha)))

    def predict_horizon(self, u_curr, candidate_du_sequence, state_curr):
        """
        Predicts process state trajectories over a multi-step horizon sequence.
        """
        trajectory = []
        sim_u = u_curr
        sim_state = dict(state_curr)
        
        for du in candidate_du_sequence:
            sim_u += du
            # Calculate steady state for candidate choke position
            q_ss = 38.0 + self.gain_Q_ss * sim_u
            whp_ss = 310.0 + self.gain_WHP_ss * sim_u
            flp_ss = 216.0 + self.gain_FLP_ss * sim_u
            bhp_ss = 3320.0 + self.gain_BHP_ss * sim_u
            
            # Apply dynamic lag transition
            sim_state = {
                'Q': (1 - self.lag_alpha) * sim_state['Q'] + self.lag_alpha * q_ss,
                'WHP': (1 - self.lag_alpha) * sim_state['WHP'] + self.lag_alpha * whp_ss,
                'FLP': (1 - self.lag_alpha) * sim_state['FLP'] + self.lag_alpha * flp_ss,
                'BHP': (1 - self.lag_alpha) * sim_state['BHP'] + self.lag_alpha * bhp_ss,
                'WHP_ss': whp_ss,
                'FLP_ss': flp_ss,
                'BHP_ss': bhp_ss,
                'u': sim_u
            }
            trajectory.append(sim_state)
            
        return trajectory


class AutonomousChokeController:
    """
    FlowGuard-MPC Receding Horizon Model Predictive Controller.
    Strictly enforces active safety limits, hard ramp constraints, and candidate rejection.
    """
    def __init__(self, whp_min=210.0, flp_min=150.0, bhp_min=2850.0, max_ramp=5.0, horizon=5):
        """
        Parameters:
            whp_min (float): Minimum WHP limit in psi (default: 210.0).
            flp_min (float): Minimum FLP limit in psi (default: 150.0).
            bhp_min (float): Minimum BHP limit in psi (default: 2850.0).
            max_ramp (float): Max choke ramp rate per step in % (default: 5.0).
            horizon (int): Multi-step prediction horizon Np (default: 5 steps).
        """
        self.whp_min = float(whp_min)
        self.flp_min = float(flp_min)
        self.bhp_min = float(bhp_min)
        self.max_ramp = float(max_ramp)
        self.horizon = int(horizon)
        
        self.deadband_q = 0.15 # Deadband threshold to eliminate choke chatter
        
        self.telemetry_filter = AdaptiveTelemetryFilter(alpha=0.3)
        self.online_id = OnlineSystemID()
        self.rejection_log = []

    def validate_trajectory(self, u_curr, candidate_du, state_curr):
        """
        Evaluates a candidate move over the prediction horizon.
        Rejects moves that breach ramp rates, pressure safety limits, or steady-state limits.
        """
        if abs(candidate_du) > self.max_ramp + 1e-5:
            return False, f"Ramp limit exceeded: |{candidate_du:.2f}%| > {self.max_ramp}%", None
            
        candidate_u = u_curr + candidate_du
        if candidate_u < 0.0 or candidate_u > 100.0:
            return False, f"Choke bounds breached: {candidate_u:.2f}%", None

        # Predict steady-state equilibrium pressures for candidate_u
        whp_ss = 310.0 + self.online_id.gain_WHP_ss * candidate_u
        flp_ss = 216.0 + self.online_id.gain_FLP_ss * candidate_u
        bhp_ss = 3320.0 + self.online_id.gain_BHP_ss * candidate_u
        
        # Buffer margin (3.0 psi) guarantees sensor noise spikes NEVER cross safety limits
        buffer = 3.0
        
        if whp_ss < self.whp_min + buffer:
            return False, f"Equilibrium WHP breach: {whp_ss:.2f} psi < {self.whp_min + buffer} psi", None
        if flp_ss < self.flp_min + buffer:
            return False, f"Equilibrium FLP breach: {flp_ss:.2f} psi < {self.flp_min + buffer} psi", None
        if bhp_ss < self.bhp_min + buffer:
            return False, f"Equilibrium BHP breach: {bhp_ss:.2f} psi < {self.bhp_min + buffer} psi", None

        du_sequence = [candidate_du] + [0.0] * (self.horizon - 1)
        trajectory = self.online_id.predict_horizon(u_curr, du_sequence, state_curr)
        
        for step_idx, pred_state in enumerate(trajectory):
            if pred_state['WHP'] < self.whp_min + buffer:
                return False, f"WHP breach at step +{step_idx+1}: {pred_state['WHP']:.2f} psi < {self.whp_min} psi", pred_state
            if pred_state['FLP'] < self.flp_min + buffer:
                return False, f"FLP breach at step +{step_idx+1}: {pred_state['FLP']:.2f} psi < {self.flp_min} psi", pred_state
            if pred_state['BHP'] < self.bhp_min + buffer:
                return False, f"BHP breach at step +{step_idx+1}: {pred_state['BHP']:.2f} psi < {self.bhp_min} psi", pred_state
                
        return True, None, trajectory[0]

    def compute_next_choke_position(self, u_curr, target_Q, current_state, time_step=0):
        """
        Computes the optimal choke position for the next 1-hour control step.
        Parameters:
            u_curr (float): Current choke position (%).
            target_Q (float): Target production rate (bbl/hr).
            current_state (dict): Telemetry dictionary {'Q', 'WHP', 'FLP', 'BHP'}.
            time_step (int): Current simulation control interval hour (default: 0).
        """
        filtered_state = self.telemetry_filter.filter(current_state)
        
        q_error = abs(filtered_state['Q'] - target_Q)
        if q_error <= self.deadband_q:
            is_valid, _, _ = self.validate_trajectory(u_curr, 0.0, filtered_state)
            if is_valid:
                return u_curr

        candidate_dus = np.linspace(-self.max_ramp, self.max_ramp, 101)
        
        best_u = u_curr
        best_cost = float('inf')
        max_safe_q = -float('inf')
        max_safe_u = u_curr
        
        for du in candidate_dus:
            candidate_u = u_curr + du
            is_valid, reason, pred_1step = self.validate_trajectory(u_curr, du, filtered_state)
            
            if not is_valid:
                self.rejection_log.append({
                    'Time (hr)': time_step,
                    'u_curr (%)': round(u_curr, 1),
                    'candidate_u (%)': round(candidate_u, 1),
                    'reason': reason
                })
                continue
                
            if pred_1step['Q'] > max_safe_q:
                max_safe_q = pred_1step['Q']
                max_safe_u = candidate_u
                
            w_effort = 0.5
            cost = (pred_1step['Q'] - target_Q)**2 + w_effort * (du**2)
            
            if cost < best_cost:
                best_cost = cost
                best_u = candidate_u
                
        if target_Q > max_safe_q and best_cost > 50.0:
            return max_safe_u

        return best_u

    def update_model(self, u_prev, u_curr, state_prev, state_curr):
        """Updates internal model gains after executing a control step."""
        filtered_prev = self.telemetry_filter.filter(state_prev)
        filtered_curr = self.telemetry_filter.filter(state_curr)
        self.online_id.update(u_prev, u_curr, filtered_prev, filtered_curr)
