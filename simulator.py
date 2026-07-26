"""
Dynamic Oil Well Simulator for Offline Testing & Verification.

Author: Shyambaskar Sriram
SASTRA University - B.E. Computer Science & Engineering (AI & DS)

Description:
    Simulates physical fluid flow and dynamic pressure responses of a single naturally flowing oil well.
    Implements Honeywell's official evaluation API:
        Q, WHP, FLP, BHP = simulator.step(choke_position)
"""

import numpy as np

class OilWellSimulator:
    """
    Physical Well Simulator Class.
    
    Models steady-state inflow performance (IPR), dynamic pressure drawdown,
    and first-order dynamic lag responses for oil rate and well pressures.
    """
    def __init__(self, initial_choke=30.0, noise_std=0.3, seed=42):
        """
        Initializes well state variables and random number generator for sensor noise.
        
        Parameters:
            initial_choke (float): Initial choke position percentage (default: 30.0%).
            noise_std (float): Standard deviation of synthetic measurement noise.
            seed (int): Random seed for reproducible simulation runs.
        """
        self.rng = np.random.RandomState(seed)
        self.noise_std = noise_std
        
        # State variable: current choke position (%)
        self.choke_pos = float(initial_choke)
        
        # Initialize active process variables at 30% baseline
        self.Q = 90.0     # Oil Flow Rate (bbl/hr)
        self.WHP = 250.0  # Wellhead Pressure (psi)
        self.FLP = 180.0  # Flowline Pressure (psi)
        self.BHP = 3000.0 # Bottom Hole Pressure (psi)
        
        # Informational variables (Page 3 of problem statement)
        self.WHT = 120.0  # Wellhead Temperature (°F)
        self.AP = 450.0   # Annulus Pressure (psi)
        
        # Physical actuator constraints
        self.max_ramp_rate = 5.0 # Maximum allowed choke move per 1-hour step (%)

    def _calc_steady_state(self, u):
        """
        Calculates theoretical steady-state outputs for a choke opening u (%).
        Based on fluid inflow performance and valve flow coefficient equations.
        
        Parameters:
            u (float): Choke opening percentage in [0, 100].
            
        Returns:
            tuple: Steady-state values (Q_ss, WHP_ss, FLP_ss, BHP_ss, WHT_ss, AP_ss).
        """
        # Linearized inflow and outflow relationships fitted from dataset
        Q_ss = 38.0 + 1.83 * u       # Flow rate increases linearly with choke opening
        WHP_ss = 310.0 - 1.42 * u    # Wellhead pressure drops as choke opens (drawdown)
        FLP_ss = 216.0 - 0.94 * u    # Flowline pressure decreases with higher throughput
        BHP_ss = 3320.0 - 6.40 * u   # Reservoir bottom hole pressure drops with higher drawdown
        
        # Informational variables
        WHT_ss = 100.0 + 0.50 * u    # Wellhead temperature increases slightly with higher velocity
        AP_ss = 480.0 - 0.80 * u     # Annulus pressure drops slightly
        
        return Q_ss, WHP_ss, FLP_ss, BHP_ss, WHT_ss, AP_ss

    def step(self, candidate_choke_position):
        """
        Executes one control interval (Ts = 1 hour).
        
        Enforces physical choke movement ramp limits, updates process state using a
        first-order dynamic lag model, and adds sensor measurement noise.
        
        Parameters:
            candidate_choke_position (float): Requested next choke position percentage.
            
        Returns:
            tuple: (Q, WHP, FLP, BHP) process outputs after 1 hour.
        """
        # Step 1: Enforce physical actuator ramp rate limit (|du| <= 5% per hour)
        du = candidate_choke_position - self.choke_pos
        du_clamped = np.clip(du, -self.max_ramp_rate, self.max_ramp_rate)
        
        # Step 2: Apply position update within valve bounds [0%, 100%]
        self.choke_pos = np.clip(self.choke_pos + du_clamped, 0.0, 100.0)
        
        # Step 3: Compute target steady-state outputs for new choke opening
        Q_ss, WHP_ss, FLP_ss, BHP_ss, WHT_ss, AP_ss = self._calc_steady_state(self.choke_pos)
        
        # Step 4: Apply dynamic first-order lag response (alpha = 0.45 dynamic transition speed)
        alpha = 0.45
        
        # Update state telemetry with dynamic lag + Gaussian sensor noise
        self.Q = (1 - alpha) * self.Q + alpha * Q_ss + self.rng.normal(0, self.noise_std * 0.5)
        self.WHP = (1 - alpha) * self.WHP + alpha * WHP_ss + self.rng.normal(0, self.noise_std)
        self.FLP = (1 - alpha) * self.FLP + alpha * FLP_ss + self.rng.normal(0, self.noise_std * 0.8)
        self.BHP = (1 - alpha) * self.BHP + alpha * BHP_ss + self.rng.normal(0, self.noise_std * 2.0)
        
        # Update informational state telemetry
        self.WHT = (1 - alpha) * self.WHT + alpha * WHT_ss
        self.AP = (1 - alpha) * self.AP + alpha * AP_ss
        
        # Return measured telemetry matching Honeywell evaluation API signature
        return self.Q, self.WHP, self.FLP, self.BHP
