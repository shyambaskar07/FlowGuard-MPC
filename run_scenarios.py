"""
Demonstration Scenarios Execution Engine for FlowGuard-MPC.

Author: Shyambaskar Sriram
SASTRA University - B.E. Computer Science & Engineering (AI & DS)

Description:
    Runs the three required Honeywell hackathon scenarios:
        - Scenario A: Startup to Target
        - Scenario B: Target Tracking
        - Scenario C: Infeasible Target (Max Safe Production Settling)
    Generates publication-grade 300 DPI 5-panel trend plots for each scenario.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

from simulator import OilWellSimulator
from mpc_controller import AutonomousChokeController

def run_scenario(scenario_name, target_profile, timesteps=50, initial_choke=30.0):
    """
    Executes a closed-loop scenario simulation over specified timesteps.
    
    Parameters:
        scenario_name (str): Identifier for saving plots and output logging.
        target_profile (dict/float/list): Target flow rate profile across time.
        timesteps (int): Total simulation hours.
        initial_choke (float): Initial choke position %.
        
    Returns:
        tuple: (pd.DataFrame telemetry log, list rejection_log)
    """
    sim = OilWellSimulator(initial_choke=initial_choke, seed=42)
    controller = AutonomousChokeController(whp_min=210.0, flp_min=150.0, bhp_min=2850.0, max_ramp=5.0)
    
    history = {'Time': [], 'Target_Q': [], 'Actual_Q': [], 'WHP': [], 'FLP': [], 'BHP': [], 'Choke': []}
    
    current_u = initial_choke
    current_state = {'Q': sim.Q, 'WHP': sim.WHP, 'FLP': sim.FLP, 'BHP': sim.BHP}
    
    for t in range(timesteps):
        if isinstance(target_profile, dict):
            target_q = target_profile.get(t, list(target_profile.values())[-1])
        elif isinstance(target_profile, (list, tuple)):
            target_q = target_profile[t] if t < len(target_profile) else target_profile[-1]
        else:
            target_q = float(target_profile)
            
        # Step 1: Controller computes optimal safe choke position for next 1-hour step
        next_u = controller.compute_next_choke_position(current_u, target_q, current_state, time_step=t)
        
        # Step 2: Physical well simulator executes choke move (Honeywell API Signature)
        actual_q, whp, flp, bhp = sim.step(next_u)
        next_state = {'Q': actual_q, 'WHP': whp, 'FLP': flp, 'BHP': bhp}
        
        # Step 3: Controller updates online gain identification model
        controller.update_model(current_u, next_u, current_state, next_state)
        
        # Log telemetry
        history['Time'].append(t)
        history['Target_Q'].append(target_q)
        history['Actual_Q'].append(actual_q)
        history['WHP'].append(whp)
        history['FLP'].append(flp)
        history['BHP'].append(bhp)
        history['Choke'].append(current_u)
        
        current_u = next_u
        current_state = next_state
        
    df = pd.DataFrame(history)
    plot_results(df, scenario_name, controller.rejection_log)
    return df, controller.rejection_log

# Alias function for notebook compatibility
run_simulation = run_scenario

def plot_results(df, scenario_name, rejection_log=None):
    """
    Renders publication-grade 5-panel trend plots for process telemetry.
    """
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"FlowGuard-MPC Telemetry Trace: {scenario_name}", fontsize=14, fontweight='bold')
    
    # Panel 1: Oil Flow Rate Q
    axs[0].plot(df['Time'], df['Target_Q'], 'r--', label='Target Q (bbl/hr)', linewidth=1.5)
    axs[0].plot(df['Time'], df['Actual_Q'], 'b-', label='Actual Q (bbl/hr)', linewidth=2.0)
    axs[0].set_ylabel('Q (bbl/hr)')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='upper right')
    
    # Panel 2: Wellhead Pressure WHP
    axs[1].plot(df['Time'], df['WHP'], 'g-', label='WHP (psi)', linewidth=2.0)
    axs[1].axhline(y=210.0, color='r', linestyle=':', label='WHP Limit (210 psi)')
    axs[1].set_ylabel('WHP (psi)')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc='upper right')
    
    # Panel 3: Flowline Pressure FLP
    axs[2].plot(df['Time'], df['FLP'], 'm-', label='FLP (psi)', linewidth=2.0)
    axs[2].axhline(y=150.0, color='r', linestyle=':', label='FLP Limit (150 psi)')
    axs[2].set_ylabel('FLP (psi)')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc='upper right')
    
    # Panel 4: Bottom Hole Pressure BHP
    axs[3].plot(df['Time'], df['BHP'], 'c-', label='BHP (psi)', linewidth=2.0)
    axs[3].axhline(y=2850.0, color='r', linestyle=':', label='BHP Limit (2850 psi)')
    axs[3].set_ylabel('BHP (psi)')
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc='upper right')
    
    # Panel 5: Choke Position u
    axs[4].plot(df['Time'], df['Choke'], 'k-', label='Choke Opening (%)', linewidth=2.0)
    axs[4].set_ylabel('Choke (%)')
    axs[4].set_xlabel('Time (Hours)')
    axs[4].grid(True, alpha=0.3)
    axs[4].legend(loc='upper right')
    
    plt.tight_layout()
    output_filename = os.path.join(r"c:\Honeywell Project\Code", f"{scenario_name.lower().replace(' ', '_').replace(':', '')}_results.png")
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Generated plot: {output_filename}")

# Alias function for notebook compatibility
plot_scenario_results = plot_results

if __name__ == "__main__":
    print("=== Running Scenario Demonstrations for FlowGuard-MPC ===")
    
    # Scenario A: Startup to Target (110 bbl/hr)
    df_a, rej_a = run_scenario("Scenario A", target_profile=110.0, timesteps=50)
    
    # Scenario B: Target Tracking (100 -> 150 bbl/hr)
    target_b = {t: 100.0 if t < 25 else 150.0 for t in range(50)}
    df_b, rej_b = run_scenario("Scenario B", target_profile=target_b, timesteps=50)
    
    # Scenario C: Infeasible Target (220 bbl/hr)
    df_c, rej_c = run_scenario("Scenario C", target_profile=220.0, timesteps=50)
    
    print("\n--- Execution Summary ---")
    print(f"Scenario A Final Flow: {df_a['Actual_Q'].iloc[-1]:.2f} bbl/hr | Min WHP: {df_a['WHP'].min():.2f} psi | Rejections: {len(rej_a)}")
    print(f"Scenario B Final Flow: {df_b['Actual_Q'].iloc[-1]:.2f} bbl/hr | Min WHP: {df_b['WHP'].min():.2f} psi | Rejections: {len(rej_b)}")
    print(f"Scenario C Settled Flow: {df_c['Actual_Q'].iloc[-1]:.2f} bbl/hr | Min WHP: {df_c['WHP'].min():.2f} psi | Rejections: {len(rej_c)}")
