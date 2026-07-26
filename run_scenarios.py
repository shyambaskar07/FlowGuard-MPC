"""
FlowGuard-MPC Scenario Execution & Plot Generation Script.

Author: Shyambaskar Sriram
SASTRA University - B.E. Computer Science & Engineering (AI & DS)

Executes Honeywell Round 2 Scenarios for FlowGuard-MPC:
    - Scenario A: Startup to Target (110 bbl/hr)
    - Scenario B: Target Tracking (100 -> 150 bbl/hr)
    - Scenario C: Infeasible Target (220 bbl/hr)
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from simulator import OilWellSimulator
from mpc_controller import AutonomousChokeController


def run_simulation(scenario_name, target_profile, timesteps=50, initial_choke=30.0):
    """
    Executes a simulation run over the specified horizon and target profile.
    """
    sim = OilWellSimulator(initial_choke=initial_choke, seed=42)
    controller = AutonomousChokeController(whp_min=210.0, flp_min=150.0, bhp_min=2850.0, max_ramp=5.0)
    
    history = {
        'Time_hr': [],
        'Target_Q': [],
        'Actual_Q': [],
        'WHP': [],
        'FLP': [],
        'BHP': [],
        'Choke_pct': []
    }
    
    current_u = initial_choke
    current_state = {'Q': sim.Q, 'WHP': sim.WHP, 'FLP': sim.FLP, 'BHP': sim.BHP}
    
    for t in range(timesteps):
        target_q = target_profile[t] if t < len(target_profile) else target_profile[-1]
        
        # Calculate optimal choke command via FlowGuard-MPC
        next_u = controller.compute_next_choke_position(current_u, target_q, current_state)
        
        # Step simulator environment
        actual_q, whp, flp, bhp = sim.step(next_u)
        next_state = {'Q': actual_q, 'WHP': whp, 'FLP': flp, 'BHP': bhp}
        
        # Update online gain model
        controller.update_model(current_u, next_u, current_state, next_state)
        
        current_u = next_u
        current_state = next_state
        
        history['Time_hr'].append(t)
        history['Target_Q'].append(target_q)
        history['Actual_Q'].append(actual_q)
        history['WHP'].append(whp)
        history['FLP'].append(flp)
        history['BHP'].append(bhp)
        history['Choke_pct'].append(current_u)
        
    df_res = pd.DataFrame(history)
    return df_res, controller.rejection_log


def plot_scenario_results(df_res, scenario_title, filename):
    """
    Generates a 5-panel trend plot matching Honeywell submission specs.
    """
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"FlowGuard-MPC Performance - {scenario_title}", fontsize=13, fontweight='bold')
    
    # 1. Oil Rate
    axes[0].plot(df_res['Time_hr'], df_res['Target_Q'], 'r--', linewidth=1.8, label='Target Q (bbl/hr)')
    axes[0].plot(df_res['Time_hr'], df_res['Actual_Q'], 'b-', linewidth=1.8, label='Actual Q (bbl/hr)')
    axes[0].set_ylabel("Oil Rate\n(bbl/hr)")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')
    
    # 2. WHP
    axes[1].plot(df_res['Time_hr'], df_res['WHP'], 'g-', linewidth=1.8, label='WHP (psi)')
    axes[1].axhline(y=210.0, color='r', linestyle=':', linewidth=1.8, label='WHP Limit (210 psi)')
    axes[1].set_ylabel("WHP\n(psi)")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')
    
    # 3. FLP
    axes[2].plot(df_res['Time_hr'], df_res['FLP'], 'm-', linewidth=1.8, label='FLP (psi)')
    axes[2].axhline(y=150.0, color='r', linestyle=':', linewidth=1.8, label='FLP Limit (150 psi)')
    axes[2].set_ylabel("FLP\n(psi)")
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].legend(loc='upper right')
    
    # 4. BHP
    axes[3].plot(df_res['Time_hr'], df_res['BHP'], 'c-', linewidth=1.8, label='BHP (psi)')
    axes[3].axhline(y=2850.0, color='r', linestyle=':', linewidth=1.8, label='BHP Limit (2850 psi)')
    axes[3].set_ylabel("BHP\n(psi)")
    axes[3].grid(True, linestyle='--', alpha=0.5)
    axes[3].legend(loc='upper right')
    
    # 5. Choke Position
    axes[4].plot(df_res['Time_hr'], df_res['Choke_pct'], 'k-', linewidth=1.8, label='Choke Opening (%)')
    axes[4].set_ylabel("Choke Position\n(%)")
    axes[4].set_xlabel("Time (Hours)")
    axes[4].grid(True, linestyle='--', alpha=0.5)
    axes[4].legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Generated plot: {filename}")


def main():
    print("=== Running Scenario Demonstrations for FlowGuard-MPC ===")
    output_dir = r"c:\Honeywell Project\Code"
    
    # Scenario A: Startup to Target (110 bbl/hr)
    target_A = [110.0] * 50
    df_A, rejections_A = run_simulation("Scenario A", target_A, timesteps=50, initial_choke=30.0)
    plot_scenario_results(df_A, "Scenario A: Startup to Target (110 bbl/hr)", os.path.join(output_dir, "scenario_A_results.png"))
    
    # Scenario B: Target Tracking (100 -> 150 bbl/hr)
    target_B = [100.0] * 25 + [150.0] * 25
    df_B, rejections_B = run_simulation("Scenario B", target_B, timesteps=50, initial_choke=35.0)
    plot_scenario_results(df_B, "Scenario B: Target Tracking (100 -> 150 bbl/hr)", os.path.join(output_dir, "scenario_B_results.png"))
    
    # Scenario C: Infeasible Target (220 bbl/hr)
    target_C = [220.0] * 50
    df_C, rejections_C = run_simulation("Scenario C", target_C, timesteps=50, initial_choke=30.0)
    plot_scenario_results(df_C, "Scenario C: Infeasible Target (Safe Envelope Settling)", os.path.join(output_dir, "scenario_C_results.png"))
    
    print("\n--- Execution Summary ---")
    print(f"Scenario A Final Flow: {df_A['Actual_Q'].iloc[-1]:.2f} bbl/hr | Min WHP: {df_A['WHP'].min():.2f} psi | Rejections: {len(rejections_A)}")
    print(f"Scenario B Final Flow: {df_B['Actual_Q'].iloc[-1]:.2f} bbl/hr | Min WHP: {df_B['WHP'].min():.2f} psi | Rejections: {len(rejections_B)}")
    print(f"Scenario C Settled Flow: {df_C['Actual_Q'].iloc[-1]:.2f} bbl/hr | Min WHP: {df_C['WHP'].min():.2f} psi | Rejections: {len(rejections_C)}")


if __name__ == "__main__":
    main()
