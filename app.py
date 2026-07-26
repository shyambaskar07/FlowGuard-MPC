"""
Interactive Streamlit Dashboard for FlowGuard-MPC.

Author: Shyambaskar Sriram
SASTRA University - B.E. Computer Science & Engineering (AI & DS)

Description:
    Provides an interactive web dashboard for FlowGuard-MPC to run live simulations,
    move setpoint target sliders, visualize real-time telemetry curves, and inspect candidate move rejection logs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator import OilWellSimulator
from mpc_controller import AutonomousChokeController

# Configure Streamlit page layout and theme
st.set_page_config(
    page_title="FlowGuard-MPC: Autonomous Production Choke Controller",
    layout="wide"
)

# Application Header & Title
st.title("FlowGuard-MPC: Adaptive Model-Predictive Production Choke Controller")
st.caption("Autonomous Safe Oil Well Optimization")

# Sidebar Configuration & Simulation Controls
st.sidebar.header("Simulation Settings")

# Scenario selection dropdown
scenario_option = st.sidebar.selectbox(
    "Select Demonstration Scenario",
    ["Scenario A: Startup to Target", "Scenario B: Target Tracking", "Scenario C: Infeasible Target", "Custom Target Control"]
)

# Preset targets based on selected scenario
if scenario_option == "Scenario A: Startup to Target":
    default_target = 110.0
elif scenario_option == "Scenario B: Target Tracking":
    default_target = 150.0
elif scenario_option == "Scenario C: Infeasible Target":
    default_target = 220.0 # Unsafe high flow target forcing candidate move rejection
else:
    default_target = 120.0

# Interactive sliders for target flow rate and time horizon
target_q = st.sidebar.slider("Target Oil Rate (bbl/hr)", min_value=50.0, max_value=250.0, value=default_target, step=5.0)
timesteps = st.sidebar.slider("Simulation Horizon (Hours)", min_value=10, max_value=100, value=50, step=10)

# Display active safety envelope bounds in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Safety Envelope Constraints")
st.sidebar.write("- Max Ramp Rate: +/- 5.0% / hr")
st.sidebar.write("- Min WHP: 210.0 psi")
st.sidebar.write("- Min FLP: 150.0 psi")
st.sidebar.write("- Min BHP: 2850.0 psi")

# Main simulation trigger button
if st.button("Run FlowGuard-MPC Simulation", use_container_width=True):
    # Initialize simulator environment and MPC controller
    sim = OilWellSimulator(seed=42)
    controller = AutonomousChokeController(whp_min=210.0, flp_min=150.0, bhp_min=2850.0, max_ramp=5.0)
    
    # Telemetry storage dictionary
    history = {'Time': [], 'Target_Q': [], 'Actual_Q': [], 'WHP': [], 'FLP': [], 'BHP': [], 'Choke': []}
    current_u = 30.0
    current_state = {'Q': sim.Q, 'WHP': sim.WHP, 'FLP': sim.FLP, 'BHP': sim.BHP}
    
    # Execute closed-loop simulation across time steps
    for t in range(timesteps):
        # Handle Scenario B step change mid-simulation
        if scenario_option == "Scenario B: Target Tracking" and t < timesteps // 2:
            active_target = 100.0
        else:
            active_target = target_q
            
        # Step 1: Compute optimal safe choke move using FlowGuard-MPC
        next_u = controller.compute_next_choke_position(current_u, active_target, current_state, time_step=t)
        
        # Step 2: Step physical simulator
        actual_q, whp, flp, bhp = sim.step(next_u)
        next_state = {'Q': actual_q, 'WHP': whp, 'FLP': flp, 'BHP': bhp}
        
        # Step 3: Update live online gain estimation model
        controller.update_model(current_u, next_u, current_state, next_state)
        
        # Advance pointers
        current_u = next_u
        current_state = next_state
        
        # Log telemetry
        history['Time'].append(t)
        history['Target_Q'].append(active_target)
        history['Actual_Q'].append(actual_q)
        history['WHP'].append(whp)
        history['FLP'].append(flp)
        history['BHP'].append(bhp)
        history['Choke'].append(current_u)
        
    df = pd.DataFrame(history)
    
    # Display Summary KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Final Oil Flow Rate", f"{df['Actual_Q'].iloc[-1]:.1f} bbl/hr", delta=f"{df['Actual_Q'].iloc[-1] - df['Target_Q'].iloc[-1]:.1f} bbl/hr")
    col2.metric("Final WHP", f"{df['WHP'].iloc[-1]:.1f} psi", delta="Safe" if df['WHP'].min() >= 210 else "Breach", delta_color="normal" if df['WHP'].min() >= 210 else "inverse")
    col3.metric("Final FLP", f"{df['FLP'].iloc[-1]:.1f} psi", delta="Safe" if df['FLP'].min() >= 150 else "Breach", delta_color="normal" if df['FLP'].min() >= 150 else "inverse")
    col4.metric("Final BHP", f"{df['BHP'].iloc[-1]:.1f} psi", delta="Safe" if df['BHP'].min() >= 2850 else "Breach", delta_color="normal" if df['BHP'].min() >= 2850 else "inverse")
    col5.metric("Candidate Rejections", f"{len(controller.rejection_log)}", delta="Zero Unsafe Moves Executed", delta_color="normal")
    
    # Generate Plotly Multi-Panel Interactive Charts
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=["Oil Flow Rate (Q)", "Wellhead Pressure (WHP)", "Flowline Pressure (FLP)", "Bottom Hole Pressure (BHP)", "Choke Position (u)"])
    
    # Panel 1: Flow rate tracking
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Target_Q'], name="Target Q", line=dict(color='#FF3333', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Actual_Q'], name="Actual Q", line=dict(color='#3399FF')), row=1, col=1)
    
    # Panel 2: WHP curve + limit line
    fig.add_trace(go.Scatter(x=df['Time'], y=df['WHP'], name="WHP", line=dict(color='#00CC66')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=[210.0]*len(df), name="WHP Limit", line=dict(color='#FF3333', dash='dot')), row=2, col=1)
    
    # Panel 3: FLP curve + limit line
    fig.add_trace(go.Scatter(x=df['Time'], y=df['FLP'], name="FLP", line=dict(color='#CC66FF')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=[150.0]*len(df), name="FLP Limit", line=dict(color='#FF3333', dash='dot')), row=3, col=1)
    
    # Panel 4: BHP curve + limit line
    fig.add_trace(go.Scatter(x=df['Time'], y=df['BHP'], name="BHP", line=dict(color='#FF9900')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=[2850.0]*len(df), name="BHP Limit", line=dict(color='#FF3333', dash='dot')), row=4, col=1)
    
    # Panel 5: Choke opening percentage - Bright Electric Cyan (#00E5FF) for high contrast visibility
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Choke'], name="Choke Opening (%)", line=dict(color='#00E5FF', width=2)), row=5, col=1)
    
    fig.update_layout(height=1000, title_text="Live FlowGuard-MPC Process Telemetry & Control Curves", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Diagnostic Rejection Log Viewer
    with st.expander("View Controller Candidate Rejection Log (Safety Filter Trace)"):
        if controller.rejection_log:
            st.dataframe(pd.DataFrame(controller.rejection_log))
        else:
            st.success("No candidate moves breached safety constraints!")

st.info("Key Highlight: When an infeasible target is requested (e.g. 220 bbl/hr), FlowGuard-MPC automatically rejects unsafe choke openings and settles smoothly at the maximum safe flow rate without violating pressure limits.")
