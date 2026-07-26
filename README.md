# FlowGuard-MPC: Adaptive Model-Predictive Production Choke Controller for Autonomous Safe Oil Well Optimization

Honeywell Hackathon Round 2 Deliverable  
Author: Shyambaskar Sriram  
Institution: SASTRA University (B.E. Computer Science and Engineering - AI & DS)  
Live Web Dashboard: https://flowguard-mpc.streamlit.app/

---

## Executive Summary

This project presents FlowGuard-MPC, an Autonomous Model Predictive Controller (MPC) designed to optimize production choke valve opening for a single naturally flowing oil well. The solution maximizes oil flow rate (Q) while enforcing active safety pressure bounds (WHP >= 210 psi, FLP >= 150 psi, BHP >= 2850 psi) and actuator ramp-rate limits (|du| <= 5% / hour).

Key highlights of the implementation:
- Plug-and-Play API Compliance: Directly interfaces with Honeywell's evaluation API signature: Q, WHP, FLP, BHP = simulator.step(choke_position)
- Online System Identification: Recursively updates process sensitivity gains (dQ/du, dWHP/du, dFLP/du, dBHP/du) live during operation.
- Candidate Rejection Engine: Evaluates proposed choke moves and instantly voids/rejects any move predicted to breach safety thresholds.
- Infeasible Target Resiliency: Automatically settles at the maximum safe achievable production rate when targets exceed operating bounds.

---

## System Architecture

+---------------------------------------------------------------------------------+
|                             INTERACTIVE DASHBOARD                               |
|                     (Streamlit Web UI: python -m streamlit run app.py)          |
+---------------------------------------------------------------------------------+
                                       |
                                       v
+---------------------------------------------------------------------------------+
|                    FLOWGUARD-MPC AUTONOMOUS CONTROLLER                          |
|  - Multi-Step Horizon Trajectory Prediction                                      |
|  - Adaptive Telemetry Noise Filtering                                           |
|                                                                                 |
|  [HARD SAFETY FILTER & CANDIDATE REJECTION ENGINE]                              |
|  1. Predicts WHP, FLP, BHP for candidate choke moves                            |
|  2. REJECTS any candidate action breaching WHP_min, FLP_min, BHP_min            |
|  3. Dynamic fallback to maximum safe choke move within envelope                 |
+---------------------------------------------------------------------------------+
                                       |
                   Reads State /       | Sends Safe Choke
                   Telemetry           | Move u_t
                                       v
+---------------------------------------------------------------------------------+
|                    HONEYWELL OIL WELL SIMULATOR                                 |
|  - Evaluator Interface: Q, WHP, FLP, BHP = simulator.step(choke_position)       |
|  - Fixed Control Interval (Ts): 1 Hour                                          |
+---------------------------------------------------------------------------------+

---

## Repository Directory & Deliverables

| File Name | Description |
| :--- | :--- |
| mpc_controller.py | Core FlowGuard-MPC Controller with Candidate Rejection Engine and Online System ID. |
| simulator.py | Dynamic physical well simulator for offline testing matching official API signature. |
| system_id.py | Step-test dataset regression fitting and dynamic relative path resolver module. |
| run_scenarios.py | Scenario execution engine generating publication-grade 5-panel trend plots. |
| app.py | Interactive Streamlit web dashboard for live demo visualization and rejection log inspection. |
| choke_controller_solution.ipynb | Master Jupyter Notebook deliverable with dynamic fitting and scenario cell outputs. |
| requirements.txt | Dependency list for 1-command evaluator environment setup. |
| FlowGuard_MPC_Architecture_Diagram.png | 16:9 widescreen system architecture diagram. |
| scenario_A_results.png | 5-panel trend plot for Scenario A (Startup to Target). |
| scenario_B_results.png | 5-panel trend plot for Scenario B (Target Tracking). |
| scenario_C_results.png | 5-panel trend plot for Scenario C (Infeasible Target Safe Settling). |

---

## Step-by-Step Execution Instructions

### Step 1: Install Dependencies
pip install -r requirements.txt

### Step 2: Run Scenario Simulations & Generate Trend Plots
python run_scenarios.py

### Step 3: Launch Interactive Web Dashboard
python -m streamlit run app.py

### Step 4: Open Master Jupyter Notebook
python -m notebook choke_controller_solution.ipynb

---

## Scenario Performance & Safety Summary

| Scenario | Target Flow Rate | Actual Flow Settled | Min WHP (Limit >= 210) | Unsafe Moves Rejected | Safety Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Scenario A: Startup | 110.0 bbl/hr | 120.38 bbl/hr | 225.15 psi | 36 | 100% Passed |
| Scenario B: Target Tracking | 100 -> 150 bbl/hr | 162.58 bbl/hr | 212.71 psi | 1,155 | 100% Passed |
| Scenario C: Infeasible Target | 220.0 bbl/hr | 164.12 bbl/hr | 211.46 psi | 4,150 | 100% Passed (Zero Breaches) |
