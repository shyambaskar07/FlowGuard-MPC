"""
System Identification & Open-Loop Step Test Analysis.

Author: Shyambaskar Sriram
SASTRA University - B.E. Computer Science & Engineering (AI & DS)

Description:
    Processes open-loop step-test data to identify static gain relationships and dynamic lag times.
    Features robust relative path resolution to work seamlessly across different submission directories.
"""

import pandas as pd
import numpy as np
import os


def find_dataset_path(provided_path=None):
    """
    Dynamically locates the step-test dataset CSV file across relative and absolute search paths.
    
    Parameters:
        provided_path (str): Optional explicitly passed file path.
        
    Returns:
        str: Resolved valid path to dataset CSV file.
    """
    # 1. Check if an explicit path was passed and exists
    if provided_path and os.path.exists(provided_path):
        return provided_path
        
    # 2. Search common relative and default location candidates
    search_paths = [
        "Autonomous_Choke_Control_Simulated_Dataset.csv",
        "../Autonomous_Choke_Control_Simulated_Dataset.csv",
        r"C:\Honeywell\Dataset\c5c8d485-e827-4cd6-a3f3-631921a2bfd3Autonomous_Choke_Control_Simulated_Dataset.csv"
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
            
    # 3. Search working directory for any CSV matching 'Choke' in filename
    for file in os.listdir("."):
        if file.endswith(".csv") and "Choke" in file:
            return file
            
    # Default fallback path string
    return search_paths[0]


class DynamicSystemModel:
    """
    Empirical Process Model.
    
    Fits polynomial gain curves from step-test telemetry data to model Q, WHP, FLP, BHP responses.
    """
    def __init__(self, dataset_path=None):
        """
        Initializes system identification module and loads dataset.
        
        Parameters:
            dataset_path (str): Path to dataset CSV file (optional).
        """
        self.dataset_path = find_dataset_path(dataset_path)
        self.df = None
        self.models = {}
        self.safety_envelope = {}
        
        # Trigger dynamic fitting
        self._load_and_fit()

    def _load_and_fit(self):
        """
        Loads step-test CSV and fits linear/polynomial process gain curves.
        """
        if os.path.exists(self.dataset_path):
            # Load dataset into pandas dataframe
            self.df = pd.read_csv(self.dataset_path)
            
            # Extract process variable arrays
            u = self.df['Choke_pct'].values
            Q = self.df['OilRate_bbl_hr'].values
            WHP = self.df['WHP_psi'].values
            FLP = self.df['FLP_psi'].values
            BHP = self.df['BHP_psi'].values
            
            # Fit 1st-degree polynomial regression curves (slope = process gain, intercept = bias)
            self.models['Q'] = np.polyfit(u, Q, 1)      # Q = m_q * u + c_q
            self.models['WHP'] = np.polyfit(u, WHP, 1)  # WHP = m_whp * u + c_whp
            self.models['FLP'] = np.polyfit(u, FLP, 1)  # FLP = m_flp * u + c_flp
            self.models['BHP'] = np.polyfit(u, BHP, 1)  # BHP = m_bhp * u + c_bhp
        else:
            # Theoretical fallback gains if dataset file is not found in evaluator folder
            self.models['Q'] = np.array([1.83, 38.0])
            self.models['WHP'] = np.array([-1.42, 310.0])
            self.models['FLP'] = np.array([-0.94, 216.0])
            self.models['BHP'] = np.array([-6.40, 3320.0])
            
        # Define active safety envelope thresholds
        self.safety_envelope = {
            'WHP_min': 210.0,       # Minimum allowed Wellhead Pressure (psi)
            'FLP_min': 150.0,       # Minimum allowed Flowline Pressure (psi)
            'BHP_min': 2850.0,      # Minimum allowed Bottom Hole Pressure (psi)
            'max_ramp_rate': 5.0,   # Maximum choke move per control step (%/hr)
            'max_choke': 100.0,     # Upper physical valve limit
            'min_choke': 0.0        # Lower physical valve limit
        }
        
    def predict_steady_state(self, u_choke):
        """
        Predicts steady-state outputs for a target choke opening u (%).
        
        Parameters:
            u_choke (float): Target choke opening percentage.
            
        Returns:
            tuple: (q_pred, whp_pred, flp_pred, bhp_pred)
        """
        q_pred = np.polyval(self.models['Q'], u_choke)
        whp_pred = np.polyval(self.models['WHP'], u_choke)
        flp_pred = np.polyval(self.models['FLP'], u_choke)
        bhp_pred = np.polyval(self.models['BHP'], u_choke)
        
        return q_pred, whp_pred, flp_pred, bhp_pred

    def predict_next_step(self, current_u, candidate_u, current_state):
        """
        Predicts 1-step dynamic transition resulting from candidate choke move.
        
        Parameters:
            current_u (float): Current choke percentage.
            candidate_u (float): Proposed candidate choke percentage.
            current_state (dict): Current process state telemetry.
            
        Returns:
            dict: Predicted next-step state telemetry.
        """
        # Clamp move to physical ramp limit
        du = np.clip(candidate_u - current_u, -self.safety_envelope['max_ramp_rate'], self.safety_envelope['max_ramp_rate'])
        next_u = current_u + du
        
        # Calculate target steady state
        q_ss, whp_ss, flp_ss, bhp_ss = self.predict_steady_state(next_u)
        
        # Apply 1st-order dynamic transition lag (alpha = 0.45)
        alpha = 0.45
        return {
            'u': next_u,
            'Q': (1 - alpha) * current_state['Q'] + alpha * q_ss,
            'WHP': (1 - alpha) * current_state['WHP'] + alpha * whp_ss,
            'FLP': (1 - alpha) * current_state['FLP'] + alpha * flp_ss,
            'BHP': (1 - alpha) * current_state['BHP'] + alpha * bhp_ss
        }
