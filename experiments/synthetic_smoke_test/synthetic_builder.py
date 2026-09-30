import os
import numpy as np
import pandas as pd
from tqdm import tqdm

MAX_SEQ_LEN = 36  # Last 3 years of monthly data

def generate_mock_longitudinal_data(df: pd.DataFrame, max_seq_len: int = MAX_SEQ_LEN) -> np.ndarray:
    '''
    Generates a mock 3D temporal tensor of shape (N, T, F_temp).
    WARNING: Architecture-validation only! These sequences are pure random noise 
    to avoid data leakage. They do not correlate with TARGET.
    '''
    N = len(df)
    F = 5
    X_seq = np.zeros((N, max_seq_len, F), dtype=np.float32)
    
    # We use vectorised random generation for speed
    rng = np.random.default_rng(42)
    
    print(f"Generating un-leaked mock temporal data for {N} applicants...")
    
    for i in tqdm(range(N)):
        # FORCE NO LEAKAGE: We do not check targets[i]
        
        # Decide how many historical months exist for this applicant
        seq_len = rng.integers(12, max_seq_len + 1)
        start_idx = max_seq_len - seq_len
        
        # Base distributions (randomly assign one of the modes to provide variance)
        if rng.random() < 0.5:
            pay_ratio_mean = rng.uniform(0.7, 0.95)
            days_late_mean = rng.uniform(5, 30)
            overdue_prob = 0.4
        else:
            pay_ratio_mean = rng.uniform(0.95, 1.0)
            days_late_mean = rng.uniform(-5, 5)  # Paid early or slightly late
            overdue_prob = 0.05
            
        for t in range(seq_len):
            idx = start_idx + t
            
            # 1. Installment amount
            inst = rng.uniform(0.1, 0.5)
            
            # 2. Payment ratio
            deterioration = 1.0 - 0.3 * (t / seq_len) if rng.random() < 0.5 else 1.0
            pay_ratio = np.clip(rng.normal(pay_ratio_mean * deterioration, 0.1), 0.0, 1.0)
            
            # 3. Days late
            late_trend = 15 * (t / seq_len) if rng.random() < 0.5 else 0
            days_late = rng.normal(days_late_mean + late_trend, 5.0)
            days_late = max(0, days_late)
            
            # 4. Active credit lines
            lines = rng.integers(1, 6)
            
            # 5. Overdue balance
            overdue = 0.0
            if rng.random() < overdue_prob:
                overdue = rng.uniform(0.1, 1.0)
                if rng.random() < 0.5:
                    overdue += 0.5 * (t / seq_len)
                    
            X_seq[i, idx, 0] = inst
            X_seq[i, idx, 1] = pay_ratio
            X_seq[i, idx, 2] = days_late / 30.0
            X_seq[i, idx, 3] = lines / 10.0
            X_seq[i, idx, 4] = overdue

    return X_seq
