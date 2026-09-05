import os
import numpy as np
import pandas as pd
from tqdm import tqdm

MAX_SEQ_LEN = 36  # Last 3 years of monthly data

def generate_mock_longitudinal_data(df: pd.DataFrame, max_seq_len: int = MAX_SEQ_LEN) -> np.ndarray:
    '''
    Generates a mock 3D temporal tensor of shape (N, T, F_temp).
    Injects a clear signal so that TARGET=1 have worse histories.
    Features (F=5):
      0: installment_amount (normalized ~ [0, 1])
      1: payment_ratio (payment / installment)
      2: days_late
      3: active_credit_lines
      4: overdue_balance
    '''
    N = len(df)
    F = 5
    X_seq = np.zeros((N, max_seq_len, F), dtype=np.float32)
    
    targets = df['TARGET'].values
    
    # We use vectorised random generation for speed
    rng = np.random.default_rng(42)
    
    print(f"Generating mock temporal data for {N} applicants...")
    
    for i in tqdm(range(N)):
        is_default = (targets[i] == 1)
        
        # Decide how many historical months exist for this applicant (pad the rest with 0s at the BEGINNING)
        seq_len = rng.integers(12, max_seq_len + 1)
        start_idx = max_seq_len - seq_len
        
        # Base distributions
        if is_default:
            pay_ratio_mean = rng.uniform(0.7, 0.95)
            days_late_mean = rng.uniform(5, 30)
            overdue_prob = 0.4
        else:
            pay_ratio_mean = rng.uniform(0.95, 1.0)
            days_late_mean = rng.uniform(-5, 5)  # Paid early or slightly late
            overdue_prob = 0.05
            
        # Time-varying trends (recent deterioration for defaulters)
        for t in range(seq_len):
            idx = start_idx + t
            
            # 1. Installment amount (stable mostly)
            inst = rng.uniform(0.1, 0.5)
            
            # 2. Payment ratio
            # Deterioration: as t -> seq_len (recent), defaulters pay less
            deterioration = 1.0
            if is_default:
                deterioration = 1.0 - 0.3 * (t / seq_len)
            
            pay_ratio = np.clip(rng.normal(pay_ratio_mean * deterioration, 0.1), 0.0, 1.0)
            
            # 3. Days late
            # Deterioration: as t -> seq_len, defaulters get later
            late_trend = 0
            if is_default:
                late_trend = 15 * (t / seq_len)
            days_late = rng.normal(days_late_mean + late_trend, 5.0)
            days_late = max(0, days_late)  # Relu
            
            # 4. Active credit lines
            lines = rng.integers(1, 6)
            
            # 5. Overdue balance
            overdue = 0.0
            if rng.random() < overdue_prob:
                overdue = rng.uniform(0.1, 1.0)
                if is_default:
                    overdue += 0.5 * (t / seq_len)
                    
            X_seq[i, idx, 0] = inst
            X_seq[i, idx, 1] = pay_ratio
            X_seq[i, idx, 2] = days_late / 30.0  # Normalize to rough months
            X_seq[i, idx, 3] = lines / 10.0      # Normalize
            X_seq[i, idx, 4] = overdue

    return X_seq

def process_static_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray, list, list):
    '''
    Extracts continuous and categorical static features.
    Returns X_cont, X_cat, cont_cols, cat_cols.
    '''
    exclude = ['SK_ID_CURR', 'TARGET']
    cols = [c for c in df.columns if c not in exclude]
    
    # Simple heuristic: object/bool are cat, others are cont
    # We already preprocessed in phase 1, so most are numeric. 
    # Let's consider features with < 15 unique values as categorical for FT-Transformer.
    cat_cols = []
    cont_cols = []
    from pandas.api.types import is_numeric_dtype
    for c in cols:
        if df[c].nunique() < 15 or not is_numeric_dtype(df[c]):
            cat_cols.append(c)
        else:
            cont_cols.append(c)
            
    # For categoricals, we need them to be 0-indexed integers
    X_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for i, c in enumerate(cat_cols):
        # pd.factorize handles strings/floats/ints uniquely
        codes, _ = pd.factorize(df[c])
        # Handle NaNs which become -1 -> shift to 0, use 0 as UNK/NaN
        X_cat[:, i] = codes + 1 
        
    X_cont = df[cont_cols].fillna(0).values.astype(np.float32)
    
    # Scale continuous to roughly N(0, 1) using robust scaling
    for i in range(X_cont.shape[1]):
        col_data = X_cont[:, i]
        q25, q75 = np.percentile(col_data, [25, 75])
        iqr = q75 - q25
        if iqr > 1e-6:
            X_cont[:, i] = (col_data - np.median(col_data)) / iqr
        else:
            std = np.std(col_data)
            if std > 1e-6:
                X_cont[:, i] = (col_data - np.mean(col_data)) / std
                
    return X_cont, X_cat, cont_cols, cat_cols

def build_datasets():
    os.makedirs('data/tensors', exist_ok=True)
    splits = ['train', 'cal', 'test']
    
    # Ensure reproducibility
    np.random.seed(42)
    
    for split in splits:
        path = f'data/{split}_reference.csv'
        if not os.path.exists(path):
            print(f"Skipping {split}, file not found.")
            continue
            
        print(f"--- Processing {split} split ---")
        df = pd.read_csv(path)
        
        # 1. Temporal
        X_seq = generate_mock_longitudinal_data(df)
        
        # 2. Static
        X_cont, X_cat, cont_cols, cat_cols = process_static_data(df)
        
        # 3. Target
        y = df['TARGET'].values.astype(np.float32)
        
        # Save
        np.save(f'data/tensors/{split}_X_seq.npy', X_seq)
        np.save(f'data/tensors/{split}_X_cont.npy', X_cont)
        np.save(f'data/tensors/{split}_X_cat.npy', X_cat)
        np.save(f'data/tensors/{split}_y.npy', y)
        
    # Save feature metadata for the model definition
    import json
    metadata = {
        'num_continuous': len(cont_cols),
        'num_categorical': len(cat_cols),
        'cat_cardinalities': [],
        'temporal_features': 5,
        'max_seq_len': MAX_SEQ_LEN
    }
    
    # We need cardinalities from the TRAIN set to define embedding layers
    train_df = pd.read_csv('data/train_reference.csv')
    _, X_cat_train, _, _ = process_static_data(train_df)
    
    cardinalities = []
    for i in range(X_cat_train.shape[1]):
        # Cardinality = max code + 1 (since 0-indexed)
        card = int(np.max(X_cat_train[:, i])) + 1
        cardinalities.append(card)
        
    metadata['cat_cardinalities'] = cardinalities
    
    with open('data/tensors/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print("Dataset construction complete.")

if __name__ == '__main__':
    build_datasets()
