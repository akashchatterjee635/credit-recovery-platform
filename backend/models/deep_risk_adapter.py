import os
import torch
import numpy as np
import pandas as pd
from backend.models.base_model import BaseRiskAdapter
from backend.models.deep.architectures import FTTransformer, TCN
from backend.models.deep.fusion import TemporalStaticFusionModel
from backend.data.temporal_builder import process_static_data

class DeepRiskAdapter(BaseRiskAdapter):
    """
    Adapter for PyTorch Deep Risk Models (e.g., Fusion model).
    Maintains the historical sequence as immutable while accepting perturbations
    to the static applicant features for recourse generation.
    """
    def __init__(self, model_path=None, meta_path='data/tensors/metadata.json', 
                 test_seq_path='data/tensors/test_X_seq.npy', test_df_path='data/test_reference.csv'):
        super().__init__()
        self.model_path = model_path
        self.meta_path = meta_path
        self.model = None
        self.device = 'cpu'
        
        # Load historical sequences and build SK_ID_CURR mapping
        self.X_seq_test = np.load(test_seq_path)
        self.test_df = pd.read_csv(test_df_path)
        self.id_to_seq = {}
        for i, row in self.test_df.iterrows():
            if 'SK_ID_CURR' in row:
                self.id_to_seq[int(row['SK_ID_CURR'])] = self.X_seq_test[i]
                
    def load(self):
        import json
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
            
        ft_params = {
            'num_continuous': meta['num_continuous'],
            'cat_cardinalities': meta['cat_cardinalities'],
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dropout': 0.1
        }
        
        # We assume the Fusion model is the main one used
        self.model = TemporalStaticFusionModel(
            temporal_dim=meta['temporal_features'], 
            hidden_dim=64, 
            static_dim=64, 
            temporal_model_type='TCN', 
            ft_params=ft_params
        )
        
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained deep model weights not found at {self.model_path}. Please run experiments/10_temporal_baselines.py first to train the model.")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            
        self.model.to(self.device)
        self.model.eval()

    def predict_risk(self, applicant: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            self.load()
            
        # 1. Process static features from the current dataframe (which may be counterfactually modified)
        X_cont, X_cat, _, _ = process_static_data(applicant)
        X_cont_t = torch.tensor(X_cont, dtype=torch.float32).to(self.device)
        X_cat_t = torch.tensor(X_cat, dtype=torch.long).to(self.device)
        
        # 2. Retrieve immutable historical sequences
        X_seq_list = []
        for _, row in applicant.iterrows():
            sk_id = int(row.get('SK_ID_CURR', -1))
            if sk_id in self.id_to_seq:
                X_seq_list.append(self.id_to_seq[sk_id])
            else:
                # Fallback to zeros if not found
                import json
                with open(self.meta_path, 'r') as f:
                    meta = json.load(f)
                X_seq_list.append(np.zeros((meta['max_seq_len'], meta['temporal_features']), dtype=np.float32))
                
        X_seq_t = torch.tensor(np.array(X_seq_list), dtype=torch.float32).to(self.device)
        
        # 3. Predict
        with torch.no_grad():
            logits, _ = self.model(X_seq_t, X_cont_t, X_cat_t)
            probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            
        # Ensure it returns 1D array matching batch size
        if probs.ndim == 0:
            probs = np.array([probs])
            
        return probs
