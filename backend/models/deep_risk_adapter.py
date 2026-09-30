import os
import torch
import numpy as np
import pandas as pd
from backend.models.base_model import BaseRiskAdapter
from backend.models.deep.fusion import TemporalStaticFusionModel
from backend.data.temporal_preprocessor import (
    TEMPORAL_PREPROCESSOR_VERSION,
    TemporalPreprocessor,
)

class DeepRiskAdapter(BaseRiskAdapter):
    """
    Adapter for PyTorch Deep Risk Models (e.g., Fusion model).
    Maintains the historical sequence as immutable while accepting perturbations
    to the static applicant features for recourse generation.
    """
    def __init__(self, model_path=None, meta_path='artifacts/temporal_metadata.json',
                 preprocessor_path='artifacts/temporal_preprocessor.joblib',
                 test_seq_path='data/tensors/test_X_seq.npy',
                 test_mask_path='data/tensors/test_mask.npy',
                 test_df_path='data/test_reference.csv'):
        super().__init__()
        self.model_path = model_path
        self.meta_path = meta_path
        self.model = None
        self.device = 'cpu'
        self.preprocessor = None
        self.preprocessor_path = preprocessor_path
        self.metadata = None

        # Load historical sequences and build SK_ID_CURR mapping if files exist
        self.id_to_seq = {}
        self.id_to_mask = {}
        if os.path.exists(test_seq_path) and os.path.exists(test_df_path):
            self.X_seq_test = np.load(test_seq_path)
            if os.path.exists(test_mask_path):
                self.sequence_masks = np.load(test_mask_path).astype(bool)
            else:
                self.sequence_masks = np.any(self.X_seq_test != 0, axis=2)
            self.test_df = pd.read_csv(test_df_path)
            for i, row in self.test_df.iterrows():
                if 'SK_ID_CURR' in row:
                    self.id_to_seq[int(row['SK_ID_CURR'])] = self.X_seq_test[i]
                    self.id_to_mask[int(row['SK_ID_CURR'])] = self.sequence_masks[i]
                
    def load(self):
        import json
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
        self.metadata = meta
        expected_preprocessor_version = meta.get(
            'preprocessor_version', TEMPORAL_PREPROCESSOR_VERSION
        )
        self.preprocessor = TemporalPreprocessor.load(
            self.preprocessor_path,
            expected_version=expected_preprocessor_version,
        )
        if meta.get('continuous_features') != self.preprocessor.cont_cols:
            raise ValueError('Deep artifact mismatch: continuous feature order differs')
        if meta.get('categorical_features') != self.preprocessor.cat_cols:
            raise ValueError('Deep artifact mismatch: categorical feature order differs')
            
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
        if self.model is None or self.preprocessor is None:
            self.load()

        # 1. Process static features from the current dataframe (which may be counterfactually modified)
        X_cont, X_cat = self.preprocessor.transform_static(applicant)
        X_cont_t = torch.tensor(X_cont, dtype=torch.float32).to(self.device)
        X_cat_t = torch.tensor(X_cat, dtype=torch.long).to(self.device)
        
        # 2. Retrieve immutable historical sequences
        X_seq_list = []
        mask_list = []
        for _, row in applicant.iterrows():
            sk_id = int(row.get('SK_ID_CURR', -1))
            if sk_id in self.id_to_seq:
                X_seq_list.append(self.id_to_seq[sk_id])
                mask_list.append(self.id_to_mask[sk_id])
            else:
                X_seq_list.append(np.zeros(
                    (self.metadata['max_seq_len'], self.metadata['temporal_features']),
                    dtype=np.float32,
                ))
                mask_list.append(np.zeros(self.metadata['max_seq_len'], dtype=bool))
                
        sequences, masks = self.preprocessor.transform_sequence(
            np.asarray(X_seq_list), np.asarray(mask_list)
        )
        X_seq_t = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        mask_t = torch.tensor(masks, dtype=torch.bool).to(self.device)
        
        # 3. Predict
        with torch.no_grad():
            logits, _ = self.model(X_seq_t, X_cont_t, X_cat_t, sequence_mask=mask_t)
            probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
            
        # Ensure it returns 1D array matching batch size
        if probs.ndim == 0:
            probs = np.array([probs])
            
        return probs
