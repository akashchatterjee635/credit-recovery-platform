import sys
import os
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.deep.trainer import CreditDataset, train_model, evaluate_model
from backend.models.deep.fusion import StaticOnlyModel, TemporalOnlyModel, TemporalStaticFusionModel

def run_experiment():
    print("Loading datasets...")
    train_dataset = CreditDataset('train')
    cal_dataset = CreditDataset('cal')
    test_dataset = CreditDataset('test')
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    cal_loader = DataLoader(cal_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    with open('data/tensors/metadata.json', 'r') as f:
        meta = json.load(f)
        
    ft_params = {
        'num_continuous': meta['num_continuous'],
        'cat_cardinalities': meta['cat_cardinalities'],
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.1
    }
    
    temporal_dim = meta['temporal_features']
    hidden_dim = 64
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    models = {
        'Static Only (FT-Transformer)': StaticOnlyModel(ft_params),
        'Temporal Only (TCN)': TemporalOnlyModel(temporal_dim, hidden_dim, temporal_model_type='TCN'),
        'Temporal Only (GRU)': TemporalOnlyModel(temporal_dim, hidden_dim, temporal_model_type='GRU'),
        'Fusion (TCN + FT-Trans)': TemporalStaticFusionModel(temporal_dim, hidden_dim, hidden_dim, 'TCN', ft_params=ft_params),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}\nTraining {name}\n{'='*50}")
        # Train
        model = train_model(model, train_loader, cal_loader, epochs=5, lr=1e-3, early_stopping=2, device=device)
        if name == 'Fusion (TCN + FT-Trans)':
            torch.save(model.state_dict(), 'backend/models/deep_fusion_model.pth')
        
        # Test
        print(f"\nEvaluating {name} on TEST set...")
        metrics = evaluate_model(model, test_loader, device=device)
        results[name] = metrics
        
        print(f"Results for {name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
    print("\n\n" + "="*80)
    print("FINAL BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Model':<30} | {'ROC-AUC':<10} | {'PR-AUC':<10} | {'Brier':<10} | {'ECE':<10}")
    print("-" * 80)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['roc_auc']:<10.4f} | {metrics['pr_auc']:<10.4f} | {metrics['brier']:<10.4f} | {metrics['ece']:<10.4f}")
        
if __name__ == '__main__':
    run_experiment()
