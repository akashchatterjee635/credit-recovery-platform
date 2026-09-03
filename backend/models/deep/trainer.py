import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import numpy as np

class CreditDataset(Dataset):
    def __init__(self, split='train'):
        # Load from disk lazily or aggressively
        self.X_seq = np.load(f'data/tensors/{split}_X_seq.npy')
        self.X_cont = np.load(f'data/tensors/{split}_X_cont.npy')
        self.X_cat = np.load(f'data/tensors/{split}_X_cat.npy')
        self.y = np.load(f'data/tensors/{split}_y.npy')
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return {
            'X_seq': torch.tensor(self.X_seq[idx], dtype=torch.float32),
            'X_cont': torch.tensor(self.X_cont[idx], dtype=torch.float32),
            'X_cat': torch.tensor(self.X_cat[idx], dtype=torch.long),
            'y': torch.tensor(self.y[idx], dtype=torch.float32)
        }

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.sum(mask) > 0:
            prob_pred = np.mean(y_prob[mask])
            prob_true = np.mean(y_true[mask])
            ece += np.abs(prob_pred - prob_true) * np.sum(mask) / len(y_true)
    return ece

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, early_stopping=3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_auc = 0.0
    best_weights = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_seq = batch['X_seq'].to(device)
            x_cont = batch['X_cont'].to(device)
            x_cat = batch['X_cat'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            
            # Forward depending on model type
            if hasattr(model, 'temporal_attention'):
                # Fusion or temporal models return (logits, alpha)
                out = model(x_seq, x_cont, x_cat)
                if isinstance(out, tuple):
                    logits, _ = out
                else:
                    logits = out
            else:
                # Static models
                out = model(x_seq, x_cont, x_cat)
                if isinstance(out, tuple):
                    logits, _ = out
                else:
                    logits = out
                    
            loss = criterion(logits.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        val_auc = val_metrics['roc_auc']
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping:
                print("Early stopping triggered.")
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return model

def evaluate_model(model, loader, device='cpu'):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            x_seq = batch['X_seq'].to(device)
            x_cont = batch['X_cont'].to(device)
            x_cat = batch['X_cat'].to(device)
            y = batch['y'].numpy()
            
            out = model(x_seq, x_cont, x_cat)
            if isinstance(out, tuple):
                logits, _ = out
            else:
                logits = out
                
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y)
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    roc_auc = roc_auc_score(all_targets, all_preds)
    pr_auc = average_precision_score(all_targets, all_preds)
    brier = brier_score_loss(all_targets, all_preds)
    ece = expected_calibration_error(all_targets, all_preds)
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier': brier,
        'ece': ece
    }
