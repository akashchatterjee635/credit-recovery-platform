import torch
import torch.nn as nn
from backend.models.deep.architectures import FTTransformer, TCN, RecurrentModel

class TemporalAttention(nn.Module):
    """
    Computes scalar attention weights over the sequence steps to distinguish
    recent vs. historical behavior.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, H):
        # H shape: (batch_size, seq_len, hidden_dim)
        
        # e_t = v^T * tanh(W_h * h_t + b_h)
        # shape: (batch_size, seq_len, 1)
        energy = self.v(self.tanh(self.W(H)))
        
        # alpha_t = softmax(e_t)
        # shape: (batch_size, seq_len, 1)
        alpha = torch.softmax(energy, dim=1)
        
        # h* = sum(alpha_t * h_t)
        # shape: (batch_size, hidden_dim)
        context = torch.sum(alpha * H, dim=1)
        
        return context, alpha


class TemporalStaticFusionModel(nn.Module):
    """
    Combines Temporal Representation (from TCN/GRU) with Static Representation (FT-Transformer)
    using a Gated Fusion mechanism.
    """
    def __init__(self, temporal_dim, static_dim, hidden_dim, temporal_model_type='TCN', 
                 tcn_channels=None, rnn_layers=2, ft_params=None):
        super().__init__()
        
        # 1. Temporal Branch
        if temporal_model_type == 'TCN':
            if tcn_channels is None:
                tcn_channels = [64, 64, hidden_dim]
            self.temporal_encoder = TCN(temporal_dim, tcn_channels)
        elif temporal_model_type in ('GRU', 'LSTM'):
            self.temporal_encoder = RecurrentModel(temporal_dim, hidden_dim, rnn_layers, model_type=temporal_model_type)
        else:
            raise ValueError(f"Unknown temporal_model_type: {temporal_model_type}")
            
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # 2. Static Branch (FT-Transformer)
        if ft_params is None:
            raise ValueError("ft_params (num_continuous, cat_cardinalities) must be provided.")
        self.static_encoder = FTTransformer(**ft_params)
        
        # 3. Gated Fusion
        # Project static representation to hidden_dim if needed
        self.static_proj = nn.Linear(ft_params['d_model'], hidden_dim) if ft_params['d_model'] != hidden_dim else nn.Identity()
        
        # Gate: g = sigmoid(W_g [h_t; h_s] + b_g)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 4. Final Classification Head
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_seq, x_cont, x_cat):
        # -- Temporal --
        # H shape: (batch, seq_len, hidden_dim)
        H = self.temporal_encoder(x_seq)
        
        # h_t shape: (batch, hidden_dim)
        h_t, alpha = self.temporal_attention(H)
        
        # -- Static --
        # h_s shape: (batch, d_model) -> (batch, hidden_dim)
        h_s = self.static_encoder(x_cont, x_cat)
        h_s = self.static_proj(h_s)
        
        # -- Fusion --
        # g shape: (batch, hidden_dim)
        g = torch.sigmoid(self.gate(torch.cat([h_t, h_s], dim=1)))
        
        # h shape: (batch, hidden_dim)
        h = g * h_t + (1 - g) * h_s
        
        # -- Prediction --
        logits = self.classifier(h)
        
        return logits, alpha

class StaticOnlyModel(nn.Module):
    def __init__(self, ft_params):
        super().__init__()
        self.static_encoder = FTTransformer(**ft_params)
        self.classifier = nn.Linear(ft_params['d_model'], 1)
        
    def forward(self, x_seq, x_cont, x_cat):
        h_s = self.static_encoder(x_cont, x_cat)
        return self.classifier(h_s), None

class TemporalOnlyModel(nn.Module):
    def __init__(self, temporal_dim, hidden_dim, temporal_model_type='TCN', tcn_channels=None, rnn_layers=2):
        super().__init__()
        if temporal_model_type == 'TCN':
            if tcn_channels is None:
                tcn_channels = [64, 64, hidden_dim]
            self.temporal_encoder = TCN(temporal_dim, tcn_channels)
        elif temporal_model_type in ('GRU', 'LSTM'):
            self.temporal_encoder = RecurrentModel(temporal_dim, hidden_dim, rnn_layers, model_type=temporal_model_type)
            
        self.temporal_attention = TemporalAttention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_seq, x_cont, x_cat):
        H = self.temporal_encoder(x_seq)
        h_t, alpha = self.temporal_attention(H)
        return self.classifier(h_t), alpha

