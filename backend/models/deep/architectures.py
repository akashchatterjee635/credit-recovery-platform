import math
import torch
import torch.nn as nn

class FTTransformer(nn.Module):
    """
    Lightweight FT-Transformer for static tabular data.
    Embeds both categorical and continuous features, then applies a Transformer encoder.
    """
    def __init__(self, num_continuous, cat_cardinalities, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Continuous feature embeddings: each feature gets a linear projection to d_model
        # Shape: (batch, num_cont) -> (batch, num_cont, d_model)
        self.num_continuous = num_continuous
        if num_continuous > 0:
            self.cont_embeddings = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_continuous)])
            
        # Categorical feature embeddings
        self.num_categorical = len(cat_cardinalities)
        if self.num_categorical > 0:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(card, d_model, padding_idx=0) for card in cat_cardinalities
            ])
            
        # CLS token for pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x_cont, x_cat):
        batch_size = x_cont.size(0)
        embeddings = []
        
        # [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings.append(cls_tokens)
        
        # Continuous
        if self.num_continuous > 0:
            for i in range(self.num_continuous):
                # shape (batch, 1) -> (batch, 1, d_model)
                e = self.cont_embeddings[i](x_cont[:, i:i+1]).unsqueeze(1)
                embeddings.append(e)
                
        # Categorical
        if self.num_categorical > 0:
            for i in range(self.num_categorical):
                e = self.cat_embeddings[i](x_cat[:, i]).unsqueeze(1)
                embeddings.append(e)
                
        # Stack into sequence: (batch, seq_len=1+num_cont+num_cat, d_model)
        x = torch.cat(embeddings, dim=1)
        
        # Transformer
        x = self.transformer(x)
        
        # Output is the [CLS] representation
        return x[:, 0, :]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Causal convolution
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for sequential data.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size, dropout=dropout
            ))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, seq_len, num_inputs)
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # out shape: (batch, channels, seq_len)
        # Return transpose back: (batch, seq_len, channels)
        return out.transpose(1, 2)


class RecurrentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, model_type='GRU', dropout=0.2):
        super().__init__()
        self.model_type = model_type
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.rnn(x)
        return out
