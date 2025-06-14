# models/nn_models.py

import torch
import torch.nn as nn

class EnhancedLSTM(nn.Module):
    """
    Bidirectional LSTM with LayerNorm and a small head.
    """
    def __init__(self, in_dim, hid_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.norm = nn.LayerNorm(hid_dim*2)
        self.head = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim//2, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # (B, hid_dim*2)
        normed = self.norm(last)
        return self.head(normed).squeeze(1)
    


class SmallLSTM(nn.Module):
    """
    Simple unidirectional LSTM for 6‐month sequences.
    """
    def __init__(self, in_dim, hid_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hid_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

class LargeLSTM(nn.Module):
    """
    Bidirectional LSTM with deeper head.
    """
    def __init__(self, in_dim, hid_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hid_dim*2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(1)


class StockTransformer(nn.Module):
    """
    Transformer‐based classifier over time‐series window.
    """
    def __init__(self, in_dim, window, d_model=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, window, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        # x: (batch, seq_len, in_dim)
        x = self.input_proj(x) + self.pos_embed[:, :x.size(1), :]
        out = self.transformer(x)            # (batch, seq_len, d_model)
        return self.classifier(out[:, -1, :]).squeeze(1)



class InceptionTime(nn.Module):
    """
    InceptionTime network for time‐series classification.
    """
    def __init__(self, in_dim, num_blocks=3, out_channels=32, 
                 kernel_sizes=[10,20,40], bottleneck_channels=32, 
                 use_residual=True, dropout=0.2):
        super().__init__()
        blocks = []
        channels = in_dim
        for _ in range(num_blocks):
            blocks.append(InceptionModule(
                in_channels=channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                use_residual=use_residual,
                dropout=dropout
            ))
            channels = out_channels * (len(kernel_sizes) + 1)
        self.network = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out = self.network(x)
        # out: (batch, seq_len, channels)
        out = out.transpose(1, 2)            # -> (batch, channels, seq_len)
        pooled = self.global_pool(out).squeeze(2)
        return self.classifier(pooled).squeeze(1)

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[10,20,40], 
                 bottleneck_channels=32, use_residual=False, dropout=0.2):
        super().__init__()
        self.use_residual = use_residual
        # 1x1 Bottleneck
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1) \
                          if in_channels > 1 else nn.Identity()
        # Convolutions with different kernel sizes
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(bottleneck_channels, out_channels, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        # MaxPool branch
        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.batchnorm = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * (len(kernel_sizes) + 1), kernel_size=1),
                nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
            )

    def forward(self, x):
        # x: (batch, seq_len, features) -> for conv: (batch, features, seq_len)
        x_in = x.transpose(1, 2)
        if hasattr(self, 'bottleneck') and not isinstance(self.bottleneck, nn.Identity):
            x_b = self.bottleneck(x_in)
        else:
            x_b = x_in
        branches = [conv(x_b) for conv in self.conv_branches]
        branches.append(self.maxpool_branch(x_in))
        x_cat = torch.cat(branches, dim=1)
        x_cat = self.batchnorm(x_cat)
        if self.use_residual:
            x_res = self.residual(x_in)
            x_cat = x_cat + x_res
        x_cat = self.activation(x_cat)
        x_cat = self.dropout(x_cat)
        return x_cat.transpose(1, 2)

