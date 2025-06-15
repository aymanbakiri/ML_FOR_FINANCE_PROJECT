# models/nn_models.py

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

class NamedModule(nn.Module):
    MODEL_NAME: str  # must be overridden

    @property
    def name(self) -> str:
        return self.MODEL_NAME


class EnhancedLSTM(NamedModule):
    """
    Bidirectional LSTM with LayerNorm and a small head.
    """

    MODEL_NAME = "enhancedLSTM"
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


class SmallLSTM(NamedModule):
    """
    Simple unidirectional LSTM for 6‐month sequences.
    """

    MODEL_NAME = "smallLSTM"
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



class LargeLSTM(NamedModule):
    """
    Bidirectional LSTM with deeper head.
    """

    MODEL_NAME = "largeLSTM"
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


class StockTransformer(NamedModule):
    """
    Transformer‐based classifier over time‐series window.
    """

    MODEL_NAME = "stockTramsformer"
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


class InceptionTime(NamedModule):
    """
    InceptionTime network for time‐series classification.
    """

    MODEL_NAME = "inceptionTime"
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




def train(model, dl_train, dl_val, optimizer, scheduler, criterion, device, n_epochs=101, patience=-1, path):
    best_val_auc = 0.0
    trials = 0
    clip_grad = 1.0
    stop = 0
    losses = [100.0]
    for epoch in range(1, n_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(dl_train.dataset)

        # validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                probs = torch.sigmoid(model(xb)).cpu().numpy()
                preds.extend(probs)
                trues.extend(yb.numpy())
        val_auc = roc_auc_score(trues, preds)
        scheduler.step(val_auc)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc, trials = val_auc, 0
            torch.save(model.state_dict(), f"{path}/{model.name}.pth")
        else:
            trials += 1
            if trials >= patience and patience > 0 :
                print(f"Early stopping at epoch {epoch}")
                break

    


