"""Recurrent models for Parts (b), (c) and (d)."""
import torch
import torch.nn as nn

import parameters as P


class StockLSTM(nn.Module):
    """Stacked LSTM -> dropout -> linear head (5 horizons)."""
    def __init__(self, F):
        super().__init__()
        self.lstm = nn.LSTM(F, P.HIDDEN, P.LAYERS, batch_first=True, dropout=P.DROPOUT)
        self.drop = nn.Dropout(P.DROPOUT)
        self.fc   = nn.Linear(P.HIDDEN, P.D)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))


class StockGRU(nn.Module):
    """Stacked GRU -> dropout -> linear head (5 horizons)."""
    def __init__(self, F):
        super().__init__()
        self.gru  = nn.GRU(F, P.HIDDEN, P.LAYERS, batch_first=True, dropout=P.DROPOUT)
        self.drop = nn.Dropout(P.DROPOUT)
        self.fc   = nn.Linear(P.HIDDEN, P.D)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.drop(out[:, -1, :]))


class MAFeatureAug(nn.Module):
    """Depthwise 1D conv producing moving-average auxiliary features (F -> 2F)."""
    def __init__(self, F, kernel=5):
        super().__init__()
        self.conv = nn.Conv1d(F, F, kernel_size=kernel, padding=kernel // 2, groups=F)
    def forward(self, x):                       # x: (B, T, F)
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return torch.cat([x, z], dim=-1)        # (B, T, 2F)


class BiLSTM(nn.Module):
    """Bidirectional LSTM with MA feature augmentation (Part d, binary)."""
    def __init__(self, F):
        super().__init__()
        self.ma   = MAFeatureAug(F)
        self.lstm = nn.LSTM(F * 2, P.HIDDEN, P.LAYERS, batch_first=True,
                            dropout=P.DROPOUT, bidirectional=True)
        self.drop = nn.Dropout(P.DROPOUT)
        self.fc   = nn.Linear(P.HIDDEN * 2, 1)
    def forward(self, x):
        out, _ = self.lstm(self.ma(x))
        return self.fc(self.drop(out[:, -1, :]))


class BiGRU(nn.Module):
    """Bidirectional GRU with MA feature augmentation (Part d, binary)."""
    def __init__(self, F):
        super().__init__()
        self.ma   = MAFeatureAug(F)
        self.gru  = nn.GRU(F * 2, P.HIDDEN, P.LAYERS, batch_first=True,
                           dropout=P.DROPOUT, bidirectional=True)
        self.drop = nn.Dropout(P.DROPOUT)
        self.fc   = nn.Linear(P.HIDDEN * 2, 1)
    def forward(self, x):
        out, _ = self.gru(self.ma(x))
        return self.fc(self.drop(out[:, -1, :]))
