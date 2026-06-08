"""Data download, chronological split, scaling, and sliding-window builders."""
import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

import parameters as P


def download_data(tickers=P.TICKERS, verbose=True):
    data = {}
    for t in tickers:
        df = yf.download(t, start="2020-01-01", end="2025-12-31",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):          # flatten new yfinance columns
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        data[t] = df
        if verbose:
            print(f"{t}: {len(df)} trading days")
    return data


def split_and_scale(df):
    train = df[df.index <= P.TRAIN_END]
    val   = df[(df.index > P.TRAIN_END) & (df.index <= P.VAL_END)]
    test  = df[df.index > P.VAL_END]
    scaler = StandardScaler()
    train_sc = scaler.fit_transform(train.values)          # fit on train only
    val_sc   = scaler.transform(val.values)
    test_sc  = scaler.transform(test.values)
    return train_sc, val_sc, test_sc, train, val, test


def make_windows(scaled, raw_df, lookback=P.T, horizons=P.D):
    """Exact d-day return target (Part b). Window includes the reference day i."""
    close = raw_df["Close"].values
    X, y = [], []
    for i in range(lookback - 1, len(scaled) - horizons):
        X.append(scaled[i - lookback + 1 : i + 1])
        p_t = close[i]
        y.append([(close[i + d] - p_t) / p_t for d in range(1, horizons + 1)])
    return np.array(X, np.float32), np.array(y, np.float32)


def make_rolling_windows(scaled, raw_df, lookback=P.T, horizons=P.D, l=P.L):
    """Weighted rolling-average return target (Part c)."""
    w = np.array([1.0 / (l + 1)] * (l + 1))
    close = raw_df["Close"].values
    X, y = [], []
    for i in range(lookback - 1, len(scaled) - horizons):
        X.append(scaled[i - lookback + 1 : i + 1])
        p_t = close[i]
        row = []
        for d in range(1, horizons + 1):
            prices = close[i + d - l : i + d + 1]           # [p_{t+d-l} .. p_{t+d}]
            row.append((np.dot(w, prices[::-1]) - p_t) / p_t)
        y.append(row)
    return np.array(X, np.float32), np.array(y, np.float32)


def make_turning_windows(scaled, raw_df, lookback=P.T, horizons=P.D,
                         thr=P.RETURN_THRESHOLD):
    """Buy/pass label (Part d): 1 if any d-day max-price return exceeds thr."""
    high, close = raw_df["High"].values, raw_df["Close"].values
    X, y = [], []
    for i in range(lookback - 1, len(scaled) - horizons):
        X.append(scaled[i - lookback + 1 : i + 1])
        p_t, label = close[i], 0
        for d in range(1, horizons + 1):
            if (high[i + d] - p_t) / p_t > thr:
                label = 1
                break
        y.append(label)
    return np.array(X, np.float32), np.array(y, np.float32)


def build_split(raw, builder):
    """Apply a window builder to every ticker and concatenate the splits."""
    Xtr, ytr, Xva, yva, Xte, yte = [], [], [], [], [], []
    for tk in P.TICKERS:
        tr_sc, v_sc, te_sc, tr_raw, v_raw, te_raw = split_and_scale(raw[tk])
        a, b = builder(tr_sc, tr_raw); Xtr.append(a); ytr.append(b)
        a, b = builder(v_sc,  v_raw);  Xva.append(a); yva.append(b)
        a, b = builder(te_sc, te_raw); Xte.append(a); yte.append(b)
    return (np.concatenate(Xtr), np.concatenate(ytr),
            np.concatenate(Xva), np.concatenate(yva),
            np.concatenate(Xte), np.concatenate(yte))


class SeqDataset(Dataset):
    """Sequence dataset; set cls=True for binary classification targets."""
    def __init__(self, X, y, cls=False):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y).unsqueeze(1) if cls else torch.tensor(y)
    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def compute_pos_weight(y):
    pos = y.sum(); neg = len(y) - pos
    return torch.tensor(neg / pos if pos > 0 else 1.0, dtype=torch.float32)


def generate_messages(n):
    """Random messages from {0..ALPHABET-1}^SEQ_LEN (Part 2)."""
    return torch.randint(0, P.ALPHABET, (n, P.SEQ_LEN))
