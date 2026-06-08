"""Evaluation and metric helpers (no training here)."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

import parameters as P
from models import run_communication


def evaluate_reg(model, loader, crit, device):
    model.eval(); total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            total += crit(model(X), y).item() * len(X)
    return total / len(loader.dataset)


def per_horizon_mse(model, loader, device):
    model.eval(); P_, T_ = [], []
    with torch.no_grad():
        for X, y in loader:
            P_.append(model(X.to(device)).cpu().numpy()); T_.append(y.numpy())
    P_, T_ = np.concatenate(P_), np.concatenate(T_)
    return ((P_ - T_) ** 2).mean(axis=0)            # (D,)


def evaluate_cls(model, loader, crit, device):
    """Return (loss, sigmoid probabilities, true labels)."""
    model.eval(); total = 0
    probs, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total += crit(logits, y).item() * len(X)
            probs.append(torch.sigmoid(logits).cpu())
            labels.append(y.cpu())
    p = torch.cat(probs).numpy().flatten()
    l = torch.cat(labels).numpy().flatten()
    return total / len(loader.dataset), p, l


def best_threshold(probs, labels):
    """Threshold that maximizes the positive-class F1 on validation."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def auc(probs, labels):
    return roc_auc_score(labels, probs)


def compute_accuracy(logits, targets):
    preds   = logits.argmax(dim=-1)
    sym_acc = (preds == targets).float().mean().item()
    msg_acc = (preds == targets).all(dim=-1).float().mean().item()
    return sym_acc, msg_acc


def eval_comm(encoder, decoder, loader, device, sigma2=P.SIGMA2):
    encoder.eval(); decoder.eval(); total = 0
    sa, ma = [], []
    with torch.no_grad():
        for (m,) in loader:
            m = m.to(device)
            logits = run_communication(encoder, decoder, m, sigma2)
            total += F.cross_entropy(logits.view(-1, P.ALPHABET), m.view(-1)).item() * len(m)
            s, mm = compute_accuracy(logits, m)
            sa.append(s); ma.append(mm)
    return total / len(loader.dataset), np.mean(sa), np.mean(ma)
