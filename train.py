"""Training loops for regression, classification and the communication system."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import parameters as P
from models import run_communication
from test import evaluate_reg, evaluate_cls


def train_epoch(model, loader, opt, crit, device):
    """One epoch for any single-input model (regression or classification)."""
    model.train(); total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


def fit_regression(model, tr, va, device, name="model"):
    opt, crit = torch.optim.AdamW(model.parameters(), lr=P.LR), nn.MSELoss()
    best_val, best_state, tc, vc = float("inf"), None, [], []
    for ep in range(1, P.EPOCHS + 1):
        tl = train_epoch(model, tr, opt, crit, device)
        vl = evaluate_reg(model, va, crit, device)
        tc.append(tl); vc.append(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0:
            print(f"[{name}] Epoch {ep:3d} | Train {tl:.6f} | Val {vl:.6f}")
    model.load_state_dict(best_state)
    return best_val, tc, vc


def fit_classification(model, tr, va, device, pos_weight, name="model"):
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    opt  = torch.optim.AdamW(model.parameters(), lr=P.LR)
    best_val, best_state, tc, vc = float("inf"), None, [], []
    for ep in range(1, P.EPOCHS + 1):
        tl = train_epoch(model, tr, opt, crit, device)
        vl, _, _ = evaluate_cls(model, va, crit, device)
        tc.append(tl); vc.append(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0:
            print(f"[{name}] Epoch {ep:3d} | Train {tl:.6f} | Val {vl:.6f}")
    model.load_state_dict(best_state)
    return crit, tc, vc


def train_epoch_comm(encoder, decoder, loader, opt, device):
    encoder.train(); decoder.train(); total = 0
    params = list(encoder.parameters()) + list(decoder.parameters())
    for (m,) in loader:
        m = m.to(device)
        opt.zero_grad()
        logits = run_communication(encoder, decoder, m)
        loss = F.cross_entropy(logits.view(-1, P.ALPHABET), m.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        total += loss.item() * len(m)
    return total / len(loader.dataset)
