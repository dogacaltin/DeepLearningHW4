"""Entry point. Run a specific part or all of them:

    python main.py --part ab
    python main.py --part c
    python main.py --part d
    python main.py --part bonus
    python main.py --part all
"""
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

import parameters as P
import datasets as D
import train as TR
import test as TE
from models import StockLSTM, StockGRU, BiLSTM, BiGRU, TXEncoder, RXDecoder


class Tee:
    """Mirror stdout to both the console and a log file."""
    def __init__(self, path):
        self.file = open(path, "w"); self.stdout = sys.stdout
    def write(self, s): self.stdout.write(s); self.file.write(s)
    def flush(self): self.stdout.flush(); self.file.flush()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loaders(raw, builder, cls=False):
    Xtr, ytr, Xva, yva, Xte, yte = D.build_split(raw, builder)
    tr = DataLoader(D.SeqDataset(Xtr, ytr, cls), batch_size=P.BATCH, shuffle=True)
    va = DataLoader(D.SeqDataset(Xva, yva, cls), batch_size=P.BATCH)
    te = DataLoader(D.SeqDataset(Xte, yte, cls), batch_size=P.BATCH)
    return tr, va, te, Xtr.shape[2], (Xtr, ytr, Xte, yte)


def run_ab(raw, device):
    print("\n===== Part (a)+(b): exact d-day returns =====")
    tr, va, te, F, _ = _loaders(raw, D.make_windows)
    for Model, name, fname in [(StockLSTM, "LSTM", "lstm_model.pt"),
                               (StockGRU, "GRU", "gru_model.pt")]:
        m = Model(F).to(device)
        best_val, _, _ = TR.fit_regression(m, tr, va, device, name)
        import torch.nn as nn
        test_mse = TE.evaluate_reg(m, te, nn.MSELoss(), device)
        print(f"[{name}] Best Val {best_val:.6f} | Test {test_mse:.6f}")
        torch.save(m.state_dict(), fname)


def run_c(raw, device):
    print("\n===== Part (c): rolling-average returns =====")
    tr, va, te, F, _ = _loaders(raw, D.make_rolling_windows)
    import torch.nn as nn
    for Model, name in [(StockLSTM, "LSTM"), (StockGRU, "GRU")]:
        m = Model(F).to(device)
        best_val, _, vc = TR.fit_regression(m, tr, va, device, name)
        test_mse = TE.evaluate_reg(m, te, nn.MSELoss(), device)
        std = np.std(vc[-10:])
        print(f"[{name}] Best Val {best_val:.6f} | Test {test_mse:.6f} | "
              f"val-std(last10) {std:.6f}")


def run_d(raw, device):
    print("\n===== Part (d): turning-point detection =====")
    tr, va, te, F, (Xtr, ytr, _, yte) = _loaders(raw, D.make_turning_windows, cls=True)
    print(f"Buy ratio -> Train {ytr.mean():.3f} | Test {yte.mean():.3f}")
    pos_weight = D.compute_pos_weight(ytr)
    print(f"pos_weight: {pos_weight.item():.2f}")
    for Model, name in [(BiLSTM, "BiLSTM"), (BiGRU, "BiGRU")]:
        m = Model(F).to(device)
        crit, _, _ = TR.fit_classification(m, tr, va, device, pos_weight, name)
        _, vp, vl = TE.evaluate_cls(m, va, crit, device)
        thr, _ = TE.best_threshold(vp, vl)
        _, tp, tl = TE.evaluate_cls(m, te, crit, device)
        preds = (tp >= thr).astype(int)
        print(f"\n[{name}] threshold {thr:.2f} | Test AUC {TE.auc(tp, tl):.3f}")
        print(classification_report(tl, preds, target_names=["Pass", "Buy"],
                                    digits=4, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(tl, preds))


def run_bonus(device):
    print("\n===== Part 2 (bonus): interactive AWGN communication =====")
    tr = DataLoader(TensorDataset(D.generate_messages(P.N_TRAIN)), batch_size=P.B_BATCH, shuffle=True)
    va = DataLoader(TensorDataset(D.generate_messages(P.N_VAL)),   batch_size=P.B_BATCH)
    te = DataLoader(TensorDataset(D.generate_messages(P.N_TEST)),  batch_size=P.B_BATCH)

    enc, dec = TXEncoder().to(device), RXDecoder().to(device)
    params = list(enc.parameters()) + list(dec.parameters())
    print("Total parameters:", f"{sum(p.numel() for p in params):,}")
    opt = torch.optim.Adam(params, lr=P.B_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=P.B_EPOCHS)

    best_val, best = float("inf"), None
    for ep in range(1, P.B_EPOCHS + 1):
        tl = TR.train_epoch_comm(enc, dec, tr, opt, device)
        vl, sa, ma = TE.eval_comm(enc, dec, va, device)
        sched.step()
        if vl < best_val:
            best_val = vl
            best = ({k: v.clone() for k, v in enc.state_dict().items()},
                    {k: v.clone() for k, v in dec.state_dict().items()})
        if ep % 10 == 0 or ep == 1:
            print(f"Epoch {ep:3d} | Train {tl:.4f} | Val {vl:.4f} | Sym {sa:.4f} | Msg {ma:.4f}")
    enc.load_state_dict(best[0]); dec.load_state_dict(best[1])
    _, sa, ma = TE.eval_comm(enc, dec, te, device)
    print(f"\nTest -> Symbol {sa:.4f} ({sa*100:.1f}%) | Message {ma:.4f} ({ma*100:.1f}%)")
    torch.save({"encoder": best[0], "decoder": best[1]}, "comm_protocol.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["ab", "c", "d", "bonus", "all"], default="all")
    args = ap.parse_args()

    sys.stdout = Tee(f"output_{args.part}.txt")
    device = get_device()
    print("Device:", device)

    if args.part in ("ab", "c", "d", "all"):
        raw = D.download_data()
        if args.part in ("ab", "all"):    run_ab(raw, device)
        if args.part in ("c", "all"):     run_c(raw, device)
        if args.part in ("d", "all"):     run_d(raw, device)
    if args.part in ("bonus", "all"):
        run_bonus(device)


if __name__ == "__main__":
    main()
