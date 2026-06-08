"""Generate all report figures into ./visualizations using the shared modules."""
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import parameters as P
import datasets as D
import train as TR
import test as TE
from models import StockLSTM, StockGRU, BiLSTM, BiGRU, TXEncoder, RXDecoder, run_communication

OUT = "visualizations"
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reduced bonus settings for fast figure generation (raise to match exact numbers)
B_EPOCHS, B_NTRAIN = 60, 20000


def loss_plot(tc, vc, title, fname, ylabel="Loss"):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(tc) + 1), tc, label="Train")
    plt.plot(range(1, len(vc) + 1), vc, label="Val")
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=150); plt.close()


def cm_plot(cm, title, fname):
    plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="Blues"); plt.colorbar()
    plt.xticks([0, 1], ["Pass", "Buy"]); plt.yticks([0, 1], ["Pass", "Buy"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=150); plt.close()


def main():
    raw = D.download_data()

    # Fig 1: price trajectories
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, tk in zip(axes, P.TICKERS):
        df = raw[tk]
        tr = df[df.index <= P.TRAIN_END]
        va = df[(df.index > P.TRAIN_END) & (df.index <= P.VAL_END)]
        te = df[df.index > P.VAL_END]
        ax.plot(tr.index, tr["Close"], "tab:blue", label="Train")
        ax.plot(va.index, va["Close"], "tab:orange", label="Val")
        ax.plot(te.index, te["Close"], "tab:red", label="Test")
        ax.set_title(tk); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_price_trajectories.png"), dpi=150); plt.close()

    # Part b
    tr, va, te, F = _loaders(raw, D.make_windows)
    lstm = StockLSTM(F).to(DEV); _, tc, vc = TR.fit_regression(lstm, tr, va, DEV, "LSTM")
    loss_plot(tc, vc, "Part (b) StockLSTM - MSE Loss", "fig2_lstm_loss_partb.png", "MSE")
    lstm_h = TE.per_horizon_mse(lstm, te, DEV)
    gru = StockGRU(F).to(DEV); _, tc, vc = TR.fit_regression(gru, tr, va, DEV, "GRU")
    loss_plot(tc, vc, "Part (b) StockGRU - MSE Loss", "fig3_gru_loss_partb.png", "MSE")
    gru_h = TE.per_horizon_mse(gru, te, DEV)
    ds, w = np.arange(1, P.D + 1), 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(ds - w / 2, lstm_h, w, label="LSTM"); plt.bar(ds + w / 2, gru_h, w, label="GRU")
    plt.xlabel("Horizon d"); plt.ylabel("Test MSE"); plt.title("Part (b) per-horizon MSE")
    plt.xticks(ds); plt.legend(); plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig4_perhorizon_mse_partb.png"), dpi=150); plt.close()

    # Part c
    tr2, va2, te2, _ = _loaders(raw, D.make_rolling_windows)
    lr = StockLSTM(F).to(DEV); TR.fit_regression(lr, tr2, va2, DEV, "LSTM-roll")
    gr = StockGRU(F).to(DEV);  TR.fit_regression(gr, tr2, va2, DEV, "GRU-roll")
    lh, gh = TE.per_horizon_mse(lr, te2, DEV), TE.per_horizon_mse(gr, te2, DEV)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, ex, rl, nm in [(ax[0], lstm_h, lh, "LSTM"), (ax[1], gru_h, gh, "GRU")]:
        a.bar(ds - w / 2, ex, w, label="Exact return"); a.bar(ds + w / 2, rl, w, label="Rolling return")
        a.set_title(f"{nm}: Exact vs Rolling MSE"); a.set_xlabel("Horizon d"); a.set_ylabel("Test MSE")
        a.set_xticks(ds); a.legend(); a.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig5_exact_vs_rolling_mse.png"), dpi=150); plt.close()

    # Part d
    tr3, va3, te3, F3, ytr = _loaders_cls(raw)
    pos_weight = D.compute_pos_weight(ytr)
    for Model, lossf, cmf, nm in [(BiLSTM, "fig6_bilstm_loss_partd.png", "fig8_confusion_bilstm.png", "Bi-LSTM"),
                                  (BiGRU,  "fig7_bigru_loss_partd.png",  "fig9_confusion_bigru.png",  "Bi-GRU")]:
        m = Model(F3).to(DEV)
        crit, tc, vc = TR.fit_classification(m, tr3, va3, DEV, pos_weight, nm)
        loss_plot(tc, vc, f"Part (d) {nm} - BCE Loss", lossf, "BCE Loss")
        _, vp, vl = TE.evaluate_cls(m, va3, crit, DEV); thr, _ = TE.best_threshold(vp, vl)
        _, tp, tl = TE.evaluate_cls(m, te3, crit, DEV)
        cm_plot(confusion_matrix(tl, (tp >= thr).astype(int)),
                f"Turning Point - {nm} (thr={thr:.2f})", cmf)

    bonus_figures()
    print("All figures saved in ./visualizations")


def _loaders(raw, builder):
    Xtr, ytr, Xva, yva, Xte, yte = D.build_split(raw, builder)
    return (DataLoader(D.SeqDataset(Xtr, ytr), batch_size=P.BATCH, shuffle=True),
            DataLoader(D.SeqDataset(Xva, yva), batch_size=P.BATCH),
            DataLoader(D.SeqDataset(Xte, yte), batch_size=P.BATCH), Xtr.shape[2])


def _loaders_cls(raw):
    Xtr, ytr, Xva, yva, Xte, yte = D.build_split(raw, D.make_turning_windows)
    return (DataLoader(D.SeqDataset(Xtr, ytr, True), batch_size=P.BATCH, shuffle=True),
            DataLoader(D.SeqDataset(Xva, yva, True), batch_size=P.BATCH),
            DataLoader(D.SeqDataset(Xte, yte, True), batch_size=P.BATCH), Xtr.shape[2], ytr)


def bonus_figures():
    gen = lambda n: torch.randint(0, P.ALPHABET, (n, P.SEQ_LEN))
    tr = DataLoader(TensorDataset(gen(B_NTRAIN)), batch_size=P.B_BATCH, shuffle=True)
    va = DataLoader(TensorDataset(gen(P.N_VAL)),  batch_size=P.B_BATCH)
    te = DataLoader(TensorDataset(gen(P.N_TEST)), batch_size=P.B_BATCH)
    enc, dec = TXEncoder().to(DEV), RXDecoder().to(DEV)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=P.B_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=B_EPOCHS)
    train_c, ser, mer = [], [], []
    best_val, best = float("inf"), None
    for ep in range(B_EPOCHS):
        tl = TR.train_epoch_comm(enc, dec, tr, opt, DEV); sched.step()
        train_c.append(tl)
        vl, sa, ma = TE.eval_comm(enc, dec, va, DEV)
        ser.append(1 - sa); mer.append(1 - ma)
        if vl < best_val:
            best_val = vl
            best = ({k: v.clone() for k, v in enc.state_dict().items()},
                    {k: v.clone() for k, v in dec.state_dict().items()})
    enc.load_state_dict(best[0]); dec.load_state_dict(best[1])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(1, B_EPOCHS + 1), train_c); ax[0].set_title("Training Loss (Cross-Entropy)")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].grid(alpha=0.3)
    ax[1].plot(range(1, B_EPOCHS + 1), ser, label="Symbol Error Rate")
    ax[1].plot(range(1, B_EPOCHS + 1), mer, label="Message Error Rate")
    ax[1].set_title("Validation Error Rates"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Rate")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig10_bonus_training_curves.png"), dpi=150); plt.close()

    snr_db = np.array([-5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15])
    sers, mers = [], []
    for s in snr_db:
        _, sa, ma = TE.eval_comm(enc, dec, te, DEV, sigma2=10 ** (-s / 10))
        sers.append(max(1 - sa, 1e-5)); mers.append(max(1 - ma, 1e-5))
    plt.figure(figsize=(6, 4))
    plt.semilogy(snr_db, sers, "o-", label="SER"); plt.semilogy(snr_db, mers, "s--", label="MER")
    plt.xlabel("SNR (dB)"); plt.ylabel("Error Rate (log scale)")
    plt.title("Error Rate vs SNR - Interactive AWGN System")
    plt.legend(); plt.grid(alpha=0.3, which="both"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig11_bonus_snr_sweep.png"), dpi=150); plt.close()


if __name__ == "__main__":
    main()
