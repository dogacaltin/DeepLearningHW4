"""Transformer-based TX encoder and RX decoder for Part 2 (bonus)."""
import math
import torch
import torch.nn as nn

import parameters as P


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn  = nn.MultiheadAttention(P.D_MODEL, P.N_HEADS, dropout=P.B_DROPOUT, batch_first=True)
        self.ff    = nn.Sequential(nn.Linear(P.D_MODEL, P.D_FF), nn.ReLU(),
                                   nn.Linear(P.D_FF, P.D_MODEL))
        self.norm1 = nn.LayerNorm(P.D_MODEL)
        self.norm2 = nn.LayerNorm(P.D_MODEL)
        self.drop  = nn.Dropout(P.B_DROPOUT)
    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class TXEncoder(nn.Module):
    """Message symbols + past feedback -> one coded scalar per position."""
    def __init__(self):
        super().__init__()
        self.embed    = nn.Embedding(P.ALPHABET, P.D_MODEL)
        self.pre_mlp  = nn.Sequential(nn.Linear(P.D_MODEL + P.T_ROUNDS, P.D_MODEL), nn.ReLU(),
                                      nn.Linear(P.D_MODEL, P.D_MODEL))
        self.pe       = PositionalEncoding(P.D_MODEL)
        self.blocks   = nn.ModuleList([TransformerBlock() for _ in range(P.N_LAYERS)])
        self.post_mlp = nn.Linear(P.D_MODEL, 1)
    def forward(self, m_ids, feedback):
        z = self.pe(self.pre_mlp(torch.cat([self.embed(m_ids), feedback], dim=-1)))
        for b in self.blocks:
            z = b(z)
        coded = self.post_mlp(z).squeeze(-1)
        # Per-symbol average power = 1 (E[x_i^2] = 1). For total ||x||^2 <= 1 use:
        # coded = coded / (coded.norm(dim=-1, keepdim=True) + 1e-8)
        power = coded.pow(2).mean(dim=-1, keepdim=True) + 1e-8
        return coded / power.sqrt()


class RXDecoder(nn.Module):
    """All received signals -> per-position logits over the alphabet."""
    def __init__(self):
        super().__init__()
        self.pre_mlp = nn.Sequential(nn.Linear(P.T_ROUNDS, P.D_MODEL), nn.ReLU(),
                                     nn.Linear(P.D_MODEL, P.D_MODEL))
        self.pe      = PositionalEncoding(P.D_MODEL)
        self.blocks  = nn.ModuleList([TransformerBlock() for _ in range(P.N_LAYERS)])
        self.out_fc  = nn.Linear(P.D_MODEL, P.ALPHABET)
    def forward(self, received_all):
        z = self.pe(self.pre_mlp(received_all))
        for b in self.blocks:
            z = b(z)
        return self.out_fc(z)


def run_communication(encoder, decoder, m_ids, sigma2=P.SIGMA2):
    """Simulate T rounds: encode, AWGN channel, relay feedback, then decode once."""
    B = m_ids.size(0); device = m_ids.device
    feedback     = torch.zeros(B, P.SEQ_LEN, P.T_ROUNDS, device=device)
    received_all = torch.zeros(B, P.SEQ_LEN, P.T_ROUNDS, device=device)
    for t in range(P.T_ROUNDS):
        x_t = encoder(m_ids, feedback)
        y_t = x_t + torch.randn_like(x_t) * math.sqrt(sigma2)
        received_all[:, :, t] = y_t
        feedback[:, :, t]     = y_t              # noiseless relay feedback
    return decoder(received_all)
