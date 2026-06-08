# CS515 HW4 — Sequence Modeling for Financial Forecasting and Interactive Communication

Deep learning experiments for CS515 (Sabancı University): financial return
forecasting with LSTM/GRU networks and a Transformer-based interactive
communication system over an AWGN channel with feedback.

## Project structure

```
.
├── parameters.py        # all hyperparameters / configuration
├── datasets.py          # data download, split, scaling, sliding-window builders
├── models/
│   ├── __init__.py
│   ├── recurrent.py     # StockLSTM, StockGRU, MAFeatureAug, BiLSTM, BiGRU
│   └── transformer.py   # PositionalEncoding, TXEncoder, RXDecoder, channel sim
├── train.py             # training loops (regression / classification / comm)
├── test.py              # evaluation and metric helpers
├── main.py              # entry point (runs the experiments)
├── visualizations.py    # generates all report figures
├── visualizations/      # output figures land here
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

(Parts a–d download data via `yfinance`, so an internet connection is required.)

## Usage

```bash
python main.py --part ab       # Parts (a) + (b): exact d-day returns
python main.py --part c        # Part (c): rolling-average returns
python main.py --part d        # Part (d): turning-point detection
python main.py --part bonus    # Part 2: interactive AWGN communication
python main.py --part all      # everything

python visualizations.py       # generate all figures into visualizations/
```

Each run writes its console log to `output_<part>.txt`, and trained weights to
`*.pt`.

## Experimental setup

- Tickers: AAPL, MSFT, GOOGL (2020–2025), OHLC features, lookback T = 20
- Split: train (2020–Jul 2024), validation (Aug–Dec 2024), test (2025)
- Optimizer: AdamW, lr 1e-3; MSE (regression) / weighted BCE (classification)
- Part 2: T = 4 rounds, sigma^2 = 0.25, alphabet {1..8}, message length 4

## Results (test set)

| Part | Result |
|------|--------|
| (b) exact returns | LSTM MSE 0.001114, GRU MSE 0.001104 |
| (c) rolling average | MSE ~0.00047 (lower and more stable than exact) |
| (d) turning point | Bi-LSTM / Bi-GRU AUC ~0.58 (5% threshold, validation-tuned) |
| 2 (bonus) | Symbol accuracy 99.6%, message accuracy 98.4% at sigma^2 = 0.25 |
