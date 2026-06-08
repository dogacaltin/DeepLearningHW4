"""Hyperparameters and configuration shared across all parts."""

# ---- Part 1: financial forecasting ----
TICKERS    = ["AAPL", "MSFT", "GOOGL"]
T          = 20          # lookback window
D          = 5           # number of forecast horizons (d = 1..5)
L          = 3           # rolling-average window (Part c)
BATCH      = 64
EPOCHS     = 30
LR         = 1e-3
HIDDEN     = 64
LAYERS     = 2
DROPOUT    = 0.2
TRAIN_END  = "2024-07-31"
VAL_END    = "2024-12-31"
RETURN_THRESHOLD = 0.05  # Part d gain threshold (5%); assignment's 10% is too rare

# ---- Part 2: interactive communication (bonus) ----
ALPHABET   = 8
SEQ_LEN    = 4
T_ROUNDS   = 4
SIGMA2     = 0.25
D_MODEL    = 64
N_HEADS    = 4
N_LAYERS   = 2
D_FF       = 128
B_DROPOUT  = 0.1
B_BATCH    = 256
B_EPOCHS   = 100
B_LR       = 1e-3
N_TRAIN    = 50000
N_VAL      = 5000
N_TEST     = 5000
