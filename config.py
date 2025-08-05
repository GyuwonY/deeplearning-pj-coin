DAYCANDLE_DIR = 'daycandle'
FEATURE_COLS = [
    'price_change_rate',
    'volume_change_rate',
    'EMA_12',
    'SMA_20',
    'SMA_60',
    'ADX',
    'OBV',
    'MACD',
    'MACDSIGNAL',
    'MACDHIST',
    'UpperBand',
    'MiddleBand',
    'LowerBand',
    'RSI',
    'MFI',
    'CCI',
    'STOCH_K',
    'STOCH_D',
    'ROC',
    'ATR',
    'CMF'
]
TARGET_COL = 'price_change_rate'

# --- Data Splitting Configuration ---
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# --- Model & Sequence Configuration ---
WINDOW_SIZE = 60
PREDICTION_HORIZONS = [1, 3, 7]

# --- Training Configuration ---
BATCH_SIZE = 32
NUM_TRAIN_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6

# --- Directories ---
OUTPUT_DIR = "./patchtst_results"
LOGGING_DIR = "./patchtst_logs"
