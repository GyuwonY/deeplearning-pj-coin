import os
import pandas as pd
import talib
from sklearn.preprocessing import LabelEncoder, RobustScaler
import joblib

import config

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators for a given single-coin DataFrame.
    """
    df = df.copy()
    close = df['trade_price'].values
    high = df['high_price'].values
    low = df['low_price'].values
    volume = df['candle_acc_trade_volume'].values

    df['price_change_rate'] = df['trade_price'].pct_change()
    df['volume_change_rate'] = df['candle_acc_trade_volume'].pct_change()

    df['EMA_12'] = talib.EMA(close, timeperiod=12)
    df['SMA_20'] = talib.SMA(close, timeperiod=20)
    df['SMA_60'] = talib.SMA(close, timeperiod=60)
    df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    df['OBV'] = talib.OBV(close, volume)
    df['RSI'] = talib.RSI(close, timeperiod=14)
    df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
    df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
    df['ROC'] = talib.ROC(close, timeperiod=10)
    df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACDSIGNAL'] = macdsignal
    df['MACDHIST'] = macdhist

    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['UpperBand'] = upperband
    df['MiddleBand'] = middleband
    df['LowerBand'] = lowerband
    
    df = add_cmf(df)
    
    df.dropna(inplace=True)
    return df

def load_all_coin_data(directory: str) -> pd.DataFrame:
    """
    Loads all CSV files from a directory, performs basic preprocessing,
    and returns a single DataFrame.
    """
    df_list = []
    daycandle_files = os.listdir(directory)

    for file_name in daycandle_files:
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)

        df['candle_date_time_utc'] = pd.to_datetime(df['candle_date_time_utc'], errors='coerce')
        df.sort_values(by='candle_date_time_utc', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['coin_symbol'] = df['market'].str.replace('KRW-', '')
        df_list.append(df)

    processed_df = pd.concat(df_list).reset_index(drop=True)
    processed_df.set_index('candle_date_time_utc', inplace=True)

    le = LabelEncoder()
    processed_df['coin_id'] = le.fit_transform(processed_df['coin_symbol'])
    processed_df.sort_values(by=['coin_symbol', processed_df.index.name], inplace=True)
    
    return processed_df

def add_cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Calculates Chaikin Money Flow (CMF) and adds it to the DataFrame.
    """
    close_price = df['trade_price']
    low_price = df['low_price']
    high_price = df['high_price']
    volume = df['candle_acc_trade_volume']

    mfm = ((close_price - low_price) - (high_price - close_price)) / (high_price - low_price)
    mfm = mfm.fillna(0)

    mfv = mfm * volume
    
    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    df['CMF'] = cmf
    return df

def prepare_and_save_data():
    """
    Loads, processes, splits, scales, and saves the datasets for training and testing.
    This is the authoritative source for data preparation.
    """
    print("--- Starting Data Preparation ---")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    all_coin_data = load_all_coin_data(config.DAYCANDLE_DIR)

    train_dfs, val_dfs, test_dfs = [], [], []

    min_sequence_length = config.WINDOW_SIZE + max(config.PREDICTION_HORIZONS)
    min_total_length = int(min_sequence_length / (1 - config.TRAIN_RATIO - config.VAL_RATIO)) if (1 - config.TRAIN_RATIO - config.VAL_RATIO) > 0 else min_sequence_length

    coin_lengths = all_coin_data.groupby('coin_symbol').size()
    long_enough_coins = coin_lengths[coin_lengths >= min_total_length].index
    
    filtered_coins_df = all_coin_data[all_coin_data['coin_symbol'].isin(long_enough_coins)]
    print(f"Found {len(long_enough_coins)} coins with sufficient data for processing.")

    for coin_symbol in filtered_coins_df['coin_symbol'].unique():
        coin_df = filtered_coins_df[filtered_coins_df['coin_symbol'] == coin_symbol].copy()

        # First, add technical indicators to the entire dataset for the coin
        coin_df_with_indicators = add_technical_indicators(coin_df)

        # Then, split the data with indicators
        total_len = len(coin_df_with_indicators)
        if total_len < min_total_length: # Re-check length after adding indicators and dropping NaNs
            print(f"Skipping {coin_symbol} due to insufficient length after adding indicators.")
            continue

        train_len = int(total_len * config.TRAIN_RATIO)
        val_len = int(total_len * config.VAL_RATIO)

        train_part = coin_df_with_indicators.iloc[:train_len]
        val_part = coin_df_with_indicators.iloc[train_len : train_len + val_len]
        test_part = coin_df_with_indicators.iloc[train_len + val_len :]

        # Final check to ensure each part is usable
        if (len(train_part) < min_sequence_length or
            len(val_part) < max(config.PREDICTION_HORIZONS) or
            len(test_part) < max(config.PREDICTION_HORIZONS)):
            print(f"Skipping {coin_symbol} due to one of the splits being too short after processing.")
            continue

        train_dfs.append(train_part)
        val_dfs.append(val_part)
        test_dfs.append(test_part)

    if not train_dfs:
        raise ValueError("No coins with sufficient data to create datasets.")

    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)

    print(f"Final dataset sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    feature_scaler = RobustScaler()
    train_df[config.FEATURE_COLS] = feature_scaler.fit_transform(train_df[config.FEATURE_COLS])
    val_df[config.FEATURE_COLS] = feature_scaler.transform(val_df[config.FEATURE_COLS])
    test_df[config.FEATURE_COLS] = feature_scaler.transform(test_df[config.FEATURE_COLS])
    
    scaler_path = os.path.join(config.OUTPUT_DIR, "feature_scaler.joblib")
    joblib.dump(feature_scaler, scaler_path)
    print(f"Feature scaler saved to {scaler_path}")

    train_df.to_csv(os.path.join(config.OUTPUT_DIR, "train_set.csv"))
    val_df.to_csv(os.path.join(config.OUTPUT_DIR, "val_set.csv"))
    test_df.to_csv(os.path.join(config.OUTPUT_DIR, "test_set.csv"))
    print(f"Processed datasets saved to {config.OUTPUT_DIR}")
    print("--- Data Preparation Finished ---")

def prepare_tuning_data(max_rows_per_coin=1500):
    """
    Prepares a smaller, subsampled dataset for hyperparameter tuning.
    Does not save files, returns dataframes directly.
    Correctly splits before adding indicators to prevent lookahead bias.
    """
    all_coin_data = load_all_coin_data(config.DAYCANDLE_DIR)

    train_dfs, val_dfs = [], []

    min_sequence_length = config.WINDOW_SIZE + max(config.PREDICTION_HORIZONS)
    min_total_length = int(min_sequence_length / (1 - config.TRAIN_RATIO - config.VAL_RATIO))

    coin_lengths = all_coin_data.groupby('coin_symbol').size()
    long_enough_coins = coin_lengths[coin_lengths >= min_total_length].index
    
    filtered_coins_df = all_coin_data[all_coin_data['coin_symbol'].isin(long_enough_coins)]

    for coin_symbol in filtered_coins_df['coin_symbol'].unique():
        coin_df = filtered_coins_df[filtered_coins_df['coin_symbol'] == coin_symbol].copy()
        
        if len(coin_df) > max_rows_per_coin:
            coin_df = coin_df.iloc[-max_rows_per_coin:]

        # First, add technical indicators
        coin_df_with_indicators = add_technical_indicators(coin_df)

        # Then, split the data
        total_len = len(coin_df_with_indicators)
        if total_len < min_sequence_length: # Check after adding indicators
            continue

        train_len = int(total_len * config.TRAIN_RATIO)
        
        train_part = coin_df_with_indicators.iloc[:train_len]
        val_part = coin_df_with_indicators.iloc[train_len:]

        # Final check for sequence length
        if len(train_part) < min_sequence_length or len(val_part) < max(config.PREDICTION_HORIZONS) + config.WINDOW_SIZE:
            continue

        train_dfs.append(train_part)
        val_dfs.append(val_part)

    if not train_dfs:
        return None, None

    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)

    feature_scaler = RobustScaler()

    train_df[config.FEATURE_COLS] = feature_scaler.fit_transform(train_df[config.FEATURE_COLS])
    val_df[config.FEATURE_COLS] = feature_scaler.transform(val_df[config.FEATURE_COLS])

    return train_df, val_df
