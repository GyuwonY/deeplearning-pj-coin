import os
import pandas as pd
import talib
from sklearn.preprocessing import LabelEncoder

import config

def load_and_process_data(directory):
    df_list = []
    daycandle_files = os.listdir(directory)

    for file_name in daycandle_files:
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)

        df['candle_date_time_utc'] = pd.to_datetime(df['candle_date_time_utc'], errors='coerce')
        df.sort_values(by='candle_date_time_utc', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['price_change_rate'] = df['trade_price'].pct_change()
        df['volume_change_rate'] = df['candle_acc_trade_volume'].pct_change()
        df['coin_symbol'] = df['market'].str.replace('KRW-', '')

        close = df['trade_price'].values
        high = df['high_price'].values
        low = df['low_price'].values
        volume = df['candle_acc_trade_volume'].values

        df['EMA_12'] = talib.EMA(close, timeperiod=12)
        df['SMA_60'] = talib.SMA(close, timeperiod=60)
        df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        df['OBV'] = talib.OBV(close, volume)
        df['RSI'] = talib.RSI(close, timeperiod=14)
        df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        

        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACDSIGNAL'] = macdsignal
        df['MACDHIST'] = macdhist

        upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['UpperBand'] = upperband
        df['MiddleBand'] = middleband
        df['LowerBand'] = lowerband

        df.dropna(inplace=True)
        df_list.append(df)

    processed_df = pd.concat(df_list).reset_index(drop=True)
    processed_df.set_index('candle_date_time_utc', inplace=True)

    le = LabelEncoder()
    processed_df['coin_id'] = le.fit_transform(processed_df['coin_symbol'])
    processed_df.sort_values(by=['coin_symbol', processed_df.index.name], inplace=True)
    
    return processed_df

def filter_and_split_data(df):
    min_sequence_length = config.WINDOW_SIZE + max(config.PREDICTION_HORIZONS)
    min_total_length_for_split = min_sequence_length / min(config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO)
    
    coin_lengths = df.groupby('coin_symbol').size()
    
    MIN_ROW_COUNT = 2700

    long_enough_coins = coin_lengths[(coin_lengths >= min_total_length_for_split) & (coin_lengths >= MIN_ROW_COUNT)].index
    
    filtered_df = df[df['coin_symbol'].isin(long_enough_coins)].copy()

    def split_data_by_coin(coin_df_group):
        total_len = len(coin_df_group)
        train_len = int(total_len * config.TRAIN_RATIO)
        val_len = int(total_len * config.VAL_RATIO)
        
        train_split = coin_df_group.iloc[:train_len]
        val_split = coin_df_group.iloc[train_len : train_len + val_len]
        test_split = coin_df_group.iloc[train_len + val_len :]
        
        return train_split, val_split, test_split

    train_dfs, val_dfs, test_dfs = [], [], []
    for _, group_df in filtered_df.groupby('coin_symbol'):
        # For faster hyperparameter tuning, use only the most recent data
        MAX_ROWS_PER_COIN = 1500
        if len(group_df) > MAX_ROWS_PER_COIN:
            group_df = group_df.iloc[-MAX_ROWS_PER_COIN:]

        train_part, val_part, test_part = split_data_by_coin(group_df)
        train_dfs.append(train_part)
        val_dfs.append(val_part)
        test_dfs.append(test_part)

    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)

    print(f"Total rows for training: {len(train_df)}")

    return train_df, val_df, test_df, filtered_df
