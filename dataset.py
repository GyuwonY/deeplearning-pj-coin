import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, features_data, target_data):
        self.features = features_data
        self.targets = target_data

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Convert numpy arrays to float32 tensors, which is expected by the trainer for fp16 training.
        return {
            "past_values": torch.tensor(self.features[idx], dtype=torch.float32),
            "future_values": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


def create_sequences(
    features_np_array: np.ndarray,
    prices_np_array: np.ndarray,
    coin_ids_series: np.ndarray,
    window_size: int,
    prediction_length: int,
):
    X_seq, y_seq = [], []
    unique_coin_ids = np.unique(coin_ids_series)
    for symbol_id in unique_coin_ids:
        coin_mask = coin_ids_series == symbol_id
        features_for_symbol = features_np_array[coin_mask]
        prices_for_symbol = prices_np_array[coin_mask]

        if len(features_for_symbol) < window_size + prediction_length:
            continue

        for i in range(len(features_for_symbol) - window_size - prediction_length + 1):
            # Input sequence
            X_seq.append(features_for_symbol[i : i + window_size])

            # Target sequence based on cumulative return
            start_price = prices_for_symbol[i + window_size - 1]
            
            future_prices = prices_for_symbol[
                i + window_size : i + window_size + prediction_length
            ]
            
            # Calculate cumulative return, handle potential division by zero
            if start_price == 0:
                cumulative_returns = np.zeros_like(future_prices, dtype=float)
            else:
                cumulative_returns = (future_prices / start_price) - 1
            
            y_seq.append(cumulative_returns.reshape(-1, 1))

    return np.array(X_seq), np.array(y_seq)

