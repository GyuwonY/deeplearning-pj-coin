import numpy as np
import torch
from torch.utils.data import Dataset

import config

def create_sequences(features_np_array, coin_ids_series, window_size, prediction_horizons):
    X_seq, y_seq, coin_ids_seq = [], [], []
    max_horizon = max(prediction_horizons)

    for symbol_id in coin_ids_series.unique():
        features_for_symbol = features_np_array[coin_ids_series.values == symbol_id]

        if len(features_for_symbol) < window_size + max_horizon:
            continue

        for i in range(len(features_for_symbol) - window_size - max_horizon + 1):
            X_seq.append(features_for_symbol[i : i + window_size, :])
            
            y_targets_for_horizons = []
            for h in prediction_horizons:
                y_targets_for_horizons.append(features_for_symbol[i + window_size + h - 1, :])
            y_seq.append(y_targets_for_horizons)
            
            coin_ids_seq.append(symbol_id)

    return np.array(X_seq), np.array(y_seq), np.array(coin_ids_seq)

class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for time series data.
    It combines the time-varying features with the static coin ID feature
    into a single `past_values` tensor.
    """
    def __init__(self, X_data, y_data, id_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)
        # Ensure id_data is a tensor
        id_data_tensor = torch.tensor(id_data, dtype=torch.float32)
        # Reshape id_data to be (num_samples, 1, 1) for broadcasting
        self.id_seq = id_data_tensor.reshape(-1, 1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        past_features = self.X[idx] # Shape: (window_size, num_features)
        future_features = self.y[idx] # Shape: (prediction_length, num_features)
        
        # Get the coin_id for the current sample, shape (1, 1)
        coin_id_tensor = self.id_seq[idx]

        # Expand coin_id to match the sequence lengths of past and future features
        # Shape becomes (window_size, 1)
        past_static_feature = coin_id_tensor.expand(past_features.shape[0], -1)
        # Shape becomes (prediction_length, 1)
        future_static_feature = coin_id_tensor.expand(future_features.shape[0], -1)

        # Concatenate the coin_id feature as the first channel
        # Shape: (window_size, 1 + num_features)
        past_values_combined = torch.cat([past_static_feature, past_features], dim=-1)
        # Shape: (prediction_length, 1 + num_features)
        future_values_combined = torch.cat([future_static_feature, future_features], dim=-1)

        return {
            "past_values": past_values_combined,
            "future_values": future_values_combined,
        }