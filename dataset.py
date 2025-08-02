import numpy as np
import torch
from torch.utils.data import Dataset

import config


def create_sequences(
    features_data: np.ndarray,
    target_data: np.ndarray,
    coin_ids_series: np.ndarray,
    window_size: int,
    prediction_length: int,
):
    """
    Create sequences of past features, future targets, and corresponding coin IDs.
    """
    X_seq, y_seq, coin_ids_seq = [], [], []

    unique_coin_ids = np.unique(coin_ids_series)

    for symbol_id in unique_coin_ids:
        # Get all data for the current coin
        coin_mask = coin_ids_series == symbol_id
        features_for_symbol = features_data[coin_mask]
        target_for_symbol = target_data[coin_mask]

        if len(features_for_symbol) < window_size + prediction_length:
            continue

        for i in range(len(features_for_symbol) - window_size - prediction_length + 1):
            # Past values: window_size sequence of features
            X_seq.append(features_for_symbol[i : i + window_size])

            # Future values: prediction_length sequence of the target variable
            y_seq.append(
                target_for_symbol[i + window_size : i + window_size + prediction_length]
            )

            # Static feature: coin_id
            coin_ids_seq.append(symbol_id)

    return np.array(X_seq), np.array(y_seq), np.array(coin_ids_seq)


class TimeSeriesDataset(Dataset):
    """
    Custom PyTorch Dataset for time series data, compatible with Hugging Face's PatchTST.
    """

    def __init__(self, X_data, y_data, id_data):
        self.past_values = torch.tensor(X_data, dtype=torch.float32)
        self.future_values = torch.tensor(y_data, dtype=torch.float32)
        # Static features are expected to be categorical
        self.static_categorical_features = torch.tensor(id_data, dtype=torch.long)

    def __len__(self):
        return len(self.past_values)

    def __getitem__(self, idx):
        return {
            "past_values": self.past_values[idx],
            "future_values": self.future_values[idx],
            "static_categorical_features": self.static_categorical_features[idx],
        }
