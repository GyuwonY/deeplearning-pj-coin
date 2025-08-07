import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import t as student_t_distribution
from torch.utils.data import DataLoader
from transformers import PatchTSMixerForPrediction

import config
import data_utils
import dataset
from train import CustomTrainer, compute_metrics


def analyze_predictions(test_results):
    """
    Analyzes the probabilistic predictions and prints insights.
    The predictions and labels are cumulative returns and are not scaled,
    so no inverse transformation is needed.
    """
    print("\n--- Probabilistic Prediction Analysis ---")

    # Extract distribution parameters and labels
    loc, scale, df = test_results.predictions
    labels = test_results.label_ids

    # Squeeze the last dimension which is of size 1
    loc = loc.squeeze(-1)
    scale = scale.squeeze(-1)
    df = df.squeeze(-1)
    labels = labels.squeeze(-1)

    # Ensure scale and df are positive using softplus
    scale = F.softplus(torch.tensor(scale)).numpy()
    df = F.softplus(torch.tensor(df)).numpy()

    # Select only the relevant prediction horizons
    prediction_indices = [h - 1 for h in config.PREDICTION_HORIZONS]
    loc_orig = loc[:, prediction_indices]
    scale_orig = scale[:, prediction_indices]
    df_orig = df[:, prediction_indices]
    labels_orig = labels[:, prediction_indices]

    # --- Analyze a few examples ---
    num_examples_to_show = 5
    print(f"\nShowing analysis for the first {num_examples_to_show} test samples...")

    for i in range(num_examples_to_show):
        print(f"\n--- Sample {i+1} ---")
        for j, horizon in enumerate(config.PREDICTION_HORIZONS):
            l, s, d = loc_orig[i, j], scale_orig[i, j], df_orig[i, j]
            true_value = labels_orig[i, j]

            # Use scipy's t distribution for stable cdf and ppf (icdf) calculations
            lower_bound = student_t_distribution.ppf(0.05, df=d, loc=l, scale=s)
            upper_bound = student_t_distribution.ppf(0.95, df=d, loc=l, scale=s)
            prob_increase = 1 - student_t_distribution.cdf(0.0, df=d, loc=l, scale=s)

            print(f"  Horizon: +{horizon} day(s)")
            print(f"    True Value (Cumulative Return): {true_value:+.4%}")
            print(f"    Prediction (Mean): {l:+.4%}")
            print(f"    90% Confidence Interval: [{lower_bound:+.4%}, {upper_bound:+.4%}]")
            print(f"    Probability of Price Increase (>0%): {prob_increase:.2%}")


def main():
    """
    Main function to load a trained model and evaluate it on the test set.
    """
    test_path = os.path.join(config.OUTPUT_DIR, "test_set.csv")
    
    print("--- Loading Pre-processed Test Data ---")
    try:
        test_df = pd.read_csv(test_path, index_col='candle_date_time_utc', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Test data not found at {test_path}. Please run train.py first.")
        return

    # The feature data in test_set.csv is already scaled.
    X_test_scaled = test_df[config.FEATURE_COLS].to_numpy()

    # The target (y) is calculated from the unscaled trade_price, which is in the test_df
    prediction_length = max(config.PREDICTION_HORIZONS)
    X_test_seq, y_test_seq = dataset.create_sequences(
        X_test_scaled,
        test_df[config.TARGET_COL].to_numpy(),
        test_df["coin_id"].to_numpy(),
        config.WINDOW_SIZE,
        prediction_length,
    )

    test_dataset = dataset.TimeSeriesDataset(X_test_seq, y_test_seq)

    print("Loading best model...")
    final_model_path = f"{config.OUTPUT_DIR}/best_model"
    
    try:
        model = PatchTSMixerForPrediction.from_pretrained(final_model_path)
        print(f"Successfully loaded model from {final_model_path}")
    except OSError:
        print(f"Error: Model not found at {final_model_path}")
        print("Please run train.py successfully to save the final model.")
        return

    # We need a trainer to run prediction
    trainer = CustomTrainer(model=model)

    print("Evaluating model on the test set...")
    test_results = trainer.predict(test_dataset)

    # --- Quantitative Metrics ---
    print("\n--- Quantitative Metrics (on Test Set) ---")
    metrics = compute_metrics(test_results)
    print(metrics)
    
    # --- Qualitative Analysis ---
    analyze_predictions(test_results)


if __name__ == "__main__":
    main()