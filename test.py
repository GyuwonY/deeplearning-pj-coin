import joblib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t as student_t_distribution
from torch.utils.data import DataLoader
from transformers import TrainingArguments, PatchTSTForPrediction

import config
import data_utils
import dataset
import model as model_utils
from train import CustomTrainer, compute_metrics


def analyze_predictions(test_results, target_scaler):
    """
    Analyzes the probabilistic predictions, inverse transforms them, and prints insights.
    """
    print("\n--- Probabilistic Prediction Analysis ---")

    # Extract distribution parameters and labels
    loc, scale, df = test_results.predictions
    labels = test_results.label_ids

    # The model outputs predictions for all features, select the target one.
    try:
        target_col_index = config.FEATURE_COLS.index(config.TARGET_COL)
    except ValueError:
        target_col_index = 0
    
    loc = loc[:, :, target_col_index]
    scale = scale[:, :, target_col_index]
    df = df[:, :, target_col_index]
    labels = labels.squeeze(-1) # Remove the last dimension

    # Ensure scale and df are positive using softplus
    scale = F.softplus(torch.tensor(scale)).numpy()
    df = F.softplus(torch.tensor(df)).numpy()

    # Select only the relevant prediction horizons
    prediction_indices = [h - 1 for h in config.PREDICTION_HORIZONS]
    loc = loc[:, prediction_indices]
    scale = scale[:, prediction_indices]
    df = df[:, prediction_indices]
    labels = labels[:, prediction_indices]

    # --- Inverse Transform to Original Scale ---
    loc_orig = target_scaler.inverse_transform(loc)
    scale_orig = scale * target_scaler.scale_
    df_orig = df
    labels_orig = target_scaler.inverse_transform(labels)

    # --- Analyze a few examples ---
    num_examples_to_show = 5
    print(f"\nShowing analysis for the first {num_examples_to_show} test samples...")

    for i in range(num_examples_to_show):
        print(f"\n--- Sample {i+1} ---")
        for j, horizon in enumerate(config.PREDICTION_HORIZONS):
            l, s, d = loc_orig[i, j], scale_orig[i, j], df_orig[i, j]
            true_value = labels_orig[i, j]

            # Use scipy's t distribution for stable cdf and ppf (icdf) calculations
            # ppf is the Percent Point Function, which is the inverse of the cdf.
            lower_bound = student_t_distribution.ppf(0.05, df=d, loc=l, scale=s)
            upper_bound = student_t_distribution.ppf(0.95, df=d, loc=l, scale=s)
            prob_increase = 1 - student_t_distribution.cdf(0.0, df=d, loc=l, scale=s)

            print(f"  Horizon: +{horizon} day(s)")
            print(f"    True Value: {true_value:+.4%}")
            print(f"    Prediction (Mean): {l:+.4%}")
            print(f"    90% Confidence Interval: [{lower_bound:+.4%}, {upper_bound:+.4%}]")
            print(f"    Probability of Price Increase (>0%): {prob_increase:.2%}")


def main():
    """
    Main function to load a trained model and evaluate it on the test set.
    """
    print("Loading test data...")
    processed_df = data_utils.load_and_process_data(config.DAYCANDLE_DIR)
    _, _, test_df, filtered_df = data_utils.filter_and_split_data(processed_df)

    print("Loading scalers...")
    try:
        feature_scaler = joblib.load(f"{config.OUTPUT_DIR}/feature_scaler.joblib")
        target_scaler = joblib.load(f"{config.OUTPUT_DIR}/target_scaler.joblib")
    except FileNotFoundError:
        print("Error: Scaler files not found. Please run train.py first.")
        return

    X_test_scaled = feature_scaler.transform(test_df[config.FEATURE_COLS])
    y_test_scaled = target_scaler.transform(test_df[[config.TARGET_COL]])

    prediction_length = max(config.PREDICTION_HORIZONS)
    X_test_seq, y_test_seq, test_coin_ids_seq = dataset.create_sequences(
        X_test_scaled, y_test_scaled, test_df["coin_id"].values,
        config.WINDOW_SIZE, prediction_length,
    )

    test_dataset = dataset.TimeSeriesDataset(X_test_seq, y_test_seq, test_coin_ids_seq)

    print("Loading best model...")
    final_model_path = f"{config.OUTPUT_DIR}/best_model"
    
    try:
        # Load the model from the final saved directory
        model = PatchTSTForPrediction.from_pretrained(final_model_path)
        print(f"Successfully loaded model from {final_model_path}")
    except OSError:
        print(f"Error: Model not found at {final_model_path}")
        print("Please run train.py successfully to save the final model.")
        return

    # We need to use the same CustomTrainer for prediction
    trainer = CustomTrainer(model=model)

    print("Evaluating model on the test set...")
    test_results = trainer.predict(test_dataset)

    # --- Quantitative Metrics ---
    metrics = compute_metrics(test_results)
    
    # --- Qualitative Analysis ---
    analyze_predictions(test_results, target_scaler)


if __name__ == "__main__":
    main()
