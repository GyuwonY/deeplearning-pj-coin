import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    PatchTSMixerConfig,
    PatchTSMixerForPrediction,
)
import optuna

import config
import data_utils
import dataset
from train import CustomTrainer, compute_metrics


def objective(trial: optuna.Trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True)
    d_model = trial.suggest_categorical("d_model", [512, 1024])
    num_mixer_layers = trial.suggest_int("num_mixer_layers", 2, 6)
    expansion_factor = trial.suggest_categorical("expansion_factor", [1, 2, 4])
    patch_len = trial.suggest_categorical("patch_len", [5, 6, 10, 12])
    stride_strategy = trial.suggest_categorical("stride_strategy", ["full", "half"])
    patch_stride = patch_len if stride_strategy == "full" else patch_len // 2
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)


    # --- Data Preparation for Tuning ---
    train_df, val_df = data_utils.prepare_tuning_data(max_rows_per_coin=1500)

    if train_df is None or val_df is None:
        raise optuna.exceptions.TrialPruned("No coins with sufficient data for this trial.")

    # The data is already scaled. We just need the column values.
    X_train_scaled = train_df[config.FEATURE_COLS].to_numpy()
    X_val_scaled = val_df[config.FEATURE_COLS].to_numpy()

    prediction_length = max(config.PREDICTION_HORIZONS)
    X_train_seq, y_train_seq = dataset.create_sequences(
        X_train_scaled, train_df[config.TARGET_COL].to_numpy(),
        train_df["coin_id"].to_numpy(), config.WINDOW_SIZE, prediction_length
    )
    X_val_seq, y_val_seq = dataset.create_sequences(
        X_val_scaled, val_df[config.TARGET_COL].to_numpy(),
        val_df["coin_id"].to_numpy(), config.WINDOW_SIZE, prediction_length
    )

    train_dataset = dataset.TimeSeriesDataset(X_train_seq, y_train_seq)
    val_dataset = dataset.TimeSeriesDataset(X_val_seq, y_val_seq)

    model_config = PatchTSMixerConfig(
        num_input_channels=len(config.FEATURE_COLS),
        prediction_channel_indices=[config.FEATURE_COLS.index(config.TARGET_COL)],
        context_length=config.WINDOW_SIZE,
        prediction_length=prediction_length,
        patch_length=6,
        patch_stride=6,
        d_model=d_model,
        num_layers=8,
        expansion_factor=1,
        dropout=0.2,
        num_targets=3,
        loss="nll",
        distribution_output="student_t",
    )
    model = PatchTSMixerForPrediction(model_config)

    output_dir = f"./tmp_trainer/trial_{trial.number}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="nll",
        greater_is_better=False,
        report_to="none",
        fp16=True,
        dataloader_num_workers=4,
        warmup_steps=100,
        optim="adamw_torch",
        lr_scheduler_type="cosine_with_restarts",
        seed=config.SEED,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,
        early_stopping_threshold=0.001
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    try:
        trainer.train()
        eval_results = trainer.evaluate()
    except Exception as e:
        print(f"Trial pruned due to an error: {e}")
        raise optuna.exceptions.TrialPruned(f"Pruned due to error: {e}")
    
    return eval_results["eval_nll"]


def run_tuning(n_trials=50):
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        study_name="patchtsmixer_hyperparameter_tuning"
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=3600*3)

    print("Hyperparameter tuning finished.")
    print("Best trial:")
    best_trial = study.best_trial

    print(f"  Value (NLL): {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
    best_params_path = os.path.join(config.OUTPUT_DIR, "best_hyperparameters.txt")
    with open(best_params_path, "w") as f:
        f.write(str(best_trial.params))
    print(f"Best hyperparameters saved to {best_params_path}")

    return study


if __name__ == "__main__":
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs("./tmp_trainer", exist_ok=True)
    
    run_tuning(n_trials=100) # Run 100 trials
