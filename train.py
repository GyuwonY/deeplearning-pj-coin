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
)

import config
import data_utils
import dataset
import model as model_utils
import os


import torch.nn.functional as F

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # store horizon indices (0-based)
        self.prediction_horizons = torch.tensor(
            [h - 1 for h in config.PREDICTION_HORIZONS]
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        future_values = inputs.pop("future_values")
        outputs = model(**inputs)
        loc, scale, df = outputs.prediction_outputs

        # Squeeze: [B, T, 1] → [B, T]
        loc, scale, df = (x.squeeze(-1) for x in (loc, scale, df))
        future_values = future_values.squeeze(-1)

        # Ensure positivity and stability
        scale = F.softplus(scale) + 1e-6
        df = F.softplus(df) + 2.0

        indices = self.prediction_horizons.to(loc.device)
        masked_loc = loc[:, indices]
        masked_scale = scale[:, indices]
        masked_df = df[:, indices]
        masked_labels = future_values[:, indices]

        masked_dist = dist.StudentT(df=masked_df, loc=masked_loc, scale=masked_scale)
        loss = -masked_dist.log_prob(masked_labels).mean()

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop("future_values")
        with torch.no_grad():
            outputs = model(**inputs)

        loc, scale, df = outputs.prediction_outputs
        loc, scale, df = (x.squeeze(-1) for x in (loc, scale, df))
        labels = labels.squeeze(-1)

        # Ensure positivity
        scale = F.softplus(scale) + 1e-6
        df = F.softplus(df) + 2.0

        indices = self.prediction_horizons.to(loc.device)
        masked_loc = loc[:, indices]
        masked_scale = scale[:, indices]
        masked_df = df[:, indices]
        masked_labels = labels[:, indices]

        masked_dist = dist.StudentT(df=masked_df, loc=masked_loc, scale=masked_scale)
        loss = -masked_dist.log_prob(masked_labels).mean()

        logits = torch.stack([loc, scale, df], dim=-1)

        return (loss, logits, labels.unsqueeze(-1))


def compute_metrics(eval_preds):
    logits = torch.tensor(eval_preds.predictions)
    labels = torch.tensor(eval_preds.label_ids).squeeze(-1)

    loc, scale, df = logits.unbind(dim=-1)

    scale = F.softplus(scale) + 1e-6
    df = F.softplus(df) + 2.0

    indices = torch.tensor([h - 1 for h in config.PREDICTION_HORIZONS], device=logits.device)
    preds_loc = loc[:, indices]
    preds_scale = scale[:, indices]
    preds_df = df[:, indices]
    labels_actual = labels[:, indices]

    # Point-wise metrics
    mae = mean_absolute_error(labels_actual.cpu().numpy().flatten(), preds_loc.cpu().numpy().flatten())
    rmse = np.sqrt(mean_squared_error(labels_actual.cpu().numpy().flatten(), preds_loc.cpu().numpy().flatten()))

    # Probabilistic metric (NLL)
    student_t = dist.StudentT(df=preds_df, loc=preds_loc, scale=preds_scale)
    nll = -student_t.log_prob(labels_actual).mean().item()

    return {"mae": mae, "rmse": rmse, "nll": nll}


def main():
    """
    Main function to run the data processing and model training pipeline.
    """
    # 1. Prepare and save data if it doesn't exist, or load it
    train_path = os.path.join(config.OUTPUT_DIR, "train_set.csv")
    val_path = os.path.join(config.OUTPUT_DIR, "val_set.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("Processed data not found. Running data preparation...")
        data_utils.prepare_and_save_data()
    else:
        print("Found pre-processed data.")

    print("--- Loading Pre-processed Data ---")
    train_df = pd.read_csv(train_path, index_col='candle_date_time_utc', parse_dates=True)
    val_df = pd.read_csv(val_path, index_col='candle_date_time_utc', parse_dates=True)
    
    # The data is already scaled. We just need the column values.
    X_train_scaled = train_df[config.FEATURE_COLS].to_numpy()
    X_val_scaled = val_df[config.FEATURE_COLS].to_numpy()
    
    print(f"Train set size: {len(train_df)}, Val set size: {len(val_df)}")


    # 2. Create Sequences with Cumulative Return Target
    prediction_length = max(config.PREDICTION_HORIZONS)
    X_train_seq, y_train_seq = dataset.create_sequences(
        X_train_scaled,
        train_df[config.TARGET_COL].to_numpy(),
        train_df["coin_id"].to_numpy(),
        config.WINDOW_SIZE,
        prediction_length,
    )
    X_val_seq, y_val_seq = dataset.create_sequences(
        X_val_scaled,
        val_df[config.TARGET_COL].to_numpy(),
        val_df["coin_id"].to_numpy(),
        config.WINDOW_SIZE,
        prediction_length,
    )

    # 3. Create Datasets
    train_dataset = dataset.TimeSeriesDataset(
        X_train_seq, y_train_seq
    )
    val_dataset = dataset.TimeSeriesDataset(X_val_seq, y_val_seq)

    # 4. Create Model
    patchtst_model = model_utils.create_patchtsmixer_model()

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        eval_strategy="epoch",
        logging_dir=config.LOGGING_DIR,
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="nll",
        greater_is_better=False,
        report_to="tensorboard",
        fp16=True,
        dataloader_num_workers=4,
        warmup_steps=100,
        optim="adamw_torch",
        lr_scheduler_type="cosine_with_restarts",
        seed=config.SEED,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=15, early_stopping_threshold=0.0005
    )

    trainer = CustomTrainer(
        model=patchtst_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    trainer.train()

    # Save the path of the best model for later use
    if trainer.state.best_model_checkpoint:
        best_model_path = trainer.state.best_model_checkpoint
        print(f"Best model found at: {best_model_path}")
        with open(f"{config.OUTPUT_DIR}/best_model_path.txt", "w") as f:
            f.write(best_model_path)
        print("Best model path saved to best_model_path.txt")

    # Save the final best model to a dedicated directory
    final_model_path = f"{config.OUTPUT_DIR}/best_model"
    trainer.save_model(final_model_path)
    print(f"Final best model saved to {final_model_path}")

    print("Training process finished.")


if __name__ == "__main__":
    main()