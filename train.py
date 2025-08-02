import joblib
import numpy as np
import torch
import torch.distributions as dist
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import config
import data_utils
import dataset
import model as model_utils


import torch.nn.functional as F

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create a mask for the loss calculation
        self.prediction_horizons = torch.tensor(
            [h - 1 for h in config.PREDICTION_HORIZONS], device=self.args.device
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss only on the specified prediction horizons.
        We pop the labels from the inputs dictionary to prevent the model from computing loss internally.
        """
        # Pop the labels from the inputs dictionary
        future_values = inputs.pop("future_values")

        # Forward pass without labels, so the model returns raw predictions
        outputs = model(**inputs)
        
        # In this version, the distribution parameters are in the `prediction_outputs` tuple
        # prediction_outputs = (loc, scale, df)
        loc, scale, df = outputs.prediction_outputs

        # Ensure scale and df are positive using the softplus function
        scale = F.softplus(scale)
        df = F.softplus(df)

        # Select the predictions and labels for the specified horizons
        masked_loc = loc[:, self.prediction_horizons]
        masked_scale = scale[:, self.prediction_horizons]
        masked_df = df[:, self.prediction_horizons]
        masked_labels = future_values[:, self.prediction_horizons]

        # Create a new distribution with the masked parameters
        masked_dist = dist.StudentT(
            df=masked_df,
            loc=masked_loc,
            scale=masked_scale,
        )

        # Calculate the negative log-likelihood loss
        loss = -masked_dist.log_prob(masked_labels).mean()

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        Perform a prediction step for evaluation.
        This overrides the default behavior to prevent the model from computing loss internally.
        """
        # Pop labels to prevent internal loss calculation
        labels = inputs.pop("future_values")

        with torch.no_grad():
            # Forward pass
            outputs = model(**inputs)

        # The 'logits' for our probabilistic model are the distribution parameters
        logits = outputs.prediction_outputs
        
        # We are not in `prediction_loss_only` mode, so we compute the loss here
        # This is the same logic as in `compute_loss`
        loc, scale, df = logits
        scale = F.softplus(scale)
        df = F.softplus(df)
        
        masked_loc = loc[:, self.prediction_horizons]
        masked_scale = scale[:, self.prediction_horizons]
        masked_df = df[:, self.prediction_horizons]
        masked_labels = labels[:, self.prediction_horizons]

        masked_dist = dist.StudentT(df=masked_df, loc=masked_loc, scale=masked_scale)
        loss = -masked_dist.log_prob(masked_labels).mean()

        return (loss, logits, labels)


def compute_metrics(eval_preds):
    """
    Computes MAE, RMSE, and Negative Log-Likelihood (NLL) metrics
    only for the specified prediction horizons.
    """
    # eval_preds.predictions is a tuple of (loc, scale, df)
    loc, scale, df = eval_preds.predictions
    labels = eval_preds.label_ids

    # The model outputs predictions for all 18 features.
    # We select the one corresponding to our TARGET_COL.
    try:
        target_col_index = config.FEATURE_COLS.index(config.TARGET_COL)
    except ValueError:
        # Fallback if TARGET_COL is not in FEATURE_COLS, though it should be.
        target_col_index = 0 

    loc = loc[:, :, target_col_index]
    scale = scale[:, :, target_col_index]
    df = df[:, :, target_col_index]

    # Ensure scale and df are positive
    scale = F.softplus(torch.tensor(scale))
    df = F.softplus(torch.tensor(df))
    
    # Select the relevant horizons (indices are 0-based)
    prediction_indices = [h - 1 for h in config.PREDICTION_HORIZONS]

    preds_loc = loc[:, prediction_indices]
    preds_scale = scale[:, prediction_indices]
    preds_df = df[:, prediction_indices]
    # Squeeze the last dimension of labels to match the shape of predictions
    labels_actual = labels[:, prediction_indices].squeeze(-1)

    # 1. Point-based metrics (MAE, RMSE) using the distribution's mean (loc)
    # Flatten the arrays to 1D for scikit-learn's metrics
    mae = mean_absolute_error(labels_actual.flatten(), preds_loc.flatten())
    rmse = np.sqrt(mean_squared_error(labels_actual.flatten(), preds_loc.flatten()))

    # 2. Probabilistic metric (NLL)
    student_t_dist = dist.StudentT(
        df=preds_df,
        loc=torch.tensor(preds_loc),
        scale=preds_scale,
    )
    log_prob = student_t_dist.log_prob(torch.tensor(labels_actual))
    nll = -torch.mean(log_prob).item()

    return {"mae": mae, "rmse": rmse, "nll": nll}


def main():
    """
    Main function to run the data processing and model training pipeline.
    """
    # 1. Load and Process Data
    processed_df = data_utils.load_and_process_data(config.DAYCANDLE_DIR)
    train_df, val_df, test_df, filtered_df = data_utils.filter_and_split_data(
        processed_df
    )

    # 2. Scale Features
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit scalers on training data only
    X_train_scaled = feature_scaler.fit_transform(train_df[config.FEATURE_COLS])
    y_train_scaled = target_scaler.fit_transform(train_df[[config.TARGET_COL]])

    # Transform validation and test data
    X_val_scaled = feature_scaler.transform(val_df[config.FEATURE_COLS])
    y_val_scaled = target_scaler.transform(val_df[[config.TARGET_COL]])
    X_test_scaled = feature_scaler.transform(test_df[config.FEATURE_COLS])
    y_test_scaled = target_scaler.transform(test_df[[config.TARGET_COL]])

    # Save the scalers for later use in testing/inference
    joblib.dump(feature_scaler, f"{config.OUTPUT_DIR}/feature_scaler.joblib")
    joblib.dump(target_scaler, f"{config.OUTPUT_DIR}/target_scaler.joblib")
    print("Scalers saved.")

    # 3. Create Sequences
    prediction_length = max(config.PREDICTION_HORIZONS)
    X_train_seq, y_train_seq, train_coin_ids_seq = dataset.create_sequences(
        X_train_scaled,
        y_train_scaled,
        train_df["coin_id"].values,
        config.WINDOW_SIZE,
        prediction_length,
    )
    X_val_seq, y_val_seq, val_coin_ids_seq = dataset.create_sequences(
        X_val_scaled,
        y_val_scaled,
        val_df["coin_id"].values,
        config.WINDOW_SIZE,
        prediction_length,
    )

    # 4. Create Datasets
    train_dataset = dataset.TimeSeriesDataset(
        X_train_seq, y_train_seq, train_coin_ids_seq
    )
    val_dataset = dataset.TimeSeriesDataset(X_val_seq, y_val_seq, val_coin_ids_seq)

    # 5. Create Model
    num_coins = filtered_df["coin_id"].nunique()
    patchtst_model = model_utils.create_patchtst_model(num_coins)

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
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5, early_stopping_threshold=0.001
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