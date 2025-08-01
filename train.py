import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import mean_absolute_error

import config
import data_utils
import dataset
import model
import torch

import torch.distributions as dist

def compute_metrics(eval_preds):
    """
    Computes MAE, RMSE, and Negative Log-Likelihood (NLL) metrics.
    The first channel (coin_id) is ignored for metric calculation.
    """
    # For probabilistic output, predictions is a tuple of distribution parameters
    # For Student-T: (loc, scale, degrees_of_freedom)
    preds_params = eval_preds.predictions
    labels = eval_preds.label_ids

    # Ignore the first channel (coin_id)
    preds_loc = preds_params[0][:, :, 1:]
    preds_scale = preds_params[1][:, :, 1:]
    preds_df = preds_params[2][:, :, 1:]
    labels_actual = labels[:, :, 1:]

    # 1. Point-based metrics (MAE, RMSE) using the distribution's mean (loc)
    mae = np.mean(np.abs(preds_loc - labels_actual))
    rmse = np.sqrt(np.mean((preds_loc - labels_actual)**2))

    # 2. Probabilistic metric (NLL)
    # Create the Student-T distribution object from the predicted parameters
    student_t_dist = dist.StudentT(
        df=torch.tensor(preds_df),
        loc=torch.tensor(preds_loc),
        scale=torch.tensor(preds_scale)
    )
    # Calculate the log probability of the true labels
    log_prob = student_t_dist.log_prob(torch.tensor(labels_actual))
    
    # NLL is the negative of the mean log probability
    nll = -torch.mean(log_prob).item()

    return {"mae": mae, "rmse": rmse, "nll": nll}

def main():
    """
    Main function to run the data processing and model training pipeline.
    """
    # 1. Load and Process Data
    processed_df = data_utils.load_and_process_data(config.DAYCANDLE_DIR)
    train_df, val_df, test_df, filtered_df = data_utils.filter_and_split_data(processed_df)

    # 2. Scale Features
    feature_scaler = StandardScaler()
    
    X_train_scaled = feature_scaler.fit_transform(train_df[config.FEATURE_COLS])
    X_val_scaled = feature_scaler.transform(val_df[config.FEATURE_COLS])
    X_test_scaled = feature_scaler.transform(test_df[config.FEATURE_COLS])

    # 3. Create Sequences
    X_train_seq, y_train_seq, train_coin_ids_seq = dataset.create_sequences(
        X_train_scaled, train_df['coin_id'], config.WINDOW_SIZE, config.PREDICTION_HORIZONS
    )
    X_val_seq, y_val_seq, val_coin_ids_seq = dataset.create_sequences(
        X_val_scaled, val_df['coin_id'], config.WINDOW_SIZE, config.PREDICTION_HORIZONS
    )
    X_test_seq, y_test_seq, test_coin_ids_seq = dataset.create_sequences(
        X_test_scaled, test_df['coin_id'], config.WINDOW_SIZE, config.PREDICTION_HORIZONS
    )

    # 4. Create Datasets and DataLoaders
    train_dataset = dataset.TimeSeriesDataset(X_train_seq, y_train_seq, train_coin_ids_seq)
    val_dataset = dataset.TimeSeriesDataset(X_val_seq, y_val_seq, val_coin_ids_seq)
    test_dataset = dataset.TimeSeriesDataset(X_test_seq, y_test_seq, test_coin_ids_seq)

    # 5. Create Model
    num_coins = filtered_df['coin_id'].nunique()
    patchtst_model = model.create_patchtst_model(num_coins)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
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
        dataloader_num_workers=4
    )
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5, # 예: 5 에포크 동안 개선 없으면 중단
        early_stopping_threshold=0.001 # 예: 손실이 0.001 미만으로 감소하면 개선으로 보지 않음
    )
    
    
    trainer = Trainer(
        model=patchtst_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
