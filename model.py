from transformers import PatchTSTConfig, PatchTSTForPrediction

import config


def create_patchtst_model(num_coins: int):
    model_config = PatchTSTConfig(
        # --- Time series properties ---
        num_input_channels=len(config.FEATURE_COLS),
        context_length=config.WINDOW_SIZE,
        prediction_length=max(config.PREDICTION_HORIZONS),
        # --- Static features ---
        num_static_categorical_features=1,  # We have one static feature: coin_id
        cardinality=[num_coins],  # The number of unique values for the static feature
        # --- Patching ---
        patch_length=10,
        stride=10,
        # --- Transformer architecture ---
        d_model=128,
        num_attention_heads=4,
        num_hidden_layers=4,
        ffn_dim=256,
        dropout=0.1,
        # --- Loss and Distribution for Probabilistic Forecasting ---
        loss="nll",
        distribution_output="student_t",
        num_output_channels=1
    )

    model = PatchTSTForPrediction(model_config)
    return model