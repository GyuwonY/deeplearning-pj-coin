from transformers import PatchTSTConfig, PatchTSTForPrediction

import config


def create_patchtst_model():
    model_config = PatchTSTConfig(
        # --- Time series properties ---
        num_input_channels=len(config.FEATURE_COLS),
        context_length=config.WINDOW_SIZE,
        prediction_length=max(config.PREDICTION_HORIZONS),
        # --- Static features ---
        num_static_categorical_features=0,  # No static features
        # --- Patching ---
        patch_length=5,
        patch_stride=5,
        # --- Transformer architecture ---
        d_model=128,
        num_attention_heads=8,
        num_hidden_layers=4,
        dropout=0.3,
        ffn_dim=256,
        # --- Loss and Distribution for Probabilistic Forecasting ---
        loss="nll",
        distribution_output="student_t",
        num_output_channels=1
    )

    model = PatchTSTForPrediction(model_config)
    return model