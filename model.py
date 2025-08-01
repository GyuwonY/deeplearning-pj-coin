from transformers import PatchTSTConfig, PatchTSTForPrediction

import config

def create_patchtst_model(num_static_categorical_features):
    """
    Creates and returns a PatchTSTForPrediction model with the specified configuration.

    Args:
        num_static_categorical_features (int): The number of unique static categorical features (e.g., coin IDs).

    Returns:
        PatchTSTForPrediction: The configured PatchTST model.
    """
    model_config = PatchTSTConfig(
        # Data related
        # Add 1 to input channels for the static coin_id feature
        num_input_channels=len(config.FEATURE_COLS) + 1,
        context_length=config.WINDOW_SIZE,
        prediction_length=len(config.PREDICTION_HORIZONS),
        
        # Patching
        patch_length=10,
        stride=10,
        
        # Transformer architecture
        d_model=128,
        num_attention_heads=4,
        num_hidden_layers=3,
        ffn_dim=256,
        dropout=0.1,
        
        # Loss and Distribution for Probabilistic Forecasting
        loss="nll",
        distribution_output="student_t"
    )

    model = PatchTSTForPrediction(model_config)
    return model
