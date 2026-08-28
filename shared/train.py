"""
This script loads features for temperature, trains a 48h prediction model,
and saves the model artifact.
"""

import sys
import os


# Add the parent directory to the Python path to allow for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from shared.multi_horizon_forest import MultiHorizonForest
from shared.model_config import ModelConfig

from shared.data_config import DATA_DIR, MODELS_DIR

def read_data(config: ModelConfig) -> pd.DataFrame:
    """
    Read features from the CSV file created by features.py.
    """

    print(f"Reading data for {config.CONFIG_NAME}...")

    features_path = os.path.join(config.DATA_DIR, f"{config.CONFIG_NAME}_features_train.csv")
    Y_path = os.path.join(config.DATA_DIR, f"{config.CONFIG_NAME}_targets_train.csv")

    df_features = pd.read_csv(features_path, index_col=0)
    df_targets = pd.read_csv(Y_path, index_col=0)

    importances_path = os.path.join(config.DATA_DIR, f"{config.CONFIG_NAME}_feature_importances.csv")
    importances = pd.read_csv(importances_path)

    return df_features, df_targets, importances


def main(config: ModelConfig, X_train=None, Y_train=None, importances=None):
    """
    Main function to train and save the temperature model.
    """

    if X_train is None or Y_train is None or importances is None:
        print("Loading data from CSV files...")
        X_train, Y_train, importances = read_data(config)


    # create a list of top features based on the importances
    top_features = importances.sort_values(by='importance', ascending=False).head(config.TOP_N_FEATURES)['feature'].tolist()
    X_top_features = [feat for feat in top_features if feat in X_train.columns]
    X_top = X_train[X_top_features]

    X_additional = []
    if config.ADDITIONAL_FORECASTS:
        for location, variables in config.ADDITIONAL_FORECASTS.items():
            for var in variables:
                if f"{location}_{var}" in  top_features:
                    additional_forecast_path = os.path.join(config.DATA_DIR, f"arso_forecasts_{location}_{var}.csv")
                    if os.path.exists(additional_forecast_path):
                        df_additional = pd.read_csv(additional_forecast_path, index_col=0, parse_dates=True)
                        X_additional.append(df_additional)
                    else:
                        print(f"Warning: ARSO forecast file {additional_forecast_path} not found. Skipping this additional feature.")
                else:
                    print(f"ARSO forecast feature {location}_{var} not in top features. Skipping this additional feature.")

    # Train model
    print("Training model...")
    model = MultiHorizonForest(n_horizons = config.HORIZON, model_type=config.MODEL_TYPE, **config.MODEL_KWARGS)

    now = pd.Timestamp.now()
    model.fit(X_top, Y_train, X_additional=X_additional)

    train_time = pd.Timestamp.now() - now
    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{config.CONFIG_NAME}_model.pkl")
    print(f"Saving model to {model_path}")
    
    # Store metadata alongside the model as a dictionary
    # model_data = {
    #     "model": model,
    #     "feature_names": top_features,
    #     "additional_forecasts_used": config.ADDITIONAL_FORECASTS,
    #     "additional_forecasts_mapping": config.ADDITIONAL_FORECASTS_MAPPING,
    #     "horizon": config.HORIZON,
    #     "model_type": config.MODEL_TYPE.__name__ if config.MODEL_TYPE else None,
    #     "model_kwargs": config.MODEL_KWARGS,
    #     "target_name": config.TARGET_NAME,
    #     "feature_params": config.FEATURE_PARAMS,
    #     "train_time": train_time.total_seconds()
    # }

    model_data = {
        "model": model,
        "feature_names": top_features,
        "config": config,
        "train_time_seconds": train_time.total_seconds()
    }
    joblib.dump(model_data, model_path)


if __name__ == "__main__":
    main(None)
