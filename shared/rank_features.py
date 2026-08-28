"""
This script loads features for temperature, trains a 48h prediction model,
and saves the model artifact.
"""

import sys
import os

from sklearn.ensemble import RandomForestRegressor

# Add the parent directory to the Python path to allow for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from shared.multi_horizon_forest import MultiHorizonForest
from shared.model_config import ModelConfig

def read_data(config: ModelConfig) -> pd.DataFrame:
    """
    Read features from the CSV file created by features.py.
    """

    print(f"Reading data for {config.CONFIG_NAME}...")

    features_path = os.path.join(config.DATA_DIR, f"{config.CONFIG_NAME}_features_train.csv")
    Y_path = os.path.join(config.DATA_DIR, f"{config.CONFIG_NAME}_targets_train.csv")

    df_features = pd.read_csv(features_path, index_col=0)
    df_targets = pd.read_csv(Y_path, index_col=0)
    return df_features, df_targets



def main(config: ModelConfig, X_train=None, Y_train=None):
    """
    Main function to train and save the temperature model.
    """

    if X_train is None or Y_train is None:

        print("Loading data from CSV files...")
        X_train, Y_train = read_data(config)

    X_additional = []
    if config.ADDITIONAL_FORECASTS:
        for location, variables in config.ADDITIONAL_FORECASTS.items():
            for var in variables:
                additional_forecast_path = os.path.join(config.DATA_DIR, f"arso_forecasts_{location}_{var}.csv")
                if os.path.exists(additional_forecast_path):
                    df_additional = pd.read_csv(additional_forecast_path, index_col=0, parse_dates=True)
                    X_additional.append(df_additional)
                else:
                    print(f"Warning: ARSO forecast file {additional_forecast_path} not found. Skipping this additional feature.")
    
    # Train model
    print("Training model...")
    model = MultiHorizonForest(n_horizons = config.HORIZON, model_type=RandomForestRegressor, n_estimators=10, random_state=42, max_features='log2')
    model.fit(X_train, Y_train, X_additional=X_additional)

    # Save importances

    importances = model.feature_importances()

    #sort importances by importance 
    importances = importances.sort_values(by='importance', ascending=False)

    #save to weather_prediction/temperature/temperature_feature_importances.csv
    importances_path = os.path.join(config.DATA_DIR, f"{config.CONFIG_NAME}_feature_importances.csv")
    importances.to_csv(importances_path, index=False)
    print(f"Feature importances saved to {importances_path}")

    return importances


if __name__ == "__main__":
    main(None)
