"""
This script evaluates a trained model by loading it from disk,
making predictions on the test set, and calculating performance metrics.
"""
from networkx import config
import pandas as pd
import joblib
import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add the parent directory to the Python path to allow for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.experiments import HEAT_PUMP_CONFIG
from shared.experiments import HEAT_PUMP_CONFIG
from shared.multi_horizon_forest import MultiHorizonForest
from shared.data_config import MODELS_DIR, PLOTS_DIR
from shared.model_config import ModelConfig
import matplotlib.pyplot as plt


def load_model(config: ModelConfig):
    """
    Load the trained model and its metadata for the given variable.
    """
    model_path = os.path.join(MODELS_DIR, f"{config.CONFIG_NAME}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    print(f"Loading model from {model_path}")
    data = joblib.load(model_path)
     
    # Support both naked models and dictionary-wrapped models backwards compatibly
    if isinstance(data, dict) and "model" in data:
        return data["model"]
    return data

def load_test_data(config: ModelConfig):
    """
    Loads the test features and targets from CSV files.
    """
    print("Loading test data...")
    features_path = os.path.join(config.DATA_DIR, f"{config.CONFIG_NAME}_features_test.csv")
    targets_path = os.path.join(config.DATA_DIR, f"{config.CONFIG_NAME}_targets_test.csv")

    if not os.path.exists(features_path) or not os.path.exists(targets_path):
        raise FileNotFoundError("Test data not found. Please run the feature engineering step first.")

    X_test = pd.read_csv(features_path, index_col=0, parse_dates=True)
    Y_test = pd.read_csv(targets_path, index_col=0, parse_dates=True)
    
    return X_test, Y_test

def evaluate(model, X_test: pd.DataFrame, Y_test: pd.DataFrame, config: ModelConfig, X_additional=None):
    """
    Makes predictions and calculates evaluation metrics.
    """
    print("Making predictions on the test set...")
    predictions = model.predict(X_test, X_additional=X_additional)

    # Ensure predictions is a DataFrame with the same index and columns as Y_test
    predictions_df = pd.DataFrame(predictions, index=Y_test.index, columns=Y_test.columns)

    mse = mean_squared_error(Y_test, predictions_df)
    mae = mean_absolute_error(Y_test, predictions_df)

    #Mean Absolute Scaled Error (MASE)
    naive_forecast = Y_test.shift(config.HORIZON)  # Naive forecast using the value from the same time in the previous day

    print(naive_forecast[config.HORIZON:].head())
    mase = mae / mean_absolute_error(Y_test[config.HORIZON:], naive_forecast[config.HORIZON:])

    metrics = {
        "mse": mse,
        "mae": mae,
        "mase": mase
    }

    print("\n--- Evaluation Metrics ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Scaled Error (MASE): {mase:.4f}")

    plot_predictions(predictions_df, Y_test, config, X_additional=X_additional, num_days=1)

    
    return metrics

def plot_predictions(predictions: pd.DataFrame, Y_test: pd.DataFrame, config: ModelConfig, X_additional=None, num_days: int = 2):
    """
    Plots the predictions against the true values.
    """

    fig, axs = plt.subplots(num_days, 1, figsize=(10, 3 * num_days), sharex=False)

    axs = np.atleast_1d(axs)

    all_handles = {}
    secondary_axes = []

    for day in range(num_days):
        ax = axs[day]
        ax2 = ax.twinx()  # Secondary y-axis for additional features
        secondary_axes.append(ax2)

        # Main predictions
        h1, = ax.plot(
            predictions.index[day*config.HORIZON:(day+1)*config.HORIZON],
            predictions.iloc[day*config.HORIZON, :],
            label="Predicted",
            color="red",
        )

        h2, = ax.plot(
            Y_test.index[day*config.HORIZON:(day+1)*config.HORIZON],
            Y_test.iloc[day*config.HORIZON, :],
            label="True",
            color="blue",
        )

        all_handles[h1.get_label()] = h1
        all_handles[h2.get_label()] = h2

        # Additional features on secondary axis
        if X_additional is not None:
            for additional_df in X_additional:
                if not additional_df.empty:
                    additional_df = additional_df.loc[predictions.index]

                    h, = ax2.plot(
                        additional_df.index[day*config.HORIZON:(day+1)*config.HORIZON],
                        additional_df.iloc[day*config.HORIZON, :],
                        label=additional_df.columns[0][:-3],
                        linestyle="--",
                    )
                    all_handles[h.get_label()] = h

        ax.tick_params(axis="x", rotation=45)

        # Optional axis labels
        ax.set_ylabel("Prediction")
        ax2.set_ylabel("Additional features")

    fig.supxlabel(
        "Time",
        y=0.01,
        fontsize=14,
    )
    fig.suptitle(
    f"Predictions vs True Values for {config.CONFIG_NAME}",
    x=0.45,
    y=0.90
    )

    fig.legend(
        all_handles.values(),
        all_handles.keys(),
        loc="upper right",
        bbox_to_anchor=(0.98, 0.99),
        fontsize="small",
    )

    fig.subplots_adjust(
        top=0.75,
        bottom=0.33,   # more room for date labels
        left=0.12,     # more room for left y-axis labels
        right=0.88,    # more room for right y-axis label
    )


    plt.savefig(os.path.join(PLOTS_DIR, f"{config.CONFIG_NAME}_predictions_plot.pdf"))

def main(config: ModelConfig, model: MultiHorizonForest = None, X_test: pd.DataFrame = None, Y_test: pd.DataFrame = None, importances: pd.DataFrame = None):
    """
    Main function to run the model evaluation.
    """
    
    # 1. Load the trained model
    if model is None:
        model = load_model(config)
    
    # 2. Load the test data
    if X_test is None or Y_test is None:
        X_test, Y_test = load_test_data(config)
    
    # 3. Select top features used for training
    if importances is None:
        importances_path = os.path.join(config.DATA_DIR, f"{config.CONFIG_NAME}_feature_importances.csv")
        importances = pd.read_csv(importances_path)

    # create a list of top features based on the importances
    top_features = importances.sort_values(by='importance', ascending=False).head(config.TOP_N_FEATURES)['feature'].tolist()
    X_top_features = [feat for feat in top_features if feat in X_test.columns]
    X_top = X_test[X_top_features]

    X_additional = []
    if config.ADDITIONAL_FORECASTS:
        for location, variables in config.ADDITIONAL_FORECASTS.items():
            for var in variables:
                if f"{location}_{var}" in top_features:
                    additional_forecast_path = os.path.join(config.DATA_DIR, f"arso_forecasts_{location}_{var}.csv")
                    if os.path.exists(additional_forecast_path):
                        df_additional = pd.read_csv(additional_forecast_path, index_col=0, parse_dates=True)
                        X_additional.append(df_additional)
                    else:
                        print(f"Warning: ARSO forecast file {additional_forecast_path} not found. Skipping this additional feature.")
                else:
                    print(f"ARSO forecast feature {location}_{var} not in top features. Skipping this additional feature.")

    metrics = evaluate(model, X_top, Y_test, config, X_additional=X_additional)
    
    return metrics

if __name__ == "__main__":
    main(HEAT_PUMP_CONFIG)
    