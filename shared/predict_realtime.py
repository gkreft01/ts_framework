"""
This script generates a real-time forecast for the next `config.HORIZON` steps
using a pre-trained model.
"""
import matplotlib
import pandas as pd
import joblib
import sys
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pvlib
import numpy as np
from sqlalchemy import text


# Add the parent directory to the Python path to allow for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.data_config import  MODELS_DIR, OUTPUT_DB_SERVER, OUTPUT_DB_DATABASE, OUTPUT_DB_USER, OUTPUT_DB_PASSWORD, OUTPUT_DB_TABLE, OUTPUT_DB_TYPE
from shared.model_config import ModelConfig
from shared.features import extract_air_quality_features, extract_all_features

from shared.db import fetch_arso_forecasts, reshape_arso_forecasts, get_engine

def load_model(config_name: str):
    """
    Load the trained model for the given variable.
    """
    model_path = os.path.join(MODELS_DIR, f"{config_name}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    print(f"Loading model from {model_path}")
    data = joblib.load(model_path)
    if isinstance(data, dict) and "model" in data:
        # if "additional_forecasts_mapping" in data and data["additional_forecasts_mapping"]:
        #     config.ADDITIONAL_FORECASTS_MAPPING = data["additional_forecasts_mapping"]
        return data["model"], data["config"], data["feature_names"]
    return data, None

def fetch_latest_data(config: ModelConfig, feature_names=None, end_date: datetime = None) -> pd.DataFrame:
    """
    Fetches the most recent chunk of data needed for feature engineering.
    The lookback period is determined by the largest window size used in feature creation.
    """
    feat_params = config.FEATURE_PARAMS
    max_roll_size = max(raw_params.win_size for _,params in feat_params.items() for raw_params in params)
    lookback_days = max_roll_size / 96 + 2  # Convert 15-min intervals to days and add a buffer
    
    if end_date is None:
        end_date = datetime.now()

    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    #round end_date down to the nearest 15-minute mark
    end_date = end_date - timedelta(minutes=end_date.minute % 15, seconds=end_date.second, microseconds=end_date.microsecond)
    start_date = end_date - timedelta(days=lookback_days)

    
    all_features = extract_all_features(config, start_date.strftime('%Y-%m-%d %H:%M'), end_date.strftime('%Y-%m-%d %H:%M'))
    df_latest = pd.concat(all_features, axis=1)

    # air_quality_features_train = extract_air_quality_features(start_date.strftime('%Y-%m-%d %H:%M'), end_date.strftime('%Y-%m-%d %H:%M'), location="Kranj")

    #merge air_quality_features_train with df_latest on index, keeping all rows from df_latest
    # df_latest = df_latest.merge(air_quality_features_train, left_index=True, right_index=True, how='left')

    features_to_use = [feat for feat in feature_names if feat in df_latest.columns]
    df_latest = df_latest[features_to_use]
    return df_latest

def fetch_arso_forecasts_for_realtime(config: ModelConfig, curr_date: str, feature_names=None) -> list[pd.DataFrame]:
    """
    Fetches ARSO forecasts for the specified variables and date range, and reshapes them into DataFrames.
    """

    X_additional = []
    if config.ADDITIONAL_FORECASTS:
        for location, variables in config.ADDITIONAL_FORECASTS.items():
            for arso_var in variables:
                if f"{location}_{arso_var}" in feature_names:
                    prev_day = (datetime.strptime(curr_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
                    raw_arso_data = fetch_arso_forecasts(prev_day, curr_date, location=location)
                    mapping = config.ADDITIONAL_FORECASTS_MAPPING.get(arso_var) if config.ADDITIONAL_FORECASTS_MAPPING else None
                    df_arso = reshape_arso_forecasts(raw_arso_data, arso_var, location, mapping=mapping)
                    X_additional.append(df_arso)
                else:
                    print(f"ARSO forecast feature {location}_{arso_var} not in model features. Skipping this additional feature.")
    return X_additional

def plot_predictions(forecast_df: pd.DataFrame, X_additional: list, config: ModelConfig):

    fig, ax1 = plt.subplots()

    # Real-time forecast
    ax1.plot(
        forecast_df.index,
        forecast_df['forecast'],
        label='Real-time Forecast',
        color='red'
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Predicted Value")

    # Second y-axis for additional forecasts
    ax2 = ax1.twinx()

    for i, additional_df in enumerate(X_additional):
        if not additional_df.empty:
            ax2.plot(
                forecast_df.index,
                additional_df.iloc[0, :],
                label=additional_df.columns[0][:-3],
                linestyle='--'
            )

    ax2.set_ylabel("Additional Forecast Value")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    plt.title(f"Real-time Forecast for {config.CONFIG_NAME}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def tranform(config: ModelConfig, forecast_df: pd.DataFrame) -> pd.DataFrame:
        print("Applying pvlib decomposition and calculating POA irradiance...")
        
        # We need the solar position. Using Bled latitude/longitude approx.
        latitude = 46.3683
        longitude = 14.1146
        
        # The index should be a timezone aware DatetimeIndex.
        # Make sure our index is localized. Assuming Central European Time (CET/CEST) if tz-naive
        times = forecast_df.index
        # if times.tz is None:
        #     times = times.tz_localize('Europe/Ljubljana')
            
        solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
        
        poa_all_horizons = forecast_df.copy()
        for idx, col in enumerate(forecast_df.columns):
            # 1. Decompose GHI to DNI, DHI using Erbs model
            # Note: pvlib.irradiance.erbs takes GHI, zenith, datetime metadata
            ghi = forecast_df[col].clip(lower=0)  # GHI cannot be negative
            zenith = solpos['apparent_zenith']
            
            # Predict DNI and DHI
            erbs_res = pvlib.irradiance.erbs(ghi, zenith, times)
            dni = erbs_res['dni'].clip(lower=0)
            dhi = erbs_res['dhi'].clip(lower=0)

            # 2. Calculate POA
            surface_tilt = 30    # Default PV tilt 
            surface_azimuth = 180 # Default facing South
            
            poa_res = pvlib.irradiance.get_total_irradiance(
                surface_tilt, surface_azimuth, 
                zenith, solpos['azimuth'], 
                dni, ghi, dhi
            )
            # Store poa_global back into our tracked predictions
            poa_all_horizons[col] = poa_res['poa_global']
            
        # Swap out predictions with our POAs for subsequent plotting/metrics!
        return poa_all_horizons




def main(config_name: str, forecast_date: str = None):
    """
    Main function to generate and display a real-time forecast.
    """
    print(f"--- Generating real-time forecast for: {config_name} ---")
    # 1. Load the pre-trained model
    model, config, feat_names = load_model(config_name)


    # 3. Engineer features for the fetched data
    df_features = fetch_latest_data(config,feature_names=feat_names, end_date=forecast_date)

    # 4. Get the most recent feature set
    # We take the last row, which corresponds to the latest timestamp
    latest_features = df_features.tail(1)

    # 5. Fetch ARSO forecasts for the next `config.HORIZON` steps
    curr_date = latest_features.index[0].strftime('%Y-%m-%d')
    X_additional = fetch_arso_forecasts_for_realtime(config, curr_date, feature_names=feat_names)
    X_additional = [df[df.index.isin(latest_features.index)] for df in X_additional]



    # 6. Make a prediction for the next `config.HORIZON` steps
    print("\nMaking prediction...")
    prediction_values = model.predict(latest_features, X_additional=X_additional)

    # 7. Create a forecast DataFrame with a proper DatetimeIndex
    last_known_timestamp = latest_features.index[0]
    forecast_index = pd.date_range(
        start=last_known_timestamp + timedelta(minutes=15),
        periods=config.HORIZON,
        freq='15min'
    )
    
    # Reshape the predictions to ensure it's a single column
    forecast_df = pd.DataFrame(prediction_values.values.reshape(-1, 1), index=forecast_index, columns=['forecast'])

    print("\n--- Forecast for the next", config.HORIZON, "steps ---")
    print(forecast_df)

    upload(forecast_df, config, end_date=forecast_date)

    # 8. Plot the forecast
    plot_predictions(forecast_df, X_additional, config)

    # 9. Optionally, apply transformations like pvlib decomposition and POA calculation
    if getattr(config, 'PVLIB_TRANSFORM', False):
        poa_global = tranform(config, forecast_df)
        print("\n--- Forecast after pvlib transformation ---")
        print(poa_global)

        plt.figure(figsize=(10, 5))
        plt.plot(poa_global.index, poa_global['forecast'], label='POA Forecast', color='green')
        plt.plot(poa_global.index, forecast_df['forecast'], label='Original Forecast', color='red', linestyle='--')
        plt.legend()
        plt.title(f"Real-time POA Forecast for {config.CONFIG_NAME}")
        plt.xlabel("Time")
        plt.ylabel("Predicted POA Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# upload to postgres database
def upload(forecast_df: pd.DataFrame, config: ModelConfig, end_date: str = None):
    """
    Uploads the forecast DataFrame to the PostgreSQL database.
    """
    print(f"Uploading forecast for {config.CONFIG_NAME} to database...")

    forecast_df['name'] = config.CONFIG_NAME

    if end_date is None:
        forecast_df["curr_time"] = datetime.now()
    else:
        forecast_df["curr_time"] = pd.to_datetime(end_date)

    forecast_df = forecast_df.reset_index().rename(columns={'index': 'forecast_time', 'forecast': 'value'})


    engine = get_engine(OUTPUT_DB_TYPE, user=OUTPUT_DB_USER, password=OUTPUT_DB_PASSWORD, server=OUTPUT_DB_SERVER, database=OUTPUT_DB_DATABASE)


    rows = forecast_df[["name", "curr_time", "forecast_time", "value"]].copy()

    rows["curr_time"] = rows["curr_time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    rows["forecast_time"] = rows["forecast_time"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")

    rows = rows.to_dict("records")

    with engine.begin() as conn:
        conn.execute(
            text(f"""
                INSERT INTO {OUTPUT_DB_TABLE}
                    (name, curr_time, forecast_time, value)
                VALUES
                    (:name, :curr_time, :forecast_time, :value)
            """),
            rows,
        )

    print(f"Upload complete. {len(forecast_df)} rows uploaded.")


if __name__ == "__main__":
    # This allows the script to be run directly for a real-time prediction
    # using a default configuration.

    # Use user input --config_name
    import argparse
    parser = argparse.ArgumentParser(description="Generate real-time forecast for a specified configuration.")
    parser.add_argument("--config-name", type=str, help="The name of the configuration to use for the forecast.")
    parser.add_argument("--forecast-date", type=str, help="The date for which to generate the forecast in YYYY-MM-DD format. Defaults to today.", default=datetime.now())
    args = parser.parse_args()
    main(args.config_name, args.forecast_date)
