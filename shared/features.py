"""
This script reads raw temperature data, applies imputation, engineers features,
and saves the feature DataFrame.
"""
import pandas as pd
import holidays
import numpy as np
import sys
import os

# Add the parent directory to the Python path to allow for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.db import fetch_air_quality_history, fetch_timeseries, fetch_arso_forecasts, reshape_arso_forecasts
from shared.impute import impute_timeseries
from shared.data_config import DATA_DIR

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series, impute
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

from shared.model_config import ModelConfig, FeatureSpec
from shared.data_config import COUNTRY_CODE

def extract_tsfresh_features(
    df: pd.DataFrame,
    raw_params: FeatureSpec,  # FeatureSpec object
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Extract tsfresh features for a rolling window size.

    roll_size is expressed in original 15-minute steps.
    aggregation_size=4 means aggregate 15-min data to hourly data before rolling.
    """


    df = df.copy()
    original_index = df.index.copy()

    roll_size = raw_params.win_size
    aggregation_size = raw_params.agg_size
    minimal = raw_params.minimal
    smooth_size = raw_params.smooth_size
    diff_lag = raw_params.diff_lag

    if roll_size % aggregation_size != 0:
        raise ValueError("roll_size must be divisible by aggregation_size")

    aggregated_roll_size = roll_size // aggregation_size

    df = df.reset_index(names=["ds"])
    tag_name = df["TagName"].iloc[0]

    # Differencing if specified
    if diff_lag > 0:
        df["TagValue"] = df["TagValue"].diff(diff_lag)
        df = df.dropna()

    # Smoothing if specified
    if smooth_size > 1:
        df["TagValue"] = df["TagValue"].rolling(window=smooth_size, min_periods=1).mean()

    # Orignal freq is 15 min
    agg_freq = f"{15 * aggregation_size}min"

    # Aggregate before rolling
    df_agg = (
        df.set_index("ds")
          .groupby("TagName")
          .resample(agg_freq)["TagValue"]
          .mean()
          .dropna()
          .reset_index()
    )

    df_agg["ds"] = pd.to_datetime(df_agg["ds"])
    df_agg = df_agg.sort_values(["TagName", "ds"]).reset_index(drop=True)

    df_rolled = roll_time_series(
        df_agg,
        column_id="TagName",
        column_sort="ds",
        max_timeshift=aggregated_roll_size,
        min_timeshift=aggregated_roll_size,
    )


    X = extract_features(
        df_rolled,
        column_id="id",
        column_sort="ds",
        column_value="TagValue",
        default_fc_parameters=MinimalFCParameters() if minimal else EfficientFCParameters(),
        impute_function=impute,
        n_jobs=n_jobs,
    )

    impute(X)

    # tsfresh rolling index is usually (TagName, ds); keep ds only
    X = X.reset_index(level=0, drop=True)

    # Convert hourly features back to original 15-minute index
    X = X.sort_index().reindex(original_index, method="ffill")

    # Add original 15-minute value as feature
    original_values = df.set_index("ds")["TagValue"]
    X = X.merge(original_values, left_index=True, right_index=True)

    X.columns = [f"{tag_name}_roll{roll_size}__{col}" for col in X.columns]

    return X


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract datetime features from the index of the DataFrame.
    """
    # df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_Saturday'] = (df.index.dayofweek == 5).astype(int)
    df['is_Sunday'] = (df.index.dayofweek == 6).astype(int)
    df['is_weekend'] = df['is_Saturday'] | df['is_Sunday']

    country_holidays = holidays.country_holidays(
    country=COUNTRY_CODE,
    years=range(df.index.min().year, df.index.max().year + 1),
)

    bdc = np.busdaycalendar(holidays=sorted(country_holidays.keys()))

    df['days_until_workday'] = [
        (np.busday_offset(d.date(), 0, roll='forward', busdaycal=bdc) - d.date()).days
        for d in pd.to_datetime(df.index)
    ]

    return df

def extract_all_features(config: ModelConfig, start_date: str, end_date: str) -> list[pd.DataFrame]:
    """
    Extract features for all variables specified in the config.
    """
    
    all_features = []
    for name, params in config.FEATURE_PARAMS.items():
        for raw_params in params:

            print(f"Extracting features for {name} with roll size {raw_params.win_size} and {'minimal' if raw_params.minimal else 'efficient'} features...")
            df_raw = fetch_timeseries(name, start_date, end_date)
            df_imputed = impute_timeseries(df_raw)  
            features = extract_tsfresh_features(df_imputed, raw_params, n_jobs=4)
            all_features.append(features)

    dt_features = extract_datetime_features(pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq="15min")))
    all_features.append(dt_features)

    return all_features

def create_targets(df: pd.DataFrame, horizons: int) -> pd.DataFrame:
    """
    Create target columns for the next 'horizon' hours.
    """
    Y = pd.DataFrame(index=df.index)
    for i in range(1, horizons + 1):
        Y[f'h{i}'] = df.iloc[:, 1].shift(-i)
    return Y

def align_data(features: list[pd.DataFrame], targets: pd.DataFrame):
    """
    Align features and targets by dropping rows with NaN values in targets.
    """

    combined = targets.copy()
    for feature_df in features:
        combined = combined.merge(feature_df, left_index=True, right_index=True, how='left')
    combined = combined.dropna()
    
    X_aligned = combined[[col for feature_df in features for col in feature_df.columns]]
    Y_aligned = combined[targets.columns]
    
    return X_aligned, Y_aligned

# def split_train_test(X: pd.DataFrame, Y: pd.DataFrame, test_size: float = 0.1):
#     """
#     Split the data into training and testing sets based on time.
#     """

#     split_index = int(len(X) * (1 - test_size))
#     X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
#     Y_train, Y_test = Y.iloc[:split_index], Y.iloc[split_index:]
    
#     return X_train, X_test, Y_train, Y_test

def extract_air_quality_features(start_date: str, end_date: str, location: str) -> pd.DataFrame:
    """
    Extract air quality features for a specific location and date range.
    """
    df_raw = fetch_air_quality_history(start_date, end_date, location)
    
    # create a 15 min interval index for the entire date range
    full_index = pd.date_range(start=start_date, end=end_date, freq='15min')
    df_reindexed = df_raw.reindex(full_index, method='ffill')

    #remove columns datum and merilno_mesto
    df_reindexed = df_reindexed.drop(columns=['datum', 'merilno_mesto'])

    return df_reindexed


def save_features(X_train: pd.DataFrame, X_test: pd.DataFrame, Y_train: pd.DataFrame, Y_test: pd.DataFrame, config: ModelConfig):
    """
    Save features and targets to CSV files.
    """
    print(f"Saving to: {config.DATA_DIR}")
    config.DATA_DIR.mkdir(exist_ok=True)
    X_train.to_csv(config.DATA_DIR / f"{config.CONFIG_NAME}_features_train.csv")
    X_test.to_csv(config.DATA_DIR / f"{config.CONFIG_NAME}_features_test.csv")
    Y_train.to_csv(config.DATA_DIR / f"{config.CONFIG_NAME}_targets_train.csv")
    Y_test.to_csv(config.DATA_DIR / f"{config.CONFIG_NAME}_targets_test.csv")


def main(config: ModelConfig, start_date: str = "2026-02-01", end_date: str = "2026-04-01") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to generate and save features for a specific variable.
    """

    print(f"Generating features for {config.TARGET_NAME} from {start_date} to {end_date}...")

    # create target, optional differencing
    target_raw = fetch_timeseries(config.TARGET_NAME, start_date, end_date)
    target_imputed = impute_timeseries(target_raw)


    # target differencing if specified
    if config.TARGET_DIFF_LAG > 0:
        target_imputed.iloc[:, 1] = target_imputed.iloc[:, 1].diff(config.TARGET_DIFF_LAG)
        target_imputed = target_imputed.dropna()



    # target smoothing if specified
    if config.TARGET_SMOOTH_SIZE > 1:
        target_imputed.iloc[:, 1] = target_imputed.iloc[:, 1].rolling(window=config.TARGET_SMOOTH_SIZE, min_periods=1).mean()

    targets = create_targets(target_imputed, horizons=config.HORIZON)

    #calculate start date of test set based on the last timestamp in the target data and the horizon
    test_start_date = target_imputed.index[-1] - pd.Timedelta(days=config.TEST_SIZE)

    all_features_train = extract_all_features(config, start_date, test_start_date)
    all_features_test = extract_all_features(config, test_start_date, end_date)

    
    # AIR QUALITY FEATURES
    # air_quality_features_train = extract_air_quality_features(start_date, test_start_date, location="Kranj")
    # air_quality_features_test = extract_air_quality_features(test_start_date, end_date, location="Kranj")

    # # Add air quality features to the list of all features
    # all_features_train.append(air_quality_features_train)
    # all_features_test.append(air_quality_features_test)

    X_train, Y_train = align_data(all_features_train, targets)
    X_test, Y_test = align_data(all_features_test, targets)


    #X_train, X_test, Y_train, Y_test = split_train_test(X, Y)
    save_features(X_train, X_test, Y_train, Y_test, config)


    # # Fetch additional forecasts
    if config.ADDITIONAL_FORECASTS:

        for location, variables in config.ADDITIONAL_FORECASTS.items():
            additional_forecasts = fetch_arso_forecasts(start_date, end_date, location=location)

            for var in variables:
                # Automatically generate mapping if dealing with string/object columns
                if additional_forecasts[var].dtype == 'object' or additional_forecasts[var].dtype.name in ['string', 'string[python]']:
                    if config.ADDITIONAL_FORECASTS_MAPPING is None:
                        config.ADDITIONAL_FORECASTS_MAPPING = {}
                    if var not in config.ADDITIONAL_FORECASTS_MAPPING:
                        unique_values = additional_forecasts[var].dropna().unique()
                        config.ADDITIONAL_FORECASTS_MAPPING[var] = {str(val): i for i, val in enumerate(unique_values)}
                        print(f"Generated automatic mapping for {var}: {config.ADDITIONAL_FORECASTS_MAPPING[var]}")

                mapping = config.ADDITIONAL_FORECASTS_MAPPING.get(var) if config.ADDITIONAL_FORECASTS_MAPPING else None
                reshaped_forecasts = reshape_arso_forecasts(additional_forecasts, variable_col=var, location=location, mapping=mapping)
                reshaped_forecasts.to_csv(config.DATA_DIR / f"arso_forecasts_{location}_{var}.csv")

    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    #main(GLOBAL_RADIATION_BASE_CONFIG)

    #test extract_datetime_features
    df_test = extract_air_quality_features("2024-04-02", "2024-12-31", "Kranj")
    print(df_test.head())