"""
This script handles the database connection and fetching of raw time series data.
"""
import pandas as pd
import pymssql
import psycopg2
import sys, os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.data_config import DB_SERVER, DB_DATABASE, DB_TYPE, DB_USER, DB_PASSWORD, DB_TABLE, ROOT
from shared.data_config import FORECASTS_SERVER, FORECASTS_DATABASE, FORECASTS_USER, FORECASTS_PASSWORD, FORECASTS_TABLE, FORECASTS_TYPE

from sqlalchemy import create_engine

def get_engine(db_type: str, user: str = None, password: str = None, server: str = None, database: str = None):
    if db_type == "local":
        return create_engine(f"sqlite:///{ROOT / database}")

    elif db_type == "mssql":
        return create_engine(
            f"mssql+pymssql://{user}:{password}"
            f"@{server}/{database}"
        )

    elif db_type == "postgres":
        return create_engine(
            f"postgresql+psycopg2://{user}:{password}"
            f"@{server}/{database}"
        )

    else:
        raise ValueError(
            f"Unknown database type: {db_type}. "
            "Expected 'local', 'mssql', or 'postgres'."
        )

def fetch_timeseries(variable_name: str, start: str, end: str) -> pd.DataFrame:
    """
    Connects to the database and fetches time series data for a given variable.
    Args:
        variable_name (str): The name of the variable to fetch.
        start (str): The start date of the time series.
        end (str): The end date of the time series.
    Returns:
        pd.DataFrame: A DataFrame with the time series data.            
    """

    print(f"Fetching {variable_name} from {start} to {end}...")
    
    engine = get_engine(DB_TYPE, user=DB_USER, password=DB_PASSWORD, server=DB_SERVER, database=DB_DATABASE)

    query = f"SELECT * FROM {DB_TABLE} WHERE TagName = '{variable_name}' AND TagTime BETWEEN '{start}' AND '{end}' ORDER BY TagName, TagTime"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = pd.read_sql(query, engine)
    
    df['TagTime'] = pd.to_datetime(df['TagTime'], errors='coerce')
    df = df.set_index('TagTime')
    df = df.sort_index()


    return df

def fetch_arso_forecasts(start: str, end: str, location: str) -> pd.DataFrame:
    """
    Fetches the latest ARSO forecasts from the postgres database.
    Args:
        start (str): The start date of the forecasts.
        end (str): The end date of the forecasts.
        location (str): The location for which to fetch forecasts.
    Returns:
        pd.DataFrame: A DataFrame with the ARSO forecasts.
    """
    print("Fetching ARSO forecasts...")
    engine = get_engine(FORECASTS_TYPE, user=FORECASTS_USER, password=FORECASTS_PASSWORD, server=FORECASTS_SERVER, database=FORECASTS_DATABASE)
    query = f"SELECT * FROM {FORECASTS_TABLE} WHERE location = '{location}' AND curr_date BETWEEN '{start}' AND '{end}' ORDER BY curr_date, forecast_time DESC"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = pd.read_sql(query, engine)

    # create an index from curr_date and forecast_time
    df['forecast_time'] = pd.to_datetime(df['forecast_time'])
    df['curr_date'] = pd.to_datetime(df['curr_date'])
    df = df.set_index(['curr_date', 'forecast_time'])
    return df

def reshape_arso_forecasts(df: pd.DataFrame, variable_col: str, location: str, mapping: dict = None ) -> pd.DataFrame:

    df_flat = df.reset_index()[['curr_date', 'forecast_time', variable_col]].copy()
    if mapping is not None:
        df_flat[variable_col] = df_flat[variable_col].map(mapping)

    df_flat['curr_date'] = pd.to_datetime(df_flat['curr_date']).dt.tz_localize(None)
    df_flat['forecast_time'] = pd.to_datetime(df_flat['forecast_time']).dt.tz_localize(None)


    min_date = df_flat['curr_date'].min()
    max_date = df_flat['curr_date'].max()

    index = pd.date_range(
        start=min_date,
        end=max_date + pd.Timedelta(days=2),
        freq='15min'
    )[:-1]

    columns = [f'{location}_{variable_col}_h{h+1}' for h in range(96)]
    result = pd.DataFrame(index=index, columns=columns, dtype=float)

    for ts in index:
        available = (
            df_flat[df_flat['curr_date'] <= ts]
            .sort_values('curr_date')
            .drop_duplicates(subset='forecast_time', keep='last')  # keep most recent curr_date
            .set_index('forecast_time')[variable_col]
        )

        if available.empty:
            continue

        # target times for all 96 horizons
        target_times = [ts + pd.Timedelta(minutes=(h + 1) * 15) for h in range(96)]

        
        # reindex to target times and backfill from nearest earlier forecast
        values = (
            available
            .reindex(available.index.union(target_times))
            .bfill()
            .infer_objects(copy=False)
            .reindex(target_times)
        )

        result.loc[ts] = values.values

    return result

def fetch_air_quality_history(start: str, end: str, location: str) -> pd.DataFrame:
    """
    Fetches the historical ARSO data from the postgres database.
    Args:
        start (str): The start date of the historical data.
        end (str): The end date of the historical data.
    Returns:
        pd.DataFrame: A DataFrame with the historical ARSO data.
    """
    print("Fetching ARSO history...")
    engine = get_engine("postgres", user=FORECASTS_USER, password=FORECASTS_PASSWORD, server=FORECASTS_SERVER, database=FORECASTS_DATABASE)
    query = f"SELECT datum, merilno_mesto, pm10_dnevna FROM air_quality WHERE merilno_mesto = '{location}' AND datum BETWEEN '{start}' AND '{end}' ORDER BY datum"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = pd.read_sql(query, engine)
    df['timestamp'] = pd.to_datetime(df['datum'])
    df = df.set_index('timestamp')
    return df

if __name__ == "__main__":
    # Example usage
    df = fetch_timeseries("BS_Vremenska_Postaja_Temperatura_Zraka", '2026-03-01', '2026-04-01')
