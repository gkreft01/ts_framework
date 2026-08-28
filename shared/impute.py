"""
This script contains the imputation logic to be shared across all variables.
"""
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.db import fetch_timeseries

def get_correct_index(df:pd.DataFrame) -> pd.DatetimeIndex:
    """
    Generates a correct datetime index with 15-minute intervals.
    """

    min_time = df.index.min()
    max_time = df.index.max()
    return pd.date_range(start=min_time, end=max_time, freq='15min')

def impute_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame with missing values.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled.
    """
    # This is a placeholder for the actual imputation logic.
    # You might use methods like forward fill, backward fill, or interpolation.
    print("Imputing missing values...")
    
    correct_index = get_correct_index(df)
    missing = correct_index.difference(df.index)

    #print missing index values
    # if len(missing) > 0:
    #     print(f"Missing timestamps: {missing}")
    # print(f"Found {len(missing)} missing timestamps.")

    df = df.reindex(correct_index, method='ffill')
    return df

if __name__ == "__main__":
    db = fetch_timeseries("BS_Vremenska_Postaja_Temperatura_Zraka", '2026-03-01', '2026-04-01')
    imputed_db = impute_timeseries(db)
    print(imputed_db.head())

    missing_after = get_correct_index(imputed_db).difference(imputed_db.index)
    print(f"Missing timestamps after imputation: {missing_after}")
    
