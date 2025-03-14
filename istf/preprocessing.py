from typing import List

import pandas as pd


# Number of distinct values for each time feature
TIME_N_VALUES = {
    'D': 31,
    'DW': 7,
    'M': 12,
    'WY': 53
}


def time_encoding(df: pd.DataFrame, codes: List[str]) -> pd.DataFrame:
    # Create a copy of the input series
    df_new = df.copy()

    # Convert the date index series to datetime type
    datetime_series = pd.to_datetime(df.index)

    # Check code format
    for code in codes:
        if code not in ['D', 'DW', 'WY', 'M']:
            raise ValueError(f"Code {code} is not supported, it must be ['D', 'DW', 'WY', 'M']")

    if 'D' in codes:
        # Extract day value
        df_new['D'] = pd.Series(datetime_series.day).values - 1
    if 'DW' in codes:
        # Extract day of the week (Monday: 0, Sunday: 6)
        df_new['DW'] = pd.Series(datetime_series.dayofweek).values
    if 'M' in codes:
        # Extract month
        df_new['M'] = pd.Series(datetime_series.month).values - 1
    if 'WY' in codes:
        # Extract week of the year
        df_new['WY'] = pd.Series(datetime_series.isocalendar().week).values - 1

    return df_new
