from typing import Literal

import numpy as np
import pandas as pd


def insert_nulls_max_consecutive_thr(time_series: np.ndarray, missing_rate: float, max_consecutive: int):
    n = len(time_series)
    pre_existing_nulls = np.isnan(time_series)
    total_pre_existing_nulls = pre_existing_nulls.sum()

    # Calculate the number of nulls to introduce
    total_missing = int(missing_rate * n)
    nulls_to_introduce = max(0, total_missing - total_pre_existing_nulls)

    time_series = time_series.copy()
    missing_indices = set(np.where(pre_existing_nulls)[0])
    valid_indices = set(range(1, n-1)) - (missing_indices | {idx - 1 for idx in missing_indices if idx > 0} | {idx + 1 for idx in missing_indices if idx < n - 1})

    while len(missing_indices) - total_pre_existing_nulls < nulls_to_introduce and len(valid_indices) > 0:
        # Pick a random starting point
        start_idx = np.random.choice(list(valid_indices))

        # Determine the consecutive missing length, bounded by max_consecutive
        consecutive_length = np.random.randint(1, max_consecutive + 1)

        # Generate the new interval and include bounds for gap validation
        end_idx = min(start_idx + consecutive_length, n)
        new_indices = set(range(start_idx, end_idx))
        new_indices_and_bounds = new_indices | {start_idx - 1} | {end_idx}  # Bounds to enforce gaps

        # Check for overlap or contiguity with existing intervals
        if not new_indices_and_bounds & missing_indices:
            # Add the new interval if it satisfies the constraints
            missing_indices.update(new_indices)
            valid_indices -= new_indices_and_bounds

        # Replace selected indices with NaN
    for idx in missing_indices - set(np.where(pre_existing_nulls)[0]):
        time_series[idx] = np.nan

    return time_series


def transpose_dict_of_dicts(original_dict):
    transposed_dict = {}
    all_keys = set(key for nested in original_dict.values() for key in nested)

    for outer_key, nested_dict in original_dict.items():
        for key in all_keys:
            if key not in transposed_dict:
                transposed_dict[key] = {}
            value = nested_dict.get(key, None)
            if value is not None:
                transposed_dict[key][outer_key] = value
    return transposed_dict


def reindex_ts(ts: pd.DataFrame, freq: Literal['M', 'W', 'D']):
    min_dt, max_dt = ts.index.min(), ts.index.max()  # Read min max date
    # Create a new monthly or week index
    new_index = pd.date_range(start=min_dt, end=max_dt, freq=freq).date
    # Check index constraint
    assert np.isin(ts.index, new_index).all()
    # Reindex the time-series DataFrame
    ts = ts.reindex(new_index)
    return ts
