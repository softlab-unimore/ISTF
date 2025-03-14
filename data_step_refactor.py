import numpy as np
import pandas as pd
from tqdm import tqdm


def time_encoding(df: pd.DataFrame, codes) -> pd.DataFrame:
    # Convert the date index series to datetime type
    datetime_series = pd.to_datetime(df.index)

    # Check code format
    for code in codes:
        if code not in ['D', 'DW', 'WY', 'M']:
            raise ValueError(f"Code {code} is not supported, it must be ['D', 'DW', 'WY', 'M']")

    if 'D' in codes:
        # Extract day value
        df['D'] = pd.Series(datetime_series.day).values - 1
    if 'DW' in codes:
        # Extract day of the week (Monday: 0, Sunday: 6)
        df['DW'] = pd.Series(datetime_series.dayofweek).values
    if 'M' in codes:
        # Extract month
        df['M'] = pd.Series(datetime_series.month).values - 1
    if 'WY' in codes:
        # Extract week of the year (1-53)
        df['WY'] = pd.Series(datetime_series.isocalendar().week).values - 1

    return df


def null_distance_array(is_null: np.ndarray):
    # Initialize arrays
    distance_array = np.zeros(len(is_null))
    last_observed_index = -1
    for i, val in enumerate(is_null):
        if not val:  # Not null value
            # Set distance_array to 0
            distance_array[i] = 0
            # Reset last_observed_index
            last_observed_index = i
        else:  # Compute distance from last_observed_index
            if last_observed_index >= 0:
                # Linear distance
                distance_array[i] = i - last_observed_index
            else:
                distance_array[i] = np.nan

    return distance_array


def link_spatial_data_water_body(
        ts_dict: dict[str, pd.DataFrame],
        label_col: str,
        num_spt: int,
        spt_dict: dict[str, pd.Series],
        *args, **kwargs
):
    spt_dict = {
        stn: dists[(dists < float("inf")) & (dists.index.isin(ts_dict.keys()))]
        for stn, dists in spt_dict.items()
    }

    ts_dict_ = dict()
    for stn_1, ts_1 in tqdm(ts_dict.items(), desc='Station'):
        dists = spt_dict[stn_1]
        if len(dists) < num_spt:
            spt_dict.pop(stn_1)
            continue

        stn_neighbors = dists[:num_spt].index.values

        for s, stn_2 in enumerate(stn_neighbors):
            ts_2 = ts_dict[stn_2][[label_col, f"{label_col}_is_null"]]
            ts_2.columns = [f"spt{s}", f"spt{s}_is_null"]
            ts_1 = ts_1.join(ts_2, how='outer')
        ts_dict_[stn_1] = ts_1

    return ts_dict_


def link_spatial_data(
        ts_dict: dict[str, pd.DataFrame],
        label_col: str,
        num_spt: int,
        spt_dict: dict[str, pd.Series],
        max_dist_th: float = None,
):
    if num_spt == 0:
        return ts_dict

    max_dist_th = max_dist_th if max_dist_th else float('inf')

    spt_dict = {
        stn: dists[(dists <= max_dist_th) & (dists.index.isin(ts_dict.keys()))]
        for stn, dists in spt_dict.items()
    }

    null_counts = {
        stn: ts_dict[stn][f"{label_col}_is_null"].sum() for stn in ts_dict
    }

    ts_dict_ = dict()
    # for stn_1 in list(spt_dict.keys()):
    for stn_1, ts_1 in tqdm(ts_dict.items(), desc='Station'):
        dists = spt_dict[stn_1]
        if len(dists) < num_spt:
            spt_dict.pop(stn_1)
            continue

        neighbor_nulls = np.array([null_counts[stn_2] for stn_2 in dists.index])
        selector = np.argpartition(neighbor_nulls, num_spt - 1)[:num_spt]
        stn_neighbors = dists.index.values[selector]

        # ts_1 = ts_dict[stn_1]
        for s, stn_2 in enumerate(stn_neighbors):
            ts_2 = ts_dict[stn_2][[label_col, f"{label_col}_is_null"]]
            ts_2.columns = [f"spt{s}", f"spt{s}_is_null"]
            ts_1 = ts_1.join(ts_2, how='outer')
        ts_dict_[stn_1] = ts_1

    return ts_dict_


def extract_windows(ts_dict, label_col, exg_cols, num_spt, time_feats, num_past, num_fut, max_null_th):

    X, y = [], []
    X_exg, X_spt = {col: [] for col in exg_cols}, {f"spt{col}": [] for col in range(num_spt)}
    time_array = []
    id_array = []

    window = num_past + num_fut
    cols = [label_col] + exg_cols + [f"spt{s}" for s in range(num_spt)]
    cols_null_dist = [f"{col}_null_dist" for col in cols]

    all_columns = np.array(cols + [f"{col}_is_null" for col in cols] + time_feats + cols_null_dist)
    # Create boolean masks for selecting columns
    cols_null_dist_mask = np.isin(all_columns, cols_null_dist)
    label_col_mask = all_columns == label_col
    label_col_is_null_mask = all_columns == f"{label_col}_is_null"
    time_feats_mask = np.isin(all_columns, time_feats)
    exg_cols_masks = [
        np.isin(all_columns, [col, f"{col}_is_null"]) | time_feats_mask for col in exg_cols
    ]
    spt_cols_masks = [
        np.isin(all_columns, [f"spt{col}", f"spt{col}_is_null"]) | time_feats_mask
        for col in range(num_spt)
    ]

    for stn, ts in tqdm(ts_dict.items(), desc='Station'):
        ts = ts[all_columns]  # preserve the order of columns
        ts_index = ts.index.values
        ts = ts.values

        n_windows = 1 + len(ts) - window
        for i in range(n_windows):
            window_values = ts[i:i + window]

            # Check null distribution threshold
            window_null_dist = window_values[:, cols_null_dist_mask]
            if np.max(window_null_dist) > max_null_th:
                continue

            # Get the label and null flag for the target column
            y_window = window_values[-1]
            if y_window[label_col_is_null_mask]:
                continue

            # Append the target label
            y.append([y_window[label_col_mask].item()])

            # Append features for X
            X.append(window_values[:num_past, label_col_mask | label_col_is_null_mask | time_feats_mask])

            # Append features for external columns (X_exg)
            for j, mask in enumerate(exg_cols_masks):
                X_exg[exg_cols[j]].append(window_values[:num_past, mask])

            # Append features for spatial columns (X_spt)
            for j, mask in enumerate(spt_cols_masks):
                X_spt[f"spt{j}"].append(window_values[:num_past, mask])

            # Append time indices and station IDs
            time_array.append(ts_index[[i, i+num_past-1, i+num_past+num_fut-1]])
            id_array.append(stn)

    # Convert to numpy arrays
    X = np.array(X, dtype=float)
    X_exg = [np.array(x, dtype=float) for x in X_exg.values()]
    X_spt = [np.array(x, dtype=float) for x in X_spt.values()]
    y = np.array(y, dtype=float)
    time_array = np.array(time_array)
    id_array = np.array(id_array)

    return X, X_exg, X_spt, y, time_array, id_array
