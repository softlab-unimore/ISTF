from typing import Dict, List, Tuple, Literal, Optional

import numpy as np
import pandas as pd


def reindex_ts(ts: pd.DataFrame, freq: Literal['M', 'W', 'D']):
    min_dt, max_dt = ts.index.min(), ts.index.max()  # Read min max date
    # Create a new monthly or week index
    new_index = pd.date_range(start=min_dt, end=max_dt, freq=freq).date
    # Check index constraint
    assert np.isin(ts.index, new_index).all()
    # Reindex the time-series DataFrame
    ts = ts.reindex(new_index)
    return ts


def define_feature_mask(base_features: List[str], null_feat: str = None, time_feats: List[str] = None) -> List[int]:
    # Return the type of feature (0: raw, 1: null encoding, 2: time encoding) in each timestamp
    features_mask = [0 for _ in base_features]
    if null_feat and null_feat in ['code_lin', 'code_bool']:
        features_mask += [1]
    elif null_feat and null_feat not in ['code_lin', 'code_bool']:
        features_mask += [0]
    if time_feats:
        features_mask += [2 for _ in time_feats]
    return features_mask


def prepare_train_test(
        x_array: np.ndarray,
        y_array: np.ndarray,
        time_array: np.ndarray,
        id_array: np.ndarray,
        spt_array: List[np.ndarray],
        exg_array: List[np.ndarray],
        test_start: str,
        valid_start: str,
        spt_dict: dict[str, pd.Series]
) -> dict:
    is_train = time_array[:, -1] < pd.to_datetime(valid_start).date()
    is_valid = (time_array[:, -1] >= pd.to_datetime(valid_start).date()) & (time_array[:, -1] < pd.to_datetime(test_start).date())
    is_test = (time_array[:, -1] >= pd.to_datetime(test_start).date())

    spt_dict = {k: v.to_dict() for k, v in spt_dict.items()}

    res = {
        'x_train': x_array[is_train],
        'y_train': y_array[is_train],
        'time_train': time_array[is_train],
        'id_train': id_array[is_train],
        'spt_train': [arr[is_train] for arr in spt_array],
        'exg_train': [arr[is_train] for arr in exg_array],

        'x_test': x_array[is_test],
        'y_test': y_array[is_test],
        'time_test': time_array[is_test],
        'id_test': id_array[is_test],
        'spt_test': [arr[is_test] for arr in spt_array],
        'exg_test': [arr[is_test] for arr in exg_array],
        'spt_dict': spt_dict,

        'x_valid': x_array[is_valid],
        'y_valid': y_array[is_valid],
        'time_valid': time_array[is_valid],
        'id_valid': id_array[is_valid],
        'spt_valid': [arr[is_valid] for arr in spt_array],
        'exg_valid': [arr[is_valid] for arr in exg_array],
    }

    return res
