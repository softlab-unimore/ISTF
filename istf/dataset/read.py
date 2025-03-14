from typing import Dict, Tuple

import pandas as pd

from .french.read import load_frenchpiezo_data
from .ushcn.read import load_ushcn_data


def load_data(
        ts_filename: str,
        context_filename: str,
        ex_filename: str,
        data_type: str,
        ts_features: list[str],
        exg_features: list[str],
        nan_percentage: float = 0,
        exg_cols_stn: list[str] = None,
        exg_cols_stn_scaler: str = 'standard',
        num_past=0, num_future=0, max_null_th=float('inf')
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:

    if data_type == 'french':
        return load_frenchpiezo_data(
            ts_filename=ts_filename,
            ts_cols=ts_features,
            exg_cols=exg_features,
            context_filename=context_filename,
            subset_filename=ex_filename,
            nan_percentage=nan_percentage,
            exg_cols_stn=exg_cols_stn,
            exg_cols_stn_scaler=exg_cols_stn_scaler,
            min_length=num_past+num_future,
            max_null_th=max_null_th
        )
    elif data_type == 'ushcn':
        return load_ushcn_data(
            ts_filename=ts_filename,
            ts_cols=ts_features,
            exg_cols=exg_features,
            subset_filename=ex_filename,
            nan_percentage=nan_percentage,
            min_length=num_past+num_future,
            max_null_th=max_null_th
        )
    else:
        raise ValueError(f'Dataset {data_type} is not supported, it must be: adbpo, french, or ushcn')
