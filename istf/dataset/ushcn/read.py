from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import haversine_distances
from tqdm import tqdm

from ..french.read import create_ts_dict
from ..utils import insert_nulls_max_consecutive_thr
from ...preparation import reindex_ts


def read_ushcn(filename: str, id_col: str, date_col: str, cols: List[str]) -> Dict[str, pd.DataFrame]:
    # Read ushcn dataset
    df = pd.read_csv(filename)

    # Transform timestamp column into datetime object
    # df[date_col] = df[date_col].apply(lambda x: timedelta(days=x) + datetime(year=1950, month=1, day=1))
    df[date_col] = pd.to_datetime(df[date_col]).dt.date

    # Split time-series based on date_col and keep only the selected cols
    ts_dict = create_ts_dict(df=df, id_col=id_col, date_col=date_col, cols=cols)

    # Reindex time-series with daily frequency
    ts_dict = {k: reindex_ts(df, 'D') for k, df in ts_dict.items()}

    return ts_dict


def extract_ushcn_context(filename: str, id_col: str, x_col: str, y_col: str):
    # Read ushcn dataset
    df = pd.read_csv(filename)

    # Create an empty dict to save x and y coordinates for each id
    ctx_dict = {}
    # Create a dictionary where for each id is associated its time-series
    ts_dict: Dict[str, pd.DataFrame] = dict(list(df.groupby(id_col)))
    for k, df_k in ts_dict.items():
        # Check coordinates uniqueness
        assert df_k[x_col].duplicated(keep=False).all(), f'Different x coordinates for series {k}'
        assert df_k[y_col].duplicated(keep=False).all(), f'Different y coordinates for series {k}'

        # Extract coordinates from k series
        ctx_dict[k] = {
            'x': df_k[x_col].iloc[0],
            'y': df_k[y_col].iloc[0],
        }
    return ctx_dict


def load_ushcn_data(
        ts_filename: str,
        ts_cols: List[str],
        exg_cols: List[str],
        subset_filename: str = None,
        nan_percentage: float = 0,
        min_length=0,
        max_null_th=float('inf')
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:

    label_col = ts_cols[0]
    cols = [label_col] + exg_cols

    # Read irregular time series
    ts_dict = read_ushcn(
        filename=ts_filename,
        id_col='COOP_ID',  # id_col='UNIQUE_ID',
        date_col='DATE',  # date_col='TIME_STAMP',
        cols=cols
    )

    # Filter based on a minimum length
    ts_dict = {k: ts for k, ts in ts_dict.items() if len(ts) >= min_length}

    # Filter based on a subset if any
    if subset_filename:
        subset = pd.read_csv(subset_filename)['UNIQUE_ID'].to_list()
        ts_dict = {k: ts_dict[k] for k in subset if k in ts_dict}

    # Extract coordinates from ushcn series
    ctx_dict = extract_ushcn_context(
        filename=ts_filename,
        id_col='COOP_ID', # id_col='UNIQUE_ID',
        x_col='X', y_col='Y'
    )

    # Remove time-series without context information
    ts_dict = {stn: ts for stn, ts in ts_dict.items() if stn in ctx_dict}

    # Remove context information without time-series
    ctx_dict = {stn: ctx for stn, ctx in ctx_dict.items() if stn in ts_dict}

    # Create distance matrix for each pair of irregular time series by computing the haversine distance
    dist_matrix = create_spatial_matrix(ctx_dict, with_haversine=True)
    spt_dict = {}
    for k in ts_dict.keys():
        dists = dist_matrix.loc[k]
        dists = dists.drop(k)
        dists = dists.sort_values(ascending=True)
        spt_dict[k] = dists

    nan, tot = 0, 0
    for stn in ts_dict:
        for col in cols:
            nan += ts_dict[stn][col].isnull().sum()
            tot += len(ts_dict[stn][col])
    print(f"Missing values: {nan}/{tot} ({nan/tot:.2%})")

    # Loop through the time-series and insert NaN values at the random indices
    if nan_percentage > 0:
        nan, tot = 0, 0
        for stn in tqdm(ts_dict.keys(), desc='Injecting null values'):
            for col in cols:
                ts_dict[stn][col] = insert_nulls_max_consecutive_thr(ts_dict[stn][col].to_numpy(), nan_percentage, max_null_th)
                nan += ts_dict[stn][col].isnull().sum()
                tot += len(ts_dict[stn][col])
        print(f"Missing values after injection: {nan}/{tot} ({nan/tot:.2%})")

    return ts_dict, spt_dict


def create_spatial_matrix(coords_dict, with_haversine: bool = False) -> pd.DataFrame:
    """ Create the spatial matrix """
    # Read id array
    ids = list(coords_dict.keys())
    # Extract x and y coords for each point
    xy_data = [{'x': val['x'], 'y': val['y']} for val in coords_dict.values()]
    xy_data = pd.DataFrame(xy_data).values
    if not with_haversine:
        # Compute pairwise euclidean distances for each pair of coords
        dist_matrix = pairwise_distances(xy_data)
    else:
        # Compute pairwise haversine distances for each pair of coords
        dist_matrix = haversine_distances(xy_data)

    dist_matrix = pd.DataFrame(dist_matrix, columns=ids, index=ids)
    return dist_matrix
