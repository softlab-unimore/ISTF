import argparse
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# from ablation import parse_params
from data_step_refactor import time_encoding, link_spatial_data_water_body, link_spatial_data, null_distance_array, \
    extract_windows
from istf.dataset.read import load_data
from istf.preparation import define_feature_mask, prepare_train_test
from istf.utils import IQRMasker


def apply_iqr_masker_by_stn(ts_dict, features_to_mask, train_end_excl):
    for stn in ts_dict:
        for f in features_to_mask:
            iqr_masker = IQRMasker()

            ts_train = ts_dict[stn].loc[ts_dict[stn].index < train_end_excl, f]
            if ts_train.isna().all(): continue
            ts_train = iqr_masker.fit_transform(ts_train.values.reshape(-1, 1))
            ts_dict[stn].loc[ts_dict[stn].index < train_end_excl, f] = ts_train.reshape(-1)

            ts_test = ts_dict[stn].loc[ts_dict[stn].index >= train_end_excl, f]
            if ts_test.isna().all(): continue
            ts_test = iqr_masker.transform(ts_test.values.reshape(-1, 1))
            ts_dict[stn].loc[ts_dict[stn].index >= train_end_excl, f] = ts_test.reshape(-1)

    return ts_dict


def apply_iqr_masker(ts_dict, features_to_mask, train_end_excl):
    for f in features_to_mask:
        ts_all_train = []
        for stn in ts_dict:
            ts = ts_dict[stn][f]
            ts_all_train.append(ts.loc[ts.index < train_end_excl].values)
        ts_all_train = np.concatenate(ts_all_train)
        iqr_masker = IQRMasker().fit(ts_all_train.reshape(-1, 1))

        for stn in ts_dict:
            ts_train = ts_dict[stn].loc[ts_dict[stn].index < train_end_excl, f]
            if ts_train.isna().all(): continue
            ts_train = iqr_masker.transform(ts_train.values.reshape(-1, 1))
            ts_dict[stn].loc[ts_dict[stn].index < train_end_excl, f] = ts_train.reshape(-1)

            ts_test = ts_dict[stn].loc[ts_dict[stn].index >= train_end_excl, f]
            if ts_test.isna().all(): continue
            ts_test = iqr_masker.transform(ts_test.values.reshape(-1, 1))
            ts_dict[stn].loc[ts_dict[stn].index >= train_end_excl, f] = ts_test.reshape(-1)

    return ts_dict


def apply_scaler_by_stn(ts_dict, features, train_end_excl, scaler_init):
    scalers = dict()
    for stn, ts in ts_dict.items():
        scalers[stn] = dict()
        for f in features:
            scaler = scaler_init()

            train_ts = ts.loc[ts.index < train_end_excl, f]
            if train_ts.isna().all(): continue
            train_ts = scaler.fit_transform(train_ts.values.reshape(-1, 1))
            ts.loc[ts.index < train_end_excl, f] = train_ts.reshape(-1)

            scalers[stn][f] = scaler

            test_ts = ts.loc[ts.index >= train_end_excl, f]
            if test_ts.isna().all(): continue
            test_ts = scaler.transform(test_ts.values.reshape(-1, 1))
            ts.loc[ts.index >= train_end_excl, f] = test_ts.reshape(-1)

    return ts_dict, scalers


def apply_scaler(ts_dict, features, train_end_excl, scaler_init):
    scalers = dict()
    for f in features:
        ts_all_train = []
        for stn in ts_dict:
            ts = ts_dict[stn][f]
            ts_all_train.append(ts.loc[ts.index < train_end_excl].values)
        ts_all_train = np.concatenate(ts_all_train)
        scaler = scaler_init().fit(ts_all_train.reshape(-1, 1))

        for stn in ts_dict:
            ts_train = ts_dict[stn].loc[ts_dict[stn].index < train_end_excl, f]
            if ts_train.isna().all(): continue
            ts_train = scaler.transform(ts_train.values.reshape(-1, 1))
            ts_dict[stn].loc[ts_dict[stn].index < train_end_excl, f] = ts_train.reshape(-1)

            ts_test = ts_dict[stn].loc[ts_dict[stn].index >= train_end_excl, f]
            if ts_test.isna().all(): continue
            ts_test = scaler.transform(ts_test.values.reshape(-1, 1))
            ts_dict[stn].loc[ts_dict[stn].index >= train_end_excl, f] = ts_test.reshape(-1)

        scalers[f] = scaler

    return ts_dict, scalers


def get_conf_name(dataset, nan_percentage, num_past, num_fut, num_spt, max_dist_th, dev):
    subset = f"spt{num_spt}_th{max_dist_th}"
    if dev: subset += "_dev"
    return f"{dataset}_{subset}_nan{int(nan_percentage * 10)}_np{num_past}_nf{num_fut}"


def main(path_params: dict, prep_params: dict, eval_params: dict):
    ts_params = prep_params["ts_params"]
    spt_params = prep_params["spt_params"]
    exg_params = prep_params["exg_params"]

    conf_name = get_conf_name(
        dataset=ts_params['dataset'],
        nan_percentage=ts_params['nan_percentage'],
        num_past=ts_params['num_past'],
        num_fut=ts_params['num_fut'],
        num_spt=spt_params['num_spt'],
        max_dist_th=spt_params['max_dist_th'],
        dev=path_params['dev']
    )
    print('Configuration:', conf_name)

    label_col = ts_params["label_col"]
    exg_cols = exg_params["features"]
    cols = [label_col] + exg_cols
    scaler_type = ts_params["scaler_type"]

    # Load dataset
    ts_dict, spt_dict = load_data(
        ts_filename=path_params['ts_filename'],
        context_filename=path_params['ctx_filename'],
        ex_filename=path_params['ex_filename'],
        data_type=ts_params['dataset'],
        ts_features=[label_col],
        exg_features=exg_cols,
        nan_percentage=ts_params['nan_percentage'],
        exg_cols_stn=exg_params['features_stn'] if 'features_stn' in exg_params else None,
        exg_cols_stn_scaler=scaler_type,
        num_past=ts_params['num_past'],
        num_future=ts_params['num_fut'],
        max_null_th=eval_params['null_th']
    )

    train_end_excl = pd.to_datetime(eval_params["valid_start"]).date()

    ts_dict = apply_iqr_masker(ts_dict, cols, train_end_excl)

    nan, tot = 0, 0
    for stn in ts_dict:
        for col in cols:
            nan += ts_dict[stn][col].isna().sum()
            tot += len(ts_dict[stn][col])
    print(f"Missing values after IQR masking: {nan}/{tot} ({nan/tot:.2%})")

    if scaler_type == "minmax":
        Scaler = MinMaxScaler
    elif scaler_type == "standard":
        Scaler = StandardScaler

    stns_no_data = list()
    for stn, ts in ts_dict.items():
        for f in [label_col]:
            ts_train = ts.loc[ts.index < train_end_excl, f]
            if ts_train.isna().all():
                stns_no_data.append(stn)
                continue

    for stn in stns_no_data:
        ts_dict.pop(stn)

    ts_dict, spt_scalers = apply_scaler_by_stn(ts_dict, cols, train_end_excl, Scaler)
    spt_scalers = {
        stn: {label_col: vars(spt_scalers[stn][label_col])} for stn in spt_scalers
    }

    exg_cols = exg_cols + (exg_params['features_stn'] if 'features_stn' in exg_params else [])
    cols = [label_col] + exg_cols

    time_feats = ts_params["time_feats"]
    for stn, ts in ts_dict.items():
        for col in cols:
            ts[f"{col}_is_null"] = ts[col].isnull().astype(int)
        ts = time_encoding(ts, time_feats)
        ts[cols] = ts[cols].ffill()
        ts_dict[stn] = ts

    # Compute feature mask and time encoding max sizes
    x_feature_mask = define_feature_mask(
        base_features=[label_col],
        null_feat="code_bool",
        time_feats=time_feats
    )
    print(f'Feature mask: {x_feature_mask}')

    num_spt = spt_params["num_spt"]

    # A minimum number of nearby stations is used as a heuristic to reduce the size of the dataset
    link_spatial_data_fn = {
        "french": link_spatial_data_water_body,
        "ushcn": link_spatial_data,
        "adbpo": link_spatial_data_water_body,
    }[ts_params['dataset']]
    ts_dict = link_spatial_data_fn(
        ts_dict=ts_dict,
        label_col=label_col,
        num_spt=num_spt,
        spt_dict=spt_dict,
        max_dist_th=spt_params["max_dist_th"]
    )

    cols = [label_col] + exg_cols + [f"spt{s}" for s in range(num_spt)]
    for stn in list(ts_dict.keys()):
        if stn not in spt_dict:
            ts_dict.pop(stn)
            continue

        ts = ts_dict[stn]
        for col in cols:
            ts[f"{col}_null_dist"] = null_distance_array(ts[f"{col}_is_null"])
        ts = ts.dropna(subset=cols)  # drop values that could not be forward filled
        ts_dict[stn] = ts

    x_array, exg_array, spt_array, y_array, time_array, id_array = extract_windows(
        ts_dict=ts_dict,
        label_col=label_col,
        exg_cols=exg_cols,
        num_spt=spt_params["num_spt"],
        time_feats=ts_params["time_feats"],
        num_past=ts_params["num_past"],
        num_fut=ts_params["num_fut"],
        max_null_th=eval_params["null_th"]
    )

    D = prepare_train_test(
        x_array=x_array,
        y_array=y_array,
        time_array=time_array,
        id_array=id_array,
        spt_array=spt_array,
        exg_array=exg_array,
        test_start=eval_params['test_start'],
        valid_start=eval_params['valid_start'],
        spt_dict=spt_dict
    )
    print(f"X train: {len(D['x_train'])}")
    print(f"X valid: {len(D['x_valid'])}")
    print(f"X test: {len(D['x_test'])}")

    # Save extra params in train test dictionary
    D['x_feat_mask'] = x_feature_mask
    D['scalers'] = spt_scalers

    arr_list = (
            [D['x_train']] + [D['x_test']] + [D['x_valid']] +
            # D['spt_train'] + D['spt_test'] + D['spt_valid'] +
            D['exg_train'] + D['exg_test'] + D['exg_valid']
    )
    nan, tot = 0, 0
    for x in arr_list:
        nan += x[:, :, 1].sum().sum()
        tot += x[:, :, 1].size
    tot -= tot * len(exg_params["features_stn"]) / (len(exg_cols)+1)
    print(f"Missing values in windows (excluding static features): {int(nan)}/{int(tot)} ({nan/tot:.2%})")

    # add static features to the exogenous arrays, accoding to the station indicated in id_array

    pickle_path = os.path.join('./output/pickle', f"{conf_name}.pickle")
    with open(pickle_path, "wb") as f:
        print('Saving to', pickle_path, '...', end='', flush=True)
        pickle.dump(D, f)
        print(' done!')

    # return D


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset")
    parser.add_argument('--nan-percentage', type=float, required=True, help='Percentage of NaN values to insert')
    parser.add_argument('--num-past', type=int, required=True, help='Number of past values to consider')
    parser.add_argument('--num-future', type=int, required=True, help='Number of future values to predict')
    parser.add_argument("--scaler-type", type=str, default="standard", help="Scaler type")
    parser.add_argument('--dev', action='store_true', help='Run on development data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    print(args)
    conf_file = f'./data/params_{args.dataset}.json'
    assert os.path.exists(conf_file), 'Configuration file does not exist'
    with open(conf_file, 'r') as f:
        conf = json.load(f)
    conf["model_params"] = conf.get("model_params", dict())

    conf['path_params']['dev'] = args.dev

    conf["prep_params"]["ts_params"]["dataset"] = args.dataset
    conf['prep_params']['ts_params']["nan_percentage"] = args.nan_percentage
    conf['prep_params']['ts_params']['num_past'] = args.num_past
    conf['prep_params']['ts_params']['num_fut'] = args.num_future
    conf['prep_params']['ts_params']['scaler_type'] = args.scaler_type

    if args.dev:
        ts_name, ts_ext = os.path.splitext(conf['path_params']['ts_filename'])
        conf['path_params']['ts_filename'] = f"{ts_name}_dev{ts_ext}"
        if conf['path_params']['ctx_filename']:
            ctx_name, ctx_ext = os.path.splitext(conf['path_params']['ctx_filename'])
            conf['path_params']['ctx_filename'] = f"{ctx_name}_dev{ctx_ext}"

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    main(conf['path_params'], conf['prep_params'], conf['eval_params'])
