import argparse
import json
import os
import pickle
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from istf.model.wrapper import ModelWrapper
from data_step_utils import TIME_N_VALUES, get_conf_name


def compute_metrics(y_true, y_preds):
    return {
        'r2': r2_score(y_true, y_preds),
        'mae': mean_absolute_error(y_true, y_preds),
        'mse': mean_squared_error(y_true, y_preds),
    }


def get_suffix(train_test_dict):
    def to_scientific_notation(number):
        mantissa, exponent = f"{number:.0e}".split("e")
        return float(mantissa), int(exponent)

    suffix = []

    num_layers = train_test_dict['params']['model_params']['nn_params']['num_layers']
    suffix.append(f'encs{num_layers}')
    d_model = train_test_dict['params']['model_params']['nn_params']['d_model']
    suffix.append(f'd{d_model}')
    num_heads = train_test_dict['params']['model_params']['nn_params']['num_heads']
    suffix.append(f'h{num_heads}')
    dff = train_test_dict['params']['model_params']['nn_params']['dff']
    suffix.append(f'dff{dff}')
    gru = train_test_dict['params']['model_params']['nn_params']['gru']
    suffix.append(f'gru{gru}')
    fff = train_test_dict['params']['model_params']['nn_params']['fff']
    suffix.append(f'fff{"+".join([str(x) for x in fff])}')

    dropout_rate = train_test_dict['params']['model_params']['nn_params']['dropout_rate']
    suffix.append(f'dro{int(dropout_rate*10)}')

    l2_reg = train_test_dict['params']['model_params']['nn_params']['l2_reg']
    m, e = to_scientific_notation(l2_reg)
    suffix.append(f'reg{int(m)}{"+" if e>0 else ""}{e}')
    lr = train_test_dict['params']['model_params']['lr']
    m, e = to_scientific_notation(lr)
    suffix.append(f'lr{int(m)}{"+" if e>0 else ""}{e}')

    return '_'.join(suffix)


def null_indicator_to_mask(train_test_dict):
    null_id = np.where(np.array(train_test_dict['x_feat_mask']) == 1)[0]
    if len(null_id) == 0:
        return train_test_dict
    for n in ['train', 'test', 'valid']:
        X = train_test_dict[f'x_{n}']
        X[:, :, null_id] = 1 - X[:, :, null_id]
        for X in train_test_dict[f'exg_{n}']:
            X[:, :, null_id] = 1 - X[:, :, null_id]
    return train_test_dict


def scale_time_features(x, feature_mask, time_features):
    if time_features is None:
        return x
    time_ids = np.where(feature_mask == 2)[0]
    assert len(time_ids) == len(time_features)
    for i, t in zip(time_ids, time_features):
        t_max = TIME_N_VALUES[t] - 1  # 0-indexed
        x[:, :, i] = x[:, :, i] / t_max - 0.5
        assert np.all(x[:, :, i] >= -0.5) and np.all(x[:, :, i] <= 0.5)
    return x


import data_step


def model_step(train_test_dict: dict, checkpoint_dir: str) -> dict:
    model_params = train_test_dict['params']['model_params']

    nn_params = model_params['nn_params']
    loss = model_params['loss']
    lr = model_params['lr']
    epochs = model_params['epochs']
    patience = model_params['patience']
    batch_size = model_params['batch_size']

    # Insert data params in nn_params for building the correct model
    nn_params['feature_mask'] = train_test_dict['x_feat_mask']
    nn_params["time_features"] = train_test_dict['params']["prep_params"]["ts_params"]['time_feats']

    model = ModelWrapper(
        checkpoint_dir=checkpoint_dir,
        model_params=nn_params,
        loss=loss,
        lr=lr,
        dev=train_test_dict['params']['path_params']['dev']
    )

    X = np.stack([train_test_dict['x_train']] + train_test_dict['exg_train'], axis=1)
    X_val = np.stack([train_test_dict['x_valid']] + train_test_dict['exg_valid'], axis=1)
    X_test = np.stack([train_test_dict['x_test']] + train_test_dict['exg_test'], axis=1)

    model.fit(
        X=X,
        y=train_test_dict['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        X_val=X_val,
        y_val=train_test_dict['y_valid'],
        early_stop_patience=patience,
    )

    res = {}
    scalers = train_test_dict['scalers']
    scaler_type = train_test_dict['scaler_type']
    for id in scalers:
        if isinstance(scalers[id], dict):
            scaler = {
                "standard": StandardScaler,
                "minmax": MinMaxScaler,
            }[scaler_type]()
            for k, v in scalers[id].items():
                setattr(scaler, k, v)
            scalers[id] = scaler

    y_true = train_test_dict['y_test']
    y_pred = model.predict(X=X_test)

    id_array = train_test_dict['id_test']
    y_true = np.array([scalers[id].inverse_transform([y_])[0] for y_, id in zip(y_true, id_array)])
    y_pred = np.array([scalers[id].inverse_transform([y_])[0] for y_, id in zip(y_pred, id_array)])
    res_test = compute_metrics(y_true=y_true, y_preds=y_pred)
    res_test = {f'test_{k}': val for k, val in res_test.items()}
    res.update(res_test)
    print(model.model.summary())
    print(res_test)

    res['loss'] = model.history.history['loss']
    res['val_loss'] = model.history.history['val_loss']
    res["val_mse"] = model.history.history["val_mse"]
    res['epoch_times'] = model.epoch_times
    return res


def main(path_params, prep_params, eval_params, model_params, seed: int = 42):
    results_dir = './output/results'
    pickle_dir = './output/pickle'
    model_dir = './output/model'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(pickle_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    conf_name = get_conf_name(
        prep_params['ts_params']['dataset'],
        prep_params['ts_params']['nan_percentage'],
        prep_params['ts_params']['num_past'],
        prep_params['ts_params']['num_fut'],
        prep_params['spt_params']['num_spt'],
        prep_params['spt_params']['max_dist_th'],
        path_params['dev']
    )

    pickle_file = os.path.join(pickle_dir, f"{conf_name}.pickle")
    if path_params['force_data_step'] or not os.path.exists(pickle_file):
        data_step.main(path_params, prep_params, eval_params)
    else:
        print("Configuration:", conf_name)

    results_file = os.path.join(results_dir, f"{conf_name}.csv")
    checkpoint_dir = os.path.join(model_dir, conf_name)

    print('Loading from', pickle_file, '...', end='', flush=True)
    with open(pickle_file, "rb") as f:
        train_test_dict = pickle.load(f)
    print(' done!')
    train_test_dict['params'] = {
        'path_params': deepcopy(path_params),
        'prep_params': deepcopy(prep_params),
        'eval_params': deepcopy(eval_params),
        'model_params': deepcopy(model_params),
    }
    train_test_dict = null_indicator_to_mask(train_test_dict)
    D = train_test_dict  # for brevity

    def _scale_time_features():
        feature_mask = np.array(D['x_feat_mask'])
        time_features = D['params']["prep_params"]["ts_params"]['time_feats']
        for split in ['train', 'test', 'valid']:
            D[f'x_{split}'] = scale_time_features(D[f'x_{split}'], feature_mask, time_features)
            D[f'spt_{split}'] = [scale_time_features(x, feature_mask, time_features) for x in D[f'spt_{split}']]
            D[f'exg_{split}'] = [scale_time_features(x, feature_mask, time_features) for x in D[f'exg_{split}']]

    abl = ""
    if not model_params["embedder"]:
        D['params']['model_params']['nn_params']['do_emb'] = False
        D['params']['model_params']['nn_params']['num_heads'] = 1
        _scale_time_features()
        abl = "_NoEmb"
    if not model_params["local_global"]:
        D['params']['model_params']['nn_params']['encoder_layer_cls'] = 'MVEncoderLayer'
        abl = "_NoLGA"
    if not model_params["gru"]:
        D["params"]["model_params"]['nn_params']["predictor_cls"] = "PredictorFlatten"
        abl = "_NoGRU"

    name = get_suffix(train_test_dict)
    checkpoint_dir = checkpoint_dir + "/" + name
    os.makedirs(checkpoint_dir, exist_ok=True)
    name += abl
    print("Architecture:", name)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model_res = model_step(train_test_dict, checkpoint_dir)

    if os.path.exists(results_file):
        results = pd.read_csv(results_file, index_col=0).T.to_dict()
    else:
        results = {}
    results[name] = model_res
    pd.DataFrame(results).T.to_csv(results_file, index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True, help="Dataset [french, ushcn]")
    parser.add_argument('--nan-percentage', type=float, required=True, help='Percentage of NaN values to insert')
    parser.add_argument('--num-past', type=int, required=True, help='Number of past values to consider')
    parser.add_argument('--num-future', type=int, required=True, help='Number of future values to predict')
    parser.add_argument("--scaler-type", type=str, default="standard", help="Scaler type [standard, minmax]")

    parser.add_argument("--kernel-size", type=int, default=5, help="Embedder kernel size")
    parser.add_argument("--d-model", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dff", type=int, default=64, help="Encoder internal feed-forward dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--gru", type=int, default=64, help="GRU units")
    parser.add_argument("--fff", type=int, nargs="+", default=[128], help="Sequence of feed-forward layer dimensions")
    parser.add_argument("--l2-reg", type=float, default=1e-2, help="L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--loss", type=str, default="mse", help="Loss function")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience; -1 no early stopping")

    parser.add_argument("--no-embedder", action='store_true', help='Ablation without embedder')
    parser.add_argument("--no-local-global", action='store_true', help='Ablation without local-global attention')
    parser.add_argument("--no-gru", action='store_true', help='Ablation without GRU')

    parser.add_argument('--force-data-step', action='store_true', help='Run data step even if pickle exists')
    parser.add_argument('--dev', action='store_true', help='Run on development data')
    parser.add_argument('--cpu', action='store_true', help='Run on CPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    print(args)
    conf_file = f'./data/params_{args.dataset}.json'
    assert os.path.exists(conf_file), 'Configuration file does not exist'

    with open(conf_file, 'r') as f:
        conf = json.load(f)
    conf["model_params"] = conf.get("model_params", dict())
    conf["model_params"]["nn_params"] = conf["model_params"].get("nn_params", dict())

    conf['path_params']['force_data_step'] = args.force_data_step
    conf['path_params']['dev'] = args.dev
    conf['model_params']['cpu'] = args.cpu
    # conf['model_params']['seed'] = args.seed

    conf["prep_params"]["ts_params"]["dataset"] = args.dataset
    conf['prep_params']['ts_params']["nan_percentage"] = args.nan_percentage
    conf['prep_params']['ts_params']['num_past'] = args.num_past
    conf['prep_params']['ts_params']['num_fut'] = args.num_future
    conf['prep_params']["ts_params"]['scaler_type'] = args.scaler_type

    conf["model_params"]["embedder"] = not args.no_embedder
    conf["model_params"]["local_global"] = not args.no_local_global
    conf["model_params"]["gru"] = not args.no_gru
    # only one ablation at a time
    assert sum([args.no_embedder, args.no_local_global, args.no_gru]) <= 1, 'Only one ablation at a time'

    conf['model_params']['nn_params']['kernel_size'] = args.kernel_size
    conf['model_params']['nn_params']['d_model'] = args.d_model
    conf['model_params']['nn_params']['num_heads'] = args.num_heads
    conf['model_params']['nn_params']['dff'] = args.dff
    conf['model_params']['nn_params']['num_layers'] = args.num_layers
    conf['model_params']['nn_params']['gru'] = args.gru
    fff = args.fff
    if fff[-1] == 1:
        fff = fff[:-1]
    conf['model_params']['nn_params']['fff'] = fff
    conf['model_params']['nn_params']['l2_reg'] = args.l2_reg
    conf['model_params']['nn_params']['dropout_rate'] = args.dropout
    conf['model_params']['nn_params']['activation'] = args.activation

    conf['model_params']['lr'] = args.lr
    conf['model_params']['loss'] = args.loss
    conf['model_params']['batch_size'] = args.batch_size
    conf['model_params']['epochs'] = args.epochs
    conf['model_params']['patience'] = args.patience

    if args.dev:
        ts_name, ts_ext = os.path.splitext(conf['path_params']['ts_filename'])
        conf['path_params']['ts_filename'] = f"{ts_name}_dev{ts_ext}"
        if conf['path_params']['ctx_filename']:
            ctx_name, ctx_ext = os.path.splitext(conf['path_params']['ctx_filename'])
            conf['path_params']['ctx_filename'] = f"{ctx_name}_dev{ctx_ext}"
        conf['model_params']['epochs'] = 3
        conf['model_params']['patience'] = 1

    if args.cpu:
        tf.config.set_visible_devices([], 'GPU')
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    main(conf['path_params'], conf['prep_params'], conf['eval_params'], conf['model_params'], args.seed)
