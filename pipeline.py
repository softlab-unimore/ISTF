import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from istf.metrics import compute_metrics
from istf.model.wrapper import ModelWrapper


def model_step(train_test_dict: dict, model_params: dict, checkpoint_dir: str) -> dict:
    model_type = model_params['model_type']
    transform_type = model_params['transform_type']
    nn_params = model_params['nn_params']
    loss = model_params['loss']
    lr = model_params['lr']
    epochs = model_params['epochs']
    patience = model_params['patience']
    batch_size = model_params['batch_size']

    # Insert data params in nn_params for building the correct model
    nn_params['feature_mask'] = train_test_dict['x_feat_mask']
    nn_params['spatial_size'] = len(train_test_dict['spt_train']) + 1 # target
    nn_params['exg_size'] = len(train_test_dict['exg_train']) + 1 # target
    nn_params["time_features"] = train_test_dict['params']["prep_params"]["feat_params"]['time_feats']
    if 'encoder_cls' in model_params:
        nn_params['encoder_cls'] = model_params['encoder_cls']
    if 'encoder_layer_cls' in model_params:
        nn_params['encoder_layer_cls'] = model_params['encoder_layer_cls']

    model = ModelWrapper(
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
        model_params=nn_params,
        loss=loss,
        lr=lr,
        dev=train_test_dict['params']['path_params']['dev']
    )

    valid_args = dict(
        val_x=train_test_dict['x_valid'],
        val_spt=train_test_dict['spt_valid'],
        val_exg=train_test_dict['exg_valid'],
        val_y=train_test_dict['y_valid']
    )

    model.fit(
        x=train_test_dict['x_train'],
        spt=train_test_dict['spt_train'],
        exg=train_test_dict['exg_train'],
        y=train_test_dict['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        **valid_args,
        early_stop_patience=patience,
    )

    res = {}
    scalers = train_test_dict['scalers']
    for id in scalers:
        for f in scalers[id]:
            if isinstance(scalers[id][f], dict):
                scaler = {
                    "standard": StandardScaler,
                    # "minmax": MinMaxScaler,
                }[transform_type]()
                for k, v in scalers[id][f].items():
                    setattr(scaler, k, v)
                scalers[id][f] = scaler

    preds = model.predict(
        x=train_test_dict['x_test'],
        spt=train_test_dict['spt_test'],
        exg=train_test_dict['exg_test'],
    )

    id_array = train_test_dict['id_test']
    y_true = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
                       for y_, id in zip(train_test_dict['y_test'], id_array)])
    y_preds = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
                        for y_, id in zip(preds, id_array)])
    res_test = compute_metrics(y_true=y_true, y_preds=y_preds)
    res_test = {f'test_{k}': val for k, val in res_test.items()}
    res.update(res_test)
    print(model.model.summary())
    print(res_test)

    res['loss'] = model.history.history['loss']
    res['val_loss'] = model.history.history['val_loss']
    if 'test_loss' in model.history.history: res['test_loss'] = model.history.history['test_loss']
    res['epoch_times'] = model.epoch_times
    return res


import tensorflow as tf
from data_step import parse_params, data_step


def main():
    path_params, prep_params, eval_params, model_params = parse_params()
    if model_params['cpu']:
        tf.config.set_visible_devices([], 'GPU')
    _seed = model_params['seed']
    if _seed is not None:
        random.seed(_seed)
        np.random.seed(_seed)
        tf.random.set_seed(_seed)

    res_dir = './output/results'
    data_dir = './output/pickle' + ('_seed' + str(_seed) if _seed != 42 else '')
    model_dir = './output/model' + ('_seed' + str(_seed) if _seed != 42 else '')

    subset = os.path.basename(path_params['ex_filename']).replace('subset_agg_', '').replace('.csv', '')
    nan_percentage = path_params['nan_percentage']
    num_fut = prep_params['ts_params']['num_fut']

    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    out_name = f"{path_params['type']}_{subset}_nan{int(nan_percentage * 10)}_nf{num_fut}"
    print('out_name:', out_name)
    results_path = os.path.join(res_dir, f"{out_name}.csv")
    pickle_path = os.path.join(data_dir, f"{out_name}.pickle")
    checkpoint_path = os.path.join(model_dir, f"{out_name}")

    # if os.path.exists(pickle_path):
    #     print('Loading from', pickle_path, '...', end='')
    #     with open(pickle_path, "rb") as f:
    #         D = pickle.load(f)
    #     print(' done!')
    # else:
    if True:
        # from data_step import data_step
        train_test_dict = data_step(
            path_params, prep_params, eval_params, scaler_type=model_params['transform_type']
        )
        with open(pickle_path, "wb") as f:
            print('Saving to', pickle_path, '...', end='')
            pickle.dump(train_test_dict, f)
            print(' done!')

    train_test_dict['params'] = {
        'path_params': path_params,
        'prep_params': prep_params,
        'eval_params': eval_params,
        'model_params': model_params,
    }

    if os.path.exists(results_path):
        results = pd.read_csv(results_path, index_col=0).T.to_dict()
    else:
        results = {}

    selected_model = train_test_dict['params']['model_params']['model_type'][:3].upper()

    results[selected_model] = model_step(train_test_dict, train_test_dict['params']['model_params'], checkpoint_path)

    pd.DataFrame(results).T.to_csv(results_path, index=True)

    print('Done!')


if __name__ == '__main__':
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
    main()
