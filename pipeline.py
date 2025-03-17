import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from istf.metrics import compute_metrics
from istf.model.wrapper import ModelWrapper


def model_step(train_test_dict: dict, model_params: dict, checkpoint_dir: str) -> dict:
    scaler_type = model_params['scaler_type']
    nn_params = model_params['nn_params']
    loss = model_params['loss']
    lr = model_params['lr']
    epochs = model_params['epochs']
    patience = model_params['patience']
    batch_size = model_params['batch_size']

    # Insert data params in nn_params for building the correct model
    nn_params['feature_mask'] = train_test_dict['x_feat_mask']
    nn_params["time_features"] = train_test_dict['params']["prep_params"]["ts_params"]['time_feats']
    if 'encoder_layer_cls' in model_params:
        nn_params['encoder_layer_cls'] = model_params['encoder_layer_cls']

    model = ModelWrapper(
        checkpoint_dir=checkpoint_dir,
        model_params=nn_params,
        loss=loss,
        lr=lr,
        dev=train_test_dict['params']['path_params']['dev']
    )

    X = np.stack([train_test_dict['x_train']] + train_test_dict['exg_train'], axis=1)
    val_X = np.stack([train_test_dict['x_valid']] + train_test_dict['exg_valid'], axis=1)

    model.fit(
        X=X,
        y=train_test_dict['y_train'],
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        val_X=val_X,
        val_y=train_test_dict['y_valid'],
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
                }[scaler_type]()
                for k, v in scalers[id][f].items():
                    setattr(scaler, k, v)
                scalers[id][f] = scaler

    X_test = np.stack([train_test_dict['x_test']] + train_test_dict['exg_test'], axis=1)
    y_true = train_test_dict['y_test']
    y_pred = model.predict(X=X_test)

    id_array = train_test_dict['id_test']
    y_true = np.array([list(scalers[id].values())[0].inverse_transform([y_])[0] for y_, id in zip(y_true, id_array)])
    y_pred = np.array([list(scalers[id].values())[0].inverse_transform([y_])[0] for y_, id in zip(y_pred, id_array)])
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
    nan_percentage = prep_params['ts_params']['nan_percentage']
    num_fut = prep_params['ts_params']['num_fut']

    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    dataset = prep_params['ts_params']['dataset']
    out_name = f"{dataset}_{subset}_nan{int(nan_percentage * 10)}_nf{num_fut}"
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
            path_params, prep_params, eval_params, scaler_type=model_params['scaler_type']
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
