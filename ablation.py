import datetime
import os
import pickle
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf

from data_step import parse_params, data_step
from istf.preprocessing import TIME_N_VALUES


def no_ablation(train_test_dict) -> dict:
    return train_test_dict


def ablation_embedder_no_feat(train_test_dict, code) -> dict:
    for n in ['train', 'test', 'valid']:
        cond_x = [x != code for x in train_test_dict['x_feat_mask']]
        train_test_dict[f'x_{n}'] = train_test_dict[f'x_{n}'][:, :, cond_x]
        train_test_dict[f'spt_{n}'] = [x[:, :, cond_x] for x in train_test_dict[f'spt_{n}']]
        train_test_dict[f'exg_{n}'] = [x[:, :, cond_x] for x in train_test_dict[f'exg_{n}']]

    train_test_dict['x_feat_mask'] = [x for x in train_test_dict['x_feat_mask'] if x != code]

    if code == 1:
        train_test_dict['params']['model_params']['nn_params']['is_null_embedding'] = False

    if code == 2:
        train_test_dict['params']["prep_params"]["ts_params"]['time_feats'] = None

    return train_test_dict


def ablation_embedder_no_time(train_test_dict) -> dict:
    train_test_dict = ablation_embedder_no_feat(train_test_dict, 2)
    return train_test_dict


def ablation_embedder_no_null(train_test_dict) -> dict:
    train_test_dict = ablation_embedder_no_feat(train_test_dict, 1)
    return train_test_dict


def ablation_embedder_no_time_null(train_test_dict) -> dict:
    train_test_dict = ablation_embedder_no_feat(train_test_dict, 1)
    train_test_dict = ablation_embedder_no_feat(train_test_dict, 2)
    return train_test_dict


def ablation_encoder_stt(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "sttransformer"
    return train_test_dict


def ablation_encoder_t(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "t"
    return train_test_dict


def ablation_encoder_s(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "s"
    return train_test_dict


def ablation_encoder_e(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "e"
    return train_test_dict


def ablation_encoder_ts(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "ts"
    return train_test_dict


def ablation_encoder_te(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "te"
    return train_test_dict


def ablation_encoder_se(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "se"
    return train_test_dict


def ablation_encoder_ts_fe(train_test_dict) -> dict:
    # Models TS by concatenating exogenous features E in the feature dimension to T.
    train_test_dict['params']['model_params']['model_type'] = "ts_fe"
    return train_test_dict


def ablation_encoder_ts_fe_nonull(train_test_dict) -> dict:
    # Models TS by concatenating exogenous features E in the feature dimension to T,
    # without null encoding.
    train_test_dict = ablation_encoder_ts_fe(train_test_dict)
    train_test_dict = ablation_embedder_no_null(train_test_dict)
    return train_test_dict


def ablation_encoder_ts_fe_nonull_notime(train_test_dict) -> dict:
    # Models TS by concatenating exogenous features E in the feature dimension to T,
    # without null and time encoding.
    train_test_dict = ablation_encoder_ts_fe(train_test_dict)
    train_test_dict = ablation_embedder_no_time_null(train_test_dict)
    return train_test_dict


def ablation_encoder_stt_se(train_test_dict) -> dict:
    # Models STT by integrating exogenous E and T similarly to the S module.
    train_test_dict['params']['model_params']['model_type'] = "stt_se"
    return train_test_dict


def ablation_encoder_stt_se_nonull(train_test_dict) -> dict:
    # Models STT by integrating exogenous E and T similarly to the S module,
    # without null encoding.
    train_test_dict = ablation_encoder_stt_se(train_test_dict)
    train_test_dict = ablation_embedder_no_null(train_test_dict)
    return train_test_dict


def ablation_encoder_se_se(train_test_dict) -> dict:
    # Models SE by integrating exogenous E and T similarly to the S module.
    train_test_dict['params']['model_params']['model_type'] = "se_se"
    return train_test_dict


def ablation_encoder_se_se_nonull(train_test_dict) -> dict:
    # Models SE by integrating exogenous E and T similarly to the S module,
    # without null encoding.
    train_test_dict = ablation_encoder_se_se(train_test_dict)
    train_test_dict = ablation_embedder_no_null(train_test_dict)
    return train_test_dict


def ablation_encoder_stt_mts_e(train_test_dict) -> dict:
    # Models STT with multivariate inputs in E.
    cond_x = [x == 0 for x in train_test_dict['x_feat_mask']]
    for n in ['train', 'test']:
        x = train_test_dict[f'x_{n}'][:, :, cond_x].copy()

        train_test_dict[f'exg_{n}'] = np.concatenate([train_test_dict[f'exg_{n}'], x], axis=2)

    x_feat_mask = [x for x in train_test_dict['x_feat_mask'] if x == 0]
    train_test_dict['exg_feat_mask'] = train_test_dict['exg_feat_mask'] + x_feat_mask

    return train_test_dict


def ablation_no_global_encoder(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "no_glb"
    return train_test_dict


def ablation_multivariate(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = 'stt_mv'
    train_test_dict['params']['model_params']['multivar'] = True
    return train_test_dict


def ablation_multivariate_no_global_encoder(train_test_dict) -> dict:
    train_test_dict = ablation_multivariate(train_test_dict)
    train_test_dict = ablation_no_global_encoder(train_test_dict)
    train_test_dict['params']['model_params']['model_type'] = "mv_no_glb"
    return train_test_dict


def ablation_multivariate_no_null(train_test_dict) -> dict:
    train_test_dict = ablation_multivariate(train_test_dict)
    train_test_dict = ablation_embedder_no_null(train_test_dict)
    return train_test_dict


def ablation_multivariate_ts(train_test_dict) -> dict:
    train_test_dict = ablation_multivariate(train_test_dict)
    train_test_dict['params']['model_params']['model_type'] = 'mv_ts'
    return train_test_dict


def ablation_multivariate_te(train_test_dict) -> dict:
    train_test_dict = ablation_multivariate(train_test_dict)
    train_test_dict['params']['model_params']['model_type'] = 'mv_te'
    return train_test_dict


def ablation_multivariate_ts_no_null_no_global_encoder(train_test_dict) -> dict:
    train_test_dict = ablation_multivariate_no_null(train_test_dict)
    train_test_dict['params']['model_params']['model_type'] = "mv_ts_no_glb"
    return train_test_dict


def ablation_target_only(train_test_dict) -> dict:
    train_test_dict['spt_train'] = []
    train_test_dict['exg_train'] = []
    train_test_dict['spt_test'] = []
    train_test_dict['exg_test'] = []
    return train_test_dict


def ablation_stt_2(train_test_dict) -> dict:
    train_test_dict['params']['model_params']['model_type'] = "stt2"
    return train_test_dict


def ablation_impute_mean(train_test_dict) -> dict:
    for n in ['train', 'test', 'valid']:
        X = train_test_dict[f'x_{n}']
        X[:, :, 0][X[:, :, 1].astype(bool)] = 0.
        for X in train_test_dict[f'spt_{n}']:
            X[:, :, 0][X[:, :, 1].astype(bool)] = 0.
        for X in train_test_dict[f'exg_{n}']:
            X[:, :, 0][X[:, :, 1].astype(bool)] = 0.
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


def apply_ablation_code(abl_code: str, D):
    T, S, E = 'T' in abl_code, 'S' in abl_code, 'E' in abl_code
    n, t = 'n' in abl_code, 't' in abl_code
    abl_1 = '1' in abl_code
    abl_2 = '2' in abl_code
    abl_3 = '3' in abl_code
    abl_4 = '4' in abl_code
    abl_5 = '5' in abl_code
    abl_6 = '6' in abl_code
    abl_7 = '7' in abl_code
    abl_8 = '8' in abl_code
    abl_9 = '9' in abl_code

    def _scale_time_features():
        feature_mask = np.array(D['x_feat_mask'])
        time_features = D['params']["prep_params"]["ts_params"]['time_feats']
        for split in ['train', 'test', 'valid']:
            D[f'x_{split}'] = scale_time_features(D[f'x_{split}'], feature_mask, time_features)
            D[f'spt_{split}'] = [scale_time_features(x, feature_mask, time_features) for x in D[f'spt_{split}']]
            D[f'exg_{split}'] = [scale_time_features(x, feature_mask, time_features) for x in D[f'exg_{split}']]
        return D

    D['params']['model_params']['model_type'] = "sttN"

    if S and E:
        T = False  # no need to force the target series in anymore
    if abl_1:
        D['params']['model_params']['model_type'] = "baseline"
        n = False
    if abl_2:
        D['params']['model_params']['nn_params']['do_emb'] = False
        D['params']['model_params']['nn_params']['num_heads'] = 1
        D = _scale_time_features()
    if abl_3:
        D['params']['model_params']['encoder_layer_cls'] = 'MVEncoderLayer'
    if abl_5:
        n = False
    if abl_6:
        D['params']['model_params']['model_type'] = "baseline"
        n = False
        D['params']['model_params']['nn_params']['do_emb'] = False
        D['params']['model_params']['nn_params']['num_heads'] = 1
        D = _scale_time_features()
    if abl_7:
        D['params']['model_params']['model_type'] = "baseline"
        D['params']['model_params']['nn_params']['do_emb'] = False
        D['params']['model_params']['nn_params']['num_heads'] = 1
        D = _scale_time_features()
    if abl_8:
        D['params']['model_params']['model_type'] = "baseline"
    if abl_9:
        D['params']['model_params']['model_type'] = "emb_gru"

    if not n:
        D = ablation_embedder_no_feat(D, 1)

    D['params']['model_params']['nn_params']['do_exg'] = E
    D['params']['model_params']['nn_params']['do_spt'] = S
    D['params']['model_params']['nn_params']['force_target'] = T

    abl_code = []

    _abl_code = []
    if E: _abl_code.append('E')
    if S: _abl_code.append('S')
    if T: _abl_code.append('T')
    abl_code.append(''.join(_abl_code))

    _abl_code = []
    if n: _abl_code.append('n')
    if t: _abl_code.append('t')
    abl_code.append(''.join(_abl_code))

    if abl_1: abl_code.append('1')
    if abl_2: abl_code.append('2')
    if abl_3: abl_code.append('3')
    if abl_4: abl_code.append('4')
    if abl_5: abl_code.append('5')
    if abl_6: abl_code.append('6')
    if abl_7: abl_code.append('7')
    if abl_8: abl_code.append('8')
    if abl_9: abl_code.append('9')

    abl_code = '_'.join(abl_code)
    return abl_code, D


from pipeline import model_step


def get_suffix(train_test_dict):
    def to_scientific_notation(number):
        mantissa, exponent = f"{number:.0e}".split("e")
        return float(mantissa), int(exponent)

    suffix = []

    num_layers = train_test_dict['params']['model_params']['nn_params']['num_layers']
    suffix.append(f'encs={num_layers}')
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
        for X in train_test_dict[f'spt_{n}']:
            X[:, :, null_id] = 1 - X[:, :, null_id]
        for X in train_test_dict[f'exg_{n}']:
            X[:, :, null_id] = 1 - X[:, :, null_id]
    return train_test_dict


def ablation(
        # train_test_dict: dict,
        pickle_file: str,
        results_file: str,
        checkpoint_basedir: str,
        path_params: dict,
        prep_params: dict,
        eval_params: dict,
        model_params: dict,
):
    ablations_mapping = [
        'E_nt',
        # 'E_t',
        # 'E_nt_1',
        # 'E_nt_2',
        # 'E_nt_3',
        # 'E_nt_4',
        # 'E_nt_6',
        # 'E_nt_7',
        # 'E_nt_8',
        # 'E_nt_9',
        # 'E_nt_A',
    ]

    for name in ablations_mapping:
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

        name, train_test_dict = apply_ablation_code(name, train_test_dict)
        suffix = get_suffix(train_test_dict)
        if suffix: name = f"{name}#{suffix}"
        if '#' not in name:
            name += '#'
        # name += "_sk"
        # name += "_EmbReg"
        # name += "_ImpMean"
        # name += "_B"
        if name.endswith('#'):
            name = name[:-1]
        # train_test_dict = ablation_impute_mean(train_test_dict)

        print(f"\n{name}: {train_test_dict['params']['model_params']['model_type']}")
        # seed = train_test_dict['params']['model_params']['seed']
        seed = model_params['seed']
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
        if seed != 42:
            name += '_seed' + str(train_test_dict['params']['model_params']['seed'])

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        checkpoint_dir = checkpoint_basedir + "/" + timestamp
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_res = model_step(train_test_dict, train_test_dict['params']['model_params'], checkpoint_dir)

        # non-grid results
        if os.path.exists(results_file):
            results = pd.read_csv(results_file, index_col=0).T.to_dict()
        else:
            results = {}
        results[name] = model_res
        pd.DataFrame(results).T.to_csv(results_file, index=True)

        """# grid results
        import json
        model_res["name"] = name
        model_res["params"] = train_test_dict["params"]["model_params"]
        results_path = results_file.replace('.csv', '/')
        os.makedirs(results_path, exist_ok=True)
        results_path += timestamp + '.json'
        with open(results_path, 'w') as f:
            # model_res["params"]["nn_params"]["null_max_size"] = int(model_res["params"]["nn_params"]["null_max_size"])
            model_res["test_mae"] = float(model_res["test_mae"])
            model_res["test_mse"] = float(model_res["test_mse"])
            json.dump(model_res, f, indent=4)"""


def main():
    path_params, prep_params, eval_params, model_params = parse_params()
    if model_params['cpu']:
        tf.config.set_visible_devices([], 'GPU')
    seed = model_params['seed']
    # if seed is not None:
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     tf.random.set_seed(seed)

    results_dir = './output/results'
    pickle_dir = './output/pickle' + ('_seed' + str(seed) if seed != 42 else '')
    model_dir = './output/model' + ('_seed' + str(seed) if seed != 42 else '')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(pickle_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    subset = path_params['ex_filename']
    dataset = prep_params["ts_params"]['dataset']
    if dataset == 'adbpo' and 'exg_w_tp_t2m' in subset:
        subset = os.path.basename(subset).replace('exg_w_tp_t2m', 'all').replace('.pickle', '')
    elif 'all' in subset:
        path_params['ex_filename'] = None
    else:
        subset = os.path.basename(subset).replace('subset_agg_', '').replace('.csv', '')
    nan_percentage = prep_params["ts_params"]['nan_percentage']
    num_past = prep_params['ts_params']['num_past']
    num_fut = prep_params['ts_params']['num_fut']

    conf_name = f"{dataset}_{subset}_nan{int(nan_percentage * 10)}_np{num_past}_nf{num_fut}"
    print('configuration:', conf_name)
    results_file = os.path.join(results_dir, f"{conf_name}.csv")
    pickle_file = os.path.join(pickle_dir, f"{conf_name}.pickle")
    checkpoint_dir = os.path.join(model_dir, conf_name)

    """if os.path.exists(pickle_file) and not path_params['force_data_step']:
        print('Loading from', pickle_file, '...', end='', flush=True)
        with open(pickle_file, "rb") as f:
            train_test_dict = pickle.load(f)
        print(' done!')
    else:
        train_test_dict = data_step(
            path_params, prep_params, eval_params, scaler_type=model_params['transform_type']
        )

        with open(pickle_file, "wb") as f:
            print('Saving to', pickle_file, '...', end='', flush=True)
            pickle.dump(train_test_dict, f)
            print(' done!')"""
    # Invert the previous if-else
    if path_params['force_data_step'] or not os.path.exists(pickle_file):
        train_test_dict = data_step(
            path_params, prep_params, eval_params, scaler_type=model_params['scaler_type']
        )
        with open(pickle_file, "wb") as f:
            print('Saving to', pickle_file, '...', end='', flush=True)
            pickle.dump(train_test_dict, f)
            print(' done!')
        del train_test_dict

    # train_test_dict['params'] = {
    #     'path_params': path_params,
    #     'prep_params': prep_params,
    #     'eval_params': eval_params,
    #     'model_params': model_params,
    # }

    ablation(
        # train_test_dict=train_test_dict,
        pickle_file=pickle_file,
        results_file=results_file,
        checkpoint_basedir=checkpoint_dir,
        path_params=path_params,
        prep_params=prep_params,
        eval_params=eval_params,
        model_params=model_params,
    )

    print('Hello World!')


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
