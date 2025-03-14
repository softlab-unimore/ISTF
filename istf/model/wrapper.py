import os
import time
from typing import List

import numpy as np
import tensorflow as tf

# from .baseline import BaselineModel
# from .emb_gru import EmbGRUModel
from .model import ISTFModel


# def get_model(model_type: str, model_params) -> tf.keras.Model:
#     if model_type == 'sttN':
#         return ISTFModel(**model_params)
#     # if model_type == 'baseline':
#     #     return BaselineModel(**model_params)
#     # if model_type == 'emb_gru':
#     #     return EmbGRUModel(**model_params)
#
#     raise ValueError(f'Model "{model_type}" is not supported')


# def custom_mae_loss(y_true, y_pred):
#     factor_levels = tf.unique(y_true[:, 1]).y
#     loss = tf.constant(0.0)
#
#     for level in factor_levels:
#         mask = tf.equal(y_true[:, 1], level)
#         true_subset = tf.boolean_mask(y_true[:, 0], mask)
#         pred_subset = tf.boolean_mask(y_pred[:, 0], mask)
#         mae = tf.reduce_mean(tf.abs(true_subset - pred_subset))
#         loss += (1.0 / tf.cast(level, dtype=tf.float32)) * mae
#
#     return loss


# def custom_mse_loss(y_true, y_pred):
#     factor_levels = tf.unique(y_true[:, 1]).y
#     loss = tf.constant(0.0)
#
#     for level in factor_levels:
#         mask = tf.equal(y_true[:, 1], level)
#         true_subset = tf.boolean_mask(y_true[:, 0], mask)
#         pred_subset = tf.boolean_mask(y_pred[:, 0], mask)
#         mse = tf.reduce_mean(tf.square(true_subset - pred_subset))
#         loss += (1.0 / tf.cast(level, dtype=tf.float32)) * mse
#
#     return loss


# def mse_multivar_ignore_nan(y_true, y_pred):
#     mask = tf.math.is_nan(y_true)
#     y_true = tf.boolean_mask(y_true, ~mask)
#     y_pred = tf.boolean_mask(y_pred, ~mask)
#     return tf.keras.losses.mean_squared_error(y_true, y_pred)


# def mae_multivar_ignore_nan(y_true, y_pred):
#     mask = tf.math.is_nan(y_true)
#     y_true = tf.boolean_mask(y_true, ~mask)
#     y_pred = tf.boolean_mask(y_pred, ~mask)
#     return tf.keras.losses.mean_absolute_error(y_true, y_pred)


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()  # Start timing at the beginning of the epoch

    def on_epoch_end(self, epoch, logs=None):
        end_time = time.time()  # End timing at the end of the epoch
        elapsed_time = end_time - self.start_time
        self.epoch_times.append(elapsed_time)


class ModelWrapper(object):
    def __init__(
            self,
            checkpoint_dir: str,
            model_type: str,
            model_params: dict,
            loss: str = 'mse',
            lr: float = 0.001,
            dev = False
    ):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = os.path.join(self.checkpoint_dir, 'cp.weights.h5')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model = ISTFModel(**model_params)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=['mae', 'mse'],
            run_eagerly=dev,
            # run_eagerly=False,
        )

        self.history = None

    @staticmethod
    def _get_spatial_array(x: np.ndarray, spt: List[np.ndarray]) -> List[np.ndarray]:
        if len(spt) == 0:
            return [x]
        spt_x = [x] + spt
        return spt_x

    def fit(
            self,
            x: np.ndarray,
            spt: List[np.ndarray],
            exg: List[np.ndarray],
            y: np.ndarray,
            epochs: int = 50,
            batch_size: int = 32,
            verbose: int = 0,
            val_x: np.ndarray = None, val_spt: List[np.ndarray] = None, val_exg: List[np.ndarray] = None, val_y: np.ndarray = None,
            early_stop_patience: int = None,
            checkpoint_threshold: float = None
    ):
        spt = self._get_spatial_array(x, spt)
        exg = self._get_spatial_array(x, exg)

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=1,
            initial_value_threshold=checkpoint_threshold
        )
        timing_callback = TimingCallback()
        callbacks = [model_checkpoint, timing_callback]

        if early_stop_patience:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stop_patience,
                mode='min',
                verbose=1,
                restore_best_weights=False,
                start_from_epoch=20,
                min_delta=2e-4,
            )
            callbacks.append(early_stopping)

        val_spt, val_exg = self._get_spatial_array(val_x, val_spt), self._get_spatial_array(val_x, val_exg)
        val_x = [np.stack(val_exg, axis=1), np.stack(val_spt, axis=1)]
        validation_data = (tuple(val_x), val_y)

        X = [np.stack(exg, axis=1), np.stack(spt, axis=1)]

        self.history = self.model.fit(
            x=tuple(X),
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=verbose,
            callbacks=callbacks
        )
        self.epoch_times = timing_callback.epoch_times

        # Load best model
        self.model.load_weights(self.checkpoint_path)
        self.model.save(self.checkpoint_dir + '/model.keras')
        os.remove(self.checkpoint_path)

    def predict(
            self, x: np.ndarray,
            spt: List[np.ndarray],
            exg: List[np.ndarray],
    ):
        spt = self._get_spatial_array(x, spt)
        exg = self._get_spatial_array(x, exg)

        X = [np.stack(exg, axis=1), np.stack(spt, axis=1)]

        y_preds = self.model.predict(
            tuple(X)
        )

        return y_preds

    def evaluate(
            self, x: np.ndarray,
            spt: List[np.ndarray],
            exg: List[np.ndarray],
            y: np.ndarray,
    ):
        spt = self._get_spatial_array(x, spt)
        exg = self._get_spatial_array(x, exg)

        X = [np.stack(exg, axis=1), np.stack(spt, axis=1)]

        metrics = self.model.evaluate(
            tuple(X),
            y,
            verbose=1
        )

        return {n: m for n, m in zip(self.model.metrics_names, metrics)}
