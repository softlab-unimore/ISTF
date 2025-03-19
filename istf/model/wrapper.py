import os
import time
from typing import List

import numpy as np
import tensorflow as tf

from .model import ISTFModel


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

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 50,
            batch_size: int = 32,
            verbose: int = 0,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            early_stop_patience: int = -1,
            checkpoint_threshold: float = None
    ):
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

        if early_stop_patience >= 0:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stop_patience,
                mode='min',
                verbose=1,
                restore_best_weights=False,
                start_from_epoch=0,
                min_delta=2e-4,
            )
            callbacks.append(early_stopping)

        self.history = self.model.fit(
            x=X,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=verbose,
            callbacks=callbacks
        )
        self.epoch_times = timing_callback.epoch_times

        # Load best model
        self.model.load_weights(self.checkpoint_path)
        self.model.save(self.checkpoint_dir + '/model.keras')
        os.remove(self.checkpoint_path)

    def predict(self, X: np.ndarray):
        y_preds = self.model.predict(X)
        return y_preds

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        metrics = self.model.evaluate(X, y, verbose=1)
        return {n: m for n, m in zip(self.model.metrics_names, metrics)}
