import numpy as np
import tensorflow as tf

from istf.model.embedding import TemporalEmbedding
from istf.model.encoder import GlobalSelfAttention, FeedForward, EncoderLayer


class ISTFModel(tf.keras.Model):

    def __init__(
            self, *,
            feature_mask,
            kernel_size,
            d_model,
            num_heads,
            dff,
            activation='relu',
            num_layers=1,
            dropout_rate=0.1,
            time_features=None,
            do_emb=True,
            encoder_layer_cls="MVEncoderLayerLocalGlobalAttn",
            predictor_cls="PredictorGRU",
            l2_reg=None,
    ):
        super().__init__()
        feature_mask = np.array(feature_mask)
        value_ids = np.arange(len(feature_mask))

        arg_null = feature_mask == 1
        if arg_null.any():
            self.feature_mask, self.value_ids = feature_mask[~arg_null], value_ids[~arg_null]
            self.attn_mask_id = value_ids[arg_null][0]
        else:
            self.feature_mask, self.value_ids = feature_mask, value_ids
            self.attn_mask_id = None

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.activation = activation
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.time_features = time_features
        self.do_emb = do_emb
        self.encoder_layer_cls = encoder_layer_cls
        self.predictor_cls = predictor_cls
        self.l2_reg = l2_reg

        if self.do_emb:
            self.embedder = TemporalEmbedding(
                d_model=self.d_model,
                kernel_size=self.kernel_size,
                feature_mask=self.feature_mask,
                time_features=self.time_features,
                activation=self.activation,
                l2_reg=self.l2_reg
            )
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        else:
            self.embedder = tf.keras.layers.Lambda(lambda x: x)
            self.dropout = tf.keras.layers.Lambda(lambda x: x)

        encoder_layer_cls = (
            MVEncoderLayer if self.encoder_layer_cls == 'MVEncoderLayer' else MVEncoderLayerLocalGlobalAttn
        )
        self.encoders = [encoder_layer_cls(
            d_model=(self.d_model if self.do_emb else len(self.feature_mask)),
            num_heads=self.num_heads,
            dff=self.dff,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg
        ) for _ in range(self.num_layers)]

        if self.predictor_cls == 'PredictorFlatten':
            self.predictor = PredictorFlatten(
                hidden_units=self.dff, activation=self.activation, dropout_rate=self.dropout_rate, l2_reg=self.l2_reg
            )
        else:  # if self.predictor_cls == 'gru':
            self.predictor = PredictorGRU(
                gru_units=self.d_model, hidden_units=self.dff, activation=self.activation,
                dropout_rate=self.dropout_rate, l2_reg=self.l2_reg
            )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'feature_mask': self.feature_mask.tolist(),
            'kernel_size': self.kernel_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            # 'gru': self.gru,
            # 'hidden_units': self.hidden_units,
            'activation': self.activation,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'time_features': self.time_features,
            'do_emb': self.do_emb,
            'encoder_layer_cls': self.encoder_layer_cls,
            'predictor_cls': self.predictor_cls,
            'l2_reg': self.l2_reg
        })
        return config

    def split_data_attn_mask(self, x):
        if self.attn_mask_id is None:
            return x, None
        return tf.gather(x, self.value_ids, axis=-1), tf.expand_dims(x[:, :, :, self.attn_mask_id], -1)

    def call(self, inputs):  # (b, v, t, f)
        X, attn_mask = self.split_data_attn_mask(inputs)
        X = tf.unstack(X, axis=1)
        X = [self.embedder(x) for x in X]
        X = [self.dropout(x) for x in X]
        X = tf.stack(X, axis=1)

        for i in range(self.num_layers):
            X = self.encoders[i](X, attn_mask=attn_mask)

        pred = self.predictor(X[:, 0])

        return pred


class MVEncoderLayerLocalGlobalAttn(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, activation='relu', dropout_rate=0.1, l2_reg=None):
        super().__init__()

        reg = {}
        if l2_reg:
            reg['kernel_regularizer'] = tf.keras.regularizers.l2(l2_reg)

        self.loc_attn = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            **reg
        )

        self.glb_attn = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            **reg
        )

        self.ffn = FeedForward(
            d_model=d_model,
            dff=dff,
            activation=activation,
            dropout_rate=dropout_rate, **reg
        )

    def call(self, x, attn_mask=None):  # x: (b, v, t, e) attn_mask: (b, v, t, 1)
        shape = tf.shape(x)
        b, v, t, e = shape[0], shape[1], shape[2], shape[3]

        if attn_mask is None:
            attn_mask_loc = None
        else:
            attn_mask_loc = tf.reshape(attn_mask, (b*v, 1, t))  # attn_mask: (b*v, t, 1)
        x = tf.reshape(x, (b*v, t, e))  # x: (b*v, t, e)
        x = self.loc_attn(x, attention_mask=attn_mask_loc)

        if attn_mask is None:
            attn_mask_glb = None
        else:
            attn_mask_glb = tf.reshape(attn_mask, (b, 1, v*t))  # attn_mask: (b, v*t, 1)
        x = tf.reshape(x, (b, v, t, e))  # x: (b, v, t, e)
        x = tf.reshape(x, (b, v*t, e))  # x: (b, v*t, e)
        x = self.glb_attn(x, attention_mask=attn_mask_glb)

        x = self.ffn(x)
        x = tf.reshape(x, (b, v, t, e))  # x: (b, v, t, e)
        return x


class PredictorGRU(tf.keras.layers.Layer):

    def __init__(self, gru_units, hidden_units, activation="gelu", dropout_rate=0.1, l2_reg=None):
        super().__init__()
        l2_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
        self.gru = tf.keras.layers.GRU(gru_units, kernel_regularizer=l2_reg, recurrent_regularizer=l2_reg)
        final_layers = []
        for units in hidden_units:
            final_layers.extend([
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=l2_reg)
            ])
        final_layers.extend([
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        self.final_layers = tf.keras.models.Sequential(final_layers)

    def call(self, x, mask=None):
        x = self.gru(x, mask=mask)
        x = self.final_layers(x)
        return x


class PredictorFlatten(tf.keras.layers.Layer):

    def __init__(self, hidden_units, activation="gelu", dropout_rate=0.1, l2_reg=None):
        super().__init__()
        l2_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
        self.flatten = tf.keras.layers.Flatten()
        final_layers = []
        for units in hidden_units:
            final_layers.extend([
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=l2_reg)
            ])
        final_layers.extend([
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        self.final_layers = tf.keras.models.Sequential(final_layers)

    def call(self, x):
        x = self.flatten(x)
        x = self.final_layers(x)
        return x


class MVEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, activation='relu', dropout_rate=0.1, l2_reg=None):
        super().__init__()

        self.encoder = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            activation=activation,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg
        )

    def call(self, x, attn_mask=None):  # x: (b, v, t, e) attn_mask: (b, v, t, 1)
        shape = tf.shape(x)
        b, v, t, e = shape[0], shape[1], shape[2], shape[3]
        x = tf.reshape(x, (b, v * t, e))  # x: (b, v*t, e)
        if attn_mask is not None:
            attn_mask = tf.reshape(attn_mask, (b, v*t, 1))  # attn_mask: (b, v*t, 1)
        x = self.encoder(x, attention_mask=attn_mask)
        x = tf.reshape(x, (b, v, t, e))  # x: (b, v, t, e)
        return x
