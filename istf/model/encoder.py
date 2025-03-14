import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    last_attn_scores = None

    def call(self, x, context, attention_mask=None):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True,
            attention_mask=attention_mask
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    last_attn_scores = None

    def call(self, x, attention_mask=None):
        attn_output, attn_scores = self.mha(
            query=x,
            key=x,
            value=x,
            return_attention_scores=True,
            attention_mask=attention_mask
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, activation='relu', dropout_rate=0.1, kernel_regularizer=None):
        super().__init__()

        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation=activation, kernel_regularizer=kernel_regularizer),
            tf.keras.layers.Dense(d_model, kernel_regularizer=kernel_regularizer),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, **kwargs):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, activation='relu', dropout_rate=0.1, l2_reg=None):
        super().__init__()

        reg = {}
        if l2_reg:
            reg['kernel_regularizer'] = tf.keras.regularizers.l2(l2_reg)

        self.self_attention = GlobalSelfAttention(
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
        self.last_attn_scores = None

    def call(self, x, attention_mask=None):
        x = self.self_attention(x, attention_mask)
        self.last_attn_scores = self.self_attention.last_attn_scores
        x = self.ffn(x)

        return x


class CrossEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, activation='relu', dropout_rate=0.1, l2_reg=None):
        super(CrossEncoderLayer, self).__init__()

        reg = {}
        if l2_reg:
            reg['kernel_regularizer'] = tf.keras.regularizers.l2(l2_reg)

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            **reg
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            **reg
        )

        self.ffn = FeedForward(d_model, dff, activation=activation,
                               dropout_rate=dropout_rate, **reg)
        self.last_attn_scores = None

    def call(self, x, context):
        x = self.self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class EncoderAttnMaskLayer(tf.keras.layers.Layer):

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

    def call(self, x, attn_mask=None):  # x: (v, b, t, e) attn_mask: (v, b, t)
        shape = tf.shape(x)
        v, b, t, e = shape[0], shape[1], shape[2], shape[3]
        x = tf.transpose(x, perm=[1, 0, 2, 3])  # x: (b, v, t, e)
        x = tf.reshape(x, (b, v * t, e))  # x: (b, v*t, e)
        if attn_mask is not None:
            attn_mask = tf.transpose(attn_mask, perm=[1, 0, 2])  # attn_mask: (b, v, t)
            attn_mask = tf.reshape(attn_mask, (b, v * t))  # attn_mask: (b, v*t)
            attn_mask = tf.expand_dims(attn_mask, axis=-1) * tf.expand_dims(attn_mask, axis=1)  # attn_mask: (b, v*t, v*t)
        x = self.encoder(x, attn_mask)
        x = tf.reshape(x, (b, v, t, e))  # x: (b, v, t, e)
        x = tf.transpose(x, perm=[1, 0, 2, 3])  # x: (v, b, t, e)
        return x
