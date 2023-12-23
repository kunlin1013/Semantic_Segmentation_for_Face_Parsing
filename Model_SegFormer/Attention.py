import tensorflow as tf
import math

class Attention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, sr_ratio, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, name="Attention", **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads      # the dimension of each attention head, allows the model to learn different features in distinct heads

        self.units = self.num_heads * self.head_dim     # the total dimension across all heads, it ensures that the output dimensions are appropriately divided into multiple heads
        self.sqrt_of_units = math.sqrt(self.head_dim)   # used for scaling the attention weights, is common technique which prevents excessively large values during dot products, thereby stabilizing the training process

        self.q = tf.keras.layers.Dense(self.units)
        self.k = tf.keras.layers.Dense(self.units)
        self.v = tf.keras.layers.Dense(self.units)

        self.attn_drop = tf.keras.layers.Dropout(attn_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # if sr_ratio > 1, it will add a conv layer and a layernormlized to reduce the sequence's dimension
            self.sr = tf.keras.layers.Conv2D(filters=dim, kernel_size=sr_ratio, strides=sr_ratio, name='sr')
            self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-05)
           
        self.proj = tf.keras.layers.Dense(dim)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)
        
        self.qkv_bias = qkv_bias

    def call(self, x, H, W):
        # x: input (batch_size, sequence_length, feature_dim), e.g. "My name is Peter" => (4, 128, 1) , 128 is the output dimension from encoder(like one-hot encoding or word embedding)
        # H, W: dimension
        get_shape = tf.shape(x)
        B = get_shape[0]
        C = x.shape[2]
        
        q = self.q(x)
        # shape: (batch_size, let tensorflow auto calculate the dimention, num_heads, head_dim)
        q = tf.reshape(q, shape=(tf.shape(q)[0], -1, self.num_heads, self.head_dim))
        # (batch_size, the length of sequence, num_heads, head_dim) => (batch_size, num_heads, the length of sequence, head_dim)
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        
        # check whether dimention reduction is required
        if self.sr_ratio > 1:
            x = tf.reshape(x, (B, H, W, C))
            x = self.sr(x)
            x = tf.reshape(x, (B, -1, C))
            x = self.norm(x)

        # operations on k and v is as same as q
        k = self.k(x)
        k = tf.reshape(k, shape=(tf.shape(k)[0], -1, self.num_heads, self.head_dim))
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        
        v = self.v(x)
        v = tf.reshape(v, shape=(tf.shape(v)[0], -1, self.num_heads, self.head_dim))
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        
        # the equation of attention: Attention(Q, K, V) = softmax(Q*K^T / sqrt(dimention of head))*V 
        attn = tf.matmul(q, k, transpose_b=True)                # dot product in q and k
        scale = tf.cast(self.sqrt_of_units, dtype=attn.dtype)   # data type conversion
        attn = tf.divide(attn, scale)                           # computes division of attn by scale

        attn = tf.nn.softmax(logits=attn, axis=-1)              # after applying softmax
        attn = self.attn_drop(attn)                             # applying dropout to attention weights, which prevents overfitting
        x = tf.matmul(attn, v)                                  # complete softmax(Q*K^T / sqrt(dimention of head))*V 
        x = tf.transpose(x, perm=[0, 2, 1, 3])                  # convert back to the original data order
        x = tf.reshape(x, (B, -1, self.units))                  
        x = self.proj(x)                                        # the attention sequence is processed through a fully connected layer to output n nodes
        x = self.proj_drop(x)                                   # prevent overfitting
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'sr_ratio': self.sr_ratio,
            'qkv_bias': self.qkv_bias,
            'attn_drop': self.attn_drop,
            'proj_drop': self.proj_drop
        })
        return config