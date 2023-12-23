import tensorflow as tf
from .Attention import Attention
from .utils import DropPath


class DWConv(tf.keras.layers.Layer):
    # 768 is from ViT's patch, patch size:(16, 16, 3) => flatten => 768 
    # DWconv: Depthwise convolution
    def __init__(self, filters=768, name="DWConv", **kwargs):
        super(DWConv, self).__init__(**kwargs)
        # group convolution: reduce parameters
        self.filters = filters
        self.dwconv = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding="same", groups=filters)

    def call(self, x, H, W):
        # x shape: (batch_size, H, W, C)
        get_shape_1 = tf.shape(x)
        x = tf.reshape(x, (get_shape_1[0], H, W, x.shape[-1]))
        x = self.dwconv(x)
        get_shape_2 = tf.shape(x)
        x = tf.reshape(x, (get_shape_2[0], x.shape[1] * x.shape[2], x.shape[3]))
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters})
        return config


class Mlp(tf.keras.layers.Layer):
    # Mix-FFN: From equation (3) in paper
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, name="Mlp", **kwargs):
        super(Mlp, self).__init__(**kwargs)
        self.in_features = in_features
        self.hidden_features = in_features
        self.out_features = in_features
        self.drop1 = drop
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = tf.keras.layers.Dense(hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = tf.keras.layers.Activation("gelu")
        self.fc2 = tf.keras.layers.Dense(out_features)
        self.drop = tf.keras.layers.Dropout(drop)
        
        # Why should we use Gelu as our activation function?
        # 1. Using sigmoid may cause the gradient to vanish.
        # 2. Relu can mitigate gradient vanishing, but it may lead to dead neuron which do not update anymore.
        # 3. Gelu can give more smooth gradient than relu.

    def call(self, x, H, W):
        x = self.fc1(x)             # MLP(xin)
        x = self.dwconv(x, H, W)    # Conv(MLP(xin))
        x = self.act(x)             # GELU(Conv(MLP(xin)))
        x = self.drop(x)            # prevent overfitting
        x = self.fc2(x)             # MLP(GELU(Conv(MLP(xin))))
        x = self.drop(x)            # prevent overfitting
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'in_features': self.in_features,
            'hidden_features': self.hidden_features,
            'out_features': self.out_features,
            'drop': self.drop1
        })
        return config


class Block(tf.keras.layers.Layer):
    # reference from ViT's encoder block
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, drop_path=0.0, sr_ratio=1, name="Block", **kwargs):
        super(Block, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.sr_ratio = sr_ratio
        
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-05)
        self.attn = Attention(dim, num_heads, sr_ratio, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = (DropPath(drop_path) if drop_path > 0.0 else tf.keras.layers.Layer())  # DropPath can randomly drop a part of path to prevent overfitting.
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-05)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def call(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads ,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'drop': self.drop,
            'attn_drop': self.attn_drop,
            'drop_path': self.drop_path,
            'sr_ratio': self.sr_ratio
        })
        return config
    

class OverlapPatchEmbed(tf.keras.layers.Layer):
    # Overlapped Patch Merging
    def __init__(self, img_size=224, patch_size=7, stride=4, filters=768, name="OverlapPatchEmbed", **kwargs):
        super(OverlapPatchEmbed, self).__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.filters = filters
        
        self.pad = tf.keras.layers.ZeroPadding2D(padding=patch_size // 2)
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=patch_size, strides=stride, padding="VALID", name='proj')
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-05)

    def call(self, x):
        x = self.conv(self.pad(x))
        get_shapes = x.shape
        H = get_shapes[1]
        W = get_shapes[2]
        C = get_shapes[3]
        x = tf.reshape(x, (-1, H * W, C))
        x = self.norm(x)
        return x, H, W
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'img_size': self.img_size,
            'patch_size': self.patch_size,  # 假设 kernel_size 是正方形
            'stride': self.stride,          # 假设 stride 在所有维度上相同
            'filters': self.filters
        })
        return config


class MixVisionTransformer(tf.keras.layers.Layer):
    def __init__(self, img_size=512, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], name="MixVisionTransformer", **kwargs):
        super(MixVisionTransformer, self).__init__(**kwargs)
        self.img_size = img_size
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.sr_ratios = sr_ratios
        
        self.depths = depths
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, filters=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, filters=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, filters=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, filters=embed_dims[3])

        dpr = [x for x in tf.linspace(0.0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = [Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, drop=drop_rate, 
                             attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[0]) for i in range(depths[0])
                       ]
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-05)

        cur += depths[0]
        self.block2 = [Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, drop=drop_rate,
                             attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[1]) for i in range(depths[1])
                       ]
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-05)

        cur += depths[1]
        self.block3 = [Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, drop=drop_rate, 
                             attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[2]) for i in range(depths[2])
                       ]
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-05)

        cur += depths[2]
        self.block4 = [Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, drop=drop_rate,
                             attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[3]) for i in range(depths[3])]
        self.norm4 = tf.keras.layers.LayerNormalization(epsilon=1e-05)

    def call_features(self, x):
        B = tf.shape(x)[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = tf.reshape(x, (B, H, W, x.shape[-1]))
        
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = tf.reshape(x, (B, H, W, x.shape[-1]))
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = tf.reshape(x, (B, H, W, x.shape[-1]))
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = tf.reshape(x, (B, H, W, x.shape[-1]))
        outs.append(x)

        return outs

    def call(self, x):
        x = self.call_features(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'img_size': self.img_size,
            'embed_dims': self.embed_dims,
            'num_heads': self.num_heads,
            'mlp_ratios': self.mlp_ratios,
            'qkv_bias': self.qkv_bias,
            'drop_rate': self.drop_rate,
            'attn_drop_rate': self.attn_drop_rate,
            'drop_path_rate': self.drop_path_rate,
            'depths': self.depths,
            'sr_ratios': self.sr_ratios
        })
        return config