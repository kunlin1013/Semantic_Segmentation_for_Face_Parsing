import tensorflow as tf


class MLP(tf.keras.layers.Layer):
    def __init__(self, decode_dim, name="MLP", **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.decode_dim = decode_dim
        self.proj = tf.keras.layers.Dense(decode_dim)

    def call(self, x):
        x = self.proj(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({'decode_dim': self.decode_dim})
        return config


class ConvModule(tf.keras.layers.Layer):
    def __init__(self, decode_dim, name="ConvModule", **kwargs):
        super(ConvModule, self).__init__(**kwargs)
        self.decode_dim = decode_dim
        self.conv = tf.keras.layers.Conv2D(filters=decode_dim, kernel_size=1, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self.activate = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'decode_dim': self.decode_dim,
        })
        return config


class SegFormerHead(tf.keras.layers.Layer):
    # corresponding to the MLP Layer in the paper
    def __init__(self, num_mlp_layers=4, decode_dim=768, num_classes=19, name="SegFormerHead", **kwargs):
        super(SegFormerHead, self).__init__(**kwargs)
        self.num_mlp_layers = num_mlp_layers
        self.decode_dim = decode_dim
        self.num_classes = num_classes
        
        self.linear_layers = []
        for _ in range(num_mlp_layers):
            self.linear_layers.append(MLP(decode_dim))

        self.linear_fuse = ConvModule(decode_dim)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.linear_pred = tf.keras.layers.Conv2D(num_classes, kernel_size=1)

    def call(self, inputs):
        # input feature map which is the output from Transformer Block N in the paper
        # in the paper, feature map will have 4 pictures
        # first feature map's shape: (image height / 4, image width / 4, C1) 
        H = inputs[0].shape[1]  # the height from first feature map
        W = inputs[0].shape[2]  # the width from first feature map
        outputs = []

        for x, mlps in zip(inputs, self.linear_layers):
            x = mlps(x)                                             # equation (4-1) in the paper
            x = tf.image.resize(x, size=(H, W), method="bilinear")  # let the shape of all feature maps be equal to the first feature map, equation (4-2) in the paper
            outputs.append(x)

        x = self.linear_fuse(tf.concat(outputs[::-1], axis=3))      # 1. concat 4 feature maps from equation (4-2), and you wull obtain data with 4*C layers. 2. use a fully connected layer to conver channel from 4*C to C. 
        x = self.dropout(x)                                         # prevent overfitting
        x = self.linear_pred(x)                                     # output layer

        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_mlp_layers': self.num_mlp_layers,
            'decode_dim': self.decode_dim,
            'num_classes': self.num_classes
        })
        return config