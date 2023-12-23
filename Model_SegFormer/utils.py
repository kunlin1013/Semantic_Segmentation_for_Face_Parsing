import tensorflow as tf


class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, height, width, name="ResizeLayer", **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.height = height
        self.width = width

    def call(self, inputs):
        resized = tf.image.resize(inputs, size=(self.height, self.width), method=tf.image.ResizeMethod.BILINEAR)
        return resized
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'height': self.height,
            'width': self.width
        })
        return config


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_path, name="DropPath", **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path      # drop ratio

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'drop_path': self.drop_path
        })
        return config