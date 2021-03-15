from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.python.keras.utils import conv_utils
import tensorflow as tf
import tensorflow.keras.backend as K

def normalize_data_format(value):
    if value is None:
        value = K.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format
    

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        
        return tf.image.resize(inputs, [height, width], method=tf.image.ResizeMethod.BILINEAR)

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # k_reduce_mean = Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True, name='global_avg_pooling'))
    # embedding_result=k_reduce_mean(stem_result)


class KerasReduceMean(Layer):
    def __init__(self, axis=(1, 2), keep_dim=True, **kwargs):
        super(KerasReduceMean, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)
        self.axis = axis
        self.keep_dim = keep_dim

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        channel_size = input_shape[3]
        return (batch_size,
                1,
                1,
                channel_size)

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis, keepdims=True, name='global_avg_pooling')

    def get_config(self):
        config = {"axis": self.axis, "keep_dim": self.keep_dim}
        base_config = super(KerasReduceMean, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
