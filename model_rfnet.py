import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Conv2D, ReLU, Concatenate, LeakyReLU, MaxPool2D,UpSampling2D


class ESANet(Model):
    def __init__(self, out_channel):
        super(ESANet, self).__init__()
        temp_channel = out_channel // 4
        self.out_channel = out_channel
        self.conv1 = Conv2D(temp_channel, kernel_size=(1, 1))
        self.conv_f = Conv2D(temp_channel, kernel_size=(1, 1))
        self.conv_max = Conv2D(temp_channel, kernel_size=(3, 3), padding='same')
        self.conv2 = Conv2D(temp_channel, kernel_size=(3, 3), strides=2, padding='same')
        self.conv3 = Conv2D(temp_channel, kernel_size=(3, 3), padding='same')
        self.conv3_ = Conv2D(temp_channel, kernel_size=(3, 3), padding='same')
        self.conv4 = Conv2D( out_channel, kernel_size=(1, 1))
        # self.sigmod = sigmoid()
        self.relu = ReLU()
        self.max_pool2d = MaxPool2D(pool_size=(7, 7), strides=3, padding='same')
        # self.resize = UpSampling2D(size=(4, 4), interpolation="bilinear")

    def __call__(self, inputs):
        input_shape = inputs.get_shape().as_list()
        c1_ = self.conv1(inputs)
        c1 = self.conv2(c1_)
        v_max = self.max_pool2d(c1)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = tf.image.resize(c3, [input_shape[1], input_shape[2]], method=tf.image.ResizeMethod.BILINEAR)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = sigmoid(c4)
        return inputs * m


class RFDB(Model):
    def __init__(self, out_channel):
        super(RFDB, self).__init__()
        filter_left = int(out_channel / 2)
        filter_right = out_channel
        self.out_channel = out_channel
        self.conv_left_1 = Conv2D(filter_left, kernel_size=(1, 1))
        self.act_left_1 = LeakyReLU(alpha=0.05)
        self.conv_left_2 = Conv2D(filter_left, kernel_size=(1, 1))
        self.act_left_2 = LeakyReLU(alpha=0.05)
        self.conv_left_3 = Conv2D(filter_left, kernel_size=(1, 1))
        self.act_left_3 = LeakyReLU(alpha=0.05)
        self.conv_right_final = Conv2D(filter_left, kernel_size=(3, 3), padding='same')
        self.act_right_final = LeakyReLU(alpha=0.05)
        self.concat = Concatenate(axis=-1)
        self.conv_concat = Conv2D(filter_right, kernel_size=(1, 1))
        self.esa = ESANet(filter_right)


    def SRB(self, X):
        X1 = (Conv2D(self.out_channel, kernel_size=(3, 3), padding='same')(X))
        return LeakyReLU(alpha=0.05)(X + X1)

    def __call__(self, inputs):
        left_1 = self.conv_left_1(inputs)
        left_1 = self.act_left_1(left_1)
        right_1 = self.SRB(inputs)

        left_2 = self.conv_left_2(right_1)
        left_2 = self.act_left_2(left_2)
        right_2 = self.SRB(right_1)

        left_3 = self.conv_left_3(right_2)
        left_3 = self.act_left_3(left_3)
        right_3 = self.SRB(right_2)

        right_final = self.conv_right_final(right_3)
        right_final = self.act_right_final(right_final)

        concat = self.concat([left_1, left_2, left_3, right_final])
        concate_1 = self.conv_concat(concat)
        result = self.esa(concate_1)
        return result

class RFDN(Model):
    def __init__(self, out_channel=50, out_class=1, scale_factor=4):
        super(RFDN, self).__init__()
        self.out_class = out_class
        self.scale_factor = scale_factor
        self.fea_conv = Conv2D(out_channel, kernel_size=(3, 3), padding='same')
        self.B1 = RFDB(out_channel)
        self.B2 = RFDB(out_channel)
        self.B3 = RFDB(out_channel)
        self.B4 = RFDB(out_channel)
        self.LR_conv = Conv2D(out_channel, kernel_size=(3, 3), padding='same')
        self.concat = Concatenate(axis=-1)
        self.conv_C = Conv2D(out_channel, kernel_size=(1, 1))
        self.act_C =LeakyReLU(alpha=0.05)

    def __call__(self, inputs):
        out_fea = self.fea_conv(inputs)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B = self.concat([out_B1, out_B2, out_B3, out_B4])
        out_B = self.act_C(self.conv_C(out_B))
        out_lr = self.LR_conv(out_B) + out_fea
        result = Conv2D(self.out_class * (self.scale_factor ** 2), kernel_size=(3, 3), strides=1, padding='same')(out_lr)
        pixel_shuffle_result = tf.nn.depth_to_space(result, self.scale_factor)
        out = Conv2D(1, kernel_size=(1, 1))(pixel_shuffle_result)
        return out
