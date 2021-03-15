from tensorflow.keras_applications.imagenet_utils import _obtain_input_shape
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.engine.topology import get_source_inputs
from tensorflow.keras.layers import Activation, Add, Concatenate, Conv2D, Multiply
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D, AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D
from layers import KerasReduceMean, BilinearUpSampling2D

def BisenetV2(include_top=True,
              input_tensor=None,
              input_shape=(224, 224, 3),
              weights=None
              ):
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only tensorflow supported for now')
    name = "bisenetv2"
    input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=28, require_flatten=include_top,
                                      data_format=K.image_data_format())
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    img_input1 = Conv2D(16, kernel_size=(3, 3), strides=2, padding="same", use_bias=False, name="stem_block/conv_block_1")(img_input)
    img_input1 = BatchNormalization(axis=-1, name="stem_block/conv_block_1/bn_1")(img_input1)
    img_input1 = Activation(activation="relu", name="stem_block/conv_block_1/activate_1")(img_input1)

    branch_left_output = Conv2D(int(16/2), kernel_size=(1, 1), strides=1, padding="same", use_bias=False, name="stem_block/downsample_branch_left/1x1_conv_block")(img_input1)
    branch_left_output = BatchNormalization(axis=-1, name="stem_block/downsample_branch_left/1x1_conv_block/bn_1")(branch_left_output)
    branch_left_output = Activation(activation="relu", name="stem_block/downsample_branch_left/1x1_conv_block/activate_1")(branch_left_output)


    branch_left_output = Conv2D(16, kernel_size=(3, 3), strides=2, padding="same", use_bias=False,
                                name="stem_block/downsample_branch_left/3x3_conv_block")(branch_left_output)
    branch_left_output = BatchNormalization(axis=-1, name="stem_block/downsample_branch_left/3x3_conv_block/bn_1")(branch_left_output)
    branch_left_output = Activation(activation="relu", name="stem_block/downsample_branch_left/3x3_conv_block/activate_1")(branch_left_output)


    branch_right_output = MaxPool2D(pool_size=(3, 3), strides=2, padding='same', name="stem_block/downsample_branch_right/maxpooling_block")(img_input1)
    stem_result = Concatenate(axis=-1, name="stem_block/concate_features")([branch_left_output, branch_right_output])
    stem_result = Conv2D(16, kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name="stem_block/final_conv_block")(stem_result)
    stem_result = BatchNormalization(axis=-1, name="stem_block/final_conv_block/bn_1")(stem_result)
    stem_result = Activation(activation="relu", name="stem_block/final_conv_block/activate_1")(stem_result)

    # k_reduce_mean = Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True, name='global_avg_pooling'))
    # embedding_result=k_reduce_mean(stem_result)
    # embedding_result = K.mean(stem_result, axis=[1, 2], keepdims=True)
    embedding_result = KerasReduceMean(axis=(1, 2), keep_dim=True, name="global_avg_pooling")(stem_result)

    embedding_result = BatchNormalization(axis=-1, name="context_embedding_block/bn")(embedding_result)
    output_channels = stem_result.get_shape().as_list()[-1]
    embedding_result = Conv2D(output_channels, kernel_size=(1, 1), strides=1, padding="same", use_bias=False,
                              name="context_embedding_block/conv_block_1")(embedding_result)
    embedding_result = BatchNormalization(axis=-1, name="context_embedding_block/conv_block_1/bn_1")(embedding_result)
    embedding_result = Activation(activation="relu", name="context_embedding_block/conv_block_1/activate_1")(embedding_result)
    embedding_result = Add(name="context_embedding_block/fused_features")([embedding_result, stem_result])
    embedding_result = Conv2D(output_channels, kernel_size=(3, 3), strides=1, padding="same", use_bias=False, name="context_embedding_block/final_conv_block")(embedding_result)


    output_channels = embedding_result.get_shape().as_list()[-1]
    gather_expansion_result = Conv2D(output_channels, kernel_size=(3, 3), strides=1, padding="same", use_bias=False,
                                     name="ge_block_with_stride_1/stride_equal_one_module/3x3_conv_block")(embedding_result)
    gather_expansion_result = BatchNormalization(axis=-1, name="ge_block_with_stride_1/stride_equal_one_module/3x3_conv_block/bn_1")(gather_expansion_result)
    gather_expansion_result = Activation(activation="relu", name="ge_block_with_stride_1/stride_equal_one_module/3x3_conv_block/activate_1")(gather_expansion_result)

    gather_expansion_result = DepthwiseConv2D(kernel_size=3, strides=1, depth_multiplier=6, padding='same',
                                              name="ge_block_with_stride_1/stride_equal_one_module/depthwise_conv_block")(gather_expansion_result)
    gather_expansion_result = BatchNormalization(axis=-1, name="ge_block_with_stride_1/stride_equal_one_module/dw_bn")(gather_expansion_result)

    gather_expansion_result = Conv2D(output_channels, kernel_size=(1, 1), strides=1, padding="same", use_bias=False,
                                     name="ge_block_with_stride_1/stride_equal_one_module/1x1_conv_block")(gather_expansion_result)
    gather_expansion_result = Add(name="ge_block_with_stride_1/stride_equal_one_module/fused_features")([embedding_result, gather_expansion_result])
    gather_expansion_result = Activation(activation="relu", name="ge_block_with_stride_1/stride_equal_one_module/ge_output")(gather_expansion_result)

    gather_expansion_proj_result = DepthwiseConv2D(kernel_size=3, depth_multiplier=1, strides=2, padding="same",
                                                   name="ge_block_with_stride_2/stride_equal_two_module/input_project_dw_conv_block")(gather_expansion_result)
    gather_expansion_proj_result = BatchNormalization(axis=-1, name="ge_block_with_stride_2/stride_equal_two_module/input_project_bn")(gather_expansion_proj_result)
    gather_expansion_proj_result = Conv2D(128, kernel_size=(1, 1), strides=1, padding="same", use_bias=False, activation=None)(gather_expansion_proj_result)
    input_tensor_channels = gather_expansion_result.get_shape().as_list()[-1]
    gather_expansion_stride2_result = Conv2D(input_tensor_channels, kernel_size=(3, 3), strides=1, padding="same",
                                             use_bias=False, name="ge_block_with_stride_2/stride_equal_two_module/3x3_conv_block")(gather_expansion_result)
    gather_expansion_stride2_result = BatchNormalization(axis=-1, name="ge_block_with_stride_2/stride_equal_two_module/3x3_conv_block/bn_1")(gather_expansion_stride2_result)
    gather_expansion_stride2_result = Activation(activation="relu", name="ge_block_with_stride_2/stride_equal_two_module/3x3_conv_block/activate_1")(gather_expansion_stride2_result)

    gather_expansion_stride2_result = DepthwiseConv2D(kernel_size=3, depth_multiplier=6, strides=2, padding="same",
                                                      name="ge_block_with_stride_2/stride_equal_two_module/depthwise_conv_block_1")(gather_expansion_stride2_result)
    gather_expansion_stride2_result = BatchNormalization(axis=-1, name="ge_block_with_stride_2/stride_equal_two_module/dw_bn_1")(gather_expansion_stride2_result)
    gather_expansion_stride2_result = DepthwiseConv2D(kernel_size=3, depth_multiplier=1, strides=1, padding="same",
                                                      name="ge_block_with_stride_2/stride_equal_two_module/depthwise_conv_block_2")(gather_expansion_stride2_result)
    gather_expansion_stride2_result = BatchNormalization(axis=-1, name="ge_block_with_stride_2/stride_equal_two_module/dw_bn_2")(gather_expansion_stride2_result)
    gather_expansion_stride2_result = Conv2D(128, kernel_size=(1, 1), strides=1, padding="same",
                                             use_bias=False, activation=None, name="ge_block_with_stride_2/stride_equal_two_module/1x1_conv_block")(gather_expansion_stride2_result)
    gather_expansion_total_result = Add(name="ge_block_with_stride_2/stride_equal_two_module/fused_features")([gather_expansion_proj_result, gather_expansion_stride2_result])
    gather_expansion_total_result = Activation(activation="relu", name="ge_block_with_stride_2/stride_equal_two_module/ge_output")(gather_expansion_total_result)


    gather_expansion_proj2_result = DepthwiseConv2D(kernel_size=3, depth_multiplier=1, strides=2, padding="same",
                                                   name="ge_block_with_stride_2_repeat/stride_equal_two_module/input_project_dw_conv_block")(gather_expansion_total_result)
    gather_expansion_proj2_result = BatchNormalization(axis=-1, name="ge_block_with_stride_2_repeat/stride_equal_two_module/input_project_bn")(gather_expansion_proj2_result)
    gather_expansion_proj2_result = Conv2D(128, kernel_size=(1, 1), strides=1, padding="same", use_bias=False, activation=None)(gather_expansion_proj2_result)
    input_tensor_channels = gather_expansion_total_result.get_shape().as_list()[-1]
    gather_expansion_stride2_result_repeat = Conv2D(input_tensor_channels, kernel_size=(3, 3), strides=1,  padding="same",
                                             use_bias=False, name="ge_block_with_stride_2_repeat/stride_equal_two_module/3x3_conv_block")(gather_expansion_total_result)
    gather_expansion_stride2_result_repeat = BatchNormalization(axis=-1, name="ge_block_with_stride_2_repeat/stride_equal_two_module/3x3_conv_block/bn_1")(gather_expansion_stride2_result_repeat)
    gather_expansion_stride2_result_repeat = Activation(activation="relu", name="ge_block_with_stride_2_repeat/stride_equal_two_module/3x3_conv_block/activate_1")(gather_expansion_stride2_result_repeat)

    gather_expansion_stride2_result_repeat = DepthwiseConv2D(kernel_size=3, depth_multiplier=6, strides=2, padding="same",
                                                      name="ge_block_with_stride_2_repeat/stride_equal_two_module/depthwise_conv_block_1")(gather_expansion_stride2_result_repeat)
    gather_expansion_stride2_result_repeat = BatchNormalization(axis=-1, name="ge_block_with_stride_2_repeat/stride_equal_two_module/dw_bn_1")(gather_expansion_stride2_result_repeat)
    gather_expansion_stride2_result_repeat = DepthwiseConv2D(kernel_size=3, depth_multiplier=1, strides=1, padding="same",
                                                      name="ge_block_with_stride_2_repeat/stride_equal_two_module/depthwise_conv_block_2")(gather_expansion_stride2_result_repeat)
    gather_expansion_stride2_result_repeat = BatchNormalization(axis=-1, name="ge_block_with_stride_2_repeat/stride_equal_two_module/dw_bn_2")(gather_expansion_stride2_result_repeat)
    gather_expansion_stride2_result_repeat = Conv2D(128, kernel_size=(1, 1), strides=1, padding="same",
                                             use_bias=False, activation=None, name="ge_block_with_stride_2_repeat/stride_equal_two_module/1x1_conv_block")(gather_expansion_stride2_result_repeat)
    gather_expansion_total_result_repeat = Add(name="ge_block_with_stride_2_repeat/stride_equal_two_module/fused_features")([gather_expansion_proj2_result, gather_expansion_stride2_result_repeat])
    gather_expansion_total_result_repeat = Activation(activation="relu", name="ge_block_with_stride_2_repeat/stride_equal_two_module/ge_output")(gather_expansion_total_result_repeat)

    detail_input_tensor = stem_result
    semantic_input_tensor = gather_expansion_total_result_repeat
    output_channels = stem_result.get_shape().as_list()[-1]
    detail_branch_remain = DepthwiseConv2D(kernel_size=3, strides=1, padding="same", depth_multiplier=1,
                                           name="guided_aggregation_block/detail_branch/3x3_dw_conv_block")(detail_input_tensor)
    detail_branch_remain = BatchNormalization(axis=-1, name="guided_aggregation_block/detail_branch/bn_1")(detail_branch_remain)
    detail_branch_remain = Conv2D(output_channels, kernel_size=(1, 1), padding="same", strides=1, use_bias=False,
                                  name="guided_aggregation_block/detail_branch/1x1_conv_block")(detail_branch_remain)

    detail_branch_downsample = Conv2D(output_channels, kernel_size=(3, 3), strides=2, use_bias=False, activation=None,
                                      padding="same", name="guided_aggregation_block/detail_branch/3x3_conv_block")(detail_input_tensor)

    detail_branch_downsample = AveragePooling2D(pool_size=(3, 3), strides=2, padding="same", name="guided_aggregation_block/detail_branch/avg_pooling_block")(detail_branch_downsample)

    semantic_branch_remain = DepthwiseConv2D(kernel_size=3, strides=1, padding="same", depth_multiplier=1,
                                             name="guided_aggregation_block/semantic_branch/3x3_dw_conv_block")(semantic_input_tensor)
    semantic_branch_remain = BatchNormalization(axis=-1, name="guided_aggregation_block/semantic_branch/bn_1")(semantic_branch_remain)
    semantic_branch_remain = Conv2D(output_channels, kernel_size=(1, 1), strides=1, use_bias=False, activation=None, padding="same",
                                    name="guided_aggregation_block/semantic_branch/1x1_conv_block")(semantic_branch_remain)
    # semantic_branch_remain = sigmoid(semantic_branch_remain)
    # keras_sigmoid = Lambda(lambda x: tf.nn.sigmoid(x, name="guided_aggregation_block/semantic_branch/semantic_remain_sigmoid"))
    # semantic_branch_remain = keras_sigmoid(semantic_branch_remain)
    semantic_branch_remain = Activation("sigmoid", name="guided_aggregation_block/semantic_branch/semantic_remain_sigmoid")(semantic_branch_remain)

    semantic_branch_upsample = Conv2D(output_channels, kernel_size=(3, 3), strides=1, padding="same", use_bias=False,
                                      activation=None, name="guided_aggregation_block/semantic_branch/3x3_conv_block")(semantic_input_tensor)
    # semantic_branch_upsample = resize_images(semantic_branch_upsample, 4, 4, data_format="channels_last", interpolation='bilinear')

    # upsample_bilinear0 = Lambda(lambda x: tf.image.resize_bilinear(x, size=stem_result.get_shape().as_list()[1:3],
    #                                                               name="guided_aggregation_block/semantic_branch/semantic_upsample_features"))
    # semantic_branch_upsample = upsample_bilinear0(semantic_branch_upsample)
    semantic_branch_upsample = BilinearUpSampling2D((4, 4), name="guided_aggregation_block/semantic_branch/semantic_upsample_features")(semantic_branch_upsample)
    semantic_branch_upsample = Activation("sigmoid", name="guided_aggregation_block/semantic_branch/semantic_branch_upsample_sigmoid")(semantic_branch_upsample)
    # keras_sigmoid_1 = Lambda(lambda x: tf.nn.sigmoid(x, name="guided_aggregation_block/semantic_branch/semantic_branch_upsample_sigmoid"))
    # semantic_branch_upsample = keras_sigmoid_1(semantic_branch_upsample)
    # semantic_branch_upsample = sigmoid(semantic_branch_upsample)

    guided_features_remain = Multiply(name="guided_aggregation_block/aggregation_features/guided_detail_features")([detail_branch_remain, semantic_branch_upsample])
    guided_features_downsample = Multiply(name="guided_aggregation_block/aggregation_features/guided_semantic_features")([detail_branch_downsample, semantic_branch_remain])

    # upsample_bilinear1 = Lambda(lambda x: tf.image.resize_bilinear(x, size=stem_result.get_shape().as_list()[1:3],
    #                                        name="guided_aggregation_block/aggregation_features/guided_upsample_features"))
    #
    # guided_features_upsample = upsample_bilinear1(guided_features_downsample)
    guided_features_upsample = BilinearUpSampling2D((4, 4), name="guided_aggregation_block/aggregation_features/guided_upsample_features")(guided_features_downsample)
    # guided_features_upsample = resize_images(guided_features_downsample, 4, 4, data_format="channels_last", interpolation='bilinear')

    guided_features = Add(name="guided_aggregation_block/aggregation_features/fused_features")([guided_features_remain, guided_features_upsample])
    guided_features = Conv2D(output_channels, kernel_size=(3, 3), strides=1, use_bias=False, padding="same",
                             name="guided_aggregation_block/aggregation_features/aggregation_feature_output")(guided_features)
    guided_features = BatchNormalization(axis=-1, name="guided_aggregation_block/aggregation_features/aggregation_feature_output/bn_1")(guided_features)
    guided_features = Activation(activation="relu", name="guided_aggregation_block/aggregation_features/aggregation_feature_output/activate_1")(guided_features)

    # input_tensor_size = [int(tmp * 4)for tmp in guided_features.get_shape().as_list()[1:3]]
    result = Conv2D(8, kernel_size=(3, 3), strides=1, use_bias=False, padding="same", name="seg_head_block/3x3_conv_block")(guided_features)
    result = BatchNormalization(axis=-1, name="seg_head_block/bn_1")(result)
    result = Activation("relu", name="seg_head_block/activate_1")(result)

    # upsample_bilinear2 = Lambda(lambda x: tf.image.resize_bilinear(x, size=input_tensor_size, name="seg_head_block/segmentation_head_logits"))
    # result = upsample_bilinear2(result)
    result = BilinearUpSampling2D((4, 4), name="seg_head_block/segmentation_head_upsample")(result)
    # result = resize_images(result, 4, 4, data_format="channels_last", interpolation='bilinear')

    result = Conv2D(1, kernel_size=(1, 1), strides=1, use_bias=False, padding="same",
                    name="seg_head_block/1x1_conv_block")(result)
    if input_tensor:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, result, name=name)

    if weights:
        model.load_weights(weights, by_name=True)

    return model
