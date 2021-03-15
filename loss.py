import tensorflow as tf
import tensorflow.keras.backend as K


def DepthNorm(x, maxDepth):
    return maxDepth / x

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0 / 10.0):
    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta
    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))


def loss_rmse_DepthNorm(truth, pred, mindepth, maxdepth):
    truth = DepthNorm(truth, maxdepth)
    one_shape = tf.ones_like(truth, dtype=tf.float32)
    zero_shape = tf.zeros_like(truth, dtype=tf.float32)
    thresh = tf.constant(mindepth, dtype=tf.float32)
    mask = tf.where(tf.greater(truth, thresh), one_shape, zero_shape)
    num_pixels = K.cast(K.sum(mask), tf.float32)
    # pred = pred * 1000
    pred = DepthNorm(pred, maxdepth)
    pred = tf.clip_by_value(pred, mindepth, maxdepth)
    diff = tf.math.multiply(pred - truth, mask) / 1000.0    # mapping the distance from millimeters to meters

    loss_mse = K.sum(K.square(diff)) / num_pixels
    loss_rmse = K.sqrt(loss_mse)
    return loss_rmse


def loss_sirmse_DepthNorm(truth, pred, mindepth, maxdepth):
    truth = DepthNorm(truth, maxdepth)
    one_shape = tf.ones_like(truth, dtype=tf.float32)
    zero_shape = tf.zeros_like(truth, dtype=tf.float32)
    thresh = tf.constant(mindepth, dtype=tf.float32)
    mask = tf.where(tf.greater(truth, thresh), one_shape, zero_shape)
    num_pixels = K.cast(K.sum(mask), tf.float32)
    # pred = pred * 1000
    pred = DepthNorm(pred, maxdepth)
    pred = tf.clip_by_value(pred, mindepth, maxdepth)
    log_diff = tf.math.multiply((tf.math.log(pred) - tf.math.log(truth)), mask)
    loss_si_rmse = K.sqrt(K.sum(K.square(log_diff)) / num_pixels - K.square(K.sum(log_diff)) / K.square(num_pixels))
    return loss_si_rmse


def loss_rmse_log(truth, pred, mindepth, maxdepth):
    truth = DepthNorm(truth, maxdepth)
    one_shape = tf.ones_like(truth, dtype=tf.float32)
    zero_shape = tf.zeros_like(truth, dtype=tf.float32)
    thresh = tf.constant(mindepth, dtype=tf.float32)
    mask = tf.where(tf.greater(truth, thresh), one_shape, zero_shape)
    num_pixels = K.cast(K.sum(mask), tf.float32)
    pred = K.exp(pred)
    pred = tf.clip_by_value(pred, mindepth, maxdepth)
    diff = tf.math.multiply(pred - truth, mask) / 1000.0  # mapping the distance from millimeters to meters

    loss_mse = K.sum(K.square(diff)) / num_pixels
    loss_rmse = K.sqrt(loss_mse)
    return loss_rmse


def loss_sirmse_log(truth, pred, mindepth, maxdepth):
    truth = DepthNorm(truth, maxdepth)
    one_shape = tf.ones_like(truth, dtype=tf.float32)
    zero_shape = tf.zeros_like(truth, dtype=tf.float32)
    thresh = tf.constant(mindepth, dtype=tf.float32)
    mask = tf.where(tf.greater(truth, thresh), one_shape, zero_shape)
    num_pixels = K.cast(K.sum(mask), tf.float32)

    pred = K.exp(pred)
    pred = tf.clip_by_value(pred, mindepth, maxdepth)
    log_diff = tf.math.multiply((tf.math.log(pred) - tf.math.log(truth)), mask)
    loss_si_rmse = K.sqrt(K.sum(K.square(log_diff)) / num_pixels - K.square(K.sum(log_diff)) / K.square(num_pixels))
    return loss_si_rmse


# baseline、baseline1
def loss_sirmse_baseline(truth, pred, mindepth, maxdepth):
    one_shape = tf.ones_like(truth, dtype=tf.float32)
    zero_shape = tf.zeros_like(truth, dtype=tf.float32)
    thresh = tf.constant(mindepth, dtype=tf.float32)
    mask = tf.where(tf.greater(truth, thresh), one_shape, zero_shape)
    num_pixels = K.cast(K.sum(mask), tf.float32)
    pred = pred * 1000
    pred = tf.clip_by_value(pred, mindepth, maxdepth)
    log_diff = tf.math.multiply((tf.math.log(pred) - tf.math.log(truth)), mask)
    loss_si_rmse = K.sqrt(K.sum(K.square(log_diff)) / num_pixels - K.square(K.sum(log_diff)) / K.square(num_pixels))
    return loss_si_rmse


def loss_rmse_baseline(truth, pred, mindepth, maxdepth):
    one_shape = tf.ones_like(truth, dtype=tf.float32)
    zero_shape = tf.zeros_like(truth, dtype=tf.float32)
    thresh = tf.constant(mindepth, dtype=tf.float32)
    mask = tf.where(tf.greater(truth, thresh), one_shape, zero_shape)
    num_pixels = K.cast(K.sum(mask), tf.float32)
    pred = pred * 1000
    pred = tf.clip_by_value(pred, mindepth, maxdepth)
    diff = tf.math.multiply(pred - truth, mask) / 1000.0    # mapping the distance from millimeters to meters

    loss_mse = K.sum(K.square(diff)) / num_pixels
    loss_rmse = K.sqrt(loss_mse)
    return loss_rmse