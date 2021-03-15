from tensorflow.keras import Input, Model
from model_rfnet import RFDN

import os, pathlib, argparse
import tensorflow as tf
import time
import tensorflow.keras.backend as K
from data import DataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from callbacks import get_nyu_callbacks

SCALE_VALE = 4

# Argument Parser:wq
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--data_path', default='/home/dujinhua/depth_train', type=str, help='Training dataset.')
parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
parser.add_argument('--bs', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--gpuids', type=str, default="0,1,2,3", help='IDs of GPUs to use')
parser.add_argument('--mindepth', type=float, default=1.0, help='Minimum of input depths')
parser.add_argument('--maxdepth', type=float, default=40000.0, help='Maximum of input depths')
parser.add_argument('--name', type=str, default='rfnet_Norm1', help='A name to attach to the training session')
parser.add_argument('--checkpoint', type=str,
                    default="",
                    help='Start training from an existing model.')
parser.add_argument('--full', dest='full', action='store_true', help='Full training with metrics, checkpoints, and image samples.')
parser.add_argument('--is-augment', type=bool, default=False,  help='dataset augmentation')
parser.add_argument('--is-scale', type=bool, default=False,  help='scale image')
parser.add_argument('--is-deconv', type=bool, default=True,  help='add deconv op')
parser.add_argument('--theta', type=float, default=0.1,  help='theta weight')
parser.add_argument('--select_label_mode', type=str, default="Norm1",  help='select_label_mode:log1/log/Norm/Norm1/baseline/baseline1')
args = parser.parse_args()

args.name = str(args.name)
args.is_augment = bool(args.is_augment)
args.is_deconv = bool(args.is_deconv)
args.theta = float(args.theta)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuids
# Inform about multi-gpu training
tf.test.is_gpu_available()

def DepthNorm(x, maxDepth):
    return maxDepth / x


def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=args.maxdepth):
    # Point-wise depth
    y_pred = y_pred * 1000
    y_pred = tf.clip_by_value(y_pred, args.mindepth, args.maxdepth)
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


def loss_rmse_DepthNorm(truth, pred, mindepth=args.mindepth, maxdepth=args.maxdepth):
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


def loss_sirmse_DepthNorm(truth, pred, mindepth=args.mindepth, maxdepth=args.maxdepth):
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


def loss_rmse_log(truth, pred, mindepth=args.mindepth, maxdepth=args.maxdepth):
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


def loss_sirmse_log(truth, pred,mindepth=args.mindepth, maxdepth=args.maxdepth):
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
def loss_sirmse_baseline(truth, pred, mindepth=args.mindepth, maxdepth=args.maxdepth):
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


def loss_rmse_baseline(truth, pred, mindepth=args.mindepth, maxdepth=args.maxdepth):
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


# model = DepthEstimate()
# Data loaders
print("load data star...")
train_generator = DataGenerator(data_path=args.data_path, min_depth=args.mindepth,
                                max_depth=args.maxdepth, batch_size=args.bs, image_shape=(480, 640), depth_shape=(240, 320),
                                n_channels=3, is_augment=args.is_augment, shuffle=True, mode="train",
                                is_flip=args.is_augment, is_addnoise=args.is_augment, is_erase=args.is_augment,
                                is_scale=args.is_scale, is_deconv=True, select_label_mode=args.select_label_mode, scale_value=SCALE_VALE)

val_generator = DataGenerator(data_path=args.data_path, min_depth=args.mindepth,
                              max_depth=args.maxdepth, batch_size=args.bs, image_shape=(480, 640), depth_shape=(240, 320),
                              n_channels=3, is_augment=False, shuffle=False, mode="val",
                              is_flip=False, is_addnoise=False, is_erase=False,
                              is_scale=False, is_deconv=True, select_label_mode=args.select_label_mode, scale_value=SCALE_VALE)

# rfanet_x = RFDNNet()
# x = Input(shape=(480, 640, 3))
# out = rfanet_x.main_model(x)
# model = Model(inputs=x, outputs=out)
# model.summary()


print("load data finish")

# Training session details
runID = str(int(time.time())) + '-n' + str(len(train_generator)) + '-e' + str(args.epochs) + '-bs' + str(args.bs) + '-lr' + str(args.lr) + '-' + args.name
outputPath = './models/'
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

# Multi-gpu setup:
strategy = tf.distribute.MirroredStrategy()
optimizer = Adam(lr=args.lr, amsgrad=True)

# rfanet_x = RFDNNet()
# x = Input(shape=(120, 160, 3))
# out = rfanet_x.main_model(x, SCALE_VALE)
# parallel_model = Model(inputs=x, outputs=out)
# parallel_model.compile(loss=loss_rmse_DepthNorm, optimizer=optimizer, metrics=[loss_rmse_DepthNorm, loss_sirmse_DepthNorm])

# log1/log/Norm/Norm1/baseline/baseline1
if args.select_label_mode == "log1" or args.select_label_mode == "log1":
    metrics = [loss_rmse_log, loss_sirmse_log]
    val_loss = "val_loss_sirmse_log"
elif args.select_label_mode == "Norm" or args.select_label_mode == "Norm1":
    metrics = [loss_rmse_DepthNorm, loss_sirmse_DepthNorm]
    val_loss = "val_loss_sirmse_DepthNorm"
elif args.select_label_mode == "baseline" or args.select_label_mode == "baseline1":
    metrics = [loss_rmse_baseline, loss_sirmse_baseline]
    val_loss = "val_loss_sirmse_baseline"
else:
    exit(0)

print("label process mode:", args.select_label_mode)
loss_fn = tf.keras.losses.MeanSquaredError()

with strategy.scope():
    rfanet_x = RFDN()
    x = Input(shape=(120, 160, 3))
    out = rfanet_x(x)
    parallel_model = Model(inputs=x, outputs=out)
    parallel_model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)

parallel_model.summary()
print('Ready for training!\n')

# Callbacks
callbacks = get_nyu_callbacks(parallel_model, train_generator, val_generator, runPath,
                              totaL_epochs=args.epochs, warmup_epoch=5, batch_size=args.bs, lr=args.lr, val_loss=val_loss)

# Start training
parallel_model.fit(train_generator, validation_data=val_generator, callbacks=callbacks, epochs=args.epochs, shuffle=True, batch_size=args.bs, verbose=1)
