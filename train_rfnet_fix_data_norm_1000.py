from tensorflow.keras import Input, Model
from model_rfnet import RFDN
import csv
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from augment import BasicPolicy
from loss import DepthNorm
import os, pathlib, argparse
import tensorflow as tf
import time
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD
from callbacks import get_nyu_callbacks

SCALE_VALE = 4
SACLE_Y = 1000.0

class DataGenerator(Sequence):
    def __init__(self, data_path, min_depth, max_depth, batch_size, image_shape, depth_shape, n_channels, is_augment=False, shuffle=False, mode="train",
                 is_flip=False, is_addnoise=False, is_erase=False,
                   is_scale=False, scale_value=None):
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.depth_shape = depth_shape
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.is_augment = is_augment
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scale_value = scale_value
        self.policy = BasicPolicy(image_shape=image_shape, color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2,
                                  add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5,
                                  scale_ratio=-1 if not is_scale else 0.2)
        train_csv_path = os.path.join(self.data_path, "data/nyu2_train.csv")
        valid_csv_path = os.path.join(self.data_path, "data/nyu2_test.csv")
        if mode == "train":
            self.imagePath_labelPath_list = self.read_csv(train_csv_path)
            np.random.shuffle(self.imagePath_labelPath_list)
            self.on_epoch_end()
        else:
            self.imagePath_labelPath_list = self.read_csv(valid_csv_path)

    def load_image(self, image_path):
        image = np.asarray(Image.open(image_path)).reshape(*self.image_shape, 3)
        image = cv2.resize(image, (int(self.image_shape[1]/self.scale_value), int(self.image_shape[0]/self.scale_value)), interpolation=cv2.INTER_CUBIC)
        image = image.astype(np.float32)/255.0
        # image = np.clip(image, 0, 1)
        return image

    def __getitem__(self, index):
        batch_imagePath_labelPath_list = self.imagePath_labelPath_list[index*self.batch_size:(index+1)*self.batch_size]
        X, Y = self.generate_batch_X_Y(batch_imagePath_labelPath_list)
        return X, Y

    def generate_batch_X_Y(self, batch_list):
        X = np.zeros((self.batch_size, int(self.image_shape[0]/self.scale_value), int(self.image_shape[1]/self.scale_value), self.n_channels), dtype=np.float32)
        Y = np.zeros((self.batch_size, *self.image_shape, 1), dtype=np.float32)
        for indx, (image_path, label_path) in enumerate(batch_list):
            image_path = os.path.join(self.data_path, image_path)
            label_path = os.path.join(self.data_path, label_path)
            image = np.asarray(Image.open(image_path)).reshape(*self.image_shape, 3)
            label = np.asarray(Image.open(label_path)).astype(np.float32).reshape(*self.image_shape, 1)
            label_mask = np.where(label == 0.0, 0.0, 1.0).astype(np.float32)
            image_new = image * label_mask
            image_new = cv2.resize(image_new, (int(self.image_shape[1] / self.scale_value), int(self.image_shape[0] / self.scale_value)), interpolation=cv2.INTER_LINEAR)
            image_new = image_new.astype(np.float32) / 255.0
            # label = np.clip(label, self.min_depth, self.max_depth)
            label = label / SACLE_Y
            if self.is_augment:
                image, label = self.policy(image, label)
            X[indx, ] = image_new
            Y[indx, ] = label
        return X, Y

    def __len__(self):
        return int(np.floor(len(self.imagePath_labelPath_list)/self.batch_size))

    def read_csv(self, csv_path):
        result = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for item in reader:
                result.append((item[0], item[1]))
        return result

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.imagePath_labelPath_list)

# Argument Parser:wq
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--data_path', default='/home/dujinhua/depth_train', type=str, help='Training dataset.')
parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
parser.add_argument('--bs', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--gpuids', type=str, default="4,5,6,7", help='IDs of GPUs to use')
parser.add_argument('--mindepth', type=float, default=1.0, help='Minimum of input depths')
parser.add_argument('--maxdepth', type=float, default=40000.0, help='Maximum of input depths')
parser.add_argument('--name', type=str, default='rfnet_Norm_1000', help='A name to attach to the training session')
parser.add_argument('--checkpoint', type=str,
                    default="",
                    help='Start training from an existing model.')
parser.add_argument('--full', dest='full', action='store_true', help='Full training with metrics, checkpoints, and image samples.')
parser.add_argument('--is-augment', type=bool, default=False,  help='dataset augmentation')
parser.add_argument('--is-scale', type=bool, default=False,  help='scale image')
parser.add_argument('--is-deconv', type=bool, default=True,  help='add deconv op')
parser.add_argument('--theta', type=float, default=0.1,  help='theta weight')
parser.add_argument('--select_label_mode', type=str, default="baseline",  help='select_label_mode:log1/log/Norm/Norm1/baseline/baseline1')
args = parser.parse_args()

args.name = str(args.name)
args.is_augment = bool(args.is_augment)
args.is_deconv = bool(args.is_deconv)
args.theta = float(args.theta)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuids
# Inform about multi-gpu training
tf.test.is_gpu_available()


def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=args.maxdepth):
    # Point-wise depth
    y_pred = y_pred * SACLE_Y
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


# baseline、baseline1
def loss_sirmse_baseline(truth, pred, mindepth=args.mindepth, maxdepth=args.maxdepth):
    max_thresh = tf.constant(SACLE_Y, dtype=tf.float32)
    truth = truth * max_thresh
    one_shape = tf.ones_like(truth, dtype=tf.float32)
    zero_shape = tf.zeros_like(truth, dtype=tf.float32)
    thresh = tf.constant(mindepth, dtype=tf.float32)
    mask = tf.where(tf.greater(truth, thresh), one_shape, zero_shape)
    num_pixels = K.cast(K.sum(mask), tf.float32)
    # pred = pred * 1000
    pred = pred * max_thresh
    pred = tf.clip_by_value(pred, mindepth, maxdepth)
    truth = tf.clip_by_value(truth, mindepth, maxdepth)
    log_diff = tf.math.multiply((tf.math.log(pred) - tf.math.log(truth)), mask)
    loss_si_rmse = K.sqrt(K.sum(K.square(log_diff)) / num_pixels - K.square(K.sum(log_diff)) / K.square(num_pixels))
    return loss_si_rmse


def loss_rmse_baseline(truth, pred, mindepth=args.mindepth, maxdepth=args.maxdepth):
    max_thresh = tf.constant(SACLE_Y, dtype=tf.float32)
    truth = truth * max_thresh
    one_shape = tf.ones_like(truth, dtype=tf.float32)
    zero_shape = tf.zeros_like(truth, dtype=tf.float32)
    thresh = tf.constant(mindepth, dtype=tf.float32)
    mask = tf.where(tf.greater(truth, thresh), one_shape, zero_shape)
    num_pixels = K.cast(K.sum(mask), tf.float32)
    # pred = pred * 1000
    # truth = truth * max_thresh
    pred = pred * max_thresh
    pred = tf.clip_by_value(pred, mindepth, maxdepth)
    truth = tf.clip_by_value(truth, mindepth, maxdepth)
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
                                is_scale=args.is_scale, scale_value=SCALE_VALE)

val_generator = DataGenerator(data_path=args.data_path, min_depth=args.mindepth,
                              max_depth=args.maxdepth, batch_size=args.bs, image_shape=(480, 640), depth_shape=(240, 320),
                              n_channels=3, is_augment=False, shuffle=False, mode="val",
                              is_flip=False, is_addnoise=False, is_erase=False,
                              is_scale=False, scale_value=SCALE_VALE)


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


print("label process mode:", args.select_label_mode)
loss_fn = tf.keras.losses.MeanSquaredError()
metrics = [loss_rmse_baseline, loss_sirmse_baseline]

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
                              totaL_epochs=args.epochs, warmup_epoch=5, batch_size=args.bs, lr=args.lr,val_loss="val_loss_sirmse_baseline")

# Start training
parallel_model.fit(train_generator, validation_data=val_generator, callbacks=callbacks, epochs=args.epochs, shuffle=True, batch_size=args.bs, verbose=1)
