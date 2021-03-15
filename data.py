import os
import csv
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from augment import BasicPolicy
from loss import DepthNorm


class DataGenerator(Sequence):
    def __init__(self, data_path, min_depth, max_depth, batch_size, image_shape, depth_shape, n_channels, is_augment=False, shuffle=False, mode="train",
                 is_flip=False, is_addnoise=False, is_erase=False,
                   is_scale=False, is_deconv=False, select_label_mode=None, scale_value=None):
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.depth_shape = depth_shape
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.is_augment = is_augment
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.is_deconv = is_deconv
        self.scale_value = scale_value
        self.select_label_mode = select_label_mode
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
        if self.scale_value != None:
            X = np.zeros((self.batch_size, int(self.image_shape[0]/self.scale_value), int(self.image_shape[1]/self.scale_value), self.n_channels))
        else:
            X = np.zeros((self.batch_size, *self.image_shape, self.n_channels))
        if not self.is_deconv:
            Y = np.zeros((self.batch_size, *self.depth_shape, 1))
        else:
            Y = np.zeros((self.batch_size, *self.image_shape, 1))
        for indx, (image_path, label_path) in enumerate(batch_list):
            image_path = os.path.join(self.data_path, image_path)
            label_path = os.path.join(self.data_path, label_path)
            image = self.load_image(image_path)
            label = self.load_label(label_path)
            if not self.is_deconv:
                from skimage.transform import resize
                label = resize(label, output_shape=(int(self.depth_shape[0]), int(self.depth_shape[1])), preserve_range=True, mode='reflect', anti_aliasing=True)
            if self.is_augment:
                image, label = self.policy(image, label)
            X[indx, ] = image
            Y[indx, ] = label
        return X, Y

    def __len__(self):
        return int(np.floor(len(self.imagePath_labelPath_list)/self.batch_size))

    def load_label(self, label_path):
        label = np.asarray(Image.open(label_path)).astype(np.float32).reshape(*self.image_shape, 1)
        if self.select_label_mode == "baseline1":
            label = np.where(label == 0.0, self.max_depth, label)
            label = np.clip(label, self.min_depth, self.max_depth)
        elif self.select_label_mode == "baseline":
            label = np.clip(label, self.min_depth, self.max_depth)
        elif self.select_label_mode == "Norm1":
            label = np.where(label == 0.0, self.max_depth, label)
            label = np.clip(label, self.min_depth, self.max_depth)
            label = DepthNorm(label, self.max_depth)
        elif self.select_label_mode == "Norm":
            label = np.clip(label, self.min_depth, self.max_depth)
            label = DepthNorm(label, self.max_depth)
        elif self.select_label_mode == "log":
            label = np.clip(label, self.min_depth, self.max_depth)
            label = np.log(label)
        elif self.select_label_mode == "log1":
            label = np.where(label == 0.0, self.max_depth, label)
            label = np.clip(label, self.min_depth, self.max_depth)
            label = np.log(label)
        return label


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