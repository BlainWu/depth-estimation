import io
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def make_image(tensor):
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor.astype('uint8'))
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=90)
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)

def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    if total_steps < warmup_steps:
       raise ValueError('total_steps must be larger or equal to  warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
        (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        #线性增长的实现
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        #只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
        return np.where(global_step > total_steps, 0.0, learning_rate)
   # """
   # 参数：
   # 		global_step: 上面定义的Tcur，记录当前执行的步数。
   # 		learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降。
   # 		total_steps: 是总的训练的步数，等于epoch*sample_count/batch_size,(sample_count是样本总数，epoch是总的循环次数)
   # 		warmup_learning_rate: 这是warm up阶段线性增长的初始值
   # 		warmup_steps: warm_up总的需要持续的步数
   # 		hold_base_rate_steps: 这是可选的参数，即当warm up阶段结束后保持学习率不变，知道hold_base_rate_steps结束后才开始学习率下降
   # """


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        #learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


def get_nyu_callbacks(model, train_generator, test_generator, runPath,
                      totaL_epochs, batch_size, lr, warmup_epoch=5, sirmse=None):
    callbacks = []

    # Callback: Tensorboard
    # class LRTensorBoard(keras.callbacks.TensorBoard):
    #     def __init__(self, log_dir):
    #         super().__init__(log_dir=log_dir)
    #
    #         self.num_samples = 6
    #         self.train_idx = np.random.randint(low=0, high=len(train_generator), size=10)
    #         self.test_idx = np.random.randint(low=0, high=len(test_generator), size=10)
    #
    #     def on_epoch_end(self, epoch, logs=None):
    #         if not test_set == None:
    #             # Samples using current model
    #             import matplotlib.pyplot as plt
    #             from skimage.transform import resize
    #             plasma = plt.get_cmap('plasma')
    #
    #             minDepth, maxDepth = 10, 1000
    #
    #             train_samples = []
    #             test_samples = []
    #
    #             for i in range(self.num_samples):
    #                 x_train, y_train = train_generator.__getitem__(self.train_idx[i], False)
    #                 x_test, y_test = test_generator[self.test_idx[i]]
    #
    #                 x_train, y_train = x_train[0], np.clip(DepthNorm(y_train[0], maxDepth=1000), minDepth, maxDepth) / maxDepth
    #                 x_test, y_test = x_test[0], np.clip(DepthNorm(y_test[0], maxDepth=1000), minDepth, maxDepth) / maxDepth
    #
    #                 h, w = y_train.shape[0], y_train.shape[1]
    #
    #                 rgb_train = resize(x_train, (h,w), preserve_range=True, mode='reflect', anti_aliasing=True)
    #                 rgb_test = resize(x_test, (h,w), preserve_range=True, mode='reflect', anti_aliasing=True)
    #
    #                 gt_train = plasma(y_train[:,:,0])[:,:,:3]
    #                 gt_test = plasma(y_test[:,:,0])[:,:,:3]
    #
    #                 predict_train = plasma(predict(model, x_train, minDepth=minDepth, maxDepth=maxDepth)[0,:,:,0])[:,:,:3]
    #                 predict_test = plasma(predict(model, x_test, minDepth=minDepth, maxDepth=maxDepth)[0,:,:,0])[:,:,:3]
    #
    #                 train_samples.append(np.vstack([rgb_train, gt_train, predict_train]))
    #                 test_samples.append(np.vstack([rgb_test, gt_test, predict_test]))
    #
    #             self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Train', image=make_image(255 * np.hstack(train_samples)))]), epoch)
    #             self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Test', image=make_image(255 * np.hstack(test_samples)))]), epoch)
    #
    #             # Metrics
    #             e = evaluate(model, test_set['rgb'], test_set['depth'], test_set['crop'], batch_size=6, verbose=True)
    #             logs.update({'rel': e[3]})
    #             logs.update({'rms': e[4]})
    #             logs.update({'log10': e[5]})
    #
    #         super().on_epoch_end(epoch, logs)
    # callbacks.append(LRTensorBoard(log_dir=runPath))
    # Create the Learning rate scheduler.
    # warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=lr,
    #                                         total_steps=int(totaL_epochs * len(train_generator)),
    #                                         warmup_learning_rate=4e-05,
    #                                         warmup_steps=int(warmup_epoch * len(train_generator)),
    #                                         hold_base_rate_steps=5,
    #                                         )
    # callbacks.append(warm_up_lr)

    # Callback: Learning Rate Scheduler
    lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, min_lr=0.00000000009)
    callbacks.append(lr_schedule)  # reduce learning rate when stuck

    # earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min')
    # callbacks.append(earlyStopping)

    # Callback: save checkpoints
    best_weights_filepath = os.path.join(runPath,"{epoch:d}_{" + str(sirmse) + ":.5f}")
    callbacks.append(keras.callbacks.ModelCheckpoint(filepath=best_weights_filepath, monitor=str(sirmse),
        verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=2))
    return callbacks
