__author__ = 'Pavel Yatvetsky'

import keras as keras
import os
from struct import pack, unpack

class TensorboardCallback(keras.callbacks.TensorBoard):
    def __init__(self, TrainStart=-1, **kwargs):
        super(TensorboardCallback, self).__init__(**kwargs)
        self.TrainStart = TrainStart

    def set_model(self, model):
        super(TensorboardCallback, self).set_model(model)

        log_filename = self.log_dir + '/TB2.log'
        if os.path.exists(log_filename):
            self.params_file = open(log_filename, 'r+b')
            self.current_epoch = unpack('I', self.params_file.read())[0] + self.TrainStart + 1
            self.TrainStart = -1
        else:
            self.params_file = open(log_filename, 'w+b')
            self.current_epoch = 0
            self.save_epoch()

        print('Starting from epoch', self.current_epoch + 1)

    def __del__(self):
        self.on_train_end(None)

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        print('Saving epoch', self.current_epoch)
        super(TensorboardCallback, self).on_epoch_end(epoch=self.current_epoch, logs=logs)
        self.save_epoch()

    def on_train_end(self, epoch, logs=None):
        self.params_file.close()
        super(TensorboardCallback, self).on_train_end(epoch)

    def save_epoch(self):
        self.params_file.seek(0)
        self.params_file.write(pack('I', self.current_epoch))
        self.params_file.flush()

class CheckpointCallback(keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(CheckpointCallback, self).__init__(*args, **kwargs)
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # print('Saving checkpoint epoch', self.current_epoch)
        super(CheckpointCallback, self).on_epoch_end(epoch=self.current_epoch, logs=logs)
        self.current_epoch += 1