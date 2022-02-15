# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import fire
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import utils
import shutil
from utils import plot_x, plot_results, plot_3dheat
import tensorflow as tf
from utils import build_autoencoder, data_augmentation
from Model_define_tf import NMSE
from tensorflow.keras.callbacks import EarlyStopping
import random as python_random

random_seed=666
np.random.seed(random_seed)
python_random.seed(random_seed)
tf.random.set_seed(random_seed)

gpu_avai = len(tf.config.list_physical_devices('GPU'))>0


class ModelTrain():
    def __init__(self):
        self.model_path = './model'
        self._load_data()

    def pipeline(self,
                 # load_model='',  # 不加载权重
                 # load_model='./model/checkpoint/nmse_52.55.hdf5',  # 加载checkpoint中模型权重 *
                 # load_model='./model/temp/54.03',  # 加载temp中模型权重 **
                 load_model='./model/project56.71',  # 加载model根目录下模型权重  ***
                 epoch=0,
                 ):
        self.load_model = load_model

        self.epoch = epoch if gpu_avai else min(1, epoch)

        self._build_model()
        if self.load_model:
            print(f'load mode weight from {self.load_model}')
            self._load_model()
        self._fit()
        self.evaluate()
        self.save_model()

    @utils.timmer
    def _load_data(self):
        self.data_path = './data'
        self.x_train, self.x_test = utils.get_data(self.data_path)
        if not gpu_avai:
            self.x_train = self.x_train[:100]

    def _build_model(self):
        from tensorflow.keras.optimizers import Adam, RMSprop
        
        self.encoder, self.decoder, self.autoencoder = build_autoencoder()
        # self.autoencoder.compile(optimizer='rmsprop',
        self.autoencoder.compile(optimizer='adam',
        # self.autoencoder.compile(optimizer='rmsprop',
                                 loss='mse',
                                 metrics=[utils.nmse]  # ,utils.mae, utils.mape, 
                                 )
        
        print(self.autoencoder.summary())
        # print(self.encoder.summary())
        # print(self.decoder.summary())
        # keras.utils.plot_model(self.autoencoder, show_shapes=True)  # , rankdir="LR"

    @utils.timmer
    def _load_model(self):
        if self.load_model.split('/')[2]=='checkpoint':
            self.autoencoder.load_weights(self.load_model)            
        else:
            self.encoder.load_weights(self.load_model + '/encoder.h5')
            self.decoder.load_weights(self.load_model + '/decoder.h5')
    @utils.timmer
    def _fit(self):
        self.history = self.autoencoder.fit(x=self.x_train, y=self.x_train,
                                            batch_size=128,
                                            epochs=self.epoch,
                                            # validation_split=0.1,
                                            validation_data=(self.x_test, self.x_test),
                                            callbacks=[
                                                EarlyStopping(monitor='val_nmse', mode='max', patience=200, verbose=2),
                                                tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1),
                                                # 注意清理检查点目录
                                                tf.keras.callbacks.ModelCheckpoint(
                                                    os.path.join(self.model_path+'/checkpoint', "nmse_{val_nmse:.2f}.hdf5"),
                                                    save_weights_only=True,
                                                    mode='max',
                                                    verbose=1,
                                                    save_best_only='True',
                                                    monitor='val_nmse'
                                                            )
                                                      ],
                                            )

        # history = pd.DataFrame(self.history.history)
        # utils.plot_his(history, 'mae')
        # utils.plot_his(history, 'mape')

    @utils.timmer
    def evaluate(self):
        self.y_test = self.autoencoder.predict(self.x_test)
        self.NMSE = str(NMSE(self.x_test, self.y_test))
        print('The mean NMSE is ' + str(NMSE(self.x_test, self.y_test)))

    @utils.timmer
    def save_model(self):
        self.encoder.save(self.model_path + '/temp/' + 'project' + self.NMSE + '/encoder.h5')
        self.decoder.save(self.model_path + '/temp/' + 'project' + self.NMSE + '/decoder.h5')
        shutil.copy('Model_define_tf.py', self.model_path + '/temp/' + 'project' + self.NMSE + '/Model_define_tf.py')


# %%
if __name__ == '__main__':
    t = ModelTrain()
    fire.Fire(t.pipeline)

    # %% test or eval
    # from utils import plot_x, plot_results, plot_pq
    # # plot_3dheat(t.x_test, 10)
    # plot_pq(t.x_test, 50, 5, 10)    
    # plot_results(t.x_test, t.y_test, 5, 10)
    print('OK!!!!!!!!!!!!')
