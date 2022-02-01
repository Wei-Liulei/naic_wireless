#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import utils
import fire
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from Model_define_tf import Encoder, Decoder, NMSE, get_custom_objects


class ModelTrain():
    def __init__(self,
                 data_path='/media/D/data_set/WirelessCommunication/train',
                 model_load_path='./Modelsave/nmse0537',
                 fit=False,
                 epoch=1,
                 model_save_path='./Modelsave',
                 ):
        self.data_path = data_path
        self.model_load_path = model_load_path
        self.fit = fit
        self.model_save_path = model_save_path
        self.pipeline()

    def pipeline(self):
        self._load_data()
        self._build_model()
        if self.model_load_path:
            print(f'load mode weight from {self.model_load_path}')
            self._load_model()
        if self.fit:
            self._fit()
        self.evaluate()
        # self.save_model()

    @utils.timmer
    def _load_data(self):
        self.x_train, self.x_test = utils.get_data(self.data_path)
        self.x_train = self.x_train[:100]

    def _build_model(self):
        feedback_bits = 512
        img_height = 126  # shape=N*126*128*2
        img_width = 128
        img_channels = 2
        # encoder model
        Encoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="encoder_input")
        Encoder_output = Encoder(Encoder_input, feedback_bits)
        self.encoder = keras.Model(inputs=Encoder_input, outputs=Encoder_output, name='encoder')
        # decoder model
        Decoder_input = keras.Input(shape=(feedback_bits,), name='decoder_input')
        Decoder_output = Decoder(Decoder_input, feedback_bits)
        self.decoder = keras.Model(inputs=Decoder_input, outputs=Decoder_output, name="decoder")

        # autoencoder model
        autoencoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="original_img")
        encoder_out = self.encoder(autoencoder_input)
        decoder_out = self.decoder(encoder_out)
        self.autoencoder = keras.Model(inputs=autoencoder_input, outputs=decoder_out, name='autoencoder')
        self.autoencoder.compile(optimizer='adam', loss='mse')

        # print(self.autoencoder.summary())
        # print(self.encoder.summary())
        print(self.decoder.summary())
        # keras.utils.plot_model(self.autoencoder, show_shapes=True)  # , rankdir="LR"
        # keras.utils.plot_model(self.decoder, show_shapes=True)  # , rankdir="LR"

    @utils.timmer
    def _load_model(self):
        self.encoder.load_weights(self.model_load_path + '/encoder.h5')
        self.decoder.load_weights(self.model_load_path + '/decoder.h5')

    @utils.timmer
    def _fit(self):
        self.autoencoder.fit(x=self.x_train, y=self.x_train,
                             batch_size=16,
                             epochs=1,
                             validation_split=0.1
                             )

    @utils.timmer
    def evaluate(self):
        y_test = self.autoencoder.predict(self.x_test)
        print('The mean NMSE is ' + np.str(NMSE(self.x_test, y_test)))


    @utils.timmer
    def save_model(self):
        self.encoder.save(self.model_save_path + '/encoder.h5')
        self.decoder.save(self.model_save_path + '/decoder.h5')


if __name__ == '__main__':
    t = ModelTrain()
    
    
    # %%test
    # keras.utils.plot_model(t.autoencoder.layers[1], show_shapes=True)  # , rankdir="LR"
    # fire.Fire(ModelTrain)
    print('OK!!!!!!!!!!!!')
    
