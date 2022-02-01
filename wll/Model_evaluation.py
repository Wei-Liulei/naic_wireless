import numpy as np
import scipy.io as scio
import tensorflow as tf
from Model_define_tf import get_custom_objects, NMSE


# %%
def Score(NMSE):
    score = (1 - NMSE) * 100
    return score

# Data loading
# data_load_address = './data'
data_path = '/media/D/data_set/WirelessCommunication/train'
mat = scio.loadmat(data_path+'/Htest.mat')
x_test = mat['H_test']
x_test = x_test.astype('float32')

# load model
encoder_address = './Modelsave/encoder.h5'
decoder_address = './Modelsave/decoder.h5'
_custom_objects = get_custom_objects()  # load keywords of Custom layers
model_encoder = tf.keras.models.load_model(encoder_address, custom_objects=_custom_objects)
model_decoder = tf.keras.models.load_model(decoder_address, custom_objects=_custom_objects)

# predict
encode_feature = model_encoder.predict(x_test)
print("feedbackbits length is ", np.shape(encode_feature)[-1])

# save encoder_output
# np.save('./Modelsave/encoder_output.npy', encode_feature)
# decode_input = np.load('./Modelsave/encoder_output.npy')

decode_input = encode_feature
# eval
y_test = model_decoder.predict(decode_input)
print('The NMSE is ' + np.str(NMSE(x_test, y_test)))

NMSE_test = NMSE(x_test, y_test)
scr = Score(NMSE_test)
if scr < 0:
    scr=0
else:
    scr=scr

result = 'score=', np.str(scr)
print(result)
