'''
reference: https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow
all the credit refer to BertMoons on QuantizedNeuralNetworks-Keras-Tensorflow

@author: Yung-Yu Tsai

'''

from keras.models import Sequential, Model
from keras import regularizers
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, Dropout
from keras.regularizers import l2
from keras import metrics
#from metrics import top2_acc
import numpy as np

from layers.quantized_layers import QuantizedConv2D,QuantizedDense
from layers.quantized_ops import quantized_relu as quantize_op


def qunatized_lenet5(nbits=8, input_shape=(28,28,1), num_classes=10):
    
    model = Sequential()
    model.add(QuantizedConv2D(filters=16,
                              H=1,
                              nb=nbits,
                              kernel_size=(5,5),
                              padding='same',
                              strides=(1, 1),
                              input_shape=input_shape,
                              activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(QuantizedConv2D(filters=36,
                              H=1,
                              nb=nbits,
                              kernel_size=(5,5),
                              padding='same',
                              strides=(1, 1),
                              activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(QuantizedDense(128,
                             H=1,
                             nb=nbits,
                             activation='relu'))
    model.add(QuantizedDense(num_classes,
                             H=1,
                             nb=nbits,
                             activation='softmax'))
#    model.summary()
#    model.compile(loss='categorical_crossentropy',
#    			  optimizer='adam',metrics=['accuracy'])

    model.summary()

    return model

def quantized_4C2F(nbits=8, input_shape=(32,32,3), num_classes=10):
    model = Sequential()
    model.add(QuantizedConv2D(filters=32,
                              H=1,
                              nb=nbits,
                              kernel_size=(3, 3),
                              padding='same',
                              strides=(1, 1),
                              input_shape=input_shape,
                              activation='relu'))
    model.add(QuantizedConv2D(filters=32,
                              H=1,
                              nb=nbits,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(QuantizedConv2D(filters=64,
                              H=1,
                              nb=nbits,
                              kernel_size=(3, 3),
                              padding='same',
                              strides=(1, 1),
                              activation='relu'))
    model.add(QuantizedConv2D(filters=64,
                              H=1,
                              nb=nbits,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(QuantizedDense(512,
                             H=1,
                             nb=nbits,
                             activation='relu'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(QuantizedDense(num_classes,
                             H=1,
                             nb=nbits,
                             activation='softmax'))
    
#    model.compile(loss='categorical_crossentropy',
#                  optimizer='adam',
#                  metrics=['accuracy', top2_acc])
    
    model.summary()

    return model

def load_weights(model, weight_reader):
    weight_reader.reset()

    for i in range(len(model.layers)):
        if 'conv' in model.layers[i].name:
            if 'batch' in model.layers[i + 1].name:
                norm_layer = model.layers[i + 1]
                size = np.prod(norm_layer.get_weights()[0].shape)

                beta = weight_reader.read_bytes(size)
                gamma = weight_reader.read_bytes(size)
                mean = weight_reader.read_bytes(size)
                var = weight_reader.read_bytes(size)

                weights = norm_layer.set_weights([gamma, beta, mean, var])

            conv_layer = model.layers[i]
            if len(conv_layer.get_weights()) > 1:
                bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel, bias])
            else:
                kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                kernel = kernel.transpose([2, 3, 1, 0])
                conv_layer.set_weights([kernel])
    return model
