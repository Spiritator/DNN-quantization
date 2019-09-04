# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:26:37 2019

@author: Yung-Yu Tsai

An example of distributed convolution layer
"""

# setup

import keras
import numpy as np
import keras.backend as K
import time

from simulator.utils_tool.weight_conversion import convert_original_weight_layer_name
from simulator.utils_tool.dataset_setup import dataset_setup
from simulator.utils_tool.confusion_matrix import show_confusion_matrix
from simulator.metrics.topk_metrics import top2_acc
from simulator.approximation.estimate import comp_num_estimate

from keras.models import Model
from keras.layers import Activation, Input, MaxPooling2D, Add
from simulator.layers.quantized_layers import QuantizedConv2D, QuantizedDense, QuantizedFlatten, QuantizedDistributedConv2D
from simulator.layers.quantized_ops import quantizer,build_layer_quantizer
from simulator.models.model_library import quantized_4C2F
from simulator.models.model_mods import exchange_distributed_conv

#%%
# model setup

weight_name='../mnist_lenet5_weight.h5'
batch_size=25

def quantized_lenet5(nbits=8, fbits=4, rounding_method='nearest', input_shape=(28,28,1), num_classes=10, batch_size=None, ifmap_fault_dict_list=None, ofmap_fault_dict_list=None, weight_fault_dict_list=None, quant_mode='hybrid', overflow_mode=False, stop_gradient=False,):
    
    print('\nBuilding model : Quantized Lenet 5')
    
    layer_quantizer=build_layer_quantizer(nbits,fbits,rounding_method,overflow_mode,stop_gradient)
    
    if ifmap_fault_dict_list is None:
        ifmap_fault_dict_list=[None for i in range(8)]
    else:
        print('Inject input fault')
    if ofmap_fault_dict_list is None:
        ofmap_fault_dict_list=[None for i in range(8)]
    else:
        print('Inject output fault')
    if weight_fault_dict_list is None:
        weight_fault_dict_list=[[None,None] for i in range(8)]
    else:
        print('Inject weight fault')
        
    print('Building Layer 0')
    input_shape = Input(shape=input_shape, batch_shape=(batch_size,)+input_shape)
    print('Building Layer 1')
    x = QuantizedConv2D(filters=16,
                        quantizers=layer_quantizer,
                        kernel_size=(5,5),
                        padding='same',
                        strides=(1, 1),                              
                        activation='relu',
                        ifmap_sa_fault_injection=ifmap_fault_dict_list[1],
                        ofmap_sa_fault_injection=ofmap_fault_dict_list[1],
                        weight_sa_fault_injection=weight_fault_dict_list[1],
                        quant_mode=quant_mode)(input_shape)
    print('Building Layer 2')
    x = MaxPooling2D(pool_size=(2,2))(x)
    print('Building Layer 3')
    x = QuantizedDistributedConv2D(filters=36,
                                   split_type=['channel','k_seq'],
                                   splits=[4,5],
                                   quantizers=layer_quantizer,
                                   kernel_size=(5,5),
                                   padding='same',
                                   strides=(1, 1),
                                   ifmap_sa_fault_injection=ifmap_fault_dict_list[3],
                                   ofmap_sa_fault_injection=ofmap_fault_dict_list[3],
                                   weight_sa_fault_injection=weight_fault_dict_list[3],
                                   quant_mode=quant_mode)(x)
    print('Building Layer 4')
    x = Add()(x)
    print('Building Layer 5')
    x = Activation('relu')(x)

    print('Building Layer 6')
    x = MaxPooling2D(pool_size=(2,2))(x)
    print('Building Layer 7')
    x = QuantizedFlatten()(x)
    print('Building Layer 8')
    x = QuantizedDense(128,
                       quantizers=layer_quantizer,
                       activation='relu',
                       ifmap_sa_fault_injection=ifmap_fault_dict_list[6],
                       ofmap_sa_fault_injection=ofmap_fault_dict_list[6],
                       weight_sa_fault_injection=weight_fault_dict_list[6],
                       quant_mode=quant_mode)(x)
    print('Building Layer 9')
    x = QuantizedDense(num_classes,
                       quantizers=layer_quantizer,
                       activation='softmax',
                       ifmap_sa_fault_injection=ifmap_fault_dict_list[7],
                       ofmap_sa_fault_injection=ofmap_fault_dict_list[7],
                       weight_sa_fault_injection=weight_fault_dict_list[7],
                       quant_mode=quant_mode)(x)

    model=Model(inputs=input_shape, outputs=x)

    return model


model=quantized_lenet5(nbits=8,fbits=3,rounding_method='nearest',batch_size=batch_size,quant_mode=None)

weight_name=convert_original_weight_layer_name(weight_name)
model.load_weights(weight_name)
print('orginal weight loaded')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])



#%%
#dataset setup

x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('mnist')

#%%
# view test result
t = time.time()

test_result = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)

t = time.time()-t
print('\nruntime: %f s'%t)
print('\nTest loss:', test_result[0])
print('Test top1 accuracy:', test_result[1])
print('Test top2 accuracy:', test_result[2])

#computaion_esti=comp_num_estimate(model)
#print('\nTotal # of computations:', computaion_esti['total_MAC'])
#print('Total # of MAC bits:', computaion_esti['total_MAC_bits'])

#%%
# draw confusion matrix

print('\n')
prediction = model.predict(x_test, verbose=1,batch_size=batch_size)
prediction = np.argmax(prediction, axis=1)

show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',normalize=False)

##%%
#K.clear_session()
#
##%%
#
#weight_name='../cifar10_4C2FBN_weight_fused_BN.h5'
#batch_size=25
#
#t = time.time()
#model=quantized_4C2F(nbits=12,
#                     fbits=6,
#                     rounding_method='nearest',
#                     batch_size=batch_size,
#                     quant_mode='hybrid',)
#
#model=exchange_distributed_conv(model,[2,6],[[11,11,10],2],fault_dict_conversion=True)
#
#t = time.time()-t
#
#print('\nModel build time: %f s'%t)
#
#print('Model compiling...')
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])
#print('Model compiled !')
#model.load_weights(weight_name)
#print('orginal weight loaded')
#
##%%
##dataset setup
#
#x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('cifar10')
#
##%%
## view test result
#
#t = time.time()
#
#test_result = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)
#
#t = time.time()-t
#  
#print('\nruntime: %f s'%t)
#print('\nTest loss:', test_result[0])
#print('Test top1 accuracy:', test_result[1])
#print('Test top2 accuracy:', test_result[2])
#
##%%
## draw confusion matrix
#
#print('\n')
#prediction = model.predict(x_test, verbose=1, batch_size=batch_size)
#prediction = np.argmax(prediction, axis=1)
#
#show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',figsize=(8,6),normalize=False)



