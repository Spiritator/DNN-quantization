# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 11:33:23 2018

@author: Yung-Yu Tsai

evaluate quantized testing result with custom Keras quantize layer 
"""

# setup

import keras
import numpy as np
import keras.backend as K
import time
import scipy


from models.model_library import quantized_lenet5
from utils_tool.weight_conversion import convert_original_weight_layer_name
from utils_tool.dataset_setup import dataset_setup
from utils_tool.confusion_matrix import show_confusion_matrix
from metrics.topk_metrics import top2_acc

#%%
# setting parameter

weight_name='../../mnist_lenet5_weight.h5'
model_word_length=8
model_factorial_bit=4
rounding_method='nearest'
batch_size=32
fault_rate=0.01

#%%
# fault generation
model=quantized_lenet5(nbits=model_word_length,fbits=model_factorial_bit,rounding_method=rounding_method)
weight_name=convert_original_weight_layer_name(weight_name)
model.load_weights(weight_name)

model_depth=len(model.layers)
model_ifmap_fault_dict_list=[None]
model_ofmap_fault_dict_list=[None]
model_weight_fault_dict_list=[None]
for layer_num in range(1,model_depth):
    layer_input_shape=model.layers[layer_num].input_shape
    layer_output_shape=model.layers[layer_num].output_shape
    layer_weight_shape=[weight_shape.shape for weight_shape in model.layers[layer_num].get_weights()]
    
    if len(layer_weight_shape)==0:
        model_ifmap_fault_dict_list.append(None)
        model_ofmap_fault_dict_list.append(None)
        model_weight_fault_dict_list.append(None)
        continue
    
    ifmap_fault_dict=dict()
    ofmap_fault_dict=dict()
    weight_fault_dict=[dict() for i in range(len(layer_weight_shape))]
    
    layer_ifmap_fault_num=np.floor(np.prod(layer_input_shape[1:]) * batch_size * model_word_length * fault_rate)
    layer_ofmap_fault_num=np.floor(np.prod(layer_output_shape[1:]) * batch_size * model_word_length * fault_rate)
    layer_weight_fault_num=[int(np.prod(shapes) * batch_size * model_word_length * fault_rate) for shapes in layer_weight_shape]
    
    
    # ifmap fault generation
    fault_count=0
    while fault_count<layer_ifmap_fault_num:
        coordinate=list()
        coordinate.append(np.random.randint(batch_size))
        for j in range(1,len(layer_input_shape)):
            coordinate.append(np.random.randint(layer_input_shape[j]))
        coordinate=tuple(coordinate)
        fault_bit=np.random.randint(model_word_length)
        
        if coordinate in ifmap_fault_dict.keys():
            if isinstance(ifmap_fault_dict[coordinate]['fault_bit'],list):
                if fault_bit in ifmap_fault_dict[coordinate]['fault_bit']:
                    continue
                else:
                    ifmap_fault_dict[coordinate]['fault_type'].append('flip')
                    ifmap_fault_dict[coordinate]['fault_bit'].append(fault_bit)
            else:
                if fault_bit == ifmap_fault_dict[coordinate]['fault_bit']:
                    continue
                else:
                    ifmap_fault_dict[coordinate]['fault_type']=[ifmap_fault_dict[coordinate]['fault_type'],'flip']
                    ifmap_fault_dict[coordinate]['fault_bit']=[ifmap_fault_dict[coordinate]['fault_bit'],fault_bit]
                fault_count += 1
        else:
            ifmap_fault_dict[coordinate]={'fault_type':'flip',
                                          'fault_bit' : fault_bit}
            fault_count += 1
    
    model_ifmap_fault_dict_list.append(ifmap_fault_dict)    
    
    
    # ifmap fault generation
    fault_count=0
    while fault_count<layer_ofmap_fault_num:
        coordinate=list()
        coordinate.append(np.random.randint(batch_size))
        for j in range(1,len(layer_output_shape)):
            coordinate.append(np.random.randint(layer_output_shape[j]))
        coordinate=tuple(coordinate)
        fault_bit=np.random.randint(model_word_length)
        
        if coordinate in ofmap_fault_dict.keys():
            if isinstance(ofmap_fault_dict[coordinate]['fault_bit'],list):
                if fault_bit in ofmap_fault_dict[coordinate]['fault_bit']:
                    continue
                else:
                    ofmap_fault_dict[coordinate]['fault_type'].append('flip')
                    ofmap_fault_dict[coordinate]['fault_bit'].append(fault_bit)
            else:
                if fault_bit == ofmap_fault_dict[coordinate]['fault_bit']:
                    continue
                else:
                    ofmap_fault_dict[coordinate]['fault_type']=[ofmap_fault_dict[coordinate]['fault_type'],'flip']
                    ofmap_fault_dict[coordinate]['fault_bit']=[ofmap_fault_dict[coordinate]['fault_bit'],fault_bit]
                fault_count += 1
        else:
            ofmap_fault_dict[coordinate]={'fault_type':'flip',
                                          'fault_bit' : fault_bit}
            fault_count += 1
    
    model_ofmap_fault_dict_list.append(ofmap_fault_dict)    
    
    
    # ifmap fault generation
    fault_count=0
    while fault_count<layer_ifmap_fault_num:
        coordinate=list()
        coordinate.append(np.random.randint(batch_size))
        for j in range(1,len(layer_input_shape)):
            coordinate.append(np.random.randint(layer_input_shape[j]))
        coordinate=tuple(coordinate)
        fault_bit=np.random.randint(model_word_length)
        
        if coordinate in ifmap_fault_dict.keys():
            if isinstance(ifmap_fault_dict[coordinate]['fault_bit'],list):
                if fault_bit in ifmap_fault_dict[coordinate]['fault_bit']:
                    continue
                else:
                    ifmap_fault_dict[coordinate]['fault_type'].append('flip')
                    ifmap_fault_dict[coordinate]['fault_bit'].append(fault_bit)
            else:
                if fault_bit == ifmap_fault_dict[coordinate]['fault_bit']:
                    continue
                else:
                    ifmap_fault_dict[coordinate]['fault_type']=[ifmap_fault_dict[coordinate]['fault_type'],'flip']
                    ifmap_fault_dict[coordinate]['fault_bit']=[ifmap_fault_dict[coordinate]['fault_bit'],fault_bit]
                fault_count += 1
        else:
            ifmap_fault_dict[coordinate]={'fault_type':'flip',
                                          'fault_bit' : fault_bit}
            fault_count += 1
    
    model_ifmap_fault_dict_list.append(ifmap_fault_dict)    


#%%
# model setup
model=quantized_lenet5(nbits=model_word_length,fbits=model_factorial_bit,rounding_method=rounding_method)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])
model.load_weights(weight_name)
print('orginal weight loaded')

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

#%%
# draw confusion matrix

print('\n')
prediction = model.predict(x_test, verbose=1, batch_size=batch_size)
prediction = np.argmax(prediction, axis=1)

show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',normalize=False)


