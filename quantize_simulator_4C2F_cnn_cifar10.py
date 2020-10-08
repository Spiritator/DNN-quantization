# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:29:34 2018

@author: Yung-Yu Tsai

evaluate quantized testing result with custom Keras quantize layer of 4C2F CNN 
"""

# setup

import numpy as np
import tensorflow.keras.backend as K
import time

from simulator.models.model_library import quantized_4C2F
from simulator.utils_tool.weight_conversion import convert_original_weight_layer_name
from simulator.utils_tool.dataset_setup import dataset_setup
from simulator.utils_tool.confusion_matrix import show_confusion_matrix
from tensorflow.keras.losses import categorical_crossentropy
from simulator.metrics.topk_metrics import top2_acc
from simulator.metrics.FT_metrics import acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50
from simulator.approximation.estimate import comp_num_estimate
from simulator.inference.evaluate import evaluate_FT

#%%
# model setup

#weight_name='../cifar10_4C2F_weight.h5'
weight_name='../cifar10_4C2FBN_weight_fused_BN.h5'
batch_size=25


# model setup
# all arguments use the same quantize precision
#model=quantized_4C2F(nbits=8,fbits=4,rounding_method='nearest')
#model=quantized_4C2F(nbits=16,fbits=8,rounding_method='nearest',batch_size=batch_size,quant_mode='hybrid',overflow_mode=[True,False,True])
model=quantized_4C2F(nbits=10,fbits=6,rounding_method='nearest',batch_size=batch_size,quant_mode='hybrid')
# each argument uses different quantize precision. information list [input, weight, output]
#model=quantized_4C2F(nbits=[12,6,12],fbits=[6,3,6],rounding_method='nearest')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',top2_acc])
weight_name=convert_original_weight_layer_name(weight_name)
model.load_weights(weight_name)
print('orginal weight loaded')

#%%
#dataset setup

x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('cifar10')

#%%
# view test result

t = time.time()

prediction = model.predict(x_test, verbose=1,batch_size=batch_size)
test_result = evaluate_FT('lenet',prediction=prediction,test_label=y_test,loss_function=categorical_crossentropy,metrics=['accuracy',top2_acc,acc_loss,relative_acc,pred_miss,top2_pred_miss,conf_score_vary_10,conf_score_vary_50])

t = time.time()-t
print('\nruntime: %f s'%t)
for key in test_result.keys():
    print('Test %s\t:'%key, test_result[key])

computaion_esti=comp_num_estimate(model)
print('\nTotal # of computations:', computaion_esti['total_MAC'])
print('Total # of MAC bits:', computaion_esti['total_MAC_bits'])

#%%
# draw confusion matrix

print('\n')
#prediction = model.predict(x_test, verbose=1, batch_size=batch_size)
prediction = np.argmax(prediction, axis=1)

show_confusion_matrix(np.argmax(y_test, axis=1),prediction,class_indices,'Confusion Matrix',figsize=(8,6),normalize=False)

