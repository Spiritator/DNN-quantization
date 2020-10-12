# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:32:50 2018

@author: Yung-Yu Tsai

evaluate fault injection testing result of MobileNetV1
"""

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from simulator.models.mobilenet import QuantizedMobileNetV1FusedBN
from simulator.utils_tool.dataset_setup import dataset_setup
from simulator.utils_tool.confusion_matrix import show_confusion_matrix
from simulator.metrics.topk_metrics import top5_acc
import time
from simulator.fault.fault_list import generate_model_stuck_fault
from simulator.fault.fault_core import generate_model_modulator
from tensorflow.keras.losses import categorical_crossentropy
from simulator.metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top5_pred_miss, conf_score_vary_10, conf_score_vary_50
from simulator.inference.evaluate import evaluate_FT
from simulator.models.model_mods import make_ref_model

# dimensions of our images.
img_width, img_height = 224, 224

set_size=2
class_number=1000
batch_size=20
model_word_length=16
model_fractional_bit=9
rounding_method='nearest'
fault_rate=1e-8
if set_size in [50,'full',None]:
    validation_data_dir = '../../dataset/imagenet_val_imagedatagenerator'
else:
    validation_data_dir = '../../dataset/imagenet_val_imagedatagenerator_setsize_%d'%set_size


#%% fault generation

# model for get configuration
ref_model=make_ref_model(QuantizedMobileNetV1FusedBN(weights='../mobilenet_1_0_224_tf_fused_BN.h5', 
                                                     nbits=model_word_length,
                                                     fbits=model_fractional_bit, 
                                                     rounding_method=rounding_method,
                                                     batch_size=batch_size,
                                                     quant_mode=None,
                                                     verbose=False))

model_ifmap_fault_dict_list, model_ofmap_fault_dict_list, model_weight_fault_dict_list\
=generate_model_stuck_fault(ref_model,
                            fault_rate,
                            batch_size,
                            model_word_length,
                            param_filter=[True,True,True],
                            fast_gen=True,
                            return_modulator=True,
                            bit_loc_distribution='uniform',
                            bit_loc_pois_lam=None,
                            fault_type='flip')

#model_ifmap_fault_dict_list, model_ofmap_fault_dict_list, model_weight_fault_dict_list\
#=generate_model_modulator(model,
#                          model_word_length,
#                          model_fractional_bit,
#                          model_ifmap_fault_dict_list, 
#                          model_ofmap_fault_dict_list, 
#                          model_weight_fault_dict_list,
#                          fast_gen=True)


#%% model setup

# print('Building model...')
# t = time.time()
# model = QuantizedMobileNetV1FusedBN(weights='../mobilenet_1_0_224_tf_fused_BN.h5', 
#                                     nbits=model_word_length,
#                                     fbits=model_fractional_bit, 
#                                     rounding_method=rounding_method,
#                                     batch_size=batch_size,
#                                     quant_mode='hybrid',
#                                     ifmap_fault_dict_list=model_ifmap_fault_dict_list,
#                                     ofmap_fault_dict_list=model_ofmap_fault_dict_list,
#                                     weight_fault_dict_list=model_weight_fault_dict_list)
# #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top5_acc])
# t = time.time()-t
# model.summary()
# print('model build time: %f s'%t)

# multi GPU model
print('Building multi GPU model...')
t = time.time()
strategy = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])
with strategy.scope():
    parallel_model = QuantizedMobileNetV1FusedBN(weights='../mobilenet_1_0_224_tf_fused_BN.h5', 
                                                 nbits=16,
                                                 fbits=8, 
                                                 rounding_method='nearest',
                                                 batch_size=batch_size,
                                                 quant_mode='hybrid')

    parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top5_acc])
    parallel_model.summary()

t = time.time()-t

print('multi GPU model build time: %f s'%t)

#%% dataset setup

print('preparing dataset...')
x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('ImageDataGenerator', img_rows = img_width, img_cols = img_height, batch_size = batch_size*strategy.num_replicas_in_sync, data_augmentation = False, data_dir = validation_data_dir)
print('dataset ready')


#%% test

t = time.time()
print('evaluating...')

prediction = parallel_model.predict(datagen, verbose=1, steps=len(datagen))
#prediction = model.predict(datagen, verbose=1, steps=len(datagen))
test_result = evaluate_FT('mobilenet',prediction=prediction,test_label=to_categorical(datagen.classes,1000),loss_function=categorical_crossentropy,metrics=['accuracy',top5_acc,acc_loss,relative_acc,pred_miss,top5_pred_miss,conf_score_vary_10,conf_score_vary_50],fuseBN=True,setsize=set_size)

t = time.time()-t
print('\nruntime: %f s'%t)
for key in test_result.keys():
    print('Test %s\t:'%key, test_result[key])

#%% draw confusion matrix

#print('\n')
#prediction = model.predict(datagen, verbose=1, steps=len(datagen))
#prediction = np.argmax(prediction, axis=1)
#
#show_confusion_matrix(datagen.classes,prediction,datagen.class_indices.keys(),'Confusion Matrix',figsize=(10,8),normalize=False,big_matrix=True)

