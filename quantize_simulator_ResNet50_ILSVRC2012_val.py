# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:32:50 2018

@author: Yung-Yu Tsai

evaluate quantized testing result with custom Keras quantize layer of ResNet50
"""

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from simulator.models.resnet50 import QuantizedResNet50, QuantizedResNet50FusedBN, preprocess_input
from simulator.utils_tool.dataset_setup import dataset_setup
from simulator.utils_tool.confusion_matrix import show_confusion_matrix
from tensorflow.keras.losses import categorical_crossentropy
from simulator.metrics.topk_metrics import top5_acc
from simulator.metrics.FT_metrics import acc_loss, relative_acc, pred_miss, top5_pred_miss, conf_score_vary_10, conf_score_vary_50
from simulator.inference.evaluate import evaluate_FT

import time


# dimensions of our images.
img_width, img_height = 224, 224

class_number=1000
batch_size=40

set_size=2
validation_data_dir = '../../dataset/imagenet_val_imagedatagenerator_setsize_2'
nb_validation_samples = 50000

#%%
# model setup

# print('Building model...')
# t = time.time()
# model = QuantizedResNet50(weights='../resnet50_weights_tf_dim_ordering_tf_kernels.h5', 
#                           nbits=28,
#                           fbits=10, 
#                           BN_nbits=28, 
#                           BN_fbits=10,
#                           rounding_method='nearest',
#                           batch_size=batch_size,
#                           quant_mode='hybrid')

#model = QuantizedResNet50FusedBN(weights='../resnet50_weights_tf_dim_ordering_tf_kernels_fused_BN.h5', 
#                          nbits=20,
#                          fbits=10, 
#                          rounding_method='nearest',
#                          batch_size=batch_size,
#                          quant_mode='hybrid')

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top5_acc])
# model.summary()
# t = time.time()-t
# print('model build time: %f s'%t)

# multi GPU model
print('Building multi GPU model...')
t = time.time()
strategy = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])
with strategy.scope():
    parallel_model = QuantizedResNet50FusedBN(weights='../resnet50_weights_tf_dim_ordering_tf_kernels_fused_BN.h5', 
                              nbits=[16,16,16],
                              fbits=[8,12,8], 
                              rounding_method='nearest',
                              batch_size=batch_size,
                              quant_mode='hybrid')
    parallel_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top5_acc])
    parallel_model.summary()

t = time.time()-t
print('multi GPU model build time: %f s'%t)

#%%
#dataset setup

print('preparing dataset...')
x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup('ImageDataGenerator', img_rows = img_width, img_cols = img_height, batch_size = batch_size*strategy.num_replicas_in_sync, data_augmentation = False, data_dir = validation_data_dir, preprocessing_function = preprocess_input)
print('dataset ready')


#%%
# test

t = time.time()
print('evaluating...')

prediction = parallel_model.predict(datagen, verbose=1, steps=len(datagen))
#prediction = model.predict(datagen, verbose=1, steps=len(datagen))
test_result = evaluate_FT('resnet',prediction=prediction,test_label=to_categorical(datagen.classes,1000),loss_function=categorical_crossentropy,metrics=['accuracy',top5_acc,acc_loss,relative_acc,pred_miss,top5_pred_miss,conf_score_vary_10,conf_score_vary_50],fuseBN=True,setsize=set_size)

t = time.time()-t
print('\nruntime: %f s'%t)
for key in test_result.keys():
    print('Test %s\t:'%key, test_result[key])

#%%
# draw confusion matrix

#print('\n')
#prediction = model.predict(datagen, verbose=1, steps=len(datagen))
#prediction = np.argmax(prediction, axis=1)
#
#show_confusion_matrix(datagen.classes,prediction,datagen.class_indices.keys(),'Confusion Matrix',figsize=(10,8),normalize=False,big_matrix=True)

