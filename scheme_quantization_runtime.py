# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:11:14 2019

@author: Yung-Yu Tsai

Quantization mode runtime analysis
"""

from simulator.inference.scheme import inference_scheme
from simulator.models.model_library import quantized_lenet5,quantized_4C2F
from simulator.models.mobilenet import QuantizedMobileNetV1FusedBN
from simulator.models.mobilenet import preprocess_input as preprocess_input_mob
from simulator.models.resnet50 import QuantizedResNet50FusedBN
from simulator.models.resnet50 import preprocess_input as preprocess_input_res

#%%
# Lenet 5
result_save_file='../test_result/mnist_lenet5_runtime.csv'
weight_name='../mnist_lenet5_weight.h5'
batch_size=25
test_rounds=200
model_word_length=8
model_fractional_bit=3

model_config={'nbits':model_word_length,
              'fbits':model_fractional_bit,
              'rounding_method':'nearest',
              'batch_size':batch_size,
              'quant_mode':None}

model_augment=[model_config for i in range(test_rounds)]

compile_augment={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy']}

dataset_augment={'dataset':'mnist'}

inference_scheme(quantized_lenet5, 
                 model_augment, 
                 compile_augment, 
                 dataset_augment, 
                 result_save_file, 
                 weight_load=True, 
                 weight_name=weight_name, 
                 save_runtime=True)

#%%
# 4C2F

result_save_file_BN_fused='../test_result/cifar10_4C2FBN_fused_runtime.csv'
weight_name_BN_fused='../cifar10_4C2FBN_weight_fused_BN.h5'
batch_size=25
test_rounds=200
model_word_length=10
model_fractional_bit=6

model_config={'nbits':model_word_length,
              'fbits':model_fractional_bit,
              'rounding_method':'nearest',
              'batch_size':batch_size,
              'quant_mode':None}

model_augment_BN=[model_config for i in range(test_rounds)]

compile_augment={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy']}

dataset_augment={'dataset':'cifar10'}

inference_scheme(quantized_4C2F, 
                 model_augment_BN, 
                 compile_augment, 
                 dataset_augment, 
                 result_save_file_BN_fused, 
                 weight_load=True, 
                 weight_name=weight_name_BN_fused, 
                 save_runtime=True)

#%%
# MobileNet

result_save_file='../test_result/imagenet_MobileNet_fused_BN_runtime.csv'
weight_name='../mobilenet_1_0_224_tf_fused_BN.h5'
img_width, img_height = 224, 224
class_number=1000
batch_size=40
validation_data_dir = '../../dataset/imagenet_val_imagedatagenerator_setsize_2'
nb_validation_samples = 50000
model_word_length=16
model_fractional_bit=9
test_rounds=200

model_config={'weights':weight_name,
              'nbits':model_word_length,
              'fbits':model_fractional_bit,
              'rounding_method':'nearest',
              'batch_size':batch_size,
              'quant_mode':None}

model_augment_BN=[model_config for i in range(test_rounds)]

compile_augment={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy']}

dataset_augment={'dataset':'ImageDataGenerator',
                 'img_rows':img_width,
                 'img_cols':img_height,
                 'batch_size':batch_size,
                 'data_augmentation':False,
                 'data_dir':validation_data_dir,
                 'preprocessing_function':preprocess_input_mob}

inference_scheme(QuantizedMobileNetV1FusedBN, 
                 model_augment, 
                 compile_augment, 
                 dataset_augment, 
                 result_save_file, 
                 save_runtime=True)


#%%
# ResNet50

result_save_file='../test_result/imagenet_Resnet50_runtime.csv'
weight_name='../resnet50_weights_tf_dim_ordering_tf_kernels_fused_BN.h5'
img_width, img_height = 224, 224
class_number=1000
batch_size=40
validation_data_dir = '../../dataset/imagenet_val_imagedatagenerator_setsize_2'
nb_validation_samples = 50000
model_word_length=[16,16,16]
model_fractional_bit=[8,12,8]
test_rounds=200

model_config={'weights':weight_name,
              'nbits':model_word_length,
              'fbits':model_fractional_bit,
              'rounding_method':'nearest',
              'batch_size':batch_size,
              'quant_mode':None}

model_augment_BN=[model_config for i in range(test_rounds)]

compile_augment={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy']}

dataset_augment={'dataset':'ImageDataGenerator',
                 'img_rows':img_width,
                 'img_cols':img_height,
                 'batch_size':batch_size,
                 'data_augmentation':False,
                 'data_dir':validation_data_dir,
                 'preprocessing_function':preprocess_input_res}


inference_scheme(QuantizedResNet50FusedBN, 
                 model_augment, 
                 compile_augment, 
                 dataset_augment, 
                 result_save_file,
                 save_runtime=True)


