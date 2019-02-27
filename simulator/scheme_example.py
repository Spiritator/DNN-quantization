# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:46:08 2019

@author: Yung-Yu Tsai

An example of using inference scheme to arange analysis and save result.
"""

from inference.scheme import inference_scheme
#from models.model_library import quantized_lenet5
from models.resnet50 import QuantizedResNet50, preprocess_input
from metrics.topk_metrics import top2_acc,top5_acc

#result_save_file='../../test_result/mnist_lenet5_hybrid.csv'
#weight_name='../../mnist_lenet5_weight.h5'
#batch_size=25
#
#model_augment=[{'nbits':16,'fbits':8,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'},
#               {'nbits':14,'fbits':7,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'},
#               {'nbits':12,'fbits':6,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'},
#               {'nbits':10,'fbits':5,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'},
#               {'nbits':8,'fbits':4,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'}]
#
#compile_augment={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top2_acc]}
#
#dataset_augment={'dataset':'mnist'}
#
#
#inference_scheme(quantized_lenet5, model_augment, compile_augment, dataset_augment, result_save_file, weight_load=True, weight_name=weight_name)

#%%

result_save_file='../../test_result/imagenet_Resnet50_extrinsic.csv'
weight_name='../../resnet50_weights_tf_dim_ordering_tf_kernels.h5'
img_width, img_height = 224, 224
class_number=1000
batch_size=40
validation_data_dir = '../../../dataset/imagenet_val_imagedatagenerator_setsize_2'
nb_validation_samples = 50000

model_augment=[{'weights':weight_name,'nbits':28,'fbits':14,'BN_nbits':28,'BN_fbits':14,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
               {'weights':weight_name,'nbits':26,'fbits':13,'BN_nbits':26,'BN_fbits':13,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
               {'weights':weight_name,'nbits':24,'fbits':12,'BN_nbits':24,'BN_fbits':12,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
               {'weights':weight_name,'nbits':22,'fbits':11,'BN_nbits':22,'BN_fbits':11,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
               {'weights':weight_name,'nbits':20,'fbits':10,'BN_nbits':20,'BN_fbits':10,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
               {'weights':weight_name,'nbits':18,'fbits':9,'BN_nbits':18,'BN_fbits':9,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
               {'weights':weight_name,'nbits':16,'fbits':8,'BN_nbits':16,'BN_fbits':8,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
               {'weights':weight_name,'nbits':14,'fbits':7,'BN_nbits':14,'BN_fbits':7,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
               {'weights':weight_name,'nbits':12,'fbits':6,'BN_nbits':12,'BN_fbits':6,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
               {'weights':weight_name,'nbits':10,'fbits':5,'BN_nbits':10,'BN_fbits':5,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
               {'weights':weight_name,'nbits':8,'fbits':4,'BN_nbits':8,'BN_fbits':4,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
               ]

compile_augment={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top5_acc]}

dataset_augment={'dataset':'ImageDataGenerator','img_rows':img_width,'img_cols':img_height,'batch_size':batch_size,'data_augmentation':False,'data_dir':validation_data_dir,'preprocessing_function':preprocess_input}


inference_scheme(QuantizedResNet50, model_augment, compile_augment, dataset_augment, result_save_file, multi_gpu=True, gpu_num=2)

