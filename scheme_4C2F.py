# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:46:08 2019

@author: Yung-Yu Tsai

evaluate quantization testing result of 4C2F CNN
An example of using inference scheme to arange analysis and save result.
"""

from simulator.inference.scheme import inference_scheme
from simulator.models.model_library import quantized_4C2F,quantized_4C2FBN
from simulator.metrics.topk_metrics import top2_acc,top5_acc

result_save_file_BN_fused='../test_result/cifar10_4C2FBN_fused.csv'
result_save_file_BN='../test_result/cifar10_4C2FBN.csv'
weight_name_BN='../cifar10_4C2FBN_weight.h5'
weight_name_BN_fused='../cifar10_4C2FBN_weight_fused_BN.h5'
batch_size=25

model_argument_BN=[{'nbits':16,'fbits':8,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
                  {'nbits':14,'fbits':7,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
                  {'nbits':12,'fbits':6,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
                  {'nbits':10,'fbits':5,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
                  {'nbits':8,'fbits':4,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'extrinsic'},
                  {'nbits':16,'fbits':8,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'},
                  {'nbits':14,'fbits':7,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'},
                  {'nbits':12,'fbits':6,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'},
                  {'nbits':10,'fbits':5,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'},
                  {'nbits':8,'fbits':4,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'},
                  {'nbits':16,'fbits':8,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'intrinsic'},
                  {'nbits':14,'fbits':7,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'intrinsic'},
                  {'nbits':12,'fbits':6,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'intrinsic'},
                  {'nbits':10,'fbits':5,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'intrinsic'},
                  {'nbits':8,'fbits':4,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'intrinsic'}]

compile_argument={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top2_acc]}

dataset_argument={'dataset':'cifar10'}

# 4C2FBN
inference_scheme(quantized_4C2FBN, model_argument_BN, compile_argument, dataset_argument, result_save_file_BN, weight_load=True, weight_name=weight_name_BN)

# 4C2FBN fused
inference_scheme(quantized_4C2F, model_argument_BN, compile_argument, dataset_argument, result_save_file_BN_fused, weight_load=True, weight_name=weight_name_BN_fused)


