# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:46:08 2019

@author: Yung-Yu Tsai

An example of using inference scheme to arange analysis and save result.
evaluate quantization testing result of LeNet-5
"""

from simulator.inference.scheme import inference_scheme
from simulator.models.model_library import quantized_lenet5
from simulator.metrics.topk_metrics import top2_acc

result_save_file='../test_result/mnist_lenet5_hybrid.csv'
weight_name='../mnist_lenet5_weight.h5'
batch_size=25

model_argument=[{'nbits':16,'fbits':8,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'},
               {'nbits':14,'fbits':7,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'},
               {'nbits':12,'fbits':6,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'},
               {'nbits':10,'fbits':5,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'},
               {'nbits':8,'fbits':4,'rounding_method':'nearest','batch_size':batch_size,'quant_mode':'hybrid'}]

compile_argument={'loss':'categorical_crossentropy','optimizer':'adam','metrics':['accuracy',top2_acc]}

dataset_argument={'dataset':'mnist'}


inference_scheme(quantized_lenet5, model_argument, compile_argument, dataset_argument, result_save_file, weight_load=True, weight_name=weight_name)


