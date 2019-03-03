# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:41:51 2019

@author: Yung-Yu Tsai

Plan for multiple inferece setting and write into file
"""

import keras, os, csv
from keras.utils import multi_gpu_model
from utils_tool.weight_conversion import convert_original_weight_layer_name
from utils_tool.dataset_setup import dataset_setup
from metrics.topk_metrics import *
import time

def inference_scheme(model_func, model_augment, compile_augment, dataset_augment, result_save_file, weight_load=False, weight_name=None, show_summary=False, multi_gpu=False, gpu_num=2):
    """Take scheme as input and run different setting of inference automaticly. Write the results into a csv file.

    # Arguments
        model_func: The callable function which returns a DNN model. (Keras funtional model API recommmanded).
        model_augment: List of Dictionarys. The augments for DNN model function.
        compile_augment: Dictionary. The augments for model compile augment.
        dataset_augment: Dictionary. The augments for dataset setup.
        result_save_file: String. The file and directory to the result csv file.
        weight_load: Bool. Need load weight proccess outside model_func or not.
        weight_name: String. The weight file to load. (if weight_load is True)
        multi_gpu: Bool. Using multi GPU inference or not.
        gpu_num: Integer. The number of GPUs in your system setup. (for multi_gpu = True only)

    # Returns
        ==================
    """
    if not callable(model_func):
        raise TypeError('The model_func augment must be a callable function which returns a Keras DNN model.')
        
        
    print('preparing dataset...')
    x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup( **dataset_augment)
    print('dataset ready')

    
    for scheme_num in range(len(model_augment)):
        print('Running inference scheme %d/%d'%(scheme_num,len(model_augment)))
        
        print('Building model...')
        
        t = time.time()

        model=model_func( **model_augment[scheme_num])
        
        if weight_load:
            weight_name=convert_original_weight_layer_name(weight_name)
            model.load_weights(weight_name)
        
        if show_summary:
            model.summary()

        
        if multi_gpu:
            t = time.time()-t
            print('model build time: %f s'%t)
            
            print('Building multi GPU model...')
            t = time.time()            
            parallel_model = multi_gpu_model(model, gpus=gpu_num)
            parallel_model.compile( **compile_augment)
            if show_summary:
                parallel_model.summary()
            t = time.time()-t
            print('multi GPU model build time: %f s'%t)            
        else:
            model.compile( **compile_augment)
            t = time.time()-t
            print('model build time: %f s'%t)
            
        
        t = time.time()
        print('evaluating...')
        
        if multi_gpu:
            if datagen is None:
                test_result = parallel_model.evaluate(x_test, y_test, verbose=1, batch_size=model_augment[scheme_num]['batch_size'])
            else:
                test_result = parallel_model.evaluate_generator(datagen, verbose=1)
        else:
            if datagen is None:
                test_result = model.evaluate(x_test, y_test, verbose=1, batch_size=model_augment[scheme_num]['batch_size'])
            else:
                test_result = model.evaluate_generator(datagen, verbose=1)
        
        t = time.time()-t
        print('evaluate done')
        print('\nruntime: %f s'%t)        
        print('\nTest loss:', test_result[0])
        for i in range(1,len(test_result)):
            print('Test metric %d :'%i, test_result[i])
            
        if scheme_num is 0:     
            with open(result_save_file, 'w', newline='') as csvfile:
                fieldnames=['loss']
                test_result_dict={'loss':test_result[0]}
                for i in range(1,len(test_result)):
                    fieldnames.append('metric %d'%i)
                    test_result_dict['metric %d'%i]=test_result[i]
                writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(test_result_dict)
        else:
            with open(result_save_file, 'a', newline='') as csvfile:
                    test_result_dict={'loss':test_result[0]}
                    for i in range(1,len(test_result)):
                        test_result_dict['metric %d'%i]=test_result[i]
                    writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(test_result_dict)
                    
        print('\n===============================================\n')


