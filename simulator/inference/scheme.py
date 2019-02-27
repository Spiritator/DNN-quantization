# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:41:51 2019

@author: Yung-Yu Tsai

Plan for multiple inferece setting and write into file
"""

import keras
from keras.utils import multi_gpu_model
from utils_tool.dataset_setup import dataset_setup
from metrics.topk_metrics import top2_acc,top3_acc,top5_acc
import time

def inference_scheme(model_func, model_augment, compile_augment, dataset_augment, multi_gpu=False, gpu_num=2):
    """Take scheme as input and run different setting of inference automaticly. Write the results into a csv file.

    # Arguments
        model_func: The callable function which returns a DNN model. (Keras funtional model API recommmanded).
        model_augment: List of Dictionarys. The augments for DNN model function.
        compile_augment: List of Dictionarys. The augments for model compile augment.
        dataset_augment: List of Dictionarys. The augments for dataset setup.
        multi_gpu: Using multi GPU inference or not.
        gpu_num: Integer. The number of GPUs in your system setup. (for multi_gpu = True only)

    # Returns
        ==================
    """
    if not callable(model_func):
        raise TypeError('The model_func augment must be a callable function which returns a Keras DNN model.')
        
    print('preparing dataset...')
    x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup( **dataset_augment)
    print('dataset ready')

    
    for scheme_num in len(model_augment):
        print('Running inference scheme %d/%d'%(scheme_num,len(model_augment)))
        
        print('Building model...')
        
        t = time.time()

        model=model_func( **model_augment[scheme_num])
        
        model.summary()

        
        if multi_gpu:
            model.summary()
            t = time.time()-t
            print('model build time: %f s'%t)
            
            print('Building multi GPU model...')
            t = time.time()            
            parallel_model = multi_gpu_model(model, gpus=gpu_num)
            parallel_model.compile( **compile_augment)
            parallel_model.summary()
            t = time.time()-t
            print('multi GPU model build time: %f s'%t)            
        else:
            model.compile( **compile_augment)
            model.summary()
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
        

