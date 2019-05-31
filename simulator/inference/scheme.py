# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:41:51 2019

@author: Yung-Yu Tsai

Plan for multiple inferece setting and write into file
"""

import keras, os, csv
import keras.backend as K
from keras.utils import multi_gpu_model,to_categorical
from utils_tool.weight_conversion import convert_original_weight_layer_name
from utils_tool.dataset_setup import dataset_setup
from inference.evaluate import evaluate_FT
import time

def inference_scheme(model_func, model_augment, compile_augment, dataset_augment, result_save_file, weight_load=False, weight_name=None, FT_evaluate=False, FT_augment=None, show_summary=False, multi_gpu=False, gpu_num=2, name_tag=None):
    """Take scheme as input and run different setting of inference automaticly. Write the results into a csv file.

    # Arguments
        model_func: The callable function which returns a DNN model. (Keras funtional model API recommmanded).
        model_augment: List of Dictionarys. The augments for DNN model function.
        compile_augment: Dictionary. The augments for model compile augment.
        dataset_augment: Dictionary. The augments for dataset setup.
        result_save_file: String. The file and directory to the result csv file.
        weight_load: Bool. Need load weight proccess outside model_func or not.
        weight_name: String. The weight file to load. (if weight_load is True)
        FT_evaluate: Bool. Doing fault tolerance analysis or not.
        FT_augment: Dictionary. The augments for fault tolerance analysis. (if FT_evaluate is True)
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
        if name_tag is None:
            name_tag=' '
        print('Running inference scheme %s %d/%d'%(name_tag,scheme_num+1,len(model_augment)))
        
        print('Building model...')
        
        t = time.time()

        model=model_func( **model_augment[scheme_num])
        
        if weight_load:
            weight_name_convert=convert_original_weight_layer_name(weight_name)
            model.load_weights(weight_name_convert)
        
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
                if FT_evaluate:
                    prediction = parallel_model.predict(x_test, verbose=1,batch_size=model_augment[scheme_num]['batch_size'])
                    FT_augment['prediction']=prediction
                    FT_augment['test_label']=y_test
                    test_result = evaluate_FT( **FT_augment)
                else:
                    test_result = parallel_model.evaluate(x_test, y_test, verbose=1, batch_size=model_augment[scheme_num]['batch_size'])
            else:
                if FT_evaluate:
                    prediction = parallel_model.predict_generator(datagen, verbose=1,steps=len(datagen))
                    FT_augment['prediction']=prediction
                    FT_augment['test_label']=to_categorical(datagen.classes,len(class_indices))
                    test_result = evaluate_FT( **FT_augment)
                else:
                    test_result = parallel_model.evaluate_generator(datagen, verbose=1, steps=len(datagen))
        else:
            if datagen is None:
                if FT_evaluate:
                    prediction = model.predict(x_test, verbose=1,batch_size=model_augment[scheme_num]['batch_size'])
                    FT_augment['prediction']=prediction
                    FT_augment['test_label']=y_test
                    test_result = evaluate_FT( **FT_augment)
                else:
                    test_result = model.evaluate(x_test, y_test, verbose=1, batch_size=model_augment[scheme_num]['batch_size'])
            else:
                if FT_evaluate:
                    prediction = model.predict_generator(datagen, verbose=1,steps=len(datagen))
                    FT_augment['prediction']=prediction
                    FT_augment['test_label']=to_categorical(datagen.classes,len(class_indices))
                    test_result = evaluate_FT( **FT_augment)
                else:
                    test_result = model.evaluate_generator(datagen, verbose=1, steps=len(datagen))
        
        t = time.time()-t
        print('evaluate done')
        print('\nruntime: %f s'%t)        
        
        if FT_evaluate:
            for key in test_result.keys():
                print('Test %s\t:'%key, test_result[key])
        else:
            for i in range(len(test_result)):
                print('Test %s\t:'%model.metrics_names[i], test_result[i])
            
        if scheme_num == 0:     
            with open(result_save_file, 'w', newline='') as csvfile:
                fieldnames=list()
                test_result_dict=dict()
                if FT_evaluate:
                    for key in test_result.keys():
                        fieldnames.append(key)
                        test_result_dict[key]=test_result[key]
                else:
                    for i in range(len(test_result)):
                        fieldnames.append(model.metrics_names[i])
                        test_result_dict[model.metrics_names[i]]=test_result[i]
                writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(test_result_dict)
        else:
            with open(result_save_file, 'a', newline='') as csvfile:
                test_result_dict=dict()
                if FT_evaluate:
                    for key in test_result.keys():
                        test_result_dict[key]=test_result[key]
                else:
                    for i in range(len(test_result)):
                        test_result_dict[model.metrics_names[i]]=test_result[i]
                writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(test_result_dict)
                    
            
        K.clear_session()
                            
        print('\n===============================================\n')


