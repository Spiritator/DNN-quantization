# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:41:51 2019

@author: Yung-Yu Tsai

Plan for multiple inferece setting and write into file
"""

import keras, os, csv
import keras.backend as K
from keras.utils import multi_gpu_model,to_categorical
from ..utils_tool.weight_conversion import convert_original_weight_layer_name
from ..utils_tool.dataset_setup import dataset_setup
from .evaluate import evaluate_FT
from ..testing.fault_list import generate_model_stuck_fault
import time
import numpy as np

def inference_scheme(model_func, 
                     model_augment, 
                     compile_augment, 
                     dataset_augment, 
                     result_save_file, 
                     weight_load=False, 
                     weight_name=None, 
                     save_runtime=False,
                     fault_gen=False, 
                     fault_param=None,
                     FT_evaluate=False, 
                     FT_augment=None, 
                     show_summary=False, 
                     multi_gpu=False, 
                     gpu_num=2, 
                     name_tag=None):
    """Take scheme as input and run different setting of inference automaticly. Write the results into a csv file.

    # Arguments
        model_func: The callable function which returns a DNN model. (Keras funtional model API recommmanded).
        model_augment: List of Dictionarys. The augments for DNN model function.
        compile_augment: Dictionary. The augments for model compile augment.
        dataset_augment: Dictionary. The augments for dataset setup.
        result_save_file: String. The file and directory to the result csv file.
        weight_load: Bool. Need load weight proccess outside model_func or not.
        weight_name: String. The weight file to load. (if weight_load is True)
        save_runtime: Bool. Save runtime in result file or not.
        fault_gen: Bool. If True, generate fault dict list inside inference_scheme (slower, consume less memory). 
                         If False, using the fault dict list from model_augment (faster, consume huge memory).
        fault_param: Dictionay. The augment for fault generation function.
        FT_evaluate: Bool. Doing fault tolerance analysis or not.
        FT_augment: Dictionary. The augments for fault tolerance analysis. (if FT_evaluate is True)
        multi_gpu: Bool. Using multi GPU inference or not.
        gpu_num: Integer. The number of GPUs in your system setup. (for multi_gpu = True only)
        name_tag: String. The messege to show in terminal represent current simulation

    # Returns
        ==================
    """
    if not callable(model_func):
        raise TypeError('The model_func augment must be a callable function which returns a Keras DNN model.')
        
        
    print('preparing dataset...')
    x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup( **dataset_augment)
    if datagen is not None:
        datagen_y_test=to_categorical(datagen.classes,datagen.num_classes)
    
    print('dataset ready')
        
    
    for scheme_num in range(len(model_augment)):
        if name_tag is None:
            name_tag=' '
        print('Running inference scheme %s %d/%d'%(name_tag,scheme_num+1,len(model_augment)))
        
        print('Building model...')
        
            
        if datagen is not None:
            x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup( **dataset_augment)
        
        t = time.time()
        
        modelaug_tmp=model_augment[scheme_num]
        
        if fault_gen:
            model_ifmap_fdl,model_ofmap_fdl,model_weight_fdl=generate_model_stuck_fault( **fault_param)
            modelaug_tmp['ifmap_fault_dict_list']=model_ifmap_fdl
            modelaug_tmp['ofmap_fault_dict_list']=model_ofmap_fdl
            modelaug_tmp['weight_fault_dict_list']=model_weight_fdl

        model=model_func( **modelaug_tmp)
        
        if weight_load:
            weight_name_convert=convert_original_weight_layer_name(weight_name)
            model.load_weights(weight_name_convert)
        
        if show_summary:
            model.summary()

        
        if multi_gpu:
            t = time.time()-t
            print('model build time: %f s'%t)
            
            print('Building multi GPU model...',end=' ')
            t = time.time()            
            parallel_model = multi_gpu_model(model, gpus=gpu_num)
            parallel_model.compile( **compile_augment)
            if show_summary:
                parallel_model.summary()
            t = time.time()-t
            print('/rmulti GPU model build time: %f s'%t)            
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
                    FT_augment['test_label']=datagen_y_test
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
                    FT_augment['test_label']=datagen_y_test
                    test_result = evaluate_FT( **FT_augment)
                else:
                    test_result = model.evaluate_generator(datagen, verbose=1, steps=len(datagen))
        
        t = time.time()-t
        #print('evaluate done')
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
                    if save_runtime:
                        fieldnames.append('runtime')
                        test_result_dict['runtime']=t
                else:
                    for i in range(len(test_result)):
                        fieldnames.append(model.metrics_names[i])
                        test_result_dict[model.metrics_names[i]]=test_result[i]
                    if save_runtime:
                        fieldnames.append('runtime')
                        test_result_dict['runtime']=t
                writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(test_result_dict)
        else:
            with open(result_save_file, 'a', newline='') as csvfile:
                test_result_dict=dict()
                if FT_evaluate:
                    for key in test_result.keys():
                        test_result_dict[key]=test_result[key]
                    if save_runtime:
                        test_result_dict['runtime']=t
                else:
                    for i in range(len(test_result)):
                        test_result_dict[model.metrics_names[i]]=test_result[i]
                    if save_runtime:
                        test_result_dict['runtime']=t
                writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(test_result_dict)
                    
            
        K.clear_session()
        del model
                            
        print('\n===============================================\n')


def gen_test_round_list(num_of_bit,upper_bound,lower_bound,left_bound=-3,right_bound=0):
    """Genrate test round list with number decade exponentially

    # Arguments
        num_of_bit_bit: Integer. The total amount of bits in the unit for fault injection.
        upper_bound: Integer. The maximum number for test rounds.
        lower_bound: Integer. The minimun number for test rounds.
        left_bound: Integer. The number for line y=exp(x) left bound. For generate decade number.
        right_bound: Integer. The number for line y=exp(x) right bound. For generate decade number.
        
    # Returns
        The fault rate list and test round list.
        (fault rate list is float, test round list is integer)
    """
    fault_rate_list=list()

    def append_frl(num_inv,fr):
        if fr>num_inv:
            for i in [1,2,5]:
                fr_tmp=fr/i
                if fr_tmp>num_inv:
                    fault_rate_list.append(fr_tmp)
                    
            append_frl(num_inv,fr/10)
            
    append_frl(1/num_of_bit,0.1)
    fault_rate_list.reverse()
    
    test_rounds_lists=np.linspace(left_bound,right_bound,num=len(fault_rate_list))
    test_rounds_lists=-np.exp(test_rounds_lists)
    scaling_factor=(upper_bound-lower_bound)/(np.max(test_rounds_lists)-np.min(test_rounds_lists))
    test_rounds_lists=(test_rounds_lists-np.min(test_rounds_lists))*scaling_factor+lower_bound
    test_rounds_lists=test_rounds_lists.astype(int)
    return fault_rate_list,test_rounds_lists
                
        

