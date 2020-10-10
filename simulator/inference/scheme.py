# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:41:51 2019

@author: Yung-Yu Tsai

Plan for multiple inferece setting and write into file
"""

import os, csv
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical #TODO new multi gpu task
from ..utils_tool.weight_conversion import convert_original_weight_layer_name
from ..utils_tool.dataset_setup import dataset_setup
from .evaluate import evaluate_FT
from ..fault.fault_list import generate_model_stuck_fault
import time
import numpy as np

def inference_scheme(model_func, 
                     model_argument, 
                     compile_argument, 
                     dataset_argument, 
                     result_save_file, 
                     append_save_file=False,
                     weight_load_name=None, 
                     save_runtime=False,
                     fault_gen_param=None,
                     FT_evaluate_argument=None, 
                     multi_gpu_num=None, 
                     name_tag=None,
                     save_file_add_on=None,
                     verbose=7):
    """Take scheme as input and run different setting of inference automaticly. Write the results into a csv file.

    Arguments
    ---------
    model_func: The callable function which returns a DNN model. 
        (Keras funtional model API recommmanded).
    model_argument: List of Dictionarys. 
        The arguments for DNN model function.
    compile_argument: Dictionary. 
        The arguments for model compile argument.
    dataset_argument: Dictionary. 
        The arguments for dataset setup.
    result_save_file: String. 
        The file and directory to the result csv file.
    append_save_file: Bool. 
        Append the save file no matter what.
    weight_load_name: String. Default is None.
        | The weight file to load. (if weight_load is not None)
        | If None, don't need to load weight proccess outside model_func.
    save_runtime: Bool. 
        Save runtime in result file or not.
    fault_gen_param: Dictionay. Default is None.
        | If is dtype dictionary (fault generation parameter), generate fault dict list inside inference_scheme (slower, consume less memory). 
        | If None, using the fault dict list from model_argument (faster, consume huge memory).
    fault_param: Dictionay. 
        The argument for fault generation function.
    FT_evaluate_argument: Dictionary. Default is None.
        | The arguments for fault tolerance analysis. (if FT_evaluate is True) Doing fault tolerance analysis.
        | If None, using model.evaluate for only have loss, accuracy and top-k accuracy.
    multi_gpu_num: Integer or List of String. 
        | If None, using CPU or single GPU inference.
        | If Integer > 1, the number of GPUs use in the inference scheme. 
        | If List of String, the specific device name wanted to be used in this inference scheme.
    name_tag: String. 
        The messege to show in terminal represent current simulation
    save_file_add_on: Dictionary.
        The add on information wanted to be saved to output result csv file. These add on information
        may not generate during inference. Therefore, they need to given beforehand. The item data list
        should be the same as inference rounds.
        
        format:
            | {'item name 1':[item data list], 
            |  'item name 2':[item data list], 
            |  ...}
            
    verbose: Integer. 
        | The verbosity of inference run printing information max 7 (print all info), min 0 (print nothing).
        | The description below shows the minimum verbosity for info to print.
        | Dataset Infos: dataset name (6), prepare and ready (6), data shape/number (7).
        | Scheme Run Index (1)
        | Model Building: building start (7), layer progress (5), build time (3).
        | Show the model compile summary (8)
        | Evaluation: evaluating start (6), inference steps (4), runtime (3).
        | Fault Tolerance Metrics (2)
        | Scheme Run Seperation Line (2)
        
    Returns
    -------
    None
        Running the inference scheme.
    """
    if not callable(model_func):
        raise TypeError('The model_func argument must be a callable function which returns a Keras DNN model.')
        
    if verbose>5:
        print('preparing dataset...')
    if multi_gpu_num is not None:
        num_gpus=len(tf.config.list_physical_devices('GPU'))
        if isinstance(multi_gpu_num,int):
            if multi_gpu_num<2:
                raise ValueError('Since you are using multi GPU why your number of GPU are less than 2?')
            if multi_gpu_num>num_gpus:
                raise ValueError('System have %d GPUs, but require %d GPUs.'%(num_gpus,multi_gpu_num))
            gpu_device=['/gpu:%d'%i for i in range(multi_gpu_num)]
            dataset_argument['batch_size']*=multi_gpu_num
        elif isinstance(multi_gpu_num,list):
            if len(multi_gpu_num)>num_gpus:
                raise ValueError('System have %d GPUs, but require %d GPUs.'%(num_gpus,len(multi_gpu_num)))
            gpu_device=multi_gpu_num
            dataset_argument['batch_size']*=len(multi_gpu_num)
        else:
            raise TypeError('multi_gpu_num must be either number of GPUs (Integer) or the List of GPU device names.')
            
    x_train, x_test, y_train, y_test, class_indices, datagen, input_shape = dataset_setup(verbose=verbose-5, **dataset_argument)
    if datagen is not None:
        y_test=to_categorical(datagen.classes,datagen.num_classes)
    if verbose>5:
        print('dataset ready')
        
    n_scheme=len(model_argument)
    for scheme_num in range(n_scheme):
        if name_tag is None:
            name_tag=' '
        if n_scheme>1 and verbose>0:
            print('Running inference scheme %s %d/%d'%(name_tag,scheme_num+1,n_scheme))
        
        modelaug_tmp=model_argument[scheme_num]
        
        if fault_gen_param is not None:
            model_ifmap_fdl,model_ofmap_fdl,model_weight_fdl=generate_model_stuck_fault( **fault_gen_param)
            modelaug_tmp['ifmap_fault_dict_list']=model_ifmap_fdl
            modelaug_tmp['ofmap_fault_dict_list']=model_ofmap_fdl
            modelaug_tmp['weight_fault_dict_list']=model_weight_fdl

        if multi_gpu_num is None:
            if verbose>6:
                print('Building model...')
            model=model_func(verbose=verbose>4, **modelaug_tmp)
            t = time.time()
            
            if weight_load_name is not None:
                model.load_weights(weight_load_name)
            
            model.compile( **compile_argument)
            
            if verbose>7:
                model.summary()
            t = time.time()-t
            if verbose>2:
                print('model build time: %f s'%t)
                
        else:            
            if verbose>6:
                print('Building multi GPU model...',end=' ')
            t = time.time()
            
            strategy = tf.distribute.MirroredStrategy(gpu_device)
            with strategy.scope():
                model=model_func(verbose=verbose>4, **modelaug_tmp)
                
                if weight_load_name is not None:
                    model.load_weights(weight_load_name)
                    
                model.compile( **compile_argument)
                
                if verbose>7:
                    model.summary()
            t = time.time()-t
            if verbose>2:
                print('/rmulti GPU model build time: %f s'%t)            
            
        
        t = time.time()
        if verbose>5:
            print('evaluating...')
        if verbose>3:
            infverbose=1
        else:
            infverbose=0
            
        if FT_evaluate_argument is not None:
            if datagen is None:
                prediction = model.predict(x_test, verbose=infverbose,batch_size=model_argument[scheme_num]['batch_size'])
            else:
                prediction = model.predict(datagen, verbose=infverbose,steps=len(datagen))
            FT_evaluate_argument['prediction']=prediction
            FT_evaluate_argument['test_label']=y_test
            test_result = evaluate_FT( **FT_evaluate_argument)
        else:
            if datagen is None:
                test_result = model.evaluate(x_test, y_test, verbose=infverbose, batch_size=model_argument[scheme_num]['batch_size'])
            else:
                test_result = model.evaluate(datagen, verbose=infverbose, steps=len(datagen))
        
        t = time.time()-t
        if verbose>2:
            print('\nruntime: %f s'%t)        
        
        if verbose>1:
            if FT_evaluate_argument is not None:
                for key in test_result.keys():
                    print('Test %s\t:'%key, test_result[key])
            else:
                for i in range(len(test_result)):
                    print('Test %s\t:'%model.metrics_names[i], test_result[i])
            
        if (scheme_num == 0 and not append_save_file) or (append_save_file and not os.path.exists(result_save_file)):
            with open(result_save_file, 'w', newline='') as csvfile:
                fieldnames=list()
                test_result_dict=dict()
                if FT_evaluate_argument is not None:
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
                if save_file_add_on is not None:
                    for key in save_file_add_on.keys():
                        fieldnames.append(key)
                        test_result_dict[key]=save_file_add_on[key][scheme_num]
                writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(test_result_dict)
        else:
            with open(result_save_file, 'a', newline='') as csvfile:
                fieldnames=list()
                test_result_dict=dict()
                if FT_evaluate_argument is not None:
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
                if save_file_add_on is not None:
                    for key in save_file_add_on.keys():
                        fieldnames.append(key)
                        test_result_dict[key]=save_file_add_on[key][scheme_num]
                writer=csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(test_result_dict)
                    
            
        K.clear_session()
        del model
                  
        if verbose>1:          
            print('\n===============================================\n')


def gen_test_round_list(num_of_bit,upper_bound,lower_bound,left_bound=-3,right_bound=0):
    """Genrate test round list with number decade exponentially

    Arguments
    ---------
    num_of_bit_bit: Integer. 
        The total amount of bits in the unit for fault injection.
    upper_bound: Integer. 
        The maximum number for test rounds.
    lower_bound: Integer. 
        The minimun number for test rounds.
    left_bound: Integer. 
        The number for line y=exp(x) left bound. For generate decade number.
    right_bound: Integer. 
        The number for line y=exp(x) right bound. For generate decade number.
        
    Returns
    -------
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
                
        

