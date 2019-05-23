# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:35:57 2019

@author: Yung-Yu Tsai

Fault tolerance evaluation functions
"""

import inspect
from metrics.FT_metrics import FT_metric_setup
import keras.backend as K
from keras.metrics import categorical_accuracy
import tensorflow as tf

def evaluate_FT(model_name,prediction,test_label,loss_function,metrics,fuseBN=None,setsize=50,score=None,fault_free_pred=None):
    ff_score,ff_pred=FT_metric_setup(model_name,fuseBN=fuseBN,setsize=setsize,score=score,fault_free_pred=fault_free_pred)
    
    test_output=list()
    test_result=['loss']
    
    test_output.append(K.mean(loss_function(tf.Variable(test_label),tf.Variable(prediction))))
    
    for metric in metrics:
        if metric in ('accuracy', 'acc'):
            test_output.append(K.mean(categorical_accuracy(test_label,prediction)))
            test_result.append(metric)
        else:
            test_result.append(metric.__name__)
            if 'ff_score' in inspect.signature(metric).parameters and 'ff_pred' in inspect.signature(metric).parameters:
                test_output.append(K.mean(metric(test_label,prediction,ff_score,ff_pred)))
            elif 'ff_score' in inspect.signature(metric).parameters:
                test_output.append(K.mean(metric(test_label,prediction,ff_score)))
            elif 'ff_pred' in inspect.signature(metric).parameters:
                test_output.append(K.mean(metric(test_label,prediction,ff_pred)))
            else:
                test_output.append(K.mean(metric(test_label,prediction)))
     
    test_output=K.eval(K.stack(test_output,axis=0))
    
    test_result=dict(zip(test_result,test_output))
    
    return test_result
