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

def evaluate_FT(model_name,prediction,test_label,metrics,fuseBN=None,setsize=50,score=None,fault_free_pred=None):
    ff_score,ff_pred=FT_metric_setup(model_name,fuseBN=fuseBN,setsize=setsize,score=score,fault_free_pred=fault_free_pred)
    
    test_result=dict()
    
    for metric in metrics:
        if metric in ('accuracy', 'acc'):
            test_result['accuracy']=K.eval(K.mean(categorical_accuracy(test_label,prediction)))
        else:
            if 'ff_score' in inspect.signature(metric).parameters and 'ff_pred' in inspect.signature(metric).parameters:
                test_result[metric.__name__]=K.eval(K.mean(metric(test_label,prediction,ff_score,ff_pred)))
            elif 'ff_score' in inspect.signature(metric).parameters:
                test_result[metric.__name__]=K.eval(K.mean(metric(test_label,prediction,ff_score)))
            elif 'ff_pred' in inspect.signature(metric).parameters:
                test_result[metric.__name__]=K.eval(K.mean(metric(test_label,prediction,ff_pred)))
            else:
                test_result[metric.__name__]=K.eval(K.mean(metric(test_label,prediction)))

    return test_result
