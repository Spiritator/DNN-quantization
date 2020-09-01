# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:35:57 2019

@author: Yung-Yu Tsai

Fault tolerance evaluation functions
"""

import inspect
from ..metrics.FT_metrics import FT_metric_setup
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import categorical_accuracy
import tensorflow as tf

def evaluate_FT(model_name,prediction,test_label,loss_function,metrics,fuseBN=None,setsize=50,score=None,fault_free_pred=None):
    """
    Run the evaluation of given fault tolerance metrics

    Parameters
    ----------
    model_name : String
        Name of model. Support LeNet-5, Custom 4C2F, MobileNetV1, ResNet50.
    prediction : Ndarray
        The output probability of DNN model.
    test_label : Ndarray
        The label of test set image data.
    loss_function : Callable TensorFlow function
        The loss function for DNN under test.
    metrics : List of String or Callable TensorFlow function
        The metrics for DNN under test.
    fuseBN : Bool, optional
        Flag for identify the DNN under test is a Fused BatchNormalization case or not. The default is None.
    setsize : Integer, optional. One of 2, 10, 50.
        For the case of DNN under test using ImageNet as benchmark dataset. There are a few presets of dataset size. 
        The setsize number represent the number of images in each ImageNet classes. The default is 50.
    score : List of Float, optional
        The base [Loss, Top-1 Accuracy, Top-K Accuracy] for comparing-to-fault-free based metrics. 
        If default as None, function will automaticly get stored golden stats. The default is None.
    fault_free_pred : Ndarray, optional
        The base golden prediction probabilities for all classes for comparing-to-fault-free based metrics. 
        If default as None, function will automaticly get stored golden output probabilities. The default is None.

    Returns
    -------
    test_result : List of Float
        The result of given metrics.

    """
    ff_score,ff_pred=FT_metric_setup(model_name,fuseBN=fuseBN,setsize=setsize,score=score,fault_free_pred=fault_free_pred)
    
    test_label_tf=tf.constant(test_label)
    prediction_tf=tf.constant(prediction)
    ff_score=tf.constant(ff_score)
    ff_pred=tf.constant(ff_pred)
    
    test_output=list()
    test_result=['loss']
    
    test_output.append(K.mean(loss_function(test_label_tf,prediction_tf)))
    
    for metric in metrics:
        if metric in ('accuracy', 'acc'):
            test_output.append(K.mean(categorical_accuracy(test_label_tf,prediction_tf)))
            test_result.append(metric)
        else:
            test_result.append(metric.__name__)
            if 'ff_score' in inspect.signature(metric).parameters and 'ff_pred' in inspect.signature(metric).parameters:
                test_output.append(K.mean(metric(test_label_tf,prediction_tf,ff_score,ff_pred)))
            elif 'ff_score' in inspect.signature(metric).parameters:
                test_output.append(K.mean(metric(test_label_tf,prediction_tf,ff_score)))
            elif 'ff_pred' in inspect.signature(metric).parameters:
                test_output.append(K.mean(metric(test_label_tf,prediction_tf,ff_pred)))
            else:
                test_output.append(K.mean(metric(test_label_tf,prediction_tf)))
     
    test_output=K.eval(K.stack(test_output,axis=0))
    
    test_result=dict(zip(test_result,test_output))
    
    return test_result
