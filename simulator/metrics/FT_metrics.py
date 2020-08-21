# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:16:15 2019

@author: Yung-Yu Tsai

metirc for fault tolerance analysis
"""

import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import backend as K
import numpy as np


def FT_metric_setup(model,fuseBN=None,setsize=50,score=None,fault_free_pred=None):
    # score of original floating-point fault free NN
    lenet5_mnist_stat=[0.025439078912439465,0.9920999966561794,0.9986999993026257]
    C4F2fusedBN_cifar10_stat=[0.391714577046223,0.8702999973297119,0.9513999938964843]
    mobilenet_imagenet_stat=[1.6203491767287255,0.6226999995172023,0.8393599996805191]
    mobilenet_fusedBN_imagenet_stat=[1.47549153089523,0.625499997138977,0.843499995470047]
    resnet50_imagenet_stat=[1.35314666219651,0.680959999883174,0.883000002408027]
    resnet50_fusedBN_imagenet_stat=[1.3531466766715,0.680959999883174,0.883000002408027]

    if not isinstance(fuseBN,bool) and fuseBN is not None:
        raise ValueError('Augment fusedBN is a indicator for whether the model is fused BN or not. Please use bool type.')
        
    if not isinstance(model,str):
        raise ValueError('Augment model is the name of model please use string to assign model name.')
        
    if model.lower() in ['lenet','lenet5','le']:
        ff_score=lenet5_mnist_stat
        ff_pred=np.load('../fault_free_pred/lenet5_mnist_fault_free_pred.npy')
    elif model.lower() in ['4c2f','c4f2','cifar10']:
        ff_score=C4F2fusedBN_cifar10_stat
        ff_pred=np.load('../fault_free_pred/C4F2_cifar10_fault_free_pred.npy')
    elif model.lower() in ['mobile','mobilenet','mobilenetv1','mobilenet-v1']:
        if fuseBN:
            ff_score=mobilenet_fusedBN_imagenet_stat
        else:
            ff_score=mobilenet_imagenet_stat
            
        if setsize==50:
            ff_pred=np.load('../fault_free_pred/mobilenet_imagenet_fault_free_pred.npy')
        elif setsize==10:
            ff_pred=np.load('../fault_free_pred/mobilenet_imagenet_fault_free_pred_setsize_10.npy')
        elif setsize==2:
            ff_pred=np.load('../fault_free_pred/mobilenet_imagenet_fault_free_pred_setsize_2.npy')
        else:
            raise ValueError('setsize %d doesn\'t exist!'%setsize)
            
    elif model.lower() in ['res','resnet','resnet50']:
        if fuseBN:
            ff_score=resnet50_fusedBN_imagenet_stat
        else:
            ff_score=resnet50_imagenet_stat
            
        if setsize==50:
            ff_pred=np.load('../fault_free_pred/resnet50_imagenet_fault_free_pred.npy')
        elif setsize==10:
            ff_pred=np.load('../fault_free_pred/resnet50_imagenet_fault_free_pred_setsize_10.npy')
        elif setsize==2:
            ff_pred=np.load('../fault_free_pred/resnet50_imagenet_fault_free_pred_setsize_2.npy')
        else:
            raise ValueError('setsize %d doesn\'t exist!'%setsize)        
            
    else:
        raise ValueError('model %s doesn\'t exist!'%model)
    
    if score is not None:
        ff_score=score
    if fault_free_pred is not None:
        ff_pred=fault_free_pred
        
    return ff_score,ff_pred

def acc_loss(y_true,y_pred,ff_score):
    pred_acc=K.mean(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)))
    return K.clip(tf.subtract(ff_score[1],pred_acc),0.0,None)

def relative_acc(y_true,y_pred,ff_score):
    pred_acc=K.mean(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)))
    return K.clip(tf.divide(pred_acc,ff_score[1]),0.0,1.0)

def pred_miss(y_true,y_pred,ff_pred):
    return tf.subtract(1.0,K.mean(K.equal(K.argmax(y_pred, axis=-1),K.argmax(ff_pred, axis=-1))))

def top2_pred_miss(y_true,y_pred,ff_pred):
    return tf.subtract(1.0,K.mean(K.in_top_k(y_pred, K.argmax(ff_pred, axis=-1), 2), axis=-1))

def top3_pred_miss(y_true,y_pred,ff_pred):
    return tf.subtract(1.0,K.mean(K.in_top_k(y_pred, K.argmax(ff_pred, axis=-1), 3), axis=-1))

def top5_pred_miss(y_true,y_pred,ff_pred):
    return tf.subtract(1.0,K.mean(K.in_top_k(y_pred, K.argmax(ff_pred, axis=-1), 5), axis=-1))

def conf_score_vary_10(y_true,y_pred,ff_pred):
    ff_indices=K.concatenate([K.reshape(tf.range(y_pred.shape.dims[0].value,dtype=tf.int64),[-1,1]),K.reshape(K.argmax(y_pred,axis=-1),[-1,1])],axis=1)
    ff_pred_class=tf.gather_nd(ff_pred,ff_indices)
    return K.greater(K.abs(tf.divide(tf.subtract(K.max(y_pred,axis=-1),ff_pred_class),ff_pred_class)),0.1)

def conf_score_vary_20(y_true,y_pred,ff_pred):
    ff_indices=K.concatenate([K.reshape(tf.range(y_pred.shape.dims[0].value,dtype=tf.int64),[-1,1]),K.reshape(K.argmax(y_pred,axis=-1),[-1,1])],axis=1)
    ff_pred_class=tf.gather_nd(ff_pred,ff_indices)
    return K.greater(K.abs(tf.divide(tf.subtract(K.max(y_pred,axis=-1),ff_pred_class),ff_pred_class)),0.2)
    
def conf_score_vary_50(y_true,y_pred,ff_pred):
    ff_indices=K.concatenate([K.reshape(tf.range(y_pred.shape.dims[0].value,dtype=tf.int64),[-1,1]),K.reshape(K.argmax(y_pred,axis=-1),[-1,1])],axis=1)
    ff_pred_class=tf.gather_nd(ff_pred,ff_indices)
    return K.greater(K.abs(tf.divide(tf.subtract(K.max(y_pred,axis=-1),ff_pred_class),ff_pred_class)),0.5)


    