# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 15:38:34 2018

@author: Yung-Yu Tsai

custom topk metrics
"""

from keras import metrics

# Let's train the model using RMSprop
def top2_acc(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=2)

def top3_acc(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=3)

def top5_acc(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=5)
