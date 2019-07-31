# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:45:33 2018

@author: Yung-Yu Tsai

Generate random stuck at fault on a model. 
Only generate the layers have weights.
assign a fault rate to generate.
support different type of fault distribution
"""

import numpy as np
from .fault_core import generate_stuck_at_fault_modulator_fast
        
def coordinate_gen_fmap(data_shape,batch_size,distribution='uniform',poisson_lam=None, mean=None, std=None, concentration=None):
    """Generate the coordinate of a feature map base on its shape and with specific distibution type.

    # Arguments
        data_shape: Tuple. The shape of feature map.
        batch_size: Integer. The batch size of fault tolerance evaluation process.
        distribution: String. The distribution type of coordinate in feature map. Must be one of 'uniform', 'poisson', 'normal'.
        poisson_lam: Tuple. The lambda of poisson distribution.
        mean: Tuple or Float. The mean value for normal or center distribution in coordinate. Tuple for normal distribution. Float for concentration of center distribution.
        std: Tuple or Float. The standard deviation value for normal or center distribution in coordinate. Tuple for normal distribution. Float for concentration of center distribution.
        concentration: Float. A float number between 1 and 0. 1 means set the lambda of 2D feature map to center of image, 0 means set to corner of the feature map. (can only be use in conv 2D layer)

    # Returns
        The coordinate Tuple.
    """
    coordinate=list()
    
    if distribution=='uniform':
        coordinate.append(np.random.randint(batch_size))
        for j in range(1,len(data_shape)):
            coordinate.append(np.random.randint(data_shape[j]))
    elif distribution=='poisson':
        if not isinstance(poisson_lam,tuple) or len(poisson_lam)!=len(data_shape)-1:
            raise TypeError('Poisson distribution lambda setting must be a tuple same length as input data shape which indicates the lamda of poisson distribution.')
        
        coordinate.append(np.random.randint(batch_size))
        for j in range(1,len(data_shape)):
            if isinstance(poisson_lam[j-1],int) and poisson_lam[j-1]>=0 and poisson_lam[j-1]<data_shape[j]:
                coor_tmp=np.random.poisson(poisson_lam[j-1])
                while coor_tmp>=data_shape[j]:
                    coor_tmp=np.random.poisson(poisson_lam[j-1])
                coordinate.append(coor_tmp)
            else:
                raise ValueError('Poisson distribution Lambda must within feature map shape. Feature map shape %s but got lambda input %s'%(str(data_shape),str(poisson_lam)))
    elif distribution=='normal':
        if not isinstance(mean,tuple) or len(mean)!=len(data_shape):
            raise TypeError('Normal distribution mean setting must be a tuple same length as input data shape which indicates the lamda of poisson distribution.')
        if not isinstance(std,tuple) or len(std)!=len(data_shape):
            raise TypeError('Normal distribution std setting must be a tuple same length as input data shape which indicates the lamda of poisson distribution.')
        
        coordinate.append(np.random.randint(batch_size))
        for j in range(1,len(data_shape)):
            if isinstance(mean[j-1],int) and mean[j-1]>=0 and mean[j-1]<data_shape[j]:
                coor_tmp=np.random.normal(mean[j-1],std[j-1])
                while coor_tmp>=data_shape[j]:
                    coor_tmp=np.random.normal(mean[j-1],std[j-1])
                coordinate.append(coor_tmp)
            else:
                raise ValueError('Normal distribution Mean must within feature map shape. Feature map shape %s but got lambda input %s'%(str(data_shape),str(poisson_lam)))
    elif distribution=='center':
        if len(data_shape)<4:
            return None

        if concentration<0 or concentration>1:
            raise ValueError('Center distribution concentration setting must be within [0,1] (inclusive).')
        if std is None:
            std=0.075
            
        coordinate.append(np.random.randint(batch_size))

        dist=np.linalg.norm(data_shape[1:3])
        radius=np.random.normal(1-concentration,std)
        theta=np.random.uniform(high=2*np.pi)
        width=radius*np.cos(theta)
        height=radius*np.sin(theta)
        width=int(np.clip(width*dist/2+data_shape[1]/2,0,data_shape[1]-1))
        height=int(np.clip(height*dist/2+data_shape[2]/2,0,data_shape[2]-1))
        
        coordinate.append(width)
        coordinate.append(height)
        
        coordinate.append(np.random.randint(data_shape[3]))
    else:
        raise NameError('Invalid type of random generation distribution. Please choose between uniform, poisson, normal.')
    
    coordinate=tuple(coordinate)
    return coordinate

def coordinate_gen_fmap_fast(data_shape,batch_size,fault_num,distribution='uniform',poisson_lam=None, mean=None, std=None, concentration=None):
    """Generate the coordinate of a feature map base on its shape and with specific distibution type.
       Faster generation version not multiple fault in a parameter.

    # Arguments
        data_shape: Tuple. The shape of feature map.
        batch_size: Integer. The batch size of fault tolerance evaluation process.
        fault_num: Integer. The number of faults in fmap.
        distribution: String. The distribution type of coordinate in feature map. Must be one of 'uniform', 'poisson', 'normal'.
        poisson_lam: Tuple. The lambda of poisson distribution.
        mean: Tuple or Float. The mean value for normal or center distribution in coordinate. Tuple for normal distribution. Float for concentration of center distribution.
        std: Tuple or Float. The standard deviation value for normal or center distribution in coordinate. Tuple for normal distribution. Float for concentration of center distribution.
        concentration: Float. A float number between 1 and 0. 1 means set the lambda of 2D feature map to center of image, 0 means set to corner of the feature map. (can only be use in conv 2D layer)

    # Returns
        The coordinate Tuple.
    """
    coordinate=list()
    
    if distribution=='uniform':
        coordinate.append(np.random.randint(batch_size,size=fault_num))
        for j in range(1,len(data_shape)):
            coordinate.append(np.random.randint(data_shape[j],size=fault_num))
    elif distribution=='poisson':
        if not isinstance(poisson_lam,tuple) or len(poisson_lam)!=len(data_shape)-1:
            raise TypeError('Poisson distribution lambda setting must be a tuple same length as input data shape which indicates the lamda of poisson distribution.')
        
        coordinate.append(np.random.randint(batch_size,size=fault_num))
        for j in range(1,len(data_shape)):
            if isinstance(poisson_lam[j-1],int) and poisson_lam[j-1]>=0 and poisson_lam[j-1]<data_shape[j]:
                coor_tmp=np.random.poisson(poisson_lam[j-1],size=fault_num)
                coor_tmp=np.clip(coor_tmp,0,data_shape[j]-1)
                coordinate.append(coor_tmp)
            else:
                raise ValueError('Poisson distribution Lambda must within feature map shape. Feature map shape %s but got lambda input %s'%(str(data_shape),str(poisson_lam)))
    elif distribution=='normal':
        if not isinstance(mean,tuple) or len(mean)!=len(data_shape):
            raise TypeError('Normal distribution mean setting must be a tuple same length as input data shape which indicates the lamda of poisson distribution.')
        if not isinstance(std,tuple) or len(std)!=len(data_shape):
            raise TypeError('Normal distribution std setting must be a tuple same length as input data shape which indicates the lamda of poisson distribution.')
        
        coordinate.append(np.random.randint(batch_size,size=fault_num))
        for j in range(1,len(data_shape)):
            if isinstance(mean[j-1],int) and mean[j-1]>=0 and mean[j-1]<data_shape[j]:
                coor_tmp=np.random.normal(mean[j-1],std[j-1],size=fault_num)
                coor_tmp=np.clip(coor_tmp,0,data_shape[j]-1)
                coordinate.append(coor_tmp)
            else:
                raise ValueError('Normal distribution Mean must within feature map shape. Feature map shape %s but got lambda input %s'%(str(data_shape),str(poisson_lam)))
    elif distribution=='center':
        if len(data_shape)<4:
            return None
        
        if concentration<0 or concentration>1:
            raise ValueError('Center distribution concentration setting must be within [0,1] (inclusive).')
        if std is None:
            std=0.075
            
        coordinate.append(np.random.randint(batch_size,size=fault_num))

        dist=np.linalg.norm(data_shape[1:3])
        radius=np.random.normal(1-concentration,std,size=fault_num)
        theta=np.random.uniform(high=2*np.pi,size=fault_num)
        width=np.multiply(radius,np.cos(theta))
        height=np.multiply(radius,np.sin(theta))
        width=(np.clip(width*(dist/2)+data_shape[1]/2,0,data_shape[1]-1)).astype(int)
        height=(np.clip(height*(dist/2)+data_shape[2]/2,0,data_shape[2]-1)).astype(int)
        
        coordinate.append(width)
        coordinate.append(height)
        
        coordinate.append(np.random.randint(data_shape[3],size=fault_num))
    else:
        raise NameError('Invalid type of random generation distribution. Please choose between uniform, poisson, normal.')
    
    coordinate=list(zip(*coordinate))
    return coordinate

def coordinate_gen_wght(data_shape,distribution='uniform',poisson_lam=None, mean=None, std=None, concentration=None):
    """Generate the coordinate of a weights base on its shape and with specific distibution type.

    # Arguments
        data_shape: Tuple. The shape of weights.
        distribution: String. The distribution type of coordinate in weights. Must be one of 'uniform', 'poisson', 'normal'.
        poisson_lam: Tuple. The lambda of poisson distribution.
        mean: Tuple or Float. The mean value for normal or center distribution in coordinate. Tuple for normal distribution. Float for concentration of center distribution.
        std: Tuple or Float. The standard deviation value for normal or center distribution in coordinate. Tuple for normal distribution. Float for concentration of center distribution.
        concentration: Float. A float number between 1 and 0. 1 means set the lambda of 2D feature map to center of image, 0 means set to corner of the feature map. (can only be use in conv 2D layer)

    # Returns
        The coordinate Tuple.
    """

    coordinate=list()
    
    if distribution=='uniform':
        for j in range(len(data_shape)):
            coordinate.append(np.random.randint(data_shape[j]))
    elif distribution=='poisson':
        if not isinstance(poisson_lam,tuple) or len(poisson_lam)!=len(data_shape):
            raise TypeError('Poisson distribution lambda setting must be a tuple same length as input data shape which indicates the lamda of poisson distribution.')
        
        for j in range(len(data_shape)):
            if isinstance(poisson_lam[j],int) and poisson_lam[j]>=0 and poisson_lam[j]<data_shape[j]:
                coor_tmp=np.random.poisson(poisson_lam[j])
                while coor_tmp>=data_shape[j]:
                    coor_tmp=np.random.poisson(poisson_lam[j])
                coordinate.append(coor_tmp)
            else:
                raise ValueError('Poisson distribution Lambda must within feature map shape. Feature map shape %s but got lambda input %s'%(str(data_shape),str(poisson_lam)))
    elif distribution=='normal':
        if not isinstance(mean,tuple) or len(mean)!=len(data_shape):
            raise TypeError('Normal distribution mean setting must be a tuple same length as input data shape which indicates the lamda of poisson distribution.')
        if not isinstance(std,tuple) or len(std)!=len(data_shape):
            raise TypeError('Normal distribution std setting must be a tuple same length as input data shape which indicates the lamda of poisson distribution.')
        
        for j in range(len(data_shape)):
            if isinstance(mean[j],int) and mean[j]>=0 and mean[j]<data_shape[j]:
                coor_tmp=np.random.normal(mean[j],std[j])
                while coor_tmp>=data_shape[j]:
                    coor_tmp=np.random.normal(mean[j],std[j])
                coordinate.append(coor_tmp)
            else:
                raise ValueError('Normal distribution Mean must within feature map shape. Feature map shape %s but got lambda input %s'%(str(data_shape),str(poisson_lam)))
    elif distribution=='center':
        if len(data_shape)<4:
            return None

        if concentration<0 or concentration>1:
            raise ValueError('Center distribution concentration setting must be within [0,1] (inclusive).')
        if std is None:
            std=0.075
            
        dist=np.linalg.norm(data_shape[:2])
        radius=np.random.normal(1-concentration,std)
        theta=np.random.uniform(high=2*np.pi)
        width=radius*np.cos(theta)
        height=radius*np.sin(theta)
        width=int(np.clip(width*dist/2+data_shape[0]/2,0,data_shape[0]-1))
        height=int(np.clip(height*dist/2+data_shape[1]/2,0,data_shape[1]-1))
        
        coordinate.append(width)
        coordinate.append(height)
        
        coordinate.append(np.random.randint(data_shape[2]))
        coordinate.append(np.random.randint(data_shape[3]))
    else:
        raise NameError('Invalid type of random generation distribution. Please choose between uniform, poisson, normal.')
    
    coordinate=tuple(coordinate)
    return coordinate

def coordinate_gen_wght_fast(data_shape,fault_num,distribution='uniform' ,poisson_lam=None, mean=None, std=None, concentration=None):
    """Generate the coordinate of a weights base on its shape and with specific distibution type.
       Faster generation version not multiple fault in a parameter.
       
    # Arguments
        data_shape: Tuple. The shape of weights.
        fault_num: Integer. The number of faults in weight.
        distribution: String. The distribution type of coordinate in weights. Must be one of 'uniform', 'poisson', 'normal'.
        poisson_lam: Tuple. The lambda of poisson distribution.
        mean: Tuple or Float. The mean value for normal or center distribution in coordinate. Tuple for normal distribution. Float for concentration of center distribution.
        std: Tuple or Float. The standard deviation value for normal or center distribution in coordinate. Tuple for normal distribution. Float for concentration of center distribution.
        concentration: Float. A float number between 1 and 0. 1 means set the lambda of 2D feature map to center of image, 0 means set to corner of the feature map. (can only be use in conv 2D layer)

    # Returns
        The coordinate Tuple.
    """

    coordinate=list()
    
    if distribution=='uniform':
        for j in range(len(data_shape)):
            coordinate.append(np.random.randint(data_shape[j],size=fault_num))
    elif distribution=='poisson':
        if not isinstance(poisson_lam,tuple) or len(poisson_lam)!=len(data_shape):
            raise TypeError('Poisson distribution lambda setting must be a tuple same length as input data shape which indicates the lamda of poisson distribution.')
        
        for j in range(len(data_shape)):
            if isinstance(poisson_lam[j],int) and poisson_lam[j]>=0 and poisson_lam[j]<data_shape[j]:
                coor_tmp=np.random.poisson(poisson_lam[j],size=fault_num)
                coor_tmp=np.clip(coor_tmp,0,data_shape[j]-1)
                coordinate.append(coor_tmp)
            else:
                raise ValueError('Poisson distribution Lambda must within feature map shape. Feature map shape %s but got lambda input %s'%(str(data_shape),str(poisson_lam)))
    elif distribution=='normal':
        if not isinstance(mean,tuple) or len(mean)!=len(data_shape):
            raise TypeError('Normal distribution mean setting must be a tuple same length as input data shape which indicates the lamda of poisson distribution.')
        if not isinstance(std,tuple) or len(std)!=len(data_shape):
            raise TypeError('Normal distribution std setting must be a tuple same length as input data shape which indicates the lamda of poisson distribution.')
        
        for j in range(len(data_shape)):
            if isinstance(mean[j],int) and mean[j]>=0 and mean[j]<data_shape[j]:
                coor_tmp=np.random.normal(mean[j],std[j],size=fault_num)
                coor_tmp=np.clip(coor_tmp,0,data_shape[j]-1)
                coordinate.append(coor_tmp)
            else:
                raise ValueError('Normal distribution Mean must within feature map shape. Feature map shape %s but got lambda input %s'%(str(data_shape),str(poisson_lam)))
    elif distribution=='center':
        if len(data_shape)<4:
            return None

        if concentration<0 or concentration>1:
            raise ValueError('Center distribution concentration setting must be within [0,1] (inclusive).')
        if std is None:
            std=0.075
            
        dist=np.linalg.norm(data_shape[:2])
        radius=np.random.normal(1-concentration,std,size=fault_num)
        theta=np.random.uniform(high=2*np.pi,size=fault_num)
        width=radius*np.cos(theta)
        height=radius*np.sin(theta)
        width=(np.clip(width*dist/2+data_shape[0]/2,0,data_shape[0]-1)).astype(int)
        height=(np.clip(height*dist/2+data_shape[1]/2,0,data_shape[1]-1)).astype(int)
        
        coordinate.append(width)
        coordinate.append(height)
        
        coordinate.append(np.random.randint(data_shape[2],size=fault_num))
        coordinate.append(np.random.randint(data_shape[3],size=fault_num))
    else:
        raise NameError('Invalid type of random generation distribution. Please choose between uniform, poisson, normal.')
    
    coordinate=list(zip(*coordinate))
    return coordinate


def fault_bit_loc_gen(model_word_length,distribution='uniform',poisson_lam=None, mean=None, std=None):
    """Generate the location of a fault bit in a parameter base on its word length and with specific distibution type.

    # Arguments
        model_word_length: Integer. The word length of model parameters.
        distribution: String. The distribution type of locaton in parameters. Must be one of 'uniform', 'poisson', 'normal'.
        poisson_lam: Integer. The lambda of poisson distribution.
        mean: Float. The mean value for normal or center distribution in bit location. 
        std: Float. The standard deviation value for normal or center distribution in bit location.

    # Returns
        The location index (Integer).
    """

    if distribution=='uniform':
        fault_bit=np.random.randint(model_word_length)
    elif distribution=='poisson':
        if isinstance(poisson_lam,int) and poisson_lam>=0 and poisson_lam<model_word_length:
            fault_bit=np.random.poisson(poisson_lam)
            while fault_bit>=model_word_length:
                fault_bit=np.random.poisson(poisson_lam)
        else:
            raise ValueError('Poisson distribution Lambda must within model word length.')
    elif distribution=='normal':
        if isinstance(mean,int) and mean>=0 and mean<model_word_length:
            fault_bit=np.random.normal(mean,std)
            while fault_bit>=model_word_length:
                fault_bit=np.random.normal(mean,std)
        else:
            raise ValueError('Normal distribution Mean must within model word length.')
    else:
        raise NameError('Invalid type of random generation distribution. Please choose between uniform, poisson, normal.')
    return fault_bit

def fault_bit_loc_gen_fast(model_word_length,fault_num,distribution='uniform',poisson_lam=None, mean=None, std=None):
    """Generate the location of a fault bit in a parameter base on its word length and with specific distibution type.
       Faster generation version not multiple fault in a parameter.

    # Arguments
        model_word_length: Integer. The word length of model parameters.
        fault_num: Integer. The number of faults in parameter.
        distribution: String. The distribution type of locaton in parameters. Must be one of 'uniform', 'poisson', 'normal'.
        poisson_lam: Integer. The lambda of poisson distribution.
        mean: Float. The mean value for normal or center distribution in bit location. 
        std: Float. The standard deviation value for normal or center distribution in bit location.

    # Returns
        The location index (Integer).
    """

    if distribution=='uniform':
        fault_bit=np.random.randint(model_word_length,size=fault_num)
    elif distribution=='poisson':
        if isinstance(poisson_lam,int) and poisson_lam>=0 and poisson_lam<model_word_length:
            fault_bit=np.random.poisson(poisson_lam,size=fault_num)
            fault_bit=np.clip(fault_bit,0,model_word_length-1)
        else:
            raise ValueError('Poisson distribution Lambda must within model word length.')
    elif distribution=='normal':
        if isinstance(mean,int) and mean>=0 and mean<model_word_length:
            fault_bit=np.random.normal(mean,std,size=fault_num)
            fault_bit=np.clip(fault_bit,0,model_word_length-1)
        else:
            raise ValueError('Normal distribution Mean must within model word length.')
    else:
        raise NameError('Invalid type of random generation distribution. Please choose between uniform, poisson, normal.')
    return fault_bit

def fault_num_gen_fmap(data_shape,fault_rate,batch_size,model_word_length):
    if isinstance(data_shape,list):
        fault_num=[int(np.prod(shapes[1:]) * batch_size * model_word_length * fault_rate) for shapes in data_shape]
    else:
        fault_num=int(np.prod(data_shape[1:]) * batch_size * model_word_length * fault_rate)
    return fault_num

def fault_num_gen_wght(data_shape,fault_rate,model_word_length):
    fault_num=[int(np.prod(shapes) * model_word_length * fault_rate) for shapes in data_shape]
    return fault_num 

def get_model_total_bits(model,batch_size,model_word_length):
    model_depth=len(model.layers)
    total_ifmap_bits=0
    total_ofmap_bits=0
    total_weight_bits=0
    
    for layer_num in range(1,model_depth):
        layer=model.layers[layer_num]
        layer_input_shape=layer.input_shape
        layer_output_shape=layer.output_shape
        layer_weight_shape=[weight_shape.shape for weight_shape in layer.get_weights()]
        
        if len(layer_weight_shape)==0:
            continue
        
        if isinstance(layer_input_shape,list):
            for i in range(len(layer_input_shape)):
                bits_tmp=int(np.prod(layer_input_shape[i][1:]) * batch_size * model_word_length)
                total_ifmap_bits+=bits_tmp
        else:
            bits_tmp=int(np.prod(layer_input_shape[1:]) * batch_size * model_word_length)
            total_ifmap_bits+=bits_tmp
            
        if isinstance(layer_output_shape,list):
            for i in range(len(layer_output_shape)):
                bits_tmp=int(np.prod(layer_output_shape[i][1:]) * batch_size * model_word_length)
                total_ofmap_bits+=bits_tmp
        else:
            bits_tmp=int(np.prod(layer_output_shape[1:]) * batch_size * model_word_length)
            total_ofmap_bits+=bits_tmp
            
        for i in range(len(layer_weight_shape)):
            bits_tmp=int(np.prod(layer_weight_shape[i][1:]) * batch_size * model_word_length)
            total_weight_bits+=bits_tmp
            
    return total_ifmap_bits,total_ofmap_bits,total_weight_bits

def fault_num_gen_model(model,fault_rate,batch_size,model_word_length):
    model_depth=len(model.layers)
    ifmap_param_bits=[0 for _ in range(model_depth)]
    wght_param_bits=[[0,0] for _ in range(model_depth)]
    ofmap_param_bits=[0 for _ in range(model_depth)]
    total_ifmap_bits=0
    total_ofmap_bits=0
    total_weight_bits=0
    if not isinstance(model_word_length,list):
        model_word_length=[model_word_length,model_word_length,model_word_length]
        
    for layer_num in range(1,model_depth):
        layer=model.layers[layer_num]
        layer_input_shape=layer.input_shape
        layer_output_shape=layer.output_shape
        layer_weight_shape=[weight_shape.shape for weight_shape in layer.get_weights()]
        
        if len(layer_weight_shape)==0:
            continue
        
        if isinstance(layer_input_shape,list):
            layer_ifmap_param_bits=list()
            for i in range(len(layer_input_shape)):
                bits_tmp=int(np.prod(layer_input_shape[i][1:]) * batch_size * model_word_length[0])
                layer_ifmap_param_bits.append(bits_tmp)
                total_ifmap_bits+=bits_tmp
            ifmap_param_bits[layer_num]=layer_ifmap_param_bits
        else:
            bits_tmp=int(np.prod(layer_input_shape[1:]) * batch_size * model_word_length[0])
            ifmap_param_bits[layer_num]=bits_tmp
            total_ifmap_bits+=bits_tmp
            
        if isinstance(layer_output_shape,list):
            layer_ofmap_param_bits=list()
            for i in range(len(layer_output_shape)):
                bits_tmp=int(np.prod(layer_output_shape[i][1:]) * batch_size * model_word_length[2])
                layer_ofmap_param_bits.append(bits_tmp)
                total_ofmap_bits+=bits_tmp
            ofmap_param_bits[layer_num]=layer_ofmap_param_bits
        else:
            bits_tmp=int(np.prod(layer_output_shape[1:]) * batch_size * model_word_length[2])
            ofmap_param_bits[layer_num]=bits_tmp
            total_ofmap_bits+=bits_tmp
            
        layer_weight_param_bits=list()
        for i in range(len(layer_weight_shape)):
            bits_tmp=int(np.prod(layer_weight_shape[i][1:]) * batch_size * model_word_length[1])
            layer_weight_param_bits.append(bits_tmp)
            total_weight_bits+=bits_tmp
        wght_param_bits[layer_num]=layer_weight_param_bits
        
    ifmap_fault_num=int(total_ifmap_bits*fault_rate)
    ofmap_fault_num=int(total_ofmap_bits*fault_rate)
    weight_fault_num=int(total_weight_bits*fault_rate)
    
    dtind=np.max(np.array([total_ifmap_bits,total_ofmap_bits,total_weight_bits]))
    
    ifmap_fault_list=np.random.randint(total_ifmap_bits,size=ifmap_fault_num,dtype=dtind.dtype)
    ofmap_fault_list=np.random.randint(total_ofmap_bits,size=ofmap_fault_num,dtype=dtind.dtype)
    weight_fault_list=np.random.randint(total_weight_bits,size=weight_fault_num,dtype=dtind.dtype)
    
    ifmap_fault_num_list=[0]
    ofmap_fault_num_list=[0]
    weight_fault_num_list=[0]
    
    filter_fault_num_ifmap_low=0
    filter_fault_num_ofmap_low=0
    filter_fault_num_weight_low=0
    
    for layer_num in range(1,model_depth):
        if isinstance(ifmap_param_bits[layer_num],list):
            layer_ifmap_fault_num=list()
            for i in range(len(ifmap_param_bits[layer_num])):
                filter_fault_num_ifmap_high=filter_fault_num_ifmap_low+ifmap_param_bits[layer_num][i]
                layer_ifmap_fault_num.append(np.sum(np.prod(np.where([ifmap_fault_list>filter_fault_num_ifmap_low,ifmap_fault_list<filter_fault_num_ifmap_high],1,0),axis=0,keepdims=False)))
                filter_fault_num_ifmap_low=filter_fault_num_ifmap_high
            ifmap_fault_num_list.append(layer_ifmap_fault_num)
        else:
            filter_fault_num_ifmap_high=filter_fault_num_ifmap_low+ifmap_param_bits[layer_num]
            ifmap_fault_num_list.append(np.sum(np.prod(np.where([ifmap_fault_list>filter_fault_num_ifmap_low,ifmap_fault_list<filter_fault_num_ifmap_high],1,0),axis=0,keepdims=False)))
            filter_fault_num_ifmap_low=filter_fault_num_ifmap_high
            
        if isinstance(ofmap_param_bits[layer_num],list):
            layer_ofmap_fault_num=list()
            for i in range(len(ofmap_param_bits[layer_num])):
                filter_fault_num_ofmap_high=filter_fault_num_ofmap_low+ofmap_param_bits[layer_num][i]
                layer_ofmap_fault_num.append(np.sum(np.prod(np.where([ofmap_fault_list>filter_fault_num_ofmap_low,ofmap_fault_list<filter_fault_num_ofmap_high],1,0),axis=0,keepdims=False)))
                filter_fault_num_ofmap_low=filter_fault_num_ofmap_high
            ofmap_fault_num_list.append(layer_ofmap_fault_num)
        else:
            filter_fault_num_ofmap_high=filter_fault_num_ofmap_low+ofmap_param_bits[layer_num]
            ofmap_fault_num_list.append(np.sum(np.prod(np.where([ofmap_fault_list>filter_fault_num_ofmap_low,ofmap_fault_list<filter_fault_num_ofmap_high],1,0),axis=0,keepdims=False)))
            filter_fault_num_ofmap_low=filter_fault_num_ofmap_high
            
        layer_weight_fault_num=list()
        for i in range(len(wght_param_bits[layer_num])):
            filter_fault_num_weight_high=filter_fault_num_weight_low+wght_param_bits[layer_num][i]
            layer_weight_fault_num.append(np.sum(np.prod(np.where([weight_fault_list>filter_fault_num_weight_low,weight_fault_list<filter_fault_num_weight_high],1,0),axis=0,keepdims=False)))
            filter_fault_num_weight_low=filter_fault_num_weight_high
        weight_fault_num_list.append(layer_weight_fault_num)
 
    return ifmap_fault_num_list,ofmap_fault_num_list,weight_fault_num_list,total_ifmap_bits,total_ofmap_bits,total_weight_bits
    

def gen_fault_dict_list_fmap(data_shape,
                             fault_rate,
                             batch_size,
                             model_word_length,
                             fault_num=None,
                             fast_gen=False,
                             return_modulator=False,
                             coor_distribution='uniform',
                             coor_pois_lam=None,
                             coor_mean=None,
                             coor_std=None,
                             concentration=None,
                             bit_loc_distribution='uniform',
                             bit_loc_pois_lam=None,
                             bit_loc_mean=None, 
                             bit_loc_std=None,
                             fault_type='flip',
                             **kwargs):
    """Generate the fault dictionary list of a feature map base on its shape and with specific distibution type.

    # Arguments
        data_shape: Tuple. The shape of feature map.
        fault_rate: Float. The probability of fault occurance in feature map.
        batch_size: Integer. The batch size of fault tolerance evaluation process.
        model_word_length: Integer. The word length of model parameters.
        fault_num: Integer. The number of faults in fmap.
        fast_gen: Bool. Use fast generation or not. Fast generation doesn't have multiple fault in single parameter, thus the fault_num maybe inaccurate.
        return_modulator: Bool. Return fault modulator or not. Return fault modulator in fault list generation phase. Further improve generation time. Only available when the fast_gen is True.
        coor_distribution: String. The distribution type of coordinate in feature map. Must be one of 'uniform', 'poisson', 'normal'.
        coor_pois_lam: Tuple. The lambda of poisson distribution of feature map coordinate.
        coor_mean: Tuple or Float. The mean value for normal or center distribution in coordinate. Tuple for normal distribution. Float for concentration of center distribution.
        coor_std: Tuple or Float. The standard deviation value for normal or center distribution in coordinate. Tuple for normal distribution. Float for concentration of center distribution.
        concentration: Float. A float number between 1 and 0. 1 means set the lambda of 2D feature map to center of image, 0 means set to corner of the feature map. (can only be use in conv 2D layer)
        bit_loc_distribution: String. The distribution type of locaton in parameters. Must be one of 'uniform', 'poisson', 'normal'.
        bit_loc_pois_lam: Integer. The lambda of poisson distribution.
        bit_loc_mean: Float. The mean value for normal or center distribution in bit location. 
        bit_loc_std: Float. The standard deviation value for normal or center distribution in bit location.
        fault_type: String. The type of fault.

    # Returns
        The fault information Dictionary. The number of fault generated Integer.
    """

    fault_count=0
    if isinstance(data_shape,list):
        fault_dict=[dict() for _ in range(len(data_shape))]
    else:
        fault_dict=dict()
        
    if fault_num is None:
        fault_num=fault_num_gen_fmap(data_shape,fault_rate,batch_size,model_word_length)
    
    if fast_gen:
        if isinstance(data_shape,list):
            for i in range(len(fault_num)):
                coordinate=coordinate_gen_fmap_fast(data_shape[i],
                                                    batch_size,
                                                    fault_num[i],
                                                    distribution=coor_distribution,
                                                    poisson_lam=coor_pois_lam,
                                                    mean=coor_mean,
                                                    std=coor_std,
                                                    concentration=concentration,
                                                    **kwargs)
                fault_bit=fault_bit_loc_gen_fast(model_word_length,
                                                 fault_num[i],
                                                 distribution=bit_loc_distribution,
                                                 poisson_lam=bit_loc_pois_lam,
                                                 mean=bit_loc_mean,
                                                 std=bit_loc_std,
                                                 **kwargs)
                if coordinate is not None:
                    if return_modulator:
                        tensor_modulator0=None
                        tensor_modulator1=None
                        tensor_modulatorF=None
                        modulator=generate_stuck_at_fault_modulator_fast(data_shape[i],coordinate,fault_type,fault_bit)
                        if fault_type == '0':
                            tensor_modulator0=modulator
                        elif fault_type == '1':
                            tensor_modulator1=modulator
                        elif fault_type == 'flip':
                            tensor_modulatorF=modulator
                        fault_dict[i]=[tensor_modulator0,tensor_modulator1,tensor_modulatorF]
                    else:
                        fault_bit=[{'SA_type':fault_type,'SA_bit':bit} for bit in fault_bit]
                        fault_dict[i]=dict(zip(coordinate,fault_bit))
        else:
            coordinate=coordinate_gen_fmap_fast(data_shape,
                                                batch_size,
                                                fault_num,
                                                distribution=coor_distribution,
                                                poisson_lam=coor_pois_lam,
                                                mean=coor_mean,
                                                std=coor_std,
                                                concentration=concentration,
                                                **kwargs)
            fault_bit=fault_bit_loc_gen_fast(model_word_length,
                                             fault_num,
                                             distribution=bit_loc_distribution,
                                             poisson_lam=bit_loc_pois_lam,
                                             mean=bit_loc_mean,
                                             std=bit_loc_std,
                                             **kwargs)
            if coordinate is not None:
                if return_modulator:
                    tensor_modulator0=None
                    tensor_modulator1=None
                    tensor_modulatorF=None
                    modulator=generate_stuck_at_fault_modulator_fast(data_shape,coordinate,fault_type,fault_bit)
                    if fault_type == '0':
                        tensor_modulator0=modulator
                    elif fault_type == '1':
                        tensor_modulator1=modulator
                    elif fault_type == 'flip':
                        tensor_modulatorF=modulator
                    fault_dict=[tensor_modulator0,tensor_modulator1,tensor_modulatorF]
                else:
                    fault_bit=[{'SA_type':fault_type,'SA_bit':bit} for bit in fault_bit]
                    fault_dict=dict(zip(coordinate,fault_bit))
    else:
        if isinstance(data_shape,list):
            for i in range(len(fault_num)):
                fault_count=0
                while fault_count<fault_num[i]:
                    coordinate=coordinate_gen_fmap(data_shape[i],
                                                   batch_size,
                                                   distribution=coor_distribution,
                                                   poisson_lam=coor_pois_lam,
                                                   mean=coor_mean,
                                                   std=coor_std,
                                                   concentration=concentration,
                                                   **kwargs)
                    fault_bit=fault_bit_loc_gen(model_word_length,
                                                distribution=bit_loc_distribution,
                                                poisson_lam=bit_loc_pois_lam,
                                                mean=bit_loc_mean,
                                                std=bit_loc_std,
                                                **kwargs)
                    if coordinate is None:
                        break

                    if coordinate in fault_dict[i].keys():
                        if isinstance(fault_dict[i][coordinate]['SA_bit'],list):
                            if fault_bit in fault_dict[i][coordinate]['SA_bit']:
                                #print('error 1')
                                continue
                            else:
                                fault_dict[i][coordinate]['SA_type'].append(fault_type)
                                fault_dict[i][coordinate]['SA_bit'].append(fault_bit)
                                fault_count += 1
                        else:
                            if fault_bit == fault_dict[i][coordinate]['SA_bit']:
                                #print('error 2')
                                continue
                            else:
                                fault_dict[i][coordinate]['SA_type']=[fault_dict[i][coordinate]['SA_type'],fault_type]
                                fault_dict[i][coordinate]['SA_bit']=[fault_dict[i][coordinate]['SA_bit'],fault_bit]
                                fault_count += 1
                    else:
                        fault_dict[i][coordinate]={'SA_type':fault_type,
                                                      'SA_bit' : fault_bit}
                        fault_count += 1
        else: 
            while fault_count<fault_num:
                coordinate=coordinate_gen_fmap(data_shape,
                                               batch_size,
                                               distribution=coor_distribution,
                                               poisson_lam=coor_pois_lam,
                                               mean=coor_mean,
                                               std=coor_std,
                                               concentration=concentration,
                                               **kwargs)
                fault_bit=fault_bit_loc_gen(model_word_length,
                                            distribution=bit_loc_distribution,
                                            poisson_lam=bit_loc_pois_lam,
                                            mean=coor_mean,
                                            std=coor_std,
                                            **kwargs)
                
                if coordinate is None:
                        break
                
                if coordinate in fault_dict.keys():
                    if isinstance(fault_dict[coordinate]['SA_bit'],list):
                        if fault_bit in fault_dict[coordinate]['SA_bit']:
                            continue
                        else:
                            fault_dict[coordinate]['SA_type'].append(fault_type)
                            fault_dict[coordinate]['SA_bit'].append(fault_bit)
                            fault_count += 1
                    else:
                        if fault_bit == fault_dict[coordinate]['SA_bit']:
                            continue
                        else:
                            fault_dict[coordinate]['SA_type']=[fault_dict[coordinate]['SA_type'],fault_type]
                            fault_dict[coordinate]['SA_bit']=[fault_dict[coordinate]['SA_bit'],fault_bit]
                            fault_count += 1
                else:
                    fault_dict[coordinate]={'SA_type':fault_type,
                                              'SA_bit' : fault_bit}
                    fault_count += 1
        
    return fault_dict,fault_num
    
def gen_fault_dict_list_wght(data_shape,
                             fault_rate,
                             model_word_length,
                             fault_num=None,
                             fast_gen=False,
                             return_modulator=False,
                             coor_distribution='uniform',
                             coor_pois_lam=None,
                             coor_mean=None,
                             coor_std=None,
                             concentration=None,
                             bit_loc_distribution='uniform',
                             bit_loc_pois_lam=None,
                             bit_loc_mean=None, 
                             bit_loc_std=None,
                             fault_type='flip',
                             **kwargs):
    """Generate the fault dictionary list of a feature map base on its shape and with specific distibution type.

    # Arguments
        data_shape: Tuple. The shape of weights.
        fault_rate: Float. The probability of fault occurance in weights.
        batch_size: Integer. The batch size of fault tolerance evaluation process.
        model_word_length: Integer. The word length of model parameters.
        fault_num: List of integer. The number of faults in [kernel,bias] respectively.
        fast_gen: Bool. Use fast generation or not. Fast generation doesn't have multiple fault in single parameter, thus the fault_num maybe inaccurate.
        return_modulator: Bool. Return fault modulator or not. Return fault modulator in fault list generation phase. Further improve generation time. Only available when the fast_gen is True.
        coor_distribution: String. The distribution type of coordinate in weights. Must be one of 'uniform', 'poisson', 'normal'.
        coor_pois_lam: Tuple. The lambda of poisson distribution of weights coordinate.
        coor_mean: Tuple or Float. The mean value for normal or center distribution in coordinate. Tuple for normal distribution. Float for concentration of center distribution.
        coor_std: Tuple or Float. The standard deviation value for normal or center distribution in coordinate. Tuple for normal distribution. Float for concentration of center distribution.
        concentration: Float. A float number between 1 and 0. 1 means set the lambda of 2D feature map to center of image, 0 means set to corner of the feature map. (can only be use in conv 2D layer)
        bit_loc_distribution: String. The distribution type of locaton in parameters. Must be one of 'uniform', 'poisson', 'normal'.
        bit_loc_pois_lam: Integer. The lambda of poisson distribution.
        bit_loc_mean: Float. The mean value for normal or center distribution in bit location. 
        bit_loc_std: Float. The standard deviation value for normal or center distribution in bit location.
        fault_type: String. The type of fault.

    # Returns
        The fault information Dictionary. The number of fault generated Integer.
    """

    fault_count=0        
    fault_dict=[dict() for _ in range(len(data_shape))]
    if fault_num is None:
        fault_num=fault_num_gen_wght(data_shape,fault_rate,model_word_length)
            
    if fast_gen:
        for i in range(len(fault_num)):
            coordinate=coordinate_gen_wght_fast(data_shape[i],
                                                fault_num[i],
                                                distribution=coor_distribution,
                                                poisson_lam=coor_pois_lam,
                                                mean=coor_mean,
                                                std=coor_std,
                                                concentration=concentration,
                                                **kwargs)
            fault_bit=fault_bit_loc_gen_fast(model_word_length,
                                             fault_num[i],
                                             distribution=bit_loc_distribution,
                                             poisson_lam=bit_loc_pois_lam,
                                             mean=bit_loc_mean,
                                             std=bit_loc_std,
                                             **kwargs)
            if coordinate is not None:
                if return_modulator:
                    tensor_modulator0=None
                    tensor_modulator1=None
                    tensor_modulatorF=None
                    modulator=generate_stuck_at_fault_modulator_fast(data_shape[i],coordinate,fault_type,fault_bit)
                    if fault_type == '0':
                        tensor_modulator0=modulator
                    elif fault_type == '1':
                        tensor_modulator1=modulator
                    elif fault_type == 'flip':
                        tensor_modulatorF=modulator
                    fault_dict[i]=[tensor_modulator0,tensor_modulator1,tensor_modulatorF]
                else:
                    fault_bit=[{'SA_type':fault_type,'SA_bit':bit} for bit in fault_bit]
                    fault_dict[i]=dict(zip(coordinate,fault_bit))
    else:
        for i in range(len(fault_num)):
            fault_count=0
            while fault_count<fault_num[i]:
                coordinate=coordinate_gen_wght(data_shape[i],
                                               distribution=coor_distribution,
                                               poisson_lam=coor_pois_lam,
                                               concentration=concentration,
                                               mean=coor_mean,
                                               std=coor_std,
                                               **kwargs)
                fault_bit=fault_bit_loc_gen(model_word_length,
                                            distribution=bit_loc_distribution,
                                            poisson_lam=bit_loc_pois_lam,
                                            mean=bit_loc_mean,
                                            std=bit_loc_std,
                                            **kwargs)
                if coordinate is None:
                        break

                if coordinate in fault_dict[i].keys():
                    if isinstance(fault_dict[i][coordinate]['SA_bit'],list):
                        if fault_bit in fault_dict[i][coordinate]['SA_bit']:
                            #print('error 1')
                            continue
                        else:
                            fault_dict[i][coordinate]['SA_type'].append(fault_type)
                            fault_dict[i][coordinate]['SA_bit'].append(fault_bit)
                            fault_count += 1
                    else:
                        if fault_bit == fault_dict[i][coordinate]['SA_bit']:
                            #print('error 2')
                            continue
                        else:
                            fault_dict[i][coordinate]['SA_type']=[fault_dict[i][coordinate]['SA_type'],fault_type]
                            fault_dict[i][coordinate]['SA_bit']=[fault_dict[i][coordinate]['SA_bit'],fault_bit]
                            fault_count += 1
                else:
                    fault_dict[i][coordinate]={'SA_type':fault_type,
                                                  'SA_bit' : fault_bit}
                    fault_count += 1
        
    return fault_dict,fault_num

def generate_layer_stuck_fault(layer,
                               fault_rate,
                               batch_size,
                               model_word_length,
                               fault_num=None,
                               param_filter=[True,True,True],
                               fast_gen=False,
                               return_modulator=False,
                               coor_distribution='uniform',
                               coor_pois_lam=None,
                               coor_mean=None,
                               coor_std=None,
                               concentration=None,
                               bit_loc_distribution='uniform',
                               bit_loc_pois_lam=None,
                               bit_loc_mean=None, 
                               bit_loc_std=None,
                               fault_type='flip',
                               print_detail=True,
                               **kwargs):
    """Generate the fault dictionary list of a layer base on its shape and with specific distibution type.

    # Arguments
        layer: Keras.Layer. 
        fault_rate: Float. The probability of fault occurance in a layer.
        batch_size: Integer. The batch size of fault tolerance evaluation process.
        model_word_length: Integer. The word length of model parameters.
        fault_num: List of integer. The number of faults in [input,weight,output] respectively.
        fast_gen: Bool. Use fast generation or not. Fast generation doesn't have multiple fault in single parameter, thus the fault_num maybe inaccurate.
        return_modulator: Bool. Return fault modulator or not. Return fault modulator in fault list generation phase. Further improve generation time. Only available when the fast_gen is True.
        coor_distribution: String. The distribution type of coordinate in parameters. Must be one of 'uniform', 'poisson', 'normal'.
        coor_pois_lam: List of Tuple. The lambda of poisson distribution of parameters coordinate.
        coor_mean: List of (Tuple or Float). The mean value for normal or center distribution in coordinate. Double List of Tuple for normal distribution. List of Float for concentration of center distribution.
        coor_std: List of (Tuple or Float). The standard deviation value for normal or center distribution in coordinate. Double List of Tuple for normal distribution. List of Float for concentration of center distribution.
        concentration: List of Float. A list of float number between 1 and 0 [input, weight, output]. 1 means set the lambda of 2D feature map to center of image, 0 means set to corner of the feature map. (can only be use in conv 2D layer)
        bit_loc_distribution: String. The distribution type of locaton in parameters. Must be one of 'uniform', 'poisson', 'normal'.
        bit_loc_pois_lam: Integer. The lambda of poisson distribution.
        bit_loc_mean: List of Float. The mean value for normal or center distribution in bit location. 
        bit_loc_std: List of Float. The standard deviation value for normal or center distribution in bit location.
        fault_type: String. The type of fault.
        print_detail: Bool. Print generation detail or not.

    # Returns
        The fault information Dictionary. The number of fault generated Integer.
    """

    if coor_pois_lam is None:
        coor_pois_lam=[None,None,None]
    
    if coor_distribution=='poisson':
        if not isinstance(coor_pois_lam,list) or len(coor_pois_lam)!=3:
            raise ValueError('Poisson distribution lambda setting must be a list has the length of 3 (ifmap, ofmap, weight).')
            
    if fault_num is None:
        fault_num=[None,None,None]
    if not isinstance(fault_num,list) or len(fault_num) != 3:
        raise ValueError('Fault num must be a list with lenth 3 and each item is an integer that indicates the number of faults in [input,weight,output] respectively.')
        
    if concentration is None:
        concentration=[None,None,None]
    if not isinstance(concentration,list) or len(concentration) != 3:
        raise ValueError('Concentration must be a list with lenth 3 and each item is a float that indicates the level of concentration faults in [input,weight,output] respectively.')

    layer_input_shape=layer.input_shape
    layer_output_shape=layer.output_shape
    layer_weight_shape=[weight_shape.shape for weight_shape in layer.get_weights()]
    
    if len(layer_weight_shape)==0:
        if print_detail:
            print('    no weight layer Skipped!')
        return None, None, [None,None]
    
    if not isinstance(model_word_length,list):
        model_word_length=[model_word_length,model_word_length,model_word_length]

    
    # ifmap fault generation
    if param_filter[0]:
        ifmap_fault_dict,layer_ifmap_fault_num=gen_fault_dict_list_fmap(layer_input_shape,
                                                                        fault_rate,
                                                                        batch_size,
                                                                        model_word_length[0],
                                                                        fault_num=fault_num[0],
                                                                        fast_gen=fast_gen,
                                                                        return_modulator=return_modulator,
                                                                        coor_distribution=coor_distribution,
                                                                        coor_pois_lam=coor_pois_lam[0],
                                                                        coor_mean=coor_mean[0],
                                                                        coor_std=coor_std[0],
                                                                        concentration=concentration[0],
                                                                        bit_loc_distribution=bit_loc_distribution,
                                                                        bit_loc_pois_lam=bit_loc_pois_lam,
                                                                        bit_loc_mean=bit_loc_mean,
                                                                        bit_loc_std=bit_loc_std,
                                                                        fault_type=fault_type,
                                                                        **kwargs)
    else:
        ifmap_fault_dict=None
        layer_ifmap_fault_num=0
        
    if print_detail:
        print('    generated layer ifmap %s faults'%(str(layer_ifmap_fault_num)))
    
    
    # ofmap fault generation
    if param_filter[2]:
        ofmap_fault_dict,layer_ofmap_fault_num=gen_fault_dict_list_fmap(layer_output_shape,
                                                                        fault_rate,
                                                                        batch_size,
                                                                        model_word_length[2],
                                                                        fault_num=fault_num[2],
                                                                        fast_gen=fast_gen,
                                                                        return_modulator=return_modulator,
                                                                        coor_distribution=coor_distribution,
                                                                        coor_pois_lam=coor_pois_lam[2],
                                                                        coor_mean=coor_mean[2],
                                                                        coor_std=coor_std[2],
                                                                        concentration=concentration[2],
                                                                        bit_loc_distribution=bit_loc_distribution,
                                                                        bit_loc_pois_lam=bit_loc_pois_lam,
                                                                        bit_loc_mean=bit_loc_mean,
                                                                        bit_loc_std=bit_loc_std,
                                                                        fault_type=fault_type,
                                                                        **kwargs)
    else:
        ofmap_fault_dict=None
        layer_ofmap_fault_num=0
        
    if print_detail:
        print('    generated layer ofmap %s faults'%(str(layer_ofmap_fault_num)))
    
    # weight fault generation
    if param_filter[1]:
        weight_fault_dict,layer_weight_fault_num=gen_fault_dict_list_wght(layer_weight_shape,
                                                                          fault_rate,
                                                                          model_word_length[1],
                                                                          fault_num=fault_num[1],
                                                                          fast_gen=fast_gen,
                                                                          return_modulator=return_modulator,
                                                                          coor_distribution=coor_distribution,
                                                                          coor_pois_lam=coor_pois_lam[1],
                                                                          coor_mean=coor_mean[1],
                                                                          coor_std=coor_std[1],
                                                                          concentration=concentration[1],
                                                                          bit_loc_distribution=bit_loc_distribution,
                                                                          bit_loc_pois_lam=bit_loc_pois_lam,
                                                                          bit_loc_mean=bit_loc_mean,
                                                                          bit_loc_std=bit_loc_std,
                                                                          fault_type=fault_type,
                                                                          **kwargs)
    else:
        weight_fault_dict=[None,None]
        layer_weight_fault_num=0
        
    if print_detail:
        print('    generated layer weight %s faults'%(str(layer_weight_fault_num)))
            
    return ifmap_fault_dict, ofmap_fault_dict, weight_fault_dict


def generate_model_stuck_fault(model,
                               fault_rate,
                               batch_size,
                               model_word_length,
                               layer_wise=False,
                               param_filter=[True,True,True],
                               fast_gen=False,
                               return_modulator=False,
                               coor_distribution='uniform',
                               coor_pois_lam=None,
                               coor_mean=None,
                               coor_std=None,
                               concentration=None,
                               bit_loc_distribution='uniform',
                               bit_loc_pois_lam=None,
                               bit_loc_mean=None, 
                               bit_loc_std=None,
                               fault_type='flip',
                               print_detail=True,
                               layer_gen_list=None,
                               **kwargs):
    """Generate the fault dictionary list of a model base on its shape and with specific distibution type.

    # Arguments
        model: Keras.Model. 
        fault_rate: Float. The probability of fault occurance in a layer.
        batch_size: Integer. The batch size of fault tolerance evaluation process.
        model_word_length: Integer. The word length of model parameters.
        layer_wise: Bool. If true, generate fault lists on each layer individually.
        param_filter: List of Bools. The indicator for generate fault on ifmap, weight, ofmap individually. [input,weight,output]
        fast_gen: Bool. Use fast generation or not. Fast generation doesn't have multiple fault in single parameter, thus the fault_num maybe inaccurate.
        return_modulator: Bool. Return fault modulator or not. Return fault modulator in fault list generation phase. Further improve generation time. Only available when the fast_gen is True.
        coor_distribution: String. The distribution type of coordinate in parameters. Must be one of 'uniform', 'poisson', 'normal'.
        coor_pois_lam: Double List of Tuple. The lambda of poisson distribution of parameters coordinate.
        coor_mean: Double List of (Tuple or Float). The mean value for normal or center distribution in coordinate. Double List of Tuple for normal distribution. Double List of Float for concentration of center distribution.
        coor_std: Double List of (Tuple or Float). The standard deviation value for normal or center distribution in coordinate. Double List of Tuple for normal distribution. Double List of Float for concentration of center distribution.
        concentration: Double List of Float. A list of float number between 1 and 0 [layer[input, weight, output]]. 1 means set the lambda of 2D feature map to center of image, 0 means set to corner of the feature map. (can only be use in conv 2D layer)
        bit_loc_distribution: String. The distribution type of locaton in parameters. Must be one of 'uniform', 'poisson', 'normal'.
        bit_loc_pois_lam: Integer. The lambda of poisson distribution.
        bit_loc_mean: Double List of Float. The mean value for normal or center distribution in bit location. 
        bit_loc_std: Double List of Float. The standard deviation value for normal or center distribution in bit location.
        fault_type: String. The type of fault.
        print_detail: Bool. Print generation detail or not.
        layer_gen_list: List of Integer. The list of indexes for specific layer wanted to generate fault

    # Returns
        The fault information Dictionary List.
    """

    model_depth=len(model.layers)
    model_ifmap_fault_dict_list=[None for _ in range(model_depth)]
    model_ofmap_fault_dict_list=[None for _ in range(model_depth)]
    model_weight_fault_dict_list=[[None,None] for _ in range(model_depth)]

    if layer_gen_list is not None:
        layer_wise=True

    if not layer_wise:
        ifmap_fault_num_list,ofmap_fault_num_list,weight_fault_num_list,_,_,_=fault_num_gen_model(model,fault_rate,batch_size,model_word_length)  
        
    if coor_pois_lam is None:
        coor_pois_lam=[[None,None,None]]
    if coor_distribution=='poisson':
        if not isinstance(coor_pois_lam,list) or len(coor_pois_lam)!=model_depth:
            raise ValueError('Poisson distribution lambda setting must be a list has the same length as model layer number.')
    
    if not isinstance(coor_mean,list):
        coor_mean=[[coor_mean,coor_mean,coor_mean]]
    if coor_distribution in ['normal','center']:
        if not isinstance(coor_mean,list) or not (len(coor_mean)==model_depth or len(coor_mean)==1):
            raise ValueError('Normal or center distribution mean setting must be a list has the same length as model layer number or 1.')

    if not isinstance(coor_std,list):
        coor_std=[[coor_std,coor_std,coor_std]]
    if coor_distribution in ['normal','center']:
        if not isinstance(coor_std,list) or not (len(coor_std)==model_depth or len(coor_std)==1):
            raise ValueError('Normal or center distribution std setting must be a list has the same length as model layer number or 1.')

    if not isinstance(concentration,list):
        concentration=[[concentration,concentration,concentration]]
    if coor_distribution=='center':
        if not isinstance(concentration,list) or not (len(concentration)==model_depth or len(concentration)==1):
            raise ValueError('Center distribution concentration setting must be a list has the same length as model layer number or 1.')

    
    if layer_gen_list is None:
        layer_iter=range(1,model_depth)
    elif isinstance(layer_gen_list,list):
        layer_iter=layer_gen_list
    else:
        raise TypeError('layer_gen_list must be None (generate fault on every layer) or List of Integers (the list of indexes for specific layer wanted to generate fault).')
    
    for layer_num in layer_iter:
        if print_detail:
            print('\nGenerating fault on layer %d ...'%layer_num)
            
        
        if coor_distribution == 'poisson':
            lam_index=layer_num
        else:
            lam_index=0
            
        if len(concentration)==1:
            concen_index=0
        else:
            concen_index=layer_num
            
        if len(coor_mean)==1:
            mean_index=0
        else:
            mean_index=layer_num
            
        if len(coor_std)==1:
            std_index=0
        else:
            std_index=layer_num
            
                    
        layer_weight_shape=[weight_shape.shape for weight_shape in model.layers[layer_num].get_weights()]
        
        if len(layer_weight_shape)==0:
            if print_detail:
                print('    no weight layer Skipped!')
            continue
        
        if layer_wise:
            fault_num=[None,None,None]
        else:
            fault_num=[ifmap_fault_num_list[layer_num],weight_fault_num_list[layer_num],ofmap_fault_num_list[layer_num]]
        
        model_ifmap_fault_dict_list[layer_num],model_ofmap_fault_dict_list[layer_num],model_weight_fault_dict_list[layer_num]\
        =generate_layer_stuck_fault(model.layers[layer_num],
                                    fault_rate,
                                    batch_size,
                                    model_word_length,
                                    fault_num=fault_num,
                                    param_filter=param_filter,
                                    fast_gen=fast_gen,
                                    return_modulator=return_modulator,
                                    coor_distribution=coor_distribution,
                                    coor_pois_lam=coor_pois_lam[lam_index],
                                    coor_mean=coor_mean[mean_index],
                                    coor_std=coor_std[std_index],
                                    concentration=concentration[concen_index],
                                    bit_loc_distribution=bit_loc_distribution,
                                    bit_loc_pois_lam=bit_loc_pois_lam,
                                    bit_loc_mean=bit_loc_mean,
                                    bit_loc_std=bit_loc_std,
                                    fault_type=fault_type,
                                    print_detail=print_detail,
                                    **kwargs)
        
        if print_detail:
            print('    layer %d Done!'%layer_num)
        
    return model_ifmap_fault_dict_list, model_ofmap_fault_dict_list, model_weight_fault_dict_list

