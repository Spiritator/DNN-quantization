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

def coordinate_gen_fmap(data_shape,batch_size,distribution='uniform',poisson_lam=None):
    coordinate=list()
    
    if distribution=='uniform':
        coordinate.append(np.random.randint(batch_size))
        for j in range(1,len(data_shape)):
            coordinate.append(np.random.randint(data_shape[j]))
    elif distribution=='poisson':
        if not isinstance(poisson_lam,tuple) or len(poisson_lam)!=len(data_shape):
            raise TypeError('Poisson distribution lambda setting must be a tuple same length as input data shape which indicates the lamda of poisson distribution.')
        
        if isinstance(poisson_lam[0],int) and poisson_lam[0]>=0 and poisson_lam[0]<batch_size:
            coor_tmp=np.random.poisson(poisson_lam[j])
            while coor_tmp>=batch_size:
                coor_tmp=np.random.poisson(poisson_lam[j])
            coordinate.append(coor_tmp) 
        for j in range(1,len(data_shape)):
            if isinstance(poisson_lam[j],int) and poisson_lam[j]>=0 and poisson_lam[j]<data_shape[j]:
                coor_tmp=np.random.poisson(poisson_lam[j])
                while coor_tmp>=data_shape[j]:
                    coor_tmp=np.random.poisson(poisson_lam[j])
                coordinate.append(coor_tmp)
        else:
            raise ValueError('Poisson distribution Lambda must within feature map shape. Feature map shape %s but got lambda input %s'%(str(data_shape),str(poisson_lam)))
    elif distribution=='normal':
        pass 
        '''TO BE DONE'''
    else:
        raise NameError('Invalid type of random generation distribution. Please choose between uniform, poisson, normal.')
    
    coordinate=tuple(coordinate)
    return coordinate

def coordinate_gen_wght(data_shape,distribution='uniform',poisson_lam=None):
    coordinate=list()
    
    if distribution=='uniform':
        for j in range(len(data_shape)):
            coordinate.append(np.random.randint(data_shape[j]))
    elif distribution=='poisson':
        if not isinstance(poisson_lam,tuple) or len(poisson_lam)!=len(data_shape):
            raise TypeError('Poisson distribution lambda setting must be a tuple same length as input data shape which indicates the lamda of poisson distribution.')
        
        for j in range(len(data_shape)):
            if isinstance(poisson_lam,int) and poisson_lam>=0 and poisson_lam<data_shape[j]:
                coor_tmp=np.random.poisson(poisson_lam[j])
                while coor_tmp>=data_shape[j]:
                    coor_tmp=np.random.poisson(poisson_lam[j])
                coordinate.append(coor_tmp)
        else:
            raise ValueError('Poisson distribution Lambda must within feature map shape. Feature map shape %s but got lambda input %s'%(str(data_shape),str(poisson_lam)))
    elif distribution=='normal':
        pass 
        '''TO BE DONE'''   
    else:
        raise NameError('Invalid type of random generation distribution. Please choose between uniform, poisson, normal.')
    
    coordinate=tuple(coordinate)
    return coordinate

def fault_bit_loc_gen(model_word_length,distribution='uniform',poisson_lam=None):
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
        pass    
    else:
        raise NameError('Invalid type of random generation distribution. Please choose between uniform, poisson, normal.')
    return fault_bit

def fault_num_gen_fmap(data_shape,fault_rate,batch_size,model_word_length):
    fault_num=int(np.prod(data_shape[1:]) * batch_size * model_word_length * fault_rate)
    return fault_num

def fault_num_gen_wght(data_shape,fault_rate,model_word_length):
    fault_num=[int(np.prod(shapes) * model_word_length * fault_rate) for shapes in data_shape]
    return fault_num 

def gen_fault_dict_list_fmap(data_shape,fault_rate,batch_size,model_word_length,coor_distribution='uniform',coor_pois_lam=None,bit_loc_distribution='uniform',bit_loc_pois_lam=None,**kwargs):
    fault_count=0        
    fault_dict=dict()
    fault_num=fault_num_gen_fmap(data_shape,fault_rate,batch_size,model_word_length)
    
    while fault_count<fault_num:
        coordinate=coordinate_gen_fmap(data_shape,batch_size,distribution=coor_distribution,poisson_lam=coor_pois_lam,**kwargs)
        fault_bit=fault_bit_loc_gen(model_word_length,distribution=bit_loc_distribution,poisson_lam=bit_loc_pois_lam,**kwargs)
        
        if coordinate in fault_dict.keys():
            if isinstance(fault_dict[coordinate]['SA_bit'],list):
                if fault_bit in fault_dict[coordinate]['SA_bit']:
                    continue
                else:
                    fault_dict[coordinate]['SA_type'].append('flip')
                    fault_dict[coordinate]['SA_bit'].append(fault_bit)
                    fault_count += 1
            else:
                if fault_bit == fault_dict[coordinate]['SA_bit']:
                    continue
                else:
                    fault_dict[coordinate]['SA_type']=[fault_dict[coordinate]['SA_type'],'flip']
                    fault_dict[coordinate]['SA_bit']=[fault_dict[coordinate]['SA_bit'],fault_bit]
                    fault_count += 1
        else:
            fault_dict[coordinate]={'SA_type':'flip',
                                          'SA_bit' : fault_bit}
            fault_count += 1
        
    return fault_dict,fault_num
    
def gen_fault_dict_list_wght(data_shape,fault_rate,model_word_length,coor_distribution='uniform',coor_pois_lam=None,bit_loc_distribution='uniform',bit_loc_pois_lam=None,**kwargs):
    fault_count=0        
    fault_dict=[dict() for i in range(len(data_shape))]
    fault_num=fault_num_gen_wght(data_shape,fault_rate,model_word_length)
            
    for i in range(len(fault_num)):
        fault_count=0
        while fault_count<fault_num[i]:
            coordinate=coordinate_gen_wght(data_shape[i],distribution=coor_distribution,poisson_lam=coor_pois_lam,**kwargs)
            fault_bit=fault_bit_loc_gen(model_word_length,distribution=bit_loc_distribution,poisson_lam=bit_loc_pois_lam,**kwargs)
            
            if coordinate in fault_dict[i].keys():
                if isinstance(fault_dict[i][coordinate]['SA_bit'],list):
                    if fault_bit in fault_dict[i][coordinate]['SA_bit']:
                        #print('error 1')
                        continue
                    else:
                        fault_dict[i][coordinate]['SA_type'].append('flip')
                        fault_dict[i][coordinate]['SA_bit'].append(fault_bit)
                        fault_count += 1
                else:
                    if fault_bit == fault_dict[i][coordinate]['SA_bit']:
                        #print('error 2')
                        continue
                    else:
                        fault_dict[i][coordinate]['SA_type']=[fault_dict[i][coordinate]['SA_type'],'flip']
                        fault_dict[i][coordinate]['SA_bit']=[fault_dict[i][coordinate]['SA_bit'],fault_bit]
                        fault_count += 1
            else:
                fault_dict[i][coordinate]={'SA_type':'flip',
                                              'SA_bit' : fault_bit}
                fault_count += 1
        
    return fault_dict,fault_num

def generate_model_stuck_fault(model,fault_rate,batch_size,model_word_length,coor_distribution='uniform',coor_pois_lam=None,bit_loc_distribution='uniform',bit_loc_pois_lam=None,**kwargs):
    model_depth=len(model.layers)
    model_ifmap_fault_dict_list=[None]
    model_ofmap_fault_dict_list=[None]
    model_weight_fault_dict_list=[[None,None]]
    if coor_pois_lam is None:
        coor_pois_lam=[[None,None,None]]
    
    if coor_distribution=='poisson':
        if not isinstance(coor_pois_lam,list) or len(coor_pois_lam)!=model_depth:
            raise ValueError('Poisson distribution lambda setting must be a list has the same length as model layer number.')
    
    for layer_num in range(1,model_depth):
        print('\nGenerating fault on layer %d ...'%layer_num)
        
        if coor_distribution == 'poisson':
            lam_index=layer_num
        else:
            lam_index=0
        
        layer_input_shape=model.layers[layer_num].input_shape
        layer_output_shape=model.layers[layer_num].output_shape
        layer_weight_shape=[weight_shape.shape for weight_shape in model.layers[layer_num].get_weights()]
        
        if len(layer_weight_shape)==0:
            model_ifmap_fault_dict_list.append(None)
            model_ofmap_fault_dict_list.append(None)
            model_weight_fault_dict_list.append([None,None])
            print('    no weight layer Skipped!')
            continue
        
        # ifmap fault generation
        ifmap_fault_dict,layer_ifmap_fault_num=gen_fault_dict_list_fmap(layer_input_shape,
                                                                        fault_rate,
                                                                        batch_size,
                                                                        model_word_length,
                                                                        coor_distribution=coor_distribution,
                                                                        coor_pois_lam=coor_pois_lam[lam_index][0],
                                                                        bit_loc_distribution=bit_loc_distribution,
                                                                        bit_loc_pois_lam=bit_loc_pois_lam,
                                                                        **kwargs)
        model_ifmap_fault_dict_list.append(ifmap_fault_dict)    
        print('    generated layer %d ifmap %d faults'%(layer_num,layer_ifmap_fault_num))
        
        
        # ofmap fault generation
        ofmap_fault_dict,layer_ofmap_fault_num=gen_fault_dict_list_fmap(layer_output_shape,
                                                                        fault_rate,
                                                                        batch_size,
                                                                        model_word_length,
                                                                        coor_distribution=coor_distribution,
                                                                        coor_pois_lam=coor_pois_lam[lam_index][2],
                                                                        bit_loc_distribution=bit_loc_distribution,
                                                                        bit_loc_pois_lam=bit_loc_pois_lam,
                                                                        **kwargs)
        model_ofmap_fault_dict_list.append(ofmap_fault_dict)    
        print('    generated layer %d ofmap %d faults'%(layer_num,layer_ofmap_fault_num))
        
        # weight fault generation
        weight_fault_dict,layer_weight_fault_num=gen_fault_dict_list_wght(layer_weight_shape,
                                                                          fault_rate,
                                                                          model_word_length,
                                                                          coor_distribution=coor_distribution,
                                                                          coor_pois_lam=coor_pois_lam[lam_index][1],
                                                                          bit_loc_distribution=bit_loc_distribution,
                                                                          bit_loc_pois_lam=bit_loc_pois_lam,
                                                                          **kwargs)
        model_weight_fault_dict_list.append(weight_fault_dict)    
        print('    generated layer %d weight %s faults'%(layer_num,str(layer_weight_fault_num)))
        
        print('    layer %d Done!'%layer_num)
        
    return model_ifmap_fault_dict_list, model_ofmap_fault_dict_list, model_weight_fault_dict_list

    
# old function to be abandoned
def generate_model_random_stuck_fault(model,fault_rate,batch_size,model_word_length):
    model_depth=len(model.layers)
    model_ifmap_fault_dict_list=[None]
    model_ofmap_fault_dict_list=[None]
    model_weight_fault_dict_list=[[None,None]]
    for layer_num in range(1,model_depth):
        print('\nGenerating fault on layer %d ...'%layer_num)
        
        layer_input_shape=model.layers[layer_num].input_shape
        layer_output_shape=model.layers[layer_num].output_shape
        layer_weight_shape=[weight_shape.shape for weight_shape in model.layers[layer_num].get_weights()]
        
        if len(layer_weight_shape)==0:
            model_ifmap_fault_dict_list.append(None)
            model_ofmap_fault_dict_list.append(None)
            model_weight_fault_dict_list.append([None,None])
            print('    no weight layer Skipped!')
            continue
        
        ifmap_fault_dict=dict()
        ofmap_fault_dict=dict()
        weight_fault_dict=[dict() for i in range(len(layer_weight_shape))]
        
        layer_ifmap_fault_num=int(np.prod(layer_input_shape[1:]) * batch_size * model_word_length * fault_rate)
        layer_ofmap_fault_num=int(np.prod(layer_output_shape[1:]) * batch_size * model_word_length * fault_rate)
        layer_weight_fault_num=[int(np.prod(shapes) * model_word_length * fault_rate) for shapes in layer_weight_shape]
        
        
        # ifmap fault generation
        fault_count=0
        while fault_count<layer_ifmap_fault_num:
            coordinate=list()
            coordinate.append(np.random.randint(batch_size))
            for j in range(1,len(layer_input_shape)):
                coordinate.append(np.random.randint(layer_input_shape[j]))
            coordinate=tuple(coordinate)
            fault_bit=np.random.randint(model_word_length)
            
            if coordinate in ifmap_fault_dict.keys():
                if isinstance(ifmap_fault_dict[coordinate]['SA_bit'],list):
                    if fault_bit in ifmap_fault_dict[coordinate]['SA_bit']:
                        continue
                    else:
                        ifmap_fault_dict[coordinate]['SA_type'].append('flip')
                        ifmap_fault_dict[coordinate]['SA_bit'].append(fault_bit)
                        fault_count += 1
                else:
                    if fault_bit == ifmap_fault_dict[coordinate]['SA_bit']:
                        continue
                    else:
                        ifmap_fault_dict[coordinate]['SA_type']=[ifmap_fault_dict[coordinate]['SA_type'],'flip']
                        ifmap_fault_dict[coordinate]['SA_bit']=[ifmap_fault_dict[coordinate]['SA_bit'],fault_bit]
                        fault_count += 1
            else:
                ifmap_fault_dict[coordinate]={'SA_type':'flip',
                                              'SA_bit' : fault_bit}
                fault_count += 1
        
        model_ifmap_fault_dict_list.append(ifmap_fault_dict)    
        print('    generated layer %d ifmap %d faults'%(layer_num,layer_ifmap_fault_num))
        
        
        # ofmap fault generation
        fault_count=0
        while fault_count<layer_ofmap_fault_num:
            coordinate=list()
            coordinate.append(np.random.randint(batch_size))
            for j in range(1,len(layer_output_shape)):
                coordinate.append(np.random.randint(layer_output_shape[j]))
            coordinate=tuple(coordinate)
            fault_bit=np.random.randint(model_word_length)
            
            if coordinate in ofmap_fault_dict.keys():
                if isinstance(ofmap_fault_dict[coordinate]['SA_bit'],list):
                    if fault_bit in ofmap_fault_dict[coordinate]['SA_bit']:
                        continue
                    else:
                        ofmap_fault_dict[coordinate]['SA_type'].append('flip')
                        ofmap_fault_dict[coordinate]['SA_bit'].append(fault_bit)
                        fault_count += 1
                else:
                    if fault_bit == ofmap_fault_dict[coordinate]['SA_bit']:
                        continue
                    else:
                        ofmap_fault_dict[coordinate]['SA_type']=[ofmap_fault_dict[coordinate]['SA_type'],'flip']
                        ofmap_fault_dict[coordinate]['SA_bit']=[ofmap_fault_dict[coordinate]['SA_bit'],fault_bit]
                        fault_count += 1
            else:
                ofmap_fault_dict[coordinate]={'SA_type':'flip',
                                              'SA_bit' : fault_bit}
                fault_count += 1
        
        model_ofmap_fault_dict_list.append(ofmap_fault_dict)    
        print('    generated layer %d ofmap %d faults'%(layer_num,layer_ofmap_fault_num))
        
        # weight fault generation
        for i in range(len(layer_weight_fault_num)):
            fault_count=0
            while fault_count<layer_weight_fault_num[i]:
                coordinate=list()
                for j in range(len(layer_weight_shape[i])):
                    coordinate.append(np.random.randint(layer_weight_shape[i][j]))
                coordinate=tuple(coordinate)
                fault_bit=np.random.randint(model_word_length)
                
                if coordinate in weight_fault_dict[i].keys():
                    if isinstance(weight_fault_dict[i][coordinate]['SA_bit'],list):
                        if fault_bit in weight_fault_dict[i][coordinate]['SA_bit']:
                            #print('error 1')
                            continue
                        else:
                            weight_fault_dict[i][coordinate]['SA_type'].append('flip')
                            weight_fault_dict[i][coordinate]['SA_bit'].append(fault_bit)
                            fault_count += 1
                    else:
                        if fault_bit == weight_fault_dict[i][coordinate]['SA_bit']:
                            #print('error 2')
                            continue
                        else:
                            weight_fault_dict[i][coordinate]['SA_type']=[weight_fault_dict[i][coordinate]['SA_type'],'flip']
                            weight_fault_dict[i][coordinate]['SA_bit']=[weight_fault_dict[i][coordinate]['SA_bit'],fault_bit]
                            fault_count += 1
                else:
                    weight_fault_dict[i][coordinate]={'SA_type':'flip',
                                                  'SA_bit' : fault_bit}
                    fault_count += 1
            
        model_weight_fault_dict_list.append(weight_fault_dict)    
        print('    generated layer %d weight %s faults'%(layer_num,str(layer_weight_fault_num)))
        
        print('    layer %d Done!'%layer_num)
        
    return model_ifmap_fault_dict_list, model_ofmap_fault_dict_list, model_weight_fault_dict_list

