# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:45:33 2018

@author: Yung-Yu Tsai

Generate random stuck at fault on a model. 
Only generate the layers have weights.
assign a fault rate to generate.
"""

import numpy as np

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
                if isinstance(ifmap_fault_dict[coordinate]['fault_bit'],list):
                    if fault_bit in ifmap_fault_dict[coordinate]['fault_bit']:
                        continue
                    else:
                        ifmap_fault_dict[coordinate]['fault_type'].append('flip')
                        ifmap_fault_dict[coordinate]['fault_bit'].append(fault_bit)
                        fault_count += 1
                else:
                    if fault_bit == ifmap_fault_dict[coordinate]['fault_bit']:
                        continue
                    else:
                        ifmap_fault_dict[coordinate]['fault_type']=[ifmap_fault_dict[coordinate]['fault_type'],'flip']
                        ifmap_fault_dict[coordinate]['fault_bit']=[ifmap_fault_dict[coordinate]['fault_bit'],fault_bit]
                        fault_count += 1
            else:
                ifmap_fault_dict[coordinate]={'fault_type':'flip',
                                              'fault_bit' : fault_bit}
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
                if isinstance(ofmap_fault_dict[coordinate]['fault_bit'],list):
                    if fault_bit in ofmap_fault_dict[coordinate]['fault_bit']:
                        continue
                    else:
                        ofmap_fault_dict[coordinate]['fault_type'].append('flip')
                        ofmap_fault_dict[coordinate]['fault_bit'].append(fault_bit)
                        fault_count += 1
                else:
                    if fault_bit == ofmap_fault_dict[coordinate]['fault_bit']:
                        continue
                    else:
                        ofmap_fault_dict[coordinate]['fault_type']=[ofmap_fault_dict[coordinate]['fault_type'],'flip']
                        ofmap_fault_dict[coordinate]['fault_bit']=[ofmap_fault_dict[coordinate]['fault_bit'],fault_bit]
                        fault_count += 1
            else:
                ofmap_fault_dict[coordinate]={'fault_type':'flip',
                                              'fault_bit' : fault_bit}
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
                    if isinstance(weight_fault_dict[i][coordinate]['fault_bit'],list):
                        if fault_bit in weight_fault_dict[i][coordinate]['fault_bit']:
                            #print('error 1')
                            continue
                        else:
                            weight_fault_dict[i][coordinate]['fault_type'].append('flip')
                            weight_fault_dict[i][coordinate]['fault_bit'].append(fault_bit)
                            fault_count += 1
                    else:
                        if fault_bit == weight_fault_dict[i][coordinate]['fault_bit']:
                            #print('error 2')
                            continue
                        else:
                            weight_fault_dict[i][coordinate]['fault_type']=[weight_fault_dict[i][coordinate]['fault_type'],'flip']
                            weight_fault_dict[i][coordinate]['fault_bit']=[weight_fault_dict[i][coordinate]['fault_bit'],fault_bit]
                            fault_count += 1
                else:
                    weight_fault_dict[i][coordinate]={'fault_type':'flip',
                                                  'fault_bit' : fault_bit}
                    fault_count += 1
            
        model_weight_fault_dict_list.append(weight_fault_dict)    
        print('    generated layer %d weight %s faults'%(layer_num,str(layer_weight_fault_num)))
        
        print('    layer %d Done!'%layer_num)
        
    return model_ifmap_fault_dict_list, model_ofmap_fault_dict_list, model_weight_fault_dict_list
