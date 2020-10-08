# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:39:08 2020

@author: Yung-Yu Tsai

Organized flow for compuation unit fault mapping
"""
import numpy as np
import json

from .tile import tile_PE, tile_FC_PE, io_data_solver

def PE_mapping_forward(ifmap_tile,
                       wght_tile,
                       ofmap_tile,
                       PEarray,
                       ifmap_expand_config,
                       wght_expand_config,
                       ofmap_expand_config,
                       PEarray_setup_config,
                       pre_plan=False,
                       verbose=4):
    """ Data mapping high level control
        Pre-plan the dataflow for backward PE mapping.    
        Or mapping the tile fault dictionary to PE dataflow model.
        
    Arguments
    ---------
    ifmap_tile: Class (tile_PE). 
        The tile class for PE array dataflow mapping of input feature maps.
    wght_tile: Class (tile_PE). 
        The tile class for PE array dataflow mapping of kernel and bias.
    ofmap_tile: Class (tile_PE). 
        The tile class for PE array dataflow mapping of output feature maps.
    PEarray: Class (PEarray). 
        The PE dataflow model class for PE array dataflow mapping.
    ifmap_expand_config: Dictionary or String. 
        Configuration for input feature maps tile expansion.
    wght_expand_config: Dictionary or String. 
        Configuration for weight (both kernal and bias) tile expansion.
    ofmap_expand_config: Dictionary or String. 
        Configuration for output feature maps tile expansion.
    
        If data type Dictionary, the dictionary is input argument for tile expansion function that writing in dictionay format.
        It will be use as **configuration put into expansion function.
        Else if data type String, the string is the directory to the JSON file which contains the configuration of 
        tile expansion function written in the format that can be read in as dictionary and **configuration put into expansion function.
    
    PEarray_setup_config: Dictionary or String. 
        Configuration for PE array dataflow setup.
        
        If data type Dictionary, the dictionary is input argument for PE array dataflow setup function that writing in dictionay format.
        It will be use as **configuration put into expansion function.
        Else if data type String, the string is the directory to the JSON file which contains the configuration of 
        PE array dataflow setup function written in the format that can be read in as dictionary and **configuration put into PE array dataflow setup function.
    
    dataflow_pre_plan: Bool. 
        Plan the dataflow model ahead. If True there will be no actual Tile to PEarray fault dictionary list transformation.
        Only save the expansion configuration for later PEarray to Tile transform.
        
    verbose: Integer. 
        Print the progress of forward mapping.
    verbose: Integer. 
        | The verbosity of printing forward mapping progress. max 5 (print all info), min 0 (print nothing), Folding All info 5.
        | The description below shows the minimum verbosity for info to print.
        | Mapping Start (2)
        | Mapping Tasks (3)
        | Folding forward/backward mapping info (5)

    Returns
    -------
    Fault Dictionary / None
        None. If dataflow_pre_plan is True.
        

    """
    if verbose>1:
        if verbose==5:
            print('\rPE array dataflow forward mapping ...',end=' ')
        else:
            print('\nPE array dataflow forward mapping ...')
    if verbose>2:
        if verbose==5:
            print('\r    Task (1/7): Load Tile Config ...',end=' ')
        else:
            print('    Task (1/7): Load Tile Config ...',end=' ')
    # load ifmap config
    if isinstance(ifmap_expand_config,str):
        with open(ifmap_expand_config, 'r') as config_file:
            ifmap_expand_config=json.load(config_file)
    elif isinstance(ifmap_expand_config,dict):
        pass
    else:
        raise TypeError('ifmap_expand_config must be String or Dictionary.')
    
    # load wght config
    if isinstance(wght_expand_config,str):
        with open(wght_expand_config, 'r') as config_file:
            wght_expand_config=json.load(config_file)
    elif isinstance(wght_expand_config,dict):
        pass
    else:
        raise TypeError('wght_expand_config must be String or Dictionary.')
    
    # load ofmap config    
    if isinstance(ofmap_expand_config,str):
        with open(ofmap_expand_config, 'r') as config_file:
            ofmap_expand_config=json.load(config_file)
    elif isinstance(ofmap_expand_config,dict):
        pass
    else:
        raise TypeError('ofmap_expand_config must be String or Dictionary.')
        
    if verbose>2:
        print('\r    Task (2/7): Load PE Array Config ...',end=' ')
    # load PEarray config
    if isinstance(PEarray_setup_config,str):
        with open(PEarray_setup_config, 'r') as config_file:
            PEarray_setup_config=json.load(config_file)
    elif isinstance(PEarray_setup_config,dict):
        pass
    else:
        raise TypeError('PEarray_setup_config must be String or Dictionary.')
    
    if verbose>2:
        print('\r    Task (3/7): Tile Expnasion ...         ',end=' ') 
    # expand ifmap
    if 'ksizes' not in ifmap_expand_config:
        ifmap_tile.expand_reshape_data(dataflow_pre_plan=pre_plan, **ifmap_expand_config)
    else:
        ifmap_tile.expand_extract_patches(dataflow_pre_plan=pre_plan, **ifmap_expand_config)
    # expand wght
    if 'ksizes' not in wght_expand_config:
        if 'bias_slice_width' in wght_expand_config:
            bias_slice_width=wght_expand_config.pop('bias_slice_width')
            wght_tile.expand_slice_bias(bias_slice_width=bias_slice_width,dataflow_pre_plan=pre_plan)
        wght_tile.expand_reshape_data(dataflow_pre_plan=pre_plan, **wght_expand_config)
    else:
        wght_tile.expand_extract_patches(dataflow_pre_plan=pre_plan, **wght_expand_config)
    # expand ofmap
    if 'ksizes' not in ofmap_expand_config:
        ofmap_tile.expand_reshape_data(dataflow_pre_plan=pre_plan, **ofmap_expand_config)
    else:
        ofmap_tile.expand_extract_patches(dataflow_pre_plan=pre_plan, **ofmap_expand_config)

    # setup PEarray
    if verbose>2:
        print('\r    Task (4/7): Tile Expnasion ...         ',end=' ') 
    PEarray.ifmap_tile=ifmap_tile
    PEarray.wght_tile=wght_tile
    PEarray.ofmap_tile=ofmap_tile
    PEarray.setup_dataflow(**PEarray_setup_config)

    # premapping
    if verbose>2:
        print('\r    Task (5/7): Tile Pre-mapping ...',end=' ') 
    PEarray.premapping_tile('ofmap', dataflow_pre_plan=pre_plan)
    PEarray.premapping_tile('wght', dataflow_pre_plan=pre_plan)
    PEarray.premapping_tile('ifmap', dataflow_pre_plan=pre_plan)
    PEarray.premapping_tile('bias', dataflow_pre_plan=pre_plan)
    PEarray.premapping_tile('psum', dataflow_pre_plan=pre_plan)
        
    # duplication
    if verbose>2:
        print('\r    Task (6/7): Mapping Duplication ...',end=' ') 
    PEarray.duplicate_mapping('ofmap', dataflow_pre_plan=pre_plan)
    PEarray.duplicate_mapping('wght', dataflow_pre_plan=pre_plan)
    PEarray.duplicate_mapping('ifmap', dataflow_pre_plan=pre_plan)
    PEarray.duplicate_mapping('bias', dataflow_pre_plan=pre_plan)
    PEarray.duplicate_mapping('psum', dataflow_pre_plan=pre_plan)
        
    # alignment
    if verbose>2:
        print('\r    Task (7/7): Clock Cycle Alignment ...',end=' ') 
    PEarray.align_slice_pack(dataflow_pre_plan=pre_plan)
    
    PEarray.mapping_shape_save()
    if verbose>2:
        if verbose==5:
            print('\r    Task (7/7): All Done.                 ',end='')
        else:
            print('\r    Task (7/7): All Done.                 ')
            
    if pre_plan:
        return None
    else:
        return PEarray.fault_dict
    
    
def PE_mapping_backward(layer, PEarray, fault_dict=None, save2tile=False, verbose=4, return_detail=False):
    """ Data mapping high level control
        Mapping the PE dataflow model fault dictionay to layer.
        
    Arguments
    ---------
    layer: Keras.Layer. 
    PEarray: Class (PEarray). 
        The PE dataflow model class for PE array dataflow mapping.
    fault_dict: Dictionary. 
        The fault dictionary be assigned to PEarray. 
        If None assuming that the fault dictionary of PEarray is already set.
    save2tile: Bool.
        Save solved fault dictionary to respective data tile or not.
        True for getting the fault information on data tile. False for PE fault injection model fault dictionary list.
        Usually False for simulation.
    verbose: Integer. 
        | The verbosity of printing backward mapping progress. max 4 (print all info), min 0 (print nothing), Folding All info 5.
        | The description below shows the minimum verbosity for info to print.
        | Mapping Start (2)
        | Mapping Tasks (3)
        | Folding forward/backward mapping info (5)
        | Mapping Sub-Tasks (4) i.e. decompose slice pack, solve correspond I/O, Tile to Layer.
        | Mapping Result (1)
    return_detail: Bool.
        Return layer mapping detail for sum up whole model mapping information.
    
    Returns
    -------
    The fault information Dictionary of Layer.
    """        
    if verbose>1:
        if verbose==5:
            print('\rPE array dataflow backward mapping ...',end=' ')
        else:
            print('\nPE array dataflow backward mapping ...')
    if verbose<4:
        PEarray.ifmap_tile.print_detail=False
        PEarray.ofmap_tile.print_detail=False
        PEarray.wght_tile.print_detail=False
    
    layer_weight_shape=[weight_shape.shape for weight_shape in layer.get_weights()]
    if len(layer_weight_shape)==0:
        if verbose>2:
            print('    no weight layer Skipped!')
        if return_detail:
            empty_info={'num_base_coor':0,
                        'num_fault_coor':0,
                        'num_psum_idx':0,
                        'num_layer_fault_coor':0,
                        'num_layer_psum_idx':0}
            
            return None, empty_info
        return None
    
    if fault_dict is not None:
        PEarray.fault_dict=fault_dict
        
    PEarray.mapping_shape_load()
    
    if len(PEarray.fault_dict)==0:
        print('Empty fault dict. No fault.')
        if return_detail:
            empty_info={'num_base_coor':0,
                        'num_fault_coor':0,
                        'num_psum_idx':0,
                        'num_layer_fault_coor':0,
                        'num_layer_psum_idx':0}
            
            return None, empty_info
        return None
        
    if verbose>2:
        if verbose==5:
            print('\r    Task (1/6): Decompose Slice Pack ...',end=' ')
        else:
            print('    Task (1/6): Decompose Slice Pack ...',end=' ') 
    PEarray.decompose_slice_pack(print_detail=verbose>3)

    if verbose>2:
        print('\r    Task (2/6): Reduce Mapping ...     ',end=' ') 
    PEarray.reduce_mapping('ofmap')
    PEarray.reduce_mapping('wght')
    PEarray.reduce_mapping('ifmap')
    PEarray.reduce_mapping('bias')
    PEarray.reduce_mapping('psum')
    
    if verbose>2:
        print('\r    Task (3/6): Tile Demapping...      ',end=' ') 
    PEarray.demapping_tile('ofmap')
    PEarray.demapping_tile('wght')
    PEarray.demapping_tile('ifmap')
    PEarray.demapping_tile('bias')
    PEarray.demapping_tile('psum')
    
    if verbose>2:
        print('\r    Task (4/6): Tile Shrinking... ',end=' ') 
    if PEarray.ofmap_tile.expand_method=='reshape':
        PEarray.ofmap_tile.shrink_reshape_data()
        PEarray.ofmap_tile.shrink_reshape_data(psum=True)
    elif PEarray.ofmap_tile.expand_method=='extract_patches':
        PEarray.ofmap_tile.shrink_return_patches(fast_gen=PEarray.fast_gen)
        PEarray.ofmap_tile.shrink_return_patches(psum=True, fast_gen=PEarray.fast_gen)
    else:
        raise ValueError('expand_method must be either \'reshape\' or \'extract_patches\'.')
        
    if PEarray.wght_tile.expand_method=='reshape':
        PEarray.wght_tile.shrink_reshape_data()
    elif PEarray.wght_tile.expand_method=='extract_patches':
        PEarray.wght_tile.shrink_return_patches(fast_gen=PEarray.fast_gen)
    else:
        raise ValueError('expand_method must be either \'reshape\' or \'extract_patches\'.')
        
    if PEarray.wght_tile.use_bias:
        PEarray.wght_tile.shrink_slice_bias()
    
    if PEarray.ifmap_tile.expand_method=='extract_patches':
        PEarray.ifmap_tile.shrink_return_patches(fast_gen=PEarray.fast_gen)
    elif PEarray.ifmap_tile.expand_method=='reshape':
        PEarray.ifmap_tile.shrink_reshape_data()
    else:
        raise ValueError('expand_method must be either \'reshape\' or \'extract_patches\'.')
         
    if verbose>2:
        print('\r    Task (5/6): Solve Fault I/O ...                                     ',end=' ') 
    # organize fault dict and give partial sum index
    solver=io_data_solver(PEarray.ofmap_tile,PEarray.wght_tile,PEarray.ifmap_tile,fault_num=PEarray.fault_num)
    PE_mac_fault_dict=solver.solve_correspond_io(save2tile,verbose>3)
    
    if verbose>2:
        print('\r    Task (6/6): Tile Return to layer ...                              ',end=' ')
    # inter tile fault dictionary transform to layer
    if not save2tile:
        PE_mac_fault_dict=solver.tile2layer(based_tile='ofmap',layer=layer,print_detail=verbose>3)
    else:
        ifmap_fd=solver.tile2layer(based_tile='ifmap',layer=layer,print_detail=verbose>3)
        wght_fd=solver.tile2layer(based_tile='wght',layer=layer,print_detail=verbose>3)
        ofmap_fd=solver.tile2layer(based_tile='ofmap',layer=layer,print_detail=verbose>3)
        PE_mac_fault_dict=(ifmap_fd, wght_fd, None, ofmap_fd)
        
    if verbose>2:
        if verbose==5:
            print('\r    Task (6/6): All Done.                                              ',end='')
        else:
            print('\r    Task (6/6): All Done.                                              ')
    if verbose>0:
        if not save2tile:
            report_2layer=solver.report_layer_map()
            if verbose==5:
                print('\r    mapped layer faults | base coors %d | ofmap %d '%(report_2layer['num_base_coor'], report_2layer['num_layer_fault_coor']))
            else:
                print('    mapped layer faults | base coors %d | ofmap %d '%(report_2layer['num_base_coor'], report_2layer['num_layer_fault_coor']))
            print('                        | total psum index %d '%report_2layer['num_layer_psum_idx'])
        else:
            if verbose==5:
                print('\r    mapped layer faults | ifmap %d | ofmap %d | weight %s '%(len(ifmap_fd),len(ofmap_fd),str([len(wght_fd),0])))
            else:
                print('    mapped layer faults | ifmap %d | ofmap %d | weight %s '%(len(ifmap_fd),len(ofmap_fd),str([len(wght_fd),0])))

    if return_detail:
        return PE_mac_fault_dict, solver.report_layer_map()
    
    return PE_mac_fault_dict


def PE_mapping2tile(PEarray, fault_dict=None, print_detail=True):
    """ Data mapping high level control
        Mapping the PE dataflow model fault dictionay to ifmap, weight and ofmap tile.
        
    Arguments
    ---------
    PEarray: Class (PEarray). 
        The PE dataflow model class for PE array dataflow mapping.
    fault_dict: Dictionary. 
        The fault dictionary be assigned to PEarray. 
        If None assuming that the fault dictionary of PEarray is already set.
    
    Returns
    -------
    The fault information Dictionary List of Layer.
    """        
    if print_detail:
        print('\nPE array dataflow backward mapping ...')
    else:
        PEarray.ifmap_tile.print_detail=False
        PEarray.ofmap_tile.print_detail=False
        PEarray.wght_tile.print_detail=False
    
    if fault_dict is not None:
        PEarray.fault_dict=fault_dict
        
    PEarray.mapping_shape_load()
        
    if print_detail:
        print('    Task (1/4): Decompose Slice Pack ...',end=' ') 
    PEarray.decompose_slice_pack(print_detail=print_detail)

    if print_detail:
        print('\r    Task (2/4): Reduce Mapping ...\t\t',end=' ') 
    PEarray.reduce_mapping('ofmap')
    PEarray.reduce_mapping('wght')
    PEarray.reduce_mapping('ifmap')
    PEarray.reduce_mapping('bias')
    PEarray.reduce_mapping('psum')
    
    if print_detail:
        print('\r    Task (3/4): Tile Demapping...\t',end=' ') 
    PEarray.demapping_tile('ofmap')
    PEarray.demapping_tile('wght')
    PEarray.demapping_tile('ifmap')
    PEarray.demapping_tile('bias')
    PEarray.demapping_tile('psum')
    
    if print_detail:
        print('\r    Task (4/4): Tile Shrinking...',end=' ') 
    if PEarray.ofmap_tile.expand_method=='reshape':
        PEarray.ofmap_tile.shrink_reshape_data()
        PEarray.ofmap_tile.shrink_reshape_data(psum=True)
    elif PEarray.ofmap_tile.expand_method=='extract_patches':
        PEarray.ofmap_tile.shrink_return_patches(fast_gen=PEarray.fast_gen)
        PEarray.ofmap_tile.shrink_return_patches(psum=True, fast_gen=PEarray.fast_gen)
    else:
        raise ValueError('expand_method must be either \'reshape\' or \'extract_patches\'.')
        
    if PEarray.wght_tile.expand_method=='reshape':
        PEarray.wght_tile.shrink_reshape_data()
    elif PEarray.wght_tile.expand_method=='extract_patches':
        PEarray.wght_tile.shrink_return_patches(fast_gen=PEarray.fast_gen)
    else:
        raise ValueError('expand_method must be either \'reshape\' or \'extract_patches\'.')
        
    if PEarray.wght_tile.use_bias:
        PEarray.wght_tile.shrink_slice_bias()
    
    if PEarray.ifmap_tile.expand_method=='extract_patches':
        PEarray.ifmap_tile.shrink_return_patches(fast_gen=PEarray.fast_gen)
    elif PEarray.ifmap_tile.expand_method=='reshape':
        PEarray.ifmap_tile.shrink_reshape_data()
    else:
        raise ValueError('expand_method must be either \'reshape\' or \'extract_patches\'.')
                 
    if print_detail:
        print('\r    Task (4/4): All Done.\t\t\t\t')


def mapping_valid_checker(ifmap_tile,
                          wght_tile,
                          ofmap_tile,
                          PEarray,
                          ifmap_expand_config,
                          wght_expand_config,
                          ofmap_expand_config,
                          PEarray_setup_config,
                          print_detail=False):
    """ Data mapping checker
        Verify the configuration of tile and PEarray pairs is feasible or not.
        By checking the subset of forward mapping fault coordinate can get the same original fault coordinate in backward mapping.
        
    Arguments
    ---------
    ifmap_tile: Class (tile_PE). 
        The tile class for PE array dataflow mapping of input feature maps.
    wght_tile: Class (tile_PE). 
        The tile class for PE array dataflow mapping of kernel and bias.
    ofmap_tile: Class (tile_PE). 
        The tile class for PE array dataflow mapping of output feature maps.
    PEarray: Class (PEarray). The 
        PE dataflow model class for PE array dataflow mapping.
    ifmap_expand_config: Dictionary or String. 
        Configuration for input feature maps tile expansion.
    wght_expand_config: Dictionary or String. 
        Configuration for weight (both kernal and bias) tile expansion.
    ofmap_expand_config: Dictionary or String. 
        Configuration for output feature maps tile expansion.
    
        If data type Dictionary, the dictionary is input argument for tile expansion function that writing in dictionay format.
        It will be use as **configuration put into expansion function.
        Else if data type String, the string is the directory to the JSON file which contains the configuration of 
        tile expansion function written in the format that can be read in as dictionary and **configuration put into expansion function.
    
    PEarray_setup_config: Dictionary or String. 
        Configuration for PE array dataflow setup.
        
        If data type Dictionary, the dictionary is input argument for PE array dataflow setup function that writing in dictionay format.
        It will be use as **configuration put into expansion function.
        Else if data type String, the string is the directory to the JSON file which contains the configuration of 
        PE array dataflow setup function written in the format that can be read in as dictionary and **configuration put into PE array dataflow setup function.
                
    print_detail: Bool. 
        Print the progress of ckecking.
    
    Returns
    -------
    Bool 
        The mapping is feasible or not.
    """        
    if print_detail:
        print('\nDataflow mapping feasblility check...')
    
    # Generate test fault
    fault_loc_i=tuple()
    for i in range(ifmap_tile.shape_len):
        fault_loc_i+=(np.random.randint(ifmap_tile.tile_shape[i]),)
    fault_bit_i=np.random.randint(ifmap_tile.wl)
    ifmap_tile.fault_dict={fault_loc_i:{'SA_type':'flip','SA_bit':fault_bit_i}}
    
    fault_loc_w=tuple()
    for i in range(wght_tile.shape_len):
        fault_loc_w+=(np.random.randint(wght_tile.tile_shape[i]),)
    fault_bit_w=np.random.randint(wght_tile.wl)
    wght_tile.fault_dict={fault_loc_w:{'SA_type':'flip','SA_bit':fault_bit_w}}
    
    fault_loc_o=tuple()
    for i in range(ofmap_tile.shape_len):
        fault_loc_o+=(np.random.randint(ofmap_tile.tile_shape[i]),)
    fault_bit_o=np.random.randint(ofmap_tile.wl)
    ofmap_tile.fault_dict={fault_loc_o:{'SA_type':'flip','SA_bit':fault_bit_o}}
    
    # Forward mapping
    PE_fault_dict=PE_mapping_forward(ifmap_tile,wght_tile,ofmap_tile,PEarray,ifmap_expand_config,wght_expand_config,ofmap_expand_config,PEarray_setup_config,print_detail=print_detail)

    # Sampple fault list
    forward_fault_num=len(PE_fault_dict)
    forward_fault_coors=np.array(list(PE_fault_dict.keys()))
    forward_fault_vl=np.array(list(PE_fault_dict.values()))
    
    sample_ifmap_coor=None
    sample_wght_coor=None
    sample_ofmap_coor=None
    cnt=0
    while(sample_ifmap_coor is None or sample_wght_coor is None or sample_ofmap_coor is None):
        sample_idx=np.random.randint(forward_fault_num)
        sample_coor=forward_fault_coors[sample_idx]
        sample_info=forward_fault_vl[sample_idx]
        if sample_info['param']=='ifmap_in':
            sample_ifmap_coor=sample_coor
            sample_ifmap_info=sample_info
        elif sample_info['param']=='wght_in':
            sample_wght_coor=sample_coor
            sample_wght_info=sample_info
        elif sample_info['param']=='psum_out':
            sample_ofmap_coor=sample_coor
            sample_ofmap_info=sample_info
        cnt+=1
        if print_detail:
            print('\r sample iteration %d '%cnt,end=' ')
    
    sample_ifmap_info.update({'id':0})
    sample_wght_info.update({'id':1})
    sample_ofmap_info.update({'id':2})
    
    sample_coor=np.stack([sample_ifmap_coor,sample_wght_coor,sample_ofmap_coor])
    sample_info=[sample_ifmap_info,sample_wght_info,sample_ofmap_info]
    
    sample_coor=list(zip(*sample_coor.T))
    sample_fault_dict=dict(zip(sample_coor,sample_info))
    
    # Backward mapping sample
    PE_mapping2tile(PEarray,sample_fault_dict,print_detail=print_detail)
    
    # Compare Forward and Backward mapping
    check_i=fault_loc_i in list(PEarray.ifmap_tile.fault_dict.keys())
    check_w=fault_loc_w in list(PEarray.wght_tile.fault_dict.keys())
    check_o=fault_loc_o in list(PEarray.ofmap_tile.fault_dict.keys())+list(PEarray.ofmap_tile.psum_fault_dict.keys())

    check_pass=all([check_i,check_w,check_o])
    
    
    if check_pass:
        if print_detail:
            print('\r    Check Passed!!\t\t\t\t\t\n')
    else:
        print('\r    Check Failed!!\t\t\t\t\t')
        print('fault location ifmap',end=' ')
        print(fault_loc_i)
        print('mapped fault location ifmap',end=' ')
        print(list(PEarray.ifmap_tile.fault_dict.keys()))
        
        print('fault location weight',end=' ')
        print(fault_loc_w)
        print('mapped fault location weight',end=' ')
        print(list(PEarray.wght_tile.fault_dict.keys()))

        print('fault location ofmap',end=' ')
        print(fault_loc_o)
        print('mapped fault location ofmap',end=' ')
        print(list(PEarray.ofmap_tile.fault_dict.keys())+list(PEarray.ofmap_tile.psum_fault_dict.keys()))
        
        print(' ')

    return check_pass

