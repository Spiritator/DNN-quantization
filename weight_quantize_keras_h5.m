clear;
close all;

% setup
original_filename = 'navigation_droneNet_v1_250x140_weight.h5';
quantized_filename = 'navigation_droneNet_v1_250x140_quantized_8B3I4F.h5';
quant_sign = 1;
qunat_wordlength = 8;
quant_factorial_bits = 7;


% open weight h5 file
weight_file_id = H5F.open(original_filename);
base_group_id = H5G.open(weight_file_id,'/');
layer_info_id = H5A.open(base_group_id,'layer_names');
backend_id = H5A.open(base_group_id,'backend');
keras_ver_id = H5A.open(base_group_id,'keras_version');
layer_info = H5A.read(layer_info_id);
backend = H5A.read(backend_id);
keras_ver = H5A.read(keras_ver_id);

% create quantize h5 weight file
fcpl = H5P.create('H5P_FILE_CREATE');
fapl = H5P.create('H5P_FILE_ACCESS');
quant_weight_file = H5F.create(quantized_filename,'H5F_ACC_TRUNC',fcpl,fapl);
plist = 'H5P_DEFAULT';
quant_base_group = H5G.open(quant_weight_file,'/');

for i=1:3
    if i==1
        info_id = layer_info_id;
    elseif i==2
        info_id = backend_id;
    elseif i==3
        info_id = keras_ver_id;
    end
    acpl_id = H5P.create('H5P_ATTRIBUTE_CREATE');
    type_id = H5A.get_type(info_id);
    space_id = H5A.get_space(info_id);
    attr_id = H5A.create(quant_base_group,H5A.get_name(info_id),type_id,space_id,acpl_id);
    if i==1
        H5A.write(attr_id,'H5ML_DEFAULT',layer_info)
    elseif i==2
        H5A.write(attr_id,'H5ML_DEFAULT',backend)
    elseif i==3
        H5A.write(attr_id,'H5ML_DEFAULT',keras_ver)
    end
end

layer_info = layer_info';

% layer_group_id = H5G.open(base_group_id,'batch_normalization_1');
%     weight_name_id = H5A.open(layer_group_id,'weight_names');
%     info = H5A.get_info(weight_name_id);
%     weight_name = H5A.read(weight_name_id);
%     weight_name = weight_name';
% dset_id = H5D.open(layer_group_id,'batch_normalization_1/gamma:0');
%             data = H5D.read(dset_id);

for i=1:length(layer_info(:,1))
    layer_group_id = H5G.open(base_group_id,deblank(layer_info(i,:)));
    weight_name_id = H5A.open(layer_group_id,'weight_names');
    
    quant_layer_group = H5G.create(quant_base_group,deblank(layer_info(i,:)),plist,plist,plist);
    attr_info = H5A.get_info(weight_name_id);
    if attr_info.data_size ~= 0
        
        quant_dset_group = H5G.create(quant_layer_group,deblank(layer_info(i,:)),plist,plist,plist);
        
        weight_name = H5A.read(weight_name_id);
        
        
        acpl_id = H5P.create('H5P_ATTRIBUTE_CREATE');
        type_id = H5A.get_type(weight_name_id);
        space_id = H5A.get_space(weight_name_id);
        attr_id = H5A.create(quant_layer_group,H5A.get_name(weight_name_id),type_id,space_id,acpl_id);
        H5A.write(attr_id,'H5ML_DEFAULT',weight_name)
        
        weight_name = weight_name';
        
        for j=1:length(weight_name(:,1))
            dset_id = H5D.open(layer_group_id,deblank(weight_name(j,:)));
            type_id = H5D.get_type(dset_id);
            
            weight = H5D.read(dset_id); 
            quantized = fi(weight, quant_sign, qunat_wordlength, quant_factorial_bits);
            data = single(quantized);
            
            file_space_id = H5D.get_space(dset_id);
            dcpl = 'H5P_DEFAULT';
            quant_dset_name = deblank(weight_name(j,length(deblank(layer_info(i,:)))+2:end));
            quant_dset_id = H5D.create(quant_dset_group,quant_dset_name,type_id,file_space_id,dcpl);
            mem_space_id = H5S.copy(file_space_id);
            H5D.write(quant_dset_id,'H5ML_DEFAULT',mem_space_id,file_space_id,plist,data);
            
            
            % H5D.write(dset_id,'H5ML_DEFAULT',mem_space_id,file_space_id,plist,quantized);
            % h5write('navigation_droneNet_v2_140x140.h5',char('/',deblank(layer_info(i,:)),'/',deblank(weight_name(j,:))),data)
        end
    else
        % weight_name = H5A.read(weight_name_id);
        acpl_id = H5P.create('H5P_ATTRIBUTE_CREATE');
        type_id = H5A.get_type(weight_name_id);
        space_id = H5A.get_space(weight_name_id);
        attr_id = H5A.create(quant_layer_group,H5A.get_name(weight_name_id),type_id,space_id,acpl_id);
        % H5A.write(attr_id,'H5ML_DEFAULT',single(zeros(0)));
    end
end

H5A.close(layer_info_id);
H5A.close(backend_id);
H5A.close(keras_ver_id);
H5G.close(base_group_id);
H5F.close(weight_file_id);
H5G.close(layer_group_id);
H5A.close(weight_name_id);
H5D.close(dset_id);
H5T.close(type_id);
% H5A.close(space_id);
H5A.close(attr_id);
H5P.close(acpl_id);
H5G.close(quant_base_group);
H5F.close(quant_weight_file);
H5G.close(quant_layer_group);
H5D.close(quant_dset_id);
% H5D.close(file_space_id);
H5S.close(mem_space_id);
clear;
close all;