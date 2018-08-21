clear;
close all;

% setup
view_filename = 'alexnet_weights.h5';
fig_filename_prefix = 'alexnet_imagenet_weight_distribution';

% open weight h5 file
weight_file_id = H5F.open(view_filename);
base_group_id = H5G.open(weight_file_id,'/');
layer_info_id = H5A.open(base_group_id,'layer_names');
backend_id = H5A.open(base_group_id,'backend');
keras_ver_id = H5A.open(base_group_id,'keras_version');
layer_info = H5A.read(layer_info_id);
backend = H5A.read(backend_id);
keras_ver = H5A.read(keras_ver_id);

layer_info = layer_info';

for i=1:length(layer_info(:,1))
    layer_group_id = H5G.open(base_group_id,deblank(layer_info(i,:)));
    weight_name_id = H5A.open(layer_group_id,'weight_names');
    
    attr_info = H5A.get_info(weight_name_id);
    if attr_info.data_size ~= 0
        
        weight_name = H5A.read(weight_name_id);
        
        type_id = H5A.get_type(weight_name_id);
        space_id = H5A.get_space(weight_name_id);
        
        weight_name = weight_name';
        
        for j=1:length(weight_name(:,1))
            dset_id = H5D.open(layer_group_id,deblank(weight_name(j,:)));
            type_id = H5D.get_type(dset_id);
            
            weight = H5D.read(dset_id); 
            
            distribution_fig=histogram(weight);
            ylabel('# of weights');
            xlabel('weight value');
            title(replace(weight_name(j,:),'_','-'));
            fig_filename='%s_%s.png';
            fig_filename=sprintf(fig_filename,fig_filename_prefix,deblank(weight_name(j,:)));
            fig_filename=replace(fig_filename,{'/',':'},'_');
            saveas(distribution_fig,fig_filename);
            
            
            file_space_id = H5D.get_space(dset_id);
            mem_space_id = H5S.copy(file_space_id);
            
            
        end
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
H5S.close(mem_space_id);