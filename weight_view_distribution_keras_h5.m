clear;
close all;

% setup
view_filename = 'mobilenet_1_0_224_tf.h5';
fig_filename_prefix = 'mobilenet_1_0_224_tf_weight_distribution';

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

total_weight = [];

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
            
            weight_queue = reshape(weight,1,[]);
            
            total_weight = [total_weight weight_queue];
            
            if size(weight_queue,2)>18400
                distribution_fig=histogram(weight,100);
            else
                distribution_fig=histogram(weight);
            end
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

% total_weight_var = var(total_weight);
% 
% for i=1:length(total_weight)
%     if abs(total_weight(i))>0.1
%         total_weight(i)=0;
%     end
% end


distribution_fig=histogram(total_weight,100);

ylabel('# of weights');
xlabel('weight value');
title('all');
fig_filename='%s_%s.png';
fig_filename=sprintf(fig_filename,fig_filename_prefix,'all_original');
fig_filename=replace(fig_filename,{'/',':'},'_');
saveas(distribution_fig,fig_filename);

total_weight_std = std(total_weight);
total_weight_mean = mean(total_weight);
x=linspace(total_weight_mean-2*total_weight_std,total_weight_mean+2*total_weight_std,100);
distribution_fig=histogram(total_weight,x);

ylabel('# of weights');
xlabel('weight value');
title('all');
fig_filename='%s_%s.png';
fig_filename=sprintf(fig_filename,fig_filename_prefix,'all_normalize_x_axis');
fig_filename=replace(fig_filename,{'/',':'},'_');
saveas(distribution_fig,fig_filename);

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