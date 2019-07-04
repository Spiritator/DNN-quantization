clear;
close all;

model_attr = h5readatt('navigation_droneNet_v2_140x140.h5','/','layer_names');
% attr = H5A.read(attr_id);
h5disp('../navigation_droneNet_v2_140x140.h5','/');
% h5disp('navigation_droneNet_v2_140x140.h5','/conv2d_1');
% h5disp('navigation_droneNet_v2_140x140.h5','/conv2d_1/conv2d_1');
% h5disp('navigation_droneNet_v2_140x140.h5','/conv2d_1/conv2d_1/kernel:0');

h5disp('navigation_droneNet_v2_140x140.h5','/conv2d_1/conv2d_1/kernel:0');
w_conv1 = h5read('navigation_droneNet_v2_140x140.h5','/conv2d_1/conv2d_1/kernel:0');

for n_f=1:32
    for channel=1:3
        disp('filter:')
        disp(n_f)
        disp('channel:')
        disp(channel)
        disp(w_conv1(n_f,:,:,channel))
        quantized = fi(w_conv1(n_f,:,:,channel), 1, 8, 7) ;
        disp(quantized)
    end
end
