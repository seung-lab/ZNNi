function [ input_size ] = get_dense_input_size( net, output_size )

[a, ~] = get_fov(net);

input_size = output_size + a - 1;

end

