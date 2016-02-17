function [ onet ] = dense_network( inet, n, os )

is = get_dense_input_size(inet, os);
onet = dense_network_helper(inet, n, is);

end
