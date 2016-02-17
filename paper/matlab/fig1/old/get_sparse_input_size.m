function [ input_size ] = get_sparse_input_size( net, output_size )

input_size = output_size;

if isstruct(net)
    input_size = get_sparse_input_size(net.next, output_size);
    if net.filter > 0
        input_size = input_size + net.filter - 1;
    else
        input_size = input_size * abs(net.filter);
    end
end

end

