function [ cost ] = get_sparse_network_cost( net, n, isize )

cost = struct;
cost.direct = zero_cost();
cost.fft    = zero_cost();
cost.cfft   = zero_cost();

if isstruct(net)
    ncost = get_sparse_network_cost(net.next,
end

end

