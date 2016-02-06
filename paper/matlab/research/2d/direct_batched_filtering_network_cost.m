function [ memory, flops ] = direct_batched_filtering_network_cost( batch, nout, layers, width, D )

sparse = ones(size(layers));
orglayers = layers;

for x = 2:size(layers,2)
    sparse(x) = sparse(x-1);
    if layers(x) == 0
        sparse(x) = sparse(x) * 2;
    else
        layers(x) = (layers(x)-1) * sparse(x) + 1;
    end
end

nin = nout;

for x = size(layers,2):-1:1
    if layers(x) > 0
        nin = nin + layers(x) - 1;
    else
        nin = nin + 1;
    end
end

memory = 0;
flops  = 0;

for x = 1:size(layers,2)
    
    f0 = width(x);
    if x > 1
        f0 = width(x-1);
    end
    f1 = width(x);
    if x == 1
        f0 = 1;
    end
    if x == size(layers,2)
        f1 = 1;
    end

    if layers(x) > 0
        [m,f] = direct_batched_layer_cost(batch, nin, f0, f1, orglayers(x), D);
        memory = max(memory, m);
        flops  = flops  + f;
        nin = nin - layers(x) + 1;
    else
        nin = nin - 1;
        memory = max(memory, nin .* nin * batch * f1);
        flops = flops + 2 * nin .* nin * batch;
    end
end

flops = flops ./ nout ./ nout / batch;
memory = memory * 4; % bytes
memory = memory / 1024 / 1024 / 1024;

end
