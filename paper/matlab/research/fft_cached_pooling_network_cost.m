function [ memory, flops ] = fft_cached_pooling_network_cost( nout, layers, width, C, D )

nin = nout;

for x = size(layers,2):-1:1
    if layers(x) > 0
        nin = nin + layers(x) - 1;
    else
        nin = nin * 2;
    end
end

memory = 0;
flops  = 0;

for x = 1:size(layers,2)
    f0 = width;
    f1 = width;
    if x == 1
        f0 = 1;
    end
    if x == size(layers,2)
        f1 = 1;
    end
    
    if layers(x) > 0
        [m,f] = fft_cached_layer_cost(nin, f0, f1, layers(x), C, D);
        memory = memory + m;
        flops  = flops  + f;
        nin = nin - layers(x) + 1;
    else        
        nin = nin / 2;
        memory = memory + nin .* nin .* nin;        
        flops = flops + 3 * nin .* nin .* nin; 
    end
end

flops = flops ./ nout ./ nout ./ nout;
memory = memory * 4; % bytes
memory = memory / 1024 / 1024 / 1024;

end

