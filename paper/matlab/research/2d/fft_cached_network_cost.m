function [ memory, flops ] = fft_cached_network_cost( nout, nlayers, width, k, C, D )

nin = nout + nlayers * (k-1);

[memory, flops] = fft_cached_layer_cost(nin, 1, width, k, C, D);

for x = 2:nlayers-1
    nin = nin - k + 1;
    [m, f] = fft_cached_layer_cost(nin, width, width, k, C, D);
    memory = memory + m;
    flops  = flops  + f;
end

nin = nin - k + 1;
[m, f] = fft_cached_layer_cost(nin, width, 1, k, C, D);
memory = memory + m;
flops  = flops  + f;

flops = flops ./ nout ./ nout;
memory = memory * 4; % bytes
memory = memory / 1024 / 1024 / 1024;

end

