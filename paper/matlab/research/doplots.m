k = 7;
close all;
m = 32:128;

[m, c] = fft_cached_pooling_network_cost(32:64, [k,0,k,0,k,0,k,k,k,k], 40, 5, 2);

%m = 32:128;

plot(m,c, 'x-')
[m, c] = fft_cached_filtering_network_cost(32:164, [k,0,k,0,k,0,k,k,k,k], 40, 5, 2);
hold on;
plot(m,c, 'o-')

for x = 2:2:20
    

    [m, c] = fft_batched_filtering_network_cost(x, 32:1000, [k,0,k,0,k,0,k,k,k,k], 40, 5, 2);
    %m = 32:256;
plot(m,c)
end