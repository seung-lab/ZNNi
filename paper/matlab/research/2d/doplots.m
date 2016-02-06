k = 5;
close all;

%net = [3,3,0, 3,3,0, 3,3,3,3,0, 3,3,3,3,0, 3,3,3,3,0];
%w   = [64,64,64,  128,128,128,  256,256,256,256,256,  512,512,512,512,512,  512,512,512,512,512];

net = [7,0,7,0,7,0,7,7,7,7];
w   = [32,32,32,32,32,32,32,32,32,1];

[m, c] = fft_cached_pooling_network_cost(1:16, net, w, 5, 2);
plot(m,c, 'x-')

[m, c] = fft_cached_filtering_network_cost(1:16, net, w, 5, 2);
hold on;
plot(m,c, 'o-')

[m, c] = direct_batched_filtering_network_cost(1, 5:150, net, w, 2);
hold on;
plot(m,c, 'v-')


for x = 2:2:20
    

    [m, c] = fft_batched_filtering_network_cost(x, 5:150, net, w, 5, 2);
    %m = 32:256;
plot(m,c)
end