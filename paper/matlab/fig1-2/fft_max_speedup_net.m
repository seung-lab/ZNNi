function [ speedup, up, down ] = fft_max_speedup_net( w, l, n, k )

nin = n + (k-1)*l;

[ up, down ] = fft_time(nin, 1, w);

for x = 2:l-1
    nin = nin - k + 1;
    [ u, d ] = fft_time(nin, w, w);
    up = up + u;
    down = down + d;
end

nin = nin - k + 1;
[ u, d ] = fft_time(nin, w, 1);
up = up + u;
down = down + d;

down = down + n^3;

speedup = up ./ down;