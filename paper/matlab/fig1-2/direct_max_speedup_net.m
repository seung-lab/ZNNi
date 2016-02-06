function [ speedup, up, down ] = direct_max_speedup_net( w, l, n, k )

nin = n + (k-1)*l;

[ up, down ] = direct_time(nin, k, 1, w);

for x = 2:l-1
    nin = nin - k + 1;
    [ u, d ] = direct_time(nin, k, w, w);
    up = up + u;
    down = down + d;
end

nin = nin - k + 1;
[ u, d ] = direct_time(nin, k, w, 1);
up = up + u;
down = down + d;

down = down + (n-k+1)^3 * k^3;

speedup = up ./ down;