function [ serial, parallel ] = direct_time( n, k, f0, f1 )

serial   = 3 * (n-k+1)^3 * k^3 * f0 .* f1;
parallel = 3 * (n-k+1)^3 * k^3 + (n-k+1)^3 * ceil(log2(f0)) + n^3 * ceil(log2(f1));

end
