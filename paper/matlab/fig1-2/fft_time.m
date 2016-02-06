function [ serial, parallel ] = fft_time( n, f0, f1 )

C = 2.5;

serial   = 9 * C * (log2(n)) * ( f0 .* f1 + f0 + f1 ) + 12 * f0 .* f1;
parallel = 12 * C * (log2(n)) + 4 * (ceil(log2(f0)) + ceil(log2(f1)));

serial = serial * n^3;
parallel = parallel * n^3;

end
