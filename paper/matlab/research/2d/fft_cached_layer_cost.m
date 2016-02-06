function [ memory, flops ] = fft_cached_layer_cost( n, f0, f1, k, C, D )

n1 = n - k + 1;

memory = (f1 + f0 * f1) * n .* n;
flops  = 0;

flops  = flops + 2 * C * ceil(log2(n)) .* (n .* n) * ( f0 + f1 ); % FFTs
flops  = flops + 4 * f0 * f1 * n .* n;           % MADs
flops  = flops + D * n1 .* n1;

end

