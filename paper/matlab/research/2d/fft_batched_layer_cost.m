function [ memory, flops ] = fft_batched_layer_cost( batch, n, f0, f1, k, C, D )

n1 = n - k + 1;

memory = (f0 + f1) * batch * n .* n;
flops  = 0;

flops  = flops + 2 * C * ceil(log2(n)) .* (n .* n) * ( f0 + f1 ) * batch; % Image FFTs
flops  = flops + 2 * C * ceil(log2(n)) .* (n .* n) * ( f0 * f1 );         % Filter FFTs
flops  = flops + 4 * f0 * f1 * n .* n * batch;                            % MADs
flops  = flops + D * n1 .* n1 .* batch;

end

