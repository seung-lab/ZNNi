function [ memory, flops ] = direct_batched_layer_cost( batch, n, f0, f1, k, D )

n1 = n - k + 1;

memory = (f1 + f0) * n .* n * batch;
flops  = 0;

flops  = flops + f0 * f1 * n1 .* n1 * k * k * batch;           % MADs
flops  = flops + D * n1 .* n1 * batch;

end

