function [ cost ] = cached_fft_layer_cost( n, fin, fout, is, fs )

cost = struct;

C = 5;
cost.memoized = fin * fout * is .* is .* is;
cost.stack = 0 * is;

cost.memory = n * (fin + fout) * is .* is .* is;
cost.memory = cost.memory + fs .* fs .* fs * fin * fout;

% FFTs of inputs
cost.cost = n * 3 * C * ceil(log2(is)) .* is .* is .* is * fin;

% MADs
cost.cost = cost.cost + fin * fout * n * 4 * is .* is .* is;

% iFFTs of the outputs
cost.cost = cost.cost + n * 3 * C * ceil(log2(is)) .* is .* is .* is * fout;

end

