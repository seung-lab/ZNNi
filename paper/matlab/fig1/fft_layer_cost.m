function [ cost ] = fft_layer_cost( n, fin, fout, is, fs )

cost = struct;

C = 5;
cost.memoized = 0 * is;

cost.stack = is .* is .* is + fs .* fs .* is;

cost.memory = n * (fin + fout) * is .* is .* is;
cost.memory = cost.memory + fs .* fs .* fs * fin * fout;

% FFTs of inputs
cost.cost = n * 3 * C * ceil(log2(is)) .* is .* is .* is * fin;

% FFTs of kernels
cost.cost = cost.cost + fin * fout * C * ceil(log2(is)) .* is .* (is .* is + is .* fs + fs .* fs);

% MADs
cost.cost = cost.cost + fin * fout * n * 4 * is .* is .* is;

% iFFTs of the outputs
cost.cost = cost.cost + n * 3 * C * ceil(log2(is)) .* is .* is .* is * fout;

end

