function [ cost ] = direct_conv_layer_cost( n, fin, fout, is, fs )

cost = struct;

cost.memoized = 0 * is;

cost.stack = 0 * is;
os = is - fs + 1;

cost.memory = n * fin * is .* is .* is + n * fout * os .* os .* os;
cost.memory = cost.memory + fs .* fs .* fs * fin * fout;

cost.cost = n * fin * fout * os .* os .* os .* fs .* fs .*fs * 2;

end

