function [ cost ] = ...
    pooling_layer_cost( n, fin, fout, is, fs )

cost = struct;

assert(fin==fout);
assert(max(rem(is,fs))==0);

cost.memoized = 0 * is;

cost.stack = 0 * is;
os = is / fs;

cost.memory = n * fout * os .* os .* os;

cost.cost = 3 * cost.memory;

end

