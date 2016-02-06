function [ cost, memory, stack ] = ...
    direct_conv_layer_cost( n, fin, fout, is, fs )

stack = 0;
os = is - fs + 1;

memory = n * fin * is .* is .* is + n * fout * os .* os .* os;
memory = memory + fs .* fs .* fs * fin * fout;

cost = n * fin * fout * os .* os .* os .* fs .* fs .*fs * 2;

end

