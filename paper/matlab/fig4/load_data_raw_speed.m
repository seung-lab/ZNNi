function [ values ] = load_data_raw_speed( fname, tval )

f = fopen(fname, 'r');
out = textscan(f, '%f,%f,%f,%f,%f,%f;');
fclose(f);

values = ones(240,1)*50000;

n = size(out{1},1);

for j = 1:n
    if out{1}(j) == tval
        values(out{3}(j)) = out{4}(j);        
    end
end

values = 1 ./ values;

end