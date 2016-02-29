function [ w, s ] = read_data( fname )

f = fopen(fname, 'r');
out = textscan(f, '%s %f,%f,%f %f');
fclose(f);

w = out{2};
s = out{5};

end

