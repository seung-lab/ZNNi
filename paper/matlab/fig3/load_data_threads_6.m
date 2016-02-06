function [ used, vals ] = load_data_threads_6( fname )

f = fopen(fname, 'r');

out = textscan(f, '%f,%f,%f,%f,%f,%f;');

fclose(f);

used = [5:5:30,40:10:60,80:20:120];

values = ones(120,240);
serial = ones(120,1);

n = size(out{1},1);

for j = 1:n
    %if out{2}(j) == 8
        if out{3}(j) == 1
            serial(out{1}(j)) = out{4}(j);
        end
        values(out{1}(j),out{3}(j)) = serial(out{1}(j)) / out{4}(j);        
    %end
end

vals = values(used,2:240);

end