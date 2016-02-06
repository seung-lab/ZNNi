function [ used, vals, seri ] = load_data_6( fname )

f = fopen(fname, 'r');

out = textscan(f, '%f,%f,%f,%f,%f,%f;');

fclose(f);

used = [5:5:30,40:10:60,80:20:120];

serial = ones(120,1);
values = ones(120,1)*50000;

n = size(out{1},1);

for j = 1:n
    if out{2}(j) > 0
        w = out{1}(j);
        if out{3}(j) == 1
            serial(w) = out{4}(j);
        else
            t = out{4}(j);
            values(w) = min(values(w),t);
        end
    end
end

%values = serial ./ values;

vals = values(used);
seri = serial(used);



end