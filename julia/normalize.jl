using HDF5

fin = ARGS[1]
fout = ARGS[2]

raw = h5read(fin, "main");

out = zeros(Float32, size(raw))

# normalize
for z in 1:size(raw,3)
    im = Array{Float32,2}(raw[:,:,z])
    im = im ./ 255.0
    im = im - mean(im)
    im = im ./ std(im)
    out[:,:,z] = im
end

if isfile(fout)
    rm(fout)
end
h5write(fout, "main", out)
