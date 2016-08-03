using HDF5

"""
get network file
"""
function get_fnet()
    if length(ARGS) >= 1
        fnet = ARGS[1]
    else
        fnet = "net_current.h5"
    end
    @assert ishdf5(fnet)
    return fnet
end


function get_fout()
    if length(ARGS) >= 2
        fout = ARGS[2]
    else
        fout = "netbin/"
    end
    if !ispath(fout)
        mkdir(fout)
    end
    return fout
end

typealias Tnet Dict{Symbol, Dict{Symbol, Any}}

"""
read network
"""
function readnet(fnet::AbstractString)
    f = h5open(fnet, "r")
    if has(f,"processing")
        # standard IO
        pre = "/processing/znn/train/network/"
    else
        pre = "/"
    end
    net = Tnet()
    # traverse the edges and nodes
    for en in names(f[pre])
        net[Symbol(en)] = Dict{Symbol, Any}()
        # traverse the attributes
        for att in names(f[joinpath(pre, en)])
            obj = read(f[joinpath(pre, en, att)])
            if contains(att, "filters")
                # switch the input size and output size of kernel
                # in ZNN we have tensor[n_input][n_output][x][y][z] and ZNNi tensor[n_output][n_input][x][y][z]
                println("$en / $att : $(size(obj))")
                # we have znnhelper to fix dims, so no need to permute here!
                #obj = permutedims(obj, [2,1,3,4,5])
            end
            net[Symbol(en)][Symbol(att)] = obj
        end
    end
    close(f)
    return net
end

"""
write network to a binary file
"""
function net2binfile(net, fout)
    if !isdir(fout)
        mkdir(fout)
    end
    for layer in keys(net)
        flayer = joinpath(fout, string(layer))
        if !isdir(flayer)
            mkdir(flayer)
        end
        for att in keys(net[layer])
            x = net[layer][att]
            fname = joinpath(fout, string(layer), string(att))
            f = open(fname, "w")
            if isa(x, Array)
                write(f, x)
            elseif isa(x, AbstractString)
                write(f, x)
            end
            close(f)
        end
    end
end

function main()
    fnet = get_fnet()
    fout = get_fout()
    # read network file
    net = readnet(fnet)
    #@show net
    # write to binary
    net2binfile(net, fout)
end

main()
