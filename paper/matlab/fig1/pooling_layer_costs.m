function [ costs ] = pooling_layer_costs( n, fin, fout, is, fs )

    assert(fin==fout);
    assert(max(rem(is,fs))==0);

    os = is / fs;

    ret = struct('flops' , 3 * n * fout *os.*os.*os, ...
                 'memory', 2 * n * fout *os.*os.*os, ...
                 'stack' , is * 0, ...
                 'stored', is * 0);

    costs = struct('direct', ret, 'fft', ret, 'cfft', ret);

end
