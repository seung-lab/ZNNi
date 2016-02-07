function [ costs ] = conv_layer_costs( n, fin, fout, is, fs )

    costs = struct('direct', direct_layer(n,fin,fout,is,fs), ...
                   'fft'   , fft_layer(n,fin,fout,is,fs), ...
                   'cfft'  , cached_fft_layer(n,fin,fout,is,fs));

end

function [ ret ] = direct_layer( n, fin, fout, is, fs )
    os = is - fs + 1;
    flops  = 2 * n * fin * fout * os.*os.*os .* fs.*fs.*fs;

    memory = is.*is.*is * fin * n + ...
             os.*os.*os * fout * n + ...
             fs.*fs.*fs * fin * fout;

    stack  = is * 0;
    stored = is * 0;

    ret = struct('flops', flops, 'memory', memory, ...
                 'stack', stack, 'stored', stored);
end

function [ ret ] = fft_layer( n, fin, fout, is, fs )
    C = 5;

    xs = floor(is/2) + 1;

    flops  = 3*C * n * fin  * ceil(log2(is)) .*is.*is.*is + ...
             3*C * n * fout * ceil(log2(is)) .*is.*is.*is + ...
             C * fin * fout * ceil(log2(is)).*is.*fs.*fs + ...
             C * fin * fout * ceil(log2(is)).*is.*is.*fs + ...
             C * fin * fout * ceil(log2(is)).*is.*is.*is + ...
             4*n*fin*fout * xs.*is.*is;


    memory = 2*xs.*is.*is * fin * n + ...
             2*xs.*is.*is * fout * n + ...
             fs.*fs.*fs * fin * fout;

    stack  = fs.*fs.*is + 2*xs.*is.*is;
    stored = is * 0;

    ret = struct('flops', flops, 'memory', memory, ...
                 'stack', stack, 'stored', stored);
end


function [ ret ] = cached_fft_layer( n, fin, fout, is, fs )
    C = 5;

    xs = floor(is/2) + 1;

    flops  = 3*C * n * fin  * ceil(log2(is)) .*is.*is.*is + ...
             3*C * n * fout * ceil(log2(is)) .*is.*is.*is + ...
             4*n*fin*fout * xs.*is.*is;

    memory = 2*xs.*is.*is * fin * n + ...
             2*xs.*is.*is * fout * n + ...
             fs.*fs.*fs * fin * fout;

    stack  = is * 0;
    stored = 2*xs.*is.*is * fin * fout;

    ret = struct('flops', flops, 'memory', memory, ...
                 'stack', stack, 'stored', stored);
end