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

    C = 2.5;

    xs = floor(is/2) + 1;

    flops  = 3*C * n * fin  * (log2(is)) .*is.*is.*is / 2 + ...
             3*C * n * fout * (log2(is)) .*is.*is.*is / 2 + ...
             C * fin * fout * (log2(is)).*is.*fs.*fs / 2 + ...
             C * fin * fout * (log2(is)).*is.*is.*fs / 2 + ...
             C * fin * fout * (log2(is)).*is.*is.*is / 2 + ...
             8*n*fin*fout * xs.*is.*is;

    fmax = max(fin,fout);     

    memory = 4 * 2*xs.*is.*is * fmax * n + ...
             fs.*fs.*fs * fin * fout;

    stack  = 2*xs.*is.*is * fmax;
    stored = is * 0;

    ret = struct('flops', flops, 'memory', memory, ...
                 'stack', stack, 'stored', stored);
end


function [ ret ] = cached_fft_layer( n, fin, fout, is, fs )
    C = 2.5;

    xs = floor(is/2) + 1;

    flops  = 3*C * n * fin  * (log2(is)) .*is.*is.*is / 2 + ...
             3*C * n * fout * (log2(is)) .*is.*is.*is / 2 + ...
             8*n*fin*fout * xs.*is.*is;

    fmax = max(fin,fout);     
         
    memory = 2*xs.*is.*is * fmax * n + ...
             2*xs.*is.*is * fmax * n + ...
             fs.*fs.*fs * fin * fout;

    stack  = is * 0;
    stored = 2*xs.*is.*is * fin * fout;

    ret = struct('flops', flops, 'memory', memory, ...
                 'stack', stack, 'stored', stored);
end