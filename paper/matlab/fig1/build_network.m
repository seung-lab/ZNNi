function [ net ] = build_network( net_fin, filters, widths )

    assert(numel(filters)==numel(widths));

    layers = struct([]);

    fin = net_fin;

    for x = 1:numel(filters)
        layers(x).fin  = fin;
        layers(x).fout = widths(x);
        layers(x).fs   = filters(x);
        fin = widths(x);
    end

    shift  = 1;
    fov    = 1;

    for x = numel(layers):-1:1
        if layers(x).fs > 0
            fov = fov + layers(x).fs - 1;
        else
            fov    = fov * abs(layers(x).fs);
            shift  = shift  * abs(layers(x).fs);
        end
    end

    net = struct('layers', layers, 'fov', fov, 'shift', shift, ...
                 'dense_cost', @dense_cost, ...
                 'sparse_cost', @sparse_cost);

end


function [ cost ] = zero_cost( )
    cost = struct('flops', 0, 'memory', 0, 'stack', 0, 'stored', 0);
end

function [ cost ] = zero_costs( )
    cost = struct('direct', zero_cost(), 'fft', zero_cost(), 'cfft', ...
                  zero_cost());
end


function [ r ] = append_cost( a, b )

    r = struct( 'flops' , a.flops + b.flops, ...
                'memory', max(a.memory, b.memory), ...
                'stack' , max(a.stack, b.stack), ...
                'stored', a.stored + b.stored );

end

function [ r ] = append_costs( a, b )

    r = struct( 'direct', append_cost(a.direct, b.direct), ...
                'fft'   , append_cost(a.fft, b.fft), ...
                'cfft'  , append_cost(a.cfft, b.cfft) );
end


function [ r ] = scale_cost( a, b )

    GB = 1024*1024*1024;

    r = struct( 'flops' , a.flops ./ b, ...
                'memory', 4*a.memory / GB, ...
                'stack' , 4*a.stack / GB, ...
                'stored', 4*a.stored / GB, ...
                'gb', 4*(a.memory+a.stack+a.stored) / GB);

end

function [ r ] = scale_costs( a, b )

    r = struct( 'direct', scale_cost(a.direct, b), ...
                'fft'   , scale_cost(a.fft, b), ...
                'cfft'  , scale_cost(a.cfft, b) );
end


function [ r ] = dense_cost( net, nin, os )

    assert(max(abs(rem(os,net.shift)))==0);

    r = zero_costs();

    is = os + net.fov - 1;
    n  = nin;

    for x = 1:numel(net.layers)
        l  = net.layers(x);
        fs = net.layers(x).fs;
        if fs > 0
            clc = conv_layer_costs(n,l.fin,l.fout,is,fs);
            r = append_costs(r,clc);
            is = is - fs + 1;
        else
            n = n * (-fs)^3;
            is = is + fs + 1;
            r = append_costs(r, pooling_layer_costs(n,l.fin,l.fout, ...
                                                    is,-fs));
            is = is / (-fs);
        end
    end

    r = scale_costs(r, os .* os .* os * nin);

end


function [ r ] = sparse_cost( net, nin, os )

    r = zero_costs();

    is = (os-1) * net.shift + net.fov;
    n  = nin;

    for x = 1:numel(net.layers)
        l  = net.layers(x);
        fs = net.layers(x).fs;
        if fs > 0
            clc = conv_layer_costs(n,l.fin,l.fout,is,fs);
            r = append_costs(r,clc);
            is = is - fs + 1;
        else
            r = append_costs(r, pooling_layer_costs(n,l.fin,l.fout, ...
                                                    is,-fs));
            is = is / (-fs);
        end
    end

    r = scale_costs(r, os .* os .* os * n);

end
