function [ fov, bs ] = get_fov( net )

if numel(net) == 1
    fov = abs(net(1));
    bs  = 1;
else
    [rest, rbs] = get_fov(net(2:numel(net)));
    if ( net(1) > 0 )
        fov = rest + net(1) - 1;
        bs = rbs;
    else
        fov = rest * abs(net(1));
        bs = rbs * net(1);
    end
end

end

