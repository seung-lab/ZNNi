function [ fov, bs ] = get_fov( net )

bs  = 1;
fov = 1;

if isstruct(net)
    [fov, bs] = get_fov(net.next);
    if net.filter > 0
        fov = fov + net.filter - 1;
    else
        fov = fov * abs(net.filter);
        bs = bs * abs(net.filter);
    end
end

end

