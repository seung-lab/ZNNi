function [ net ] = build_network( input_size, filters, widths )

net = struct;
net.fin  = input_size;
net.fout = widths(1);
net.filter = filters(1);

if ( numel(filters) > 1 )
    net.next = build_network(widths(1), filters(2:end), widths(2:end));
else
    net.next = 0;
end

end

