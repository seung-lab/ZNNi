function [ onet ] = sparse_network( inet, n, os )

onet = os;

if isstruct(inet)
    onet = struct;
    onet.fin = inet.fin;
    onet.fout = inet.fout;
    onet.filter = inet.filter;
    onet.n = n;
    
    onet.next = sparse_network(inet.next, n, os);
    
    if isstruct(onet.next)
        onet.os = onet.next.is;
    else
        onet.os = os;
    end
    
    if inet.filter > 0
        onet.is = onet.os + onet.filter - 1;
    else
        onet.is = onet.os * abs(onet.filter);
    end
end


end

