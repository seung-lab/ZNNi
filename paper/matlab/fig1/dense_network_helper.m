function [ onet ] = dense_network_helper( inet, n, is )

onet = inet;

if isstruct(inet)
    onet = struct;
    onet.fin = inet.fin;
    onet.fout = inet.fout;
    onet.filter = inet.filter;
    onet.n = n;
    onet.is = is;
    
    if onet.filter > 0
        fs = onet.filter;
        onet.os = is - fs + 1;
        onet.next = dense_network_helper(inet.next, n, onet.os);
    else
        fs = -onet.filter;
        onet.is = onet.is - fs + 1;
        onet.n = onet.n * fs^3;
        assert(max(rem(is+1,fs))==0);
        onet.os = onet.is / fs;
        onet.next = dense_network_helper(inet.next, n*fs^3, onet.os);
    end
end

end

