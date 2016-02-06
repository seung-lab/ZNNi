function [ onet ] = network_cost( inet )

onet = 0;

if isstruct(inet)
    onet = struct;
    onet.fin = inet.fin;
    onet.fout = inet.fout;
    onet.n = inet.n;
    onet.is = inet.is;
    onet.os = inet.os;
    onet.filter = inet.filter;
    
    onet.next = network_cost(inet.next);
    
    if onet.filter > 0
        onet.direct = direct_conv_layer_cost(onet.n, onet.fin, onet.fout, onet.is, onet.filter);
        onet.fft = fft_layer_cost(onet.n, onet.fin, onet.fout, onet.is, onet.filter);
        onet.cfft = cached_fft_layer_cost(onet.n, onet.fin, onet.fout, onet.is, onet.filter);
    else
        onet.direct = pooling_layer_cost(onet.n, onet.fin, onet.fout, onet.is, -onet.filter);
        onet.fft = onet.direct;
        onet.cfft = onet.direct;
    end
    
    onet.tdirect = onet.direct;
    onet.tfft = onet.fft;
    onet.tcfft = onet.cfft;
    
    if isstruct(inet.next)
        onet.tdirect = append_cost(onet.tdirect, onet.next.tdirect);
        onet.tfft = append_cost(onet.tfft, onet.next.tfft);
        onet.tcfft = append_cost(onet.tcfft, onet.next.tcfft);
    end
end

end

