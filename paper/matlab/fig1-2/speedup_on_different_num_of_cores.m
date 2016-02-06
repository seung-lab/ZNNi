
Data1FFT = zeros(24*5,116);
Data2FFT = zeros(24*5,116);

Data1D = zeros(24*5,116);
Data2D = zeros(24*5,116);

Indices = zeros(24*5,116);

Ps = [8 18 40 60 120];

idx = 0;

for w = 4:1:40
    rfft = fft_max_speedup_net(5:120,w,1,5);
    rd   = direct_max_speedup_net(5:120,w,1,5);
    
    for j = 1:5
        idx = idx + 1;
        Indices(idx,:) = [5:120]; % / sqrt(Ps(j));
        Data1FFT(idx,:) = rfft ./ ( 1 + (rfft + 1) / Ps(j));
        Data2FFT(idx,:) = Data1FFT(idx,:) / Ps(j);

        Data1D(idx,:) = rd ./ ( 1 + (rd + 1) / Ps(j));
        Data2D(idx,:) = Data1D(idx,:) / Ps(j);
    end;
    
end;

createfigure(Indices',Data1FFT','Achievable Speedup');
createfigure(Indices',Data2FFT'*100,'Achievable Efficiency');

createfigure(Indices',Data1D','Achievable Speedup');
createfigure(Indices',Data2D'*100,'Achievable Efficiency');