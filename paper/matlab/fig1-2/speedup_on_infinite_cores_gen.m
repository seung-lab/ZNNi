figure;
hold on;

Colors = { [      0    0.4470    0.7410],...
           [ 0.8500    0.3250    0.0980] };

for w = 5:5:120

    rfft = fft_max_speedup_net(5:120,w,1,5);
    rd   = direct_max_speedup_net(5:120,w,1,5);

    W = [0 0];
    W(1) = plot(5:120, rfft, 'color', Colors{1});
    W(2) = plot(5:120, rd, 'color', Colors{2});
    
    if w == 5
        legend(W, {'Direct', 'FFT'});           
    end;
end;


hold off;
