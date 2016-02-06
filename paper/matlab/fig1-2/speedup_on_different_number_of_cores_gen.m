figure;
hold on;

for w = 5:5:120

    R1 = fft_max_speedup_net(5:128,w,1,5);
    %R2 = direct_max_speedup_net(5:128,w,1,5);
    %R2 = direct_max_speedup_net(5:128,5,1,5);

    Ps = [6 18 40 60 120];
    Colors = { [      0    0.4470    0.7410],...
               [ 0.8500    0.3250    0.0980],...
               [ 0.9290    0.6940    0.1250],...
               [ 0.4940    0.1840    0.5560],...
               [ 0.4660    0.6740    0.1880]};

    W = [0 0 0 0 0];
    
    for j = 1:5
        P1 = R1 ./ ( 1 + (R1 + 1) / Ps(j));
        %P2 = R2 ./ ( 1 + (R2 + 1) / Ps(j));
        %plot(R1, 'color', [0.1, 0.1, 0.2 + w/40]);
        W(j) = plot(5:128, P1, 'color', Colors{j});
    end;
    
    if w == 5
        legend(W, {'6 CPUs', '18 CPUs', '40 CPUs', '60 CPUs', '120 CPUs'});           
    end;
end;


hold off;
