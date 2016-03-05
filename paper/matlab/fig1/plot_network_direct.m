function [  ] = plot_network_direct( net )
% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1,'YMinorGrid','on','YGrid','on',...
    'XMinorGrid','on',...
    'XGrid','on',...
    'Color',[0.931372549019608 0.915686274509804 0.884313725490196],...
    'YMinorTick','on',...
    'XMinorTick','on',...
    'XColor',[0 0 0],...
    'FontSize',12,...
    'AmbientLightColor',[0.8 0.8 0.8]);


box(axes1,'on');
hold(axes1,'on');
% Create ylabel
ylabel('Theoretical speedup','FontSize',16);

% Create xlabel
xlabel('Output voxels','FontSize',16);

z = net.sparse_cost(net,1,1);
x = z.direct.flops;

for bp = 0:0
    b = 2^bp;
    int = 8:8:(1000/(b^(1/3)));
    z = net.dense_cost(net,b,int);
    
    label = sprintf('Batch of %d', b);
    
    plot(int.^3*b, x./z.fft.flops, 'LineWidth',2.5,'Parent',axes1,'DisplayName','Direct convolution');
end

legend1 = legend(axes1,'show');
set(legend1,...
    'Location', 'SouthEast',...
    'Color',[1 1 1],'FontSize',12);


end

