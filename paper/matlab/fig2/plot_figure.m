function [  ] = plot_figure( net, subpos, legentpos )
% Create figure
figure1 = figure('Position', [100, 100, 650, 400]);
xmax = 0;
ymax = 0;

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
ylabel('Voxels per second','FontSize',16);

% Create xlabel
xlabel('Output size','FontSize',16);

[a, b] = read_data([net '.behir']);
plot(a,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','72 Core CPU');

if exist([net '.aws'], 'file')
    [a, b] = read_data([net '.aws']);
    plot(a,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','CPU AWS');
end

if exist([net '.gpu.cudnn'], 'file')
    [a, b] = read_data([net '.gpu.cudnn']);
    xmax = max(xmax, max(a(:)));
    ymax = max(ymax, max(b(:)));
    plot(a,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','CuDNN Precomputed GEMM');
end

if exist([net '.gpu.cudnn_np'], 'file')
    [a, b] = read_data([net '.gpu.cudnn_np']);
    xmax = max(xmax, max(a(:)));
    ymax = max(ymax, max(b(:)));
    plot(a,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','CuDNN GEMM');
end

if exist([net '.gpu.fft'], 'file')
    [a, b] = read_data([net '.gpu.fft']);
    xmax = max(xmax, max(a(:)));
    ymax = max(ymax, max(b(:)));
    plot(a,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','GPU FFT');
end

axes2 = axes('Parent',figure1,'YMinorGrid','on','YGrid','on',...
    'XMinorGrid','on',...
    'XGrid','on',...
    'Color',[0.831372549019608 0.815686274509804 0.784313725490196],...
    'YMinorTick','on',...
    'XMinorTick','on',...
    'XColor',[0 0 0],...
    'FontSize',10,...
    'XLim', [0 xmax * 1.1],...
    'YLim', [0 ymax * 1.1],...
    'Position',subpos,...
    'AmbientLightColor',[0.8 0.8 0.8]);

legend1 = legend(axes1,'show');
set(legend1,...
    'Location', legentpos,...
    'Color',[1 1 1],'FontSize',12);


box(axes2,'on');
hold(axes2,'on');

[a, b] = read_data([net '.behir']);
plot(a,b,'LineWidth',2.5,'Parent',axes2,'DisplayName','CPU');

if exist([net '.aws'], 'file')
    [a, b] = read_data([net '.aws']);
    plot(a,b,'LineWidth',2.5,'Parent',axes2,'DisplayName','CPU AWS');
end

if exist([net '.gpu.cudnn'], 'file')
    [a, b] = read_data([net '.gpu.cudnn']);
    plot(a,b,'LineWidth',2.5,'Parent',axes2,'DisplayName','CuDNN Precomputed GEMM');
end

if exist([net '.gpu.cudnn_np'], 'file')
    [a, b] = read_data([net '.gpu.cudnn_np']);
    plot(a,b,'LineWidth',2.5,'Parent',axes2,'DisplayName','CuDNN GEMM');
end

if exist([net '.gpu.fft'], 'file')
    [a, b] = read_data([net '.gpu.fft']);
    plot(a,b,'LineWidth',2.5,'Parent',axes2,'DisplayName','GPU FFT');
end


end

