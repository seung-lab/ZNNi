function [  ] = plot_available_results( net )
% Create figure
figure1 = figure('Position', [100, 100, 800, 500]);

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

if exist([net '.behir'], 'file')
    [a, b] = read_data([net '.behir']);
    plot(a,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','72 Core CPU');
end

if exist([net '.aws'], 'file')
    [a, b] = read_data([net '.aws']);
    plot(a,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','CPU AWS');
end

if exist([net '.ram'], 'file')
    [a, b] = read_data([net '.ram']);
    plot(a,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','GPU + Host RAM');
end

if exist([net '.gpu.optimal'], 'file')
    [a, b] = read_data([net '.gpu.optimal']);
    plot(a,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','Titan X (Optimal)');
end

if exist([net '.gpu.cudnn'], 'file')
    [a, b] = read_data([net '.gpu.cudnn']);
    plot(a,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','CuDNN Precomputed GEMM');
end

if exist([net '.gpu.cudnn_np'], 'file')
    [a, b] = read_data([net '.gpu.cudnn_np']);
    plot(a,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','CuDNN GEMM');
end

if exist([net '.gpu.fft'], 'file')
    [a, b] = read_data([net '.gpu.fft']);
    plot(a,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','GPU FFT');
end

legend1 = legend(axes1,'show');
set(legend1,...
    'Location', 'SouthEast',...
    'Color',[1 1 1],'FontSize',10);
