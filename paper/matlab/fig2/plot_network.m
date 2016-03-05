function [  ] = plot_network( net )
% Create figure
figure1 = figure('Position', [100, 100, 800, 500]);
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

[a, b] = read_data([net '.cpu']);
plot(a,1./b,'LineWidth',2.5,'Parent',axes1,'DisplayName','CPU');

if exist([net '.aws'], 'file')
    [a, b] = read_data([net '.aws']);
    plot(a,1./b,'LineWidth',2.5,'Parent',axes1,'DisplayName','CPU AWS');
end

[a, b] = read_data([net '.gpu']);

xmax = max(xmax, max(a(:)));
ymax = max(ymax, max(1./b(:)));

plot(a,1./b,'LineWidth',2.5,'Parent',axes1,'DisplayName','CuDNN Precomputed GEMM');

[a, b] = read_data([net '.gpu_no_precomp_gemm']);

xmax = max(xmax, max(a(:)));
ymax = max(ymax, max(1./b(:)));

plot(a,1./b,'LineWidth',2.5,'Parent',axes1,'DisplayName','CuDNN Implicit GEMM');


if exist([net '.gpuram'], 'file')
    [a, b] = read_data([net '.gpuram']);
    plot(a,1./b,'LineWidth',2.5,'Parent',axes1,'DisplayName','GPU RAM');
end

xmax = max(xmax, max(a(:)));
ymax = max(ymax, max(1./b(:)));

[a, b] = read_data([net '.gpufft']);
plot(a,1./b,'LineWidth',2.5,'Parent',axes1,'DisplayName','GPU FFT');

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
    'Position',[.18 .55 .13 .13],...
    'AmbientLightColor',[0.8 0.8 0.8]);

legend1 = legend(axes1,'show');
set(legend1,...
    'Location', 'SouthEast',...
    'Color',[1 1 1],'FontSize',10);


box(axes2,'on');
hold(axes2,'on');

[a, b] = read_data([net '.cpu']);
plot(a,1./b,'LineWidth',2.5,'Parent',axes2,'DisplayName','CPU');

if exist([net '.aws'], 'file')
    [a, b] = read_data([net '.aws']);
    plot(a,1./b,'LineWidth',2.5,'Parent',axes2,'DisplayName','CPU AWS');
end

[a, b] = read_data([net '.gpu']);

plot(a,1./b,'LineWidth',2.5,'Parent',axes2,'DisplayName','CuDNN Precomputed GEMM');

[a, b] = read_data([net '.gpu_no_precomp_gemm']);

plot(a,1./b,'LineWidth',2.5,'Parent',axes2,'DisplayName','CuDNN Implicit GEMM');

[a, b] = read_data([net '.gpufft']);

plot(a,1./b,'LineWidth',2.5,'Parent',axes2,'DisplayName','GPU FFT');


end

