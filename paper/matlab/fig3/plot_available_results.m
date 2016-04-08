function [  ] = plot_available_results( net )
% Create figure
figure1 = figure('Position', [100, 100, 600, 400]);

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

GB = 1000 * 1000 * 1000;

box(axes1,'on');
hold(axes1,'on');
% Create ylabel
ylabel('Voxels per second','FontSize',20);

% Create xlabel
xlabel('Memory consumed (GB)','FontSize',20);

if exist([net '.behir'], 'file')
    [~, b, c] = read_data([net '.behir']);
    plot(c/GB,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','CPU Only');
end

if exist([net '.aws'], 'file')
    [~, b, c] = read_data([net '.aws']);
    plot(c/GB,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','CPU AWS');
end

if exist([net '.ram'], 'file')
    [~, b, c] = read_data([net '.ram']);
    plot(c/GB,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','GPU + Host RAM');
end


if exist([net '.fusion.4'], 'file')
    [~, b, c] = read_data([net '.fusion.4']);
    plot(c/GB,b,'LineWidth',2.5,'Parent',axes1,'DisplayName','CPU + GPU');
end


if exist([net '.gpu'], 'file')
    [~, b, c] = read_data([net '.gpu']);
    plot(c/GB,b,'LineWidth',3.5,'Parent',axes1,'DisplayName','GPU Only');
end


legend1 = legend(axes1,'show');
set(legend1,...
    'Location', 'SouthEast',...
    'Color',[1 1 1],'FontSize',12);
