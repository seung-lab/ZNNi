function [  ] = flops_per_output_pixel( net )
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
% ylim([-100 150000]);

% Create xlabel
xlabel('Memory required (GB)','FontSize',16);
% xlim([0 512]);

z = net.sparse_cost(net,1,1);
x = z.fft.flops;
xd = z.direct.flops;

int = 8:8:800;
z = net.dense_cost(net,1,int);
    
label = 'FFT Conv + Max Frag Pooling';
plot(int, x./z.fft.flops, 'LineWidth',2.5,'Parent',axes1,'DisplayName',label);

int = 1:1:60;
z = net.sparse_cost(net,1,int);
    
label = 'FFT Conv + Pooling';
plot(int, x./z.fft.flops, 'LineWidth',2.5,'Parent',axes1,'DisplayName',label);
% 
% int = 8:8:800;
% z = net.dense_cost(net,1,int);
%     
% label = 'Direct Conv + Max Frag Pooling';
% plot(z.direct.gb, xd./z.direct.flops, 'LineWidth',2.5,'Parent',axes1,'DisplayName',label);
% 
% int = 1:1:200;
% z = net.sparse_cost(net,1,int);
%     
% label = 'Direct Conv + Pooling';
% plot(z.direct.gb, xd./z.direct.flops, 'LineWidth',2.5,'Parent',axes1,'DisplayName',label);
% 


legend1 = legend(axes1,'show');
set(legend1,...
    'Location', 'NorthWest',...
    'Color',[1 1 1],'FontSize',12);


end

