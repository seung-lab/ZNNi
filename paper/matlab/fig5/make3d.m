% 3D

ZNN3    = [0.100 0.140 0.154 0.200 0.230];
Theano3 = [0.045 0.054 0.078 0.108 0.142];

YLim3   = max(ZNN3(:));

ZNN5    = [0.290 0.304 0.410 0.520 0.650];
Theano5 = [0.247 0.286 0.392 0.521 0.731];

YLim5   = max(Theano5(:));

ZNN7    = [0.900 1.310 1.500 1.600 1.950];
Theano7 = [1.288 1.483 1.825 2.259 2.732];

YLim7   = max(Theano7(:));

% subplotplus demo

cell31={{['-g']};{['-g']};{['-g']}};

figure(1)
C = {cell31};
[h,labelfontsize] = subplotplus(C);
x = 1:6;

subidx = 1;
set(gcf,'CurrentAxes',h(subidx));
b = bar([ZNN3; Theano3]')
set(b(1),'DisplayName','ZNN','FaceColor',[0 0.447 0.741],...
    'EdgeColor',[0 0.447 0.741]);
set(b(2),'DisplayName','Theano','FaceColor',[0.85 0.325 0.098],...
    'EdgeColor',[0.85 0.325 0.098]);

set(h(subidx), ...
    'XGrid', 'on', ...
    'YGrid', 'on', ...
    'Box', 'on', ...
    'XTick', '', ...
    'XMinorTick', 'off', ...
    'YMinorTick', 'off', ...
    'XMinorGrid', 'off', ...
    'YMinorGrid', 'off', ...
    'FontSize', 18,...
    'YLim', [0 YLim3 * 1.2],...
    'Color',[0.831372549019608 0.815686274509804 0.784313725490196], ...
    'AmbientLightColor',[0.466666666666667 0.674509803921569 0.188235294117647]);

legend1 = legend(h(subidx),'show');
set(legend1,...
    'Position',[0.108723483120437 0.764331210191083 0.124853159215329 0.0955414052601812],...
    'Orientation','horizontal',...
    'EdgeColor',[0.972549021244049 0.972549021244049 0.972549021244049],...
    'FontSize',18,...
    'Color',[1 1 1]);


set(gca,'XTickLabel',{'1x1x1', '2x2x2', '4x4x4', '6x6x6', '8x8x8', '16x16x16'})
%xlabel('x [1/(2*pi)rad]','FontSize',12);
xlabel('Output Size','FontSize',20);
%ylabel('Seconds / Update','FontSize',20);
set(gca,'XTickLabel',{'1x1x1', '2x2x2', '4x4x4', '6x6x6', '8x8x8', '16x16x16'})

subidx = subidx +1;
set(gcf,'CurrentAxes',h(subidx));
b = bar([ZNN5; Theano5]')
set(b(1),'DisplayName','ZNN','FaceColor',[0 0.447 0.741],...
    'EdgeColor',[0 0.447 0.741]);
set(b(2),'DisplayName','Theano','FaceColor',[0.85 0.325 0.098],...
    'EdgeColor',[0.85 0.325 0.098]);
set(h(subidx), ...
    'XGrid', 'on', ...
    'YGrid', 'on', ...
    'Box', 'on', ...
    'XTick', '', ...
    'XMinorTick', 'off', ...
    'YMinorTick', 'off', ...
    'XMinorGrid', 'off', ...
    'YMinorGrid', 'off', ...
    'YLim', [0 YLim5 * 1.2],...
    'FontSize', 18,...
    'YAxisLocation', 'right',...
    'Color',[0.831372549019608 0.815686274509804 0.784313725490196], ...
    'AmbientLightColor',[0.466666666666667 0.674509803921569 0.188235294117647]);
xlabel('Output Size','FontSize',20);
ylabel('Seconds / Update','FontSize',20);
set(gca,'XTickLabel',{'1x1x1', '2x2x2', '4x4x4', '6x6x6', '8x8x8', '16x16x16'})

subidx = subidx +1;
set(gcf,'CurrentAxes',h(subidx));
b = bar([ZNN7; Theano7]')
set(b(1),'DisplayName','ZNN','FaceColor',[0 0.447 0.741],...
    'EdgeColor',[0 0.447 0.741]);
set(b(2),'DisplayName','Theano','FaceColor',[0.85 0.325 0.098],...
    'EdgeColor',[0.85 0.325 0.098]);
set(h(subidx), ...
    'XGrid', 'on', ...
    'YGrid', 'on', ...
    'Box', 'on', ...
    'XTick', '', ...
    'XMinorTick', 'off', ...
    'YMinorTick', 'off', ...
    'XMinorGrid', 'off', ...
    'YMinorGrid', 'off', ...
    'FontSize', 18,...
    'YLim', [0 YLim7 * 1.2],...
    'Color',[0.831372549019608 0.815686274509804 0.784313725490196], ...
    'AmbientLightColor',[0.466666666666667 0.674509803921569 0.188235294117647]);
xlabel('Output Size','FontSize',20);
%ylabel('Seconds / Update','FontSize',20);
set(gca,'XTickLabel',{'1x1x1', '2x2x2', '4x4x4', '6x6x6', '8x8x8', '16x16x16'})