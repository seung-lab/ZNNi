net = './data/m36';
[X1, Y1] = read_data([net '.behir']);
[X2, Y2] = read_data([net '.aws']);
[X3, Y3] = read_data([net '.gpu.optimal']);

X1 = X1 + 84;
X2 = X2 + 84;
X3 = X3 + 84;

Y1 = Y1 / 1000000;
Y2 = Y2 / 1000000;
Y3 = Y3 / 1000000;

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1,'YGrid','on','XGrid','on','XColor',[0 0 0],...
    'XLim', [50 750],...
    'TickDir','out',...
    'FontSize',14,'FontWeight','normal',...
    'ColorOrder',[0 0.446999996900558 0.74099999666214;0.850000023841858 0.324999988079071 0.0979999974370003;0.465999990701675 0.674000024795532 0.187999993562698;0.929000020027161 0.694000005722046 0.125;0.493999987840652 0.184000000357628 0.555999994277954;0.300999999046326 0.745000004768372 0.933000028133392;0.634999990463257 0.0780000016093254 0.184000000357628],...
    'FontSize',12,...
    'XTickLabel','',...
    'Position',[0.12 0.55 0.38 0.35]);
box(axes1,'on');
hold(axes1,'on');

% Create ylabel
ylabel('Throughput (MVox/s)','FontSize',20,'Position',[-95.23811246141975 -0.13220338387004394 0]);

% Create xlabel
xlabel('Input size','FontSize',20,'Position',[800 -1 0]);

% Create plot
plot(X1,Y1,'DisplayName','CPU (72 cores)','LineWidth',3);

% Create plot
plot(X2,Y2,'DisplayName','CPU (16 cores)','LineWidth',3);

% Create plot
plot(X3,Y3,'DisplayName','GPU','LineWidth',3);

% Create legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.4 0.94 0.420712373167982 0.0237804878048781],...
    'Orientation','horizontal',...
    'FontSize',16,...
    'EdgeColor',[1 1 1]);

% Create textbox
annotation(figure1,'textbox',...
    [0.26 0.86 0.108357384441939 0.0280487804878049],...
    'String','n337',...
    'FontSize',18,...
    'FitBoxToText','off',...
    'EdgeColor',[1 1 1]);


net = './data/m56';
[X1, Y1] = read_data([net '.behir']);
[X2, Y2] = read_data([net '.aws']);
[X3, Y3] = read_data([net '.gpu.optimal']);

X1 = X1 + 162;
X2 = X2 + 162;
X3 = X3 + 162;

Y1 = Y1 / 1000000;
Y2 = Y2 / 1000000;
Y3 = Y3 / 1000000;


axes2 = axes('Parent',figure1,'YGrid','on','XGrid','on','XColor',[0 0 0],...
    'XLim', [50 750],...
    'TickDir','out',...
    'FontSize',14,'FontWeight','normal',...
    'ColorOrder',[0 0.446999996900558 0.74099999666214;0.850000023841858 0.324999988079071 0.0979999974370003;0.465999990701675 0.674000024795532 0.187999993562698;0.929000020027161 0.694000005722046 0.125;0.493999987840652 0.184000000357628 0.555999994277954;0.300999999046326 0.745000004768372 0.933000028133392;0.634999990463257 0.0780000016093254 0.184000000357628],...
    'FontSize',12,...
    'XTickLabel','',...
    'Position',[0.58 0.55 0.38 0.35]);

box(axes2,'on');
hold(axes2,'on');

% Create plot
plot(X1,Y1,'DisplayName','CPU (72c)','LineWidth',3);

% Create plot
plot(X2,Y2,'DisplayName','CPU (16c)','LineWidth',3);

% Create plot
plot(X3,Y3,'DisplayName','GPU','LineWidth',3);

% Create textbox
annotation(figure1,'textbox',...
    [0.72 0.86 0.108357384441939 0.0280487804878049],...
    'String','n537',...
    'FontSize',18,...
    'FitBoxToText','off',...
    'EdgeColor',[1 1 1]);

net = './data/m76';
[X1, Y1] = read_data([net '.behir']);
[X2, Y2] = read_data([net '.aws']);
[X3, Y3] = read_data([net '.gpu.optimal']);

X1 = X1 + 116;
X2 = X2 + 116;
X3 = X3 + 116;

Y1 = Y1 / 1000000;
Y2 = Y2 / 1000000;
Y3 = Y3 / 1000000;


axes3 = axes('Parent',figure1,'YGrid','on','XGrid','on','XColor',[0 0 0],...
    'XLim', [50 750],...
    'TickDir','out',...
    'FontSize',14,'FontWeight','normal',...
    'ColorOrder',[0 0.446999996900558 0.74099999666214;0.850000023841858 0.324999988079071 0.0979999974370003;0.465999990701675 0.674000024795532 0.187999993562698;0.929000020027161 0.694000005722046 0.125;0.493999987840652 0.184000000357628 0.555999994277954;0.300999999046326 0.745000004768372 0.933000028133392;0.634999990463257 0.0780000016093254 0.184000000357628],...
    'FontSize',12,...
    'Position',[0.12 0.16 0.38 0.35]);

box(axes3,'on');
hold(axes3,'on');

% Create plot
plot(X1,Y1,'DisplayName','CPU (72c)','LineWidth',3);

% Create plot
plot(X2,Y2,'DisplayName','CPU (16c)','LineWidth',3);

% Create plot
plot(X3,Y3,'DisplayName','GPU','LineWidth',3);

% Create textbox
annotation(figure1,'textbox',...
    [0.26 0.46 0.108357384441939 0.0280487804878049],...
    'String','n726',...
    'FontSize',18,...
    'FitBoxToText','off',...
    'EdgeColor',[1 1 1]);


net = './data/m96';
[X1, Y1] = read_data([net '.behir']);
[X2, Y2] = read_data([net '.aws']);
[X3, Y3] = read_data([net '.gpu.optimal']);

X1 = X1 + 154;
X2 = X2 + 154;
X3 = X3 + 154;

Y1 = Y1 / 1000000;
Y2 = Y2 / 1000000;
Y3 = Y3 / 1000000;


axes4 = axes('Parent',figure1,'YGrid','on','XGrid','on','XColor',[0 0 0],...
    'XLim', [50 750],...
    'TickDir','out',...
    'FontSize',14,'FontWeight','normal',...
    'ColorOrder',[0 0.446999996900558 0.74099999666214;0.850000023841858 0.324999988079071 0.0979999974370003;0.465999990701675 0.674000024795532 0.187999993562698;0.929000020027161 0.694000005722046 0.125;0.493999987840652 0.184000000357628 0.555999994277954;0.300999999046326 0.745000004768372 0.933000028133392;0.634999990463257 0.0780000016093254 0.184000000357628],...
    'FontSize',12,...
    'Position',[0.58 0.16 0.38 0.35]);

box(axes4,'on');
hold(axes4,'on');

% Create plot
plot(X1,Y1,'DisplayName','CPU (72c)','LineWidth',3);

% Create plot
plot(X2,Y2,'DisplayName','CPU (16c)','LineWidth',3);

% Create plot
plot(X3,Y3,'DisplayName','GPU','LineWidth',3);


% Create textbox
annotation(figure1,'textbox',...
    [0.72 0.46 0.108357384441939 0.0280487804878049],...
    'String','n926',...
    'FontSize',18,...
    'FitBoxToText','off',...
    'EdgeColor',[1 1 1]);
