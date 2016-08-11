net = './data/m36';
[~, Y1, X1] = read_data([net '.behir']);
[~, Y2, X2] = read_data([net '.gpu']);
[~, Y3, X3] = read_data([net '.ram']);
[~, Y4, X4] = read_data([net '.fusion.4']);


X1 = X1 / 1000000000;
X2 = X2 / 1000000000;
X3 = X3 / 1000000000;
X4 = X4 / 1000000000;

Y1 = Y1 / 1000000;
Y2 = Y2 / 1000000;
Y3 = Y3 / 1000000;
Y4 = Y4 / 1000000;

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1,'YGrid','on','XGrid','on','XColor',[0 0 0],...
    'XLim', [0 256],...
    'TickDir','out',...
    'FontSize',14,'FontWeight','normal',...
    'ColorOrder',[0 0.446999996900558 0.74099999666214;0.929000020027161 0.694000005722046 0.125;0.493999987840652 0.184000000357628 0.555999994277954;0.465999990701675 0.674000024795532 0.187999993562698;0.300999999046326 0.745000004768372 0.933000028133392;0.634999990463257 0.0780000016093254 0.184000000357628;0.850000023841858 0.324999988079071 0.0979999974370003],...
    'FontSize',12,...
    'XTickLabel','',...
    'Position',[0.12 0.55 0.38 0.35]);
box(axes1,'on');
hold(axes1,'on');

% Create ylabel
ylabel('Throughput (MVox/s)','FontSize',20,'Position',[-44 -0.1 0]);

% Create xlabel
xlabel('Memory Consumed (GB)','FontSize',20,'Position',[275 -1.55 0]);

plot(X1,Y1,'DisplayName','CPU (72 cores)','LineWidth',3);
plot(X3,Y3,'DisplayName','GPU+Host RAM','LineWidth',3);
plot(X4,Y4,'DisplayName','CPU+GPU','LineWidth',3);
plot(X2,Y2,'DisplayName','GPU','LineWidth',3);


% Create legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.3 0.94 0.420712373167982 0.0237804878048781],...
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
[~, Y1, X1] = read_data([net '.behir']);
[~, Y2, X2] = read_data([net '.gpu']);
[~, Y3, X3] = read_data([net '.ram']);
[~, Y4, X4] = read_data([net '.fusion.4']);


X1 = X1 / 1000000000;
X2 = X2 / 1000000000;
X3 = X3 / 1000000000;
X4 = X4 / 1000000000;

Y1 = Y1 / 1000000;
Y2 = Y2 / 1000000;
Y3 = Y3 / 1000000;
Y4 = Y4 / 1000000;


axes2 = axes('Parent',figure1,'YGrid','on','XGrid','on','XColor',[0 0 0],...
    'XLim', [0 256],...
    'TickDir','out',...
    'FontSize',14,'FontWeight','normal',...
    'ColorOrder',[0 0.446999996900558 0.74099999666214;0.929000020027161 0.694000005722046 0.125;0.493999987840652 0.184000000357628 0.555999994277954;0.465999990701675 0.674000024795532 0.187999993562698;0.300999999046326 0.745000004768372 0.933000028133392;0.634999990463257 0.0780000016093254 0.184000000357628;0.850000023841858 0.324999988079071 0.0979999974370003],...
    'FontSize',12,...
    'XTickLabel','',...
    'Position',[0.58 0.55 0.38 0.35]);

box(axes2,'on');
hold(axes2,'on');


plot(X1,Y1,'DisplayName','CPU (72 cores)','LineWidth',3);
plot(X3,Y3,'DisplayName','GPU+Host RAM','LineWidth',3);
plot(X4,Y4,'DisplayName','CPU+GPU','LineWidth',3);
plot(X2,Y2,'DisplayName','GPU','LineWidth',3);

% Create textbox
annotation(figure1,'textbox',...
    [0.72 0.86 0.108357384441939 0.0280487804878049],...
    'String','n537',...
    'FontSize',18,...
    'FitBoxToText','off',...
    'EdgeColor',[1 1 1]);

net = './data/m76';
[~, Y1, X1] = read_data([net '.behir']);
[~, Y2, X2] = read_data([net '.gpu']);
[~, Y3, X3] = read_data([net '.ram']);
[~, Y4, X4] = read_data([net '.fusion.4']);



X1 = X1 / 1000000000;
X2 = X2 / 1000000000;
X3 = X3 / 1000000000;
X4 = X4 / 1000000000;

Y1 = Y1 / 1000000;
Y2 = Y2 / 1000000;
Y3 = Y3 / 1000000;
Y4 = Y4 / 1000000;

axes3 = axes('Parent',figure1,'YGrid','on','XGrid','on','XColor',[0 0 0],...
    'XLim', [0 256],...
    'TickDir','out',...
    'FontSize',14,'FontWeight','normal',...
    'ColorOrder',[0 0.446999996900558 0.74099999666214;0.929000020027161 0.694000005722046 0.125;0.493999987840652 0.184000000357628 0.555999994277954;0.465999990701675 0.674000024795532 0.187999993562698;0.300999999046326 0.745000004768372 0.933000028133392;0.634999990463257 0.0780000016093254 0.184000000357628;0.850000023841858 0.324999988079071 0.0979999974370003],...
    'FontSize',12,...
    'Position',[0.12 0.16 0.38 0.35]);

box(axes3,'on');
hold(axes3,'on');


plot(X1,Y1,'DisplayName','CPU (72 cores)','LineWidth',3);
plot(X3,Y3,'DisplayName','GPU+Host RAM','LineWidth',3);
plot(X4,Y4,'DisplayName','CPU+GPU','LineWidth',3);
plot(X2,Y2,'DisplayName','GPU','LineWidth',3);


% Create textbox
annotation(figure1,'textbox',...
    [0.26 0.47 0.108357384441939 0.0280487804878049],...
    'String','n726',...
    'FontSize',18,...
    'FitBoxToText','off',...
    'EdgeColor',[1 1 1]);


net = './data/m96';
[~, Y1, X1] = read_data([net '.behir']);
[~, Y2, X2] = read_data([net '.gpu']);
[~, Y3, X3] = read_data([net '.ram']);
[~, Y4, X4] = read_data([net '.fusion.4']);



X1 = X1 / 1000000000;
X2 = X2 / 1000000000;
X3 = X3 / 1000000000;
X4 = X4 / 1000000000;

Y1 = Y1 / 1000000;
Y2 = Y2 / 1000000;
Y3 = Y3 / 1000000;
Y4 = Y4 / 1000000;

axes4 = axes('Parent',figure1,'YGrid','on','XGrid','on','XColor',[0 0 0],...
    'XLim', [0 256],...
    'TickDir','out',...
    'FontSize',14,'FontWeight','normal',...
    'ColorOrder',[0 0.446999996900558 0.74099999666214;0.929000020027161 0.694000005722046 0.125;0.493999987840652 0.184000000357628 0.555999994277954;0.465999990701675 0.674000024795532 0.187999993562698;0.300999999046326 0.745000004768372 0.933000028133392;0.634999990463257 0.0780000016093254 0.184000000357628;0.850000023841858 0.324999988079071 0.0979999974370003],...
    'FontSize',12,...
    'Position',[0.58 0.16 0.38 0.35]);

box(axes4,'on');
hold(axes4,'on');


plot(X1,Y1,'DisplayName','CPU (72 cores)','LineWidth',3);
plot(X3,Y3,'DisplayName','GPU+Host RAM','LineWidth',3);
plot(X4,Y4,'DisplayName','CPU+GPU','LineWidth',3);
plot(X2,Y2,'DisplayName','GPU','LineWidth',3);



% Create textbox
annotation(figure1,'textbox',...
    [0.72 0.47 0.108357384441939 0.0280487804878049],...
    'String','n926',...
    'FontSize',18,...
    'FitBoxToText','off',...
    'EdgeColor',[1 1 1]);
