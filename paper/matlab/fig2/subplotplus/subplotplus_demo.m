% subplotplus demo

cell31={{['-g']};{['-g']};{['-g']}};

figure(1)
C = {cell31};
[h,labelfontsize] = subplotplus(C);
x = [0:1e-3:1];

subidx = 1;
set(gcf,'CurrentAxes',h(subidx));
plot(x,sin(2*pi*16*x),'k-');
set(h(subidx),'XGrid','on','YGrid','on','Box','on','XMinorTick','on','YMinorTick','on','XMinorGrid','off','YMinorGrid','off');
xlabel('x [1/(2*pi)rad]','FontSize',12);	
ylabel('sin(2pi16x)','FontSize',12);

subidx = subidx +1;
set(gcf,'CurrentAxes',h(subidx));
plot(x,sin(2*pi*32*x),'b-');
set(h(subidx),'XGrid','on','YGrid','on','Box','on','XMinorTick','on','YMinorTick','on','XMinorGrid','off','YMinorGrid','off');
xlabel('x [1/(2*pi)rad]','FontSize',labelfontsize(subidx,1));	
ylabel('sin(2pi32x)','FontSize',labelfontsize(subidx,2));

subidx = subidx +1;
set(gcf,'CurrentAxes',h(subidx));
plot(x,sin(2*pi*64*x),'r-');
set(h(subidx),'XGrid','on','YGrid','on','Box','on','XMinorTick','on','YMinorTick','on','XMinorGrid','off','YMinorGrid','off');
xlabel('x [1/(2*pi)rad]','FontSize',labelfontsize(subidx,1));	
ylabel('sin(2pi64x)','FontSize',labelfontsize(subidx,2));


