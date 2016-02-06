%[a,b1] = process_data3('measure_aws_n.txt');

	
[a,b1,s1] = load_data_6('../data/aws8_3d.txt');
[~,b2,s2] = load_data_6('../data/aws18_3d.txt');
[~,b3,s3] = load_data_6('../data/intel40_3d.txt');
[~,b4,s4] = load_data_6('../data/phi_3d.txt');

r1 = s2 ./ b1;
r2 = s2 ./ b2;
r3 = s2 ./ b3;
r4 = s2 ./ b4;

makeplot(a,[r1 r2 r3 r4], 'Achieved speedup vs one core');

b1 = s1 ./ b1;
b2 = s2 ./ b2;
b3 = s3 ./ b3;
b4 = s4 ./ b4;

makeplot(a,[b1 b2 b3 b4], 'Achieved speedup');
%makeplot([a'/sqrt(8) a'/sqrt(18) a'/sqrt(40) a'/sqrt(60)],[b1 b2 b3 b4], 'Achieved speedup');
%makeplot(a,[b1 b2 b3 b4], 'Achieved speedup');

makeplot(a,[100*b1/8 b2*100/18 b3*100/40 b4*100/60], 'Efficiency (%)');
%makeplot([a'/sqrt(8) a'/sqrt(18) a'/sqrt(40) a'/sqrt(60)],[100*b1/8 100*b2/18 100*b3/40 100*b4/60], 'Efficiency (%)');
%makeplot([a'/sqrt(8) a'/sqrt(18) a'/sqrt(40) a'/sqrt(60)],[100*b1/8 100*b2/18 100*b3/40 100*b4/60], 'Efficiency (%)');

% for k = 80:20:120
% 
%     b1 = load_data_raw_speed('../data/aws8_3d.txt', k);
%     b2 = load_data_raw_speed('../data/aws18_3d.txt', k);
%     b3 = load_data_raw_speed('../data/intel40_3d.txt', k);
%     b4 = load_data_raw_speed('../data/phi_3d.txt', k);
% 
%     makeplot2(1:240,[b1 b2 b3 b4], 'Speed');
% end;