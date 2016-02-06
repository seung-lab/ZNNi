% 2D

[~, aws8core] = load_data_threads_6('../data/aws8_2d.txt');
make_speedup_threads_chart(2:240, aws8core);

[~, aws18core] = load_data_threads_6('../data/aws18_2d.txt');
make_speedup_threads_chart(2:240, aws18core);

[~, intel40core] = load_data_threads_6('../data/intel40_2d.txt');
make_speedup_threads_chart(2:240, intel40core);

[~, xeonphi] = load_data_threads_6('../data/phi_2d.txt');
make_speedup_threads_chart(2:240, xeonphi);

% 3D

[~, aws8core] = load_data_threads_6('../data/aws8_3d.txt');
make_speedup_threads_chart(2:240, aws8core);

[~, aws18core] = load_data_threads_6('../data/aws18_3d.txt');
make_speedup_threads_chart(2:240, aws18core);

[~, intel40core] = load_data_threads_6('../data/intel40_3d.txt');
make_speedup_threads_chart(2:240, intel40core);

[~, xeonphi] = load_data_threads_6('../data/phi_3d.txt');
make_speedup_threads_chart(2:240, xeonphi);
