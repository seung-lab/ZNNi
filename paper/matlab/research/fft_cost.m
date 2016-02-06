function [ ratio ] = fft_cost( k, N )

old = 3 * N .* N;
new = k * k + k * N + N .* N;

ratio = new ./ old;

end

