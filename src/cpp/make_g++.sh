g++ -Wall -Wextra -std=c++11  -O2 -DNDEBUG $1.cpp -o $1_g++ -pthread -lfftw3f -I../include -I../../ -I/usr/local/cuda/include
