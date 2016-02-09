g++ -O2 -DNDEBUG -I. -I../..  $1.cpp -o $1 -lfftw3f -lpthread  -L/usr/local/cuda/lib -I/usr/local/cuda/include  -lcudart -lcudnn -std=c++11 -lcublas
