g++ -Wall -Wextra -std=c++11  -O3 -DNDEBUG $1.cpp -o $1_g++ -pthread -lfftw3f -I../../assets/vectorclass -I../include -I../../ -I/usr/local/cuda/include -ltbb -ltbbmalloc -ltbbmalloc_proxy
