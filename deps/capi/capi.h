#define DTYPE float

extern "C" {
    struct znnilayer;
    znnilayer* crop_layer(int ni, int no, int* insz, int* outsz);

    znnilayer* conv_layer(int n, int ni, int no, int* insz,
                        int* knsz, DTYPE* kernel, DTYPE* biases, int act);
}
