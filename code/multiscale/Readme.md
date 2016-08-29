
## Build
    cd ..
    make bin/multiscale/znni_gpu

## Usage
    ./znni_gpu device_id path/of/normalized/image path/of/output/affinity/map dataset_name [output patch size: z,y,x]

example:

    ./znni_gpu 0 ../../../data/raw.float.h5 out.h5 main 17 256 256

## FAQ
* Q: What version of CUDA and CUDNN do I need?
* A: CUDA 8.0 and CUDNN 5 (for CUDA 8.0 and above)

* Q: I get an `error while loading shared libraries: libcudart.so.8.0: cannot open shared object file: No such file or directory` when starting the executable
* A: Try `sudo ldconfig /usr/local/cuda/lib64`

* Q: The output contains only NaN.
* A: Make sure the network directory is in the correct directory, i.e. with the executable. You should see a bunch of `NREAD ...` messages when starting the program.

* Q: ZNNi crashes when creating a`cudnn_conv` layer.
* A: `cudnn_conv` is not supported by your hardware. Try using the `cudnn_no_precomp_gemm_conv` primitive.

