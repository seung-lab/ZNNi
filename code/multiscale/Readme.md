* Q: What version of CUDA and CUDNN do I need?
* A: CUDA 7.5 and CUDNN 4 (for CUDA 7.0 and above)

* Q: I get an `error while loading shared libraries: libcudart.so.7.5: cannot open shared object file: No such file or directory` when starting the executable
* A: Try `sudo ldconfig /usr/local/cuda/lib64`

* Q: The output contains only NaN.
* A: Make sure the network directory is in the correct directory, i.e. with the executable. You should see a bunch of `NREAD ...` messages when starting the program.

* Q: ZNNi crashes when creating a`cudnn_conv` layer.
* A: `cudnn_conv` is not supported by your hardware. Try using the `cudnn_no_precomp_gemm_conv` primitive.