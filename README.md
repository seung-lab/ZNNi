# ZNNi
ZNNi - Maximizing 3D ConvNet Inference Throughput Using Multi-Core, Many-Core CPUs and GPUs

## Installation

### Nvidia docker
We recommend to use nvidia-docker to install ZNNi. For the image, please contact Jingpeng Wu <jingpeng.wu@gmail.com>.

### dependency

| Library       | Components    | Notes        |
| ------------- |:-------------:| ------------:|
| intel compiler| tbb, icpc     | version > 13 |
| cuda7.5       |               |              |
| cudnn7.0      |               |              |

we use gcc4.8, 4.7/4.9 do not work!
