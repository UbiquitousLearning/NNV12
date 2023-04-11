# NNV12

This repo is the official implementation of ["Boosting DNN Cold Inference on Edge Devices"](https://arxiv.org/abs/2206.07446). 
This repo is based on [ncnn](https://github.com/Tencent/ncnn/). 

---
## Benchmark on Android Phone CPU

The following instructions are executed on a PC with an operating system of Ubuntu 22.04.

1. Install NDK and adb. 

    ```shell
    $ mkdir ./NDK && cd ./NDK
    $ wget https://dl.google.com/android/repository/android-ndk-r21e-linux-x86_64.zip
    $ unzip ./android-ndk-r21e-linux-x86_64.zip
    $ echo export ANDROID_NDK=$HOME/NDK/android-ndk-r21e >> ~/.bashrc
    $ source ~/.bashrc
    $ sudo apt install adb
    ```

2. Connect PC and android phone via USB cable. 

3. Clone this repo.

    ```shell
    $ git clone --recursive https://github.com/Yeeethan00/NNV12.git 
    $ cd ./NNV12
    ```

4. Build and push models to android phone.

    ```shell
    $ cd ./scripts
    $ chmod +x ./init_arm64.sh
    $ ./init_arm64.sh
    ```
5. Deploy models.

    ```shell
    $ chmod +x ./deployed_arm64.sh
    $ ./deployed_arm64.sh
    ```

6. Run NNV12.

    ```shell
    $ chmod +x ./run_arm64.sh
    $ ./run_arm64.sh
    ```
7. The result of latency is shown in "./scripts/output.csv". 

There are results get from Meizu 16T (Snapdragon 855).
   
   | Model           | latency(ms) | 
   |-----------------|-------------|
   | alexnet         | 120.123     | 
   | googlenet       | 45.401      |
   | mobilenet       | 25.084      | 
   | mobilenet_v2    | 24.346      |
   | resnet18        | 53.672      | 
   | resnet50        | 103.136     |
   | shufflenet      | 11.576      | 
   | shufflenet_v2   | 14.033      |
   | squeezenet      | 12.139      | 
   | efficientnet_b0 | 25.235      |

---
## Benchmark on Jetson Nano GPU

The following instructions are executed on a Jetson Nano.

1. Install Vulkan and CMake.

    ```shell
    $ cd ./scripts
    $ chmod +x ./install_jetson.sh
    $ ./install_jetson.sh
    ```
   
2. Clone this repo.

    ```shell
    $ git clone --recursive https://github.com/Yeeethan00/NNV12.git 
    $ cd ./NNV12
    ```   

3. Build project.

    ```shell
    $ chmod +x ./init_jetson.sh
    $ ./init_jetson.sh
    ```

4. Deploy models.

    ```shell
    $ chmod +x ./deployed_jetson.sh
    $ ./deployed_jetson.sh
    ```

5. Run NNV12.

    ```shell
    $ chmod +x ./run_jetson.sh
    $ ./run_jetson.sh
    ```

6. The result of latency is shown in "./build/benchmark/output.csv". 

There are results get from Jetson Nano.

   | Model           | latency(ms) | 
   |-----------------|-------------|
   | alexnet         | 169.942     | 
   | googlenet       | 94.181      |
   | mobilenet       | 50.916      | 
   | mobilenet_v2    | 63.992      |
   | resnet18        | 87.460      | 
   | resnet50        | 186.534     |
   | shufflenet      | 40.941      | 
   | shufflenet_v2   | 45.309      |
   | squeezenet      | 53.929      | 
   | efficientnet_b0 | 107.834     |