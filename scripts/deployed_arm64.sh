#!/bin/bash

adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcolddeploy /data/local/tmp/cold-infer-ncnn/alexnet 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcolddeploy /data/local/tmp/cold-infer-ncnn/googlenet 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcolddeploy /data/local/tmp/cold-infer-ncnn/mobilenet 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcolddeploy /data/local/tmp/cold-infer-ncnn/mobilenet_v2 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcolddeploy /data/local/tmp/cold-infer-ncnn/resnet18 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcolddeploy /data/local/tmp/cold-infer-ncnn/resnet50 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcolddeploy /data/local/tmp/cold-infer-ncnn/shufflenet 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcolddeploy /data/local/tmp/cold-infer-ncnn/shufflenet_v2 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcolddeploy /data/local/tmp/cold-infer-ncnn/squeezenet 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcolddeploy /data/local/tmp/cold-infer-ncnn/efficientnet_b0 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcolddeploy /data/local/tmp/cold-infer-ncnn/mobilenetv2_yolov3 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcolddeploy /data/local/tmp/cold-infer-ncnn/mobilenet_yolo 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcolddeploy /data/local/tmp/cold-infer-ncnn/crnn_lite 4 0 -1 0 0