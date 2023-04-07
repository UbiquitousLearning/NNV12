#!/bin/bash

adb shell rm /data/local/tmp/cold-infer-ncnn/output.csv
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/alexnet 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/googlenet 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/mobilenet 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/mobilenet_v2 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/resnet18 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/resnet50 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/shufflenet 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/shufflenet_v2 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/squeezenet 4 0 -1 0 0
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/efficientnet_b0 4 0 -1 0 0
adb pull /data/local/tmp/cold-infer-ncnn/output.csv ./


