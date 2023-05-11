#!/bin/bash
adb shell rm /data/local/tmp/cold-infer-ncnn/output.tmp.csv
for ((i=1; i<=10; i++))
do
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/alexnet 4 0 -1 0 0
done
for ((i=1; i<=10; i++))
do
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/googlenet 4 0 -1 0 0
done
for ((i=1; i<=10; i++))
do
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/mobilenet 4 0 -1 0 0
done
for ((i=1; i<=10; i++))
do
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/mobilenet_v2 4 0 -1 0 0
done
for ((i=1; i<=10; i++))
do
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/resnet18 4 0 -1 0 0
done
for ((i=1; i<=10; i++))
do
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/resnet50 4 0 -1 0 0
done
for ((i=1; i<=10; i++))
do
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/shufflenet 4 0 -1 0 0
done
for ((i=1; i<=10; i++))
do
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/shufflenet_v2 4 0 -1 0 0
done
for ((i=1; i<=10; i++))
do
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/squeezenet 4 0 -1 0 0
done
for ((i=1; i<=10; i++))
do
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/efficientnet_b0 4 0 -1 0 0
done
for ((i=1; i<=10; i++))
do
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/mobilenetv2_yolov3 4 0 -1 0 0
done
for ((i=1; i<=10; i++))
do
adb shell taskset f0 /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/mobilenet_yolo 4 0 -1 0 0
done
for ((i=1; i<=10; i++))
do
adb shell /data/local/tmp/cold-infer-ncnn/benchcold /data/local/tmp/cold-infer-ncnn/crnn_lite 4 0 -1 0 0
done
adb pull /data/local/tmp/cold-infer-ncnn/output.tmp.csv ./
chmod +x ./get_output.py
python3 ./get_output.py
rm ./output.tmp.csv


