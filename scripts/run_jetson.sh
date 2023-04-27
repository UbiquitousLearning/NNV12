#!/bin/bash

cd ../build/benchmark || exit
rm ./output.tmp.csv
for ((i=1; i<=10; i++))
do
./benchcold alexnet 2 0 0 0 1
done
for ((i=1; i<=10; i++))
do
./benchcold googlenet 2 0 0 0 1
done
for ((i=1; i<=10; i++))
do
./benchcold mobilenet 2 0 0 0 1
done
for ((i=1; i<=10; i++))
do
./benchcold mobilenet_v2 2 0 0 0 1
done
for ((i=1; i<=10; i++))
do
./benchcold resnet18 2 0 0 0 1
done
for ((i=1; i<=10; i++))
do
./benchcold resnet50 2 0 0 0 1
done
for ((i=1; i<=10; i++))
do
./benchcold shufflenet 2 0 0 0 1
done
for ((i=1; i<=10; i++))
do
./benchcold shufflenet_v2 2 0 0 0 1
done
for ((i=1; i<=10; i++))
do
./benchcold squeezenet 2 0 0 0 1
done
for ((i=1; i<=10; i++))
do
./benchcold efficientnet_b0 2 0 0 0 1
done
for ((i=1; i<=10; i++))
do
./benchcold mobilenetv2_yolov3 2 0 0 0 1
done
for ((i=1; i<=10; i++))
do
./benchcold mobilenet_yolo 2 0 0 0 1
done
mv ./output.tmp.csv ../../scripts
cd ../../scripts || exit
chmod +x ./get_output.py
python3 ./get_output.py
rm ./output.tmp.csv