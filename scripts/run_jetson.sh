#!/bin/bash

cd ../build/benchmark || exit
./benchcold alexnet 2 0 0 0 1
./benchcold googlenet 2 0 0 0 1
./benchcold mobilenet 2 0 0 0 1
./benchcold mobilenet_v2 2 0 0 0 1
./benchcold resnet18 2 0 0 0 1
./benchcold resnet50 2 0 0 0 1
./benchcold shufflenet 2 0 0 0 1
./benchcold shufflenet_v2 2 0 0 0 1
./benchcold squeezenet 2 0 0 0 1
./benchcold efficientnet_b0 2 0 0 0 1