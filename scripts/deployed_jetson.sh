#!/bin/bash

cd ../build/benchmark || exit
./benchcolddeploy alexnet 2 0 0 0 1
./benchcolddeploy googlenet 2 0 0 0 1
./benchcolddeploy mobilenet 2 0 0 0 1
./benchcolddeploy mobilenet_v2 2 0 0 0 1
./benchcolddeploy resnet18 2 0 0 0 1
./benchcolddeploy resnet50 2 0 0 0 1
./benchcolddeploy shufflenet 2 0 0 0 1
./benchcolddeploy shufflenet_v2 2 0 0 0 1
./benchcolddeploy squeezenet 2 0 0 0 1
./benchcolddeploy efficientnet_b0 2 0 0 0 1