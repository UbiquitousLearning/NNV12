#!/bin/bash

mkdir -p ../build
pushd ../build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/jetson.toolchain.cmake -DNCNN_VULKAN=ON ..
#cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/jetson.toolchain.cmake -DNCNN_VULKAN=ON -DCMAKE_BUILD_TYPE=Release ..
make -j8
#make install
popd

cp ../benchmark/benchmark_models/* ../build/benchmark