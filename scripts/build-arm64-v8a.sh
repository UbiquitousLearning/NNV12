#!/bin/bash

mkdir -p ../build-android-arm64
pushd ../build-android-arm64
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"   -DANDROID_ABI="arm64-v8a"   -DANDROID_PLATFORM=android-24 -DNCNN_VULKAN=ON ..
make -j8
#make install
popd