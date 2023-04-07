#!/bin/bash

#adb shell echo "userspace" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
#adb shell echo "userspace" > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
#adb shell echo "userspace" > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
#adb shell echo "userspace" > /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor
#adb shell echo "userspace" > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor
#adb shell echo "userspace" > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor
#adb shell echo "1804800" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#adb shell echo "1804800" > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#adb shell echo "2419200" > /sys/devices/system/cpu/cpu4/cpufreq/scaling_setspeed
#adb shell echo "2419200" > /sys/devices/system/cpu/cpu5/cpufreq/scaling_setspeed
#adb shell echo "2419200" > /sys/devices/system/cpu/cpu6/cpufreq/scaling_setspeed
#adb shell echo "2841600" > /sys/devices/system/cpu/cpu7/cpufreq/scaling_setspeed

#echo "userspace" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
#echo "userspace" > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
#echo "userspace" > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
#echo "userspace" > /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor
#echo "userspace" > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor
#echo "userspace" > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor
#echo "1804800" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
#echo "1804800" > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
#echo "2419200" > /sys/devices/system/cpu/cpu4/cpufreq/scaling_setspeed
#echo "2419200" > /sys/devices/system/cpu/cpu5/cpufreq/scaling_setspeed
#echo "2419200" > /sys/devices/system/cpu/cpu6/cpufreq/scaling_setspeed
#echo "2841600" > /sys/devices/system/cpu/cpu7/cpufreq/scaling_setspeed

# repair
adb shell mkdir /data/local/tmp/cold-infer-ncnn/
adb push ../benchmark/benchmark_models/* /data/local/tmp/cold-infer-ncnn/
./build-arm64-v8a.sh
adb push ../build-android-arm64/benchmark/benchcolddeploy /data/local/tmp/cold-infer-ncnn/
adb push ../build-android-arm64/benchmark/benchcold /data/local/tmp/cold-infer-ncnn/