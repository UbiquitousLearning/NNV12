#!/bin/bash

# update cmake
sudo apt remove cmake
sudo rm -rf /usr/local/share/cmake*
mkdir software-install && cd software-install
wget https://github.com/Kitware/CMake/releases/download/v3.26.3/cmake-3.26.3.zip
unzip ./cmake-3.26.3.zip
cd ./cmake-3.26.3
chmod +x ./configure
./configure
make & sudo make install
cd ..

# install vukan
sudo apt-get update && sudo apt-get install git build-essential libx11-xcb-dev libxkbcommon-dev libwayland-dev libxrandr-dev cmake
git clone https://github.com/KhronosGroup/Vulkan-Loader.git
cd Vulkan-Loader && mkdir build && cd build
../scripts/update_deps.py
cmake -DCMAKE_BUILD_TYPE=Release -DVULKAN_HEADERS_INSTALL_DIR=$(pwd)/Vulkan-Headers/build/install ..
make -j8
sudo apt install