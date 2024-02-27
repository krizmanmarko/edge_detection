#!/bin/bash

install_dir="$(pwd)/opencv_install"
mkdir -p $install_dir

build_dir="$install_dir/build"
util_dir="$install_dir/util"
download_dir="$install_dir/download"

mkdir -p $build_dir
mkdir -p $util_dir
mkdir -p $download_dir

nr_cores=2
run_srun=""

################################################################################
# INSTALL DEPENDENCIES - 20 minutes (2 cores)
################################################################################

#########
# cmake #
#########
cmake="$util_dir/cmake/bin/cmake"

if [ -f "$cmake" ]
then
	echo "cmake already installed"
else
	wget -O "$download_dir/cmake.sh" "https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-x86_64.sh"
	mkdir $util_dir/cmake
	bash "$download_dir/cmake.sh" --prefix="$util_dir/cmake" --exclude-subdir --skip-license
fi
test -f "$cmake" || exit 1

############
# openblas #
############
if [ -f "$install_dir/lib64/libopenblas.a" ]
then
	echo "openblas already installed"
else
	wget -O "$download_dir/openblas.zip" "https://github.com/xianyi/OpenBLAS/releases/download/v0.3.21/OpenBLAS-0.3.21.zip"
	unzip "$download_dir/openblas.zip" -d "$download_dir"
	mkdir "$build_dir/OpenBLAS"
	cd "$build_dir/OpenBLAS" && $cmake "$download_dir/OpenBLAS-0.3.21"
	cd "$build_dir/OpenBLAS" && $run_srun make -j $nr_cores
	cd "$build_dir/OpenBLAS" && "$cmake" --install . --prefix "$install_dir"
fi
test -f "$install_dir/lib64/libopenblas.a" || exit 1


################################################################################
# INSTALL OPENCV - 2 hours (2 cores)
################################################################################

wget -O "$download_dir/opencv.zip" https://github.com/opencv/opencv/archive/4.x.zip
unzip "$download_dir/opencv.zip" -d "$download_dir"
mv "$download_dir/opencv-4.x" "$download_dir/opencv"

mkdir "$build_dir/opencv"
export OpenBLAS_HOME=$install_dir
cd "$build_dir/opencv" && "$cmake" "$download_dir/opencv" # somehow add path to compiled openblas
cd "$build_dir/opencv" && $run_srun make -j $nr_cores
cd "$build_dir/opencv" && "$cmake" --install . --prefix "$install_dir"


################################################################################
# USAGE
################################################################################

# #include <opencv2/opencv.hpp>
# g++ main.cpp -I $install_dir/include/opencv4
