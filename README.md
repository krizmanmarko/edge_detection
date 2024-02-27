# Pipelined Canny Edge Detector in C++ and OpenCV using Pthreads and MPI

## OpenCV Installation

Install OpenCV using `install-opencv.sh` script.

## Build

Put these entries into your `~/.bashrc` file. After `~/.bashrc` being run once after the change, comment the 2nd line out so that you won't pollute your `LD_LIBRARY_PATH` with the same path over and over again.

```bash
export OPENCV_DIR=/ceph/grid/home/<username>/opencv_install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENCV_DIR/lib64
```

Then compile with `g++`:
```bash
g++ -I $OPENCV_DIR/include/opencv4 -L $OPENCV_DIR/lib64 -lopencv_core -lopencv_imgcodecs -lopencv_imgproc ... main.cpp
```

`-l<something>` are the libraries that you will be using. You can find them in `$OPENCV_DIR/lib64` directory named something along the lines of `lib<something>.so*`.
