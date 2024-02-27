CC = g++
CFLAGS = -Wall -pedantic -g
INCLUDE = /ceph/grid/home/mk4462/opencv_install/include/opencv4
LIB = /ceph/grid/home/mk4462/opencv_install/lib64
#USED_LIBS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
USED_LIBS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

%: %.cpp
	$(CC) -o $@ -I $(INCLUDE) -L $(LIB) $(USED_LIBS) $^
