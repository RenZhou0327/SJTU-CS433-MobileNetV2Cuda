TARGET = mobilenet_main
SRC = mobilenet_main.cc init_model.cc

NVCC = /usr/local/cuda-10.2/bin/nvcc

$(TARGET): $(SRC)
	$(NVCC) $(SRC) -o $(TARGET)

clean:
	rm -rf *.o mobilenet_main