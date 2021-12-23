TARGET = mobilenet_main
SRC = mobilenet_main.cc init_model.cu layers.cu

NVCC = /usr/local/cuda-10.2/bin/nvcc

$(TARGET): $(SRC)
	$(NVCC) $(SRC) -o $(TARGET) -std=c++11 -lcublas 

clean:
	rm -rf *.o mobilenet_main