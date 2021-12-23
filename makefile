TARGET = mobilenet_main
SRC = mobilenet_main.cc init_model.cu layers.cu

NVCC = /usr/local/cuda-10.2/bin/nvcc

$(TARGET): $(SRC)
	$(NVCC) -lcublas $(SRC) -o $(TARGET)

clean:
	rm -rf *.o mobilenet_main