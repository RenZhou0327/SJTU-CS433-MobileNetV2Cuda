SRC = mobilenetv2.cu
TARGET = mobilenetv2

CC = /usr/local/cuda-10.2/bin/nvcc

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET)

clean:
	rm -rf *.o mobilenetv2