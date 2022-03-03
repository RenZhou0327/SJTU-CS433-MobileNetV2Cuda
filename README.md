# SJTU CS433 Parallel and Distributed Programming

## Professor
  Xiaoyao Liang
  
## Basic Info
  An optimized **MobileNet-V2** Model implemented by Cuda C
  
## My Work
  Set up MobileNet-V2 Model Structure and Test Binary Parameters in Each Layer, Write all Convolution(Conv2d, Depth-wise Conv and Point-wise Conv) Layers and Matrix Gemm Methods.

## Structure
```
\Codes
	\models
		-- mobilenet_v2.onnx	onnx model files
	\parameters
		-- bias_data.bin	model bias	(binary)
		-- bias_data.txt	model bias	(text)
		-- weight_data.bin	model weights	(binary)
		-- weight_data.txt	model weights	(text)
	\tmpfiles	Store Layer Outputs, Just for Test
	\preprocess
		-- CheckData.py	check layer output
		-- CodeGenerateFunctions.py	generate codes
		-- Img2Col.py	image to column methods
		-- InferModel.py infer standard layer data for check
		-- ReadONNX.py	read onnx model and store weights and bias
	-- mobilenet_main.cc	program entrance
	-- layers.cuh	model layers headfile
	-- layers.cu	model layers implements
	-- init_model.cuh	model init headfile
	-- init_model.cu	model init implements
	-- mobilenetInput.txt	Input Image Data
	-- mobilenetOutput.txt	Standard Outputs
```

## Running
Please put **mobilenetInput.txt**, **mobilenetOutput.txt** in /Code/ folder and put  **weight_data.bin** and **bias_data.bin** in /Code/parameters/ folder.
	
- **mobilenetInput.txt** contains a batch of images data, each line represents an images (c \* h \* w).
	
- **mobilenetOutput.txt** contains inference result from model/mobilenet_v2.onnx model, which can be seen as ground truth. The dimension of each line is 1000.
	
- **weight_data.bin** is W of **W**X + b for each layer; **bias_data.bin** is b of WX + **b** for each layer. These two binary files can be gotten from preprocess/ReadONNX.py, from where you can get two text files and then convert them to binary files.
 
​	Then execute in terminal:

```
$ make
$ ./mobilenet_main
```

