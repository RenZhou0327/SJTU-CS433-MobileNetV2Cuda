# SJTU CS433 Parallel and Distributed Programming

## Professor
  Xiaoyao Liang
  
## Basic Info
  An optimized MobileNet-V2 implementation by Cuda C
  
## My Work
  Set up the Model and Binary Parameters, Write all Convolution(Conv2d, Depth-wise Conv and Point-wise Conv) Layer and Matrix Gemm.

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
Please put **mobilenetInput.txt**, **mobilenetOutput.txt** in /Code/ direction and put  **weight_data.bin** and **bias_data.bin** in /Code/parameters/ direction.

​	Then execute in terminal:

```
$ make
$ ./mobilenet_main
```

