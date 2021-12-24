#ifndef LAYERS_CUH
#define LAYERS_CUH
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include<cublas_v2.h>

void conv2d(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int k_shape, int out_c, int stride, int pad);
void relu6();
void depth_wise_conv(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int k_shape, int out_c, int stride, int pad);
void point_wise_conv(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int out_c);
void add_layer();
void avg_pool();
void linear_layer();

#endif