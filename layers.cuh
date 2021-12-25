#ifndef LAYERS_CUH
#define LAYERS_CUH
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include<cublas_v2.h>

void conv2d(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int k_shape, int out_c, int stride, int pad, cublasHandle_t* handle_p);
void relu6();
void depth_wise_conv(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int k_shape, int out_c, int stride, int pad, bool is_log);
void point_wise_conv(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int out_c, bool is_relu, bool is_log, cublasHandle_t* handle_p);
void add_layer(float* A, float* B, float** C_p, int channels, int shape);
void avg_pool();
void linear_layer();
void store_back_up(float* in_tensor, float** out_tensor_p, int out_lens);

void check_layer_data(float* out_tensor, int out_lens, int idx, char* file_name);

#endif