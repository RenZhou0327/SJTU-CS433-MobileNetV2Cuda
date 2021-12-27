#ifndef LAYERS_CUH
#define LAYERS_CUH
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
#include<cublas_v2.h>

// 普通卷积
void Conv2d(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int k_shape, int out_c, int stride, int pad, cublasHandle_t* handle_p);
// depthwise卷积
void DepthwiseConv(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int k_shape, int out_c, int stride, int pad, bool is_log);
// pointwise卷积
void PointwiseConv(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int out_c, bool is_relu, bool is_log, cublasHandle_t* handle_p);
// 跃层连接
void AddLayer(float* A, float* B, float** C_p, int channels, int shape);
// global average pool
void GlobalAvgPool(float* in_tensor, float** out_tensor_p, int channels, int in_shape);
// 线性层
void LinearLayer(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_len, int out_len, cublasHandle_t* handle_p);
// 存储跃层连接中间结果
void StoreBackup(float* in_tensor, float** out_tensor_p, int out_lens);

void CheckLayerData(float* out_tensor, int out_lens, int idx, char* file_name);

#endif