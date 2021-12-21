#ifndef LAYERS_CUH
#define LAYERS_CUH
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void conv2d();
void relu6();
void depth_wise_conv();
void point_wise_conv();
void add_layer();
void avg_pool();
void linear_layer();

#endif