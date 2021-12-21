#include <iostream>
#include <typeinfo>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cublas.h>
#include "layers.cuh"

using namespace std;

// 注意: 现在读数据改成二进制读了, 二进制快多了

FILE *in_w, *in_b;
const int IMG_SHAPE = 3 * 244 * 244;
const int BATCH_SIZE = 16;
float images[BATCH_SIZE][IMG_SHAPE];

void read_images(char* image_path) {
    FILE *img_in = fopen(image_path, "rb");
    for (int i = 0; i < 16; ++i) {
        fread(images[i], sizeof(images[i]), 1, img_in);
    }
    fclose(img_in);
    // for (int i = 0; i < 16; ++i) {
    //     printf("%f ", images[i][IMG_SHAPE - 1]);
    // }
    // printf("\n");
}

__global__ void test_print(float* data, int lens) {
    for (int i = 0; i < lens; i += 27) {
        printf("%f ", data[i]);
    }
    printf("\n");
}

void read_params(float** w_p, float** b_p, int w_len, int b_len) {
    *w_p = (float*) calloc(w_len, sizeof(float));
    *b_p = (float*) calloc(b_len, sizeof(float));
    fread(*w_p, w_len * sizeof(float), 1, in_w);
    fread(*b_p, b_len * sizeof(float), 1, in_b);
    
    float *w_gpu = NULL, *b_gpu = NULL;
    cudaError_t e1 = cudaSuccess, e2 = cudaSuccess;
    
    e1 = cudaMalloc((void**)&w_gpu, w_len * sizeof(float));
    e2 = cudaMalloc((void**)&b_gpu, b_len * sizeof(float));
    printf("%d %d\n", (e1 == cudaSuccess), (e2 == cudaSuccess));
    
    e1 = cudaMemcpy(w_gpu, *w_p, w_len * sizeof(float), cudaMemcpyHostToDevice);
    e2 = cudaMemcpy(b_gpu, *b_p, b_len * sizeof(float), cudaMemcpyHostToDevice);
    printf("%d %d\n", (e1 == cudaSuccess), (e2 == cudaSuccess));
    
    free(*w_p);
    free(*b_p);
    *w_p = w_gpu;
    *b_p = b_gpu;
}


void infer() {
    
    // Block1: W (32, 3, 3, 3), b (32,), In (1, 3, 244, 244) Out ()
    float *w1 = NULL, *b1 = NULL;
    int w1_len = 32 * 3 * 3 * 3, b1_len = 32;
    // 传入的是指针的地址
    read_params(&w1, &b1, w1_len, b1_len);
    // test_print<<<1, 1>>>(w1, w1_len);
    // cudaDeviceSynchronize();
    // return;
    // cout<<typeid(w1).name()<<endl;
    // for (int i = 0; i < 32; ++i) {
    //     printf("%f ", w1[i * 27 + 26]);
    //     printf("%f\n", b1[i]);
    // }
    // cudaError_t e1 = cudaSuccess, e2 = cudaSuccess;
    // e1 = cudaMemcpy(w1_gpu, w1, w1_len * sizeof(float), cudaMemcpyHostToDevice);
    // e2 = cudaMemcpy(b1_gpu, b1, b1_len * sizeof(float), cudaMemcpyHostToDevice);
    // cout<<e1<<" "<<e2<<endl;
    // return;
    conv2d();
    relu6();

    // Block2:
    depth_wise_conv();
    relu6();
    point_wise_conv();

    // Block3:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    // Block4:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    add_layer();

    // Block5:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    // Block6:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    add_layer();

    // Block7:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    add_layer();

    // Block8:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    // Block9:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    add_layer();

    // Block10:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    add_layer();

    // Block11:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    add_layer();

    // Block12:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    // Block13:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    add_layer();

    // Block14:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    add_layer();

    // Block15:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    // Block16:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    add_layer();

    // Block17:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    add_layer();

    // Block18:
    point_wise_conv();
    relu6();
    depth_wise_conv();
    relu6();
    point_wise_conv();

    // Block19:
    point_wise_conv();
    relu6();

    // Block20:
    avg_pool();
    linear_layer();
    
}

int main(int argc, char* argv[]) {
    char image_path[] = "./images/images_data.bin";
    char weight_path[] = "./parameters/weight_data.bin";
    char bias_path[] = "./parameters/bias_data.bin";
    char infer_res_path[] = "./results/";
    
    in_w = fopen(weight_path, "rb");
    in_b = fopen(bias_path, "rb");
    read_images(image_path);

    infer();
    
    return 0;
}