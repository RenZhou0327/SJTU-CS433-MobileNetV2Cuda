#ifndef INIT_MODEL_CUH
#define INIT_MODEL_CUH
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include<cublas_v2.h>
#include <time.h>
#include <assert.h>


extern time_t st, et;
extern cublasStatus_t status;
extern cublasHandle_t handle;


extern const int w1_len, b1_len;
extern const int w2_len, b2_len;
extern const int w3_len, b3_len;
extern const int w4_len, b4_len;
extern const int w5_len, b5_len;
extern const int w6_len, b6_len;
extern const int w7_len, b7_len;
extern const int w8_len, b8_len;
extern const int w9_len, b9_len;
extern const int w10_len, b10_len;
extern const int w11_len, b11_len;
extern const int w12_len, b12_len;
extern const int w13_len, b13_len;
extern const int w14_len, b14_len;
extern const int w15_len, b15_len;
extern const int w16_len, b16_len;
extern const int w17_len, b17_len;
extern const int w18_len, b18_len;
extern const int w19_len, b19_len;
extern const int w20_len, b20_len;
extern const int w21_len, b21_len;
extern const int w22_len, b22_len;
extern const int w23_len, b23_len;
extern const int w24_len, b24_len;
extern const int w25_len, b25_len;
extern const int w26_len, b26_len;
extern const int w27_len, b27_len;
extern const int w28_len, b28_len;
extern const int w29_len, b29_len;
extern const int w30_len, b30_len;
extern const int w31_len, b31_len;
extern const int w32_len, b32_len;
extern const int w33_len, b33_len;
extern const int w34_len, b34_len;
extern const int w35_len, b35_len;
extern const int w36_len, b36_len;
extern const int w37_len, b37_len;
extern const int w38_len, b38_len;
extern const int w39_len, b39_len;
extern const int w40_len, b40_len;
extern const int w41_len, b41_len;
extern const int w42_len, b42_len;
extern const int w43_len, b43_len;
extern const int w44_len, b44_len;
extern const int w45_len, b45_len;
extern const int w46_len, b46_len;
extern const int w47_len, b47_len;
extern const int w48_len, b48_len;
extern const int w49_len, b49_len;
extern const int w50_len, b50_len;
extern const int w51_len, b51_len;
extern const int w52_len, b52_len;
extern const int w53_len, b53_len;


extern float *w1, *b1;
extern float *w2, *b2;
extern float *w3, *b3;
extern float *w4, *b4;
extern float *w5, *b5;
extern float *w6, *b6;
extern float *w7, *b7;
extern float *w8, *b8;
extern float *w9, *b9;
extern float *w10, *b10;
extern float *w11, *b11;
extern float *w12, *b12;
extern float *w13, *b13;
extern float *w14, *b14;
extern float *w15, *b15;
extern float *w16, *b16;
extern float *w17, *b17;
extern float *w18, *b18;
extern float *w19, *b19;
extern float *w20, *b20;
extern float *w21, *b21;
extern float *w22, *b22;
extern float *w23, *b23;
extern float *w24, *b24;
extern float *w25, *b25;
extern float *w26, *b26;
extern float *w27, *b27;
extern float *w28, *b28;
extern float *w29, *b29;
extern float *w30, *b30;
extern float *w31, *b31;
extern float *w32, *b32;
extern float *w33, *b33;
extern float *w34, *b34;
extern float *w35, *b35;
extern float *w36, *b36;
extern float *w37, *b37;
extern float *w38, *b38;
extern float *w39, *b39;
extern float *w40, *b40;
extern float *w41, *b41;
extern float *w42, *b42;
extern float *w43, *b43;
extern float *w44, *b44;
extern float *w45, *b45;
extern float *w46, *b46;
extern float *w47, *b47;
extern float *w48, *b48;
extern float *w49, *b49;
extern float *w50, *b50;
extern float *w51, *b51;
extern float *w52, *b52;
extern float *w53, *b53;


// 读取权重
void init_model();

// 分配内存空间
void alloc_mem();

// 读取weight和bias
void read_params();

// 移动到cuda上
void move_params();

// 移动图片到cuda上
void move_imgs(float* input, float** imgs, int len);

// 测试模型读入权重是否正确
void test_read_data();

// 测试模型输出值
void test_output_data(float* nums, int lens, int idx);

#endif