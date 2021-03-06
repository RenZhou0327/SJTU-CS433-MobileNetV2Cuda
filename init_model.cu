#include "init_model.cuh"

const int float_size = sizeof(float);
time_t st, et;
cublasStatus_t status;
cublasHandle_t handle;

FILE *w_in = NULL, *b_in = NULL;

const int w1_len = 32 * 3 * 3 * 3, b1_len = 32;
const int w2_len = 32 * 1 * 3 * 3, b2_len = 32;
const int w3_len = 16 * 32 * 1 * 1, b3_len = 16;
const int w4_len = 96 * 16 * 1 * 1, b4_len = 96;
const int w5_len = 96 * 1 * 3 * 3, b5_len = 96;
const int w6_len = 24 * 96 * 1 * 1, b6_len = 24;
const int w7_len = 144 * 24 * 1 * 1, b7_len = 144;
const int w8_len = 144 * 1 * 3 * 3, b8_len = 144;
const int w9_len = 24 * 144 * 1 * 1, b9_len = 24;
const int w10_len = 144 * 24 * 1 * 1, b10_len = 144;
const int w11_len = 144 * 1 * 3 * 3, b11_len = 144;
const int w12_len = 32 * 144 * 1 * 1, b12_len = 32;
const int w13_len = 192 * 32 * 1 * 1, b13_len = 192;
const int w14_len = 192 * 1 * 3 * 3, b14_len = 192;
const int w15_len = 32 * 192 * 1 * 1, b15_len = 32;
const int w16_len = 192 * 32 * 1 * 1, b16_len = 192;
const int w17_len = 192 * 1 * 3 * 3, b17_len = 192;
const int w18_len = 32 * 192 * 1 * 1, b18_len = 32;
const int w19_len = 192 * 32 * 1 * 1, b19_len = 192;
const int w20_len = 192 * 1 * 3 * 3, b20_len = 192;
const int w21_len = 64 * 192 * 1 * 1, b21_len = 64;
const int w22_len = 384 * 64 * 1 * 1, b22_len = 384;
const int w23_len = 384 * 1 * 3 * 3, b23_len = 384;
const int w24_len = 64 * 384 * 1 * 1, b24_len = 64;
const int w25_len = 384 * 64 * 1 * 1, b25_len = 384;
const int w26_len = 384 * 1 * 3 * 3, b26_len = 384;
const int w27_len = 64 * 384 * 1 * 1, b27_len = 64;
const int w28_len = 384 * 64 * 1 * 1, b28_len = 384;
const int w29_len = 384 * 1 * 3 * 3, b29_len = 384;
const int w30_len = 64 * 384 * 1 * 1, b30_len = 64;
const int w31_len = 384 * 64 * 1 * 1, b31_len = 384;
const int w32_len = 384 * 1 * 3 * 3, b32_len = 384;
const int w33_len = 96 * 384 * 1 * 1, b33_len = 96;
const int w34_len = 576 * 96 * 1 * 1, b34_len = 576;
const int w35_len = 576 * 1 * 3 * 3, b35_len = 576;
const int w36_len = 96 * 576 * 1 * 1, b36_len = 96;
const int w37_len = 576 * 96 * 1 * 1, b37_len = 576;
const int w38_len = 576 * 1 * 3 * 3, b38_len = 576;
const int w39_len = 96 * 576 * 1 * 1, b39_len = 96;
const int w40_len = 576 * 96 * 1 * 1, b40_len = 576;
const int w41_len = 576 * 1 * 3 * 3, b41_len = 576;
const int w42_len = 160 * 576 * 1 * 1, b42_len = 160;
const int w43_len = 960 * 160 * 1 * 1, b43_len = 960;
const int w44_len = 960 * 1 * 3 * 3, b44_len = 960;
const int w45_len = 160 * 960 * 1 * 1, b45_len = 160;
const int w46_len = 960 * 160 * 1 * 1, b46_len = 960;
const int w47_len = 960 * 1 * 3 * 3, b47_len = 960;
const int w48_len = 160 * 960 * 1 * 1, b48_len = 160;
const int w49_len = 960 * 160 * 1 * 1, b49_len = 960;
const int w50_len = 960 * 1 * 3 * 3, b50_len = 960;
const int w51_len = 320 * 960 * 1 * 1, b51_len = 320;
const int w52_len = 1280 * 320 * 1 * 1, b52_len = 1280;
const int w53_len = 1000 * 1280 * 1 * 1, b53_len = 1000;

float *w1, *b1;
float *w2, *b2;
float *w3, *b3;
float *w4, *b4;
float *w5, *b5;
float *w6, *b6;
float *w7, *b7;
float *w8, *b8;
float *w9, *b9;
float *w10, *b10;
float *w11, *b11;
float *w12, *b12;
float *w13, *b13;
float *w14, *b14;
float *w15, *b15;
float *w16, *b16;
float *w17, *b17;
float *w18, *b18;
float *w19, *b19;
float *w20, *b20;
float *w21, *b21;
float *w22, *b22;
float *w23, *b23;
float *w24, *b24;
float *w25, *b25;
float *w26, *b26;
float *w27, *b27;
float *w28, *b28;
float *w29, *b29;
float *w30, *b30;
float *w31, *b31;
float *w32, *b32;
float *w33, *b33;
float *w34, *b34;
float *w35, *b35;
float *w36, *b36;
float *w37, *b37;
float *w38, *b38;
float *w39, *b39;
float *w40, *b40;
float *w41, *b41;
float *w42, *b42;
float *w43, *b43;
float *w44, *b44;
float *w45, *b45;
float *w46, *b46;
float *w47, *b47;
float *w48, *b48;
float *w49, *b49;
float *w50, *b50;
float *w51, *b51;
float *w52, *b52;
float *w53, *b53;


void AllocateMemory()
{
    w1 = (float*) malloc(w1_len * float_size); b1 = (float*) malloc(b1_len * float_size);
    w2 = (float*) malloc(w2_len * float_size); b2 = (float*) malloc(b2_len * float_size);
    w3 = (float*) malloc(w3_len * float_size); b3 = (float*) malloc(b3_len * float_size);
    w4 = (float*) malloc(w4_len * float_size); b4 = (float*) malloc(b4_len * float_size);
    w5 = (float*) malloc(w5_len * float_size); b5 = (float*) malloc(b5_len * float_size);
    w6 = (float*) malloc(w6_len * float_size); b6 = (float*) malloc(b6_len * float_size);
    w7 = (float*) malloc(w7_len * float_size); b7 = (float*) malloc(b7_len * float_size);
    w8 = (float*) malloc(w8_len * float_size); b8 = (float*) malloc(b8_len * float_size);
    w9 = (float*) malloc(w9_len * float_size); b9 = (float*) malloc(b9_len * float_size);
    w10 = (float*) malloc(w10_len * float_size); b10 = (float*) malloc(b10_len * float_size);
    w11 = (float*) malloc(w11_len * float_size); b11 = (float*) malloc(b11_len * float_size);
    w12 = (float*) malloc(w12_len * float_size); b12 = (float*) malloc(b12_len * float_size);
    w13 = (float*) malloc(w13_len * float_size); b13 = (float*) malloc(b13_len * float_size);
    w14 = (float*) malloc(w14_len * float_size); b14 = (float*) malloc(b14_len * float_size);
    w15 = (float*) malloc(w15_len * float_size); b15 = (float*) malloc(b15_len * float_size);
    w16 = (float*) malloc(w16_len * float_size); b16 = (float*) malloc(b16_len * float_size);
    w17 = (float*) malloc(w17_len * float_size); b17 = (float*) malloc(b17_len * float_size);
    w18 = (float*) malloc(w18_len * float_size); b18 = (float*) malloc(b18_len * float_size);
    w19 = (float*) malloc(w19_len * float_size); b19 = (float*) malloc(b19_len * float_size);
    w20 = (float*) malloc(w20_len * float_size); b20 = (float*) malloc(b20_len * float_size);
    w21 = (float*) malloc(w21_len * float_size); b21 = (float*) malloc(b21_len * float_size);
    w22 = (float*) malloc(w22_len * float_size); b22 = (float*) malloc(b22_len * float_size);
    w23 = (float*) malloc(w23_len * float_size); b23 = (float*) malloc(b23_len * float_size);
    w24 = (float*) malloc(w24_len * float_size); b24 = (float*) malloc(b24_len * float_size);
    w25 = (float*) malloc(w25_len * float_size); b25 = (float*) malloc(b25_len * float_size);
    w26 = (float*) malloc(w26_len * float_size); b26 = (float*) malloc(b26_len * float_size);
    w27 = (float*) malloc(w27_len * float_size); b27 = (float*) malloc(b27_len * float_size);
    w28 = (float*) malloc(w28_len * float_size); b28 = (float*) malloc(b28_len * float_size);
    w29 = (float*) malloc(w29_len * float_size); b29 = (float*) malloc(b29_len * float_size);
    w30 = (float*) malloc(w30_len * float_size); b30 = (float*) malloc(b30_len * float_size);
    w31 = (float*) malloc(w31_len * float_size); b31 = (float*) malloc(b31_len * float_size);
    w32 = (float*) malloc(w32_len * float_size); b32 = (float*) malloc(b32_len * float_size);
    w33 = (float*) malloc(w33_len * float_size); b33 = (float*) malloc(b33_len * float_size);
    w34 = (float*) malloc(w34_len * float_size); b34 = (float*) malloc(b34_len * float_size);
    w35 = (float*) malloc(w35_len * float_size); b35 = (float*) malloc(b35_len * float_size);
    w36 = (float*) malloc(w36_len * float_size); b36 = (float*) malloc(b36_len * float_size);
    w37 = (float*) malloc(w37_len * float_size); b37 = (float*) malloc(b37_len * float_size);
    w38 = (float*) malloc(w38_len * float_size); b38 = (float*) malloc(b38_len * float_size);
    w39 = (float*) malloc(w39_len * float_size); b39 = (float*) malloc(b39_len * float_size);
    w40 = (float*) malloc(w40_len * float_size); b40 = (float*) malloc(b40_len * float_size);
    w41 = (float*) malloc(w41_len * float_size); b41 = (float*) malloc(b41_len * float_size);
    w42 = (float*) malloc(w42_len * float_size); b42 = (float*) malloc(b42_len * float_size);
    w43 = (float*) malloc(w43_len * float_size); b43 = (float*) malloc(b43_len * float_size);
    w44 = (float*) malloc(w44_len * float_size); b44 = (float*) malloc(b44_len * float_size);
    w45 = (float*) malloc(w45_len * float_size); b45 = (float*) malloc(b45_len * float_size);
    w46 = (float*) malloc(w46_len * float_size); b46 = (float*) malloc(b46_len * float_size);
    w47 = (float*) malloc(w47_len * float_size); b47 = (float*) malloc(b47_len * float_size);
    w48 = (float*) malloc(w48_len * float_size); b48 = (float*) malloc(b48_len * float_size);
    w49 = (float*) malloc(w49_len * float_size); b49 = (float*) malloc(b49_len * float_size);
    w50 = (float*) malloc(w50_len * float_size); b50 = (float*) malloc(b50_len * float_size);
    w51 = (float*) malloc(w51_len * float_size); b51 = (float*) malloc(b51_len * float_size);
    w52 = (float*) malloc(w52_len * float_size); b52 = (float*) malloc(b52_len * float_size);
    w53 = (float*) malloc(w53_len * float_size); b53 = (float*) malloc(b53_len * float_size);
}

void ReadParams()
{
    fread(w1, w1_len * float_size, 1, w_in); fread(b1, b1_len * float_size, 1, b_in);
    fread(w2, w2_len * float_size, 1, w_in); fread(b2, b2_len * float_size, 1, b_in);
    fread(w3, w3_len * float_size, 1, w_in); fread(b3, b3_len * float_size, 1, b_in);
    fread(w4, w4_len * float_size, 1, w_in); fread(b4, b4_len * float_size, 1, b_in);
    fread(w5, w5_len * float_size, 1, w_in); fread(b5, b5_len * float_size, 1, b_in);
    fread(w6, w6_len * float_size, 1, w_in); fread(b6, b6_len * float_size, 1, b_in);
    fread(w7, w7_len * float_size, 1, w_in); fread(b7, b7_len * float_size, 1, b_in);
    fread(w8, w8_len * float_size, 1, w_in); fread(b8, b8_len * float_size, 1, b_in);
    fread(w9, w9_len * float_size, 1, w_in); fread(b9, b9_len * float_size, 1, b_in);
    fread(w10, w10_len * float_size, 1, w_in); fread(b10, b10_len * float_size, 1, b_in);
    fread(w11, w11_len * float_size, 1, w_in); fread(b11, b11_len * float_size, 1, b_in);
    fread(w12, w12_len * float_size, 1, w_in); fread(b12, b12_len * float_size, 1, b_in);
    fread(w13, w13_len * float_size, 1, w_in); fread(b13, b13_len * float_size, 1, b_in);
    fread(w14, w14_len * float_size, 1, w_in); fread(b14, b14_len * float_size, 1, b_in);
    fread(w15, w15_len * float_size, 1, w_in); fread(b15, b15_len * float_size, 1, b_in);
    fread(w16, w16_len * float_size, 1, w_in); fread(b16, b16_len * float_size, 1, b_in);
    fread(w17, w17_len * float_size, 1, w_in); fread(b17, b17_len * float_size, 1, b_in);
    fread(w18, w18_len * float_size, 1, w_in); fread(b18, b18_len * float_size, 1, b_in);
    fread(w19, w19_len * float_size, 1, w_in); fread(b19, b19_len * float_size, 1, b_in);
    fread(w20, w20_len * float_size, 1, w_in); fread(b20, b20_len * float_size, 1, b_in);
    fread(w21, w21_len * float_size, 1, w_in); fread(b21, b21_len * float_size, 1, b_in);
    fread(w22, w22_len * float_size, 1, w_in); fread(b22, b22_len * float_size, 1, b_in);
    fread(w23, w23_len * float_size, 1, w_in); fread(b23, b23_len * float_size, 1, b_in);
    fread(w24, w24_len * float_size, 1, w_in); fread(b24, b24_len * float_size, 1, b_in);
    fread(w25, w25_len * float_size, 1, w_in); fread(b25, b25_len * float_size, 1, b_in);
    fread(w26, w26_len * float_size, 1, w_in); fread(b26, b26_len * float_size, 1, b_in);
    fread(w27, w27_len * float_size, 1, w_in); fread(b27, b27_len * float_size, 1, b_in);
    fread(w28, w28_len * float_size, 1, w_in); fread(b28, b28_len * float_size, 1, b_in);
    fread(w29, w29_len * float_size, 1, w_in); fread(b29, b29_len * float_size, 1, b_in);
    fread(w30, w30_len * float_size, 1, w_in); fread(b30, b30_len * float_size, 1, b_in);
    fread(w31, w31_len * float_size, 1, w_in); fread(b31, b31_len * float_size, 1, b_in);
    fread(w32, w32_len * float_size, 1, w_in); fread(b32, b32_len * float_size, 1, b_in);
    fread(w33, w33_len * float_size, 1, w_in); fread(b33, b33_len * float_size, 1, b_in);
    fread(w34, w34_len * float_size, 1, w_in); fread(b34, b34_len * float_size, 1, b_in);
    fread(w35, w35_len * float_size, 1, w_in); fread(b35, b35_len * float_size, 1, b_in);
    fread(w36, w36_len * float_size, 1, w_in); fread(b36, b36_len * float_size, 1, b_in);
    fread(w37, w37_len * float_size, 1, w_in); fread(b37, b37_len * float_size, 1, b_in);
    fread(w38, w38_len * float_size, 1, w_in); fread(b38, b38_len * float_size, 1, b_in);
    fread(w39, w39_len * float_size, 1, w_in); fread(b39, b39_len * float_size, 1, b_in);
    fread(w40, w40_len * float_size, 1, w_in); fread(b40, b40_len * float_size, 1, b_in);
    fread(w41, w41_len * float_size, 1, w_in); fread(b41, b41_len * float_size, 1, b_in);
    fread(w42, w42_len * float_size, 1, w_in); fread(b42, b42_len * float_size, 1, b_in);
    fread(w43, w43_len * float_size, 1, w_in); fread(b43, b43_len * float_size, 1, b_in);
    fread(w44, w44_len * float_size, 1, w_in); fread(b44, b44_len * float_size, 1, b_in);
    fread(w45, w45_len * float_size, 1, w_in); fread(b45, b45_len * float_size, 1, b_in);
    fread(w46, w46_len * float_size, 1, w_in); fread(b46, b46_len * float_size, 1, b_in);
    fread(w47, w47_len * float_size, 1, w_in); fread(b47, b47_len * float_size, 1, b_in);
    fread(w48, w48_len * float_size, 1, w_in); fread(b48, b48_len * float_size, 1, b_in);
    fread(w49, w49_len * float_size, 1, w_in); fread(b49, b49_len * float_size, 1, b_in);
    fread(w50, w50_len * float_size, 1, w_in); fread(b50, b50_len * float_size, 1, b_in);
    fread(w51, w51_len * float_size, 1, w_in); fread(b51, b51_len * float_size, 1, b_in);
    fread(w52, w52_len * float_size, 1, w_in); fread(b52, b52_len * float_size, 1, b_in);
    fread(w53, w53_len * float_size, 1, w_in); fread(b53, b53_len * float_size, 1, b_in);
}

void MoveItem(float** w_p, int w_len, float** b_p, int b_len)
{
    float *w_gpu = NULL, *b_gpu = NULL;

    cudaError_t e1 = cudaSuccess, e2 = cudaSuccess;
    e1 = cudaMalloc((void**)&w_gpu, w_len * float_size);
    e2 = cudaMalloc((void**)&b_gpu, b_len * float_size);
    assert(e1 == cudaSuccess && e2 == cudaSuccess);
    
    e1 = cudaMemcpy(w_gpu, *w_p, w_len * float_size, cudaMemcpyHostToDevice);
    e2 = cudaMemcpy(b_gpu, *b_p, b_len * float_size, cudaMemcpyHostToDevice);
    assert(e1 == cudaSuccess && e2 == cudaSuccess);
    
    free(*w_p);
    free(*b_p);
    *w_p = w_gpu;
    *b_p = b_gpu;
}

void MoveImgs(float* input, float** imgs_p, int len)
{
    float *imgs_gpu = NULL;

    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&imgs_gpu, len * float_size);
    assert(err == cudaSuccess);
    
    err = cudaMemcpy(imgs_gpu, input, len * float_size, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
    
    *imgs_p = imgs_gpu;
}

void MoveParams()
{
    MoveItem(&w1, w1_len, &b1, b1_len);
    MoveItem(&w2, w2_len, &b2, b2_len);
    MoveItem(&w3, w3_len, &b3, b3_len);
    MoveItem(&w4, w4_len, &b4, b4_len);
    MoveItem(&w5, w5_len, &b5, b5_len);
    MoveItem(&w6, w6_len, &b6, b6_len);
    MoveItem(&w7, w7_len, &b7, b7_len);
    MoveItem(&w8, w8_len, &b8, b8_len);
    MoveItem(&w9, w9_len, &b9, b9_len);
    MoveItem(&w10, w10_len, &b10, b10_len);
    MoveItem(&w11, w11_len, &b11, b11_len);
    MoveItem(&w12, w12_len, &b12, b12_len);
    MoveItem(&w13, w13_len, &b13, b13_len);
    MoveItem(&w14, w14_len, &b14, b14_len);
    MoveItem(&w15, w15_len, &b15, b15_len);
    MoveItem(&w16, w16_len, &b16, b16_len);
    MoveItem(&w17, w17_len, &b17, b17_len);
    MoveItem(&w18, w18_len, &b18, b18_len);
    MoveItem(&w19, w19_len, &b19, b19_len);
    MoveItem(&w20, w20_len, &b20, b20_len);
    MoveItem(&w21, w21_len, &b21, b21_len);
    MoveItem(&w22, w22_len, &b22, b22_len);
    MoveItem(&w23, w23_len, &b23, b23_len);
    MoveItem(&w24, w24_len, &b24, b24_len);
    MoveItem(&w25, w25_len, &b25, b25_len);
    MoveItem(&w26, w26_len, &b26, b26_len);
    MoveItem(&w27, w27_len, &b27, b27_len);
    MoveItem(&w28, w28_len, &b28, b28_len);
    MoveItem(&w29, w29_len, &b29, b29_len);
    MoveItem(&w30, w30_len, &b30, b30_len);
    MoveItem(&w31, w31_len, &b31, b31_len);
    MoveItem(&w32, w32_len, &b32, b32_len);
    MoveItem(&w33, w33_len, &b33, b33_len);
    MoveItem(&w34, w34_len, &b34, b34_len);
    MoveItem(&w35, w35_len, &b35, b35_len);
    MoveItem(&w36, w36_len, &b36, b36_len);
    MoveItem(&w37, w37_len, &b37, b37_len);
    MoveItem(&w38, w38_len, &b38, b38_len);
    MoveItem(&w39, w39_len, &b39, b39_len);
    MoveItem(&w40, w40_len, &b40, b40_len);
    MoveItem(&w41, w41_len, &b41, b41_len);
    MoveItem(&w42, w42_len, &b42, b42_len);
    MoveItem(&w43, w43_len, &b43, b43_len);
    MoveItem(&w44, w44_len, &b44, b44_len);
    MoveItem(&w45, w45_len, &b45, b45_len);
    MoveItem(&w46, w46_len, &b46, b46_len);
    MoveItem(&w47, w47_len, &b47, b47_len);
    MoveItem(&w48, w48_len, &b48, b48_len);
    MoveItem(&w49, w49_len, &b49, b49_len);
    MoveItem(&w50, w50_len, &b50, b50_len);
    MoveItem(&w51, w51_len, &b51, b51_len);
    MoveItem(&w52, w52_len, &b52, b52_len);
    MoveItem(&w53, w53_len, &b53, b53_len);
}

void InitModel() 
{
    char weight_path[] = "./parameters/weight_data.bin";
    char bias_path[] = "./parameters/bias_data.bin";
    w_in = fopen(weight_path, "rb");
    b_in = fopen(bias_path, "rb");

    AllocateMemory();
    ReadParams();

    // TestReadData();
    
    MoveParams();
    status = cublasCreate(&handle);
    assert(status == CUBLAS_STATUS_SUCCESS);
}

void TestReadData()
{
    printf("%d\n", float_size);
    printf("%f %f\n", w1[w1_len - 1], b1[b1_len - 1]);
    printf("%f %f\n", w2[w2_len - 1], b2[b2_len - 1]);
    printf("%f %f\n", w3[w3_len - 1], b3[b3_len - 1]);
    printf("%f %f\n", w4[w4_len - 1], b4[b4_len - 1]);
    printf("%f %f\n", w5[w5_len - 1], b5[b5_len - 1]);
    printf("%f %f\n", w6[w6_len - 1], b6[b6_len - 1]);
    printf("%f %f\n", w7[w7_len - 1], b7[b7_len - 1]);
    printf("%f %f\n", w8[w8_len - 1], b8[b8_len - 1]);
    printf("%f %f\n", w9[w9_len - 1], b9[b9_len - 1]);
    printf("%f %f\n", w10[w10_len - 1], b10[b10_len - 1]);
    printf("%f %f\n", w11[w11_len - 1], b11[b11_len - 1]);
    printf("%f %f\n", w12[w12_len - 1], b12[b12_len - 1]);
    printf("%f %f\n", w13[w13_len - 1], b13[b13_len - 1]);
    printf("%f %f\n", w14[w14_len - 1], b14[b14_len - 1]);
    printf("%f %f\n", w15[w15_len - 1], b15[b15_len - 1]);
    printf("%f %f\n", w16[w16_len - 1], b16[b16_len - 1]);
    printf("%f %f\n", w17[w17_len - 1], b17[b17_len - 1]);
    printf("%f %f\n", w18[w18_len - 1], b18[b18_len - 1]);
    printf("%f %f\n", w19[w19_len - 1], b19[b19_len - 1]);
    printf("%f %f\n", w20[w20_len - 1], b20[b20_len - 1]);
    printf("%f %f\n", w21[w21_len - 1], b21[b21_len - 1]);
    printf("%f %f\n", w22[w22_len - 1], b22[b22_len - 1]);
    printf("%f %f\n", w23[w23_len - 1], b23[b23_len - 1]);
    printf("%f %f\n", w24[w24_len - 1], b24[b24_len - 1]);
    printf("%f %f\n", w25[w25_len - 1], b25[b25_len - 1]);
    printf("%f %f\n", w26[w26_len - 1], b26[b26_len - 1]);
    printf("%f %f\n", w27[w27_len - 1], b27[b27_len - 1]);
    printf("%f %f\n", w28[w28_len - 1], b28[b28_len - 1]);
    printf("%f %f\n", w29[w29_len - 1], b29[b29_len - 1]);
    printf("%f %f\n", w30[w30_len - 1], b30[b30_len - 1]);
    printf("%f %f\n", w31[w31_len - 1], b31[b31_len - 1]);
    printf("%f %f\n", w32[w32_len - 1], b32[b32_len - 1]);
    printf("%f %f\n", w33[w33_len - 1], b33[b33_len - 1]);
    printf("%f %f\n", w34[w34_len - 1], b34[b34_len - 1]);
    printf("%f %f\n", w35[w35_len - 1], b35[b35_len - 1]);
    printf("%f %f\n", w36[w36_len - 1], b36[b36_len - 1]);
    printf("%f %f\n", w37[w37_len - 1], b37[b37_len - 1]);
    printf("%f %f\n", w38[w38_len - 1], b38[b38_len - 1]);
    printf("%f %f\n", w39[w39_len - 1], b39[b39_len - 1]);
    printf("%f %f\n", w40[w40_len - 1], b40[b40_len - 1]);
    printf("%f %f\n", w41[w41_len - 1], b41[b41_len - 1]);
    printf("%f %f\n", w42[w42_len - 1], b42[b42_len - 1]);
    printf("%f %f\n", w43[w43_len - 1], b43[b43_len - 1]);
    printf("%f %f\n", w44[w44_len - 1], b44[b44_len - 1]);
    printf("%f %f\n", w45[w45_len - 1], b45[b45_len - 1]);
    printf("%f %f\n", w46[w46_len - 1], b46[b46_len - 1]);
    printf("%f %f\n", w47[w47_len - 1], b47[b47_len - 1]);
    printf("%f %f\n", w48[w48_len - 1], b48[b48_len - 1]);
    printf("%f %f\n", w49[w49_len - 1], b49[b49_len - 1]);
    printf("%f %f\n", w50[w50_len - 1], b50[b50_len - 1]);
    printf("%f %f\n", w51[w51_len - 1], b51[b51_len - 1]);
    printf("%f %f\n", w52[w52_len - 1], b52[b52_len - 1]);
    printf("%f %f\n", w53[w53_len - 1], b53[b53_len - 1]);
}

void FreeMemory() 
{
    cudaError_t err;
    cublasStatus_t status = cublasDestroy(handle); assert(status == CUBLAS_STATUS_SUCCESS);
    err = cudaFree(w1); assert(err == cudaSuccess); err = cudaFree(b1); assert(err == cudaSuccess);
    err = cudaFree(w2); assert(err == cudaSuccess); err = cudaFree(b2); assert(err == cudaSuccess);
    err = cudaFree(w3); assert(err == cudaSuccess); err = cudaFree(b3); assert(err == cudaSuccess);
    err = cudaFree(w4); assert(err == cudaSuccess); err = cudaFree(b4); assert(err == cudaSuccess);
    err = cudaFree(w5); assert(err == cudaSuccess); err = cudaFree(b5); assert(err == cudaSuccess);
    err = cudaFree(w6); assert(err == cudaSuccess); err = cudaFree(b6); assert(err == cudaSuccess);
    err = cudaFree(w7); assert(err == cudaSuccess); err = cudaFree(b7); assert(err == cudaSuccess);
    err = cudaFree(w8); assert(err == cudaSuccess); err = cudaFree(b8); assert(err == cudaSuccess);
    err = cudaFree(w9); assert(err == cudaSuccess); err = cudaFree(b9); assert(err == cudaSuccess);
    err = cudaFree(w10); assert(err == cudaSuccess); err = cudaFree(b10); assert(err == cudaSuccess);
    err = cudaFree(w11); assert(err == cudaSuccess); err = cudaFree(b11); assert(err == cudaSuccess);
    err = cudaFree(w12); assert(err == cudaSuccess); err = cudaFree(b12); assert(err == cudaSuccess);
    err = cudaFree(w13); assert(err == cudaSuccess); err = cudaFree(b13); assert(err == cudaSuccess);
    err = cudaFree(w14); assert(err == cudaSuccess); err = cudaFree(b14); assert(err == cudaSuccess);
    err = cudaFree(w15); assert(err == cudaSuccess); err = cudaFree(b15); assert(err == cudaSuccess);
    err = cudaFree(w16); assert(err == cudaSuccess); err = cudaFree(b16); assert(err == cudaSuccess);
    err = cudaFree(w17); assert(err == cudaSuccess); err = cudaFree(b17); assert(err == cudaSuccess);
    err = cudaFree(w18); assert(err == cudaSuccess); err = cudaFree(b18); assert(err == cudaSuccess);
    err = cudaFree(w19); assert(err == cudaSuccess); err = cudaFree(b19); assert(err == cudaSuccess);
    err = cudaFree(w20); assert(err == cudaSuccess); err = cudaFree(b20); assert(err == cudaSuccess);
    err = cudaFree(w21); assert(err == cudaSuccess); err = cudaFree(b21); assert(err == cudaSuccess);
    err = cudaFree(w22); assert(err == cudaSuccess); err = cudaFree(b22); assert(err == cudaSuccess);
    err = cudaFree(w23); assert(err == cudaSuccess); err = cudaFree(b23); assert(err == cudaSuccess);
    err = cudaFree(w24); assert(err == cudaSuccess); err = cudaFree(b24); assert(err == cudaSuccess);
    err = cudaFree(w25); assert(err == cudaSuccess); err = cudaFree(b25); assert(err == cudaSuccess);
    err = cudaFree(w26); assert(err == cudaSuccess); err = cudaFree(b26); assert(err == cudaSuccess);
    err = cudaFree(w27); assert(err == cudaSuccess); err = cudaFree(b27); assert(err == cudaSuccess);
    err = cudaFree(w28); assert(err == cudaSuccess); err = cudaFree(b28); assert(err == cudaSuccess);
    err = cudaFree(w29); assert(err == cudaSuccess); err = cudaFree(b29); assert(err == cudaSuccess);
    err = cudaFree(w30); assert(err == cudaSuccess); err = cudaFree(b30); assert(err == cudaSuccess);
    err = cudaFree(w31); assert(err == cudaSuccess); err = cudaFree(b31); assert(err == cudaSuccess);
    err = cudaFree(w32); assert(err == cudaSuccess); err = cudaFree(b32); assert(err == cudaSuccess);
    err = cudaFree(w33); assert(err == cudaSuccess); err = cudaFree(b33); assert(err == cudaSuccess);
    err = cudaFree(w34); assert(err == cudaSuccess); err = cudaFree(b34); assert(err == cudaSuccess);
    err = cudaFree(w35); assert(err == cudaSuccess); err = cudaFree(b35); assert(err == cudaSuccess);
    err = cudaFree(w36); assert(err == cudaSuccess); err = cudaFree(b36); assert(err == cudaSuccess);
    err = cudaFree(w37); assert(err == cudaSuccess); err = cudaFree(b37); assert(err == cudaSuccess);
    err = cudaFree(w38); assert(err == cudaSuccess); err = cudaFree(b38); assert(err == cudaSuccess);
    err = cudaFree(w39); assert(err == cudaSuccess); err = cudaFree(b39); assert(err == cudaSuccess);
    err = cudaFree(w40); assert(err == cudaSuccess); err = cudaFree(b40); assert(err == cudaSuccess);
    err = cudaFree(w41); assert(err == cudaSuccess); err = cudaFree(b41); assert(err == cudaSuccess);
    err = cudaFree(w42); assert(err == cudaSuccess); err = cudaFree(b42); assert(err == cudaSuccess);
    err = cudaFree(w43); assert(err == cudaSuccess); err = cudaFree(b43); assert(err == cudaSuccess);
    err = cudaFree(w44); assert(err == cudaSuccess); err = cudaFree(b44); assert(err == cudaSuccess);
    err = cudaFree(w45); assert(err == cudaSuccess); err = cudaFree(b45); assert(err == cudaSuccess);
    err = cudaFree(w46); assert(err == cudaSuccess); err = cudaFree(b46); assert(err == cudaSuccess);
    err = cudaFree(w47); assert(err == cudaSuccess); err = cudaFree(b47); assert(err == cudaSuccess);
    err = cudaFree(w48); assert(err == cudaSuccess); err = cudaFree(b48); assert(err == cudaSuccess);
    err = cudaFree(w49); assert(err == cudaSuccess); err = cudaFree(b49); assert(err == cudaSuccess);
    err = cudaFree(w50); assert(err == cudaSuccess); err = cudaFree(b50); assert(err == cudaSuccess);
    err = cudaFree(w51); assert(err == cudaSuccess); err = cudaFree(b51); assert(err == cudaSuccess);
    err = cudaFree(w52); assert(err == cudaSuccess); err = cudaFree(b52); assert(err == cudaSuccess);
    err = cudaFree(w53); assert(err == cudaSuccess); err = cudaFree(b53); assert(err == cudaSuccess);
}

void TestOutputData(float* nums, int lens, int idx)
{
    float *nums_cpu = NULL;
    nums_cpu = (float*) malloc(lens * float_size);
    cudaMemcpy(nums_cpu, nums, lens * float_size, cudaMemcpyDeviceToHost);
    printf("%f\n", nums_cpu[idx]);
    free(nums_cpu);
}
