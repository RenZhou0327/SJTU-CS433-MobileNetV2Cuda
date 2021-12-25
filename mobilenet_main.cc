#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include "init_model.cuh"
#include "layers.cuh"

#define INPUTSHAPE 3 * 244 * 244
#define OUTPUTSHAPE 1000
#define TESTNUM 10
#define ITERNUM 500
float inputArr[TESTNUM][INPUTSHAPE];
float benchOutArr[TESTNUM][OUTPUTSHAPE];

void readInput(char *filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < INPUTSHAPE; j++)
            fscanf(fp, "%f", &inputArr[i][j]);
}

void readOutput(char *filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < OUTPUTSHAPE; j++)
            fscanf(fp, "%f", &benchOutArr[i][j]);
}

void checkOutput(float *out1, float *out2)
{
    float maxDiff = 0;
    for (int i = 0; i < OUTPUTSHAPE; i++)
    {
        maxDiff = (fabs(out1[i] - out2[i]) > maxDiff) ? fabs(out1[i] - out2[i]) : maxDiff;
    }
    if (maxDiff > 1e-5)
    {
        printf("Output dismatch. MaxDiff is %.7f\n", maxDiff);
    }
}

// TODO: 读取权重
void initModel();

// TODO: 实现自己的inference
void inference(float *input, float *output);


int main()
{
    
    initModel(); // 读取网络权重
    
    readInput("./mobilenetInput.txt");   // 读取输入
    readOutput("./mobilenetOutput.txt"); // 读取标准输出


    float sumTime = 0;
    for (int i = 0; i < TESTNUM; i++)
    {
        float inferOut[1000];
        for (int j = 0; j < ITERNUM; j++)
        {
            float Onetime;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

            // 执行Inference
            inference(inputArr[i], inferOut);
            
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&Onetime, start, stop);
            // 累加单次推理消耗时间
            sumTime += Onetime;
        }
        checkOutput(benchOutArr[i], inferOut);
    }
    printf("Average Time is: %f\n", (sumTime / TESTNUM / ITERNUM));
}

void initModel() {
    init_model();
}

void inference(float *input, float *output)
{
    // 注意这里imgs是CHW格式
    float *in_tensor = NULL, *out_tensor = NULL, *backup_tensor = NULL;
    st = clock();
    move_imgs(input, &in_tensor, INPUTSHAPE);
    et = clock();
    printf("move_img: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);

   
    // Block1
    int in_shape = 244, in_c = 3;
    int k_shape = 3, out_c = 32;
    int stride = 2, pad = 1;
    int out_lens = 32 * 122 * 122;
    bool is_relu, is_log = false;
    // conv2d + relu6()
    st = clock();
    conv2d(in_tensor, &out_tensor, w1, b1, in_shape, in_c, k_shape, out_c, stride, pad, &handle);
    et = clock();
    printf("conv1: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // relu6();
    in_tensor = out_tensor;
    out_tensor = NULL;
    // printf("addr: %p %p\n", in_tensor, out_tensor);
    // test_output_data(in_tensor, 32 * 122 * 122, 71680);
    // exit(0);

    // Block2:
    in_shape = 122, in_c = 32;
    k_shape = 3, out_c = 32;
    stride = 1, pad = 1;
    out_lens = 32 * 122 * 122;
    st = clock();
    depth_wise_conv(in_tensor, &out_tensor, w2, b2, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    et = clock();
    printf("conv2: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // relu6();
    in_tensor = out_tensor;
    out_tensor = NULL;
    // printf("addr: %p %p\n", in_tensor, out_tensor);
    // test_output_data(in_tensor, 32 * 122 * 122, 71680);
    // exit(0);

    in_shape = 122, in_c = 32;
    k_shape = 1, out_c = 16;
    stride = 1, pad = 0;
    out_lens =  16 * 122 * 122;
    st = clock();
    is_relu = false;
    point_wise_conv(in_tensor, &out_tensor, w3, b3, in_shape, in_c, out_c, is_relu, is_log, &handle);
    et = clock();
    printf("conv3: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    in_tensor = out_tensor;
    out_tensor = NULL;

    // Block3:
    in_shape = 122, in_c = 16;
    k_shape = 1, out_c = 96;
    stride = 1, pad = 0;
    out_lens = 96 * 122 * 122;
    is_relu = true;
    st = clock();
    point_wise_conv(in_tensor, &out_tensor, w4, b4, in_shape, in_c, out_c, is_relu, is_log, &handle);
    et = clock();
    printf("conv4: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, 1000, "./tmpfiles/325_relu.txt");
    in_tensor = out_tensor;
    out_tensor = NULL;
    // exit(0);

    // relu6();
    in_shape = 122, in_c = 96;
    k_shape = 3, out_c = 96;
    stride = 2, pad = 1;
    out_lens = 96 * 61 * 61;
    st = clock();
    depth_wise_conv(in_tensor, &out_tensor, w5, b5, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    et = clock();
    printf("conv5: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, 1000, "./tmpfiles/486.txt");
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 61, in_c = 96;
    k_shape = 1, out_c = 24;
    stride = 1, pad = 0;
    out_lens = 24 * 61 * 61;
    is_relu = false;
    st = clock();
    point_wise_conv(in_tensor, &out_tensor, w6, b6, in_shape, in_c, out_c, is_relu, is_log, &handle);
    et = clock();
    printf("conv6: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/489.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    store_back_up(in_tensor, &backup_tensor, out_lens);


    // Block4:
    in_shape = 61, in_c = 24;
    k_shape = 1, out_c = 144;
    stride = 1, pad = 0;
    out_lens = 144 * 61 * 61;
    is_relu = true;
    st = clock();
    point_wise_conv(in_tensor, &out_tensor, w7, b7, in_shape, in_c, out_c, is_relu, is_log, &handle);
    et = clock();
    printf("conv7: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/333_relu.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 61, in_c = 144;
    k_shape = 3, out_c = 144;
    stride = 1, pad = 1;
    out_lens = 144 * 61 * 61;
    st = clock();
    depth_wise_conv(in_tensor, &out_tensor, w8, b8, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    et = clock();
    printf("conv8: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/336_relu.txt");
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 61, in_c = 144;
    k_shape = 1, out_c = 24;
    stride = 1, pad = 0;
    out_lens = 24 * 61 * 61;
    is_relu = false;
    st = clock();
    point_wise_conv(in_tensor, &out_tensor, w9, b9, in_shape, in_c, out_c, is_relu, is_log, &handle);
    et = clock();
    printf("conv9: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/498.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 61;
    in_c = 24;
    st = clock();
    add_layer(in_tensor, backup_tensor, &out_tensor, in_c, in_shape);
    et = clock();
    printf("add1: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/339.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 61, in_c = 24;
    k_shape = 1, out_c = 144;
    stride = 1, pad = 0;
    out_lens = 144 * 61 * 61;
    is_relu = true;
    st = clock();
    point_wise_conv(in_tensor, &out_tensor, w10, b10, in_shape, in_c, out_c, is_relu, is_log, &handle);
    et = clock();
    printf("conv10: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/324.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 61, in_c = 144;
    k_shape = 3, out_c = 144;
    stride = 2, pad = 1;
    out_lens = 144 * 31 * 31;
    st = clock();
    depth_wise_conv(in_tensor, &out_tensor, w11, b11, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    et = clock();
    printf("conv11: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/345.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 31, in_c = 144;
    k_shape = 1, out_c = 32;
    stride = 1, pad = 0;
    out_lens = 32 * 31 * 31;
    is_relu = false;
    st = clock();
    point_wise_conv(in_tensor, &out_tensor, w12, b12, in_shape, in_c, out_c, is_relu, is_log, &handle);
    et = clock();
    printf("conv12: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/507.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    store_back_up(in_tensor, &backup_tensor, out_lens);


    // Block6:
    in_shape = 31, in_c = 32;
    k_shape = 1, out_c = 192;
    stride = 1, pad = 0;
    out_lens = 192 * 31 * 31;
    is_relu = true;
    st = clock();
    point_wise_conv(in_tensor, &out_tensor, w13, b13, in_shape, in_c, out_c, is_relu, is_log, &handle);
    et = clock();
    printf("conv13: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/350.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;


    in_shape = 31, in_c = 192;
    k_shape = 3, out_c = 192;
    stride = 1, pad = 1;
    out_lens = 192 * 31 * 31;
    st = clock();
    depth_wise_conv(in_tensor, &out_tensor, w14, b14, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    et = clock();
    printf("conv14: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/353.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 31, in_c = 192;
    k_shape = 1, out_c = 32;
    stride = 1, pad = 0;
    out_lens = 32 * 31 * 31;
    is_relu = false;
    st = clock();
    point_wise_conv(in_tensor, &out_tensor, w15, b15, in_shape, in_c, out_c, is_relu, is_log, &handle);
    et = clock();
    printf("conv15: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/516.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 31;
    in_c = 32;
    st = clock();
    add_layer(in_tensor, backup_tensor, &out_tensor, in_c, in_shape);
    et = clock();
    printf("add2: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/356.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    store_back_up(in_tensor, &backup_tensor, out_lens);

    // Block7:
    in_shape = 31, in_c = 32;
    k_shape = 1, out_c = 192;
    stride = 1, pad = 0;
    out_lens = 192 * 31 * 31;
    is_relu = true;
    st = clock();
    point_wise_conv(in_tensor, &out_tensor, w16, b16, in_shape, in_c, out_c, is_relu, is_log, &handle);
    et = clock();
    printf("conv16: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/350.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;


    in_shape = 31, in_c = 192;
    k_shape = 3, out_c = 192;
    stride = 1, pad = 1;
    out_lens = 192 * 31 * 31;
    st = clock();
    depth_wise_conv(in_tensor, &out_tensor, w17, b17, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    et = clock();
    printf("conv17: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/353.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 31, in_c = 192;
    k_shape = 1, out_c = 32;
    stride = 1, pad = 0;
    out_lens = 32 * 31 * 31;
    is_relu = false;
    st = clock();
    point_wise_conv(in_tensor, &out_tensor, w18, b18, in_shape, in_c, out_c, is_relu, is_log, &handle);
    et = clock();
    printf("conv18: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/516.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 31;
    in_c = 32;
    st = clock();
    add_layer(in_tensor, backup_tensor, &out_tensor, in_c, in_shape);
    et = clock();
    printf("add3: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // check_layer_data(out_tensor, out_lens, out_lens - 1, "./tmpfiles/365.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

/*
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
*/

}
