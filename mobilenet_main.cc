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
    
    readInput((char*)"./mobilenetInput.txt");   // 读取输入
    readOutput((char*)"./mobilenetOutput.txt"); // 读取标准输出


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

    FreeMemory();  // 释放动态内存
}

void initModel()
{
    InitModel();
}

void inference(float *input, float *output)
{


    // 注意这里imgs是CHW格式
    float *in_tensor = NULL, *out_tensor = NULL, *backup_tensor = NULL;
    // st = clock();
    MoveImgs(input, &in_tensor, INPUTSHAPE);   // img移动到gpu上
    // et = clock();
    // printf("time_move_img: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);

   
    // Block1   Conv2d + relu6
    int in_shape = 244, in_c = 3;
    int k_shape = 3, out_c = 32;
    int stride = 2, pad = 1;
    int out_lens = 32 * 122 * 122;
    bool is_relu, is_log = false;

    // st = clock();
    Conv2d(in_tensor, &out_tensor, w1, b1, in_shape, in_c, k_shape, out_c, stride, pad, &handle);
    // et = clock();
    // printf("time_conv1: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    in_tensor = out_tensor;
    out_tensor = NULL;


    // Block2: dw_conv + relu6 + pw_conv
    in_shape = 122, in_c = 32;
    k_shape = 3, out_c = 32;
    stride = 1, pad = 1;
    out_lens = 32 * 122 * 122;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w2, b2, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv2: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 122, in_c = 32;
    k_shape = 1, out_c = 16;
    stride = 1, pad = 0;
    out_lens =  16 * 122 * 122;
    // st = clock();
    is_relu = false;
    PointwiseConv(in_tensor, &out_tensor, w3, b3, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv3: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    in_tensor = out_tensor;
    out_tensor = NULL;


    // Block3: pw_conv + relu6 + dw_conv + relu6 + pw_conv
    in_shape = 122, in_c = 16;
    k_shape = 1, out_c = 96;
    stride = 1, pad = 0;
    out_lens = 96 * 122 * 122;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w4, b4, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv4: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, 1000, "./tmpfiles/325_relu.txt");
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 122, in_c = 96;
    k_shape = 3, out_c = 96;
    stride = 2, pad = 1;
    out_lens = 96 * 61 * 61;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w5, b5, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv5: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, 1000, "./tmpfiles/486.txt");
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 61, in_c = 96;
    k_shape = 1, out_c = 24;
    stride = 1, pad = 0;
    out_lens = 24 * 61 * 61;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w6, b6, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv6: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/489.txt");
    in_tensor = out_tensor;
    out_tensor = NULL;

    // store tensor used in add layer
    StoreBackup(in_tensor, &backup_tensor, out_lens);

    // Block4:  pw_conv + relu6 + dw_conv + relu6 + pw_conv + skip_conn
    in_shape = 61, in_c = 24;
    k_shape = 1, out_c = 144;
    stride = 1, pad = 0;
    out_lens = 144 * 61 * 61;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w7, b7, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv7: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/333_relu.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 61, in_c = 144;
    k_shape = 3, out_c = 144;
    stride = 1, pad = 1;
    out_lens = 144 * 61 * 61;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w8, b8, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv8: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/336_relu.txt");
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 61, in_c = 144;
    k_shape = 1, out_c = 24;
    stride = 1, pad = 0;
    out_lens = 24 * 61 * 61;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w9, b9, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv9: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/498.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    // skip connection
    in_shape = 61;
    in_c = 24;
    // st = clock();
    AddLayer(in_tensor, backup_tensor, &out_tensor, in_c, in_shape);
    // et = clock();
    // printf("time_add1: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/339.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;


    // Block5: pw_conv + relu6 + dw_conv + relu6 + pw_conv
    in_shape = 61, in_c = 24;
    k_shape = 1, out_c = 144;
    stride = 1, pad = 0;
    out_lens = 144 * 61 * 61;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w10, b10, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv10: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/324.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 61, in_c = 144;
    k_shape = 3, out_c = 144;
    stride = 2, pad = 1;
    out_lens = 144 * 31 * 31;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w11, b11, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv11: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/345.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 31, in_c = 144;
    k_shape = 1, out_c = 32;
    stride = 1, pad = 0;
    out_lens = 32 * 31 * 31;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w12, b12, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv12: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/507.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    StoreBackup(in_tensor, &backup_tensor, out_lens);


    // Block6: pw_conv + relu6 + dw_conv + relu6 + pw_conv + skip_conn
    in_shape = 31, in_c = 32;
    k_shape = 1, out_c = 192;
    stride = 1, pad = 0;
    out_lens = 192 * 31 * 31;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w13, b13, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv13: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/350.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;


    in_shape = 31, in_c = 192;
    k_shape = 3, out_c = 192;
    stride = 1, pad = 1;
    out_lens = 192 * 31 * 31;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w14, b14, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv14: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/353.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 31, in_c = 192;
    k_shape = 1, out_c = 32;
    stride = 1, pad = 0;
    out_lens = 32 * 31 * 31;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w15, b15, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv15: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/516.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 31;
    in_c = 32;
    // st = clock();
    AddLayer(in_tensor, backup_tensor, &out_tensor, in_c, in_shape);
    // et = clock();
    // printf("time_add2: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/356.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    StoreBackup(in_tensor, &backup_tensor, out_lens);


    // Block7:  pw_conv + relu6 + dw_conv + relu6 + pw_conv + skip_conn
    in_shape = 31, in_c = 32;
    k_shape = 1, out_c = 192;
    stride = 1, pad = 0;
    out_lens = 192 * 31 * 31;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w16, b16, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv16: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/350.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 31, in_c = 192;
    k_shape = 3, out_c = 192;
    stride = 1, pad = 1;
    out_lens = 192 * 31 * 31;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w17, b17, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv17: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/353.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 31, in_c = 192;
    k_shape = 1, out_c = 32;
    stride = 1, pad = 0;
    out_lens = 32 * 31 * 31;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w18, b18, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv18: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/516.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 31;
    in_c = 32;
    // st = clock();
    AddLayer(in_tensor, backup_tensor, &out_tensor, in_c, in_shape);
    // et = clock();
    // printf("time_add3: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/365.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;


    // Block8: pw_conv + relu6 + dw_conv + relu6 + pw_conv
    in_shape = 31, in_c = 32;
    k_shape = 1, out_c = 192;
    stride = 1, pad = 0;
    out_lens = 192 * 31 * 31;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w19, b19, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv19: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/368.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 31, in_c = 192;
    k_shape = 3, out_c = 192;
    stride = 2, pad = 1;
    out_lens = 192 * 16 * 16;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w20, b20, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv20: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/371.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16, in_c = 192;
    k_shape = 1, out_c = 64;
    stride = 1, pad = 0;
    out_lens = 64 * 16 * 16;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w21, b21, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv21: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/534.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    StoreBackup(in_tensor, &backup_tensor, out_lens);


    // Block9: pw_conv + relu6 + dw_conv + relu6 + pw_conv + skip_conn
    in_shape = 16, in_c = 64;
    k_shape = 1, out_c = 384;
    stride = 1, pad = 0;
    out_lens = 384 * 16 * 16;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w22, b22, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv22: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/376.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16, in_c = 384;
    k_shape = 3, out_c = 384;
    stride = 1, pad = 1;
    out_lens = 384 * 16 * 16;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w23, b23, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv23: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/379.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16, in_c = 384;
    k_shape = 1, out_c = 64;
    stride = 1, pad = 0;
    out_lens = 64 * 16 * 16;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w24, b24, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv24: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/543.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16;
    in_c = 64;
    // st = clock();
    AddLayer(in_tensor, backup_tensor, &out_tensor, in_c, in_shape);
    // et = clock();
    // printf("time_add4: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/382.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    StoreBackup(in_tensor, &backup_tensor, out_lens);


    // Block10: pw_conv + relu6 + dw_conv + relu6 + pw_conv + skip_conn
    in_shape = 16, in_c = 64;
    k_shape = 1, out_c = 384;
    stride = 1, pad = 0;
    out_lens = 384 * 16 * 16;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w25, b25, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv25: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/376.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16, in_c = 384;
    k_shape = 3, out_c = 384;
    stride = 1, pad = 1;
    out_lens = 384 * 16 * 16;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w26, b26, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv26: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/379.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16, in_c = 384;
    k_shape = 1, out_c = 64;
    stride = 1, pad = 0;
    out_lens = 64 * 16 * 16;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w27, b27, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv27: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/543.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16;
    in_c = 64;
    // st = clock();
    AddLayer(in_tensor, backup_tensor, &out_tensor, in_c, in_shape);
    // et = clock();
    // printf("time_add5: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/391.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    StoreBackup(in_tensor, &backup_tensor, out_lens);


    // Block11: pw_conv + relu6 + dw_conv + relu6 + pw_conv + skip_conn
    in_shape = 16, in_c = 64;
    k_shape = 1, out_c = 384;
    stride = 1, pad = 0;
    out_lens = 384 * 16 * 16;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w28, b28, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv28: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/376.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16, in_c = 384;
    k_shape = 3, out_c = 384;
    stride = 1, pad = 1;
    out_lens = 384 * 16 * 16;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w29, b29, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv29: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/379.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16, in_c = 384;
    k_shape = 1, out_c = 64;
    stride = 1, pad = 0;
    out_lens = 64 * 16 * 16;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w30, b30, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv30: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/543.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16;
    in_c = 64;
    // st = clock();
    AddLayer(in_tensor, backup_tensor, &out_tensor, in_c, in_shape);
    // et = clock();
    // printf("time_add6: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/400.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;


    // Block12: pw_conv + relu6 + dw_conv + relu6 + pw_conv
    in_shape = 16, in_c = 64;
    k_shape = 1, out_c = 384;
    stride = 1, pad = 0;
    out_lens = 384 * 16 * 16;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w31, b31, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv31: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/376.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16, in_c = 384;
    k_shape = 3, out_c = 384;
    stride = 1, pad = 1;
    out_lens = 384 * 16 * 16;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w32, b32, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv32: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/379.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16, in_c = 384;
    k_shape = 1, out_c = 96;
    stride = 1, pad = 0;
    out_lens = 96 * 16 * 16;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w33, b33, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv33: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/570.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    StoreBackup(in_tensor, &backup_tensor, out_lens);


    // Block13: pw_conv + relu6 + dw_conv + relu6 + pw_conv + skip_conn
    in_shape = 16, in_c = 96;
    k_shape = 1, out_c = 576;
    stride = 1, pad = 0;
    out_lens = 576 * 16 * 16;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w34, b34, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv34: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/376.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16, in_c = 576;
    k_shape = 3, out_c = 576;
    stride = 1, pad = 1;
    out_lens = 576 * 16 * 16;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w35, b35, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv35: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/379.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16, in_c = 576;
    k_shape = 1, out_c = 96;
    stride = 1, pad = 0;
    out_lens = 96 * 16 * 16;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w36, b36, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv36: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/417.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16;
    in_c = 96;
    // st = clock();
    AddLayer(in_tensor, backup_tensor, &out_tensor, in_c, in_shape);
    // et = clock();
    // printf("time_add7: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/417.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    StoreBackup(in_tensor, &backup_tensor, out_lens);


    // Block14: pw_conv + relu6 + dw_conv + relu6 + pw_conv + skip_conn
    in_shape = 16, in_c = 96;
    k_shape = 1, out_c = 576;
    stride = 1, pad = 0;
    out_lens = 576 * 16 * 16;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w37, b37, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv37: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/376.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16, in_c = 576;
    k_shape = 3, out_c = 576;
    stride = 1, pad = 1;
    out_lens = 576 * 16 * 16;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w38, b38, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv38: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/379.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16, in_c = 576;
    k_shape = 1, out_c = 96;
    stride = 1, pad = 0;
    out_lens = 96 * 16 * 16;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w39, b39, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv39: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/417.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16;
    in_c = 96;
    // st = clock();
    AddLayer(in_tensor, backup_tensor, &out_tensor, in_c, in_shape);
    // et = clock();
    // printf("time_add8: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/426.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;


    // Block15: pw_conv + relu6 + dw_conv + relu6 + pw_conv
    in_shape = 16, in_c = 96;
    k_shape = 1, out_c = 576;
    stride = 1, pad = 0;
    out_lens = 576 * 16 * 16;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w40, b40, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv40: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/368.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 16, in_c = 576;
    k_shape = 3, out_c = 576;
    stride = 2, pad = 1;
    out_lens = 576 * 8 * 8;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w41, b41, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv41: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/371.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 8, in_c = 576;
    k_shape = 1, out_c = 160;
    stride = 1, pad = 0;
    out_lens = 160 * 8 * 8;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w42, b42, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv42: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/597.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    StoreBackup(in_tensor, &backup_tensor, out_lens);


    // Block16: pw_conv + relu6 + dw_conv + relu6 + pw_conv + skip_conn
    in_shape = 8, in_c = 160;
    k_shape = 1, out_c = 960;
    stride = 1, pad = 0;
    out_lens = 960 * 8 * 8;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w43, b43, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv43: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/376.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 8, in_c = 960;
    k_shape = 3, out_c = 960;
    stride = 1, pad = 1;
    out_lens = 960 * 8 * 8;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w44, b44, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv44: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/379.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 8, in_c = 960;
    k_shape = 1, out_c = 160;
    stride = 1, pad = 0;
    out_lens = 160 * 8 * 8;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w45, b45, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv45: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/606.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 8;
    in_c = 160;
    // st = clock();
    AddLayer(in_tensor, backup_tensor, &out_tensor, in_c, in_shape);
    // et = clock();
    // printf("time_add9: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/443.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    StoreBackup(in_tensor, &backup_tensor, out_lens);


    // Block17: pw_conv + relu6 + dw_conv + relu6 + pw_conv + skip_conn
    in_shape = 8, in_c = 160;
    k_shape = 1, out_c = 960;
    stride = 1, pad = 0;
    out_lens = 960 * 8 * 8;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w46, b46, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv46: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/376.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 8, in_c = 960;
    k_shape = 3, out_c = 960;
    stride = 1, pad = 1;
    out_lens = 960 * 8 * 8;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w47, b47, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv47: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/449.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 8, in_c = 960;
    k_shape = 1, out_c = 160;
    stride = 1, pad = 0;
    out_lens = 160 * 8 * 8;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w48, b48, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv48: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/606.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 8;
    in_c = 160;
    // st = clock();
    AddLayer(in_tensor, backup_tensor, &out_tensor, in_c, in_shape);
    // et = clock();
    // printf("time_add10: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/452.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;


    // Block18: pw_conv + relu6 + dw_conv + relu6 + pw_conv
    in_shape = 8, in_c = 160;
    k_shape = 1, out_c = 960;
    stride = 1, pad = 0;
    out_lens = 960 * 8 * 8;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w49, b49, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv49: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/376.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 8, in_c = 960;
    k_shape = 3, out_c = 960;
    stride = 1, pad = 1;
    out_lens = 960 * 8 * 8;
    // st = clock();
    DepthwiseConv(in_tensor, &out_tensor, w50, b50, in_shape, in_c, k_shape, out_c, stride, pad, is_log);
    // et = clock();
    // printf("time_conv50: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/449.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    in_shape = 8, in_c = 960;
    k_shape = 1, out_c = 320;
    stride = 1, pad = 0;
    out_lens = 320 * 8 * 8;
    is_relu = false;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w51, b51, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv51: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/624.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;


    // Block19: pw_conv
    in_shape = 8, in_c = 320;
    k_shape = 1, out_c = 1280;
    stride = 1, pad = 0;
    out_lens = 1280 * 8 * 8;
    is_relu = true;
    // st = clock();
    PointwiseConv(in_tensor, &out_tensor, w52, b52, in_shape, in_c, out_c, is_relu, is_log, &handle);
    // et = clock();
    // printf("time_conv52: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/463.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;


    // Block20: global_avg_pool + linear
    in_shape = 8;
    in_c = 1280;
    out_lens = 1280;
    // st = clock();
    GlobalAvgPool(in_tensor, &out_tensor, in_c, in_shape);
    // et = clock();
    // printf("time_avg: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/464.txt");
    // exit(0);
    in_tensor = out_tensor;
    out_tensor = NULL;

    int in_lens = 1280;
    out_lens = 1000;
    // st = clock();
    LinearLayer(in_tensor, &out_tensor, w53, b53, in_lens, out_lens, &handle);
    // et = clock();
    // printf("time_linear: %lf\n", (double)(et - st) / CLOCKS_PER_SEC);
    // CheckLayerData(out_tensor, out_lens, out_lens - 1, "./tmpfiles/473.txt");
    // exit(0);
    cudaMemcpy(output, out_tensor, out_lens * sizeof(float), cudaMemcpyDeviceToHost);

}
