#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
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
    float *in_tensor = NULL, *out_tensor = NULL;
    move_imgs(input, &in_tensor, INPUTSHAPE);

    int in_shape = 244, in_c = 3;
    int k_shape = 3, out_c = 32;
    int stride = 2, pad = 1;
    // Block1
    // conv2d + relu6()
    conv2d(in_tensor, &out_tensor, w1, b1, in_shape, in_c, k_shape, out_c, stride, pad);
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
    depth_wise_conv(in_tensor, &out_tensor, w2, b2, in_shape, in_c, k_shape, out_c, stride, pad);
    // relu6();
    in_tensor = out_tensor;
    out_tensor = NULL;
    printf("addr: %p %p\n", in_tensor, out_tensor);
    test_output_data(in_tensor, 32 * 122 * 122, 71680);
    exit(0);

    // point_wise_conv();

/*
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
*/

}
