#include <iostream>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cublas.h>

using namespace std;

FILE *in_w, *in_b;

void conv2d();
void relu6();
void depth_wise_conv();
void point_wise_conv();
void add_layer();
void avg_pool();
void linear_layer();

void infer() {
    
    // Block1: W (32, 3, 3, 3), b (32,), In (1, 3, 244, 244) Out ()
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
    char image_path[] = "./images/batch_images.txt";
    char weight_path[] = "./parameters/weight_data.txt";
    char bias_path[] = "./parameters/bias_data.txt";
    char infer_res_path[] = "./results/";
    
    in_w = fopen(weight_path, "r");
    in_b = fopen(bias_path, "r");

    infer();
    
    // clock_t st, et;
    // st = clock();
    // float *weights = (float *)malloc(32 * (3 * 3 * 3 + 1) * sizeof(float));
    // for (int i = 0; i < 32; ++i) {
    //     for (int j = 0; j < 3 * 3 * 3; ++j) {
    //         fscanf(in_w, "%f", &weights[j * 32 + i]);
    //     }
    //     fscanf(in_b, "%f", &weights[32 * 3 * 3 * 3 + i]);
    // }
    // for (int i = 0; i < 32 * (3 * 3 * 3 + 1); ++i) {
    //     printf("%f\n", weights[i]);
    // }
    // et = clock();
    // cout<< "time:"<<double(et - st) / CLOCKS_PER_SEC<<endl;
    fclose(in_w);
    fclose(in_b);
    return 0;
}