#include "layers.cuh"


__global__ void add_bias_relu6(float* WX, float *B, int out_c, int out_shape) 
{
    // thread_j: [0, 122 * 32)
    int thread_j = blockIdx.x * blockDim.x + threadIdx.x;
    // thread_i: [0, 122)
    int thread_i = blockIdx.y;
    int num_id = thread_i * out_c * out_shape + thread_j;
    int b_id = num_id / (out_shape * out_shape);
    WX[num_id] += B[b_id];
    // RELU6
    WX[num_id] = max(WX[num_id], 0.0);
    WX[num_id] = min(WX[num_id], 6.0);
    // printf("%d %d\n", thread_i, thread_j);
}


__global__ void img2col(float *imgs, float *cols, int in_shape, int out_shape, int k_shape, int in_c, int s, int p) {
    int thread_j = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_i = blockIdx.y * blockDim.y + threadIdx.y;
    int cols_id = thread_i * (out_shape * out_shape) + thread_j;
    
    // index in cols
    int row_idx = cols_id / (out_shape * out_shape);
    int col_idx = cols_id % (out_shape * out_shape);
    
    // index in imgs
    int c_idx = row_idx / (k_shape * k_shape); // In fact, c_idx == blockIdx.y
    int i_idx = (row_idx / k_shape) % k_shape + (col_idx / out_shape) * s - p;
    int j_idx = row_idx % in_c + (col_idx % out_shape) * s - p;
    // if (d_idx == 2) {
    //     printf("%d %d %d %d\n", cols_id, d_idx, i_idx, j_idx);
    // }
    if (i_idx >= 0 && j_idx >= 0) {
        int img_idx = c_idx * (in_shape * in_shape) + i_idx * in_shape + j_idx;
        cols[cols_id] = imgs[img_idx];
    }
}


void mat_multiple(float *A, float *B, float* C, int m, int k, int n) {
	cublasStatus_t status;
	cublasHandle_t handle;

    const float al = 1.0f, bt = 0.0f;
    status = cublasCreate(&handle);
    assert(status == CUBLAS_STATUS_SUCCESS);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &al, B, n, A, k, &bt, C, n);
}


void conv2d(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int k_shape, int out_c, int stride, int pad)
{
    int out_shape = int((in_shape + 2 * pad - k_shape) / stride) + 1;
    // printf("out shape: %d\n", out_shape);

    float *in_cols = NULL;
    int threadNum = k_shape * k_shape * in_c * out_shape * out_shape;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&in_cols, threadNum * sizeof(float));
    assert(err == cudaSuccess);
    
    int bIndx = ceil(out_shape * out_shape / 32.0), bIndy = in_c;
    int tIndx = 32, tIndy = k_shape * k_shape;
    // printf("%d %d %d %d %d\n", threadNum, bIndx, bIndy, tIndx, tIndy);
    // exit(0);
    dim3 gDim1(bIndx, bIndy);
    // !!! 特别注意, tIndx * tIndy得小于1024, 否则出错无结果!!!
    dim3 bDim1(tIndx, tIndy);
    img2col<<<gDim1, bDim1>>>(in_tensor, in_cols, in_shape, out_shape, k_shape, in_c, stride, pad);
    cudaFree(in_tensor);

    // // Just for Test:
    // float *temp = (float*) malloc(threadNum * sizeof(float));
    // err = cudaMemcpy(temp, in_cols, threadNum * sizeof(float), cudaMemcpyDeviceToHost);
    // assert(err == cudaSuccess);
    // printf("%f\n", temp[400868]);
    // for (int i = 0; i < 10; ++i) {
    //     printf("%f ", temp[10388]);
    // }
    // printf("\n");
    // exit(0);

    float *out_tensor = NULL;
    int out_lens = out_c * out_shape * out_shape;
    int mat_m = out_c, mat_k = in_c * k_shape * k_shape, mat_n = out_shape * out_shape;
    err = cudaMalloc((void**)&out_tensor, out_lens * sizeof(float));
    assert(err == cudaSuccess);
    mat_multiple(w, in_cols, out_tensor, mat_m, mat_k, mat_n);
    err = cudaFree(w);
    assert(err == cudaSuccess);
    err = cudaFree(in_cols);
    assert(err == cudaSuccess);

    // printf("%d %d\n", out_c, out_shape);
    dim3 gDim2(out_shape, out_shape);
    dim3 bDim2(out_c, 1);
    add_bias_relu6<<<gDim2, bDim2>>>(out_tensor, b, out_c, out_shape);
    cudaFree(b);
    
    // // Just for Test:
    // float *temp = (float*) malloc(out_lens * sizeof(float));
    // err = cudaMemcpy(temp, out_tensor, out_lens * sizeof(float), cudaMemcpyDeviceToHost);
    // assert(err == cudaSuccess);
    // // printf("%f\n", temp[416288]);
    // FILE *test_file = fopen("tmpfiles/317_relu.txt", "w");
    // for (int i = 0; i < out_lens; ++i) {
    //     fprintf(test_file, "%f ", temp[i]);
    // }
    // fprintf(test_file, "\n");
    // fclose(test_file);
    // exit(0);
    
    *out_tensor_p = out_tensor;
};


void relu6() {};


__global__ void depthwise_kernel(float *in_tensor, float *out_tensor, float *w, float *b, int in_shape, int out_shape, int k_shape, int c, int s, int p) {
    
    int thread_j = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_i = blockIdx.y;
    int num_id = thread_i * c * out_shape + thread_j + 1;
    // if (num_id == 32 * 122 * 122 - 1) {
    //     printf("%d %d %d\n", thread_i, thread_j, num_id);
    // }

    // num_id [0, 32 * 122 * 122)

    // 确定out_tensor中num_id这个位置在(C, H, W)形式中的位置
    int out_c = num_id / (out_shape * out_shape);
    int out_i = (num_id / out_shape) % out_shape;
    int out_j = num_id % out_shape;

    int i_st = out_i - p, j_st = out_j - p;
    int i_ed = i_st + (k_shape - 1) * s, j_ed = j_st + (k_shape - 1) * s;

    float res = 0.0f;
    const float* const img_bias = in_tensor + out_c * in_shape * in_shape;
    const float* const weight_bias = w + out_c * k_shape * k_shape;

    int k_pos = 0;
    float img_value = 0.0f;
    for (int i = i_st; i <= i_ed; i += s) {
        for (int j = j_st; j <= j_ed; j += s) {
            img_value = (i < 0 || i >= in_shape || j < 0 || j >= in_shape) ? 0.0f: img_bias[i * in_shape + j]; 
            res += weight_bias[k_pos] * img_value;
            ++k_pos;
        }
    }
    res += b[out_c];
    res = max(res, 0.0);
    res = min(res, 6.0);
    out_tensor[num_id] = res;
}


void depth_wise_conv(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int k_shape, int out_c, int stride, int pad)
{
    int out_shape = int((in_shape + 2 * pad - k_shape) / stride) + 1;
    printf("%d %d %d %d %d %d\n", in_shape, in_c, k_shape, out_c, stride, pad);
    printf("out shape: %d\n", out_shape);
    
    int threadNum = out_c * out_shape * out_shape;
    printf("thd num: %d\n", threadNum);
    // exit(0);
    float* out_tensor = NULL;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&out_tensor, threadNum * sizeof(float));
    assert(err == cudaSuccess);

    dim3 gDim(out_shape, out_shape);
    dim3 dDim(out_c, 1);

    depthwise_kernel<<<gDim, dDim>>>(in_tensor, out_tensor, w, b, in_shape, out_shape, k_shape, out_c, stride, pad);
    cudaFree(in_tensor);
    cudaFree(w);
    cudaFree(b);

    // // Just for Test:
    // float *temp = (float*) malloc(threadNum * sizeof(float));
    // err = cudaMemcpy(temp, out_tensor, threadNum * sizeof(float), cudaMemcpyDeviceToHost);
    // assert(err == cudaSuccess);
    // // printf("%f\n", temp[416288]);
    // FILE *test_file = fopen("tmpfiles/320_relu.txt", "w");
    // for (int i = 0; i < threadNum; ++i) {
    //     fprintf(test_file, "%f ", temp[i]);
    // }
    // fprintf(test_file, "\n");
    // fclose(test_file);
    // exit(0);

    *out_tensor_p = out_tensor;
};


void point_wise_conv(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int out_c)
{
    // k_shape = 1, stride = 1, pad = 0
    conv2d(in_tensor, out_tensor_p, w, b, in_shape, in_c, 1, out_c, 1, 0);
};

void add_layer() {};
void avg_pool() {};
void linear_layer() {};
