#include "layers.cuh"


__global__ void img2col(float *imgs, float *cols, int in_shape, int out_shape, int k_shape, int in_c, int s, int p) {
    int thread_j = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_i = blockIdx.y * blockDim.y + threadIdx.y;
    int cols_id = thread_i * (out_shape * out_shape) + thread_j;
    
    int row_idx = cols_id / (out_shape * out_shape);
    int col_idx = cols_id % (out_shape * out_shape);
    
    int c_idx = row_idx / (k_shape * k_shape);
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

void matGemm(float *A, float *B, float* C, int m, int k, int n) {
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
    printf("out shape: %d\n", out_shape);

    float *in_cols = NULL;
    int threadNum = k_shape * k_shape * in_c * out_shape * out_shape;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&in_cols, threadNum * sizeof(float));
    assert(err == cudaSuccess);
    
    int bIndx = ceil(out_shape * out_shape / 32.0), bIndy = in_c;
    int tIndx = 32, tIndy = k_shape * k_shape;
    // printf("%d %d %d %d %d\n", threadNum, bIndx, bIndy, tIndx, tIndy);
    // exit(0);
    dim3 gDim(bIndx, bIndy);
    dim3 bDim(tIndx, tIndy);
    img2col<<<gDim, bDim>>>(in_tensor, in_cols, in_shape, out_shape, k_shape, in_c, stride, pad);
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
    matGemm(w, in_cols, out_tensor, mat_m, mat_k, mat_n);
    cudaFree(w);
    cudaFree(in_cols);

    // Just for Test:
    float *temp = (float*) malloc(out_lens * sizeof(float));
    err = cudaMemcpy(temp, out_tensor, out_lens * sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    printf("%f\n", temp[416288]);
    // for (int i = 0; i < 10; ++i) {
    //     printf("%f ", temp[10388]);
    // }
    printf("\n");
    exit(0);
    // cudaDeviceSynchronize();
};

void relu6() {};
void depth_wise_conv() {};
void point_wise_conv() {};
void add_layer() {};
void avg_pool() {};
void linear_layer() {};
