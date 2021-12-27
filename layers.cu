#include "layers.cuh"

__constant__ float const_bias[1280];
time_t s_t, e_t, in_s_t, in_e_t;

void check_layer_data(float* out_tensor, int out_lens, int idx, char* file_name) {
    cudaError_t err = cudaSuccess;
    float *temp = (float*) malloc(out_lens * sizeof(float));
    err = cudaMemcpy(temp, out_tensor, out_lens * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("%d %s.\n", err, cudaGetErrorString(err));
    assert(err == cudaSuccess);
    printf("%f\n", temp[idx]);
    FILE *test_file = fopen(file_name, "w");
    for (int i = 0; i < out_lens; ++i) {
        fprintf(test_file, "%f ", temp[i]);
    }
    fprintf(test_file, "\n");
    fclose(test_file);
}

void store_back_up(float* in_tensor, float** out_tensor_p, int out_lens) {
    
    float* out_tensor;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&out_tensor, out_lens * sizeof(float));
    assert(err == cudaSuccess);
    err = cudaMemcpy(out_tensor, in_tensor, out_lens * sizeof(float), cudaMemcpyDeviceToDevice);
    assert(err == cudaSuccess);
    *out_tensor_p = out_tensor;

}

__global__ void add_bias_relu6(float* WX, int out_c, int out_shape) 
{
    // thread_j: [0, 122 * 32)
    int thread_j = blockIdx.x * blockDim.x + threadIdx.x;
    // thread_i: [0, 122)
    int thread_i = blockIdx.y;
    int num_id = thread_i * out_c * out_shape + thread_j;
    int b_id = num_id / (out_shape * out_shape);
    WX[num_id] += const_bias[b_id];
    // RELU6
    WX[num_id] = max(WX[num_id], 0.0f);
    WX[num_id] = min(WX[num_id], 6.0f);
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


void mat_multiply_cublas(float *A, float *B, float* C, int m, int k, int n, const float al, const float bt, cublasHandle_t* handle_p)
{   
    // cublasSgemm:
    // C = alpha * A * B + beta * C
    // A:(m, k) B:(k, n) C:(m, n)
    // op(A) = A if transa == CUBLAS_OP_N, column major
    // op(A) = A^T if transa == CUBLAS_OP_T, row major
    // op(A) = A^H if transa == CUBLAS_OP_C
    // Two common usages:
    // 1. CT = BT * AT: cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,m,n,k,&alpha,A,m,B,k,&beta,C,m)
    // 2. C = AB: cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&alpha,B,n,A,k,&beta,C,n)
    in_s_t = clock();
    cublasSgemm(*handle_p, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &al, B, n, A, k, &bt, C, n);
    in_e_t = clock();
    // printf("gemm inner: %lf\n", (double)(in_e_t - in_s_t) / CLOCKS_PER_SEC);
}


void conv2d(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int k_shape, int out_c, int stride, int pad, cublasHandle_t* handle_p)
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
    dim3 gDim(bIndx, bIndy);
    // !!! 特别注意, tIndx * tIndy得小于1024, 否则无执行结果!!!
    dim3 bDim(tIndx, tIndy);
    s_t = clock();
    img2col<<<gDim, bDim>>>(in_tensor, in_cols, in_shape, out_shape, k_shape, in_c, stride, pad);
    e_t = clock();
    // printf("img2col: %lf\n", (double)(e_t - s_t) / CLOCKS_PER_SEC);
    err = cudaFree(in_tensor);
    assert(err == cudaSuccess);

    float *out_tensor = NULL;
    int out_lens = out_c * out_shape * out_shape;
    int mat_m = out_c, mat_k = in_c * k_shape * k_shape, mat_n = out_shape * out_shape;
    err = cudaMalloc((void**)&out_tensor, out_lens * sizeof(float));
    assert(err == cudaSuccess);
    s_t = clock();
    mat_multiply_cublas(w, in_cols, out_tensor, mat_m, mat_k, mat_n, 1.0f, 0.0f, handle_p);
    e_t = clock();
    // printf("gemm: %lf\n", (double)(e_t - s_t) / CLOCKS_PER_SEC);
    // err = cudaFree(w);
    // assert(err == cudaSuccess);
    err = cudaFree(in_cols);
    assert(err == cudaSuccess);

    // printf("%d %d\n", out_c, out_shape);
    dim3 gDim_bias(out_shape, out_shape);
    dim3 bDim_bias(out_c, 1);
    err = cudaMemcpyToSymbol(const_bias, b, out_c * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    assert(err == cudaSuccess);
    s_t = clock();
    add_bias_relu6<<<gDim_bias, bDim_bias>>>(out_tensor, out_c, out_shape);
    e_t = clock();
    // printf("add bias: %lf\n", (double)(e_t - s_t) / CLOCKS_PER_SEC);
    // err = cudaFree(b);
    // assert(err == cudaSuccess);
    
    *out_tensor_p = out_tensor;
};


__global__ void depthwise_kernel(float *in_tensor, float *out_tensor, float *w, float *b, int in_shape, int out_shape, int k_shape, int c, int s, int p) {
    
    // dim3 gDim(out_shape, out_shape);
    // dim3 dDim(out_c, 1);
    //
    int thread_j = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_i = blockIdx.y;
    int num_id = thread_i * c * out_shape + thread_j;
    // num_id [0, 32 * 122 * 122)

    // 确定out_tensor中num_id这个位置在(C, H, W)形式中的位置
    int out_c = num_id / (out_shape * out_shape);
    int out_i = (num_id / out_shape) % out_shape;
    int out_j = num_id % out_shape;

    
    int i_st = out_i * s - p, j_st = out_j * s - p;
    int i_ed = i_st + k_shape - 1, j_ed = j_st + k_shape - 1;
    
    float res = 0.0f;
    const float* const img_bias = in_tensor + out_c * in_shape * in_shape;
    const float* const weight_bias = w + out_c * k_shape * k_shape;

    int k_pos = 0;
    float img_value = 0.0f;
    for (int i = i_st; i <= i_ed; ++i) {
        for (int j = j_st; j <= j_ed; ++j) {
            img_value = (i < 0 || i >= in_shape || j < 0 || j >= in_shape) ? 0.0f: img_bias[i * in_shape + j]; 
            res += weight_bias[k_pos] * img_value;
            ++k_pos;
        }
    }
    res += const_bias[out_c];
    res = max(res, 0.0);
    res = min(res, 6.0);
    out_tensor[num_id] = res;
}


void depthwise_conv(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int k_shape, int out_c, int stride, int pad, bool is_log)
{
    int out_shape = int((in_shape + 2 * pad - k_shape) / stride) + 1;
    // printf("%d %d %d %d %d %d\n", in_shape, in_c, k_shape, out_c, stride, pad);
    // printf("out shape: %d\n", out_shape);
    
    int out_lens = out_c * out_shape * out_shape;

    float* out_tensor = NULL;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&out_tensor, out_lens * sizeof(float));
    assert(err == cudaSuccess);

    dim3 gDim(out_shape, out_shape);
    dim3 dDim(out_c, 1);

    // if (is_log) {
    //     check_layer_data(out_tensor, out_lens, 1000, "./tmpfiles/486.txt");
    //     exit(0);
    // }

    err = cudaMemcpyToSymbol(const_bias, b, out_c * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    assert(err == cudaSuccess);
    depthwise_kernel<<<gDim, dDim>>>(in_tensor, out_tensor, w, b, in_shape, out_shape, k_shape, out_c, stride, pad);
    cudaDeviceSynchronize();
    err = cudaFree(in_tensor);
    assert(err == cudaSuccess);
    // cudaFree(w);
    // assert(err == cudaSuccess);
    // cudaFree(b);
    // assert(err == cudaSuccess);

    *out_tensor_p = out_tensor;
};


__global__ void pw_img2col(float *imgs, float *cols, int shape, int in_c) {
    
    // dim3 gDim(in_c, in_shape);
    // dim3 bDim(in_shape, 1);
    int b_id = blockIdx.y * gridDim.x + blockIdx.x;
    int t_id = threadIdx.x;    // w_idx
    int num_id = b_id * blockDim.x + t_id;
    int c_idx = num_id / (shape * shape);
    b_id %= shape;  // h_idx
    cols[c_idx * shape * shape + b_id * shape + t_id] = imgs[num_id];
}


__global__ void add_bias(float* WX, int out_c, int out_shape) 
{
    // dim3 gDim_bias(out_shape, out_shape);
    // dim3 bDim_bias(out_c, 1);

    int thread_j = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_i = blockIdx.y;
    int num_id = thread_i * out_c * out_shape + thread_j;
    int b_id = num_id / (out_shape * out_shape);
    WX[num_id] += const_bias[b_id];
}


void pointwise_conv(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_shape, int in_c, int out_c, bool is_relu, bool is_log, cublasHandle_t* handle_p)
{
    int out_shape = in_shape;
    float *in_cols = NULL;

    int threadNum = in_c * out_shape * out_shape;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&in_cols, threadNum * sizeof(float));
    assert(err == cudaSuccess);

    dim3 gDim(in_c, in_shape);
    dim3 bDim(in_shape, 1);

    pw_img2col<<<gDim, bDim>>>(in_tensor, in_cols, out_shape, in_c);
    err = cudaFree(in_tensor);
    assert(err == cudaSuccess);

    float *out_tensor = NULL;
    int out_lens = out_c * out_shape * out_shape;

    int mat_m = out_c, mat_k = in_c, mat_n = out_shape * out_shape;
    err = cudaMalloc((void**)&out_tensor, out_lens * sizeof(float));
    assert(err == cudaSuccess);
    mat_multiply_cublas(w, in_cols, out_tensor, mat_m, mat_k, mat_n, 1.0f, 0.0f, handle_p);
    err = cudaFree(in_cols);
    assert(err == cudaSuccess);

    int bIndx = out_shape, bIndy = out_shape;
    int tIndx = out_c;
    if (out_c > 1024) {
        bIndx = out_shape * 2;
        tIndx = ceil(tIndx / 2.0);
        // printf("hello\n");
    }

    dim3 gDim_bias(bIndx, bIndy);
    dim3 bDim_bias(tIndx, 1);

    err = cudaMemcpyToSymbol(const_bias, b, out_c * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    assert(err == cudaSuccess);
    if (is_relu) {
        add_bias_relu6<<<gDim_bias, bDim_bias>>>(out_tensor, out_c, out_shape);
    }
    else {
        add_bias<<<gDim_bias, bDim_bias>>>(out_tensor, out_c, out_shape);
    }

    // if (is_log) {
    //     // printf("%d %d %d\n", mat_m, mat_k, mat_n);
    //     check_layer_data(out_tensor, out_lens, 1000, "./tmpfiles/325_relu.txt");
    //     exit(0);
    // }

    // err = cudaFree(b);
    // assert(err == cudaSuccess);

    *out_tensor_p = out_tensor;
    // exit(0);
};

__global__ void addlayer_kernel(float* A, float* B, float* C, int channels, int shape)
{
    // dim3 gDim(shape, shape);
    // dim3 bDim(channels, 1);
    int thread_j = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_i = blockIdx.y;
    int idx = thread_i * channels * shape + thread_j;
    C[idx] = A[idx] + B[idx];
}

void add_layer(float* A, float* B, float** C_p, int channels, int shape)
{
    int out_lens = channels * shape * shape;
    float* C = NULL;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&C, out_lens * sizeof(float));
    
    dim3 gDim(shape, shape);
    dim3 bDim(channels, 1);

    addlayer_kernel<<<gDim, bDim>>>(A, B, C, channels, shape);
    err = cudaFree(A);
    assert(err == cudaSuccess);
    err = cudaFree(B);
    assert(err == cudaSuccess);
    // 注意，addlayer的原始的那一层在pointwise后不能free掉
    *C_p = C;
}

__global__ void avg_pool_kernel(float* in_tensor, float* out_tensor, int channels, int in_shape)
{
    // dim3 gDim(ceil(channels / 512.0), 1);
    // dim3 bDim(32, 16);
    int thread_j = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_i = blockIdx.y * blockDim.y + threadIdx.y;
    // `out_idx` also means which slice of in_tensor we are now avgpooling
    int out_idx = thread_i * gridDim.x * blockDim.x + thread_j;

    float sum = 0.0f;
    for (int i = 0; i < in_shape; ++i)
    {
        for (int j = 0; j < in_shape; ++j)
        {
            // TODO: 有什么可以优化的吗，比如in_tensor数组取值？
            sum += in_tensor[out_idx * in_shape * in_shape + i * in_shape + j];
        }
    }
    int divisor = in_shape * in_shape; // num of total grids in one channel of in_tensor
    out_tensor[out_idx] = sum / divisor;
}

void avg_pool(float* in_tensor, float** out_tensor_p, int channels, int in_shape)
{
    // TODO:可能可以改变gDim和bDim, 应用上share memory
    // TODO:将所有的dim3都加上ceil
    dim3 gDim(ceil(channels / 512.0), 1);
    dim3 bDim(32, 16);

    float* out_tensor = NULL;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&out_tensor, channels * sizeof(float));
    assert(err == cudaSuccess);

    avg_pool_kernel<<<gDim, bDim>>>(in_tensor, out_tensor, channels, in_shape);

    err = cudaFree(in_tensor);
    assert(err == cudaSuccess);
    *out_tensor_p = out_tensor;
}


void linear_layer(float* in_tensor, float** out_tensor_p, float* w, float* b, int in_len, int out_len, cublasHandle_t* handle_p)
{
    // W:(1000, 1280) X:(1, 1280) b:(1000,) Y:(1,1000)
    float* out_tensor = NULL;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void**)&out_tensor, out_len * sizeof(float));
    assert(err == cudaSuccess);
    mat_multiply_cublas(w, in_tensor, out_tensor, out_len, in_len, 1, 1.0f, 0.0f, handle_p);
    
    err = cudaMemcpyToSymbol(const_bias, b, out_len * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    // printf("%d %s.\n", err, cudaGetErrorString(err));
    assert(err == cudaSuccess);
    dim3 gDim_bias(10, 1);
    dim3 bDim_bias(ceil(out_len / 10.0), 1);
    add_bias<<<gDim_bias, bDim_bias>>>(out_tensor, out_len, 1);

    *out_tensor_p = out_tensor;
};