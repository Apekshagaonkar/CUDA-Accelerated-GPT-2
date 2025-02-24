#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
using namespace std;

void isequal(float* a, float* b, int n){
    float maxval = -INFINITY;
    for(int i = 0;i<n;i++){
        maxval = fmaxf(maxval, fmaxf(a[i], b[i]));
    }
    float eps = 1e-5;
    for(int i = 0;i<n;i++){
        if(fabs(a[i] - b[i]) > eps * (maxval + 1)){
            cout << "Mismatch at index " << i << " CPU: " << a[i] << " GPU: " << b[i] << endl;
        }
    }
    cout << "Results match " << endl;
    for (int i = 0; i < 4; i++) {
        cout << a[i] << " " << b[i] << endl;
    }
}

void layernorm(float* out, float* x, float* w, float* b, int C){
    float mean = 0;
    float var = 0;
    for(int i = 0;i<C;i++){
        mean += x[i];
    }
    mean /= C;
    for(int i = 0;i<C;i++){
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= C;
    float scale = 1.0 / sqrt(var + 1e-6);
    for(int i = 0;i<C;i++){
        out[i] = (x[i] - mean) * scale * w[i] + b[i];
    }
}

__global__
void layernorm_kernel(float* out, float* x, float* w, float* b, int C){
    int idx = threadIdx.x;
    float mean = 0;
    __shared__ float s_mean[1024];
    __shared__ float s_var[1024];
    s_mean[idx] = 0.0f;
    s_var[idx] = 0.0f;
    __syncthreads();

    for(int i = idx;i<C;i+=blockDim.x){
        s_mean[idx] += x[i];
    }
    __syncthreads();
    if(idx == 0){
        float m = 0;
        for(int i = 0;i<blockDim.x;i++){
            m += s_mean[i];
        }
        m /= C;
        s_mean[0] = m;
    }
    __syncthreads();
    mean = s_mean[0];

    for(int i = idx;i<C;i+=blockDim.x){
        float diff = x[i] - mean;
        s_var[idx] += diff * diff;
    }
    __syncthreads();
    if(idx == 0){
        float v = 0;
        for(int i = 0;i<blockDim.x;i++){
            v += s_var[i];
        }
        v /= C;
        s_var[0] = v;
    }
    __syncthreads();
    float var = s_var[0];
    float scale = 1.0 / sqrt(var + 1e-6);
    for(int i = idx;i<C;i+=blockDim.x){
        out[i] = (x[i] - mean) * scale * w[i] + b[i];
    }
}

void layernorm_gpu(float* out, float* x, float* w, float* b, int C){
    int numThreads = 256;
    int block = 1;
    layernorm_kernel<<<block,numThreads>>>(out,x,w,b,C);
}

int main(){

    int C = 768;
    float* x = (float*)malloc(C * sizeof(float));
    float* w = (float*)malloc(C * sizeof(float));
    float* b = (float*)malloc(C * sizeof(float));
    float* out = (float*)malloc(C * sizeof(float));

    float* x_gpu,*w_gpu,*b_gpu,*out_gpu;
    cudaMalloc((void**)&x_gpu, C * sizeof(float));
    cudaMalloc((void**)&w_gpu, C * sizeof(float));
    cudaMalloc((void**)&b_gpu, C * sizeof(float));
    cudaMalloc((void**)&out_gpu, C * sizeof(float));

    for (int i = 0; i < C; i++) {
        x[i] = i*0.92;
        w[i] = i;
        b[i] = i*0.5;
    }

    cudaMemcpy(x_gpu, x, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu, w, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, C * sizeof(float), cudaMemcpyHostToDevice);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    layernorm(out, x, w, b, C);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU Execution Time: " << cpu_duration.count() << " ms\n";

    // GPU Timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    layernorm_gpu(out_gpu, x_gpu, w_gpu, b_gpu, C);
    cudaEventRecord(stop_gpu);

    cudaEventSynchronize(stop_gpu);
    float gpu_duration = 0;
    cudaEventElapsedTime(&gpu_duration, start_gpu, stop_gpu);
    std::cout << "GPU Execution Time: " << gpu_duration << " ms\n";

    float* check = (float*)malloc(C * sizeof(float));

    cudaMemcpy(check, out_gpu, C * sizeof(float), cudaMemcpyDeviceToHost);

    isequal(out, check, C);
    
    return 0;
}