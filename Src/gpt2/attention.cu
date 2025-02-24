#include <cuda_runtime.h>
#include <iostream>
#include <cublas_v2.h>
#include <float.h>
#include <assert.h>
#include <chrono>
using namespace std;

void rand_init(float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = (float)rand() / RAND_MAX;
    }
}

void isequal(float* a, float* b, int n) {
    float maxval = -INFINITY;
    for (int i = 0; i < n; i++) {
        maxval = fmaxf(maxval, fmaxf(a[i], b[i]));
    }
    float eps = 1e-5;

    for (int i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > eps * (maxval + 1)) {
            cout << "Mismatches" << endl;
            for (int j = i; j < min(n, i + 10); j++) {
                cout << a[j] << " " << b[j] << endl;
            }
            return;
        }
    }
    cout << "Results match " << endl;
    for (int i = 0; i < 4; i++) {
        cout << a[i] << " " << b[i] << endl;
    }
}

void softmax(float* x, int N) {
    float max = x[0];
    for (int i = 1; i < N; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }
    float sum = 0;
    for (int i = 0; i < N; i++) {
        x[i] = exp(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < N; i++) {
        x[i] /= sum;
    }
}

void attention(float* out, float* att, float* qkv, float* key_cache, float* value_cache, int l, int pos, int C, int NH, int T) {

    int head_size = C / NH;

    float* q = qkv;
    memcpy(key_cache + l * C * T + pos * C, qkv + C, C * sizeof(float));
    memcpy(value_cache + l * C * T + pos * C, qkv + 2 * C, C * sizeof(float));

    float scale = 1.0 / sqrt(head_size);

    float* k = key_cache + l * C * T;
    float* v = value_cache + l * C * T;

    int h;
#pragma omp parallel for private(h)
    for (h = 0; h < NH; h++) {

        float* qh = q + h * head_size;
        float* atth = att + h * T;

        for (int t = 0; t < T; t++) {
            float* kh = k + t * C + h * head_size;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += qh[i] * kh[i];
            }
            score *= scale;
            atth[t] = score;
        }
        for (int t = pos + 1; t < T; t++) {
            atth[t] = -FLT_MAX;
        }

        softmax(atth, T);

        float* outh = out + h * head_size;
        memset(outh, 0, head_size * sizeof(float));
        for (int t = 0; t < T; t++) {
            float* vh = v + t * C + h * head_size;
            float score = atth[t];
            for (int i = 0; i < head_size; i++) {
                outh[i] += score * vh[i];
            }
        }
    }
}

/*
Plan :
qkv is shape of 3 * C
fill qkv_k into key_cache
fill qkv_v into value_cache
now the problem is caches are of size L * T * NH * HS
but we need caches to be of size L * NH * T * HS
so we directly copy permuted qkv to caches for every position for every layer
this way caches remain in shape L * NH * T * HS
*/

__global__
void fill_cache(float* key_cache, float* value_cache, float* arr, int l, int pos, int C, int NH, int T) {

    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread index is within bounds of the input size
    if (idx < C) {

        // Calculate the head size for each attention head
        int head_size = C / NH;

        // Determine the head number (nh) and element index (h) within the head
        int nh = idx / head_size; // Head number (which head this element belongs to)
        int h = idx % head_size;  // Index within the head (head size index)

        // Pointers to the key and value caches for the current layer
        float* key_cache_l = key_cache + l * T * NH * head_size;
        float* value_cache_l = value_cache + l * T * NH * head_size;

        // Map the source position in the `arr` array (current position in input tensors)
        // to the destination position in the cache arrays (reorganized layout for efficiency)

        // Source index in the input array (`arr`):
        // Each head's data is stored contiguously in the input tensor.
        // `from` refers to the element in the head before reorganization.
        int from = nh * head_size + h;

        // Destination index in the cache:
        // Cache format is [NH, T, HS], so we need to map the data accordingly.
        int to = nh * T * head_size + pos * head_size + h;

        // Copy the corresponding key and value elements from the input array to the cache.
        // - Keys are stored in `arr[from + C]` (shifted by `C` to skip over queries).
        // - Values are stored in `arr[from + 2*C]` (shifted by `2*C` to skip over queries and keys).
        key_cache_l[to] = arr[from + C];        // Store key in the cache
        value_cache_l[to] = arr[from + 2 * C];  // Store value in the cache
    }
}



__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__
void softmax_kernel3(float* x, int T, int NH, int pos, float scale) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;

    int warpsPerBlock = blockDim.x / 32;

    extern __shared__ float shared[];
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    float* x_h = x + idx * T;
    float maxval = -FLT_MAX;
    for (int i = tid; i < pos; i += blockDim.x) {
        x_h[i] *= scale;
        maxval = fmaxf(maxval, x_h[i]);
    }
    maxval = warpReduceMax(maxval);
    if (laneId == 0) {
        maxvals[warpId] = maxval;
    }
    __syncthreads();
    if (tid == 0) {
        float val = -FLT_MAX;
        for (int i = 0; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();
    float offset = maxvals[0];

    for (int i = tid; i < pos; i += blockDim.x) {
        x_h[i] = expf(x_h[i] - offset);
    }
    //sum
    float sumval = 0.0f;
    for (int i = tid; i < pos; i += blockDim.x) {
        sumval += x_h[i];
    }
    sumval = warpReduceSum(sumval);
    if (laneId == 0) {
        sumvals[warpId] = sumval;
    }
    __syncthreads();
    if (tid == 0) {
        float val = 0;
        for (int i = 0; i < warpsPerBlock; i++) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    float sum = sumvals[0];
    for (int i = tid; i < T; i += blockDim.x) {
        x_h[i] = (i < pos) ? x_h[i] / sum : 0;
    }
}

__global__
void compute_att_kernel(
    const float* __restrict__ q,     // Query matrix (shape: [NH, HS])
    const float* __restrict__ k,     // Key matrix (shape: [NH, T, HS])
    float* __restrict__ att,         // Output attention scores (shape: [NH, T])
    int NH,                          // Number of attention heads
    int T,                           // Sequence length
    int HS,                          // Head size (dimensionality of each attention head)
    int pos                          // Current position in the sequence
) {
    // Each block corresponds to one attention head
    int h = blockIdx.x; // Head index (one block per head)

    // Each thread computes one element along the time dimension for the current head
    int t = threadIdx.x; // Time index (one thread per time step)

    // Ensure that the current thread operates only on valid indices
    if (h < NH && t <= pos) {

        // Initialize the dot product sum for the current position
        float sum = 0.0f;

        // Calculate the offsets for accessing `q` and `k` matrices
        int q_offset = h * HS;             // Query offset for the current head
        int k_offset = h * T * HS + t * HS; // Key offset for the current head and time step

        // Compute the dot product between the query vector and the key vector
#pragma unroll 4 // Unroll the loop for better performance
        for (int i = 0; i < HS; i++) {
            sum += q[q_offset + i] * k[k_offset + i];
        }

        // Store the computed attention score in the output matrix
        // `att` is indexed as [head, time]
        att[h * T + t] = sum;
    }
}
__global__
void compute_out_kernel(
    const float* __restrict__ att,  // Attention scores (shape: [NH, T])
    const float* __restrict__ v,    // Value matrix (shape: [NH, T, HS])
    float* __restrict__ out,        // Output matrix (shape: [NH, HS])
    int NH,                         // Number of attention heads
    int T,                          // Sequence length
    int HS,                         // Head size (dimensionality per attention head)
    int pos                         // Current position in the sequence
) {
    // Identify the current attention head (`h`) and head size index (`i`)
    int h = blockIdx.x;  // Each block corresponds to a single attention head
    int i = threadIdx.x; // Each thread computes one element in the head size dimension (HS)

    // Ensure the current thread operates within valid bounds
    if (h < NH && i < HS) {
        // Initialize a variable to accumulate the weighted sum of values
        float sum = 0.0f;

        // Calculate offsets for accessing value matrix (`v`) and attention scores (`att`)
        int v_head_offset = h * T * HS;  // Offset to the start of the current head in `v`
        int att_head_offset = h * T;     // Offset to the start of the current head in `att`

        // Compute the weighted sum of values using attention scores
        // For each time step `t`, multiply the attention score by the corresponding value vector
#pragma unroll 4
        for (int t = 0; t <= pos; t++) {
            sum += att[att_head_offset + t] * v[v_head_offset + t * HS + i];
        }

        // Store the computed value in the output matrix
        // Output is indexed as [NH, HS] (flattened as 1D array)
        out[h * HS + i] = sum;
    }
}


void attention_gpu(float* out, float* att, float* qkv, float* key_cache, float* value_cache, int l, int pos, int C, int NH, int T) {

    // Compute head size: Each head gets an equal portion of the total dimension C
    int head_size = C / NH;

    // Determine the number of threads per block for `fill_cache` kernel
    int numThreads = 32; // Small value to handle elements of `qkv`
    int blocks = (C + numThreads - 1) / numThreads; // Number of blocks needed for `fill_cache`

    // Step 1: Copy keys and values to their respective caches
    fill_cache << <blocks, numThreads >> > (key_cache, value_cache, qkv, l, pos, C, NH, T);

    // Scale factor for the attention scores: 1 / sqrt(head_size) for numerical stability
    float scale = 1.0f / sqrtf((float)head_size);

    // Pointer setup for query (q), key (k), and value (v) tensors
    float* q = qkv;                          // Query: First C elements of qkv
    float* k = key_cache + l * C * T;        // Keys: Points to the start of key cache for layer `l`
    float* v = value_cache + l * C * T;      // Values: Points to the start of value cache for layer `l`

    // Step 2: Compute att = q * K^T (attention scores for each head)
    // Attention tensor `att` shape: [NH, T]
    // Query tensor `q` shape: [NH, HS]
    // Key tensor `k` shape: [NH, T, HS]
    {
        int threads = pos + 1; // Threads needed to compute scores up to `pos`
        if (threads > 1024) {
            // If the sequence length T is large, you may need to split work into smaller chunks
            threads = 1024; // Limit threads per block to 1024 (hardware maximum)
        }
        // Launch kernel to compute attention scores
        compute_att_kernel << <NH, threads >> > (q, k, att, NH, T, head_size, pos);
    }

    // Step 3: Apply the softmax operation to normalize attention scores
    // Softmax is performed in place on `att`, operating on `NH` heads and the time dimension `T`
    size_t memory = 2 * 1024 * sizeof(float) / 32; // Shared memory size for softmax kernel
    softmax_kernel3 << <NH, 1024, memory >> > (att, T, NH, pos + 1, scale);

    // Step 4: Compute output = att * V (weighted sum of values)
    // Output tensor `out` shape: [NH, HS]
    // Attention tensor `att` shape: [NH, T]
    // Value tensor `v` shape: [NH, T, HS]
    {
        int threads = head_size; // Threads needed to compute output for `head_size`
        if (threads > 1024) {
            // For very large head sizes, consider splitting work across multiple launches
            threads = 1024; // Limit threads per block to 1024
        }
        // Launch kernel to compute the final output
        compute_out_kernel << <NH, threads >> > (att, v, out, NH, T, head_size, pos);
    }
}



int main() {

    int C = 1024;
    int NH = 16;
    int head_size = C / NH;
    int T = 1200;
    int L = 8;


    float* out, * att, * qkv, * key_cache, * value_cache;
    float* d_out, * d_att, * d_qkv, * d_key_cache, * d_value_cache;

    if (true) {
        out = (float*)malloc(NH * head_size * sizeof(float));
        att = (float*)malloc(NH * T * sizeof(float));
        qkv = (float*)malloc(3 * C * sizeof(float));
        key_cache = (float*)malloc(L * NH * T * head_size * sizeof(float));
        value_cache = (float*)malloc(L * NH * T * head_size * sizeof(float));

        rand_init(qkv, 3 * C);
        memset(key_cache, 0, L * NH * T * head_size * sizeof(float));
        memset(value_cache, 0, L * NH * T * head_size * sizeof(float));

        cudaMalloc(&d_out, NH * head_size * sizeof(float));
        cudaMalloc(&d_att, NH * T * sizeof(float));
        cudaMalloc(&d_qkv, 3 * C * sizeof(float));
        cudaMalloc(&d_key_cache, L * NH * T * head_size * sizeof(float));
        cudaMalloc(&d_value_cache, L * NH * T * head_size * sizeof(float));

        cudaMemcpy(d_qkv, qkv, 3 * C * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_key_cache, key_cache, L * NH * T * head_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_value_cache, value_cache, L * NH * T * head_size * sizeof(float), cudaMemcpyHostToDevice);
    }


    int l = 7;
    int pos = 1023;

    auto start_cpu = std::chrono::high_resolution_clock::now();
    attention(out, att, qkv, key_cache, value_cache, l, pos, C, NH, T);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU Execution Time: " << cpu_duration.count() << " ms\n";

    // GPU Timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    attention_gpu(d_out, d_att, d_qkv, d_key_cache, d_value_cache, l, pos, C, NH, T);
    cudaEventRecord(stop_gpu);

    cudaEventSynchronize(stop_gpu);
    float gpu_duration = 0;
    cudaEventElapsedTime(&gpu_duration, start_gpu, stop_gpu);
    std::cout << "GPU Execution Time: " << gpu_duration << " ms\n";


    float* check_att = (float*)malloc(NH * T * sizeof(float));
    cudaMemcpy(check_att, d_att, NH * T * sizeof(float), cudaMemcpyDeviceToHost);
    isequal(att, check_att, NH * T);

    float* check_out = (float*)malloc(NH * head_size * sizeof(float));
    cudaMemcpy(check_out, d_out, NH * head_size * sizeof(float), cudaMemcpyDeviceToHost);
    isequal(out, check_out, NH * head_size);

    return 0;
}