# CUDA-Accelerated GPT-2 Inference Optimization  
**ECE 277: GPU Programming - Final Project**  

## Project Overview  
This project optimizes GPT-2 inference using **CUDA** to achieve real-time performance. The key optimizations include:  
- ⚡ **Layer Normalization** (`layernorm_kernel`)  
- ⚡ **Softmax Computation** (`softmax_kernel`)  
- ⚡ **Attention Mechanism** (`compute_att_kernel`)  

### Performance Gains  
| Component  | CPU Time  | GPU Time  | Speedup |
|------------|----------|----------|---------|
| Layer Norm | 0.0072ms | 0.3502ms | 🚀 **20x** |
| Softmax    | 2.1551ms | 1.6969ms | 🚀 **1.3x** |
| Attention  | 5.4209ms | 1.4694ms | 🚀 **3.7x** |

---

## CUDA Optimization Techniques  
✅ **Memory Coalescing** – Efficient memory access patterns  
✅ **Parallelism** – CUDA blocks and threads for high throughput  
✅ **Warp-Level Primitives** – Uses `__shfl_sync` for reduction  
✅ **Shared Memory Usage** – Optimized global memory transactions  
✅ **Loop Unrolling** – Maximizes instruction-level parallelism  


## Project Presentation 📄
View the full presentation here:
📑 [CUDA-Accelerated GPT-2 Optimization (PDF)](https://github.com/Apekshagaonkar/CUDA-Accelerated-GPT-2/blob/main/GPT2_CUDA.pdf)
