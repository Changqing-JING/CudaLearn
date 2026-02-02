#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return 1; \
        } \
    } while(0)

// CUDA kernel that performs calculations on arrays
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Perform addition
        c[idx] = a[idx] + b[idx];
        
        // Perform some additional calculations
        float temp = a[idx] * b[idx];
        c[idx] = c[idx] + temp * 0.5f;
        
        // Power calculation
        c[idx] = c[idx] * c[idx];
    }
}

int main() {
    
    // Print CUDA runtime version (compiled against)
    int runtimeVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    
    // Print driver version (installed on system)
    int driverVersion = 0;
    cudaDriverGetVersion(&driverVersion);


    
    // Check for CUDA devices
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found!\n");
        return 1;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using device: %s\n", prop.name);
    
    // Allocate host memory
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c = (float *)malloc(N * sizeof(float));
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_c, N * sizeof(float)));
    
    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    printf("Launching kernel with %d blocks, %d threads per block\n", blocks, threads_per_block);
    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print some results
    printf("First 5 results:\n");
    for (int i = 0; i < 5; i++) {
        printf("c[%d] = %.2f (a[%d]=%.2f, b[%d]=%.2f)\n", 
               i, h_c[i], i, h_a[i], i, h_b[i]);
    }
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
