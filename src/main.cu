#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

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
    printf("CUDA Vector Calculation Program\n");
    
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
    cudaMalloc((void **)&d_a, N * sizeof(float));
    cudaMalloc((void **)&d_b, N * sizeof(float));
    cudaMalloc((void **)&d_c, N * sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print some results
    printf("First 5 results:\n");
    for (int i = 0; i < 5; i++) {
        printf("c[%d] = %.2f (a[%d]=%.2f, b[%d]=%.2f)\n", 
               i, h_c[i], i, h_a[i], i, h_b[i]);
    }
    
    // Synchronize to ensure kernel finishes
    cudaDeviceSynchronize();
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    printf("Done!\n");
    
    return 0;
}
