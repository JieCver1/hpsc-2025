#include <cstdio>
#include <cstdlib>
#include <vector>

// Set the costants for the kernel
const int N = 50; // Number of elements to sort
const int RANGE = 5; // range of keys

// Kernel 1: Initialize the array with zeros
__global__ void initBucketZero(int bucket[]){
        int i = threadIdx.x;
        if (i<RANGE)
                bucket[i] = 0;
}

// Kernel 2: Increment the bucket for each key
__global__ void incrementBucketKey(int bucket[], int key[], int n){
        int i = threadIdx.x + blockDim.x*blockIdx.x; // calculate the global index
        if(i<n){
                atomicAdd(&bucket[key[i]],1);
        }
}

 // Kernel 3: write the sorted values to the key array
__global__ void sortBucket(int bucket[], int key[], int offsets[]){
        int i = threadIdx.x;
        if(i<RANGE){
                int begin = offsets[i];
                int count = bucket[i];
                for (int j =0; j <count; j++){
                        key[begin+j] = i;
                }
        }
}


int main() {
    int *key, *bucket, *offset;

    // Allocate memory using cudaMallocManaged then this memory is accessible from both the CPU and GPU
    cudaMallocManaged(&key, N * sizeof(int));
    cudaMallocManaged(&bucket, RANGE * sizeof(int));
    cudaMallocManaged(&offset, RANGE * sizeof(int));

    for (int i = 0; i < N; ++i) {
        key[i] = rand() % RANGE;
        printf("%d ", key[i]);
    }
    printf("\n");

    // Launch the kernel to initialize the bucket
    initBucketZero<<<1, RANGE>>>(bucket);
    cudaDeviceSynchronize();

    // Launch the kernel to increment the bucket for each key
    incrementBucketKey<<<(N + 255)/256, 256>>>(bucket, key, N); // as (N+M-1)/M, M
    cudaDeviceSynchronize();

    // Compute offset array on the CPU
    offset[0] = 0;
    for (int i = 1; i < RANGE; ++i) {
        offset[i] = offset[i - 1] + bucket[i - 1];
    }
           
    // Launch the kernel to write the sorted values back to the key array
    sortBucket<<<1, RANGE>>>(bucket, key, offset);
    cudaDeviceSynchronize();

    /*
    std::vector<int> bucket(range);
    for (int i=0; i<range; i++) {
        bucket[i] = 0;
    }
    for (int i=0; i<n; i++) {
        bucket[key[i]]++;
    }
    for (int i=0, j=0; i<range; i++) {
        for (; bucket[i]>0; bucket[i]--) {
        key[j++] = i;
        }
    }
    */

    for (int i=0; i<N; i++) {
        printf("%d ",key[i]);
    }
    printf("\n");

    // Free the allocated memory
    cudaFree(key);
    cudaFree(bucket);
    cudaFree(offset);
    
    return 0; 
}
