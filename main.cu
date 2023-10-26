#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"


#define CUDA_CHECK_RETURN( value ) {                            \
    cudaError_t err = value;                                    \
    if( err != cudaSuccess ) {                                  \
        fprintf( stderr, "Error %s at line %d in file %s\n",    \
                cudaGetErrorString(err), __LINE__, __FILE__ );  \
        exit( 1 );                                              \
    } }

#define BLOCK_SIZE 16 // 16 * 16 = 256 threads in single block 

__global__ void matMul(int* Avalues, int* Bvalues, int* Out, int Awidth, int Bwidth){
    int value = 0;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    for (int i=0; i < Awidth; ++i){
        value += Avalues[row * Awidth + i] * Bvalues[i * Bwidth + col];
    }
    Out[row * Bwidth + col] = value;
}

int main(){
    int width_a;
    int height_a;
    printf("Input matrix A width. Must be multiple of 16 (Block size).\n");
    scanf("%d", &width_a);
    printf("Input matrix  A height. Must be multiple of 16.\n");
    scanf("%d", &height_a);
    
    // Suppose that FirstMatrix width = SecondMatrix height and FirstMatrix height = SecondMatrix width
    int A_height = height_a;
    int A_width = width_a;
    int *A_values = (int *)malloc(sizeof(int) * A_width * A_height);

    int B_height = height_a;
    int B_width = width_a;
    int *B_values = (int *)malloc(sizeof(int) * B_width * B_height);

    int C_height = A_height;
    int C_width = B_width;
    int *C_values = (int *)malloc(sizeof(int) * C_width * C_height);

    int *d_A_values;
    int *d_B_values;
    int *d_C_values;

    // initialize matrices values 
    for (int i = 0; i < A_width * A_width; ++i){
        A_values[i] = i;
    }

    for (int i = 0; i < B_width * B_height; ++i){
        B_values[i] = i;
    }

    // Allocate on CUDA
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_A_values, sizeof(int) * A_height * A_width));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_B_values, sizeof(int) * B_height * B_width));
    CUDA_CHECK_RETURN(cudaMalloc((void **)&d_C_values, sizeof(int) * C_height * C_width));

    // Copy values to CUDA 
    cudaMemcpy(d_A_values, A_values, sizeof(int) * A_height * A_width, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_values, B_values, sizeof(int) * B_height * B_width, cudaMemcpyHostToDevice);

    
    // Grid and Block dimension setup
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(C_width / dimBlock.x, C_height / dimBlock.y);
    matMul<<<dimGrid, dimBlock>>>(d_A_values, d_B_values, d_C_values, A_width, B_width);  // Run Kernel

    // Copy values from device to host
    cudaMemcpy(C_values, d_C_values, sizeof(int) * C_width * C_height, cudaMemcpyDeviceToHost);

    // print values 
    for (int i = 0; i < C_width * C_height; ++i){
        printf("%d\n", C_values[i]);
    }
    return 0;
}

