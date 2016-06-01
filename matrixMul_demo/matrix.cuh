#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <time.h>
#include <windows.h>


#define BLOCK_SIZE 32

double secondsPerTick;

__global__ void MatMulKernel_slow(float *A, float *B, float *C, int H_A, int W_A, int W_B)
{
	// Each thread computes one element of C 
	// by accumulating results into Cvalue 
	float Cvalue = 0.0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e != W_A; ++e)
		Cvalue += A[row * W_A + e] * B[e * W_B + col];
	C[row * W_B + col] = Cvalue;
}

cudaError_t MultWithCuda(float *c_slow, float *c, const float *a, const float *b, unsigned int H_A, unsigned int W_A, unsigned int W_B);

// Matrix multiplication kernel
__global__ void MatMulKernel(float *A, float *B, float *C, int H_A, int W_A, int W_B)
{
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	// Thread row and column within sub_matrix
	int row = threadIdx.y;
	int col = threadIdx.x;

	// Each thread computes one element of sub_matrix of C by accumulating results into Cvalue
	float Cvalue = 0;
	int count = W_A/BLOCK_SIZE;

	// Loop over all the sub-matrices of A and B that are required to compute Csub
	// Multiply each pair of sub-matrices together and accumulate the results
	for (int m=0; m<(W_A/BLOCK_SIZE); ++m)
	//W_A is divided by BLOCKI_SIZE here,so we don't use this form: (W_A-1)/BLOCK_SIZE+1
	{
		// Shared memory used to store Asub and Bsub respectively
		__shared__ float sub_A[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float sub_B[BLOCK_SIZE][BLOCK_SIZE];

		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		sub_A[row][col] = A[W_A*(BLOCK_SIZE*blockRow+row)+BLOCK_SIZE*m+col];
		sub_B[row][col] = B[W_B*(BLOCK_SIZE*m+row)+BLOCK_SIZE*blockCol+col];

		//  This synchronize is to make sure the sub-matrices are loaded before starting the computation
		__syncthreads();

		// Multiply Asub and Bsub together
		for (int e = 0; e != BLOCK_SIZE; ++e)
			Cvalue += sub_A[row][e] * sub_B[e][col];

		// Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write Csub to device memory
	// Each thread writes one element, too
	C[W_B*(BLOCK_SIZE*blockRow+row)+BLOCK_SIZE*blockCol+col] = Cvalue;
}