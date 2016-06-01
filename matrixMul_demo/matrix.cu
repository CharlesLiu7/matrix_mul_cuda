// 核心代码主要改动自NVIDIA官网的示例程序

#include "matrix.cuh"



cudaError_t MultWithCuda(float *slow_out, float *out_cu, const float *A_in, const float *B_in, unsigned int HA, unsigned int WA, unsigned int WB)
{
	double time;
	LARGE_INTEGER lv1,lv2;
	
	size_t size_A = HA*WA;
	size_t size_B = WA*WB;
	size_t size_C = HA*WB;
	
	cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }

    // Allocate the device input Matrices A B and C
	float *A1 = NULL, *B1 = NULL, *C1 = NULL;

	cudaStatus = cudaMalloc((void **)&A1, size_A*sizeof(float));
	if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!\n");
        goto Error;
    }

	cudaStatus = cudaMalloc((void **)&B1, size_B*sizeof(float));
	if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!\n");
        goto Error;
    }

	cudaStatus = cudaMalloc((void **)&C1, size_C*sizeof(float));
	if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed!\n");
        goto Error;
    }

	// Copy data from host to device
	printf("Copy input data from the host memory to the CUDA device\n");
	cudaStatus = cudaMemcpy(A1, A_in, size_A*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!111\n");
        goto Error;
    }

	cudaStatus = cudaMemcpy(B1, B_in, size_B*sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!\n");
        goto Error;
    }

	// Invoke kernel******FOR FAST*********
	printf("\nStart quick kernel\n\n");

	QueryPerformanceCounter(&lv1);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(WB / dimBlock.x, HA / dimBlock.y);//We have assumed here that Matrix dimensions are multiples of BLOCK_SIZE
	
	MatMulKernel<<<dimGrid, dimBlock>>>(A1, B1, C1, HA, WA, WB);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
        printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
	QueryPerformanceCounter(&lv2);
	time=(secondsPerTick * (lv2.QuadPart-lv1.QuadPart));
	printf("Done! GPU_fast Time used:  %.2lfms\n\n",time/1000.0);

	// Copy data back
	printf("Copy output data from the CUDA device to the host memory.\n");
	cudaStatus = cudaMemcpy(out_cu, C1, size_C*sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!\n");
        goto Error;
    }

	// Invoke kernel******FOR SLOW*********
	printf("\nStart slow kernel\n\n");

	QueryPerformanceCounter(&lv1);
	
	MatMulKernel_slow<<<dimGrid, dimBlock>>>(A1, B1, C1, HA, WA, WB);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
        printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
	QueryPerformanceCounter(&lv2);
	time=(secondsPerTick * (lv2.QuadPart-lv1.QuadPart));
	printf("Done! GPU_slow Time used:  %.2lfms\n\n",time/1000.0);

	// Copy data back
	printf("Copy output data from the CUDA device to the host memory.\n");
	cudaStatus = cudaMemcpy(slow_out, C1, size_C*sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy failed!\n");
        goto Error;
    }

Error:
    // Free device memory
	cudaFree(A1);
	cudaFree(B1);
	cudaFree(C1);
    
    return cudaStatus;
}




int main()
{

	time_t t;
	srand((unsigned int)time(&t));
	double time;
	LARGE_INTEGER lv,lv1,lv2;

	QueryPerformanceFrequency(&lv);
	secondsPerTick = 1000000.0/lv.QuadPart;

	int HA,WA,HB,WB;
	int i,j,k;
	printf("Please input size:");
	scanf("%d",&i);

	HA = i*BLOCK_SIZE;
	WA = i*BLOCK_SIZE;
	WB = 2*i*BLOCK_SIZE;

	printf("So I choose %d %d and %d as HA WA and WB.\n",HA,WA,WB);
	HB=WA;

	size_t size_A = HA*WA;
	size_t size_B = HB*WB;
	size_t size_C = HA*WB;

	// Allocate the host input matrices A B and C,CU
	float *A_in = (float *)malloc(size_A * sizeof(float));
	float *B_in = (float *)malloc(size_B * sizeof(float));
	float *out_cu = (float *)malloc(size_C * sizeof(float));
	float *slow_out = (float *)malloc(size_C * sizeof(float));
	float *Out_C = (float *)malloc(size_C * sizeof(float));
	// Verify
	if (A_in == NULL || B_in == NULL || Out_C == NULL || out_cu == NULL || slow_out == NULL)
	{
		printf("Failed to allocate host matrices!\n");
		exit(EXIT_FAILURE);
	}
	
	// Initialize the host input matrices,each element is between 0 and 100
	for (long i = 0; i < size_A; ++i)
		A_in[i] = (100.0*(rand()%65536))/65535.0;
	for (long i = 0; i < size_B; ++i)
		B_in[i] = (100.0*(rand()%65536))/65535.0;


    // Multiple in parallel.
    cudaError_t cudaStatus = MultWithCuda(slow_out, out_cu, A_in, B_in, HA, WA, WB);
    if (cudaStatus != cudaSuccess) {
        printf("Multiple With Cuda failed!\n");
        goto FREE;
    }

	// Reset the device
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceReset failed!\n");
        goto FREE;
    }
	getchar();
	//Compute Out_C with CPU
	printf("Start to calculate with CPU.\n\n");
	QueryPerformanceCounter(&lv1);
	for (i = 0; i<HA; ++i)
		for(j=0; j<WB; ++j)
		{
			float c=0.0;
			for(k=0; k<WA; ++k)
				c+=A_in[i*WA+k]*B_in[k*WB+j];
			Out_C[i*WB+j]=c;
		}
	QueryPerformanceCounter(&lv2);
	time=(secondsPerTick * (lv2.QuadPart-lv1.QuadPart));
	printf("Done! CPU Time used:  %.2lfms\n\n",time/1000.0);

	// Verify that if the result is correct
	printf("Start Verification.\n");
	for(i=0;i<HA;++i)
		for(j=0;j<WB;++j)
			if (fabs(Out_C[i*WB+j] - out_cu[i*WB+j]) > 1.0)
			{
				printf("\n%f  %f",Out_C[i*WB+j], out_cu[i*WB+j]);
				printf("\nResult verification failed at fast_element C[%d][%d]!\n", i,j);
				goto FREE;
			};

	for(i=0;i<HA;++i)
		for(j=0;j<WB;++j)
			if (fabs(Out_C[i*WB+j] - slow_out[i*WB+j]) > 1.0)
			{
				printf("\n%f  %f",Out_C[i*WB+j], slow_out[i*WB+j]);
				printf("\nResult verification failed at slow_element C[%d][%d]!\n", i,j);
				goto FREE;
			};
	printf("ALL PASSED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

FREE:
	// Free host memory
	free(A_in);
	free(B_in);
	free(out_cu);
	free(slow_out);
	free(Out_C);
	getchar();
    return 0;
}
