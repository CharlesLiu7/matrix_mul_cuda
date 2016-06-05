// CUDA runtime 库 + CUBLAS 库 
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <time.h>
#include <iostream>
#include <iomanip>
#include <windows.h>

using namespace std;

// 定义测试矩阵的维度
int const L = 10*32;
int const M = 10*32;
int const N = 20*32;

int main()
{
	// 定义测量时间的变量
	double secondsPerTick;
	LARGE_INTEGER lv;
	QueryPerformanceFrequency(&lv);
	secondsPerTick = 1000000.0 / lv.QuadPart;
	double time;
	LARGE_INTEGER lv1, lv2;

	// 定义状态变量
	cublasStatus_t status;

	// 在 内存 中为将要计算的矩阵开辟空间
	float *h_A = (float*)malloc(L*M*sizeof(float));
	float *h_B = (float*)malloc(N*M*sizeof(float));

	// 在 内存 中为将要存放运算结果的矩阵开辟空间
	float *h_C = (float*)malloc(L*N*sizeof(float));

	// 为待运算矩阵的元素赋予 0-100 范围内的随机数
	for (int i = 0; i<L*M; i++) {
		h_A[i] = (100.0*(rand() % 65536)) / 65535.0;
	}
	for (int i = 0; i<N*M; i++) {
		h_B[i] = (100.0*(rand() % 65536)) / 65535.0;
	}

	// 打印待测试的矩阵
	/*cout << "矩阵 A :" << endl;
	for (int i = 0; i<L*M; i++){
		cout << h_A[i] << " ";
		if ((i + 1) % M == 0) cout << endl;
	}
	cout << endl;
	cout << "矩阵 B :" << endl;
	for (int i = 0; i<N*M; i++){
		cout << h_B[i] << " ";
		if ((i + 1) % N == 0) cout << endl;
	}
	cout << endl;*/

	/*
	** GPU 计算矩阵相乘
	*/

	// 创建并初始化 CUBLAS 库对象
	cublasHandle_t handle;
	status = cublasCreate(&handle);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
			cout << "CUBLAS 对象实例化出错" << endl;
		}
		getchar();
		return EXIT_FAILURE;
	}

	float *d_A, *d_B, *d_C;
	// 在 显存 中为将要计算的矩阵开辟空间
	cudaMalloc(
		(void**)&d_A,    // 指向开辟的空间的指针
		L*M * sizeof(float)    //　需要开辟空间的字节数
		);
	cudaMalloc(
		(void**)&d_B,
		N*M * sizeof(float)
		);

	// 在 显存 中为将要存放运算结果的矩阵开辟空间
	cudaMalloc(
		(void**)&d_C,
		L*N * sizeof(float)
		);

	// 将矩阵数据传递进 显存 中已经开辟好了的空间
	cublasSetVector(
		L*M,    // 要存入显存的元素个数
		sizeof(float),    // 每个元素大小
		h_A,    // 主机端起始地址
		1,    // 连续元素之间的存储间隔
		d_A,    // GPU 端起始地址
		1    // 连续元素之间的存储间隔
		);
	cublasSetVector(
		N*M,
		sizeof(float),
		h_B,
		1,
		d_B,
		1
		);

	// 同步函数
	cudaThreadSynchronize();

	// 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
	float a = 1; float b = 0;

	// 输出时间
	QueryPerformanceCounter(&lv1);
	// 矩阵相乘。该函数必然将数组解析成列优先数组
	cublasSgemm(
		handle,    // blas 库对象 
		CUBLAS_OP_T,    // 矩阵 A 属性参数
		CUBLAS_OP_T,    // 矩阵 B 属性参数
		L,    // A, C 的行数 
		N,    // B, C 的列数
		M,    // A 的列数和 B 的行数
		&a,    // 运算式的 α 值
		d_A,    // A 在显存中的地址
		M,    // lda
		d_B,    // B 在显存中的地址
		N,    // ldb
		&b,    // 运算式的 β 值
		d_C,    // C 在显存中的地址(结果矩阵)
		N    // ldc
		);

	// 同步函数
	cudaThreadSynchronize();

	// 输出时间
	QueryPerformanceCounter(&lv2);
	time = (secondsPerTick * (lv2.QuadPart - lv1.QuadPart));
	cout << endl << "Done! cublas Sgemm Time used " << setprecision(2) << time / 1000.0 << "ms" << endl << endl;

	// 从 显存 中取出运算结果至 内存中去
	cublasGetVector(
		L*N,    //  要取出元素的个数
		sizeof(float),    // 每个元素大小
		d_C,    // GPU 端起始地址
		1,    // 连续元素之间的存储间隔
		h_C,    // 主机端起始地址
		1    // 连续元素之间的存储间隔
		);

	// 打印运算结果
	/*cout << "计算结果的转置 ( (A*B)的转置 )：" << endl;
	for (int i = 0; i<L*N; i++){
		cout << h_C[i] << " ";
		if ((i + 1) % N == 0) cout << endl;
	}*/

	// 清理掉使用过的内存
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// 释放 CUBLAS 库对象
	cublasDestroy(handle);

	getchar();

	return 0;
}