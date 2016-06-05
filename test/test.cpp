// CUDA runtime �� + CUBLAS �� 
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <time.h>
#include <iostream>
#include <iomanip>
#include <windows.h>

using namespace std;

// ������Ծ����ά��
int const L = 10*32;
int const M = 10*32;
int const N = 20*32;

int main()
{
	// �������ʱ��ı���
	double secondsPerTick;
	LARGE_INTEGER lv;
	QueryPerformanceFrequency(&lv);
	secondsPerTick = 1000000.0 / lv.QuadPart;
	double time;
	LARGE_INTEGER lv1, lv2;

	// ����״̬����
	cublasStatus_t status;

	// �� �ڴ� ��Ϊ��Ҫ����ľ��󿪱ٿռ�
	float *h_A = (float*)malloc(L*M*sizeof(float));
	float *h_B = (float*)malloc(N*M*sizeof(float));

	// �� �ڴ� ��Ϊ��Ҫ����������ľ��󿪱ٿռ�
	float *h_C = (float*)malloc(L*N*sizeof(float));

	// Ϊ����������Ԫ�ظ��� 0-100 ��Χ�ڵ������
	for (int i = 0; i<L*M; i++) {
		h_A[i] = (100.0*(rand() % 65536)) / 65535.0;
	}
	for (int i = 0; i<N*M; i++) {
		h_B[i] = (100.0*(rand() % 65536)) / 65535.0;
	}

	// ��ӡ�����Եľ���
	/*cout << "���� A :" << endl;
	for (int i = 0; i<L*M; i++){
		cout << h_A[i] << " ";
		if ((i + 1) % M == 0) cout << endl;
	}
	cout << endl;
	cout << "���� B :" << endl;
	for (int i = 0; i<N*M; i++){
		cout << h_B[i] << " ";
		if ((i + 1) % N == 0) cout << endl;
	}
	cout << endl;*/

	/*
	** GPU ����������
	*/

	// ��������ʼ�� CUBLAS �����
	cublasHandle_t handle;
	status = cublasCreate(&handle);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
			cout << "CUBLAS ����ʵ��������" << endl;
		}
		getchar();
		return EXIT_FAILURE;
	}

	float *d_A, *d_B, *d_C;
	// �� �Դ� ��Ϊ��Ҫ����ľ��󿪱ٿռ�
	cudaMalloc(
		(void**)&d_A,    // ָ�򿪱ٵĿռ��ָ��
		L*M * sizeof(float)    //����Ҫ���ٿռ���ֽ���
		);
	cudaMalloc(
		(void**)&d_B,
		N*M * sizeof(float)
		);

	// �� �Դ� ��Ϊ��Ҫ����������ľ��󿪱ٿռ�
	cudaMalloc(
		(void**)&d_C,
		L*N * sizeof(float)
		);

	// ���������ݴ��ݽ� �Դ� ���Ѿ����ٺ��˵Ŀռ�
	cublasSetVector(
		L*M,    // Ҫ�����Դ��Ԫ�ظ���
		sizeof(float),    // ÿ��Ԫ�ش�С
		h_A,    // ��������ʼ��ַ
		1,    // ����Ԫ��֮��Ĵ洢���
		d_A,    // GPU ����ʼ��ַ
		1    // ����Ԫ��֮��Ĵ洢���
		);
	cublasSetVector(
		N*M,
		sizeof(float),
		h_B,
		1,
		d_B,
		1
		);

	// ͬ������
	cudaThreadSynchronize();

	// ���ݽ�������˺����еĲ��������庬����ο������ֲᡣ
	float a = 1; float b = 0;

	// ���ʱ��
	QueryPerformanceCounter(&lv1);
	// ������ˡ��ú�����Ȼ���������������������
	cublasSgemm(
		handle,    // blas ����� 
		CUBLAS_OP_T,    // ���� A ���Բ���
		CUBLAS_OP_T,    // ���� B ���Բ���
		L,    // A, C ������ 
		N,    // B, C ������
		M,    // A �������� B ������
		&a,    // ����ʽ�� �� ֵ
		d_A,    // A ���Դ��еĵ�ַ
		M,    // lda
		d_B,    // B ���Դ��еĵ�ַ
		N,    // ldb
		&b,    // ����ʽ�� �� ֵ
		d_C,    // C ���Դ��еĵ�ַ(�������)
		N    // ldc
		);

	// ͬ������
	cudaThreadSynchronize();

	// ���ʱ��
	QueryPerformanceCounter(&lv2);
	time = (secondsPerTick * (lv2.QuadPart - lv1.QuadPart));
	cout << endl << "Done! cublas Sgemm Time used " << setprecision(2) << time / 1000.0 << "ms" << endl << endl;

	// �� �Դ� ��ȡ���������� �ڴ���ȥ
	cublasGetVector(
		L*N,    //  Ҫȡ��Ԫ�صĸ���
		sizeof(float),    // ÿ��Ԫ�ش�С
		d_C,    // GPU ����ʼ��ַ
		1,    // ����Ԫ��֮��Ĵ洢���
		h_C,    // ��������ʼ��ַ
		1    // ����Ԫ��֮��Ĵ洢���
		);

	// ��ӡ������
	/*cout << "��������ת�� ( (A*B)��ת�� )��" << endl;
	for (int i = 0; i<L*N; i++){
		cout << h_C[i] << " ";
		if ((i + 1) % N == 0) cout << endl;
	}*/

	// �����ʹ�ù����ڴ�
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// �ͷ� CUBLAS �����
	cublasDestroy(handle);

	getchar();

	return 0;
}