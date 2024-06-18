#ifndef _HLBL_KERNEL_CUH
#define _HLBL_KERNEL_CUH

#ifdef __restrict__
#define _RESTR __restrict__
#else
#define _RESTR
#endif

#ifndef CUDA_BLOCK_SIZE
#  define CUDA_BLOCK_SIZE 256
#endif

// void ker_X_prepare_ev ( cuDoubleComplex* _RESTR out, const double* _RESTR in, const double *_RESTR kerv, const int N );

__global__ void ker_X_prepare_ev ( cuDoubleComplex* _RESTR out, const double* _RESTR in, const double *_RESTR kerv, const int N );

//__global__ void test_kernel ( double * _RESTR out, const double* _RESTR in, const double *_RESTR kerv, const int N );
__global__ void test_kernel ( cuDoubleComplex * _RESTR out, const double* _RESTR in, const double *_RESTR kerv, const int N );

#endif
