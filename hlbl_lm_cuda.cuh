#ifndef _HLBL_LM_CUDA_H
#define _HLBL_LM_CUDA_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cublas_api.h>

using cuda_data_type = cuDoubleComplex;

namespace cvc {

/****************************************************
 * hlbl_lm_cuda
 ****************************************************/

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// CUDA API error checking for malloc
#define CUDA_CHECK_MALLOC(err)                                                                     \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA MALLOC error %d at %s:%d\n", err_, __FILE__, __LINE__);              \
            throw std::runtime_error("CUDA MALLOC error");                                         \
        }                                                                                          \
    } while (0)

// CUDA API error checking for free
#define CUDA_CHECK_FREE(err)                                                                       \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA FREE error %d at %s:%d\n", err_, __FILE__, __LINE__);                \
            throw std::runtime_error("CUDA FREE error");                                           \
        }                                                                                          \
    } while (0)


// CUDA API error checking for memcpy
#define CUDA_CHECK_MEMCPY(err)                                                                     \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA MEMCPY error %d at %s:%d\n", err_, __FILE__, __LINE__);              \
            throw std::runtime_error("CUDA MEMCPY error");                                         \
        }                                                                                          \
    } while (0)


#if 0
int hlbl_lm_reduce ( void * const h_p, void * const h_v, void * const h_s, const int nv, const int nx, const int ns );
#endif  // end of if 0

int hlbl_lm_reduce ( cudaStream_t stream, cublasHandle_t cublasH, double _Complex * const h_p,
    cuda_data_type * const d_v, cuda_data_type * const d_s, const int nv, const int nx, const int ns );

int project_v_dag_g_v ( cudaStream_t stream, cublasHandle_t cublasH, double _Complex * const h_p,
    const double * d_v, const double * kervx, const int nv, const int nx );

// int apply_kernel ( cudaStream_t stream, cublasHandle_t cublasH, double * d_out, const double * d_in, const double * kervx,  const int nx );
int apply_kernel ( cudaStream_t stream, cublasHandle_t cublasH, cuDoubleComplex * d_out, const double * d_in, const double * kervx,  const int nx );

}

#endif
