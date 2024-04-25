#ifndef _HLBL_LM_CUDA_H
#define _HLBL_LM_CUDA_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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



int hlbl_lm_reduce ( void * const h_p, void * const h_v, void * const h_s, const int nv, const int nx, const int ns );

#endif
