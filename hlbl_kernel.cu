#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>

#include "hlbl_kernel.cuh"


__device__ int d_coord_map ( int xi, int Li ) 
{
  return (xi >= Li / 2) ? (xi - Li) : xi;
}  // end of d_coord_map

__device__ int d_coord_map_zerohalf(int xi, int Li) 
{
  return (xi > Li / 2) ? xi - Li : ( (xi < Li / 2) ? xi : 0 );
}  // end of d_coord_map_zerohalf



__device__ __constant__ int d_gamma_permutation[16][24] = {
  {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
  {19, 18, 21, 20, 23, 22, 13, 12, 15, 14, 17, 16, 7, 6, 9, 8, 11, 10, 1, 0, 3, 2, 5, 4},
  {18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5},
  {13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10},
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
  {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
  {19, 18, 21, 20, 23, 22, 13, 12, 15, 14, 17, 16, 7, 6, 9, 8, 11, 10, 1, 0, 3, 2, 5, 4},
  {18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5},
  {13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10},
  {7, 6, 9, 8, 11, 10, 1, 0, 3, 2, 5, 4, 19, 18, 21, 20, 23, 22, 13, 12, 15, 14, 17, 16},
  {6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17},
  {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22},
  {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22},
  {6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17},
  {7, 6, 9, 8, 11, 10, 1, 0, 3, 2, 5, 4, 19, 18, 21, 20, 23, 22, 13, 12, 15, 14, 17, 16}
};
__device__ __constant__ int d_gamma_sign[16][24] = {
  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1},
  {-1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1},
  {+1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1},
  {+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1},
  {+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {-1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1},
  {+1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1},
  {-1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1},
  {+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1},
  {-1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1},
  {+1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1},
  {-1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1},
  {-1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1},
  {-1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1}
};

__device__ inline void _d_fv_pl_eq_fv_ti_re(double* _RESTR out, const double* _RESTR in, double const re)
{
#pragma unroll
  for (int i = 0; i < 24; ++i) 
  {
    out[i] = in[i] * re;
  }
}

__device__ inline void _d_fv_eq_fv_pl_fv_ti_re(double* _RESTR out, const double* _RESTR ina, const double* _RESTR inb, double const re)
{
#pragma unroll
  for (int i = 0; i < 24; ++i) 
  {
    out[i] = ina[i] + inb[i] * re;
  }
}

__device__ inline void _d_fv_eq_gamma_ti_fv(double* _RESTR out, int gamma_index, const double* _RESTR in) 
{
#pragma unroll
  for (int i = 0; i < 24; ++i) 
  {
    out[i] = in[ d_gamma_permutation[gamma_index][i]] * d_gamma_sign[gamma_index][i];
  }
}

__device__ inline void _d_fv_ti_eq_g5(double* in_out) 
{
#pragma unroll
  for (int i = 12; i < 24; ++i) 
  {
    in_out[i] *= -1;
  }
} 

__device__ inline void _d_fv_eq_zero(double* in_out) 
{
#pragma unroll
  for (int i = 0; i < 24; ++i) 
  {
    in_out[i] = 0.0;
  }
} 

__device__ inline void _d_fv_eq_fv(double* out, const double * in) 
{
#pragma unroll
  for (int i = 0; i < 24; i++ ) 
  {
    out[i] = in[i];
  }
} 

__device__ inline void _d_fv_ti_eq_re(double* _RESTR in_out, double const re)
{
#pragma unroll
  for (int i = 0; i < 24; ++i)
  {
    in_out[i] *= re;
  }
}


/***********************************************************
 * out has indexing

 * out[ 96 x spin-component + k ]
 * with k = 0,..., 96-1 the open index combinations of the kernel
 * spin-component = 0,..., 3
 *
 * kerv field must be pre-computed with indices
 *
 * kerv[4 x (sigma-rho, nu, lambda ) + mu ],
 * so summation index mu is innermost
 ***********************************************************/

__device__ inline void _d_X_prepare_ev ( cuDoubleComplex * _RESTR out, const double* _RESTR in, const double *_RESTR kerv) 
{
  // not good, hard-coded cvc -> ukqcd conversion
  int    const gamma_map_id[4]   = {   6,   10,   11,  12 };
  double const gamma_map_sign[4] = {  -1.,  +1.,  +1., +1. };

  double sp[4][24];

  _d_fv_eq_gamma_ti_fv( sp[0], gamma_map_id[0], in );
  _d_fv_ti_eq_re( sp[0], gamma_map_sign[0] );
  _d_fv_eq_gamma_ti_fv( sp[1], gamma_map_id[1], in );
  _d_fv_ti_eq_re( sp[1], gamma_map_sign[1] );
  _d_fv_eq_gamma_ti_fv( sp[2], gamma_map_id[2], in );
  _d_fv_ti_eq_re( sp[2], gamma_map_sign[2] );
  _d_fv_eq_gamma_ti_fv( sp[3], gamma_map_id[3], in );
  _d_fv_ti_eq_re( sp[3], gamma_map_sign[3] );


#pragma unroll
  for ( int i = 0; i < 12; i++ )
  {
#pragma unroll
    for ( int k = 0; k < 96; k++ )
    {
      // sum on mu
      out[96*i+k].x = sp[0][2*i+0] * kerv[4*k+0] + sp[1][2*i+0] * kerv[4*k+1] + sp[2][2*i+0] * kerv[4*k+2] + sp[3][2*i+0] * kerv[4*k+3];
      out[96*i+k].y = sp[0][2*i+1] * kerv[4*k+0] + sp[1][2*i+1] * kerv[4*k+1] + sp[2][2*i+1] * kerv[4*k+2] + sp[3][2*i+1] * kerv[4*k+3];
      
      // with complex conjugation
      //out[96*i+k].x =    sp[0][2*i+0] * kerv[4*k+0] + sp[1][2*i+0] * kerv[4*k+1] + sp[2][2*i+0] * kerv[4*k+2] + sp[3][2*i+0] * kerv[4*k+3];
      //out[96*i+k].y = -( sp[0][2*i+1] * kerv[4*k+0] + sp[1][2*i+1] * kerv[4*k+1] + sp[2][2*i+1] * kerv[4*k+2] + sp[3][2*i+1] * kerv[4*k+3] );
    }
  }

}  // end of _d_X_prepare_ev


/***********************************************************
 * kernel for X-preparation of eigenvector
 *
 * wrapper and iterator to call _d_X_prepare_ev
 ***********************************************************/
__global__ void ker_X_prepare_ev ( cuDoubleComplex* _RESTR out, const double* _RESTR in, const double *_RESTR kerv, const int N )
{
  int const index  = blockIdx.x * blockDim.x + threadIdx.x;

  int const stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i += stride ) 
  {
    size_t offset = 96 * 12 * i;
    cuDoubleComplex * _out  = out  + offset;

    offset = 24 * i;
    const double    * _in   = in   + offset;

    offset = 384 * i;
    const double    * _kerv = kerv + offset;
 
    _d_X_prepare_ev ( _out, _in, _kerv );
  }

  return;
}  // end of ker_X_prepare_ev


/***********************************************************
 * kernel for X-preparation of eigenvector
 *
 * wrapper and iterator to call _d_X_prepare_ev
 ***********************************************************/
// __global__ void test_kernel ( double * _RESTR out, const double* _RESTR in, const double *_RESTR kerv, const int N )
__global__ void test_kernel ( cuDoubleComplex * _RESTR out, const double* _RESTR in, const double *_RESTR kerv, const int N )
{

  int const index  = blockIdx.x * blockDim.x + threadIdx.x;

  int const stride = blockDim.x * gridDim.x;

  // printf ("# [test_kernel] index %6d   stride %6d\n", index, stride);

  for (int i = index; i < N; i += stride ) 
  {
    // printf ("# [test_kernel] index %6d   stride %6d   i %6d\n", index, stride, i);
    // double *       _out  = out  +  24 * i;
    size_t offset = 12 * 96 * i;
    cuDoubleComplex * _out  = out  +  offset;

    offset = 24 * i;
    const double    * _in   = in   +  offset;

    offset = 384 * i;
    const double    * _kerv = kerv + offset;
 
    // _d_fv_eq_zero( _out );
   
    /* _d_fv_eq_fv ( _out, _in );
    _d_fv_ti_eq_g5 ( _out ); */

    // _d_fv_eq_gamma_ti_fv ( _out, 12, _in );

    _d_X_prepare_ev ( _out, _in, _kerv );
  }

  return;
}  // end of test_kernel
