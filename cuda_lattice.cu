/* -*- mode: c++ -*- */

#include "cuda_lattice.h"

#include <cassert>
#include <cuda_runtime.h>

/// from global.h without pulling in the whole header
#define _GSI(_ix) (24*(_ix))
#ifdef __restrict__
#define _RESTR __restrict__
#else
#define _RESTR
#endif

__device__ __constant__ int gamma_permutation[16][24] = {
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
__device__ __constant__ int gamma_sign[16][24] = {
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
__device__ __constant__ IdxComb idx_comb {
  .comb = {
    {0,1},
    {0,2},
    {0,3},
    {1,2},
    {1,3},
    {2,3}
  }};


/**
 * See geometry comment in cuda_lattice.h
 */
const int BS = CUDA_BLOCK_SIZE;
__device__ inline Coord get_thread_origin(Geom local_geom) {
  int x = BS*(blockIdx.x * blockDim.x + threadIdx.x);
  int t = BS*(x / local_geom.LX);
  x %= local_geom.LX;
  int y = BS*(blockIdx.y * blockDim.y + threadIdx.y);
  int z = BS*(blockIdx.z * blockDim.z + threadIdx.z);
  return Coord { .t = t, .x = x, .y = y, .z = z };
}
__device__ inline size_t coord2lexic(Coord coord, Geom local_geom) {
  return (((coord.t*local_geom.LX) + coord.x)*local_geom.LY + coord.y)*local_geom.LZ + coord.z;
}
__device__ inline Coord lexic2coord(size_t ind, Geom local_geom) {
  return Coord {
    .t = (int)(ind / (local_geom.LX * local_geom.LY * local_geom.LZ)),
    .x = (int)((ind / (local_geom.LY * local_geom.LZ)) % local_geom.LX),
    .y = (int)((ind / local_geom.LZ) % local_geom.LY),
    .z = (int)(ind % local_geom.LZ)
  };
}

/**
 * Given length-24 spin vector in, multiply by appropriate gamma matrix, writing
 * to out (non-aliasing assumed).
 */
__device__ inline void _fv_eq_gamma_ti_fv(double* _RESTR out, int gamma_index, const double* _RESTR in) {
  for (int i = 0; i < 24; ++i) {
    out[i] = in[gamma_permutation[gamma_index][i]] * gamma_sign[gamma_index][i];
  }
}
__device__ inline void _fv_ti_eq_g5(double* in_out) {
  for (int i = 12; i < 24; ++i) {
    in_out[i] *= -1;
  }
}

/**
 * 1D kernels: operate over CUDA_BLOCK_SIZE spinor elements each.
 *  - `len`: num *doubles* in the input/output array (must be divisible by 24)
 */
__global__ void ker_spinor_field_eq_gamma_ti_spinor_field(
    double* _RESTR out, int gamma_index, const double* _RESTR in, size_t len) {
  int start_ind = BS*_GSI(blockIdx.x * blockDim.x + threadIdx.x);
  for (int ind = start_ind; ind < len && ind < (start_ind + BS*24); ind += 24) {
    double* rr = out + ind;
    const double* ss = in + ind;
    _fv_eq_gamma_ti_fv(rr, gamma_index, ss);
  }
}

__global__ void ker_g5_phi(double* spinor, size_t len) {
  /* invert sign of spin components 2 and 3 */
  int start_ind = BS*_GSI(blockIdx.x * blockDim.x + threadIdx.x);
  for (int ind = start_ind; ind < len && ind < (start_ind + BS*24); ind += 24) {
    _fv_ti_eq_g5(&spinor[ind]);
  }
}

/**
 * Wrap coords to [-L/2, L/2]. Assumes inputs are in [0, ..., L-1]
 */
__device__ int coord_map(int xi, int Li) {
  return (xi >= Li / 2) ? (xi - Li) : xi;
}
__device__ int coord_map_zerohalf(int xi, int Li) {
  return (xi > Li / 2) ? xi - Li : ( (xi < Li / 2) ? xi : 0 );
}


/**
 * 4D kernels: operate over CUDA_BLOCK_SIZE^4 spinor elements each.
 */
__global__ void ker_dzu_dzsu(
    double* _RESTR dzu, double* _RESTR dzsu, const double* _RESTR fwd_src, const double* _RESTR fwd_y,
    int iflavor, Coord g_proc_coords, Coord gsx,
    Geom global_geom, Geom local_geom) {

  // Coord origin = get_thread_origin(local_geom);
  int gsx_arr[4] = {gsx.t, gsx.x, gsx.y, gsx.z};
  size_t VOLUME = local_geom.T * local_geom.LX * local_geom.LY * local_geom.LZ;
  int local_geom_arr[4] = {local_geom.T, local_geom.LX, local_geom.LY, local_geom.LZ};
  int global_geom_arr[4] = {global_geom.T, global_geom.LX, global_geom.LY, global_geom.LZ};
  int proc_coord_arr[4] = {g_proc_coords.t, g_proc_coords.x, g_proc_coords.y, g_proc_coords.z};

  // double dzu_work[6 * 12 * 12 * 2] = { 0 };
  // double dzsu_work[4 * 12 * 12 * 2] = { 0 };
  double spinor_work_0[24] = { 0 };
  double spinor_work_1[24] = { 0 };

  for (int ia = 0; ia < 12; ++ia) {
    const double* fwd_base = &fwd_src[_GSI(VOLUME) * (iflavor * 12 + ia)];
    for (int k = 0; k < 6; ++k) {
      const int sigma = idx_comb.comb[k][1];
      const int rho = idx_comb.comb[k][0];
      double dzu_work[12 * 2] = { 0 };
      for (int iz = blockIdx.x * blockDim.x + threadIdx.x;
           iz < VOLUME; iz += blockDim.x * gridDim.x) {
        const Coord coord = lexic2coord(iz, local_geom);
        const int tt = coord.t;
        const int xx = coord.x;
        const int yy = coord.y;
        const int zz = coord.z;
        const double* _u = &fwd_base[_GSI(iz)];
        double* _t_sigma = spinor_work_0;
        double* _t_rho = spinor_work_1;
        _fv_eq_gamma_ti_fv(_t_sigma, sigma, _u);
        _fv_ti_eq_g5(_t_sigma);
        _fv_eq_gamma_ti_fv(_t_rho, rho, _u);
        _fv_ti_eq_g5(_t_rho);
        int coord_arr[4] = {tt, xx, yy, zz};
        int zrho = coord_arr[rho] + proc_coord_arr[rho] * local_geom_arr[rho] - gsx_arr[rho];
        zrho = (zrho + global_geom_arr[rho]) % global_geom_arr[rho];
        int zsigma = coord_arr[sigma] + proc_coord_arr[sigma] * local_geom_arr[sigma] - gsx_arr[sigma];
        zsigma = (zsigma + global_geom_arr[sigma]) % global_geom_arr[sigma];
        int factor_rho = coord_map_zerohalf(zrho, global_geom_arr[rho]);
        int factor_sigma = coord_map_zerohalf(zsigma, global_geom_arr[sigma]);
        for (int ib = 0; ib < 12; ++ib) {
          for (int i = 0; i < 12; ++i) {
            double fwd_y_re = fwd_y[((1-iflavor) * 12 + ib) * _GSI(VOLUME) + _GSI(iz) + 2*i];
            double fwd_y_im = fwd_y[((1-iflavor) * 12 + ib) * _GSI(VOLUME) + _GSI(iz) + 2*i+1];
            double s_re = (_t_sigma[2*i] * factor_rho - _t_rho[2*i] * factor_sigma);
            double s_im = (_t_sigma[2*i+1] * factor_rho - _t_rho[2*i+1] * factor_sigma);
            // dzu_work[((k * 12 + ia) * 12 + ib) * 2 + 0] += fwd_y_re * s_re + fwd_y_im * s_im;
            // dzu_work[((k * 12 + ia) * 12 + ib) * 2 + 1] += fwd_y_re * s_im - fwd_y_im * s_re;
            dzu_work[2*ib] += fwd_y_re * s_re + fwd_y_im * s_im;
            dzu_work[2*ib+1] += fwd_y_re * s_im - fwd_y_im * s_re;
          }
        }
      } // end vol loop

      // reduce (TODO faster reduce algo?)
      for (int ib = 0; ib < 12; ++ib) {
        int ind = ((k * 12 + ia) * 12 + ib) * 2;
        atomicAdd_system(&dzu[ind], dzu_work[2*ib]);
        atomicAdd_system(&dzu[ind+1], dzu_work[2*ib+1]);
      }
    }

    for (int sigma = 0; sigma < 4; ++sigma) {
      // const double* fwd_base = &fwd_src[_GSI(VOLUME) * (iflavor * 12 + ia)];
      for (int ib = 0; ib < 12; ++ib) {
        double dzsu_work_re = 0.0;
        double dzsu_work_im = 0.0;
        for (int iz = blockIdx.x * blockDim.x + threadIdx.x;
             iz < VOLUME; iz += blockDim.x * gridDim.x) {
          // const Coord coord = lexic2coord(iz, local_geom);
          // const int tt = coord.t;
          // const int xx = coord.x;
          // const int yy = coord.y;
          // const int zz = coord.z;
          const double* _u = &fwd_base[_GSI(iz)];
          double* _t = spinor_work_0;
          _fv_eq_gamma_ti_fv(_t, sigma, _u);
          _fv_ti_eq_g5(_t);

          for (int i = 0; i < 12; ++i) {
            double fwd_y_re = fwd_y[((1-iflavor) * 12 + ib) * _GSI(VOLUME) + _GSI(iz) + 2*i];
            double fwd_y_im = fwd_y[((1-iflavor) * 12 + ib) * _GSI(VOLUME) + _GSI(iz) + 2*i+1];
            double s_re = _t[2*i];
            double s_im = _t[2*i+1];
            // dzsu_work[((sigma * 12 + ia) * 12 + ib) * 2 + 0] += fwd_y_re * s_re + fwd_y_im * s_im;
            // dzsu_work[((sigma * 12 + ia) * 12 + ib) * 2 + 1] += fwd_y_re * s_im - fwd_y_im * s_re;
            dzsu_work_re += fwd_y_re * s_re + fwd_y_im * s_im;
            dzsu_work_im += fwd_y_re * s_im - fwd_y_im * s_re;
          }
        } // end vol loop

        // reduce (TODO faster reduce algo?)
        int ind = ((sigma * 12 + ia) * 12 + ib) * 2;
        atomicAdd_system(&dzsu[ind], dzsu_work_re);
        atomicAdd_system(&dzsu[ind+1], dzsu_work_im);
      }
    }
  }
}

// typedef void (*QED_kernel_LX_ptr)( const double xv[4], const double yv[4], const struct QED_kernel_temps t, double kerv[6][4][4][4] );

__device__
void KQED_LX(
    int ikernel, const double xm[4], const double ym[4],
    const struct QED_kernel_temps kqed_t, double kerv[6][4][4][4]) {
#if CUDA_N_QED_KERNEL != 3
  #error "Number of QED kernels does not match implementation"
#endif
  if (ikernel == 0) {
    QED_kernel_L0( xm, ym, kqed_t, kerv );
  }
  else if (ikernel == 1) {
    QED_kernel_L3( xm, ym, kqed_t, kerv );
  }
  else {
    QED_Mkernel_L2( 0.4, xm, ym, kqed_t, kerv );
  }
}

__global__
void ker_4pt_contraction(
    double* _RESTR kernel_sum, const double* _RESTR g_dzu, const double* _RESTR g_dzsu,
    const double* _RESTR fwd_src, const double* _RESTR fwd_y, int iflavor, Coord g_proc_coords,
    Coord gsx, Pair xunit, Coord yv, QED_kernel_temps kqed_t,
    Geom global_geom, Geom local_geom) {

  // Coord origin = get_thread_origin(local_geom);
  int gsx_arr[4] = {gsx.t, gsx.x, gsx.y, gsx.z};
  size_t VOLUME = local_geom.T * local_geom.LX * local_geom.LY * local_geom.LZ;
  int local_geom_arr[4] = {local_geom.T, local_geom.LX, local_geom.LY, local_geom.LZ};
  int global_geom_arr[4] = {global_geom.T, global_geom.LX, global_geom.LY, global_geom.LZ};
  int proc_coord_arr[4] = {g_proc_coords.t, g_proc_coords.x, g_proc_coords.y, g_proc_coords.z};

  // double corr_I[6 * 4 * 4 * 4 * 2];
  // double corr_II[6 * 4 * 4 * 4 * 2];
  // double dxu[4 * 12 * 12 * 2];
  // __shared__ double corr_I_shared[CUDA_THREAD_DIM_1D * 6 * 4 * 4 * 4];
  // __shared__ double corr_II_shared[CUDA_THREAD_DIM_1D * 6 * 4 * 4 * 4];
  // double* corr_I_re = &corr_I_shared[threadIdx.x * 6 * 4 * 4 * 4];
  // double* corr_II_re = &corr_II_shared[threadIdx.x * 6 * 4 * 4 * 4];
  double corr_I_re[6 * 4 * 4 * 4];
  double corr_II_re[6 * 4 * 4 * 4];

  double kernel_sum_work[CUDA_N_QED_KERNEL] = { 0 };
  double spinor_work_0[24], spinor_work_1[24];
  double kerv[6][4][4][4] KQED_ALIGN = { 0 };

  for (int ix = blockIdx.x * blockDim.x + threadIdx.x;
       ix < VOLUME; ix += blockDim.x * gridDim.x) {
    Coord coord = lexic2coord(ix, local_geom);
    const int tt = coord.t;
    const int xx = coord.x;
    const int yy = coord.y;
    const int zz = coord.z;
    int coord_arr[4] = {tt, xx, yy, zz};
    int xv[4], xvzh[4];
    #pragma unroll
    for (int rho = 0; rho < 4; ++rho) {
      int xrho = coord_arr[rho] + proc_coord_arr[rho] * local_geom_arr[rho] - gsx_arr[rho];
      xrho = (xrho + global_geom_arr[rho]) % global_geom_arr[rho];
      xv[rho] = coord_map(xrho, global_geom_arr[rho]);
      xvzh[rho] = coord_map_zerohalf(xrho, global_geom_arr[rho]);
    }

    #pragma unroll
    for (int mu = 0; mu < 4; ++mu) {
      /// COMPUTE DXU v1
      double dxu[12 * 12 * 2] = { 0 };
      for (int ia = 0; ia < 12; ++ia) {
        const double* _d = &fwd_src[((1-iflavor) * 12 + ia) * _GSI(VOLUME) + _GSI(ix)];
        double* _t = spinor_work_1;
        _fv_eq_gamma_ti_fv(_t, mu, _d);
        _fv_ti_eq_g5(_t);
        for (int ib = 0; ib < 12; ++ib) {
          const double* _u = &fwd_y[(iflavor * 12 + ib) * _GSI(VOLUME) + _GSI(ix)];
          for (int i = 0; i < 12; ++i) {
            double _t_re = _t[2*i];
            double _t_im = _t[2*i+1];
            double _u_re = _u[2*i];
            double _u_im = _u[2*i+1];
            /* -1 factor due to (g5 gmu)^+ = -g5 gmu */
            dxu[(ib * 12 + ia) * 2 + 0] += -(_t_re * _u_re + _t_im * _u_im);
            dxu[(ib * 12 + ia) * 2 + 1] += -(_t_re * _u_im - _t_im * _u_re);
          }
        }
      }

      #pragma unroll
      for (int k = 0; k < 6; ++k) {
        const int sigma = idx_comb.comb[k][1];
        const int rho = idx_comb.comb[k][0];
        #pragma unroll
        for (int nu = 0; nu < 4; ++nu) {
          #pragma unroll
          for (int lambda = 0; lambda < 4; ++lambda) {
            double *_corr_I_re = &corr_I_re[((k * 4 + mu) * 4 + nu) * 4 + lambda];
            double *_corr_II_re = &corr_II_re[((k * 4 + mu) * 4 + nu) * 4 + lambda];
            _corr_I_re[0] = 0.0;
            _corr_II_re[0] = 0.0;
            for (int ia = 0; ia < 12; ++ia) {
              /// COMPUTE DXU v2
              // double dxu[12 * 2];
              // const double* _u = &fwd_y[(iflavor * 12 + ia) * _GSI(VOLUME) + _GSI(ix)];
              // for (int ic = 0; ic < 12; ++ic) {
              //   const double* _d = &fwd_src[((1-iflavor) * 12 + ic) * _GSI(VOLUME) + _GSI(ix)];
              //   double* _t = spinor_work_1;
              //   _fv_eq_gamma_ti_fv(_t, mu, _d);
              //   _fv_ti_eq_g5(_t);
              //   for (int i = 0; i < 12; ++i) {
              //     double _t_re = _t[2*i];
              //     double _t_im = _t[2*i+1];
              //     double _u_re = _u[2*i];
              //     double _u_im = _u[2*i+1];
              //     /* -1 factor due to (g5 gmu)^+ = -g5 gmu */
              //     dxu[2*ic] += -(_t_re * _u_re + _t_im * _u_im);
              //     dxu[2*ic+1] += -(_t_re * _u_im - _t_im * _u_re);
              //   }
              // }
              // double *_dxu = &dxu[0];
              double *_dxu = &dxu[ia * 12 * 2];
              double *_t = spinor_work_0;
              _fv_eq_gamma_ti_fv(_t, 5, _dxu);
              double *_g_dxu = spinor_work_1;
              _fv_eq_gamma_ti_fv(_g_dxu, lambda, _t);
              for (int ib = 0; ib < 12; ++ib) {
                double u_re = _g_dxu[2*ib];
                double u_im = _g_dxu[2*ib+1];
                double v_re = g_dzu[(((k * 4 + nu) * 12 + ib) * 12 + ia) * 2];
                double v_im = g_dzu[(((k * 4 + nu) * 12 + ib) * 12 + ia) * 2 + 1];
                _corr_I_re[0] -= u_re * v_re - u_im * v_im;
                v_re = (
                    xvzh[rho] * g_dzsu[(((sigma * 4 + nu) * 12 + ib) * 12 + ia) * 2] -
                    xvzh[sigma] * g_dzsu[(((rho * 4 + nu) * 12 + ib) * 12 + ia) * 2] );
                v_im = (
                    xvzh[rho] * g_dzsu[(((sigma * 4 + nu) * 12 + ib) * 12 + ia) * 2 + 1] -
                    xvzh[sigma] * g_dzsu[(((rho * 4 + nu) * 12 + ib) * 12 + ia) * 2 + 1] );
                _corr_II_re[0] -= u_re * v_re - u_im * v_im;
              }
            }
          }
        }
      }
    }

    double const xm[4] = {
      xv[0] * xunit.a,
      xv[1] * xunit.a,
      xv[2] * xunit.a,
      xv[3] * xunit.a };

    double const ym[4] = {
      yv.t * xunit.a,
      yv.x * xunit.a,
      yv.y * xunit.a,
      yv.z * xunit.a };

    double * const _corr_I_re  = corr_I_re;
    double * const _corr_II_re = corr_II_re;

    double const xm_mi_ym[4] = {
      xm[0] - ym[0],
      xm[1] - ym[1],
      xm[2] - ym[2],
      xm[3] - ym[3] };

    for (int ikernel = 0; ikernel < CUDA_N_QED_KERNEL; ++ikernel) {
      // dtmp += (
      //     kerv1[k][mu][nu][lambda] + kerv2[k][nu][mu][lambda]
      //     - kerv3[k][lambda][nu][mu] ) * _corr_I[2*i]
      //     + kerv3[k][lambda][nu][mu] * _corr_II[2*i];
      double dtmp = 0.;
      int i;
      KQED_LX( ikernel, xm, ym, kqed_t, kerv );
      i = 0;
      for( int k = 0; k < 6; k++ ) {
        for ( int mu = 0; mu < 4; mu++ ) {
          for ( int nu = 0; nu < 4; nu++ ) {
            for ( int lambda = 0; lambda < 4; lambda++ ) {
              dtmp += kerv[k][mu][nu][lambda] * _corr_I_re[i];
              // dtmp += _corr_I_re[i];
              i++;
            }
          }
        }
      }
      KQED_LX( ikernel, ym, xm,       kqed_t, kerv );
      i = 0;
      for( int k = 0; k < 6; k++ ) {
        for ( int mu = 0; mu < 4; mu++ ) {
          for ( int nu = 0; nu < 4; nu++ ) {
            for ( int lambda = 0; lambda < 4; lambda++ ) {
              dtmp += kerv[k][nu][mu][lambda] * _corr_I_re[i];
              // dtmp += _corr_I_re[i];
              i++;
            }
          }
        }
      }
      KQED_LX( ikernel, xm, xm_mi_ym, kqed_t, kerv );
      i = 0;
      for( int k = 0; k < 6; k++ ) {
        for ( int mu = 0; mu < 4; mu++ ) {
          for ( int nu = 0; nu < 4; nu++ ) {
            for ( int lambda = 0; lambda < 4; lambda++ ) {
              dtmp -= kerv[k][lambda][nu][mu] * _corr_I_re[i];
              dtmp += kerv[k][lambda][nu][mu] * _corr_II_re[i];
              // dtmp += _corr_II_re[i] - _corr_I_re[i];
              i++;
            }
          }
        }
      }
      kernel_sum_work[ikernel] += dtmp;
    }

  } // end coord loop

  // reduce (TODO faster reduce algo?)
  for (int ikernel = 0; ikernel < CUDA_N_QED_KERNEL; ++ikernel) {
    atomicAdd_system(&kernel_sum[ikernel], kernel_sum_work[ikernel]);
  }
}

__global__
void ker_2p2_pieces(
    double* _RESTR P1, double* _RESTR P23x,
    const double* _RESTR fwd_y, int iflavor, Coord g_proc_coords,
    Coord gsw, int n_y, Coord* gycoords, Pair xunit, QED_kernel_temps kqed_t,
    Geom global_geom, Geom local_geom, int Lmax) {
  int gsw_arr[4] = {gsw.t, gsw.x, gsw.y, gsw.z};
  size_t VOLUME = local_geom.T * local_geom.LX * local_geom.LY * local_geom.LZ;
  int local_geom_arr[4] = {local_geom.T, local_geom.LX, local_geom.LY, local_geom.LZ};
  int global_geom_arr[4] = {global_geom.T, global_geom.LX, global_geom.LY, global_geom.LZ};
  int proc_coord_arr[4] = {g_proc_coords.t, g_proc_coords.x, g_proc_coords.y, g_proc_coords.z};

  double pimn[4][4];
  double spinor_work_0[24], spinor_work_1[24];
  double kerv[6][4][4][4] KQED_ALIGN = { 0 };

  for (int ix = blockIdx.x * blockDim.x + threadIdx.x;
       ix < VOLUME; ix += blockDim.x * gridDim.x) {
    Coord coord = lexic2coord(ix, local_geom);
    int coord_arr[4] = {coord.t, coord.x, coord.y, coord.z};

    for (int nu = 0; nu < 4; ++nu) {
      for (int mu = 0; mu < 4; ++mu) {
        pimn[mu][nu] = 0.0;
        // V2: Sparse summation using properties of gamma matrices
        for (int ib = 0; ib < 12; ++ib) {
          int ia = (gamma_permutation[nu][2*ib]) / 2;
          int sign_re = ((ib >= 6) ? -1 : 1) * gamma_sign[nu][2*ib];
          int sign_im = ((ib >= 6) ? -1 : 1) * gamma_sign[nu][2*ib+1];
          bool re_im_swap = gamma_permutation[nu][2*ib] % 2 == 1;
          
          const double* _u = &fwd_y[(iflavor * 12 + ia) * _GSI(VOLUME) + _GSI(ix)];
          double* _t = spinor_work_0;
          _fv_eq_gamma_ti_fv(_t, mu, _u);
          _fv_ti_eq_g5(_t);

          const double* _d = &fwd_y[((1-iflavor) * 12 + ib) * _GSI(VOLUME) + _GSI(ix)];
          for (int i = 0; i < 12; ++i) {
            const double _t_re = _t[2*i];
            const double _t_im = _t[2*i+1];
            const double _d_re = _d[2*i];
            const double _d_im = _d[2*i+1];
            if (!re_im_swap) {
              pimn[mu][nu] += sign_re * (_d_re * _t_re + _d_im * _t_im);
            }
            else {
              pimn[mu][nu] += sign_im * (_d_re * _t_im - _d_im * _t_re);
            }
          }
        }
        
        // V1: Direct loop and trace
        /*
        for (int ia = 0; ia < 12; ++ia) {
          const double* _u = &fwd_y[(iflavor * 12 + ia) * _GSI(VOLUME) + _GSI(ix)];
          double* _t = spinor_work_0;
          _fv_eq_gamma_ti_fv(_t, mu, _u);
          _fv_ti_eq_g5(_t);
          double* _s = spinor_work_1;
          for (int ib = 0; ib < 12; ++ib) {
            const double* _d = &fwd_y[((1-iflavor) * 12 + ib) * _GSI(VOLUME) + _GSI(ix)];
            double w_re = 0.0;
            double w_im = 0.0;
            for (int i = 0; i < 12; ++i) {
              const double _t_re = _t[2*i];
              const double _t_im = _t[2*i+1];
              const double _d_re = _d[2*i];
              const double _d_im = _d[2*i+1];
              w_re += _d_re * _t_re + _d_im * _t_im;
              w_im += _d_re * _t_im - _d_im * _t_re;
            }
            _s[2*ib] = w_re;
            _s[2*ib+1] = w_im;
          }
          _fv_ti_eq_g5(_s);
          _fv_eq_gamma_ti_fv(_t, nu, _s);
          // real part
          pimn[mu][nu] += _t[2*ia];
        }
        */
      }
    }

    int z[4];
    #pragma unroll
    for (int rho = 0; rho < 4; ++rho) {
      int zrho = coord_arr[rho] + proc_coord_arr[rho] * local_geom_arr[rho] - gsw_arr[rho];
      z[rho] = (zrho + global_geom_arr[rho]) % global_geom_arr[rho];
    }
    // reduce (TODO faster reduce algo?)
    for (int sigma = 0; sigma < 4; ++sigma) {
      for (int nu = 0; nu < 4; ++nu) {
        for (int rho = 0; rho < 4; ++rho) {
          atomicAdd_system(
              &P1[(((rho * 4) + sigma) * 4 + nu) * Lmax + z[rho]],
              pimn[sigma][nu]);
        }
      }
    }

    for (int yi = 0; yi < n_y; yi++) {
      // For P2: y = (gsy - gsw)
      // For P3, P4, P5: y' = (gsw - gsy)
      // We define y = (gsy - gsw) and use -y as input for P3, P4, P5.
      int const gsy_arr[4] = { gycoords[yi].t, gycoords[yi].x, gycoords[yi].y, gycoords[yi].z };
      int const y[4] = {
        (gsy_arr[0] - gsw_arr[0] + global_geom_arr[0]) % global_geom_arr[0],
        (gsy_arr[1] - gsw_arr[1] + global_geom_arr[1]) % global_geom_arr[1],
        (gsy_arr[2] - gsw_arr[2] + global_geom_arr[2]) % global_geom_arr[2],
        (gsy_arr[3] - gsw_arr[3] + global_geom_arr[3]) % global_geom_arr[3]
      };
      int const yv[4] = {
        coord_map_zerohalf(y[0], global_geom.T),
        coord_map_zerohalf(y[1], global_geom.LX),
        coord_map_zerohalf(y[2], global_geom.LY),
        coord_map_zerohalf(y[3], global_geom.LZ)
      };
      double const ym[4] = {
        yv[0] * xunit.a,
        yv[1] * xunit.a,
        yv[2] * xunit.a,
        yv[3] * xunit.a };
      double const ym_minus[4] = {
        -yv[0] * xunit.a,
        -yv[1] * xunit.a,
        -yv[2] * xunit.a,
        -yv[3] * xunit.a };
      
      int const xv[4] = {
        coord_map_zerohalf(z[0], global_geom_arr[0]),
        coord_map_zerohalf(z[1], global_geom_arr[1]),
        coord_map_zerohalf(z[2], global_geom_arr[2]),
        coord_map_zerohalf(z[3], global_geom_arr[3])
      };
      double const xm[4] = {
        xv[0] * xunit.a,
        xv[1] * xunit.a,
        xv[2] * xunit.a,
        xv[3] * xunit.a };
      double const xm_minus[4] = {
        -xv[0] * xunit.a,
        -xv[1] * xunit.a,
        -xv[2] * xunit.a,
        -xv[3] * xunit.a };
        
      // int const x_mi_y[4] = {
      //   (z[0] - y[0] + global_geom_arr[0]) % global_geom_arr[0],
      //   (z[1] - y[1] + global_geom_arr[1]) % global_geom_arr[1],
      //   (z[2] - y[2] + global_geom_arr[2]) % global_geom_arr[2],
      //   (z[3] - y[3] + global_geom_arr[2]) % global_geom_arr[3]
      // };
      // int xv_mi_yv[4] = {
      //   coord_map_zerohalf(x_mi_y[0], global_geom_arr[0]),
      //   coord_map_zerohalf(x_mi_y[1], global_geom_arr[1]),
      //   coord_map_zerohalf(x_mi_y[2], global_geom_arr[2]),
      //   coord_map_zerohalf(x_mi_y[3], global_geom_arr[3])
      // };
      // double const xm_mi_ym[4] = {
      //   xv_mi_yv[0] * xunit.a,
      //   xv_mi_yv[1] * xunit.a,
      //   xv_mi_yv[2] * xunit.a,
      //   xv_mi_yv[3] * xunit.a
      // };
      double const xm_mi_ym[4] = {
        xm[0] - ym[0],
        xm[1] - ym[1],
        xm[2] - ym[2],
        xm[3] - ym[3] };
      double const ym_mi_xm[4] = {
        ym[0] - xm[0],
        ym[1] - xm[1],
        ym[2] - xm[2],
        ym[3] - xm[3] };

      for (int ikernel = 0; ikernel < CUDA_N_QED_KERNEL; ++ikernel) {
        double local_P2_0[4][4][4] = { 0 };
        double local_P2_1[4][4][4] = { 0 };
        double local_P3[4][4][4] = { 0 };
        double local_P4_0[4][4][4] = { 0 };
        double local_P4_1[4][4][4] = { 0 };
        KQED_LX( ikernel, xm, ym, kqed_t, kerv );
        for( int k = 0; k < 6; k++ ) {
          int const rho = idx_comb.comb[k][0];
          int const sigma = idx_comb.comb[k][1];
          for ( int nu = 0; nu < 4; nu++ ) {
            local_P2_0[rho][sigma][nu] = 0.0;
            for ( int mu = 0; mu < 4; mu++ ) {
              for ( int lambda = 0; lambda < 4; lambda++ ) {
                local_P2_0[rho][sigma][nu] += kerv[k][mu][nu][lambda] * pimn[mu][lambda];
              }
            }
          }
        }
        KQED_LX( ikernel, ym, xm, kqed_t, kerv );
        for( int k = 0; k < 6; k++ ) {
          int const rho = idx_comb.comb[k][0];
          int const sigma = idx_comb.comb[k][1];
          for ( int nu = 0; nu < 4; nu++ ) {
            local_P2_1[rho][sigma][nu] = 0.0;
            for ( int mu = 0; mu < 4; mu++ ) {
              for ( int lambda = 0; lambda < 4; lambda++ ) {
                local_P2_1[rho][sigma][nu] += kerv[k][nu][mu][lambda] * pimn[mu][lambda];
              }
            }
          }
        }
        KQED_LX( ikernel, xm_mi_ym, ym_minus, kqed_t, kerv );
        for( int k = 0; k < 6; k++ ) {
          int const rho = idx_comb.comb[k][0];
          int const sigma = idx_comb.comb[k][1];
          for ( int nu = 0; nu < 4; nu++ ) {
            local_P3[rho][sigma][nu] = 0.0;
            for ( int mu = 0; mu < 4; mu++ ) {
              for ( int lambda = 0; lambda < 4; lambda++ ) {
                local_P3[rho][sigma][nu] += kerv[k][mu][lambda][nu] * pimn[mu][lambda];
              }
            }
          }
        }
        KQED_LX( ikernel, ym_mi_xm, xm_minus, kqed_t, kerv );
        for( int k = 0; k < 6; k++ ) {
          int const rho = idx_comb.comb[k][0];
          int const sigma = idx_comb.comb[k][1];
          for ( int nu = 0; nu < 4; nu++ ) {
            local_P4_0[rho][sigma][nu] = 0.0;
            for ( int mu = 0; mu < 4; mu++ ) {
              for ( int lambda = 0; lambda < 4; lambda++ ) {
                local_P4_0[rho][sigma][nu] += kerv[k][nu][lambda][mu] * pimn[mu][lambda];
              }
            }
            local_P4_1[rho][sigma][nu] = local_P4_0[rho][sigma][nu] * (yv[rho]-xv[rho]);
            local_P4_1[sigma][rho][nu] = -local_P4_0[rho][sigma][nu] * (yv[sigma]-xv[sigma]);
          }
        }

        // reduce (TODO faster reduce algo?)
        int ind;
        for (int rho = 0; rho < 4; ++rho) {
          for (int sigma = 0; sigma < 4; ++sigma) {
            for (int nu = 0; nu < 4; ++nu) {
              #if CUDA_N_QED_GEOM != 5
              #error "Number of QED kernel geometries does not match implementation"
              #endif
              ind = ((((yi*CUDA_N_QED_KERNEL + ikernel)*CUDA_N_QED_GEOM + 0)*4 + rho)*4 + sigma)*4 + nu;
              atomicAdd_system(&P23x[ind], local_P2_0[rho][sigma][nu]);
              ind = ((((yi*CUDA_N_QED_KERNEL + ikernel)*CUDA_N_QED_GEOM + 1)*4 + rho)*4 + sigma)*4 + nu;
              atomicAdd_system(&P23x[ind], local_P2_1[rho][sigma][nu]);
              ind = ((((yi*CUDA_N_QED_KERNEL + ikernel)*CUDA_N_QED_GEOM + 2)*4 + rho)*4 + sigma)*4 + nu;
              atomicAdd_system(&P23x[ind], local_P3[rho][sigma][nu]);
              ind = ((((yi*CUDA_N_QED_KERNEL + ikernel)*CUDA_N_QED_GEOM + 3)*4 + rho)*4 + sigma)*4 + nu;
              atomicAdd_system(&P23x[ind], local_P4_0[rho][sigma][nu]);
              ind = ((((yi*CUDA_N_QED_KERNEL + ikernel)*CUDA_N_QED_GEOM + 4)*4 + rho)*4 + sigma)*4 + nu;
              atomicAdd_system(&P23x[ind], local_P4_1[rho][sigma][nu]);
            }
          }
        }

      }
    }
  }
}

/**
 * Top-level operations.
 */
void cu_spinor_field_eq_gamma_ti_spinor_field(double* out, int mu, const double* in, size_t len) {
  const size_t BS_spinor = 12 * CUDA_BLOCK_SIZE * CUDA_THREAD_DIM_1D;
  size_t nx = (len + BS_spinor - 1) / BS_spinor;
  dim3 kernel_nblocks(nx);
  dim3 kernel_nthreads(CUDA_THREAD_DIM_1D);
  ker_spinor_field_eq_gamma_ti_spinor_field<<<kernel_nblocks, kernel_nthreads>>>(
      out, mu, in, len);
}

void cu_g5_phi(double* spinor, size_t len) {
  const size_t BS_spinor = 12 * CUDA_BLOCK_SIZE * CUDA_THREAD_DIM_1D;
  size_t nx = (len + BS_spinor - 1) / BS_spinor;
  dim3 kernel_nblocks(nx);
  dim3 kernel_nthreads(CUDA_THREAD_DIM_1D);
  ker_g5_phi<<<kernel_nblocks, kernel_nthreads>>>(spinor, len);
}

void cu_dzu_dzsu(
    double* d_dzu, double* d_dzsu, const double* fwd_src, const double* fwd_y,
    int iflavor, Coord proc_coords, Coord gsx,
    Geom global_geom, Geom local_geom) {
  size_t T = local_geom.T;
  size_t LX = local_geom.LX;
  size_t LY = local_geom.LY;
  size_t LZ = local_geom.LZ;
  size_t VOLUME = T * LX * LY * LZ;
  size_t kernel_nthreads = CUDA_THREAD_DIM_1D;
  size_t kernel_nblocks = (VOLUME + kernel_nthreads - 1) / kernel_nthreads;
  // const size_t BS_TX = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE * CUDA_BLOCK_SIZE;
  // const size_t BS_Y = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE;
  // const size_t BS_Z = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE;
  // size_t nx = (T*LX + BS_TX - 1) / BS_TX;
  // size_t ny = (LY + BS_Y - 1) / BS_Y;
  // size_t nz = (LZ + BS_Z - 1) / BS_Z;
  // dim3 kernel_nblocks(nx, ny, nz);
  // dim3 kernel_nthreads(CUDA_THREAD_DIM_4D, CUDA_THREAD_DIM_4D, CUDA_THREAD_DIM_4D);
  ker_dzu_dzsu<<<kernel_nblocks, kernel_nthreads>>>(
      d_dzu, d_dzsu, fwd_src, fwd_y, iflavor, proc_coords, gsx,
      global_geom, local_geom);
}

void cu_4pt_contraction(
    double* d_kernel_sum, const double* d_g_dzu, const double* d_g_dzsu,
    const double* fwd_src, const double* fwd_y, int iflavor, Coord proc_coords,
    Coord gsx, Pair xunit, Coord yv, QED_kernel_temps kqed_t,
    Geom global_geom, Geom local_geom) {
  size_t T = local_geom.T;
  size_t LX = local_geom.LX;
  size_t LY = local_geom.LY;
  size_t LZ = local_geom.LZ;
  size_t VOLUME = T * LX * LY * LZ;
  size_t kernel_nthreads = CUDA_THREAD_DIM_1D;
  size_t kernel_nblocks = (VOLUME + kernel_nthreads - 1) / kernel_nthreads;
  // const size_t BS_TX = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE * CUDA_BLOCK_SIZE;
  // const size_t BS_Y = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE;
  // const size_t BS_Z = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE;
  // size_t nx = (T*LX + BS_TX - 1) / BS_TX;
  // size_t ny = (LY + BS_Y - 1) / BS_Y;
  // size_t nz = (LZ + BS_Z - 1) / BS_Z;
  // dim3 kernel_nblocks(nx, ny, nz);
  // dim3 kernel_nthreads(CUDA_THREAD_DIM_4D, CUDA_THREAD_DIM_4D, CUDA_THREAD_DIM_4D);
  ker_4pt_contraction<<<kernel_nblocks, kernel_nthreads>>>(
      d_kernel_sum, d_g_dzu, d_g_dzsu, fwd_src, fwd_y, iflavor, proc_coords,
      gsx, xunit, yv, kqed_t, global_geom, local_geom);
}

void cu_2p2_pieces(
    double* d_P1, double* d_P23x, const double* fwd_y, int iflavor,
    Coord proc_coords, Coord gsw, int n_y, Coord* d_ycoords, Pair xunit,
    QED_kernel_temps kqed_t, Geom global_geom, Geom local_geom) {
  size_t T = local_geom.T;
  size_t LX = local_geom.LX;
  size_t LY = local_geom.LY;
  size_t LZ = local_geom.LZ;
  size_t VOLUME = T * LX * LY * LZ;
  size_t kernel_nthreads = CUDA_THREAD_DIM_1D;
  size_t kernel_nblocks = (VOLUME + kernel_nthreads - 1) / kernel_nthreads;
  int Lmax = 0;
  if (global_geom.T >= Lmax) Lmax = global_geom.T;
  if (global_geom.LX >= Lmax) Lmax = global_geom.LX;
  if (global_geom.LY >= Lmax) Lmax = global_geom.LY;
  if (global_geom.LZ >= Lmax) Lmax = global_geom.LZ;
  ker_2p2_pieces<<<kernel_nblocks, kernel_nthreads>>>(
      d_P1, d_P23x, fwd_y, iflavor, proc_coords, gsw, n_y, d_ycoords, xunit,
      kqed_t, global_geom, local_geom, Lmax);
}


/**
 * Simple interface to KQED.
 * NOTE: One should not really use these single-point evaluations in production,
 * they are only intended to test that the CUDA QED kernels work.
 */
#if 0
#include "KQED.h"

struct __attribute__((packed, aligned(8))) Vec4 {
  double x[4];
};
inline Vec4 vec4(const double xv[4]) {
  Vec4 pt;
  for (int i = 0; i < 4; ++i) {
    pt.x[i] = xv[i];
  }
  return pt;
}
struct __attribute__((packed, aligned(8))) OneKernel {
  double k[6][4][4][4] ;
};

__global__
void
ker_QED_kernel_L0(
    const Vec4 *d_xv, const Vec4 *d_yv, unsigned n,
    const struct QED_kernel_temps t, OneKernel *d_kerv ) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  QED_kernel_L0(d_xv[i].x, d_yv[i].x, t, (double (*)[4][4][4]) &d_kerv[i].k);
}
__global__
void
ker_QED_kernel_L1(
    const Vec4 *d_xv, const Vec4 *d_yv, unsigned n,
    const struct QED_kernel_temps t, OneKernel *d_kerv ) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  QED_kernel_L1(d_xv[i].x, d_yv[i].x, t, (double (*)[4][4][4]) &d_kerv[i].k);
}
__global__
void
ker_QED_kernel_L2(
    const Vec4 *d_xv, const Vec4 *d_yv, unsigned n,
    const struct QED_kernel_temps t, OneKernel *d_kerv ) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  QED_kernel_L2(d_xv[i].x, d_yv[i].x, t, (double (*)[4][4][4]) &d_kerv[i].k);
}
__global__
void
ker_QED_kernel_L3(
    const Vec4 *d_xv, const Vec4 *d_yv, unsigned n,
    const struct QED_kernel_temps t, OneKernel *d_kerv ) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  QED_kernel_L3(d_xv[i].x, d_yv[i].x, t, (double (*)[4][4][4]) &d_kerv[i].k);
}

void
cu_pt_QED_kernel_L0(
    const double xv[4] , const double yv[4] ,
    const struct QED_kernel_temps t , double kerv[6][4][4][4] ) {
  OneKernel *d_kerv;
  size_t sizeof_kerv = 6*4*4*4*sizeof(double);
  checkCudaErrors(cudaMalloc(&d_kerv, sizeof_kerv));
  Vec4 *d_xv, *d_yv;
  checkCudaErrors(cudaMalloc(&d_xv, sizeof(Vec4)));
  checkCudaErrors(cudaMalloc(&d_yv, sizeof(Vec4)));
  checkCudaErrors(cudaMemcpy(d_xv, xv, sizeof(Vec4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_yv, yv, sizeof(Vec4), cudaMemcpyHostToDevice));
  ker_QED_kernel_L0<<<1,1>>>( d_xv, d_yv, 1, t, d_kerv );
  checkCudaErrors(cudaMemcpy(kerv, d_kerv, sizeof_kerv, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_kerv));
  checkCudaErrors(cudaFree(d_xv));
  checkCudaErrors(cudaFree(d_yv));
}
void
cu_pt_QED_kernel_L1(
    const double xv[4] , const double yv[4] ,
    const struct QED_kernel_temps t , double kerv[6][4][4][4] ) {
  OneKernel *d_kerv;
  size_t sizeof_kerv = 6*4*4*4*sizeof(double);
  checkCudaErrors(cudaMalloc(&d_kerv, sizeof_kerv));
  Vec4 *d_xv, *d_yv;
  checkCudaErrors(cudaMalloc(&d_xv, sizeof(Vec4)));
  checkCudaErrors(cudaMalloc(&d_yv, sizeof(Vec4)));
  checkCudaErrors(cudaMemcpy(d_xv, xv, sizeof(Vec4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_yv, yv, sizeof(Vec4), cudaMemcpyHostToDevice));
  ker_QED_kernel_L1<<<1,1>>>( d_xv, d_yv, 1, t, d_kerv );
  checkCudaErrors(cudaMemcpy(kerv, d_kerv, sizeof_kerv, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_kerv));
  checkCudaErrors(cudaFree(d_xv));
  checkCudaErrors(cudaFree(d_yv));
}
void
cu_pt_QED_kernel_L2(
    const double xv[4] , const double yv[4] ,
    const struct QED_kernel_temps t , double kerv[6][4][4][4] ) {
  OneKernel *d_kerv;
  size_t sizeof_kerv = 6*4*4*4*sizeof(double);
  checkCudaErrors(cudaMalloc(&d_kerv, sizeof_kerv));
  Vec4 *d_xv, *d_yv;
  checkCudaErrors(cudaMalloc(&d_xv, sizeof(Vec4)));
  checkCudaErrors(cudaMalloc(&d_yv, sizeof(Vec4)));
  checkCudaErrors(cudaMemcpy(d_xv, xv, sizeof(Vec4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_yv, yv, sizeof(Vec4), cudaMemcpyHostToDevice));
  ker_QED_kernel_L2<<<1,1>>>( d_xv, d_yv, 1, t, d_kerv );
  checkCudaErrors(cudaMemcpy(kerv, d_kerv, sizeof_kerv, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_kerv));
  checkCudaErrors(cudaFree(d_xv));
  checkCudaErrors(cudaFree(d_yv));
}
void
cu_pt_QED_kernel_L3(
    const double xv[4] , const double yv[4] ,
    const struct QED_kernel_temps t , double kerv[6][4][4][4] ) {
  OneKernel *d_kerv;
  size_t sizeof_kerv = 6*4*4*4*sizeof(double);
  checkCudaErrors(cudaMalloc(&d_kerv, sizeof_kerv));
  Vec4 *d_xv, *d_yv;
  checkCudaErrors(cudaMalloc(&d_xv, sizeof(Vec4)));
  checkCudaErrors(cudaMalloc(&d_yv, sizeof(Vec4)));
  checkCudaErrors(cudaMemcpy(d_xv, xv, sizeof(Vec4), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_yv, yv, sizeof(Vec4), cudaMemcpyHostToDevice));
  ker_QED_kernel_L3<<<1,1>>>( d_xv, d_yv, 1, t, d_kerv );
  checkCudaErrors(cudaMemcpy(kerv, d_kerv, sizeof_kerv, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_kerv));
  checkCudaErrors(cudaFree(d_xv));
  checkCudaErrors(cudaFree(d_yv));
}
#endif
