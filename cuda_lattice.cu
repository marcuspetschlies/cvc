/* -*- mode: c++ -*- */

#include "cuda_lattice.h"

#include <cassert>
#include <cuda_runtime.h>

/// from global.h without pulling in the whole header
#define _GSI(_ix) (24*(_ix))

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

/**
 * Given length-24 spin vector in, multiply by appropriate gamma matrix, writing
 * to out (non-aliasing assumed).
 */
__device__ inline void _fv_eq_gamma_ti_fv(double* out, int gamma_index, const double* in) {
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
    double* out, int gamma_index, const double* in, size_t len) {
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

// __device__ void site_map(int xv[4], int const x[4], Geom global_geom ) {
//   int T_global = global_geom.T;
//   int LX_global = global_geom.LX;
//   int LY_global = global_geom.LY;
//   int LZ_global = global_geom.LZ;
//   xv[0] = ( x[0] >= T_global   / 2 ) ? (x[0] - T_global )  : x[0];
//   xv[1] = ( x[1] >= LX_global  / 2 ) ? (x[1] - LX_global)  : x[1];
//   xv[2] = ( x[2] >= LY_global  / 2 ) ? (x[2] - LY_global)  : x[2];
//   xv[3] = ( x[3] >= LZ_global  / 2 ) ? (x[3] - LZ_global)  : x[3];
// }

// __device__ void site_map_zerohalf (int xv[4], int const x[4], Geom global_geom ) {
//   int T_global = global_geom.T;
//   int LX_global = global_geom.LX;
//   int LY_global = global_geom.LY;
//   int LZ_global = global_geom.LZ;
//   xv[0] = ( x[0] > T_global   / 2 ) ? x[0] - T_global   : (  ( x[0] < T_global   / 2 ) ? x[0] : 0 );
//   xv[1] = ( x[1] > LX_global  / 2 ) ? x[1] - LX_global  : (  ( x[1] < LX_global  / 2 ) ? x[1] : 0 );
//   xv[2] = ( x[2] > LY_global  / 2 ) ? x[2] - LY_global  : (  ( x[2] < LY_global  / 2 ) ? x[2] : 0 );
//   xv[3] = ( x[3] > LZ_global  / 2 ) ? x[3] - LZ_global  : (  ( x[3] < LZ_global  / 2 ) ? x[3] : 0 );
// }

__device__ int coord_map_zerohalf(int xi, int Li) {
  return (xi > Li / 2) ? xi - Li : ( (xi < Li / 2) ? xi : 0 );
}


/**
 * 4D kernels: operate over CUDA_BLOCK_SIZE^4 spinor elements each.
 */
__global__ void ker_dzu_dzsu(
    double* d_dzu, double* d_dzsu, const double* fwd_src, const double* fwd_y,
    int iflavor, Coord g_proc_coords, Coord gsx, IdxComb idx_comb,
    Geom global_geom, Geom local_geom) {
  Coord origin = get_thread_origin(local_geom);
  int gsx_arr[4] = {gsx.t, gsx.x, gsx.y, gsx.z};
  size_t VOLUME = local_geom.T * local_geom.LX * local_geom.LY * local_geom.LZ;
  int local_geom_arr[4] = {local_geom.T, local_geom.LX, local_geom.LY, local_geom.LZ};
  int global_geom_arr[4] = {global_geom.T, global_geom.LX, global_geom.LY, global_geom.LZ};
  int proc_coord_arr[4] = {g_proc_coords.t, g_proc_coords.x, g_proc_coords.y, g_proc_coords.z};
  double dzu_work[6 * 12 * 12 * 2] = { 0 };
  double dzsu_work[6 * 12 * 12 * 2] = { 0 };
  double spinor_work_0[24] = { 0 };
  double spinor_work_1[24] = { 0 };
  for (int ia = 0; ia < 12; ++ia) {
    for (int k = 0; k < 6; ++k) {
      const int sigma = idx_comb.comb[k][1];
      const int rho = idx_comb.comb[k][0];
      const double* fwd_base = &fwd_src[_GSI(VOLUME) * (iflavor * 12 + ia)];
      for (int dt = 0; dt < BS; ++dt) {
        for (int dx = 0; dx < BS; ++dx) {
          for (int dy = 0; dy < BS; ++dy) {
            for (int dz = 0; dz < BS; ++dz) {
              const int tt = dt + origin.t;
              const int xx = dx + origin.x;
              const int yy = dy + origin.y;
              const int zz = dz + origin.z;
              if (tt >= local_geom.T || xx >= local_geom.LX ||
                  yy >= local_geom.LY || zz >= local_geom.LZ) {
                continue;
              }
              const Coord coord{
                .t = tt, .x = xx, .y = yy, .z = zz
              };
              size_t iz = coord2lexic(coord, local_geom);
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
                  dzu_work[((k * 12 + ia) * 12 + ib) * 2 + 0] += fwd_y_re * s_re + fwd_y_im * s_im;
                  dzu_work[((k * 12 + ia) * 12 + ib) * 2 + 1] += fwd_y_re * s_im - fwd_y_im * s_re;
                }
              }
            }
          }
        }
      } // end vol loop

      // reduce (TODO faster reduce algo?)
      for (int ib = 0; ib < 12; ++ib) {
        int ind = ((k * 12 + ia) * 12 + ib) * 2;
        atomicAdd_system(&d_dzu[ind], dzu_work[ind]);
        atomicAdd_system(&d_dzu[ind+1], dzu_work[ind+1]);
      }
    }

    for (int sigma = 0; sigma < 4; ++sigma) {
      const double* fwd_base = &fwd_src[_GSI(VOLUME) * (iflavor * 12 + ia)];
      for (int ib = 0; ib < 12; ++ib) {
        for (int dt = 0; dt < BS; ++dt) {
          for (int dx = 0; dx < BS; ++dx) {
            for (int dy = 0; dy < BS; ++dy) {
              for (int dz = 0; dz < BS; ++dz) {
                const int tt = dt + origin.t;
                const int xx = dx + origin.x;
                const int yy = dy + origin.y;
                const int zz = dz + origin.z;
                if (tt >= local_geom.T || xx >= local_geom.LX ||
                    yy >= local_geom.LY || zz >= local_geom.LZ) {
                  continue;
                }
                const Coord coord{
                  .t = tt, .x = xx, .y = yy, .z = zz
                };
                size_t iz = coord2lexic(coord, local_geom);
                const double* _u = &fwd_base[_GSI(iz)];
                double* _t = spinor_work_0;
                _fv_eq_gamma_ti_fv(_t, sigma, _u);
                _fv_ti_eq_g5(_t);

                for (int i = 0; i < 12; ++i) {
                  double fwd_y_re = fwd_y[((1-iflavor) * 12 + ib) * _GSI(VOLUME) + _GSI(iz) + 2*i];
                  double fwd_y_im = fwd_y[((1-iflavor) * 12 + ib) * _GSI(VOLUME) + _GSI(iz) + 2*i+1];
                  double s_re = _t[2*i];
                  double s_im = _t[2*i+1];
                  dzsu_work[((sigma * 12 + ia) * 12 + ib) * 2 + 0] += fwd_y_re * s_re + fwd_y_im * s_im;
                  dzsu_work[((sigma * 12 + ia) * 12 + ib) * 2 + 1] += fwd_y_re * s_im - fwd_y_im * s_re;
                }
              }
            }
          }
        } // end vol loop

        // reduce (TODO faster reduce algo?)
        int ind = ((sigma * 12 + ia) * 12 + ib) * 2;
        atomicAdd_system(&d_dzsu[ind], dzsu_work[ind]);
        atomicAdd_system(&d_dzsu[ind+1], dzsu_work[ind+1]);
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
    int iflavor, Coord proc_coords, Coord gsx, IdxComb idx_comb, Geom global_geom, Geom local_geom) {
  int T = local_geom.T;
  int LX = local_geom.LX;
  int LY = local_geom.LY;
  int LZ = local_geom.LZ;
  const size_t BS_TX = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE * CUDA_BLOCK_SIZE;
  const size_t BS_Y = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE;
  const size_t BS_Z = CUDA_THREAD_DIM_4D * CUDA_BLOCK_SIZE;
  size_t nx = (T*LX + BS_TX - 1) / BS_TX;
  size_t ny = (LY + BS_Y - 1) / BS_Y;
  size_t nz = (LZ + BS_Z - 1) / BS_Z;
  dim3 kernel_nblocks(nx, ny, nz);
  dim3 kernel_nthreads(CUDA_THREAD_DIM_4D, CUDA_THREAD_DIM_4D, CUDA_THREAD_DIM_4D);
  ker_dzu_dzsu<<<kernel_nblocks, kernel_nthreads>>>(
      d_dzu, d_dzsu, fwd_src, fwd_y, iflavor, proc_coords, gsx, idx_comb,
      global_geom, local_geom);
}


