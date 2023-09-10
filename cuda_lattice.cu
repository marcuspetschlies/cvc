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
    double* dzu, double* dzsu, const double* fwd_src, const double* fwd_y,
    int iflavor, Coord g_proc_coords, Coord gsx, IdxComb idx_comb,
    Geom global_geom, Geom local_geom) {
  Coord origin = get_thread_origin(local_geom);
  int gsx_arr[4] = {gsx.t, gsx.x, gsx.y, gsx.z};
  size_t VOLUME = local_geom.T * local_geom.LX * local_geom.LY * local_geom.LZ;
  int local_geom_arr[4] = {local_geom.T, local_geom.LX, local_geom.LY, local_geom.LZ};
  int global_geom_arr[4] = {global_geom.T, global_geom.LX, global_geom.LY, global_geom.LZ};
  int proc_coord_arr[4] = {g_proc_coords.t, g_proc_coords.x, g_proc_coords.y, g_proc_coords.z};
  double dzu_work[6 * 12 * 12 * 2] = { 0 };
  double dzsu_work[4 * 12 * 12 * 2] = { 0 };
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
        atomicAdd_system(&dzu[ind], dzu_work[ind]);
        atomicAdd_system(&dzu[ind+1], dzu_work[ind+1]);
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
        atomicAdd_system(&dzsu[ind], dzsu_work[ind]);
        atomicAdd_system(&dzsu[ind+1], dzsu_work[ind+1]);
      }
    }
  }
}

typedef void (*QED_kernel_LX_ptr)( const double xv[4], const double yv[4], const struct QED_kernel_temps t, double kerv[6][4][4][4] );

__global__
void ker_4pt_contraction(
    double* kernel_sum, const double* g_dzu, const double* g_dzsu,
    const double* fwd_src, const double* fwd_y, int iflavor, Coord g_proc_coords,
    Coord gsx, Pair xunit, Coord yv, IdxComb idx_comb, QED_kernel_temps kqed_t,
    Geom global_geom, Geom local_geom) {

  Coord origin = get_thread_origin(local_geom);
  int gsx_arr[4] = {gsx.t, gsx.x, gsx.y, gsx.z};
  size_t VOLUME = local_geom.T * local_geom.LX * local_geom.LY * local_geom.LZ;
  int local_geom_arr[4] = {local_geom.T, local_geom.LX, local_geom.LY, local_geom.LZ};
  int global_geom_arr[4] = {global_geom.T, global_geom.LX, global_geom.LY, global_geom.LZ};
  int proc_coord_arr[4] = {g_proc_coords.t, g_proc_coords.x, g_proc_coords.y, g_proc_coords.z};

  double corr_I[6 * 4 * 4 * 4 * 2];
  double corr_II[6 * 4 * 4 * 4 * 2];
  double dxu[4 * 12 * 12 * 2];
  double g_dxu[4 * 4 * 12 * 12 * 2];

  double kernel_sum_work[4] = { 0 };
  double spinor_work[24];
  double kerv1[6][4][4][4] KQED_ALIGN ;
  double kerv2[6][4][4][4] KQED_ALIGN ;
  double kerv3[6][4][4][4] KQED_ALIGN ;

  QED_kernel_LX_ptr KQED_LX[4] = {
    QED_kernel_L0,
    QED_kernel_L1,
    QED_kernel_L2,
    QED_kernel_L3 };
  
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
          size_t ix = coord2lexic(coord, local_geom);

          int coord_arr[4] = {tt, xx, yy, zz};
          int xv[4], xvzh[4];
          for (int rho = 0; rho < 4; ++rho) {
            int xrho = coord_arr[rho] + proc_coord_arr[rho] * local_geom_arr[rho] - gsx_arr[rho];
            xrho = (xrho + global_geom_arr[rho]) % global_geom_arr[rho];
            xv[rho] = coord_map(xv[rho], global_geom_arr[rho]);
            xvzh[rho] = coord_map_zerohalf(xv[rho], global_geom_arr[rho]);
          }
          for (int ib = 0; ib < 12; ++ib) {
            const double* _u = &fwd_y[(iflavor * 12 + ib) * _GSI(VOLUME) + _GSI(ix)];
            for (int mu = 0; mu < 4; ++mu) {
              for (int ia = 0; ia < 12; ++ia) {
                const double* _d = &fwd_src[((1-iflavor) * 12 + ia) * _GSI(VOLUME) + _GSI(ix)];
                double* _t = spinor_work;
                _fv_eq_gamma_ti_fv(_t, mu, _d);
                _fv_ti_eq_g5(_t);
                dxu[((mu * 12 + ib) * 12 + ia) * 2 + 0] = 0.0;
                dxu[((mu * 12 + ib) * 12 + ia) * 2 + 1] = 0.0;
                for (int i = 0; i < 12; ++i) {
                  double _t_re = _t[2*i];
                  double _t_im = _t[2*i+1];
                  double _u_re = _u[2*i];
                  double _u_im = _u[2*i+1];
                  /* -1 factor due to (g5 gmu)^+ = -g5 gmu */
                  dxu[((mu * 12 + ib) * 12 + ia) * 2 + 0] += -(_t_re * _u_re + _t_im * _u_im);
                  dxu[((mu * 12 + ib) * 12 + ia) * 2 + 1] += -(_t_re * _u_im - _t_im * _u_re);
                }
              }
              double* _t = spinor_work;
              const double* _dxu = &dxu[(mu * 12 + ib) * 12 * 2];
              _fv_eq_gamma_ti_fv(_t, 5, _dxu);
              for (int lambda = 0; lambda < 4; ++lambda) {
                double* _g_dxu = &g_dxu[((lambda * 4 + mu) * 12 + ib) * 12 * 2];
                _fv_eq_gamma_ti_fv(_g_dxu, lambda, _t);
              }
            }
          }

          for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
              for (int lambda = 0; lambda < 4; ++lambda) {
                for (int k = 0; k < 6; ++k) {
                  const int sigma = idx_comb.comb[k][1];
                  const int rho = idx_comb.comb[k][0];
                  double *_corr_I = &corr_I[(((k * 4 + mu) * 4 + nu) * 4 + lambda) * 2];
                  double *_corr_II = &corr_II[(((k * 4 + mu) * 4 + nu) * 4 + lambda) * 2];
                  _corr_I[0] = 0.0;
                  _corr_I[1] = 0.0;
                  _corr_II[0] = 0.0;
                  _corr_II[1] = 0.0;
                  for (int ia = 0; ia < 12; ++ia) {
                    for (int ib = 0; ib < 12; ++ib) {
                      double u_re = g_dxu[(((lambda * 4 + mu) * 12 + ia) * 12 + ib) * 2];
                      double u_im = g_dxu[(((lambda * 4 + mu) * 12 + ia) * 12 + ib) * 2 + 1];
                      double v_re = g_dzu[(((k * 4 + nu) * 12 + ib) * 12 + ia) * 2];
                      double v_im = g_dzu[(((k * 4 + nu) * 12 + ib) * 12 + ia) * 2 + 1];
                      _corr_I[0] -= u_re * v_re - u_im * v_im;
                      _corr_I[1] -= u_re * v_im + u_im * v_re;
                      v_re = (
                          xvzh[rho] * g_dzsu[(((sigma * 4 + nu) * 12 + ib) * 12 + ia) * 2] -
                          xvzh[sigma] * g_dzsu[(((rho * 4 + nu) * 12 + ib) * 12 + ia) * 2] );
                      v_im = (
                          xvzh[rho] * g_dzsu[(((sigma * 4 + nu) * 12 + ib) * 12 + ia) * 2 + 1] -
                          xvzh[sigma] * g_dzsu[(((rho * 4 + nu) * 12 + ib) * 12 + ia) * 2 + 1] );
                      _corr_II[0] -= u_re * v_re - u_im * v_im;
                      _corr_II[1] -= u_re * v_im + u_im * v_re;
                    }
                  }
                  // corr_I[(((k * 4 + mu) * 4 + nu) * 4 + lambda) * 2] = -dtmp[0];
                  // corr_I[(((k * 4 + mu) * 4 + nu) * 4 + lambda) * 2 + 1] = -dtmp[1];
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

          // double * const _kerv1   = (double * const )kerv1;
          // double * const _kerv2   = (double * const )kerv2;
          // double * const _kerv3   = (double * const )kerv3;

          double * const _corr_I  = corr_I;
          double * const _corr_II = corr_II;

          double const xm_mi_ym[4] = {
            xm[0] - ym[0],
            xm[1] - ym[1],
            xm[2] - ym[2],
            xm[3] - ym[3] };

          // TODO: Better kernel parameterization / loop?
          for (int ikernel = 0; ikernel < 4; ++ikernel) {
            // atomicAdd_system(&kernel_sum[ikernel], corr_I[0]);
            // atomicAdd_system(&kernel_sum[ikernel], corr_II[0]);
            // continue; // FORNOW
            KQED_LX[ikernel]( xm, ym,       kqed_t, kerv1 );
            KQED_LX[ikernel]( ym, xm,       kqed_t, kerv2 );
            KQED_LX[ikernel]( xm, xm_mi_ym, kqed_t, kerv3 );
            double dtmp = 0.;
            int i = 0;
            for( int k = 0; k < 6; k++ ) {
              for ( int mu = 0; mu < 4; mu++ ) {
                for ( int nu = 0; nu < 4; nu++ ) {
                  for ( int lambda = 0; lambda < 4; lambda++ ) {
                    /// FORNOW
                    kerv1[k][mu][nu][lambda] = 1.0;
                    kerv2[k][nu][mu][lambda] = 1.0;
                    kerv3[k][lambda][nu][mu] = 1.0;
                    dtmp += (
                        kerv1[k][mu][nu][lambda] + kerv2[k][nu][mu][lambda]
                        - kerv3[k][lambda][nu][mu] ) * _corr_I[2*i]
                        + kerv3[k][lambda][nu][mu] * _corr_II[2*i];
                    // dtmp += _corr_I[2*i] * _corr_II[2*i];

                    i++;
                  }
                }
              }
            }
            kernel_sum_work[ikernel] += dtmp;
          }

          // FORNOW
          // for (int ikernel = 0; ikernel < 4; ++ikernel) {
          //   atomicAdd_system(&kernel_sum[ikernel], kernel_sum_work[ikernel]);
          // }
          // return;
          

        } // end coord loop
      }
    }
  }

  // reduce (TODO faster reduce algo?)
  for (int ikernel = 0; ikernel < 4; ++ikernel) {
    atomicAdd_system(&kernel_sum[ikernel], kernel_sum_work[ikernel]);
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
    int iflavor, Coord proc_coords, Coord gsx, IdxComb idx_comb,
    Geom global_geom, Geom local_geom) {
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

void cu_4pt_contraction(
    double* d_kernel_sum, const double* d_g_dzu, const double* d_g_dzsu,
    const double* fwd_src, const double* fwd_y, int iflavor, Coord proc_coords,
    Coord gsx, Pair xunit, Coord yv, IdxComb idx_comb, QED_kernel_temps kqed_t,
    Geom global_geom, Geom local_geom) {
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
  ker_4pt_contraction<<<kernel_nblocks, kernel_nthreads>>>(
      d_kernel_sum, d_g_dzu, d_g_dzsu, fwd_src, fwd_y, iflavor, proc_coords,
      gsx, xunit, yv, idx_comb, kqed_t, global_geom, local_geom);
}


/**
 * Simple interface to KQED.
 */
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

