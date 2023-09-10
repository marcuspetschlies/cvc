#ifndef CUDA_LATTICE_H
#define CUDA_LATTICE_H

#include <cuda_runtime.h>
#include <cstdio>
#include "KQED.h"

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
            (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

struct IdxComb {
  int comb[6][2];
};

struct Geom {
  int T, LX, LY, LZ;
};

struct Coord {
  int t, x, y, z;
};

struct Pair {
  double a, b;
};

/**
 * NOTE: 3D kernel must act over MPI subvolume T x LX x LY x LZ. We map this into
 * 3D kernel launches by taking (x, y, z) = global thread (x*BS % LX, y*BS, z*BS)
 * coord with t = global thread (x*BS / LX)*BS. Then each thread gets a local
 * volume of BS^4 to operate on.
 */
#define CUDA_BLOCK_SIZE 4
#define CUDA_THREAD_DIM_1D 32
#define CUDA_THREAD_DIM_4D 4

void cu_spinor_field_eq_gamma_ti_spinor_field(
    double* out, int mu, const double* in, size_t len);
void cu_g5_phi(double* out, size_t len);
void cu_dzu_dzsu(
    double* d_dzu, double* d_dzsu, const double* fwd_src, const double* fwd_y,
    int iflavor, Coord proc_coords, Coord gsx, IdxComb idx_comb,
    Geom global_geom, Geom local_geom);
void cu_4pt_contraction(
    double* d_kernel_sum, const double* d_g_dzu, const double* d_g_dzsu,
    const double* fwd_src, const double* fwd_y, int iflavor, Coord proc_coords,
    Coord gsx, Pair xunit, Coord yv, IdxComb idx_comb, QED_kernel_temps kqed_t,
    Geom global_geom, Geom local_geom);


/**
 * Simple interface to KQED.
 */
void
cu_pt_QED_kernel_L0(
    const double xv[4] , const double yv[4] ,
    const struct QED_kernel_temps t , double kerv[6][4][4][4] );
void
cu_pt_QED_kernel_L1(
    const double xv[4] , const double yv[4] ,
    const struct QED_kernel_temps t , double kerv[6][4][4][4] );
void
cu_pt_QED_kernel_L2(
    const double xv[4] , const double yv[4] ,
    const struct QED_kernel_temps t , double kerv[6][4][4][4] );
void
cu_pt_QED_kernel_L3(
    const double xv[4] , const double yv[4] ,
    const struct QED_kernel_temps t , double kerv[6][4][4][4] );


#endif // CUDA_LATTICE_H
