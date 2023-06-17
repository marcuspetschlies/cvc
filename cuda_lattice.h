#ifndef CUDA_LATTICE_H
#define CUDA_LATTICE_H

#include <cuda_runtime.h>

struct IdxComb {
  int comb[6][2];
};

struct Geom {
  int T, LX, LY, LZ;
};

struct Coord {
  int t, x, y, z;
};

/**
 * NOTE: 3D kernel must act over MPI subvolume T x LX x LY x LZ. We map this into
 * 3D kernel launches by taking (x, y, z) = global thread (x*BS % LX, y*BS, z*BS)
 * coord with t = global thread (x*BS / LX)*BS. Then each thread gets a local
 * volume of BS^4 to operate on.
 */
#define CUDA_BLOCK_SIZE 4


/**
 * 1D kernels: operate over CUDA_BLOCK_SIZE spinor elements each.
 *  - `len`: num *doubles* in the input/output array (must be divisible by 24)
 */
__global__ void cu_spinor_field_eq_gamma_ti_spinor_field(double* out, const double* in, size_t len);
__global__ void cu_g5_phi(double* out, const double* in, size_t len);


#endif // CUDA_LATTICE_H
