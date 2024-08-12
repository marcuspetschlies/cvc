#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include "ranlxd.h"
#include "global.h"
#include "cvc_utils.h"

#include <random>

namespace cvc {
#if ( defined DUMMY_SOLVER ) || ( ! defined HAVE_TMLQCD_LIBWRAPPER )
/************************************************************************************
 * dummy solver
 ************************************************************************************/
inline void hash_combine(std::size_t& seed, const double& v)
{
  std::hash<double> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}
inline void hash_spinor(std::size_t& seed, double * const x)
{
  for (unsigned int ix = 0; ix < VOLUME; ix++)
  {
    for (int ia = 0; ia < 24; ia++)
    {
      hash_combine(seed, x[_GSI(ix)+ia]);
    }
  }
}
int dummy_solver ( double * const propagator, double * const source, int const op_id ) {
  // V3: Seeded gaussian random numbers (identical for identical sources)
  //     Note that different MPI geometries will give different results
  int exitstatus;
  std::mt19937_64 rng;
  std::size_t source_seed = 0;
  hash_spinor(source_seed, source);
  unsigned long long ss = (unsigned long long) source_seed;
  int n_rank;
#ifdef HAVE_MPI
  exitstatus = MPI_Comm_size(g_cart_grid, &n_rank);
  if (exitstatus != MPI_SUCCESS)
  {
    fprintf(stderr, "[dummy_solver] MPI_Comm_size err\n");
    EXIT(54);
  }
#endif
  std::vector<unsigned long long> ss_all(n_rank);
#ifdef HAVE_MPI
  exitstatus = MPI_Allgather(&ss, 1, MPI_UNSIGNED_LONG_LONG, ss_all.data(), 1, MPI_UNSIGNED_LONG_LONG, g_cart_grid);
  if (exitstatus != MPI_SUCCESS)
  {
    fprintf(stderr, "[dummy_solver] MPI_Allgather err\n");
    EXIT(55);
  }
#endif
  ss_all.push_back(g_cart_id);
  ss_all.push_back(op_id);
  std::seed_seq seed(ss_all.begin(), ss_all.end());
  rng.seed(seed);
  std::normal_distribution<double> dist;
  for (unsigned int ix = 0; ix < VOLUME; ix++)
  {
    for (int ia = 0; ia < 24; ia++)
    {
      propagator[_GSI(ix) + ia] = dist(rng);
    }
  }
  return 0;
  // V2: Gaussian random numbers (different for identical sources)
  // return( rangauss(propagator, _GSI(VOLUME) ) );
  // V1: All zeros
  // memcpy ( propagator, source, _GSI(VOLUME)*sizeof(double) );
  // return ( 0 ) ;
}  /* end of dummy_solver */
#endif
}  /* end of namespace cvc */
