/****************************************************
 * contract_loop.cpp
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#ifdef HAVE_HDF5
#include <hdf5.h>
#endif

#include "cvc_complex.h"
#include "iblas.h"
#include "ilinalg.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "table_init_d.h"
#include "table_init_z.h"
#include "project.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "contract_cvc_tensor.h"
#include "scalar_products.h"
#include "contract_loop_inline.h"

#define MAX_SUBGROUP_NUMBER 20

namespace cvc {

/***********************************************************
 * local loop contractions
 ***********************************************************/
int contract_local_loop_stochastic ( double *** const loop, double * const source, double * const prop ) {

  unsigned int const VOL3 = LX * LY * LZ;

  for ( int x0 = 0; x0 < T; x0++ ) {
    for ( int ix = 0; ix < VOL3; ix++ ) {
      unsigned int const iix = x0 * VOL3 + ix;
      unsigned int const offset = _GSI ( iix );
      double * const source_ = source + offset;
      double * const prop_   = prop   + offset;
      _contract_loop_x_spin_diluted ( loop[it][ix], source_ , prop_ );
    }  /* end of loop on volume */
  }  /* end of loop on timeslices */

}  /* end of contract_local_loop_stochastic */

/***********************************************************/
/***********************************************************/

}  /* end of namespace cvc */
