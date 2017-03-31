/****************************************************
 * test_sliced.cpp 
 *
 * Do 30. Jun 08:19:01 CEST 2016
 *
 * PURPOSE:
 * TODO:
 * DONE:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#  ifdef HAVE_TMLQCD_LIBWRAPPER
#    include "tmLQCD.h"
#  endif

#ifdef __cplusplus
}
#endif


#define MAIN_PROGRAM

#include "types.h"
#include "cvc_complex.h"
#include "ilinalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "read_input_parser.h"
#include "ranlxd.h"
#include "scalar_products.h"


using namespace cvc;

int main(int argc, char **argv) {
  
  int c, exitstatus;
  int i, j;
  int filename_set = 0;
  int x0, x1, x2, x3, ix, iix;
  int ix_even, ix_odd, y0, y1, y2, y3;
  /* int start_valuet=0, start_valuex=0, start_valuey=0; */
  /* int threadid, nthreads; */
  /* double diff1, diff2; */
  double plaq=0;
  double spinor1[24], spinor2[24];
  complex w, w2;
  int verbose = 0;
  char filename[200];
  FILE *ofs=NULL;
  double norm, norm2;
  unsigned int Vhalf, VOL3;
  double **eo_spinor_field = NULL;
  int no_eo_fields;


#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif


  while ((c = getopt(argc, argv, "h?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'h':
    case '?':
    default:
      exit(0);
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "apply_Dtm.input");
  if(g_cart_id==0) fprintf(stdout, "# Reading input from file %s\n", filename);
  read_input_parser(filename);


  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);


  /* initialize T etc. */
  fprintf(stdout, "# [%2d] parameters:\n"\
                  "# [%2d] T_global     = %3d\n"\
                  "# [%2d] T            = %3d\n"\
		  "# [%2d] Tstart       = %3d\n"\
                  "# [%2d] LX_global    = %3d\n"\
                  "# [%2d] LX           = %3d\n"\
		  "# [%2d] LXstart      = %3d\n"\
                  "# [%2d] LY_global    = %3d\n"\
                  "# [%2d] LY           = %3d\n"\
		  "# [%2d] LYstart      = %3d\n",\
		  g_cart_id, g_cart_id, T_global, g_cart_id, T, g_cart_id, Tstart,
		             g_cart_id, LX_global, g_cart_id, LX, g_cart_id, LXstart,
		             g_cart_id, LY_global, g_cart_id, LY, g_cart_id, LYstart);

  if(init_geometry() != 0) {
    fprintf(stderr, "ERROR from init_geometry\n");
    exit(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

  Vhalf = VOLUME / 2;
  VOL3 = LX * LY * LZ;

  for(ix=0; ix<Vhalf; ix++) {
    x0 = g_eosub2t[0][ix];
    fprintf(stdout, "proc%.4d e2sliced\t%8d\t%3d\t%8d\n", g_cart_id, ix, x0, g_eosub2sliced3d[0][ix]);
  }

  for(ix=0; ix<Vhalf; ix++) {
    x0 = g_eosub2t[1][ix];
    fprintf(stdout, "proc%.4d o2sliced\t%8d\t%3d\t%8d\n", g_cart_id, ix, x0, g_eosub2sliced3d[1][ix]);
  }

  for(x0=0; x0<T; x0++) {
    for(ix=0; ix<VOL3/2; ix++) {
      fprintf(stdout, "proc%.4d sliced2e\t%8d\t%3d\t%8d\n", g_cart_id, ix, x0, g_sliced3d2eosub[0][x0][ix]);
    }
  }

  for(x0=0; x0<T; x0++) {
    for(ix=0; ix<VOL3/2; ix++) {
      fprintf(stdout, "proc%.4d sliced2o\t%8d\t%3d\t%8d\n", g_cart_id, ix, x0, g_sliced3d2eosub[1][x0][ix]);
    }
  }

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  free_geometry();

  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_invert] %s# [test_invert] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_invert] %s# [test_invert] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }


#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}
