/****************************************************
 * test_slaph_baryon_2pt.c
 *
 * Tue Sep  6 14:05:47 CEST 2016
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
#include "set_default.h"
#include "mpi_init.h"
#include "io.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "gauge_io.h"
#include "read_input_parser.h"
#include "laplace_linalg.h"
#include "hyp_smear.h"
#include "laphs_io.h"
#include "laphs_utils.h"
#include "laphs.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, mu, nu, status, sid;
  int it_src = 1;
  int is_src = 2;
  int iv_src = 3;
  int i, j, k, ncon=-1, is, idx;
  int filename_set = 0;
  int x0, x1, x2, x3, ix, iix;
  int y0, y1, y2, y3;
  int threadid, nthreads;
  int no_fields = 2;
  double dtmp;

  double plaq=0.;
  double *gauge_field_smeared = NULL;
  int verbose = 0;
  char filename[200];
  FILE *ofs=NULL;
  double v1[6], v2[6];
  size_t items, bytes;
  complex w, w1, w2;
  double **perambulator = NULL;
  double ratime, retime;
  eigensystem_type es;
  randomvector_type rv[3];
  tripleV_type tripleV;
  perambulator_type peram[3];
  double *momentum_phase = NULL;
  int momentum[3] = {0,0,0};
  int sink_timeslice, source_timeslice;
  unsigned int VOL3;

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
      usage();
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  if(g_cart_id==0) fprintf(stdout, "# [test_slaph_baryon_2pt] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg_xspace] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  status = tmLQCD_invert_init(argc, argv, 1);
  if(status != 0) {
    EXIT(14);
  }
  status = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(status != 0) {
    EXIT(15);
  }
  status = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(status != 0) {
    EXIT(16);
  }
#endif

  /* initialize MPI parameters */
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
    fprintf(stderr, "[test_slaph_baryon_2pt] ERROR from init_geometry\n");
    exit(101);
  }

  geometry();

  VOL3 = LX*LY*LZ;

  /* initialize eigensystem */
  init_eigensystem(&es);


  /* initialize 3 perambulators */
  init_perambulator(&peram[0]);
  init_perambulator(&peram[1]);
  init_perambulator(&peram[2]);

  /* initialize 3 random vetors */
  init_randomvector(&rv[0]);
  init_randomvector(&rv[1]);
  init_randomvector(&rv[2]);

  /* initialize the eigensystem */
  status = alloc_eigensystem (&es, T, laphs_eigenvector_number);
  if(status != 0) {
    fprintf(stderr, "[test_slaph_baryon_2pt] Error from alloc_eigensystem, status was %d\n", status);
    EXIT(7);
  }

  /* read eigensystem */
  ratime = _GET_TIME;
  status = read_eigensystem(&es);
  if (status != 0) {
    fprintf(stderr, "# [test_slaph_baryon_2pt] Error from read_eigensystem, status was %d\n", status);
  }
  retime = _GET_TIME;
  fprintf(stdout, "# [test_slaph_baryon_2pt] time to read eigensystem %e\n", retime-ratime);


  /* init reduced VVV */
  init_tripleV (&tripleV, 1, 1, laphs_eigenvector_number);

  /************************************************************
   * for now we test with fixed
   */
  sink_timeslice   = 2;
  source_timeslice = 0;
  /*
   * in production, we must have a loop on source 
   * and sink timeslice
   ************************************************************/

/*
 * TEST
  ratime = _GET_TIME;
  status = test_eigensystem(&es, g_gauge_field);
  retime = _GET_TIME;
  fprintf(stdout, "# [test_slaph_baryon_2pt] time to test eigensystem %e\n", retime-ratime);
*/

  /* allocate 3 random vectors */
  status = alloc_randomvector(&rv[0], T, 4, laphs_eigenvector_number);
  status = status || alloc_randomvector(&rv[1], T, 4, laphs_eigenvector_number);
  status = status || alloc_randomvector(&rv[2], T, 4, laphs_eigenvector_number);
  if(status != 0) {
    fprintf(stderr, "[] Error from alloc_randomvector\n");
    EXIT(4);
  }

  /* read random vector */
  status = 0;
  status = status || read_randomvector(&rv[0], "u", 0);
  status = status || read_randomvector(&rv[1], "u", 1);
  status = status || read_randomvector(&rv[2], "u", 2);
  if(status != 0) {
    fprintf(stderr, "[] Error from read_randomvector\n");
    EXIT(4);
  }

  /* allocate 3 perambulators */
  status = 0;
  status = status || alloc_perambulator(&peram[0], laphs_time_src_number, laphs_spin_src_number, laphs_evec_src_number, T, 4, laphs_eigenvector_number, 3, "u", "smeared1", 0 );
  status = status || alloc_perambulator(&peram[1], laphs_time_src_number, laphs_spin_src_number, laphs_evec_src_number, T, 4, laphs_eigenvector_number, 3, "u", "smeared1", 1 );
  status = status || alloc_perambulator(&peram[2], laphs_time_src_number, laphs_spin_src_number, laphs_evec_src_number, T, 4, laphs_eigenvector_number, 3, "u", "smeared1", 2 );
  if(status != 0) {
    fprintf(stderr, "[test_slaph_baryon_2pt] Error from read_perambulator, status was %d\n", status);
    EXIT(12);
  }

  /* TEST */
  print_perambulator_info (&peram[0]);

  /* read perambulators */
  status = 0;
  status = status || read_perambulator(&peram[0]);
  status = status || read_perambulator(&peram[1]);
  status = status || read_perambulator(&peram[2]);
  if(status != 0) {
    fprintf(stderr, "[test_slaph_baryon_2pt] Error from read_perambulator, status was %d\n", status);
    exit(14);
  }

  /*****************************************************************
   * now everything is read for eigenvector id triplet 0, 1, 2
   * - start reduction
   *****************************************************************/

  /* set a momentum phase for Fourier Transform
   *
   * for the test, momentum was set to zero at the top
   * */

  momentum_phase = (double*)malloc(VOL3*2*sizeof(double));
  if(momentum_phase == NULL) {
    fprintf(stderr, "[] Error from malloc\n");
    EXIT(15);
  }

  for(x1=0; x1<LX; x1++) {
  for(x2=0; x2<LY; x2++) {
  for(x3=0; x3<LZ; x3++) {
    ix  = g_ipt[0][x1][x2][x3];
    dtmp = 2. * M_PI * (
        (x1 + g_proc_coords[1]*LX) * momentum[0] / (double)LX_global +
        (x2 + g_proc_coords[2]*LY) * momentum[1] / (double)LY_global +
        (x3 + g_proc_coords[3]*LZ) * momentum[2] / (double)LZ_global );

    momentum_phase[2*ix  ] = cos(dtmp);
    momentum_phase[2*ix+1] = sin(dtmp);
  }}}


  /* (1) reduce VVV */
  status = reduce_triple_eigensystem_timeslice ( &tripleV, &es, sink_timeslice, 0, momentum_phase);

  /* reduce stepwise VVV x p x p x r */



  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  /* fini reduced VVV */
  fini_tripleV (&tripleV);

  /* fini eigensystem */
  fini_eigensystem (&es);

  /* fini random vectors */
  fini_randomvector(&rv[0]);
  fini_randomvector(&rv[1]);
  fini_randomvector(&rv[2]);

  /* fini perambulators */
  fini_perambulator(&peram[0]);
  fini_perambulator(&peram[1]);
  fini_perambulator(&peram[2]);

  free(momentum_phase);

  free_geometry();

  g_the_time = time(NULL);
  fprintf(stdout, "# [test_slaph_baryon_2pt] %s# [test_slaph_baryon_2pt] end fo run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "# [test_slaph_baryon_2pt] %s# [test_slaph_baryon_2pt] end fo run\n", ctime(&g_the_time));
  fflush(stderr);


#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  return(0);
}

