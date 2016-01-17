/****************************************************
 * test_laph.c
 *
 * Sun May 31 17:05:41 CEST 2015
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
#ifdef MPI
#  include <mpi.h>
#endif
#ifdef OPENMP
#include <omp.h>
#endif

#define MAIN_PROGRAM

#include "types.h"
#include "cvc_complex.h"
#include "ilinalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "set_default_input.h"
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
  int i, j, ncon=-1, is, idx;
  int filename_set = 0;
  int x0, x1, x2, x3, ix, iix;
  int y0, y1, y2, y3;
  int threadid, nthreads;
  double dtmp[4], norm, norm2, norm3;

  double plaq=0.;
  double *gauge_field_smeared = NULL;
  int verbose = 0;
  char filename[200];
  FILE *ofs=NULL;
  unsigned int VOL3;
  double v1[6], v2[6];
  size_t items, bytes;
  complex w, w1, w2;
  double **perambulator = NULL;
  eigensystem_type es;
  randomvector_type rv, prv;
  perambulator_type peram;

#ifdef MPI
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
  if(g_cart_id==0) fprintf(stdout, "# [test_laph] Reading input from file %s\n", filename);
  read_input_parser(filename);


  /* some checks on the input data */
  if((T_global == 0) || (LX==0) || (LY==0) || (LZ==0)) {
    if(g_proc_id==0) fprintf(stdout, "# [test_laph] T and L's must be set\n");
    usage();
  }

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
    fprintf(stderr, "[test_laph] ERROR from init_geometry\n");
    exit(101);
  }

  geometry();

  VOL3 = LX*LY*LZ;

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [test_laph] reading gauge field from file %s\n", filename);

  if(strcmp(gaugefilename_prefix,"identity")==0) {
    status = unit_gauge_field(g_gauge_field, VOLUME);
  } else {
    // status = read_nersc_gauge_field_3x3(g_gauge_field, filename, &plaq);
    // status = read_ildg_nersc_gauge_field(g_gauge_field, filename);
    status = read_lime_gauge_field_doubleprec(filename);
    // status = read_nersc_gauge_field(g_gauge_field, filename, &plaq);
  }
  if(status != 0) {
    fprintf(stderr, "[test_laph] Error, could not read gauge field\n");
    exit(11);
  }
  // measure the plaquette
  if(g_cart_id==0) fprintf(stdout, "# [test_laph] read plaquette value 1st field: %25.16e\n", plaq);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [test_laph] measured plaquette value 1st field: %25.16e\n", plaq);

#if 0
  /* smear the gauge field */
  status = hyp_smear_3d (g_gauge_field, N_hyp, alpha_hyp, 0, 0);
  if(status != 0) {
    fprintf(stderr, "[test_laph] Error from hyp_smear_3d, status was %d\n", status);
    EXIT(7);
  }

  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [test_laph] measured plaquette value ofter hyp smearing = %25.16e\n", plaq);

  sprintf(filename, "%s_hyp.%.4d", gaugefilename_prefix, Nconf);
  fprintf(stdout, "# [test_laph] writing hyp-smeared gauge field to file %s\n", filename);

  status = write_lime_gauge_field(filename, plaq, Nconf, 64);
  if(status != 0) {
    fprintf(stderr, "[apply_lapace] Error friom write_lime_gauge_field, status was %d\n", status);
    EXIT(7);
  }
#endif

  init_eigensystem(&es);
  init_perambulator(&peram);
  init_randomvector(&rv);

  status = alloc_eigensystem (&es, T, laphs_eigenvector_number);
  if(status != 0) {
    fprintf(stderr, "[test_laph] Error from alloc_eigensystem, status was %d\n", status);
    EXIT(7);
  }

  status = alloc_randomvector(&rv, T, 4, laphs_eigenvector_number);
  status = alloc_randomvector(&prv, T, 4, laphs_eigenvector_number);
  status = read_randomvector(&rv, "u", 0);

  status = project_randomvector (&prv, &rv, 1, 2, 3);
  if(status != 0) {
    fprintf(stderr, "[test_laph] Error from project_randomvector, status was %d\n", status);
    EXIT(8);
  }

  print_randomvector(&rv, stdout);

  print_randomvector(&prv, stdout);



  status = alloc_perambulator(&peram, laphs_time_src_number, laphs_spin_src_number, laphs_evec_src_number, T, 4, laphs_eigenvector_number, 3, "u", "smeared1", 0 );

  print_perambulator_info (&peram);

  status = read_perambulator(&peram);
  if(status != 0) {
    fprintf(stderr, "[test_laph] Error from read_perambulator, status was %d\n", status);
    exit(12);
  }
#if 0
#endif

/*
  status = read_eigensystem(&es);
  if (status != 0) {
    fprintf(stderr, "# [test_laph] Error from read_eigensystem, status was %d\n", status);
  }
  status = test_eigensystem(&es, g_gauge_field);
*/

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  fini_eigensystem (&es);
  fini_randomvector(&rv);
  fini_perambulator(&peram);


  free(g_gauge_field);
  free_geometry();

  g_the_time = time(NULL);
  fprintf(stdout, "# [test_laph] %s# [test_laph] end fo run\n", ctime(&g_the_time));
  fflush(stdout);
  fprintf(stderr, "# [test_laph] %s# [test_laph] end fo run\n", ctime(&g_the_time));
  fflush(stderr);


#ifdef MPI
  MPI_Finalize();
#endif
  return(0);
}

