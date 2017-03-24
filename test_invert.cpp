/****************************************************
 * test_invert.cpp 
 *
 * Fr 8. Jan 15:38:18 CET 2016
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
#include "scalar_products.h"


using namespace cvc;

int main(int argc, char **argv) {
  
  int c, mu, nu, exitstatus;
  int i, j;
  int filename_set = 0;
  int x0, x1, x2, x3, ix, iix;
  int y0, y1, y2, y3;
  /* int start_valuet=0, start_valuex=0, start_valuey=0; */
  /* int threadid, nthreads; */
  /* double diff1, diff2; */
  double plaq=0;
  /* double spinor1[24], spinor2[24]; */
  double *pl_gather=NULL;
  /* complex prod, w; */
  int verbose = 0;
  char filename[200];
  /* FILE *ofs=NULL; */
  double norm, norm2;
  unsigned int VOL3;
  int op_id = 0;


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


#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_invert] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1);
  if(exitstatus != 0) {
    EXIT(14);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    EXIT(15);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(exitstatus != 0) {
    EXIT(16);
  }
#endif

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
    fprintf(stderr, "Error from init_geometry\n");
    exit(101);
  }

  geometry();

  VOL3 = LX*LY*LZ;

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [test_invert] reading gauge field from file %s\n", filename);
      read_lime_gauge_field_doubleprec(filename);
    } else {
      /* initialize unit matrices */
      if(g_cart_id==0) fprintf(stdout, "\n# [test_invert] initializing unit matrices\n");
      for(ix=0;ix<VOLUME;ix++) {
        _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
        _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
      }
    }
#else
   Nconf = g_tmLQCD_lat.nstore;
   if(g_cart_id== 0) fprintf(stdout, "[test_invert] Nconf = %d\n", Nconf);

   exitstatus = tmLQCD_read_gauge(Nconf);
   if(exitstatus != 0) {
     EXIT(3);
   }

   exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
   if(exitstatus != 0) {
     EXIT(4);
   }
   if(&g_gauge_field == NULL) {
     fprintf(stderr, "[test_invert] Error, &g_gauge_field is NULL\n");
     EXIT(5);
   }
#endif

#ifdef HAVE_MPI
   xchange_gauge();
#endif



  // measure the plaquette
  if(g_cart_id==0) fprintf(stdout, "# read plaquette value 1st field: %25.16e\n", plaq);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# measured plaquette value 1st field: %25.16e\n", plaq);

  no_fields=3;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);

  /****************************************
   * set a point source 
   ****************************************/
  memset(g_spinor_field[0], 0, 24*VOLUME*sizeof(double));

  strcpy(filename, filename_prefix3);
  exitstatus = read_lime_spinor(g_spinor_field[0], filename, 0);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_invert] Error from read_lime_spinor, status was %d\n", exitstatus);
    EXIT(14);
  }
  xchange_field(g_spinor_field[0]);


  /****************************************
   * invert for flavor type
   *   g_propagator_position
   ****************************************/
  op_id = 0;
  exitstatus = tmLQCD_invert(g_spinor_field[1], g_spinor_field[0], op_id, 0);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_invert] Error from tmLQCD_invert, status was %d\n", exitstatus);
    EXIT(12);
  }

  sprintf(filename, "%s.inverted", filename_prefix3);
  exitstatus = write_propagator(g_spinor_field[1], filename, 0, 64);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_invert] Error from write_propagator, status was %d\n", exitstatus);
    EXIT(15);
  }

#if 0
  /****************************************
   * invert for flavor type
   *   1 - g_propagator_position
   ****************************************/

  op_id = g_propagator_position;
  exitstatus = tmLQCD_invert(g_spinor_field[2], g_spinor_field[0], op_id, 0);

  if(exitstatus != 0) {
    fprintf(stderr, "[test_invert] Error from tmLQCD_invert, status was %d\n", exitstatus);
    EXIT(12);
  }
#endif

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  free_geometry();
  if(g_spinor_field != NULL) {
    for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
    free(g_spinor_field);
  }

  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_invert] %s# [test_invert] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_invert] %s# [test_invert] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }


#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

