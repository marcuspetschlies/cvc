/****************************************************
 * test_Xeo.cpp
 *
 * Fr 3. Jun 12:34:15 CEST 2016
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
#include "Q_phi.h"
#include "invert_Qtm.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  /* const double preset_eigenvalue = 7.864396614243382E-06; */
  const int evecs_num = 4;

  int c, status; 
  int i;
  int x0, x1, x2, x3;
  int filename_set = 0;
  int ix, iix, it;
  int threadid, nthreads;
  int no_eo_fields;
  double norm;
  double plaq=0.;
  double evecs_lambda;
  int verbose = 0;
  char filename[200];
  int printf_to_file = 0;

  FILE *ofs=NULL;
/*
  size_t items, bytes;
*/
  complex w;
  double **eo_spinor_field=NULL;
  double ratime, retime;
  unsigned int Vhalf;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ph?vf:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'p':
      printf_to_file = 1;
      fprintf(stdout, "# [test_Xeo] will print fields to file\n");
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
  if(g_cart_id==0) fprintf(stdout, "# [test_Xeo] Reading input from file %s\n", filename);
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
    fprintf(stderr, "[test_Xeo] ERROR from init_geometry\n");
    EXIT(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

  Vhalf = VOLUME / 2;

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "[test_Xeo] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private(i,threadid) 
{
  nthreads = omp_get_num_threads();
  threadid = omp_get_thread_num();
  fprintf(stdout, "# [test_Xeo] thread%.4d number of threads = %d\n", threadid, nthreads);
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_Xeo] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif


  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [test_Xeo] reading gauge field from file %s\n", filename);

  if(strcmp(gaugefilename_prefix,"identity")==0) {
    status = unit_gauge_field(g_gauge_field, VOLUME);
  } else {
    // status = read_nersc_gauge_field_3x3(g_gauge_field, filename, &plaq);
    // status = read_ildg_nersc_gauge_field(g_gauge_field, filename);
    status = read_lime_gauge_field_doubleprec(filename);
    // status = read_nersc_gauge_field(g_gauge_field, filename, &plaq);
  }
  if(status != 0) {
    fprintf(stderr, "[test_Xeo] Error, could not read gauge field\n");
    EXIT(11);
  }
  xchange_gauge();

  /* measure the plaquette */
  if(g_cart_id==0) fprintf(stdout, "# [test_Xeo] read plaquette value 1st field: %25.16e\n", plaq);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [test_Xeo] measured plaquette value 1st field: %25.16e\n", plaq);

  /* init and allocate spinor fields */
  no_fields = 1;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&g_spinor_field[i], VOLUME+RAND);


  no_eo_fields = 5;
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));
  for(i=0; i<no_eo_fields; i++) alloc_spinor_field(&eo_spinor_field[i], (VOLUME+RAND)/2);

  /***********************************************
   * read eo eigenvector
   ***********************************************/

  sprintf(filename, "%s%.5d", filename_prefix, 0);
  if(g_cart_id == 0) fprintf(stdout, "# [test_Xeo] reading C_oo_sym eigenvector from file %s\n", filename);

  status = read_lime_spinor(g_spinor_field[0], filename, 0);
  if( status != 0) {
    fprintf(stderr, "[test_Xeo] Error from read_lime_spinor, status was %d\n", status);
    EXIT(1);
  }

  ratime = _GET_TIME;
  spinor_field_unpack_lexic2eo (g_spinor_field[0], eo_spinor_field[0], eo_spinor_field[1]);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_Xeo] time for unpacking = %e\n", retime - ratime);

  ratime = _GET_TIME;
  xchange_eo_field( eo_spinor_field[0], 1);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_Xeo] time for odd exchange = %e\n", retime - ratime);

  ratime = _GET_TIME;
  xchange_eo_field( eo_spinor_field[1], 1);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_Xeo] time for odd exchange = %e\n", retime - ratime);

  /* apply C_oo */
  ratime = _GET_TIME;
  C_oo(eo_spinor_field[2], eo_spinor_field[1], g_gauge_field, -g_mu, eo_spinor_field[4]);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_Xeo] time for C_oo = %e\n", retime - ratime);

#if 0
  /* apply C_oo using X_eo */
  ratime = _GET_TIME;
  C_with_Xeo (eo_spinor_field[3], eo_spinor_field[1], g_gauge_field, -g_mu, eo_spinor_field[4] );
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_Xeo] time for C_oo with X_eo = %e\n", retime - ratime);
#endif

  ratime = _GET_TIME;
  X_eo (eo_spinor_field[4], eo_spinor_field[1], -g_mu, g_gauge_field);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_Xeo] time for X_eo = %e\n", retime - ratime);

  ratime = _GET_TIME;
  C_from_Xeo (eo_spinor_field[3], eo_spinor_field[4], eo_spinor_field[1], g_gauge_field, -g_mu);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_Xeo] time for C_from_Xeo = %e\n", retime - ratime);
    

  spinor_scalar_product_co(&w, eo_spinor_field[3], eo_spinor_field[3], Vhalf);
  w.re *= 4.*g_kappa*g_kappa;
  w.im *= 4.*g_kappa*g_kappa;
  if(g_cart_id == 0) fprintf(stdout, "# [test_Xeo] Xeo w(1) = %25.16e +I %25.16e\n", w.re, w.im);

  spinor_scalar_product_co(&w, eo_spinor_field[2], eo_spinor_field[2], Vhalf);
  w.re *= 4.*g_kappa*g_kappa;
  w.im *= 4.*g_kappa*g_kappa;
  if(g_cart_id == 0) fprintf(stdout, "# [test_Xeo] std w(1) = %25.16e +I %25.16e\n", w.re, w.im);

  spinor_scalar_product_co(&w, eo_spinor_field[1], eo_spinor_field[1], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_Xeo] norm(1) = %25.16e +I %25.16e\n", w.re, w.im);

  if(printf_to_file) {
    sprintf(filename, "coo_xeo_diff.%.2d", g_cart_id);
    ofs = fopen(filename, "w");
    for(ix=0; ix<Vhalf; ix++) {
      fprintf(ofs, "# ix = %8d\n", ix);
      for(i = 0;i<12; i++) {
        fprintf(ofs, "\t%3d%25.16e%25.16e\t%25.16e%25.16e\n", i,
            eo_spinor_field[2][_GSI(ix)+2*i], eo_spinor_field[2][_GSI(ix)+2*i+1], eo_spinor_field[3][_GSI(ix)+2*i], eo_spinor_field[3][_GSI(ix)+2*i+1]);
      }
    }
    fclose(ofs);
  }  /* of if printf to file */


  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  free(g_gauge_field);
  for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
  free(g_spinor_field);

  for(i=0; i<no_eo_fields; i++) free(eo_spinor_field[i]);
  free(eo_spinor_field);


  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
#endif

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_Xeo] %s# [test_Xeo] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_Xeo] %s# [test_Xeo] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }


#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

