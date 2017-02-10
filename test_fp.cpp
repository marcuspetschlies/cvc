/****************************************************
 * test_fp.cpp
 *
 * Tue Feb  7 15:32:56 CET 2017
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
#include "Q_clover_phi.h"
#include "ranlxd.h"
#include "scalar_products.h"
#include "contract_cvc_tensor.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, status; 
  int i, k;
  int x0, x1, x2, x3;
  int y0, y1, y2, y3;
  int z0, z1, z2, z3;
  int isboundary, start_value_t, start_value_x, start_value_y, start_value_z;
  int xsrc[4], psrc[4], have_source = 0, xsrc_iseven;
  unsigned int ixsrc;
  int filename_set = 0;
  int ix, iix;
  int no_eo_fields;
  int exitstatus;
  double plaq;
/*  double U1[18], U2[18]; */
  int verbose = 0;
  char filename[200];

  FILE *ofs=NULL;
  complex w, w2;
  double **eo_spinor_field=NULL;
  double ratime, retime;
  double *gauge_trafo=NULL, *gauge_ptr, U1[18], U2[18];
  double spinor1[24], norm, norm2;
  unsigned int Vhalf, VOL3;
  

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
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_fp] calling tmLQCD wrapper init functions\n");

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
    fprintf(stderr, "[test_fp] ERROR from init_geometry\n");
    EXIT(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "[test_fp] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  int nthreads = omp_get_num_threads();
  int threadid = omp_get_thread_num();
  fprintf(stdout, "# [test_fp] thread%.4d number of threads = %d\n", threadid, nthreads);
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_fp] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  Vhalf = VOLUME / 2;
  VOL3  = LX*LY*LZ;

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [test_fp] reading gauge field from file %s\n", filename);

  if(strcmp(gaugefilename_prefix,"identity")==0) {
    status = unit_gauge_field(g_gauge_field, VOLUME);
  } else {
    /* status = read_nersc_gauge_field_3x3(g_gauge_field, filename, &plaq); */
    /* status = read_ildg_nersc_gauge_field(g_gauge_field, filename); */
    status = read_lime_gauge_field_doubleprec(filename);
    /* status = read_nersc_gauge_field(g_gauge_field, filename, &plaq); */
  }
  if(status != 0) {
    fprintf(stderr, "[test_fp] Error, could not read gauge field\n");
    EXIT(11);
  }
#ifdef HAVE_MPI
  xchange_gauge();
#endif
                        
  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [test_fp] measured plaquette value 1st field: %25.16e\n", plaq);

  /* init rng */
  exitstatus = init_rng_stat_file (g_seed, NULL );

  fermion_propagator_type fp1, fp2;
  create_fp(&fp1);
  create_fp(&fp2);

  ofs = fopen("test_fp.data", "w");

  rangauss(fp1[0], 288);
  rangauss(fp2[0], 288);

  printf_fp( fp1, "fp1", ofs);

/*
  _co_eq_tr_fp( &w, fp1 );
  fprintf(stdout, "# [test_fp] w = %e + I %e\n", w.re, w.im);
*/

  double U[18];
  
  random_cm(U, 1.);

  printf_cm(U, "U", ofs);

  /* _fp_eq_cm_ti_fp(fp2, U, fp1); */
  /* _fp_eq_cm_dagger_ti_fp(fp2, U, fp1); */
  /* _fp_eq_fp_ti_cm(fp2, U, fp1); */
  /* _fp_eq_fp_ti_cm_dagger(fp2, U, fp1); */


  /* _co_eq_tr_fp_dagger_ti_fp (&w, fp1, fp2); */
  /* fprintf(stdout, "# [test_fp] w = %25.16e +I %25.16e\n", w.re, w.im); */
 
  /* _fp_eq_gamma_ti_fp(fp2, 10, fp1); */
  /* _fp_eq_fp_ti_gamma(fp2, 7, fp1); */

  /*
  _fp_eq_cm_ti_fp(fp2, U, fp1);
  _fp_eq_cm_dagger_ti_fp(fp1, U, fp2);
  _fp_eq_fp_ti_cm_dagger(fp2, U, fp1);
  _fp_eq_fp_ti_cm(fp1, U, fp2);
*/
  printf_fp( fp2, "fp2", ofs);

  /* _co_eq_zero(&w); */
  /* co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (&w, &fp1, &fp2, 1., 1); */
  /* fprintf(stdout, "# [test_fp] w = %25.16e +I %25.16e\n", w.re, w.im); */

  apply_propagator_constant_cvc_vertex ( &fp2, &fp1, 0, 0, U, 1 );
  printf_fp( fp2, "fp2", ofs);

 
  free_fp(&fp1);
  free_fp(&fp2);

  fclose(ofs);

#if 0

  /* init and allocate spinor fields */

  no_fields = 24;
  g_spinor_field = (double**)calloc(no_fields, sizeof(double*));
  g_spinor_field[0 ] = (double*)malloc(no_fields * _GSI(VOLUME+RAND)*sizeof(double) );
  for(i=1; i<no_fields; i++) g_spinor_field[i] = g_spinor_field[i-1] +  _GSI( VOLUME+RAND );

  no_eo_fields = 24;
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));
  eo_spinor_field = (double**)calloc(no_eo_fields * _GSI((VOLUME+RAND)/2), sizeof(double));
  for(i=1; i<no_eo_fields; i++) eo_spinor_field[i] = eo_spinor_field[i-1] + _GSI( (VOLUME+RAND)/2 );

  /* fp fields */
  fp1 = create_fp_field( (VOLUME+RAND)/2);
  fp2 = create_fp_field( (VOLUME+RAND)/2);
 

  /* set random spinor fields */
  for( i=0; i<12; i++ ) {
    rangauss(g_spinor_field[i], _GSI(VOLUME) );
  }

  /* decompose to eo spinor fields */
  for( i=0; i<12; i++ ) {
    spinor_field_lexic2eo (g_spinor_field[i], eo_spinor_field[i], eo_spinor_field[12+i]);
  }

  /* assign fp from spinor fields */
  /* even part */
  exitstatus = assign_fermion_propagaptor_from_spinor_field (fp1, eo_spinor_field, Vhalf);
  /* odd part */
  exitstatus = assign_fermion_propagaptor_from_spinor_field (fp2, eo_spinor_field+12, Vhalf);


 

  free_geometry();

  free(g_spinor_field[0]);
  free(g_spinor_field);
  free(eo_spinor_field[0]);
  free(eo_spinor_field);

  free_fp_field(&fp1);
  free_fp_field(&fp2);
#endif  /* of if 0 */
#ifdef HAVE_MPI
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
#endif


  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_fp] %s# [test_fp] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_fp] %s# [test_fp] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }


#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

