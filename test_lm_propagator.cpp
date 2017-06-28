/****************************************************
 * test_lm_propagator.cpp
 *
 * Fri Oct 21 09:33:34 CEST 2016
 *
 * - originally copied from unfinished p2gg_xspace_ama.cpp
 *
 * PURPOSE:
 * - test low-mode construction of propagators and projections
 *   of propagators
 * DONE:
 * TODO:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

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

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "matrix_init.h"
#include "prepare_source.h"
#include "project.h"
#include "Q_phi.h"
#include "scalar_products.h"
#include "gsp.h"

#define EO_FLAG_EVEN 0
#define EO_FLAG_ODD  1

#define _OP_ID_UP 0
#define _OP_ID_DN 1

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform cvc correlator conn. contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options: -v verbose\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  EXIT(0);
}

/************************************************************************************
 * main program
 ************************************************************************************/

int main(int argc, char **argv) {
  
  char outfile_name[] = "test_lm_propagator";

  const double PI2 =  2. * M_PI;

  int c, i, k, mu, ia;
  int no_eo_fields = 0, no_eo_work_fields=0;
  int op_id;
  int flavor_sign=0;
  int filename_set = 0;
  int source_location, source_location_iseven;
  int x0, x1, x2, x3;
  unsigned int ix, Vhalf;
  int gsx[4];
  int sx0, sx1, sx2, sx3;
  int isimag[4];
  int gperm[5][4], gperm2[4][4];
  int check_position_space_WI=0;
  int nthreads=-1, threadid=-1;
  int exitstatus;
  int write_ascii=0;
  int source_proc_coords[4], source_proc_id = -1;
  size_t sizeof_eo_spinor_field;
  double ***gsp_u=NULL;
  /* double ***gsp_d=NULL; */
  int verbose = 0;
  char filename[100];
  char outfile_tag[200];
  double ratime, retime;
  double plaq, norm;
#ifndef HAVE_OPENMP
  double spinor1[24], spinor2[24], U_[18];
#endif
/*  double *gauge_trafo = NULL; */
  double **pcoeff=NULL;
  double ***p3coeff=NULL;
  double *gauge_field_with_phase = NULL;
  complex w, w1;
  FILE *ofs;

  int evecs_num=0, evecs_eval_set = 0;
  double *evecs_lambdaOneHalf=NULL, *evecs_eval = NULL;
  double *eo_evecs_block[2], **eo_spinor_field=NULL, **eo_spinor_work=NULL, **full_spinor_work_halo=NULL;
  double **eo_evecs_field=NULL;

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

  g_the_time = time(NULL);

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_lm_propagators] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_lm_propagators] calling tmLQCD wrapper init functions\n");

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
/*  mpi_init_xchange_contraction(32); */

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_lm_propagators] Error from init_geometry\n");
    EXIT(2);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

  Vhalf = VOLUME / 2;
  sizeof_eo_spinor_field = 24*Vhalf*sizeof(double);

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private(nthreads,threadid)
  {
    threadid = omp_get_thread_num();
    nthreads = omp_get_num_threads();
    fprintf(stdout, "# [test_lm_propagators] proc%.4d thread%.4d using %d threads\n", g_cart_id, threadid, nthreads);
  }
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_lm_propagators] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif


#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [test_lm_propagators] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [test_lm_propagators] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[test_lm_propagators] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(3);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(4);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[test_lm_propagators] Error, &g_gauge_field is NULL\n");
    EXIT(5);
  }
#endif

#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/

  exitstatus = tmLQCD_init_deflator(_OP_ID_UP);
  if( exitstatus > 0) {
    fprintf(stderr, "[test_lm_propagators] Error from tmLQCD_init_deflator, status was %d\n", exitstatus);
    EXIT(9);
  }

  exitstatus = tmLQCD_get_deflator_params(&g_tmLQCD_defl, _OP_ID_UP);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_lm_propagators] Error from tmLQCD_get_deflator_params, status was %d\n", exitstatus);
    EXIT(30);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [test_lm_propagators] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [test_lm_propagators] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [test_lm_propagators] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [test_lm_propagators] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  eo_evecs_block[0] = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block[0] == NULL) {
    fprintf(stderr, "[test_lm_propagators] Error, eo_evecs_block is NULL\n");
    EXIT(32);
  }

  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[test_lm_propagators] Error, dimension of eigenspace is zero\n");
    EXIT(33);
  }

  exitstatus = tmLQCD_set_deflator_fields(_OP_ID_DN, _OP_ID_UP);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_lm_propagators] Error from tmLQCD_set_deflator_fields, status was %d\n", exitstatus);
    EXIT(30);
  }

#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */


  alloc_gauge_field(&gauge_field_with_phase, VOLUMEPLUSRAND);
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for( unsigned int ix=0; ix<VOLUME; ix++ ) {
    for (int mu=0; mu<4; mu++ ) {
      _cm_eq_cm_ti_co ( gauge_field_with_phase+_GGI(ix,mu), g_gauge_field+_GGI(ix,mu), &co_phase_up[mu] );
    }
  }

#ifdef HAVE_MPI
  // xchange_gauge();
  xchange_gauge_field(gauge_field_with_phase);
#endif
        
  /* measure the plaquette */
  //plaquette(&plaq);
  //if(g_cart_id==0) fprintf(stdout, "# [test_lm_propagator] measured plaquette value: %25.16e\n", plaq);
        
  plaquette2(&plaq, gauge_field_with_phase);
  if(g_cart_id==0) fprintf(stdout, "# [test_lm_propagator] gauge field with phase measured plaquette value: %25.16e\n", plaq);
        

  /***********************************************
   * allocate memory for the spinor fields
   ***********************************************/
  /* (1) eigenvector blocks */
#ifndef HAVE_TMLQCD_LIBWRAPPER
  eo_evecs_block[0] = (double*)malloc(evecs_num * sizeof_eo_spinor_field);
  if(eo_evecs_block[0] == NULL) {
    fprintf(stderr, "[test_lm_propagators] Error from malloc\n");
    EXIT(25);
  }
#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */
  eo_evecs_block[1] = (double*)malloc(evecs_num * sizeof_eo_spinor_field);
  if(eo_evecs_block[1] == NULL) {
    fprintf(stderr, "[test_lm_propagators] Error from malloc\n");
    EXIT(26);
  }

  eo_evecs_field = (double**)calloc(2*evecs_num, sizeof(double*));
  eo_evecs_field[0] = eo_evecs_block[0];
  for(i=1; i<evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + 24*Vhalf;
  eo_evecs_field[evecs_num] = eo_evecs_block[1];
  for(i=evecs_num+1; i<2*evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + 24*Vhalf;

  /* (2) fermion fields without halo sites */
  no_eo_fields = 48; /* two full propagators */
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));

  eo_spinor_field[0] = (double*)malloc( no_eo_fields * sizeof_eo_spinor_field);
  if(eo_spinor_field[0] == NULL) {
    fprintf(stderr, "[test_lm_propagators] Error from calloc\n");
    EXIT(35);
  }
  for(i=1; i<no_eo_fields; i++) eo_spinor_field[i] = eo_spinor_field[i-1] + Vhalf*24;

  /* (3) fermion fields with halo sites */
  no_eo_work_fields = 8;
  eo_spinor_work = (double**)calloc(no_eo_work_fields, sizeof(double*));
  eo_spinor_work[0] = (double*)calloc( no_eo_work_fields*12*(VOLUME+RAND), sizeof(double) );
  if(eo_spinor_work[0] == NULL) {
    fprintf(stderr, "[test_lm_propagators] Error from calloc\n");
    EXIT(36);
  }
  for(i=1; i<no_eo_work_fields; i++) {
    eo_spinor_work[i] = eo_spinor_work[i-1] + 12*(VOLUME+RAND);
  }
  full_spinor_work_halo = (double**)calloc( (no_eo_work_fields/2), sizeof(double*));
  for(i=0; i<(no_eo_work_fields/2); i++) {
    full_spinor_work_halo[i] = eo_spinor_work[2*i];
  }

  evecs_eval = (double*)malloc(evecs_num * sizeof(double));
  if(evecs_eval == NULL) {
    fprintf(stderr, "[test_lm_propagators] Error from calloc\n");
    EXIT(63);
  }

  evecs_lambdaOneHalf = (double*)malloc(evecs_num * sizeof(double));
  if(evecs_lambdaOneHalf == NULL) {
    fprintf(stderr, "[test_lm_propagators] Error from calloc\n");
    EXIT(64);
  }

  /***********************************************************
   * determine source coordinates, find out, if source_location is in this process
   ***********************************************************/
  /* global source coordinates */
  gsx[0] = g_source_location / ( LX_global * LY_global * LZ_global);
  gsx[1] = (g_source_location % ( LX_global * LY_global * LZ_global)) / (LY_global * LZ_global);
  gsx[2] = (g_source_location % ( LY_global * LZ_global)) / LZ_global;
  gsx[3] = (g_source_location % LZ_global);
  /* local source coordinates */
  sx0 = gsx[0] % T;
  sx1 = gsx[1] % LX;
  sx2 = gsx[2] % LY;
  sx3 = gsx[3] % LZ;
  source_proc_id = 0;
#ifdef HAVE_MPI
  source_proc_coords[0] = gsx[0] / T;
  source_proc_coords[1] = gsx[1] / LX;
  source_proc_coords[2] = gsx[2] / LY;
  source_proc_coords[3] = gsx[3] / LZ;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [test_lm_propagators] global source coordinates: (%3d,%3d,%3d,%3d)\n",  gsx[0], gsx[1], gsx[2], gsx[3]);
    fprintf(stdout, "# [test_lm_propagators] source proc coordinates: (%3d,%3d,%3d,%3d)\n",  source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
  }

  exitstatus = MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
  if(exitstatus !=  MPI_SUCCESS ) {
    fprintf(stderr, "[test_lm_propagators] Error from MPI_Cart_rank, status was %d\n", exitstatus);
    EXIT(9);
  }
#endif
  if( source_proc_id == g_cart_id) {
    fprintf(stdout, "# [test_lm_propagators] process %2d has source location\n", source_proc_id);
  }

  if( source_proc_id == g_cart_id) {
    source_location_iseven = g_iseven[g_ipt[sx0][sx1][sx2][sx3]];
    fprintf(stdout, "# [test_lm_propagators] source site (%d, %d, %d, %d) is even = %d\n", gsx[0], gsx[1], gsx[2], gsx[3], source_location_iseven);
  }

  /***********************************************************
   * check number of operators, maximally 2 for now
   ***********************************************************/
#ifdef HAVE_TMLQCD_LIBWRAPPER
  if(g_tmLQCD_lat.no_operators > 2) {
    if(g_cart_id == 0) fprintf(stderr, "[test_lm_propagators] Error, confused about number of operators, expected 2 operator (up-type, dn-type)\n");
    EXIT(9);
  }
#endif
  
#ifdef HAVE_TMLQCD_LIBWRAPPER
  for(i=0; i<evecs_num; i++) {
    evecs_eval[i] = ((double*)(g_tmLQCD_defl.evals))[2*i];
  }
#endif
  for(i=0; i<evecs_num; i++) {
    evecs_lambdaOneHalf[i] = 2. * g_kappa / sqrt( evecs_eval[i] );
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] eval %4d %25.16e %25.16e\n", i, evecs_eval[i], evecs_lambdaOneHalf[i]);
  }


  /***********************************************************
   * test eigenvectors
   ***********************************************************/
  for(i=0; i<evecs_num; i++) {
    C_oo (eo_spinor_work[0], eo_evecs_field[i], gauge_field_with_phase, -g_mu, eo_spinor_work[2]);
    C_oo (eo_spinor_work[1], eo_spinor_work[0], gauge_field_with_phase,  g_mu, eo_spinor_work[2]);

    spinor_scalar_product_re(&norm, eo_evecs_field[i], eo_evecs_field[i], Vhalf);
    spinor_scalar_product_co(&w, eo_spinor_work[1], eo_evecs_field[i], Vhalf);

    w.re *= 4.*g_kappa*g_kappa;
    w.im *= 4.*g_kappa*g_kappa;
   
    if(g_cart_id == 0) {
      fprintf(stdout, "# [test_lm_propagator] evec %.4d norm = %25.16e w = %25.16e +I %25.16e\n", i, norm, w.re, w.im);
    }
  }
#if 0
#endif

  /***********************************************************
   * make Wtilde field
   ***********************************************************/
  for(i=0; i<evecs_num; i++) {
    double dnorm, dnorm2;
    C_oo (eo_evecs_field[i+evecs_num], eo_evecs_field[i], gauge_field_with_phase, -g_mu, eo_spinor_work[0]);
    spinor_scalar_product_re(&dnorm,  eo_evecs_field[i], eo_evecs_field[i], Vhalf);
    spinor_scalar_product_re(&dnorm2, eo_evecs_field[i+evecs_num], eo_evecs_field[i+evecs_num], Vhalf);
    norm    = 1./sqrt(dnorm2);
    dnorm2 *= 4.*g_kappa*g_kappa;
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator] evec %.4d ||V||^2 = %25.16e ||W||^2 = %25.16e\n", i, dnorm, dnorm2);
    spinor_field_ti_eq_re(eo_evecs_field[evecs_num+i], norm, Vhalf);
  }

#if 0

#if 0
  random_spinor_field (eo_spinor_field[0], 24*Vhalf);
  random_spinor_field (eo_spinor_field[1], 24*Vhalf);

  memcpy(eo_spinor_work[0], eo_spinor_field[0], sizeof_eo_spinor_field);
  memcpy(eo_spinor_work[1], eo_spinor_field[1], sizeof_eo_spinor_field);

  Q_eo_SchurDecomp_B (eo_spinor_work[2], eo_spinor_work[3], eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_phase, g_mu, eo_spinor_work[4]);
  Q_eo_SchurDecomp_A (eo_spinor_work[0], eo_spinor_work[1], eo_spinor_work[2], eo_spinor_work[3], gauge_field_with_phase, g_mu, eo_spinor_work[4]);
  g5_phi(eo_spinor_work[0], Vhalf);
  g5_phi(eo_spinor_work[1], Vhalf);

  spinor_field_eo2lexic( full_spinor_work_halo[1], eo_spinor_work[0], eo_spinor_work[1]);
  spinor_field_eo2lexic( full_spinor_work_halo[0], eo_spinor_field[0], eo_spinor_field[1]);

  xchange_field(full_spinor_work_halo[0]);
  Q_phi(full_spinor_work_halo[2], full_spinor_work_halo[0], g_mu);

  spinor_field_norm_diff( &norm, full_spinor_work_halo[2], full_spinor_work_halo[1], VOLUME);
   if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] full norm diff %e\n", norm);

  spinor_field_mi_eq_spinor_field_ti_re(full_spinor_work_halo[2], full_spinor_work_halo[1], 1., VOLUME);
  sprintf(filename, "spinor2.%.2d.%.2d", ia, g_cart_id);
  ofs = fopen(filename, "w");
  printf_spinor_field(full_spinor_work_halo[2], 0, ofs);
  fclose(ofs);
#endif
#if 0
  random_spinor_field (eo_spinor_field[0], 24*Vhalf);
  random_spinor_field (eo_spinor_field[1], 24*Vhalf);

  memcpy(eo_spinor_field[2], eo_spinor_field[0], sizeof_eo_spinor_field);
  memcpy(eo_spinor_field[3], eo_spinor_field[1], sizeof_eo_spinor_field);

  g5_phi(eo_spinor_field[2], Vhalf);
  g5_phi(eo_spinor_field[3], Vhalf);
  // Q_eo_SchurDecomp_Ainv (eo_spinor_work[0], eo_spinor_work[1], eo_spinor_field[2], eo_spinor_field[3], gauge_field_with_phase, g_mu, eo_spinor_work[4]);
  Q_eo_SchurDecomp_Ainv (eo_spinor_field[2], eo_spinor_field[3], eo_spinor_field[2], eo_spinor_field[3], gauge_field_with_phase, g_mu, eo_spinor_work[4]);

  memset(eo_spinor_work[2], 0, sizeof_eo_spinor_field);
  memcpy(eo_spinor_work[1], eo_spinor_field[3], sizeof_eo_spinor_field);
  exitstatus = tmLQCD_invert_eo(eo_spinor_work[2], eo_spinor_work[1], op_id);
  memcpy(eo_spinor_field[3], eo_spinor_work[2], sizeof_eo_spinor_field);

  Q_eo_SchurDecomp_Binv (eo_spinor_field[2], eo_spinor_field[3], eo_spinor_field[2], eo_spinor_field[3], gauge_field_with_phase, g_mu, eo_spinor_work[4]);

  spinor_field_eo2lexic( full_spinor_work_halo[0], eo_spinor_field[2], eo_spinor_field[3]);
  xchange_field(full_spinor_work_halo[0]);
  Q_phi(full_spinor_work_halo[1], full_spinor_work_halo[0], g_mu);

  spinor_field_eo2lexic( full_spinor_work_halo[0], eo_spinor_field[0], eo_spinor_field[1]);

  double norm2 = 0.;
  spinor_scalar_product_re( &norm2, full_spinor_work_halo[0], full_spinor_work_halo[0], VOLUME);
  norm2 = sqrt(norm2);
  spinor_field_norm_diff( &norm, full_spinor_work_halo[1], full_spinor_work_halo[0], VOLUME);
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] full norm diff %e\n", norm/norm2);

  sprintf(filename, "spinor1.%.2d", g_cart_id);
  ofs = fopen(filename, "w");
  printf_spinor_field(full_spinor_work_halo[1], 0, ofs);
  fclose(ofs);

  sprintf(filename, "spinor0.%.2d", g_cart_id);
  ofs = fopen(filename, "w");
  printf_spinor_field(full_spinor_work_halo[0], 0, ofs);
  fclose(ofs);
#endif


  /**********************************************************
   * full propagators
   **********************************************************/

  flavor_sign = -1;
  op_id = (1 - flavor_sign ) / 2;
  if(g_cart_id==0) fprintf(stdout, "# [test_lm_propagators] flavor sign = %d; op id = %d\n", flavor_sign, op_id);

  /* loop on spin-color components of point source */
  ratime = _GET_TIME;


  for(ia=0; ia<12; ia++)
  {
#if 0
    random_spinor_field (eo_spinor_work[0], Vhalf);
    random_spinor_field (eo_spinor_work[1], Vhalf);
    Q_eo_SchurDecomp_Ainv (eo_spinor_work[2], eo_spinor_work[3], eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_phase, g_mu, eo_spinor_work[6]);
    Q_eo_SchurDecomp_A    (eo_spinor_work[4], eo_spinor_work[5], eo_spinor_work[2], eo_spinor_work[3], gauge_field_with_phase, g_mu, eo_spinor_work[6]);
    spinor_field_norm_diff( &norm, eo_spinor_work[0], eo_spinor_work[4], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] even resdiue %e\n", norm);
    spinor_field_norm_diff( &norm, eo_spinor_work[1], eo_spinor_work[5], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] odd  resdiue %e\n", norm);
#endif

    /* up-type propagator */

    /* prepare source */
    exitstatus = init_eo_spincolor_pointsource_propagator (eo_spinor_field[ia], eo_spinor_field[12+ia], gsx, ia, flavor_sign, source_proc_id==g_cart_id, eo_spinor_work[2]);
    if(exitstatus != 0 ) {
      fprintf(stderr, "[test_lm_propagators] Error from init_eo_spincolor_pointsource_propagator; status was %d\n", exitstatus);
      EXIT(36);
    }
#if 0
    /* 2,3 <- g5 A 0,1*/
    Q_eo_SchurDecomp_A (eo_spinor_work[0], eo_spinor_work[1], eo_spinor_field[ia], eo_spinor_field[12+ia], gauge_field_with_phase, flavor_sign*g_mu, eo_spinor_work[2]);
    g5_phi(eo_spinor_work[0], Vhalf);
    g5_phi(eo_spinor_work[1], Vhalf);

    if(source_proc_id == g_cart_id) {
      if( g_iseven[g_ipt[sx0][sx1][sx2][sx3]]) {
        eo_spinor_work[0][_GSI( g_lexic2eosub[g_ipt[sx0][sx1][sx2][sx3]] )+2*ia] -= 1.;
      } else {
        eo_spinor_work[1][_GSI( g_lexic2eosub[g_ipt[sx0][sx1][sx2][sx3]] )+2*ia] -= 1.;
      }
    }

    spinor_scalar_product_re(&norm, eo_spinor_work[0], eo_spinor_work[0], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] even norm %2d %e\n", ia, sqrt(norm));
    spinor_scalar_product_re(&norm, eo_spinor_work[1], eo_spinor_work[1], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] odd  norm %2d %e\n", ia, sqrt(norm));
#endif
  }  /* end of loop on spin-color */

#if 0
  exitstatus = project_propagator_field(eo_spinor_field[12], eo_spinor_field[12], 1, eo_evecs_block[op_id], 12, evecs_num, Vhalf);
  if(exitstatus != 0 ) {
    fprintf(stderr, "[test_lm_propagators] Error from project_propagator_field; status was %d\n", exitstatus);
    EXIT(36);
  }
#endif

#if 0
  for(ia=0; ia<12; ia++) {
    spinor_field_pl_eq_spinor_field(eo_spinor_field[ia],eo_spinor_field[24+ia], Vhalf);
    spinor_field_mi_eq_spinor_field_ti_re (eo_spinor_field[ia],eo_spinor_field[12+ia], 1., Vhalf);
  }

  init_2level_buffer(&pcoeff, 12, 24);
  exitstatus = project_reduce_from_propagator_field (pcoeff[0], eo_spinor_field[ 0], eo_spinor_field[ 0], 12, 12, Vhalf, 1);
  for(i=0; i<12; i++) {
    for(k=0; k<12; k++) {
      fprintf(stdout, "pceff %3d%3d%25.16e%25.16e\n", i,k,pcoeff[i][2*k], pcoeff[i][2*k+1]);
    }
  }

  fini_2level_buffer(&pcoeff);
#endif
#if 0
  for(ia=0; ia<12; ia++) {
    spinor_field_norm_diff(&norm, eo_spinor_field[ia], eo_spinor_field[12+ia], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator] norm idff %2d %25.16e\n", ia, norm);
  }
#endif
/*
  if(exitstatus != 0) {
    fprintf(stderr, "[test_lm_propagators] Error from project_propagator_field, %d\n", exitstatus);
    EXIT(37);
  }
*/

  for(ia=0; ia<12; ia++) {
   
    /* invert */
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] calling tmLQCD_invert_eo\n");
    memset(eo_spinor_work[1], 0, sizeof_eo_spinor_field);
    memcpy(eo_spinor_work[0], eo_spinor_field[12+ia], sizeof_eo_spinor_field);
    exitstatus = tmLQCD_invert_eo(eo_spinor_work[1], eo_spinor_work[0], op_id);
    if(exitstatus != 0) {
      fprintf(stderr, "[test_lm_propagators] Error from tmLQCD_invert_eo, status was %d\n", exitstatus);
      EXIT(35);
    }
    memcpy(eo_spinor_field[12+ia], eo_spinor_work[1], sizeof_eo_spinor_field);
/*
    C_oo ( eo_spinor_work[0], eo_spinor_work[2], gauge_field_with_phase, g_mu, eo_spinor_work[3]);
    spinor_field_ti_eq_re (eo_spinor_work[0], 2.*g_kappa, Vhalf);
    spinor_field_norm_diff( &norm, eo_spinor_work[0], eo_spinor_work[1], Vhalf);
    if(g_cart_id == 0) {
      fprintf(stdout, "# [test_lm_propagators] norm %2d %e\n", ia, norm);
    }
*/

    /* Q_eo_SchurDecomp_Binv (eo_spinor_work[0], eo_spinor_work[1], eo_spinor_field[ia], eo_spinor_field[12+ia], gauge_field_with_phase, g_mu, eo_spinor_work[4]); */

    exitstatus = fini_eo_propagator (eo_spinor_field[ia], eo_spinor_field[12+ia], eo_spinor_field[ia], eo_spinor_field[12+ia], flavor_sign, eo_spinor_work[4]);
    if(exitstatus != 0) {
      fprintf(stderr, "[test_lm_propagators] Error from fini_eo_propagator, status was %d\n", exitstatus);
      EXIT(35);
    }

/*
    spinor_field_norm_diff( &norm, eo_spinor_work[0], eo_spinor_field[ia], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] even norm %2d %e\n", ia, norm);
    spinor_field_norm_diff( &norm, eo_spinor_work[1], eo_spinor_field[12+ia], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] odd  norm %2d %e\n", ia, norm);
*/
/*
    spinor_field_eo2lexic(full_spinor_work_halo[0], eo_spinor_field[ia], eo_spinor_field[12+ia] );
    xchange_field(full_spinor_work_halo[0]);
    Q_phi( full_spinor_work_halo[1], full_spinor_work_halo[0], g_mu);
    if(source_proc_id == g_cart_id) {
      full_spinor_work_halo[1][_GSI( g_ipt[sx0][sx1][sx2][sx3] )+2*ia] -= 1.;
    }
    spinor_scalar_product_re(&norm, full_spinor_work_halo[1], full_spinor_work_halo[1], VOLUME);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] full norm %2d %e\n", ia, sqrt(norm));
    sprintf(filename, "spinor1.%.2d.%.2d", ia, g_cart_id);
    ofs = fopen(filename, "w");
    printf_spinor_field(full_spinor_work_halo[1], 0, ofs);
    fclose(ofs);
*/
  }  /* of loop on spin-color component ia */



#if 0
  /* calculate xv */
  exitstatus = gsp_calculate_xv_from_v ( &(eo_evecs_field[evecs_num]), &(eo_evecs_field[0]), &(eo_spinor_work[0]), evecs_num, -g_mu, Vhalf);

  /* calculate w from xv and v */
  exitstatus = gsp_calculate_w_from_xv_and_v (&(eo_evecs_field[evecs_num]), &(eo_evecs_field[evecs_num]), &(eo_evecs_field[0]), &(eo_spinor_work[0]), evecs_num, -g_mu, Vhalf);

  /* expand odd part in basis */
  exitstatus = project_expand_to_propagator_field(eo_spinor_field[36], pcoeff[0], eo_evecs_block[1], 12, evecs_num, Vhalf);
#endif

#if 0
  /* even new = even_old + X_eo odd_new */
  for(ia=0; ia<12; ia++) {
    memcpy(eo_spinor_work[0], eo_spinor_field[36+ia], sizeof_eo_spinor_field);
    X_eo (eo_spinor_work[1], eo_spinor_work[0], g_mu, gauge_field_with_phase);
    spinor_field_pl_eq_spinor_field(eo_spinor_field[24+ia], eo_spinor_work[1], Vhalf);
  }


  for(ia=0; ia<12; ia++) {
    spinor_field_eq_spinor_field_mi_spinor_field(eo_spinor_field[24+ia], eo_spinor_field[24+ia], eo_spinor_field[ia], Vhalf);
    spinor_scalar_product_re(&norm, eo_spinor_field[24+ia], eo_spinor_field[24+ia], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] even norm %2d %e\n", ia, norm);
  }

  for(ia=0; ia<12; ia++) {
    spinor_field_eq_spinor_field_mi_spinor_field(eo_spinor_field[36+ia], eo_spinor_field[36+ia], eo_spinor_field[12+ia], Vhalf);
    spinor_scalar_product_re(&norm, eo_spinor_field[36+ia], eo_spinor_field[36+ia], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] odd  norm %2d %e\n", ia, norm);
  }
#endif

  g_seq_source_momentum[0] = 1;
  g_seq_source_momentum[1] = 2;
  g_seq_source_momentum[2] = 3;
  g_sequential_source_timeslice = 7;
  g_sequential_source_gamma_id = 7;

  for(ia=0; ia<12; ia++) {
    exitstatus = init_eo_sequential_source(eo_spinor_field[24+ia], eo_spinor_field[36+ia], eo_spinor_field[ia], eo_spinor_field[12+ia],
        g_sequential_source_timeslice, flavor_sign, g_seq_source_momentum, g_sequential_source_gamma_id, eo_spinor_work[0]);
/*
    sprintf(filename, "spinor.%.2d.%.2d.e", ia, g_cart_id);
    ofs = fopen(filename, "w");
    printf_eo_spinor_field(eo_spinor_field[24+ia], 1, 0, ofs);
    fclose(ofs);

    sprintf(filename, "spinor.%.2d.%.2d.o", ia, g_cart_id);
    ofs = fopen(filename, "w");
    printf_eo_spinor_field(eo_spinor_field[36+ia], 0, 0, ofs);
    fclose(ofs);
 */
  }

  exitstatus = project_propagator_field(eo_spinor_field[36], eo_spinor_field[36], 1, eo_evecs_block[op_id], 12, evecs_num, Vhalf);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_lm_propagators] Error from project_propagator_fiel, status was %d\n", exitstatus);
    EXIT(35);
  }

  for(ia=0; ia<12; ia++) {
    memset(eo_spinor_work[1], 0, sizeof_eo_spinor_field);
    memcpy(eo_spinor_work[0], eo_spinor_field[36+ia], sizeof_eo_spinor_field);
    exitstatus = tmLQCD_invert_eo(eo_spinor_work[1], eo_spinor_work[0], op_id);
    if(exitstatus != 0) {
      fprintf(stderr, "[test_lm_propagators] Error from tmLQCD_invert_eo, status was %d\n", exitstatus);
      EXIT(35);
    }
    memcpy(eo_spinor_field[36+ia], eo_spinor_work[1], sizeof_eo_spinor_field);
    fini_eo_propagator(eo_spinor_field[24+ia], eo_spinor_field[36+ia], eo_spinor_field[24+ia], eo_spinor_field[36+ia], flavor_sign, eo_spinor_work[0]);
  }

#if 0
  for(ia=0; ia<12; ia++) {
    spinor_field_eo2lexic(full_spinor_work_halo[0], eo_spinor_field[24+ia], eo_spinor_field[36+ia] );
    xchange_field(full_spinor_work_halo[0]);
    Q_phi( full_spinor_work_halo[1], full_spinor_work_halo[0], g_mu);
/*
    Q_eo_SchurDecomp_Ainv (eo_spinor_field[24+ia], eo_spinor_field[36+ia], eo_spinor_field[   ia], eo_spinor_field[12+ia], gauge_field_with_phase, g_mu, eo_spinor_work[0]);
    Q_eo_SchurDecomp_A    (eo_spinor_work[1], eo_spinor_work[2], eo_spinor_field[24+ia], eo_spinor_field[36+ia], gauge_field_with_phase, g_mu, eo_spinor_work[0]);
    Q_eo_SchurDecomp_A    (eo_spinor_field[24+ia], eo_spinor_field[36+ia], eo_spinor_field[24+ia], eo_spinor_field[36+ia], gauge_field_with_phase, g_mu, eo_spinor_work[0]);
*/
/*
    spinor_field_norm_diff (&norm, eo_spinor_field[ia], eo_spinor_field[24+ia], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] even norm diff %2d %25.16e\n", ia, norm);
    spinor_field_norm_diff (&norm, eo_spinor_field[12+ia], eo_spinor_field[36+ia], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] odd  norm diff %2d %25.16e\n", ia, norm);
*/
    spinor_field_eo2lexic(full_spinor_work_halo[0], eo_spinor_field[ia], eo_spinor_field[12+ia] );
/*
    sprintf(filename, "spinor0.%.2d.%.2d", ia, g_cart_id);
    ofs = fopen(filename, "w");
    printf_spinor_field(full_spinor_work_halo[0], 0, ofs);
    fclose(ofs);
*/

    if(g_proc_coords[0] == g_sequential_source_timeslice / T) {
      fprintf(stdout, "# [test_lm_propagators] process %d applying inverted seq vertex\n", g_cart_id);
      for(x0=0; x0<T; x0++) {
        if(x0 == g_sequential_source_timeslice%T) {
          for(x1=0; x1<LX; x1++) {
          for(x2=0; x2<LY; x2++) {
          for(x3=0; x3<LZ; x3++) {
            ix = g_ipt[x0][x1][x2][x3];
            double phase = 2. * M_PI * ( (x1 + LX * g_proc_coords[1]) * g_seq_source_momentum[0] / (double)LX_global + 
                                         (x2 + LY * g_proc_coords[2]) * g_seq_source_momentum[1] / (double)LY_global + 
                                         (x3 + LZ * g_proc_coords[3]) * g_seq_source_momentum[2] / (double)LZ_global );
            complex w = (complex) {cos(phase), -sin(phase)};
            _fv_eq_fv_ti_co(spinor1, full_spinor_work_halo[1]+_GSI(ix), &w);
            _fv_eq_gamma_ti_fv(full_spinor_work_halo[1]+_GSI(ix), g_sequential_source_gamma_id, spinor1);
          }}}
        } else {
          ix = g_ipt[x0][0][0][0];
          memset(full_spinor_work_halo[0]+_GSI(ix), 0, 24*LX*LY*LZ*sizeof(double));
        }
      }
    } else {
      memset(full_spinor_work_halo[0], 0, 24*VOLUME*sizeof(double));
    }

    spinor_field_mi_eq_spinor_field_ti_re(full_spinor_work_halo[1], full_spinor_work_halo[0], -1., VOLUME);
    spinor_scalar_product_re(&norm, full_spinor_work_halo[1], full_spinor_work_halo[1], VOLUME);
    // spinor_field_norm_diff (&norm, full_spinor_work_halo[1], full_spinor_work_halo[0], VOLUME);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators] seq %2d norm diff = %25.16e\n", ia, sqrt(norm));

/*
    sprintf(filename, "spinor0.%.2d.%.2d", ia, g_cart_id);
    ofs = fopen(filename, "w");
    printf_spinor_field(full_spinor_work_halo[0], 0, ofs);
    fclose(ofs);
    sprintf(filename, "spinor1.%.2d.%.2d", ia, g_cart_id);
    ofs = fopen(filename, "w");
    printf_spinor_field(full_spinor_work_halo[1], 0, ofs);
    fclose(ofs);
*/
  }
#endif

  for(ia=0; ia<12; ia++) {
    exitstatus = init_eo_sequential_source(eo_spinor_field[ia], eo_spinor_field[12+ia], eo_spinor_field[ia], eo_spinor_field[12+ia],
        g_sequential_source_timeslice, flavor_sign, g_seq_source_momentum, g_sequential_source_gamma_id, eo_spinor_work[0]);

    if(flavor_sign == -1) {
      memcpy(eo_spinor_work[0], eo_spinor_field[12+ia], sizeof_eo_spinor_field);
      C_oo ( eo_spinor_field[12+ia], eo_spinor_work[0], gauge_field_with_phase, g_mu, eo_spinor_work[1]);
    }
  }

  init_2level_buffer(&pcoeff, 12, 2*evecs_num);

  /* odd projection coefficients v^+ sp_o */
  exitstatus = project_reduce_from_propagator_field (pcoeff[0], eo_spinor_field[12], eo_evecs_block[0], 12, evecs_num, Vhalf, 1);

  for(ia=0; ia<12; ia++) {
    for(i=0; i<evecs_num; i++) {
      /*pcoeff[ia][2*i  ] *= evecs_lambdaOneHalf[i];
      pcoeff[ia][2*i+1] *= evecs_lambdaOneHalf[i];*/
      /* norm = 1. / sqrt(evecs_eval[i]); */
      norm = 2.*g_kappa / evecs_eval[i];
      pcoeff[ia][2*i  ] *= norm;
      pcoeff[ia][2*i+1] *= norm;
    }
  }

#if 0
  /* calculate xv */
  exitstatus = gsp_calculate_xv_from_v ( &(eo_evecs_field[evecs_num]), &(eo_evecs_field[0]), &(eo_spinor_work[0]), evecs_num, -g_mu, Vhalf);
  /* calculate w from xv and v */
  exitstatus = gsp_calculate_w_from_xv_and_v (&(eo_evecs_field[evecs_num]), &(eo_evecs_field[evecs_num]), &(eo_evecs_field[0]), &(eo_spinor_work[0]), evecs_num, -g_mu, Vhalf);
  /* expand odd part in basis */
  exitstatus = project_expand_to_propagator_field(eo_spinor_field[12], pcoeff[0], eo_evecs_block[1], 12, evecs_num, Vhalf);
#endif

  exitstatus = project_expand_to_propagator_field(eo_spinor_field[12], pcoeff[0], eo_evecs_block[0], 12, evecs_num, Vhalf);
  if(flavor_sign == 1) {
    for(ia=0; ia<12; ia++) {
      memcpy(eo_spinor_work[0], eo_spinor_field[12+ia], sizeof_eo_spinor_field);
      C_oo (eo_spinor_field[12+ia], eo_spinor_work[0], gauge_field_with_phase, -g_mu, eo_spinor_work[2]);
/*
      sprintf(filename, "spinor.%.2d.%.2d.o", ia, g_cart_id);
      ofs = fopen(filename, "w");
      printf_eo_spinor_field(eo_spinor_field[12+ia], 1, 0, ofs);
      fclose(ofs);
*/
    }
  }
  fini_2level_buffer(&pcoeff);

  for(ia=0; ia<12; ia++) {
    Q_eo_SchurDecomp_Binv (eo_spinor_field[ia], eo_spinor_field[12+ia], eo_spinor_field[ia], eo_spinor_field[12+ia], gauge_field_with_phase, flavor_sign*g_mu, eo_spinor_work[4]);
  }

  for(ia=0; ia<12; ia++) {
    spinor_field_norm_diff(&norm, eo_spinor_field[ia], eo_spinor_field[24+ia], Vhalf);
    if(g_cart_id == 0) printf("# even norm %2d %25.16e\n", ia, norm);
    spinor_field_norm_diff(&norm, eo_spinor_field[12+ia], eo_spinor_field[36+ia], Vhalf);
    if(g_cart_id == 0) printf("# odd  norm %2d %25.16e\n", ia, norm);
  }

  retime = _GET_TIME;
  if(g_cart_id == 0) {
    fprintf(stdout, "# [test_lm_propagators] time for propagators = %e seconds\n", retime-ratime);
  }
#if 0
  for(ia=0; ia<evecs_num; ia++) {
    memcpy(eo_spinor_field[0], eo_evecs_field[ia], sizeof_eo_spinor_field);
    C_oo (eo_spinor_field[1], eo_spinor_field[0], gauge_field_with_phase, -g_mu, eo_spinor_work[2]);
    spinor_scalar_product_re(&norm, eo_spinor_field[1], eo_spinor_field[1], Vhalf);
    spinor_field_ti_eq_re(eo_spinor_field[1], sqrt(1./norm), Vhalf);

    memcpy(eo_spinor_work[0], eo_evecs_field[ia], sizeof_eo_spinor_field);
    gsp_calculate_xv_from_v ( &(eo_spinor_work[1]), &(eo_spinor_work[0]), &(eo_spinor_work[4]), 1, -g_mu, Vhalf);
    gsp_calculate_w_from_xv_and_v (&(eo_spinor_work[1]), &(eo_spinor_work[1]), &(eo_spinor_work[0]), &(eo_spinor_work[4]), 1, -g_mu, Vhalf);

    spinor_field_norm_diff(&norm, eo_spinor_field[1], eo_spinor_work[1], Vhalf);
    if(g_cart_id == 0) printf("# [test_lm_propagators] eo norm %4d %25.16e\n", ia, norm);
  }
#endif

#endif  /* of if 0 */

  /****************************************
   * free the allocated memory, finalize
   ****************************************/

  if( eo_spinor_field != NULL ) {
    if( eo_spinor_field[0] != NULL ) free(eo_spinor_field[0]);
    free(eo_spinor_field);
  }

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(eo_evecs_block[0]);
#endif
  free(eo_evecs_block[1]);

  if(eo_spinor_work != NULL ) {
    if(eo_spinor_work[0] != NULL ) free(eo_spinor_work[0]);
    free(eo_spinor_work);
  }

  if(eo_evecs_field        != NULL ) free(eo_evecs_field);
  if(full_spinor_work_halo != NULL ) free(full_spinor_work_halo);

  free_geometry();

  if( evecs_lambdaOneHalf != NULL ) free( evecs_lambdaOneHalf );
#ifndef HAVE_TMLQCD_LIBWRAPPER
  if( evecs_eval          != NULL ) free( evecs_eval );
#endif


#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
/*  mpi_fini_xchange_contraction(); */
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_lm_propagators] %s# [test_lm_propagators] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "[test_lm_propagators] %s[test_lm_propagators] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
