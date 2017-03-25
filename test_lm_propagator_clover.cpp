/****************************************************
 * test_lm_propagator_clover.cpp
 *
 * Thu Nov  3 16:15:53 CET 2016
 *
 * - originally copied from unfinished test_lm_propagator.cpp
 *
 * PURPOSE:
 * - same as test_lm_propagator.cpp, but with tm+clover
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
#include "Q_clover_phi.h"
#include "scalar_products.h"
#include "gsp.h"

#define EO_FLAG_EVEN 0
#define EO_FLAG_ODD  1

#define OP_ID_UP 0
#define OP_ID_DN 1

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform cvc correlator conn. contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "         -f input filename [default cvc.input]\n");
  EXIT(0);
}

/************************************************************************************
 * main program
 ************************************************************************************/

int main(int argc, char **argv) {
  
  int c, i, ia;
  int no_eo_fields = 0, no_eo_work_fields=0;
  int op_id = -1;
  int filename_set = 0;
  int source_location_iseven;
  int flavor_sign, flavor_id;
  unsigned int Vhalf;
  int gsx[4];
  int sx0, sx1, sx2, sx3;
  int exitstatus;
  int source_proc_coords[4], source_proc_id = -1;
  size_t sizeof_eo_spinor_field, sizeof_spinor_field;
  char filename[100];
  double ratime, retime;
  double plaq, norm;
  double **mzz[2], **mzzinv[2];
  double *gauge_field_with_phase = NULL;


/*  double *gauge_trafo = NULL; */
  double **pcoeff=NULL;
  /* complex w; */

  int evecs_num=0;
  double *evecs_lambdaOneHalf=NULL, *evecs_eval = NULL;
  double *eo_evecs_block[2], **eo_spinor_field=NULL, **eo_spinor_work=NULL, **full_spinor_work_halo=NULL;
  double **eo_evecs_field=NULL;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:")) != -1) {
    switch (c) {
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
  if(filename_set==0) strcpy(filename, "p2gg.input");
  fprintf(stdout, "# [test_lm_propagator_clover] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_lm_propagator_clover] calling tmLQCD wrapper init functions\n");

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

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_lm_propagator_clover] Error from init_geometry\n");
    EXIT(2);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

  Vhalf = VOLUME / 2;
  sizeof_eo_spinor_field = 24*Vhalf*sizeof(double);
  sizeof_spinor_field    = 24*VOLUME*sizeof(double);

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
  {
    fprintf(stdout, "# [test_lm_propagator_clover] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
  }
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_lm_propagator_clover] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif


#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [test_lm_propagator_clover] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [test_lm_propagator_clover] initializing unit matrices\n");
    for(ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[test_lm_propagator_clover] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(3);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if(exitstatus != 0) {
    EXIT(4);
  }
  if(&g_gauge_field == NULL) {
    fprintf(stderr, "[test_lm_propagator_clover] Error, &g_gauge_field is NULL\n");
    EXIT(5);
  }
#endif

  alloc_gauge_field(&gauge_field_with_phase, VOLUMEPLUSRAND);
#ifdef HAVE_OPENMP
#pragma omp parallel for private(mu)
#endif
  for( unsigned int ix=0; ix<VOLUME; ix++ ) {
    for (int mu=0; mu<4; mu++ ) {
      _cm_eq_cm_ti_co ( gauge_field_with_phase+_GGI(ix,mu), g_gauge_field+_GGI(ix,mu), &co_phase_up[mu] );
    }
  }

#ifdef HAVE_MPI
  xchange_gauge();
  xchange_gauge_field(gauge_field_with_phase);
#endif

  /* measure the plaquette */
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [test_lm_propagator_clover] measured plaquette value: %25.16e\n", plaq);

  plaquette2(&plaq, gauge_field_with_phase);
  if(g_cart_id==0) fprintf(stdout, "# [test_lm_propagator_clover] gauge field with phase measured plaquette value: %25.16e\n", plaq);

#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/
  op_id = OP_ID_UP;

  exitstatus = tmLQCD_init_deflator(op_id);
  if( exitstatus > 0) {
    fprintf(stderr, "[test_lm_propagator_clover] Error from tmLQCD_init_deflator, status was %d\n", exitstatus);
    EXIT(9);
  }

  exitstatus = tmLQCD_get_deflator_params(&g_tmLQCD_defl, op_id);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_lm_propagator_clover] Error from tmLQCD_get_deflator_params, status was %d\n", exitstatus);
    EXIT(30);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [test_lm_propagator_clover] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [test_lm_propagator_clover] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [test_lm_propagator_clover] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [test_lm_propagator_clover] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  eo_evecs_block[0] = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block[0] == NULL) {
    fprintf(stderr, "[test_lm_propagator_clover] Error, eo_evecs_block is NULL\n");
    EXIT(32);
  }

  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[test_lm_propagator_clover] Error, dimension of eigenspace is zero\n");
    EXIT(33);
  }

  exitstatus = tmLQCD_set_deflator_fields(1, 0);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_lm_propagator_clover] Error from tmLQCD_set_deflator_fields, status was %d\n", exitstatus);
    EXIT(30);
  }
#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */

  /***********************************************
   * allocate memory for the spinor fields
   ***********************************************/
  /* (1) eigenvector blocks */
#ifndef HAVE_TMLQCD_LIBWRAPPER
  eo_evecs_block[0] = (double*)malloc(evecs_num * sizeof_eo_spinor_field);
  if(eo_evecs_block[0] == NULL) {
    fprintf(stderr, "[test_lm_propagator_clover] Error from malloc\n");
    EXIT(25);
  }
#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */
  eo_evecs_block[1] = (double*)malloc(evecs_num * sizeof_eo_spinor_field);
  if(eo_evecs_block[1] == NULL) {
    fprintf(stderr, "[test_lm_propagator_clover] Error from malloc\n");
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
    fprintf(stderr, "[test_lm_propagator_clover] Error from calloc\n");
    EXIT(35);
  }
  for(i=1; i<no_eo_fields; i++) eo_spinor_field[i] = eo_spinor_field[i-1] + Vhalf*24;

  /* (3) fermion fields with halo sites */
  no_eo_work_fields = 8;
  eo_spinor_work = (double**)calloc(no_eo_work_fields, sizeof(double*));
  eo_spinor_work[0] = (double*)calloc( no_eo_work_fields*12*(VOLUME+RAND), sizeof(double) );
  if(eo_spinor_work[0] == NULL) {
    fprintf(stderr, "[test_lm_propagator_clover] Error from calloc\n");
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
    fprintf(stderr, "[test_lm_propagator_clover] Error from calloc\n");
    EXIT(63);
  }

  evecs_lambdaOneHalf = (double*)malloc(evecs_num * sizeof(double));
  if(evecs_lambdaOneHalf == NULL) {
    fprintf(stderr, "[test_lm_propagator_clover] Error from calloc\n");
    EXIT(64);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  clover_term_init(&g_clover, 6);

  clover_term_init(&g_mzz_up, 6);
  clover_term_init(&g_mzz_dn, 6);
  clover_term_init(&g_mzzinv_up, 8);
  clover_term_init(&g_mzzinv_dn, 8);

  ratime = _GET_TIME;
  clover_term_eo (g_clover, g_gauge_field);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] time for clover_term_eo = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_matrix (g_mzz_up, g_clover, g_mu, g_csw);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] time for clover_mzz_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_matrix (g_mzz_dn, g_clover, -g_mu, g_csw);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] time for clover_mzz_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_inv_matrix (g_mzzinv_up, g_mzz_up);
  retime = _GET_TIME;
  fprintf(stdout, "# [test_lm_propagator_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_inv_matrix (g_mzzinv_dn, g_mzz_dn);
  retime = _GET_TIME;
  fprintf(stdout, "# [test_lm_propagator_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);

  mzz[0] = g_mzz_up;
  mzz[1] = g_mzz_dn;
  mzzinv[0] = g_mzzinv_up;
  mzzinv[1] = g_mzzinv_dn;

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
    fprintf(stdout, "# [test_lm_propagator_clover] global source coordinates: (%3d,%3d,%3d,%3d)\n",  gsx[0], gsx[1], gsx[2], gsx[3]);
    fprintf(stdout, "# [test_lm_propagator_clover] source proc coordinates: (%3d,%3d,%3d,%3d)\n",  source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
  }

  exitstatus = MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
  if(exitstatus !=  MPI_SUCCESS ) {
    fprintf(stderr, "[test_lm_propagator_clover] Error from MPI_Cart_rank, status was %d\n", exitstatus);
    EXIT(9);
  }
#endif
  if( source_proc_id == g_cart_id) {
    fprintf(stdout, "# [test_lm_propagator_clover] process %2d has source location\n", source_proc_id);
  }

  if( source_proc_id == g_cart_id) {
    source_location_iseven = g_iseven[g_ipt[sx0][sx1][sx2][sx3]];
    fprintf(stdout, "# [test_lm_propagator_clover] source site (%d, %d, %d, %d) is even = %d\n", gsx[0], gsx[1], gsx[2], gsx[3], source_location_iseven);
  }

  /***********************************************************
   * check number of operators, maximally 2 for now
   ***********************************************************/
#ifdef HAVE_TMLQCD_LIBWRAPPER
  if(g_tmLQCD_lat.no_operators > 2) {
    if(g_cart_id == 0) fprintf(stderr, "[test_lm_propagator_clover] Error, confused about number of operators, expected 2 operator (up-type, dn-type)\n");
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
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] eval %4d %25.16e %25.16e\n", i, evecs_eval[i], evecs_lambdaOneHalf[i]);
  }

  /***********************************************
   * init ranlxd random number generator
   ***********************************************/
  exitstatus = init_rng_stat_file (g_seed, NULL);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_lm_propagator_clover] Error from init_rng_stat_file, status was %d\n", exitstatus);
    EXIT(32);
  }

  flavor_sign = -1;
  flavor_id = ( 1 - flavor_sign) / 2;
  op_id = flavor_id;
  if(g_cart_id==0) fprintf(stdout, "# [] flavor sign = %d, flavor id = %d, op id = %d\n", flavor_sign, flavor_id, op_id);

#if 0
  /***********************************************************
   * test eigenvectors
   ***********************************************************/
  if(g_cart_id == 0) {
    fprintf(stdout, "# [test_lm_propagator_clover] eigenvector test\n");
  }
  for(i=0; i<evecs_num; i++) {
    complex w;
    C_clover_oo (eo_spinor_field[0], eo_evecs_field[i],  gauge_field_with_phase, eo_spinor_work[2], g_mzz_dn[1], g_mzzinv_dn[0]);
    C_clover_oo (eo_spinor_field[1], eo_spinor_field[0], gauge_field_with_phase, eo_spinor_work[2], g_mzz_up[1], g_mzzinv_up[0]);

    spinor_scalar_product_re(&norm, eo_evecs_field[i], eo_evecs_field[i], Vhalf);
    spinor_scalar_product_co(&w, eo_spinor_field[1], eo_evecs_field[i], Vhalf);

    w.re *= 4.*g_kappa*g_kappa;
    w.im *= 4.*g_kappa*g_kappa;
   
    if(g_cart_id == 0) {
      fprintf(stdout, "# [] evec %.4d norm = %16.7e w = %16.7e +I %16.7e\n", i, norm, w.re, w.im);
    }
  }
#endif

#if 0
  /***********************************************************
   * make Wtilde field
   ***********************************************************/
  for(i=0; i<evecs_num; i++) {
    double dnorm, dnorm2;
    C_clover_oo (eo_evecs_field[i+evecs_num], eo_evecs_field[i], gauge_field_with_phase, eo_spinor_work[0], g_mzz_dn[1], g_mzzinv_dn[0]);
    spinor_scalar_product_re(&dnorm,  eo_evecs_field[i], eo_evecs_field[i], Vhalf);
    spinor_scalar_product_re(&dnorm2, eo_evecs_field[i+evecs_num], eo_evecs_field[i+evecs_num], Vhalf);
    norm    = 1./sqrt(dnorm2);
    dnorm2 *= 4.*g_kappa*g_kappa;
    if(g_cart_id == 0) fprintf(stdout, "# [] evec %.4d ||V||^2 = %16.7e ||W||^2 = %16.7e\n", i, dnorm, dnorm2);
    spinor_field_ti_eq_re(eo_evecs_field[evecs_num+i], norm, Vhalf);
  }
#endif

#if 0
  /***********************************************************
   * TEST A,B Schur decomposition against Q_clover_phi
   ***********************************************************/
  random_spinor_field (eo_spinor_field[0], Vhalf);
  random_spinor_field (eo_spinor_field[1], Vhalf);

  Q_clover_eo_SchurDecomp_B (eo_spinor_field[2], eo_spinor_field[3], eo_spinor_field[0], eo_spinor_field[1], gauge_field_with_phase, mzz[flavor_id][1], mzzinv[flavor_id][0], eo_spinor_work[0]);
  Q_clover_eo_SchurDecomp_A (eo_spinor_field[4], eo_spinor_field[5], eo_spinor_field[2], eo_spinor_field[3], gauge_field_with_phase, mzz[flavor_id][0], eo_spinor_work[0]);
  g5_phi(eo_spinor_field[4], Vhalf);
  g5_phi(eo_spinor_field[5], Vhalf);

  memcpy(eo_spinor_work[0], eo_spinor_field[0], sizeof_eo_spinor_field);
  memcpy(eo_spinor_work[1], eo_spinor_field[1], sizeof_eo_spinor_field);
  Q_clover_phi_matrix_eo (eo_spinor_work[2], eo_spinor_work[3], eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_phase, eo_spinor_work[4], mzz[flavor_id]);

  spinor_scalar_product_re( &norm, eo_spinor_field[0], eo_spinor_field[0], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators_clover] orig even norm %e\n", norm);
  spinor_scalar_product_re( &norm, eo_spinor_field[1], eo_spinor_field[1], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators_clover] orig odd  norm %e\n", norm);

  spinor_field_norm_diff( &norm, eo_spinor_work[2], eo_spinor_field[4], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators_clover] Q g5 A B even norm diff %e\n", norm);
  spinor_field_norm_diff( &norm, eo_spinor_work[3], eo_spinor_field[5], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators_clover] Q g5 A B odd  norm diff %e\n", norm);
  /* end of TEST */
#endif

#if 0
  /***********************************************************
   * TEST A Schur decomposition against Ainv
   ***********************************************************/
  random_spinor_field (eo_spinor_field[0], Vhalf);
  random_spinor_field (eo_spinor_field[1], Vhalf);

  Q_clover_eo_SchurDecomp_A    (eo_spinor_field[2], eo_spinor_field[3], eo_spinor_field[0], eo_spinor_field[1], gauge_field_with_phase, mzz[flavor_id][0],    eo_spinor_work[0]);
  Q_clover_eo_SchurDecomp_Ainv (eo_spinor_field[4], eo_spinor_field[5], eo_spinor_field[2], eo_spinor_field[3], gauge_field_with_phase, mzzinv[flavor_id][0], eo_spinor_work[0]);

  spinor_field_norm_diff( &norm, eo_spinor_field[4], eo_spinor_field[0], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators_clover] A^-1 A even norm diff %e\n", norm);
  spinor_field_norm_diff( &norm, eo_spinor_field[5], eo_spinor_field[1], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators_clover] A^-1 A odd  norm diff %e\n", norm);
#endif

#if 0
  /***********************************************************
   * TEST Q_clover_phi_matrix_eo and
   *   Q_clover_eo_SchurDecomp_A, Q_clover_eo_SchurDecomp_B
   *   against tmLQCD_invert with lexic return field
   ***********************************************************/
  random_spinor_field (eo_spinor_field[0], Vhalf);
  random_spinor_field (eo_spinor_field[1], Vhalf);

  spinor_field_eo2lexic (full_spinor_work_halo[0], eo_spinor_field[0], eo_spinor_field[1] );
  memset(full_spinor_work_halo[1], 0, sizeof_spinor_field);

  exitstatus = tmLQCD_invert( full_spinor_work_halo[1], full_spinor_work_halo[0], op_id, 0);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_lm_propagator_clover] Error from tmLQCD_invert_eo, status was %d\n", exitstatus);
    EXIT(35);
  }
  spinor_field_lexic2eo (full_spinor_work_halo[1], eo_spinor_work[0], eo_spinor_work[1] );

  /* apply the full operator Q_clover_phi_matrix_eo */
  Q_clover_phi_matrix_eo (eo_spinor_field[2], eo_spinor_field[3], eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_phase, eo_spinor_work[4], mzz[flavor_id]);

  spinor_field_norm_diff( &norm, eo_spinor_field[2], eo_spinor_field[0], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators_clover] Q Q_tm^-1 even norm diff %e\n", norm);
  spinor_field_norm_diff( &norm, eo_spinor_field[3], eo_spinor_field[1], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators_clover] Q Q_tm^-1 odd  norm diff %e\n", norm);

  /* apply the eo-decomposed operator Q = g5 A B */
  Q_clover_eo_SchurDecomp_B (eo_spinor_field[4], eo_spinor_field[5], eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_phase, mzz[flavor_id][1], mzzinv[flavor_id][0], eo_spinor_work[2]);
  Q_clover_eo_SchurDecomp_A (eo_spinor_field[2], eo_spinor_field[3], eo_spinor_field[4], eo_spinor_field[5], gauge_field_with_phase, mzz[flavor_id][0], eo_spinor_work[2]);
  spinor_field_eq_gamma_ti_spinor_field(eo_spinor_field[2], 5, eo_spinor_field[2], Vhalf);
  spinor_field_eq_gamma_ti_spinor_field(eo_spinor_field[3], 5, eo_spinor_field[3], Vhalf);

  spinor_field_norm_diff( &norm, eo_spinor_field[2], eo_spinor_field[0], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators_clover] g5 A B Q_tm^-1 even norm diff %e\n", norm);
  spinor_field_norm_diff( &norm, eo_spinor_field[3], eo_spinor_field[1], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators_clover] g5 A B Q_tm^-1 odd  norm diff %e\n", norm);
#endif

#if 0
  /***********************************************************
   * TEST step-wise propagator construction with
   *   Q_clover_eo_SchurDecomp_Ainv, tmLQCD_invert_eo
   *   with eo return field and Q_clover_eo_SchurDecomp_Binv
   *   agains Q_clover_eo_SchurDecomp_B, Q_clover_eo_SchurDecomp_A
   *   and spinor_field_eq_gamma_ti_spinor_field 
   ***********************************************************/
  random_spinor_field (eo_spinor_field[0], Vhalf);
  random_spinor_field (eo_spinor_field[1], Vhalf);

  spinor_field_eq_gamma_ti_spinor_field(eo_spinor_field[2], 5, eo_spinor_field[0], Vhalf);
  spinor_field_eq_gamma_ti_spinor_field(eo_spinor_field[3], 5, eo_spinor_field[1], Vhalf);

  Q_clover_eo_SchurDecomp_Ainv (eo_spinor_field[2], eo_spinor_field[3], eo_spinor_field[2], eo_spinor_field[3], gauge_field_with_phase, mzzinv[flavor_id][0], eo_spinor_work[0]);
  memcpy(eo_spinor_work[0], eo_spinor_field[3], sizeof_eo_spinor_field);
  memset(eo_spinor_work[1], 0, sizeof_eo_spinor_field);
  exitstatus = tmLQCD_invert_eo(eo_spinor_work[1], eo_spinor_work[0], op_id);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_lm_propagator_clover] Error from tmLQCD_invert_eo, status was %d\n", exitstatus);
    EXIT(35);
  }
  memcpy(eo_spinor_field[3], eo_spinor_work[1], sizeof_eo_spinor_field);
  Q_clover_eo_SchurDecomp_Binv (eo_spinor_field[2], eo_spinor_field[3], eo_spinor_field[2], eo_spinor_field[3], gauge_field_with_phase, mzzinv[flavor_id][0], eo_spinor_work[0]);

  Q_clover_eo_SchurDecomp_B (eo_spinor_field[4], eo_spinor_field[5], eo_spinor_field[2], eo_spinor_field[3], gauge_field_with_phase, mzz[flavor_id][1], mzzinv[flavor_id][0], eo_spinor_work[0]);
  Q_clover_eo_SchurDecomp_A (eo_spinor_field[2], eo_spinor_field[3], eo_spinor_field[4], eo_spinor_field[5], gauge_field_with_phase, mzz[flavor_id][0], eo_spinor_work[0]);
  spinor_field_eq_gamma_ti_spinor_field(eo_spinor_field[2], 5, eo_spinor_field[2], Vhalf);
  spinor_field_eq_gamma_ti_spinor_field(eo_spinor_field[3], 5, eo_spinor_field[3], Vhalf);

  spinor_field_norm_diff( &norm, eo_spinor_field[2], eo_spinor_field[0], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators_clover] eo step-wise propagator even norm diff %e\n", norm);
  spinor_field_norm_diff( &norm, eo_spinor_field[3], eo_spinor_field[1], Vhalf);
  if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagators_clover] eo step-wise propagator odd  norm diff %e\n", norm);
#endif  /* of if 0 */

#if 0
  /**********************************************************
   * up-type full propagators
   **********************************************************/

  /* loop on spin-color components of point source */
  ratime = _GET_TIME;

  for(ia=0; ia<12; ia++) {

    /* A^-1 g5 source */
    exitstatus = init_clover_eo_spincolor_pointsource_propagator (eo_spinor_field[ia], eo_spinor_field[12+ia], gsx, ia, gauge_field_with_phase, mzzinv[flavor_id][0], source_proc_id==g_cart_id, eo_spinor_work[2]);
    if(exitstatus != 0 ) {
      fprintf(stderr, "[test_lm_propagator_clover] Error from init_eo_spincolor_pointsource_propagator; status was %d\n", exitstatus);
      EXIT(36);
    }

    /* C^-1 */
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] calling tmLQCD_invert_eo\n");
    memset(eo_spinor_work[1], 0, sizeof_eo_spinor_field);
    memcpy(eo_spinor_work[0], eo_spinor_field[12+ia], sizeof_eo_spinor_field);
    exitstatus = tmLQCD_invert_eo(eo_spinor_work[1], eo_spinor_work[0], op_id);
    if(exitstatus != 0) {
      fprintf(stderr, "[test_lm_propagator_clover] Error from tmLQCD_invert_eo, status was %d\n", exitstatus);
      EXIT(35);
    }
    memcpy(eo_spinor_field[12+ia], eo_spinor_work[1], sizeof_eo_spinor_field);

    /* B^-1 excl. C^-1 */
    exitstatus = fini_clover_eo_propagator (eo_spinor_field[ia], eo_spinor_field[12+ia], eo_spinor_field[ia], eo_spinor_field[12+ia], gauge_field_with_phase, mzzinv[flavor_id][0], eo_spinor_work[4]);
    if(exitstatus != 0) {
      fprintf(stderr, "[test_lm_propagator_clover] Error from fini_eo_propagator, status was %d\n", exitstatus);
      EXIT(35);
    }

#if 0
    /* TEST against Q_clover_phi_matrix_eo */
    memcpy(eo_spinor_work[0], eo_spinor_field[   ia], sizeof_eo_spinor_field);
    memcpy(eo_spinor_work[1], eo_spinor_field[12+ia], sizeof_eo_spinor_field);
    Q_clover_phi_matrix_eo (eo_spinor_work[2], eo_spinor_work[3], eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_phase, eo_spinor_work[4], mzz[flavor_id]);

    if(source_proc_id == g_cart_id) {
      unsigned int ix = _GSI(g_lexic2eosub[g_ipt[sx0][sx1][sx2][sx3]]);
      if(source_location_iseven) {
        eo_spinor_work[2][ix+2*ia] -= 1.;
      } else {
        eo_spinor_work[3][ix+2*ia] -= 1.;
      }
    }

    spinor_scalar_product_re( &norm, eo_spinor_work[2], eo_spinor_work[2], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] even norm diff %e\n", sqrt(norm));
    spinor_scalar_product_re( &norm, eo_spinor_work[3], eo_spinor_work[3], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] odd  norm diff %e\n", sqrt(norm));
#endif
  }  /* of loop on spin-color component ia */


  g_seq_source_momentum[0] = 1;
  g_seq_source_momentum[1] = 2;
  g_seq_source_momentum[2] = -3;
  g_sequential_source_timeslice = 7;
  g_sequential_source_gamma_id = 14;
  int g_sequential_source_gamma_sqr_sign = -1;

  for(ia=0; ia<12; ia++) {
    exitstatus = init_clover_eo_sequential_source(eo_spinor_field[24+ia], eo_spinor_field[36+ia], eo_spinor_field[ia], eo_spinor_field[12+ia],
        g_sequential_source_timeslice, gauge_field_with_phase, mzzinv[flavor_id][0], g_seq_source_momentum, g_sequential_source_gamma_id, eo_spinor_work[0]);
  }

  for(ia=0; ia<12; ia++) {
    /* multiply with A */
    Q_clover_eo_SchurDecomp_A (eo_spinor_field[24+ia], eo_spinor_field[36+ia], eo_spinor_field[24+ia], eo_spinor_field[36+ia], gauge_field_with_phase, mzz[flavor_id][0], eo_spinor_work[0]);

    /* multiply with g5 */
    g5_phi( eo_spinor_field[24+ia], Vhalf );
    g5_phi( eo_spinor_field[36+ia], Vhalf );

    /* multiply with gseq */
    spinor_field_eq_gamma_ti_spinor_field(eo_spinor_field[24+ia], g_sequential_source_gamma_id, eo_spinor_field[24+ia], Vhalf );
    spinor_field_eq_gamma_ti_spinor_field(eo_spinor_field[36+ia], g_sequential_source_gamma_id, eo_spinor_field[36+ia], Vhalf );
    spinor_field_ti_eq_re ( eo_spinor_field[24+ia], g_sequential_source_gamma_sqr_sign, Vhalf);
    spinor_field_ti_eq_re ( eo_spinor_field[36+ia], g_sequential_source_gamma_sqr_sign, Vhalf);


    double q[3] = { 2.*M_PI * g_seq_source_momentum[0] / LX_global, 2.*M_PI * g_seq_source_momentum[1] / LY_global, 2.*M_PI * g_seq_source_momentum[2] / LZ_global};
    
    double q_offset = q[0] * g_proc_coords[1] * LX + q[1] * g_proc_coords[2] * LY + q[2] * g_proc_coords[3] * LZ;
    int x0, x1, x2, x3;
    /* multiply with inverse phase */
    for( x0=0; x0<T; x0++ ) {
    for( x1=0; x1<LX; x1++ ) {
    for( x2=0; x2<LY; x2++ ) {
    for( x3=0; x3<LZ; x3++ ) {
      unsigned int ix = g_ipt[x0][x1][x2][x3];
      unsigned int ixeosub = g_lexic2eosub[ix];
      double q_phase = q[0] * x1 + q[1] * x2 + q[2] * x3 + q_offset;
      double spinor1[24];
      complex w = { cos(q_phase), sin(-q_phase) };
      double *s_ = g_iseven[ix] ? eo_spinor_field[24+ia]+_GSI(ixeosub) : eo_spinor_field[36+ia]+_GSI(ixeosub);
      _fv_eq_fv(spinor1, s_ );
      _fv_eq_fv_ti_co(s_, spinor1, &w);
    }}}}

    if(  g_sequential_source_timeslice / T == g_proc_coords[0] ) {
      unsigned int VOL3half = LX * LY * LZ / 2;
      unsigned int offset = _GSI( ( g_sequential_source_timeslice % T ) * VOL3half );
      spinor_field_mi_eq_spinor_field_ti_re( eo_spinor_field[24+ia]+offset, eo_spinor_field[   ia]+offset, 1., VOL3half );
      spinor_field_mi_eq_spinor_field_ti_re( eo_spinor_field[36+ia]+offset, eo_spinor_field[12+ia]+offset, 1., VOL3half );
    }

    spinor_scalar_product_re(&norm, eo_spinor_field[24+ia], eo_spinor_field[24+ia], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] even part norm = %e\n", sqrt(norm));
    spinor_scalar_product_re(&norm, eo_spinor_field[36+ia], eo_spinor_field[36+ia], Vhalf);
    if(g_cart_id == 0) fprintf(stdout, "# [test_lm_propagator_clover] odd  part norm = %e\n", sqrt(norm));

  }  /* end of loop on spin-color */

#endif  /* of if 0 */
  


#if 0
  exitstatus = project_propagator_field(eo_spinor_field[36], eo_spinor_field[36], 1, eo_evecs_block[op_id], 12, evecs_num, Vhalf);
  if(exitstatus != 0) {
    fprintf(stderr, "[test_lm_propagator_clover] Error from project_propagator_field, status was %d\n", exitstatus);
    EXIT(35);
  }

  for(ia=0; ia<12; ia++) {
    memset(eo_spinor_work[1], 0, sizeof_eo_spinor_field);
    memcpy(eo_spinor_work[0], eo_spinor_field[36+ia], sizeof_eo_spinor_field);
    exitstatus = tmLQCD_invert_eo(eo_spinor_work[1], eo_spinor_work[0], op_id);
    if(exitstatus != 0) {
      fprintf(stderr, "[test_lm_propagator_clover] Error from tmLQCD_invert_eo, status was %d\n", exitstatus);
      EXIT(35);
    }
    memcpy(eo_spinor_field[36+ia], eo_spinor_work[1], sizeof_eo_spinor_field);
    fini_clover_eo_propagator(eo_spinor_field[24+ia], eo_spinor_field[36+ia], eo_spinor_field[24+ia], eo_spinor_field[36+ia], mzzinv[flavor_id][0], eo_spinor_work[0]);
  }

  for(ia=0; ia<12; ia++) {
    exitstatus = init_clover_eo_sequential_source(eo_spinor_field[ia], eo_spinor_field[12+ia], eo_spinor_field[ia], eo_spinor_field[12+ia],
        g_sequential_source_timeslice, mzzinv[flavor_id][0], g_seq_source_momentum, g_sequential_source_gamma_id, eo_spinor_work[0]);
  }

  init_2level_buffer(&pcoeff, 12, 2*evecs_num);

  /* odd projection coefficients v^+ sp_o */

  if(flavor_id == 1) {
    for(ia=0; ia<12; ia++) {
      memcpy(eo_spinor_work[0], eo_spinor_field[12+ia], sizeof_eo_spinor_field);
      C_clover_oo (eo_spinor_field[12+ia], eo_spinor_work[0], g_gauge_field, eo_spinor_work[2], mzz[1-flavor_id][1], mzzinv[1-flavor_id][0]);
    }
  }
  exitstatus = project_reduce_from_propagator_field (pcoeff[0], eo_spinor_field[12], eo_evecs_block[0], 12, evecs_num, Vhalf);

  for(ia=0; ia<12; ia++) {
    for(i=0; i<evecs_num; i++) {
      norm = 2.*g_kappa / evecs_eval[i];
      pcoeff[ia][2*i  ] *= norm;
      pcoeff[ia][2*i+1] *= norm;
    }
  }

  exitstatus = project_expand_to_propagator_field(eo_spinor_field[12], pcoeff[0], eo_evecs_block[0], 12, evecs_num, Vhalf);
  if(flavor_id == 0) {
    for(ia=0; ia<12; ia++) {
      memcpy(eo_spinor_work[0], eo_spinor_field[12+ia], sizeof_eo_spinor_field);
      C_clover_oo (eo_spinor_field[12+ia], eo_spinor_work[0], g_gauge_field, eo_spinor_work[2], mzz[1-flavor_id][1], mzzinv[1-flavor_id][0]);
    }
  }
  fini_2level_buffer(&pcoeff);

  for(ia=0; ia<12; ia++) {
    Q_clover_eo_SchurDecomp_Binv (eo_spinor_field[ia], eo_spinor_field[12+ia], eo_spinor_field[ia], eo_spinor_field[12+ia], g_gauge_field, mzzinv[flavor_id][0], eo_spinor_work[4]);
  }

  for(ia=0; ia<12; ia++) {
    spinor_field_norm_diff(&norm, eo_spinor_field[ia], eo_spinor_field[24+ia], Vhalf);
    if(g_cart_id == 0) printf("# even norm %2d %16.7e\n", ia, norm);
    spinor_field_norm_diff(&norm, eo_spinor_field[12+ia], eo_spinor_field[36+ia], Vhalf);
    if(g_cart_id == 0) printf("# odd  norm %2d %16.7e\n", ia, norm);
  }

  retime = _GET_TIME;
  if(g_cart_id == 0) {
    fprintf(stdout, "# [test_lm_propagator_clover] time for propagators = %e seconds\n", retime-ratime);
  }
#endif  /* of if 0 */

  /****************************************
   * free the allocated memory, finalize
   ****************************************/

  clover_term_fini(&g_clover);
  clover_term_fini(&g_mzz_up);
  clover_term_fini(&g_mzz_dn);
  clover_term_fini(&g_mzzinv_up);
  clover_term_fini(&g_mzzinv_dn);

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
  if( evecs_eval          != NULL ) free( evecs_eval );

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
/*  mpi_fini_xchange_contraction(); */
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_lm_propagator_clover] %s# [test_lm_propagator_clover] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_lm_propagator_clover] %s# [test_lm_propagator_clover] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
