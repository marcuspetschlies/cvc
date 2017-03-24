/****************************************************
 * ama_high.cpp
 *
 * So 10. Jul 21:08:26 CEST 2016
 *
 * PURPOSE:
 * TODO:
 * - mixed gsps V / XbarV / W / XW with xi / Xbar xi / phi / X phi
 * - purely stochastic gsps
 * DONE:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <getopt.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#define MAIN_PROGRAM

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef HAVE_TMLQCD_LIBWRAPPER
#include "tmLQCD.h"
#endif

#ifdef __cplusplus
}
#endif

#include "types.h"
#include "ilinalg.h"
#include "global.h"
#include "cvc_complex.h"
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
#include "scalar_products.h"
#include "gsp.h"
#include "prepare_source.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, status, isample;
  int i, k, ievecs;
  int filename_set = 0;
  int unpack_spinor_field = 0;
  int check_eigenvectors = 0;
  int threadid, nthreads;
  unsigned int Vhalf;

  int evecs_num=0;
  double *evecs_eval = NULL;
  double evecs_lambda;

  double norm, dtmp;
  complex w;

  char gsp_tag[100];

  double plaq=0.;
  int verbose = 0;
  char filename[200];

  double **eo_spinor_field=NULL, *eo_spinor_work0=NULL, *eo_spinor_work1=NULL, *eo_spinor_work2 = NULL, *eo_spinor_work3=NULL;
  double *eo_evecs_block[2];
  double *full_spinor_field[1];
  double **eo_evecs_field=NULL;
  double ratime, retime;

  size_t sizeof_eo_spinor_field;


  int op_id = -1;


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif


  while ((c = getopt(argc, argv, "uch?vf:n:o:")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'n':
      evecs_num = atoi(optarg);
      fprintf(stdout, "# [ama_high] dimension of eigenspace set to %d\n", evecs_num);
      break;
    case 'c':
      check_eigenvectors = 1;
      fprintf(stdout, "# [ama_high] check eigenvectors set to %d\n", check_eigenvectors);
      break;
    case 'u':
      unpack_spinor_field = 1;
      fprintf(stdout, "# [ama_high] unpack_spinor_field set to %d\n", unpack_spinor_field);
      break;
    case 'o':
      op_id = atoi(optarg);
      fprintf(stdout, "# [ama_high] set operator id to %d\n", op_id);
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
  if(g_cart_id==0) fprintf(stdout, "# [ama_high] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [ama_high] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  status = tmLQCD_invert_init(argc, argv, 1);
  if(status != 0) {
    EXIT(1);
  }
  status = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(status != 0) {
    EXIT(2);
  }
  status = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(status != 0) {
    EXIT(3);
  }
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /* initialize geometry */
  if(init_geometry() != 0) {
    fprintf(stderr, "[ama_high] Error from init_geometry\n");
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

  Vhalf = VOLUME / 2;
  sizeof_eo_spinor_field = 24 * Vhalf * sizeof(double);

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [ama_high] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private(nthreads,threadid)
{
  threadid = omp_get_thread_num();
  nthreads = omp_get_num_threads();
  fprintf(stdout, "# [ama_high] proc%.4d thread%.4d using %d threads\n", g_cart_id, threadid, nthreads);
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[ama_high] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

#ifndef HAVE_TMLQCD_LIBWRAPPER
  if (evecs_num == 0) {
    if(g_cart_id==0) fprintf(stderr, "[ama_high] eigenspace dimension is 0\n");
    EXIT(5);
  }
#endif

#ifdef HAVE_TMLQCD_LIBWRAPPER
  if(g_cart_id == 0) fprintf(stdout, "# [ama_high] calling tmLQCD_get_gauge_field_pointer\n");
  status = tmLQCD_read_gauge(Nconf);
  if (status != 0) {
    fprintf(stderr, "[ama_high] Error from tmLQCD_read_gauge, status was %d\n", status);
    EXIT(33);
  }

  status = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
  if (status != 0) {
    fprintf(stderr, "[ama_high] Error from tmLQCD_get_gauge_field_pointer, status was %d\n", status);
    EXIT(29);
  }
  if (g_gauge_field == NULL) {
    fprintf(stderr, "[ama_high] Error, gauge field pointer is NULL\n");
    EXIT(32);
  }
#else
  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [ama_high] reading gauge field from file %s\n", filename);

  if(strcmp(gaugefilename_prefix,"identity")==0) {
    status = unit_gauge_field(g_gauge_field, VOLUME);
#ifdef HAVE_MPI
    xchange_gauge();
#endif
  } else if(strcmp(gaugefilename_prefix, "NA") != 0) {
    /* status = read_nersc_gauge_field_3x3(g_gauge_field, filename, &plaq); */
    /* status = read_ildg_nersc_gauge_field(g_gauge_field, filename); */
    status = read_lime_gauge_field_doubleprec(filename);
    /* status = read_nersc_gauge_field(g_gauge_field, filename, &plaq); */
#ifdef HAVE_MPI
    xchange_gauge();
#endif
    if(status != 0) {
      fprintf(stderr, "[ama_high] Error, could not read gauge field\n");
      EXIT(7);
    }
  } else {
    if(g_cart_id == 0) fprintf(stderr, "# [ama_high] need gauge field\n");
    EXIT(8);
  }
#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */

  /* measure the plaquette */
  if(g_cart_id==0) fprintf(stdout, "# [ama_high] read plaquette value 1st field: %25.16e\n", plaq);
  plaquette(&plaq);
  if(g_cart_id==0) fprintf(stdout, "# [ama_high] measured plaquette value 1st field: %25.16e\n", plaq);


#ifdef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * retrieve deflator paramters from tmLQCD
   ***********************************************/

  status = tmLQCD_init_deflator(op_id);
  if( status > 0) {
    fprintf(stderr, "[ama_high] Error from tmLQCD_init_deflator, status was %d\n", status);
    EXIT(9);
  }


  status = tmLQCD_get_deflator_params(&g_tmLQCD_defl, op_id);
  if(status != 0) {
    fprintf(stderr, "[ama_high] Error from tmLQCD_get_deflator_params, status was %d\n", status);
    EXIT(30);
  }

  if(g_cart_id == 1) {
    fprintf(stdout, "# [ama_high] deflator type name = %s\n", g_tmLQCD_defl.type_name);
    fprintf(stdout, "# [ama_high] deflator eo prec   = %d\n", g_tmLQCD_defl.eoprec);
    fprintf(stdout, "# [ama_high] deflator precision = %d\n", g_tmLQCD_defl.prec);
    fprintf(stdout, "# [ama_high] deflator nev       = %d\n", g_tmLQCD_defl.nev);
  }

  eo_evecs_block[0] = (double*)(g_tmLQCD_defl.evecs);
  if(eo_evecs_block[0] == NULL) {
    fprintf(stderr, "[ama_high] Error, eo_evecs_block is NULL\n");
    EXIT(32);
  }
  evecs_num = g_tmLQCD_defl.nev;
  if(evecs_num == 0) {
    fprintf(stderr, "[ama_high] Error, dimension of eigenspace is zero\n");
    EXIT(33);
  }

#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */



  /***********************************************
   * allocate spinor fields
   ***********************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  eo_evecs_block[0] = (double*)malloc(evecs_num*24*Vhalf*sizeof(double));
  if(eo_evecs_block[0] == NULL) {
    fprintf(stderr, "[ama_high] Error from malloc\n");
    EXIT(25);
  }
#endif  /* of ifdef HAVE_TMLQCD_LIBWRAPPER */
  eo_evecs_field = (double**)calloc(evecs_num, sizeof(double*));
  eo_evecs_field[0] = eo_evecs_block[0];
  for(i=1; i<evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + 24*Vhalf;

  /* 2*g_nsample + 4 fields */
  eo_spinor_field    = (double**)calloc(2*g_nsample+4, sizeof(double*));

  /* first block g_nsample fields */
  eo_spinor_field[0] = (double*)calloc( g_nsample*24*Vhalf, sizeof(double) );
  if(eo_spinor_field[0] == NULL) {
    fprintf(stderr, "[ama_high] Error from calloc\n");
    EXIT(35);
  }
  /* second block of g_nsample fields */
  for(i=1; i<g_nsample; i++) eo_spinor_field[i] = eo_spinor_field[i-1] + Vhalf*24;
  eo_spinor_field[g_nsample] = (double*)calloc( g_nsample*24*Vhalf, sizeof(double) );
  if(eo_spinor_field[g_nsample] == NULL) {
    fprintf(stderr, "[ama_high] Error from calloc\n");
    EXIT(36);
  }
  for(i=g_nsample+1; i<2*g_nsample; i++) eo_spinor_field[i] = eo_spinor_field[i-1] + Vhalf*24;

  alloc_spinor_field(&eo_spinor_field[2*g_nsample], (VOLUME+RAND));
  eo_spinor_field[2*g_nsample+1] = eo_spinor_field[2*g_nsample] + 12*(VOLUME+RAND);

  alloc_spinor_field(&eo_spinor_field[2*g_nsample+2], (VOLUME+RAND));
  eo_spinor_field[2*g_nsample+3] = eo_spinor_field[2*g_nsample+2] + 12*(VOLUME+RAND);

  /* work spinor fields, last four spinor fields */
  full_spinor_field[0] = eo_spinor_field[2*g_nsample];
  eo_spinor_work0 = eo_spinor_field[2*g_nsample];
  eo_spinor_work1 = eo_spinor_field[2*g_nsample+1];
  eo_spinor_work2 = eo_spinor_field[2*g_nsample+2];
  eo_spinor_work3 = eo_spinor_field[2*g_nsample+3];

#ifndef HAVE_TMLQCD_LIBWRAPPER
  /***********************************************
   * read eo eigenvectors
   ***********************************************/
  for(ievecs = 0; ievecs<evecs_num; ievecs+=2) {
    
    sprintf(filename, "%s%.5d", filename_prefix, ievecs);
    if(g_cart_id == 0) fprintf(stdout, "# [ama_high] reading C_oo_sym eigenvector from file %s\n", filename);

    status = read_lime_spinor(full_spinor_field[0], filename, 0);
    if( status != 0) {
      fprintf(stderr, "[ama_high] Error from read_lime_spinor, status was %d\n", status);
      EXIT(9);
    }

    ratime = _GET_TIME;
    if(unpack_spinor_field) {
      spinor_field_unpack_lexic2eo (full_spinor_field[0], eo_evecs_field[ievecs], eo_evecs_field[ievecs+1]);
    } else {
      spinor_field_lexic2eo (full_spinor_field[0], eo_evecs_field[ievecs], eo_evecs_field[ievecs+1]);
    }
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [ama_high] time for unpacking = %e seconds\n", retime - ratime);

  }  /* end of loop on evecs number */
#endif  /* of ifndef HAVE_TMLQCD_LIBWRAPPER */

  /***********************************************
   * allocate memory for eigenvalues
   ***********************************************/
  evecs_eval = (double*)malloc(evecs_num * sizeof(double));
  if(evecs_eval == NULL) {
    fprintf(stderr, "[ama_high] Error from malloc\n");
    EXIT(6);
  }

  if (check_eigenvectors) {

    /***********************************************
     * check eigenvector equation
     ***********************************************/
    for(ievecs = 0; ievecs<evecs_num; ievecs++) {
  
      ratime = _GET_TIME;

      /* 0 <- V */
      memcpy(eo_spinor_work0, eo_evecs_field[ievecs], sizeof_eo_spinor_field);

      C_oo(eo_spinor_work1, eo_spinor_work0, g_gauge_field, -g_mu, eo_spinor_work3);

      C_oo(eo_spinor_work2, eo_spinor_work1, g_gauge_field,  g_mu, eo_spinor_work3);
  
      norm = 4 * g_kappa * g_kappa;
      spinor_field_ti_eq_re (eo_spinor_work2, norm, Vhalf);
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [ama_high] time to apply C_oo_sym = %e seconds\n", retime - ratime);
  
      /* determine eigenvalue */
      spinor_scalar_product_re(&norm,  eo_spinor_work0, eo_spinor_work0, Vhalf);
      spinor_scalar_product_co(&w,  eo_spinor_work0, eo_spinor_work2, Vhalf);
      evecs_lambda = w.re / norm;
      if(g_cart_id == 0) {
        fprintf(stdout, "# [ama_high] estimated eigenvalue(%d) = %25.16e\n", ievecs, evecs_lambda);
      }
  
      /* check evec equation */
      spinor_field_mi_eq_spinor_field_ti_re(eo_spinor_work2, eo_spinor_work0, evecs_lambda, Vhalf);

      spinor_scalar_product_re(&norm, eo_spinor_work2, eo_spinor_work2, Vhalf);
      if(g_cart_id == 0) {
        fprintf(stdout, "# [ama_high] eigenvector(%d) || A x - lambda x || = %16.7e\n", ievecs, sqrt(norm) );
      }
    }  /* end of loop on evecs_num */

  }  /* end of if check_eigenvectors */

  /***********************************************
   * init ranlxd random number generator
   ***********************************************/
  status = init_rng_stat_file (g_seed, NULL);
  if(status != 0) {
    fprintf(stderr, "[ama_high] Error from init_rng_stat_file, status was %d\n", status);
    EXIT(32);
  }

  /***********************************************
   * loop on stochastic samples
   ***********************************************/
  for(isample = 0; isample < g_nsample; isample++) {

#ifdef HAVE_TMLQCD_LIBWRAPPER
    /***********************************************
     * prepare a volume source
     ***********************************************/
    status = prepare_volume_source(eo_spinor_work0 , VOLUME/2);
    if(status != 0) {
      fprintf(stderr, "[ama_high] Error from prepare_volume_source, status was %d\n", status);
      EXIT(33);
    }


    /***********************************************
     * orthogonal projection of source
     ***********************************************/
 
    status = project_spinor_field(eo_spinor_field[isample], eo_spinor_work0, 0, eo_evecs_block[0], evecs_num, VOLUME/2);
    if(status != 0) {
      fprintf(stderr, "[ama_high] Error from project_spinor_field, status was %d\n", status);
      EXIT(34);
    }

    /* TEST */
    for(i=0; i<evecs_num; i++) {
      spinor_scalar_product_co(&w, eo_evecs_field[i], eo_spinor_field[isample], Vhalf);
      if(g_cart_id == 0) {
        fprintf(stdout, "# [ama_high] before %3d %25.16e %25.16e\n", i, w.re, w.im);
      }
    }


    /***********************************************
     * invert
     ***********************************************/
    memcpy(eo_spinor_work0, eo_spinor_field[isample], sizeof_eo_spinor_field);
    if(g_cart_id == 0) fprintf(stdout, "# [ama_high] calling tmLQCD_invert_eo\n");
    status = tmLQCD_invert_eo(eo_spinor_work1, eo_spinor_work0, op_id);
    if(status != 0) {
      fprintf(stderr, "[ama_high] Error from tmLQCD_invert_eo, status was %d\n", status);
      EXIT(35);
    }
    memcpy(eo_spinor_field[g_nsample+isample], eo_spinor_work1, sizeof_eo_spinor_field);


    /* TEST */
    for(i=0; i<evecs_num; i++) {
      spinor_scalar_product_co(&w, eo_evecs_field[i], eo_spinor_work1, Vhalf);
      if(g_cart_id == 0) {
        fprintf(stdout, "# [ama_high] after inversion %3d %25.16e %25.16e\n", i, w.re, w.im);
      }
    }
#endif   /* of if HAVE_TMLQCD_LIBWRAPPER */

    /***********************************************
     * check residuum, apply C_oo Cbar_oo
     ***********************************************/

    C_oo(eo_spinor_work2, eo_spinor_work1, g_gauge_field,  g_mu, eo_spinor_work3);
    norm = 2. * g_kappa;
    spinor_field_ti_eq_re (eo_spinor_work2, norm, Vhalf);
    spinor_field_norm_diff (&norm, eo_spinor_work2, eo_spinor_work0, Vhalf);
    spinor_scalar_product_re (&dtmp, eo_spinor_work0, eo_spinor_work0, Vhalf);

    if(g_cart_id == 0) { fprintf(stdout, "# [ama_high] abs norm diff = %e; rel norm diff = %e\n", norm, norm / sqrt(dtmp)); }

#if 0
    C_oo(eo_spinor_work2, eo_spinor_work1, g_gauge_field, -g_mu, eo_spinor_work3);
    C_oo(eo_spinor_work1, eo_spinor_work2, g_gauge_field,  g_mu, eo_spinor_work3);

    norm = 4. * g_kappa * g_kappa;
    spinor_field_ti_eq_re (eo_spinor_work1, norm, Vhalf);
    spinor_field_norm_diff (&norm, eo_spinor_work1, eo_spinor_work0, Vhalf);
    if(g_cart_id == 0) { fprintf(stdout, "# [ama_high] norm diff = %e\n", norm); }
#endif
  }  /* end of loop on samples */


  /***********************************************
   * scalar products
   ***********************************************/

  /* (1) xi^+ xi */
  sprintf(gsp_tag, "%s.%.4d", "gsp_xi_xi", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_high] calculating %s\n", gsp_tag);
  status = gsp_calculate_v_dag_gamma_p_w_block(&(eo_spinor_field[0]), &(eo_spinor_field[0]), g_nsample, g_source_momentum_number, g_source_momentum_list, g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag);
  if(status != 0) {
    fprintf(stderr, "[ama_high] Error from gsp_calculate_v_dag_gamma_p_w_block, status was %d\n", status);
    EXIT(11);
  }

  /* (2) xi^+ phi */
  sprintf(gsp_tag, "%s.%.4d", "gsp_xi_phi", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_high] calculating %s\n", gsp_tag);
  status = gsp_calculate_v_dag_gamma_p_w_block(&(eo_spinor_field[0]), &(eo_spinor_field[g_nsample]), g_nsample, g_source_momentum_number, g_source_momentum_list, g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag);
  if(status != 0) {
    fprintf(stderr, "[ama_high] Error from gsp_calculate_v_dag_gamma_p_w_block, status was %d\n", status);
    EXIT(12);
  }

  /* (3) phi^+ phi */
  sprintf(gsp_tag, "%s.%.4d", "gsp_phi_phi", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_high] calculating %s\n", gsp_tag);
  status = gsp_calculate_v_dag_gamma_p_w_block(&(eo_spinor_field[g_nsample]), &(eo_spinor_field[g_nsample]), g_nsample, g_source_momentum_number, g_source_momentum_list, g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag);
  if(status != 0) {
    fprintf(stderr, "[ama_high] Error from gsp_calculate_v_dag_gamma_p_w_block, status was %d\n", status);
    EXIT(14);
  }

  /* (4) (xbar_xi)^+ (xbar xi) */
  for(isample = 0; isample<g_nsample; isample++) {
    ratime = _GET_TIME;
    /* 0 <- eo_spinor_field */
    memcpy(eo_spinor_work0, eo_spinor_field[isample], sizeof_eo_spinor_field);
    /* 1 <- Xbar 0 */
    X_eo (eo_spinor_work1, eo_spinor_work0, -g_mu, g_gauge_field);
    /* XV <- 1 */
    memcpy(eo_spinor_field[isample], eo_spinor_work1, sizeof_eo_spinor_field);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [ama_high] time for X_eo = %e seconds\n", retime-ratime);
  }
  sprintf(gsp_tag, "%s.%.4d", "gsp_xbareoxi_xbareoxi", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_high] calculating %s\n", gsp_tag);
  status = gsp_calculate_v_dag_gamma_p_w_block(&(eo_spinor_field[0]), &(eo_spinor_field[0]), g_nsample, g_source_momentum_number, g_source_momentum_list, g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag);
  if(status != 0) {
    fprintf(stderr, "[ama_high] Error from gsp_calculate_v_dag_gamma_p_w_block, status was %d\n", status);
    EXIT(15);
  }

  /* (5) (x_phi)^+ (x phi) */
  for(isample = 0; isample<g_nsample; isample++) {
    ratime = _GET_TIME;
    /* 0 <- eo_spinor_field */
    memcpy(eo_spinor_work0, eo_spinor_field[g_nsample+isample], sizeof_eo_spinor_field);
    /* 1 <- X 0 */
    X_eo (eo_spinor_work1, eo_spinor_work0, g_mu, g_gauge_field);
    /* eo_spinor_field <- 1 */
    memcpy(eo_spinor_field[g_nsample+isample], eo_spinor_work1, sizeof_eo_spinor_field);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [ama_high] time for X_eo = %e seconds\n", retime-ratime);
  }
  sprintf(gsp_tag, "%s.%.4d", "gsp_xeophi_xeophi", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_high] calculating %s\n", gsp_tag);
  status = gsp_calculate_v_dag_gamma_p_w_block(&(eo_spinor_field[g_nsample]), &(eo_spinor_field[g_nsample]), g_nsample, g_source_momentum_number, g_source_momentum_list, g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag);
  if(status != 0) {
    fprintf(stderr, "[ama_high] Error from gsp_calculate_v_dag_gamma_p_w_block, status was %d\n", status);
    EXIT(16);
  }

  /* (Xbar xi)^+ (X phi) */
  sprintf(gsp_tag, "%s.%.4d", "gsp_xbareoxi_xeophi", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_high] calculating %s\n", gsp_tag);
  status = gsp_calculate_v_dag_gamma_p_w_block(&(eo_spinor_field[0]), &(eo_spinor_field[g_nsample]), g_nsample, g_source_momentum_number, g_source_momentum_list, g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag);
  if(status != 0) {
    fprintf(stderr, "[ama_high] Error from gsp_calculate_v_dag_gamma_p_w_block, status was %d\n", status);
    EXIT(17);
  }

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  if(evecs_eval != NULL) free(evecs_eval);
#ifndef HAVE_TMLQCD_LIBWRAPPER
  if(g_gauge_field != NULL) free(g_gauge_field);
#endif

  if(eo_spinor_field != NULL) {
    if(eo_spinor_field[0]             != NULL) free(eo_spinor_field[0]);
    if(eo_spinor_field[g_nsample]     != NULL) free(eo_spinor_field[g_nsample]);
    if(eo_spinor_field[2*g_nsample]   != NULL) free(eo_spinor_field[2*g_nsample]);
    if(eo_spinor_field[2*g_nsample+2] != NULL) free(eo_spinor_field[2*g_nsample+2]);
    free(eo_spinor_field);
  }

  if(eo_evecs_field != NULL) {
    free(eo_evecs_field);
    eo_evecs_field = NULL;
  }
#ifndef HAVE_TMLQCD_LIBWRAPPER
  if (eo_evecs_block[0] != NULL) {
    free(eo_evecs_block[0]);
    eo_evecs_block[0] = NULL;
  }
#endif

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
#endif

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [ama_high] %s# [ama_high] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [ama_high] %s# [ama_high] end fo run\n", ctime(&g_the_time));
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

