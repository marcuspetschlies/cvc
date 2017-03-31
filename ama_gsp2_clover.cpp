/****************************************************
 * ama_gsp2_clover.cpp
 *
 * Tue Jul  5 11:58:18 CEST 2016
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

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

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
#include "scalar_products.h"
#include "gsp.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, status;
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
  double **eo_evecs_block=NULL;
  double *full_spinor_field[1];
  double **eo_evecs_field=NULL;
  double **clover=NULL, **mzz_up=NULL, **mzz_dn=NULL, **mzzinv_up=NULL, **mzzinv_dn=NULL;
  double ratime, retime;

  size_t sizeof_eo_spinor_field;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif


  while ((c = getopt(argc, argv, "uch?vf:n:")) != -1) {
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
      fprintf(stdout, "# [ama_gsp2_clover] dimension of eigenspace set to%d\n", evecs_num);
      break;
    case 'c':
      check_eigenvectors = 1;
      fprintf(stdout, "# [ama_gsp2_clover] check eigenvectors set to %d\n", check_eigenvectors);
      break;
    case 'u':
      unpack_spinor_field = 1;
      fprintf(stdout, "# [ama_gsp2_clover] unpack_spinor_field set to %d\n", unpack_spinor_field);
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
  if(g_cart_id==0) fprintf(stdout, "# [ama_gsp2_clover] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [ama_gsp2_clover] calling tmLQCD wrapper init functions\n");

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
    fprintf(stderr, "[ama_gsp2_clover] ERROR from init_geometry\n");
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

  Vhalf = VOLUME / 2;
  sizeof_eo_spinor_field = 24 * Vhalf * sizeof(double);

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private(nthreads,threadid)
{
  threadid = omp_get_thread_num();
  nthreads = omp_get_num_threads();
  fprintf(stdout, "# [ama_gsp2_clover] proc%.4d thread%.4d using %d threads\n", g_cart_id, threadid, nthreads);
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[ama_gsp2_clover] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if (evecs_num == 0) {
    if(g_cart_id==0) fprintf(stderr, "[ama_gsp2_clover] eigenspace dimension is 0\n");
    EXIT(5);
  }
  evecs_eval = (double*)malloc(evecs_num * sizeof(double));
  if(evecs_eval == NULL) {
    fprintf(stderr, "[ama_gsp2_clover] Error from malloc\n");
    EXIT(6);
  }

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [ama_gsp2_clover] reading gauge field from file %s\n", filename);

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
      fprintf(stderr, "[ama_gsp2_clover] Error, could not read gauge field\n");
      EXIT(7);
    }

    /* measure the plaquette */
    if(g_cart_id==0) fprintf(stdout, "# [ama_gsp2_clover] read plaquette value 1st field: %25.16e\n", plaq);
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "# [ama_gsp2_clover] measured plaquette value 1st field: %25.16e\n", plaq);
  } else {
    if(g_cart_id == 0) fprintf(stderr, "# [ama_gsp2_clover] need gauge field\n");
    EXIT(8);
  }

  /***********************************************
   * allocate spinor fields
   ***********************************************/

  eo_evecs_block = (double**)malloc(2*sizeof(double*));
  eo_evecs_block[0] = (double*)malloc(evecs_num*24*Vhalf*sizeof(double));
  if(eo_evecs_block[0] == 0) {
    fprintf(stderr, "[ama_gsp2_clover] Error from malloc\n");
    EXIT(25);
  }
  eo_evecs_block[1] = (double*)malloc(evecs_num*24*Vhalf*sizeof(double));
  if(eo_evecs_block[1] == 0) {
    fprintf(stderr, "[ama_gsp2_clover] Error from malloc\n");
    EXIT(25);
  }

  eo_evecs_field = (double**)calloc(2*evecs_num, sizeof(double*));
  eo_evecs_field[0] = eo_evecs_block[0];
  for(i=1; i<evecs_num; i++) eo_evecs_field[i] = eo_evecs_field[i-1] + 24*Vhalf;
  eo_evecs_field[evecs_num] = eo_evecs_block[1];
  for(i=1; i<evecs_num; i++) eo_evecs_field[i+evecs_num] = eo_evecs_field[i-1+evecs_num] + 24*Vhalf;

  eo_spinor_field = (double**)calloc(4, sizeof(double*));
  alloc_spinor_field(&eo_spinor_field[0], (VOLUME+RAND));
  eo_spinor_field[1] = eo_spinor_field[0] + 12*(VOLUME+RAND);

  alloc_spinor_field(&eo_spinor_field[2], (VOLUME+RAND));
  eo_spinor_field[3] = eo_spinor_field[2] + 12*(VOLUME+RAND);

  full_spinor_field[0] = eo_spinor_field[0];
  eo_spinor_work0 = eo_spinor_field[0];
  eo_spinor_work1 = eo_spinor_field[1];
  eo_spinor_work2 = eo_spinor_field[2];
  eo_spinor_work3 = eo_spinor_field[3];

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  clover_term_init(&clover, 6);

  clover_term_init(&mzz_up, 6);
  clover_term_init(&mzz_dn, 6);
  clover_term_init(&mzzinv_up, 8);
  clover_term_init(&mzzinv_dn, 8);

  ratime = _GET_TIME;
  clover_term_eo (clover, g_gauge_field);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] time for clover_term_eo = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_matrix (mzz_up, clover, g_mu, g_csw);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] time for clover_mzz_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_matrix (mzz_dn, clover, -g_mu, g_csw);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] time for clover_mzz_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_inv_matrix (mzzinv_up, mzz_up);
  retime = _GET_TIME;
  fprintf(stdout, "# [ama_gsp2_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_inv_matrix (mzzinv_dn, mzz_dn);
  retime = _GET_TIME;
  fprintf(stdout, "# [ama_gsp2_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);

  /***********************************************
   * read eo eigenvectors
   ***********************************************/
  for(ievecs = 0; ievecs<evecs_num; ievecs+=2) {
    
    sprintf(filename, "%s%.5d", filename_prefix, ievecs);
    if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] reading C_oo_sym eigenvector from file %s\n", filename);

    status = read_lime_spinor(full_spinor_field[0], filename, 0);
    if( status != 0) {
      fprintf(stderr, "[ama_gsp2_clover] Error from read_lime_spinor, status was %d\n", status);
      EXIT(9);
    }

    ratime = _GET_TIME;
    if(unpack_spinor_field) {
      spinor_field_unpack_lexic2eo (full_spinor_field[0], eo_evecs_field[ievecs], eo_evecs_field[ievecs+1]);
    } else {
      spinor_field_lexic2eo (full_spinor_field[0], eo_evecs_field[ievecs], eo_evecs_field[ievecs+1]);
    }
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] time for unpacking = %e seconds\n", retime - ratime);

  }  /* end of loop on evecs number */

  if (check_eigenvectors) {

    /***********************************************
     * check eigenvector equation
     ***********************************************/
    for(ievecs = 0; ievecs<evecs_num; ievecs++) {
  
      ratime = _GET_TIME;

      /* 0 <- V */
      memcpy(eo_spinor_work0, eo_evecs_field[ievecs], sizeof_eo_spinor_field);

      C_clover_oo(eo_spinor_work1, eo_spinor_work0, g_gauge_field, eo_spinor_work3, mzz_dn[1], mzzinv_dn[0]);

      C_clover_oo(eo_spinor_work2, eo_spinor_work1, g_gauge_field, eo_spinor_work3, mzz_up[1], mzzinv_up[0]);

      norm = 4 * g_kappa * g_kappa;
      spinor_field_ti_eq_re (eo_spinor_work2, norm, Vhalf);
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] time to apply C_oo_sym = %e seconds\n", retime - ratime);
  
      /* determine eigenvalue */
      spinor_scalar_product_re(&norm,  eo_spinor_work0, eo_spinor_work0, Vhalf);
      spinor_scalar_product_co(&w,  eo_spinor_work0, eo_spinor_work2, Vhalf);
      evecs_lambda = w.re / norm;
      if(g_cart_id == 0) {
        fprintf(stdout, "# [ama_gsp2_clover] estimated eigenvalue(%d) = %25.16e\n", ievecs, evecs_lambda);
      }
  
      /* check evec equation */
      spinor_field_mi_eq_spinor_field_ti_re(eo_spinor_work2, eo_spinor_work0, evecs_lambda, Vhalf);

      spinor_scalar_product_re(&norm, eo_spinor_work2, eo_spinor_work2, Vhalf);
      if(g_cart_id == 0) {
        fprintf(stdout, "# [ama_gsp2_clover] eigenvector(%d) || A x - lambda x || = %16.7e\n", ievecs, sqrt(norm) );
      }
    }  /* end of loop on evecs_num */

  }  /* end of if check_eigenvectors */

  /***********************************************
   * (1) calculate gsp_v_v
   ***********************************************/
  /* name tag */
  sprintf(gsp_tag, "%s.%.4d", "gsp_v_v", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] calculating %s\n", gsp_tag);
  /* scalar products */
  status = gsp_calculate_v_dag_gamma_p_w_block(&(eo_evecs_block[0]), &(eo_evecs_block[0]), evecs_num, g_source_momentum_number, g_source_momentum_list,
      g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag);
  if(status != 0) {
    fprintf(stderr, "[ama_gsp2_clover] Error from gsp_calculate_v_dag_gamma_p_w, status was %d\n", status);
    EXIT(11);
  }
#
  /***********************************************
   * (2) calculate gsp_xeobarv_xeobarv
   ***********************************************/
   /* calculate Xeobar V */
  for(ievecs = 0; ievecs<evecs_num; ievecs++) {
    ratime = _GET_TIME;
    /* 0 <- V */
    memcpy(eo_spinor_work0, eo_evecs_field[ievecs], sizeof_eo_spinor_field);
    /* 1 <- 0 */
    X_clover_eo (eo_spinor_work1, eo_spinor_work0, g_gauge_field, mzzinv_dn[0]);
    /* XV <- 1 */
    memcpy(eo_evecs_field[evecs_num+ievecs], eo_spinor_work1, sizeof_eo_spinor_field);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] time for X_eo = %e seconds\n", retime-ratime);
  }
  /* name tag */
  sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeobarv", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] calculating %s\n", gsp_tag);
  
  /* scalar products */
  status = gsp_calculate_v_dag_gamma_p_w_block(&(eo_evecs_block[1]), &(eo_evecs_block[1]), evecs_num, g_source_momentum_number, g_source_momentum_list,
      g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag);
  if(status != 0) {
    fprintf(stderr, "[ama_gsp2_clover] Error from gsp_calculate_v_dag_gamma_p_w, status was %d\n", status);
    EXIT(13);
  }

  /***********************************************
   * (3) calculate gsp_w_w
   * - overwrite Xeobar V
   ***********************************************/
  /* calculate W */
  for(ievecs = 0; ievecs<evecs_num; ievecs++) {
    ratime = _GET_TIME;
    /* 0 <- V */
    memcpy(eo_spinor_work0, eo_evecs_field[          ievecs], sizeof_eo_spinor_field);
    /* 1 <- XV */
    memcpy(eo_spinor_work1, eo_evecs_field[evecs_num+ievecs], sizeof_eo_spinor_field);
    /*          input+output     Xeobar V         auxilliary */
    /* 0 <- 0,1 | 2 */
    C_clover_from_Xeo (eo_spinor_work0, eo_spinor_work1, eo_spinor_work2, g_gauge_field, mzz_dn[1]);
    /* <0|0> */
    spinor_scalar_product_re(&norm, eo_spinor_work0, eo_spinor_work0, Vhalf);
    evecs_eval[ievecs] = norm  * 4.*g_kappa*g_kappa;
    norm = 1./sqrt( norm );
    /* W <- 0 */
    spinor_field_eq_spinor_field_ti_re (eo_evecs_field[evecs_num + ievecs],  eo_spinor_work0, norm, Vhalf);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] time for C_from_Xeo = %e seconds\n", retime-ratime);
    /* TEST */
    if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] eval %4d %25.16e\n", ievecs, evecs_eval[ievecs]);
  }
  /* write eigenvalues */
  sprintf(gsp_tag, "%s.%.4d", "gsp_eval", Nconf);
  status = gsp_write_eval(evecs_eval, evecs_num, gsp_tag);
  if(status != 0) {
    fprintf(stderr, "[ama_gsp2_clover] Error from gsp_write, status was %d\n", status);
    EXIT(14);
  }

  /* name tag */
  sprintf(gsp_tag, "%s.%.4d", "gsp_w_w", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] calculating %s\n", gsp_tag);

  /* scalar product */
  status = gsp_calculate_v_dag_gamma_p_w_block(&(eo_evecs_block[1]), &(eo_evecs_block[1]), evecs_num, g_source_momentum_number, g_source_momentum_list,
      g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag);
  if(status != 0) {
    fprintf(stderr, "[ama_gsp2_clover] Error from gsp_calculate_v_dag_gamma_p_w, status was %d\n", status);
    EXIT(16);
  }

  /***********************************************
   * (4) calculate gsp_v_w
   ***********************************************/
  /* name tag */
  sprintf(gsp_tag, "%s.%.4d", "gsp_v_w", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] calculating %s\n", gsp_tag);

  /* scalar product */
  /*                                            V                      W */
  status = gsp_calculate_v_dag_gamma_p_w_block(&(eo_evecs_block[0]), &(eo_evecs_block[1]), evecs_num, g_source_momentum_number, g_source_momentum_list,
      g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag);
  if(status != 0) {
    fprintf(stderr, "[ama_gsp2_clover] Error from gsp_calculate_v_dag_gamma_p_w, status was %d\n", status);
    EXIT(18);
  }


  /***********************************************
   * (5) calculate gsp_xeow_xeow
   * - overwrite W with X_eo W
   ***********************************************/
   /* calculate Xeo W */
  for(ievecs = 0; ievecs<evecs_num; ievecs++) {
    ratime = _GET_TIME;
    /* 0 <- W */
    memcpy(eo_spinor_work0, eo_evecs_field[evecs_num+ievecs], sizeof_eo_spinor_field);
    /* 1 <- 0 */
    X_clover_eo (eo_spinor_work1, eo_spinor_work0, g_gauge_field, mzzinv_up[0]);
    /* XW <- 1 */
    memcpy(eo_evecs_field[evecs_num+ievecs], eo_spinor_work1, sizeof_eo_spinor_field);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] time for X_eo = %e seconds\n", retime-ratime);
  }
  /* name tag */
  sprintf(gsp_tag, "%s.%.4d", "gsp_xeow_xeow", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] calculating %s\n", gsp_tag);

  /* scalar products */
  status = gsp_calculate_v_dag_gamma_p_w_block(&(eo_evecs_block[1]), &(eo_evecs_block[1]), evecs_num, g_source_momentum_number, g_source_momentum_list,
      g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag);
  if(status != 0) {
    fprintf(stderr, "[ama_gsp2_clover] Error from gsp_calculate_v_dag_gamma_p_w, status was %d\n", status);
    EXIT(20);
  }

  /***********************************************
   * (6) calculate gsp_xeobarv_xeow
   * - FOR NOW: overwrite V with Xbar_eo V
   * - keep X_eo W
   ***********************************************/
   /* calculate Xeobar V */
  for(ievecs = 0; ievecs<evecs_num; ievecs++) {
    ratime = _GET_TIME;
    /* 0 <- V */
    memcpy(eo_spinor_work0, eo_evecs_field[ievecs], sizeof_eo_spinor_field);
    /* 1 <- 0 */
    X_clover_eo (eo_spinor_work1, eo_spinor_work0, g_gauge_field, mzzinv_dn[0]);
    /* XV <- 1 */
    memcpy(eo_evecs_field[ievecs], eo_spinor_work1, sizeof_eo_spinor_field);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] time for X_eo = %e seconds\n", retime-ratime);
  }
  /* name tag */
  sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeow", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp2_clover] calculating %s\n", gsp_tag);

  /* scalar products */
  status = gsp_calculate_v_dag_gamma_p_w_block(&(eo_evecs_block[0]), &(eo_evecs_block[1]), evecs_num, g_source_momentum_number, g_source_momentum_list,
      g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag);
  if(status != 0) {
    fprintf(stderr, "[ama_gsp2_clover] Error from gsp_calculate_v_dag_gamma_p_w, status was %d\n", status);
    EXIT(22);
  }

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  if(evecs_eval != NULL) free(evecs_eval);

  if(g_gauge_field != NULL) free(g_gauge_field);

  free(eo_spinor_field[0]);
  free(eo_spinor_field[2]);
  free(eo_spinor_field);

  free(eo_evecs_field);
  free(eo_evecs_block[0]);
  free(eo_evecs_block[1]);

  clover_term_fini(&clover);
  clover_term_fini(&mzz_up);
  clover_term_fini(&mzz_dn);
  clover_term_fini(&mzzinv_up);
  clover_term_fini(&mzzinv_dn);


  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
#endif

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [ama_gsp2_clover] %s# [ama_gsp2_clover] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [ama_gsp2_clover] %s# [ama_gsp2_clover] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

