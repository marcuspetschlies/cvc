/****************************************************
 * ama_gsp_clover.cpp
 *
 * Do 23. Jun 14:29:31 CEST 2016
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
#include "scalar_products.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "gsp.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, status;
  int i, ievecs;
  int filename_set = 0;
  int check_eigenvectors = 0;
  int threadid, nthreads;
  int no_eo_fields;
  unsigned int Vhalf;

  int evecs_num=0;
  double *evecs_eval = NULL;
  double evecs_lambda;

  double norm;
  complex w;

  double *****gsp = NULL;
  char gsp_tag[100];

  double plaq=0.;
  int verbose = 0;
  char filename[200], file_format[200];

  double **eo_spinor_field=NULL, *eo_spinor_work=NULL, *eo_spinor_work2 = NULL, *eo_spinor_work3=NULL, *full_spinor_work=NULL;
  double **clover=NULL, **mzz_up=NULL, **mzz_dn=NULL, **mzzinv_up=NULL, **mzzinv_dn=NULL;
  double ratime, retime;

  FILE *ifs = NULL;
  size_t items;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif


  while ((c = getopt(argc, argv, "ch?vf:n:a:")) != -1) {
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
      fprintf(stdout, "# [ama_gsp_clover] dimension of eigenspace set to%d\n", evecs_num);
      break;
    case 'c':
      check_eigenvectors = 1;
      fprintf(stdout, "# [ama_gsp_clover] check eigenvectors set to %d\n", check_eigenvectors);
      break;
    case 'a':
      strcpy( file_format, optarg );
      fprintf(stdout, "# [ama_gsp_clover] file format set to %s\n", file_format);
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
  if(g_cart_id==0) fprintf(stdout, "# [ama_gsp_clover] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [ama_gsp_clover] calling tmLQCD wrapper init functions\n");

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

  /* initialize geometry */
  if(init_geometry() != 0) {
    fprintf(stderr, "[ama_gsp_clover] Error from init_geometry\n");
    EXIT(101);
  }

  geometry();

  mpi_init_xchange_eo_spinor();

  Vhalf = VOLUME / 2;

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private(nthreads,threadid)
{
  threadid = omp_get_thread_num();
  nthreads = omp_get_num_threads();
  fprintf(stdout, "# [ama_gsp_clover] proc%.4d thread%.4d using %d threads\n", g_cart_id, threadid, nthreads);
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[ama_gsp_clover] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if (evecs_num == 0) {
    if(g_cart_id==0) fprintf(stderr, "[ama_gsp_clover] eigenspace dimension is 0\n");
    EXIT(1);
  }
  evecs_eval = (double*)malloc(evecs_num * sizeof(double));
  if(evecs_eval == NULL) {
    fprintf(stderr, "[ama_gsp_clover] Error from malloc\n");
    EXIT(117);
  }

  /* read the gauge field */
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
  if(g_cart_id==0) fprintf(stdout, "# [ama_gsp_clover] reading gauge field from file %s\n", filename);

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
      fprintf(stderr, "[ama_gsp_clover] Error, could not read gauge field\n");
      EXIT(11);
    }

    /* measure the plaquette */
    if(g_cart_id==0) fprintf(stdout, "# [ama_gsp_clover] read plaquette value 1st field: %25.16e\n", plaq);
    plaquette(&plaq);
    if(g_cart_id==0) fprintf(stdout, "# [ama_gsp_clover] measured plaquette value 1st field: %25.16e\n", plaq);
  } else {
    if(g_cart_id == 0) fprintf(stderr, "# [ama_gsp_clover] need gauge field\n");
    EXIT(20);
  }

  /***********************************************
   * allocate spinor fields
   ***********************************************/
  no_eo_fields = 2*evecs_num + 3;
  eo_spinor_field = (double**)calloc(no_eo_fields, sizeof(double*));
  for(i=0; i<no_eo_fields-2; i++) {
    alloc_spinor_field(&eo_spinor_field[i], (VOLUME+RAND)/2);
  }
  alloc_spinor_field(&eo_spinor_field[no_eo_fields-2], (VOLUME+RAND));
  eo_spinor_field[no_eo_fields-1] = eo_spinor_field[no_eo_fields-2] + 12*(VOLUME+RAND);
  /* auxilliary eo fields */
  eo_spinor_work  = eo_spinor_field[no_eo_fields - 3];
  eo_spinor_work2 = eo_spinor_field[no_eo_fields - 2];
  eo_spinor_work3 = eo_spinor_field[no_eo_fields - 1];
  /* auxilliary full spinor field */
  full_spinor_work = eo_spinor_field[no_eo_fields - 2];

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
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] time for clover_term_eo = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_matrix (mzz_up, clover, g_mu, g_csw);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] time for clover_mzz_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_matrix (mzz_dn, clover, -g_mu, g_csw);
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] time for clover_mzz_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_inv_matrix (mzzinv_up, mzz_up);
  retime = _GET_TIME;
  fprintf(stdout, "# [ama_gsp_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);

  ratime = _GET_TIME;
  clover_mzz_inv_matrix (mzzinv_dn, mzz_dn);
  retime = _GET_TIME;
  fprintf(stdout, "# [ama_gsp_clover] time for clover_mzz_inv_matrix = %e seconds\n", retime-ratime);



  /***********************************************
   * read eo eigenvectors
   ***********************************************/
  if( strcmp(file_format, "single") == 0 ) {
    for(ievecs = 0; ievecs<evecs_num; ievecs+=2) {
  
      sprintf(filename, "%s%.5d", filename_prefix, ievecs);
      if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] reading C_oo_sym eigenvector from file %s\n", filename);

      status = read_lime_spinor(full_spinor_work, filename, 0);
      if( status != 0) {
        fprintf(stderr, "[ama_gsp_clover] Error from read_lime_spinor, status was %d\n", status);
        EXIT(1);
      }

      ratime = _GET_TIME;
      spinor_field_unpack_lexic2eo (full_spinor_work, eo_spinor_field[ievecs], eo_spinor_field[ievecs+1]);
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] time for unpacking = %e\n", retime - ratime);

    }  /* end of loop on evecs number */
  } else if (strcmp(file_format, "part_file") == 0 ) {
  
    sprintf(filename, "%s.%.4d.%.5d.pt%.2dpx%.2dpy%.2dpz%.2d", filename_prefix, Nconf, evecs_num, 
       g_proc_coords[0], g_proc_coords[1], g_proc_coords[2], g_proc_coords[3]);
    if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] reading C_oo_sym eigenvector from file %s\n", filename);

    ifs = fopen(filename, "r");
    if( ifs == NULL) {
      fprintf(stderr, "[ama_gsp_clover] Error, could not open file %s for reading\n", filename);
      EXIT(132);
    }

    items = 12 * (size_t)VOLUME;
    for(ievecs = 0; ievecs<evecs_num; ievecs++) {

      if( items != fread ( eo_spinor_field[ievecs], sizeof(double), items, ifs ) ) {
        fprintf(stderr, "[ama_gsp_clover] Error, could not read proper amount of data from file %s\n", filename);
        EXIT(133);
      }
    }
    fclose(ifs);
  } else {
    fprintf(stderr, "[] Error, file format not implemented\n");
    EXIT(131);
  }

  if (check_eigenvectors) {

    /***********************************************
     * check eigenvector equation
     ***********************************************/
    for(ievecs = 0; ievecs<evecs_num; ievecs++) {
  
      ratime = _GET_TIME;
#ifdef HAVE_MPI
      xchange_eo_field(eo_spinor_field[ievecs], 1);
#endif
      C_clover_oo(eo_spinor_work, eo_spinor_field[ievecs], g_gauge_field, eo_spinor_work3, mzz_dn[1], mzzinv_dn[0]);

#ifdef HAVE_MPI
      xchange_eo_field( eo_spinor_work, 1);
#endif
      C_clover_oo(eo_spinor_work2, eo_spinor_work, g_gauge_field, eo_spinor_work3, mzz_up[1], mzzinv_up[0]);
  
      norm = 4 * g_kappa * g_kappa;
      spinor_field_ti_eq_re (eo_spinor_work2, norm, Vhalf);
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] time to apply C_oo_sym = %e\n", retime - ratime);
  
      /* determine eigenvalue */
      spinor_scalar_product_re(&norm,  eo_spinor_field[ievecs], eo_spinor_field[ievecs], Vhalf);
      spinor_scalar_product_co(&w,  eo_spinor_field[ievecs], eo_spinor_work2, Vhalf);
      evecs_lambda = w.re / norm;
      if(g_cart_id == 0) {
        fprintf(stdout, "# [ama_gsp_clover] estimated eigenvalue(%d) = %25.16e\n", ievecs, evecs_lambda);
      }
  
      /* check evec equation */
      spinor_field_mi_eq_spinor_field_ti_re(eo_spinor_work2, eo_spinor_field[ievecs], evecs_lambda, Vhalf);

      spinor_scalar_product_re(&norm, eo_spinor_work2, eo_spinor_work2, Vhalf);
      if(g_cart_id == 0) {
        fprintf(stdout, "# [ama_gsp_clover] eigenvector(%d) || A x - lambda x || = %16.7e\n", ievecs, sqrt(norm) );
      }
    }  /* end of loop on evecs_num */

  }  /* end of if check_eigenvectors */


  /***********************************************
   * (1) calculate gsp_v_v
   ***********************************************/
  /* allocate */
  status = gsp_init (&gsp, g_source_momentum_number, g_source_gamma_id_number, T, evecs_num);
  if(status != 0) {
    fprintf(stderr, "[ama_gsp_clover] Error from gsp_init, status was %d\n", status);
    EXIT(150);
  }
  /* name tag */
  sprintf(gsp_tag, "%s.%.4d", "gsp_v_v", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] calculating %s\n", gsp_tag);
  /* scalar products */
  status = gsp_calculate_v_dag_gamma_p_w(gsp, &(eo_spinor_field[0]), &(eo_spinor_field[0]), evecs_num, g_source_momentum_number, g_source_momentum_list,
      g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag, 1);
  /* deallocate */
  gsp_fini(&gsp);

  /***********************************************
   * (2) calculate gsp_xeobarv_xeobarv
   ***********************************************/
   /* calculate Xeobar V */
  for(ievecs = 0; ievecs<evecs_num; ievecs++) {
    ratime = _GET_TIME;
    X_clover_eo (eo_spinor_field[evecs_num+ievecs], eo_spinor_field[ievecs], g_gauge_field, mzzinv_dn[0]);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] time for X_clover_eo = %e seconds\n", retime-ratime);
  }
  /* name tag */
  sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeobarv", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] calculating %s\n", gsp_tag);
  /* allocate */
  status = gsp_init (&gsp, g_source_momentum_number, g_source_gamma_id_number, T, evecs_num);
  if(status != 0) {
    fprintf(stderr, "[ama_gsp_clover] Error from gsp_init, status was %d\n", status);
    EXIT(150);
  }
  /* scalar products */
  status = gsp_calculate_v_dag_gamma_p_w(gsp, &(eo_spinor_field[evecs_num]), &(eo_spinor_field[evecs_num]), evecs_num, g_source_momentum_number, g_source_momentum_list,
      g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag, 1);
  /* deallocate */
  gsp_fini(&gsp);

  /***********************************************
   * calculate gsp_w_w
   * - overwrite Xeobar V
   ***********************************************/
  /* calculate W */
  for(ievecs = 0; ievecs<evecs_num; ievecs++) {
    ratime = _GET_TIME;
    /* copy original V to eo_spinor_work */
    memcpy(eo_spinor_work, eo_spinor_field[ievecs], 24*Vhalf*sizeof(double));
    /*          input+output    Xeobar V                           auxilliary */
    C_clover_from_Xeo (eo_spinor_work, eo_spinor_field[evecs_num+ievecs], eo_spinor_work2, g_gauge_field, mzz_dn[1]);
    spinor_scalar_product_re(&norm, eo_spinor_work, eo_spinor_work, Vhalf);
    evecs_eval[ievecs] = norm  * 4.*g_kappa*g_kappa;
    norm = 1./sqrt( norm );
    /* final result starting at evecs_num */
    spinor_field_eq_spinor_field_ti_re (eo_spinor_field[evecs_num + ievecs],  eo_spinor_work, norm, Vhalf);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] time for C_clover_from_Xeo = %e seconds\n", retime-ratime);
    /* TEST */
    if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] eval %4d %25.16e\n", ievecs, evecs_eval[ievecs]);
  }
  /* write eigenvalues */
  sprintf(gsp_tag, "%s.%.4d", "gsp_eval", Nconf);
  status = gsp_write_eval(evecs_eval, evecs_num, gsp_tag);
  if(status != 0) {
    fprintf(stderr, "[ama_gsp_clover] Error from gsp_init, status was %d\n", status);
    EXIT(151);
  }

  /* name tag */
  sprintf(gsp_tag, "%s.%.4d", "gsp_w_w", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] calculating %s\n", gsp_tag);
  /* allocate */
  status = gsp_init (&gsp, g_source_momentum_number, g_source_gamma_id_number, T, evecs_num);
  if(status != 0) {
    fprintf(stderr, "[ama_gsp_clover] Error from gsp_init, status was %d\n", status);
    EXIT(150);
  }
  /* scalar product */
  status = gsp_calculate_v_dag_gamma_p_w(gsp, &(eo_spinor_field[evecs_num]), &(eo_spinor_field[evecs_num]), evecs_num, g_source_momentum_number, g_source_momentum_list,
      g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag, 1);
  /* deallocate */
  gsp_fini(&gsp);

  /***********************************************
   * calculate gsp_v_w
   ***********************************************/
  /* name tag */
  sprintf(gsp_tag, "%s.%.4d", "gsp_v_w", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] calculating %s\n", gsp_tag);
  /* allocate */
  status = gsp_init (&gsp, g_source_momentum_number, g_source_gamma_id_number, T, evecs_num);
  if(status != 0) {
    fprintf(stderr, "[ama_gsp_clover] Error from gsp_init, status was %d\n", status);
    EXIT(150);
  }
  /* scalar product */
  /*                                            V                      W */
  status = gsp_calculate_v_dag_gamma_p_w(gsp, &(eo_spinor_field[0]), &(eo_spinor_field[evecs_num]), evecs_num, g_source_momentum_number, g_source_momentum_list,
      g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag, 0);
  /* deallocate */
  gsp_fini(&gsp);


  /***********************************************
   * calculate gsp_xeow_xeow
   * - overwrite W with X_eo W
   ***********************************************/
   /* calculate Xeo W */
  for(ievecs = 0; ievecs<evecs_num; ievecs++) {
    ratime = _GET_TIME;
    /* copy W to eo_spinor_work */
    memcpy(eo_spinor_work, eo_spinor_field[evecs_num+ievecs], 24*Vhalf*sizeof(double));
    /* final result starting at evecs_num */
    X_clover_eo (eo_spinor_field[evecs_num+ievecs], eo_spinor_work, g_gauge_field, mzzinv_up[0]);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] time for X_clover_eo = %e seconds\n", retime-ratime);
  }
  /* name tag */
  sprintf(gsp_tag, "%s.%.4d", "gsp_xeow_xeow", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] calculating %s\n", gsp_tag);
  /* allocate */
  status = gsp_init (&gsp, g_source_momentum_number, g_source_gamma_id_number, T, evecs_num);
  if(status != 0) {
    fprintf(stderr, "[ama_gsp_clover] Error from gsp_init, status was %d\n", status);
    EXIT(150);
  }
  /* scalar products */
  status = gsp_calculate_v_dag_gamma_p_w(gsp, &(eo_spinor_field[evecs_num]), &(eo_spinor_field[evecs_num]), evecs_num, g_source_momentum_number, g_source_momentum_list,
      g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag, 1);
  /* deallocate */
  gsp_fini(&gsp);
  

  /***********************************************
   * calculate gsp_xeobarv_xeow
   * - FOR NOW: overwrite V with Xbar_eo V
   * - keep X_eo W
   ***********************************************/
   /* calculate Xeobar V */
  for(ievecs = 0; ievecs<evecs_num; ievecs++) {
    ratime = _GET_TIME;
    /* copy V to eo_spinor_work */
    memcpy(eo_spinor_work, eo_spinor_field[ievecs], 24*Vhalf*sizeof(double));
    /* final result starting at 0 */
    X_clover_eo (eo_spinor_field[ievecs], eo_spinor_work, g_gauge_field, mzzinv_dn[0]);
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] time for X_clover_eo = %e seconds\n", retime-ratime);
  }
  /* name tag */
  sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeow", Nconf);
  if(g_cart_id == 0) fprintf(stdout, "# [ama_gsp_clover] calculating %s\n", gsp_tag);
  /* allocate */
  status = gsp_init (&gsp, g_source_momentum_number, g_source_gamma_id_number, T, evecs_num);
  if(status != 0) {
    fprintf(stderr, "[ama_gsp_clover] Error from gsp_init, status was %d\n", status);
    EXIT(150);
  }
  /* scalar products */
  status = gsp_calculate_v_dag_gamma_p_w(gsp, &(eo_spinor_field[0]), &(eo_spinor_field[evecs_num]), evecs_num, g_source_momentum_number, g_source_momentum_list,
      g_source_gamma_id_number, g_source_gamma_id_list, gsp_tag, 0);
  /* deallocate */
  gsp_fini(&gsp);
  

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  if(evecs_eval != NULL) free(evecs_eval);

  if(g_gauge_field != NULL) free(g_gauge_field);

  for(i=0; i<no_eo_fields; i++) free(eo_spinor_field[i]);
  free(eo_spinor_field);

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
#endif

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [ama_gsp_clover] %s# [ama_gsp_clover] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [ama_gsp_clover] %s# [ama_gsp_clover] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

