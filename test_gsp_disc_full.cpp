/****************************************************
 * test_gsp_disc_full.cpp
 *
 * Fri Jul 22 11:58:58 CEST 2016
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
#include "read_input_parser.h"
#include "gsp.h"

using namespace cvc;

void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

  /*
   * to get the pair (Gamma_i, Gamma_f) one needs to use (g5 Gamma_i, g5 Gamma_f)
     0,   1,   2,   3,   id, 5,  0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
  id 0    1    2    3    4   5   6    7    8    9    10   11   12   13   14   15
g5 x 0_5, 1_5, 2_5, 3_5, 5,  id, 0,   1,   2,   3,   2_3, 1_3, 1_2, 0_3, 0_2, 0_1 
*/
const int gamma_id_g5_ti_gamma[16] {
     6,   7,   8,   9,   5,  4,  0,   1,   2,   3,   15,  14,  13,  12,  11,  10
};

const int gamma_sign_g5_ti_gamma[16] {
    -1,  -1,  -1,  -1,   1,  1, -1,  -1,  -1,  -1,   -1,   1,  -1,  -1,   1,  -1
};

int main(int argc, char **argv) {
  
  int c, status;
  int i, i2pt, io_proc, k;
  int i_gi, i_gf, i_si, i_sf, i_ti, i_tf, i_tfi, x0;
  int filename_set = 0;
#ifdef HAVE_OPENMP
  int threadid, nthreads;
#endif

  int evecs_num=0, evecs_num_contract=0;
  int sample_num_contract = 0;
  double *evecs_eval = NULL;
  double *sample_weights = NULL;

  complex w, w2, sli, slf, vli, vlf;

  double *****gsp_v_w_f = NULL, *****gsp_xv_xw_f = NULL, *****gsp_v_w_i = NULL, *****gsp_xv_xw_i = NULL;
  double *****gsp_xi_phi_f = NULL, *****gsp_xxi_xphi_f = NULL, *****gsp_xi_phi_i = NULL, *****gsp_xxi_xphi_i = NULL;
#ifdef HAVE_MPI
  int mrank, mcoords[4];
#endif
  double *evecs_loop_i=NULL, *evecs_loop_f=NULL, *sample_loop_i=NULL, *sample_loop_f=NULL;
  double **sample_diag_i=NULL, **sample_diag_f=NULL;
  double *buffer=NULL;
  double *correlator_re_re=NULL, *correlator_re_im=NULL, *correlator_im_re=NULL, *correlator_im_im=NULL;
  char gsp_tag[100];

  int verbose = 0;
  char filename[200];

  double ratime, retime, dtmp, sample_loop_norm;

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char * aff_status_str;
  double _Complex *aff_buffer = NULL;
  char aff_buffer_path[200];
  /*  uint32_t aff_buffer_size; */
#else
  FILE *ofs = NULL;
#endif

#if (defined HAVE_LHPC_AFF) || (defined HAVE_MPI)
  size_t items, bytes;
#endif

#ifdef HAVE_MPI
  MPI_Status mstatus;

  MPI_Init(&argc, &argv);
#endif


  while ((c = getopt(argc, argv, "h?vf:n:c:C:")) != -1) {
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
      fprintf(stdout, "# [test_gsp_disc_full] dimension of eigenspace set to%d\n", evecs_num);
      break;
    case 'c':
      evecs_num_contract = atoi(optarg);
      fprintf(stdout, "# [test_gsp_disc_full] use first %d eigenvectors for contraction\n", evecs_num_contract);
      break;
    case 'C':
      sample_num_contract = atoi(optarg);
      fprintf(stdout, "# [test_gsp_disc_full] use first %d samples for contraction\n", sample_num_contract);
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
  if(g_cart_id==0) fprintf(stdout, "# [test_gsp_disc_full] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_gsp_disc_full] calling tmLQCD wrapper init functions\n");

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
    fprintf(stderr, "[test_gsp_disc_full] Error from init_geometry\n");
    EXIT(4);
  }

  geometry();

#if (defined HAVE_MPI) && ( (defined PARALLELTX) ||  (defined PARALLELTXY) ||  (defined PARALLELTXYZ) )
  fprintf(stderr, "[test_gsp_disc_full] Error, for now only 1-dimensional MPI-parallelization\n");
  EXIT(5);
#endif

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc_full] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private(nthreads,threadid)
{
  threadid = omp_get_thread_num();
  nthreads = omp_get_num_threads();
  fprintf(stdout, "# [test_gsp_disc_full] proc%.4d thread%.4d using %d threads\n", g_cart_id, threadid, nthreads);
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_gsp_disc_full] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if (evecs_num == 0) {
    if(g_cart_id==0) fprintf(stderr, "[test_gsp_disc_full] eigenspace dimension is 0\n");
    EXIT(6);
  }

  /***********************************************
   * check, whether evecs_num_contract was set
   ***********************************************/
  if(evecs_num_contract == 0 || evecs_num_contract > evecs_num) {
    evecs_num_contract = evecs_num;
    if(g_cart_id == 0) fprintf(stdout, "[test_gsp_2pt] reset evecs_num_contract to default value %d\n", evecs_num_contract);
  }
  if(sample_num_contract == 0  || sample_num_contract > g_nsample) {
    sample_num_contract = g_nsample;
    if(g_cart_id == 0) fprintf(stdout, "[test_gsp_2pt] reset sample_num_contract to default value %d\n", sample_num_contract);
  }

  /***********************************************
   * read eigenvalues
   ***********************************************/
  sprintf(gsp_tag, "%s.%.4d", "gsp_eval", Nconf);
  status = gsp_read_eval(&evecs_eval, evecs_num, gsp_tag);
  if( status != 0) {
    fprintf(stderr, "[test_gsp_disc_full] Error from gsp_read_eval, status was %d\n", status);
    EXIT(7);
  }

  /***********************************************
   * inverse square root of diagonal elements
   ***********************************************/
  for(i=0; i<evecs_num; i++) {
    if(g_cart_id==0) fprintf(stdout, "# [test_gsp_disc_full] eval %4d %25.16e\n", i, evecs_eval[i]) ;
    evecs_eval[i] = 2.*g_kappa / sqrt(evecs_eval[i]);
  }

  /***********************************************
   * sample weight, set identically equal to 1
   ***********************************************/
  sample_weights = (double*)malloc(g_nsample*sizeof(double));
  if( sample_weights == NULL) {
    fprintf(stderr, "[test_gsp_disc_full] Error from malloc\n");
    EXIT(8);
  }
  for(i=0; i<g_nsample; i++) { sample_weights[i] = 1.; }

  /***********************************************
   * allocate gsp's
   ***********************************************/
  /* allocate */
  ratime = _GET_TIME;
  status = 0;

  status = status ||  gsp_init (&gsp_v_w_f,   1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_xv_xw_f, 1, 1, T, evecs_num);

  status = status ||  gsp_init (&gsp_v_w_i,   1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_xv_xw_i, 1, 1, T, evecs_num);

  status = status ||  gsp_init (&gsp_xi_phi_f,   1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_xxi_xphi_f, 1, 1, T, evecs_num);

  status = status ||  gsp_init (&gsp_xi_phi_i,   1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_xxi_xphi_i, 1, 1, T, evecs_num);

  if(status) {
    fprintf(stderr, "[test_gsp_disc_full] Error from gsp_init\n");
    EXIT(9);
  }
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc_full] time to initialize gsp = %e seconds\n", retime - ratime);

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [test_gsp_disc_full] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [test_gsp_disc_full] proc%.4d is send process\n", g_cart_id);
    } else {
      io_proc = 0;
    }
  }
#else
  io_proc = 2;
#endif

  /***********************************************
   * loops, buffer and correlators
   ***********************************************/
  evecs_loop_i = (double*)malloc(2*T_global*sizeof(double));
  if(evecs_loop_i == NULL) {
    fprintf(stderr, "[test_gsp_disc_full] Error from malloc\n");
    EXIT(10);
  }
  evecs_loop_f = (double*)malloc(2*T_global*sizeof(double));
  if(evecs_loop_f == NULL) {
    fprintf(stderr, "[test_gsp_disc_full] Error from malloc\n");
    EXIT(11);
  }

  sample_loop_i = (double*)malloc(2*T_global*sizeof(double));
  if(sample_loop_i == NULL) {
    fprintf(stderr, "[test_gsp_disc_full] Error from malloc\n");
    EXIT(12);
  }
  sample_loop_f = (double*)malloc(2*T_global*sizeof(double));
  if(sample_loop_f == NULL) {
    fprintf(stderr, "[test_gsp_disc_full] Error from malloc\n");
    EXIT(13);
  }

  sample_diag_i = (double**)malloc(T_global*sizeof(double*));
  sample_diag_i[0] = (double*)malloc(2*T_global*g_nsample*sizeof(double));
  if(sample_diag_i[0] == NULL) {
    fprintf(stderr, "[test_gsp_disc_full] Error from malloc\n");
    EXIT(14);
  }
  for(i=1; i<T_global; i++) sample_diag_i[i] = sample_diag_i[i-1] + 2*g_nsample;

  sample_diag_f = (double**)malloc(T_global*sizeof(double*));
  sample_diag_f[0] = (double*)malloc(2*T_global*g_nsample*sizeof(double));
  if(sample_diag_f[0] == NULL) {
    fprintf(stderr, "[test_gsp_disc_full] Error from malloc\n");
    EXIT(15);
  }
  for(i=1; i<T_global; i++) sample_diag_f[i] = sample_diag_f[i-1] + 2*g_nsample;
  
  buffer = (double*)malloc(2*T_global*sizeof(double));
  if( buffer == NULL ) {
    fprintf(stderr, "[test_gsp_disc] Error from malloc\n");
    EXIT(16);
  }

  if(io_proc == 2) {

    correlator_re_re = (double*)malloc(T_global*sizeof(double));
    if(correlator_re_re == NULL) {
      fprintf(stderr, "[test_gsp_disc_full] Error from malloc\n");
      EXIT(17);
    }
 
    correlator_re_im = (double*)malloc(T_global*sizeof(double));
    if(correlator_re_im == NULL) {
      fprintf(stderr, "[test_gsp_disc_full] Error from malloc\n");
      EXIT(18);
    }
 
    correlator_im_re = (double*)malloc(T_global*sizeof(double));
    if(correlator_im_re == NULL) {
      fprintf(stderr, "[test_gsp_disc_full] Error from malloc\n");
      EXIT(19);
    }
 
    correlator_im_im = (double*)malloc(T_global*sizeof(double));
    if(correlator_im_im == NULL) {
      fprintf(stderr, "[test_gsp_disc_full] Error from malloc\n");
      EXIT(20);
    }
 
  }  /* of if io_proc == 2 */

  /***********************************************
   * output file
   ***********************************************/
  if(io_proc == 2) {
#ifdef HAVE_LHPC_AFF

    aff_status_str = (char*)aff_version();
    fprintf(stdout, "# [test_gsp_disc_full] using aff version %s\n", aff_status_str);

    sprintf(filename, "%s.%.4d.nev%.4d.nsp%.4d.aff", "gsp_correlator_disc", Nconf, evecs_num_contract, sample_num_contract);
    fprintf(stdout, "# [test_gsp_disc_full] writing gsp data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_gsp_disc_full] Error from aff_writer, status was %s\n", aff_status_str);
      EXIT(21);
    }

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[test_gsp_disc_full] Error, aff writer is not initialized\n");
      EXIT(22);
    }

    aff_buffer = (double _Complex*)malloc(T_global*sizeof(double _Complex));
    if(aff_buffer == NULL) {
      fprintf(stderr, "[test_gsp_disc_full] Error from malloc\n");
      EXIT(23);
    }
#else
    sprintf(filename, "%s.%.4d.nev%.4d.nsp%.4d", "gsp_correlator_disc_full", Nconf, evecs_num_contract, sample_num_contract);
    ofs = fopen(filename, "w");
    if(ofs == NULL) {
      fprintf(stderr, "[test_gsp_disc_full] Error, could not open file %s for writing\n", filename);
      EXIT(24);
    }
#endif
  }  /* of if io_proc == 0  */


  /***********************************************
   * allocate buffer
   ***********************************************/
  buffer = (double*)malloc(2*T_global*g_nsample*sizeof(double));
  if( buffer == NULL ) {
    fprintf(stderr, "[test_gsp_disc_full] Error from malloc\n");
    EXIT(25);
  }

  /***********************************************
   * loops on configurations of momenta
   * at source and
   * sink and Gamma at source and sink
   *
   * at sink
   *   gsp_v_w_f
   *   gsp_xv_xw_f
   *
   * at source
   *   gsp_v_w_i    (also at sink)
   *   gsp_xv_xw_i  (also at sink)
   *
   ***********************************************/

  for(i2pt=0; i2pt < g_m_m_2pt_num; i2pt++) {

    /* gamma id at source and sink */
    i_gi = gamma_id_g5_ti_gamma[g_m_m_2pt_list[i2pt].gi];
    i_gf = gamma_id_g5_ti_gamma[g_m_m_2pt_list[i2pt].gf];
    /* gamma sign at source and sink */
    i_si = gamma_sign_g5_ti_gamma[g_m_m_2pt_list[i2pt].gi];
    i_sf = gamma_sign_g5_ti_gamma[g_m_m_2pt_list[i2pt].gf];

    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc_full] (%d, %d) to (%d, %d); (%d, %d)\n", g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].gf, i_gi, i_si, i_gf, i_sf);

    ratime = _GET_TIME;

    /* at source */
    sprintf(gsp_tag, "%s.%.4d", "gsp_v_w", Nconf);
    status = gsp_read_node (gsp_v_w_i[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_disc_full] Error from gsp_read_node\n");
      EXIT(26);
    }

    sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeow", Nconf);
    status = gsp_read_node (gsp_xv_xw_i[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_disc_full] Error from gsp_read_node\n");
      EXIT(27);
    }

    /* at sink */
    sprintf(gsp_tag, "%s.%.4d", "gsp_v_w", Nconf);
    status = gsp_read_node (gsp_v_w_f[0][0], evecs_num, g_m_m_2pt_list[i2pt].pf, i_gf, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_disc_full] Error from gsp_read_node\n");
      EXIT(28);
    }

    sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeow", Nconf);
    status = gsp_read_node (gsp_xv_xw_f[0][0], evecs_num, g_m_m_2pt_list[i2pt].pf, i_gf, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_disc_full] Error from gsp_read_node\n");
      EXIT(29);
    }


    /* stochastic part at source */
    sprintf(gsp_tag, "%s.%.4d", "gsp_xi_phi", Nconf);
    status = gsp_read_node (gsp_xi_phi_i[0][0], g_nsample, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_disc_full] Error from gsp_read_node\n");
      EXIT(30);
    }

    sprintf(gsp_tag, "%s.%.4d", "gsp_xbareoxi_xeophi", Nconf);
    status = gsp_read_node (gsp_xxi_xphi_i[0][0], g_nsample, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_disc_full] Error from gsp_read_node\n");
      EXIT(31);
    }

    /* stochastic part at sink */
    sprintf(gsp_tag, "%s.%.4d", "gsp_xi_phi", Nconf);
    status = gsp_read_node (gsp_xi_phi_f[0][0], g_nsample, g_m_m_2pt_list[i2pt].pf, i_gf, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_disc_full] Error from gsp_read_node\n");
      EXIT(32);
    }

    sprintf(gsp_tag, "%s.%.4d", "gsp_xbareoxi_xeophi", Nconf);
    status = gsp_read_node (gsp_xxi_xphi_f[0][0], g_nsample, g_m_m_2pt_list[i2pt].pf, i_gf, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_disc_full] Error from gsp_read_node\n");
      EXIT(33);
    }

    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc_full] time to read gsp nodes gsp = %e seconds\n", retime - ratime);


    /* reduce (trace) matrix product 
     *   for all combinations of source and sink time
     */
    memset(evecs_loop_f, 0, 2*T_global*sizeof(double));
    memset(evecs_loop_i, 0, 2*T_global*sizeof(double));
    memset(sample_loop_f, 0, 2*T_global*sizeof(double));
    memset(sample_loop_i, 0, 2*T_global*sizeof(double));
    memset(sample_diag_i[0], 0, 2*T_global*g_nsample*sizeof(double));
    memset(sample_diag_f[0], 0, 2*T_global*g_nsample*sizeof(double));

    ratime = _GET_TIME;
    for(i_ti = 0; i_ti<T; i_ti++) {
      x0 = i_ti + g_proc_coords[0] * T;

      _co_eq_zero(&w);
      /* accumulate four trace terms */
      co_eq_tr_gsp (&w2, gsp_xv_xw_i[0][0][i_ti], evecs_eval, evecs_num_contract);
      _co_pl_eq_co(&w, &w2);

      co_eq_tr_gsp (&w2, gsp_v_w_i[0][0][i_ti],   evecs_eval, evecs_num_contract);
      _co_pl_eq_co(&w, &w2);

      _co_ti_eq_re(&w, i_si);
  
      evecs_loop_i[2*x0  ] = w.re;
      evecs_loop_i[2*x0+1] = w.im;
      /* TEST */
      /* fprintf(stdout, "# [test_gsp_disc_full] correlator[%2d] = %25.16e %25.16e\n", x0, correlator[2*x0  ], correlator[2*x0+1]); */

    }  /* end of loop on tf */
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc_full] time for source evecs loop reduction = %e seconds\n", retime - ratime);

    ratime = _GET_TIME;
    for(i_tf = 0; i_tf<T; i_tf++) {
      x0 = i_tf + g_proc_coords[0] * T;

      _co_eq_zero(&w);
      /* accumulate four trace terms */
      co_eq_tr_gsp (&w2, gsp_xv_xw_f[0][0][i_tf], evecs_eval, evecs_num_contract);
      _co_pl_eq_co(&w, &w2);

      co_eq_tr_gsp (&w2, gsp_v_w_f[0][0][i_tf],   evecs_eval, evecs_num_contract);
      _co_pl_eq_co(&w, &w2);

      _co_ti_eq_re(&w, i_sf);

      evecs_loop_f[2*x0  ] = w.re;
      evecs_loop_f[2*x0+1] = w.im;

      /* TEST */
      /* fprintf(stdout, "# [test_gsp_disc_full] correlator2[%2d] = %25.16e %25.16e\n", x0, correlator2[2*x0  ], correlator2[2*x0+1]); */
    }    /* end of loop on tf */

    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc_full] time for sink evecs loop reduction = %e seconds\n", retime - ratime);

    ratime = _GET_TIME;
    /* stochastic loops at source */
    for(i_ti = 0; i_ti<T; i_ti++) {
      x0 = i_ti + g_proc_coords[0] * T;

      _co_eq_zero(&w);
      /* accumulate four trace terms */
      co_eq_tr_gsp (&w2, gsp_xxi_xphi_i[0][0][i_ti], sample_weights, sample_num_contract);
      _co_pl_eq_co(&w, &w2);

      co_eq_tr_gsp (&w2, gsp_xi_phi_i[0][0][i_ti],   sample_weights, sample_num_contract);
      _co_pl_eq_co(&w, &w2);

      _co_ti_eq_re(&w, i_si);
  
      sample_loop_i[2*x0  ] = w.re;
      sample_loop_i[2*x0+1] = w.im;

      /* set to  diagonal of gsp x */
      co_eq_gsp_diag ((complex*)(sample_diag_i[x0]), gsp_xxi_xphi_i[0][0][i_ti], sample_num_contract);
      /* add diagonal of gsp */
      co_pl_eq_gsp_diag ((complex*)(sample_diag_i[x0]), gsp_xi_phi_i[0][0][i_ti], sample_num_contract);

    }  /* end of loop on tf */
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc_full] time for source sample loop reduction = %e seconds\n", retime - ratime);

    ratime = _GET_TIME;
    /* stochastic loops at sink */
    for(i_tf = 0; i_tf<T; i_tf++) {
      x0 = i_tf + g_proc_coords[0] * T;

      _co_eq_zero(&w);
      /* accumulate four trace terms */
      co_eq_tr_gsp (&w2, gsp_xxi_xphi_f[0][0][i_tf], sample_weights, sample_num_contract);
      _co_pl_eq_co(&w, &w2);

      co_eq_tr_gsp (&w2, gsp_xi_phi_f[0][0][i_tf],   sample_weights, sample_num_contract);
      _co_pl_eq_co(&w, &w2);

      _co_ti_eq_re(&w, i_sf);

      sample_loop_f[2*x0  ] = w.re;
      sample_loop_f[2*x0+1] = w.im;

      /* set to  diagonal of gsp x */
      co_eq_gsp_diag ((complex*)(sample_diag_f[x0]), gsp_xxi_xphi_f[0][0][i_tf], sample_num_contract);
      /* add diagonal of gsp */
      co_pl_eq_gsp_diag ((complex*)(sample_diag_f[x0]), gsp_xi_phi_f[0][0][i_tf], sample_num_contract);

    }    /* end of loop on tf */
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc_full] time for sink sample loop reduction = %e seconds\n", retime - ratime);

#ifdef HAVE_MPI
    /* exchange */

    ratime = _GET_TIME;
    items = 2 * (size_t)T_global;
    bytes = items * sizeof(double);

    /* exchange evecs part of loops at source and sink */

    memcpy(buffer, evecs_loop_i, bytes);
    status = MPI_Allreduce(buffer, evecs_loop_i, (2*T_global), MPI_DOUBLE, MPI_SUM, g_cart_grid);
    if (status != MPI_SUCCESS) {
      fprintf(stderr, "[test_gsp_disc_full] Error from MPI_Allreduce\n");
      EXIT(34);
    }

    memcpy(buffer, evecs_loop_f, bytes);
    status = MPI_Allreduce(buffer, evecs_loop_f, (2*T_global), MPI_DOUBLE, MPI_SUM, g_cart_grid);
    if (status != MPI_SUCCESS) {
      fprintf(stderr, "[test_gsp_disc_full] Error from MPI_Allreduce\n");
      EXIT(35);
    }

    /* exchange sample part of loops at source and sink */

    memcpy(buffer, sample_loop_i, bytes);
    status = MPI_Allreduce(buffer, sample_loop_i, (2*T_global), MPI_DOUBLE, MPI_SUM, g_cart_grid);
    if (status != MPI_SUCCESS) {
      fprintf(stderr, "[test_gsp_disc_full] Error from MPI_Allreduce\n");
      EXIT(36);
    }

    memcpy(buffer, sample_loop_f, bytes);
    status = MPI_Allreduce(buffer, sample_loop_f, (2*T_global), MPI_DOUBLE, MPI_SUM, g_cart_grid);
    if (status != MPI_SUCCESS) {
      fprintf(stderr, "[test_gsp_disc_full] Error from MPI_Allreduce\n");
      EXIT(37);
    }

    items = 2 * (size_t)T_global * g_nsample;
    bytes = items * sizeof(double);

    memcpy(buffer, sample_diag_i, bytes);
    status = MPI_Allreduce(buffer, sample_diag_i, (2*T_global*g_nsample), MPI_DOUBLE, MPI_SUM, g_cart_grid);
    if (status != MPI_SUCCESS) {
      fprintf(stderr, "[test_gsp_disc_full] Error from MPI_Allreduce\n");
      EXIT(38);
    }

    memcpy(buffer, sample_diag_f, bytes);
    status = MPI_Allreduce(buffer, sample_diag_f, (2*T_global*g_nsample), MPI_DOUBLE, MPI_SUM, g_cart_grid);
    if (status != MPI_SUCCESS) {
      fprintf(stderr, "[test_gsp_disc_full] Error from MPI_Allreduce\n");
      EXIT(39);
    }

    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc_full] time for xchange = %e seconds\n", retime - ratime);
#endif


    /***********************************************
     * Build correlaotr and I / O
     ***********************************************/
    ratime = _GET_TIME;
#ifdef HAVE_MPI
    if(io_proc == 2) {
#endif
      memset(correlator_re_re,  0, T_global*sizeof(double));
      memset(correlator_re_im,  0, T_global*sizeof(double));
      memset(correlator_im_re,  0, T_global*sizeof(double));
      memset(correlator_im_im,  0, T_global*sizeof(double));

      sample_loop_norm = (2.*g_kappa);

      /* real source - real sink */
      for(i_ti=0; i_ti<T_global; i_ti++) {

        _co_eq_co_ti_re(&sli, (complex*)(sample_loop_i+2*i_ti), sample_loop_norm);
        _co_eq_co(&vli, (complex*)(evecs_loop_i+2*i_ti));

        for(i_tf = 0; i_tf < T_global; i_tf++) {
          i_tfi = ( i_tf - i_ti + T_global ) % T_global;

          /* _co_eq_co_ti_co( (complex*)(&buffer[2*i_tfi]), (complex*)(&correlator[2*i_ti]), (complex*)(&correlator2[2*i_tf])); */

          _co_eq_co_ti_re(&slf, (complex*)(sample_loop_f+2*i_tf), sample_loop_norm);
          _co_eq_co(&vlf, (complex*)(evecs_loop_f+2*i_tf));


          /* set the correlator */
          correlator_re_re[i_tfi] = vli.re * vlf.re + (sli.re*vlf.re + vli.re*slf.re)/sample_num_contract + sli.re*slf.re/(sample_num_contract*(sample_num_contract-1));
          correlator_re_im[i_tfi] = vli.re * vlf.im + (sli.re*vlf.im + vli.re*slf.im)/sample_num_contract + sli.re*slf.im/(sample_num_contract*(sample_num_contract-1));
          correlator_im_re[i_tfi] = vli.im * vlf.re + (sli.im*vlf.re + vli.im*slf.re)/sample_num_contract + sli.im*slf.re/(sample_num_contract*(sample_num_contract-1));
          correlator_im_im[i_tfi] = vli.im * vlf.im + (sli.im*vlf.im + vli.im*slf.im)/sample_num_contract + sli.im*slf.im/(sample_num_contract*(sample_num_contract-1));

          /* subtract the summed element-wise product of diagonals */
          /* re - re */
          dtmp = 0.;
          for(k=0; k<sample_num_contract; k++) {
            dtmp += sample_diag_i[i_ti][2*k] * sample_diag_f[i_tf][2*k];
          }
          correlator_re_re[i_tfi] -= dtmp/(sample_num_contract*(sample_num_contract-1));
       
          /* re - im */
          dtmp = 0.;
          for(k=0; k<sample_num_contract; k++) {
            dtmp += sample_diag_i[i_ti][2*k] * sample_diag_f[i_tf][2*k+1];
          }
          correlator_re_im[i_tfi] -= dtmp/(sample_num_contract*(sample_num_contract-1));

          /* im - re */
          dtmp = 0.;
          for(k=0; k<sample_num_contract; k++) {
            dtmp += sample_diag_i[i_ti][2*k+1] * sample_diag_f[i_tf][2*k];
          }
          correlator_im_re[i_tfi] -= dtmp/(sample_num_contract*(sample_num_contract-1));

          /* im - im */
          dtmp = 0.;
          for(k=0; k<sample_num_contract; k++) {
            dtmp += sample_diag_i[i_ti][2*k+1] * sample_diag_f[i_tf][2*k+1];
          }
          correlator_im_im[i_tfi] -= dtmp/(sample_num_contract*(sample_num_contract-1));

        }  /* end of loop on i_tf */
#ifdef HAVE_LHPC_AFF
        /***********************************************
         * write correlator
         ***********************************************/
        x0    = i_ti;
        items = T_global;
        sprintf(aff_buffer_path, "/re-re/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        /* if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc_full] current aff path = %s\n", aff_buffer_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
        status = aff_node_put_double (affw, affdir, correlator_re_re, (uint32_t)items);
        if(status != 0) {
          fprintf(stderr, "[test_gsp_disc_full] Error from aff_node_put_double, status was %d\n", status);
          EXIT(40);
        }

        sprintf(aff_buffer_path, "/re-im/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        /* if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc_full] current aff path = %s\n", aff_buffer_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
        status = aff_node_put_double (affw, affdir, correlator_re_im, (uint32_t)items);
        if(status != 0) {
          fprintf(stderr, "[test_gsp_disc_full] Error from aff_node_put_double, status was %d\n", status);
          EXIT(41);
        }

        sprintf(aff_buffer_path, "/im-re/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        /* if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc_full] current aff path = %s\n", aff_buffer_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
        status = aff_node_put_double (affw, affdir, correlator_im_re, (uint32_t)items);
        if(status != 0) {
          fprintf(stderr, "[test_gsp_disc_full] Error from aff_node_put_double, status was %d\n", status);
          EXIT(42);
        }

        sprintf(aff_buffer_path, "/im-im/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        /* if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc_full] current aff path = %s\n", aff_buffer_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
        status = aff_node_put_double (affw, affdir, correlator_im_im, (uint32_t)items);
        if(status != 0) {
          fprintf(stderr, "[test_gsp_disc_full] Error from aff_node_put_double, status was %d\n", status);
          EXIT(43);
        }


#else
        /***********************************************
         * write correlator
         ***********************************************/
        x0 = i_ti;
        fprintf(ofs, "# /re-re/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d\n", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        for(i_tf = 0; i_tf < T_global; i_tf++) {
          fprintf(ofs, "\t%25.16e\n", correlator_re_re[i_tf] );
        }

        fprintf(ofs, "# /re-im/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d\n", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        for(i_tf = 0; i_tf < T_global; i_tf++) {
          fprintf(ofs, "\t%25.16e\n", correlator_re_im[i_tf] );
        }

        fprintf(ofs, "# /im-re/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d\n", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        for(i_tf = 0; i_tf < T_global; i_tf++) {
          fprintf(ofs, "\t%25.16e\n", correlator_im_re[i_tf] );
        }

        fprintf(ofs, "# /im-im/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d\n", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        for(i_tf = 0; i_tf < T_global; i_tf++) {
          fprintf(ofs, "\t%25.16e\n", correlator_im_im[i_tf] );
        }
#endif  /* of if def HAVE_LHPC_AFF */
      }  /* end of loop on i_ti */

#ifdef HAVE_MPI
    }  /* end of if io_proc == 2 */
#endif

    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc_full] time for writing = %e seconds\n", retime - ratime);

  }  /* end of loop on i2pt */

  /***********************************************
   * close output file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(io_proc == 2 ) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_gsp_disc_full] Error from aff_writer_close, status was %s\n", aff_status_str);
      EXIT(44);
    }
  }
#else
  fclose(ofs);
#endif


  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  gsp_fini(&gsp_v_w_f);
  gsp_fini(&gsp_xv_xw_f);

  gsp_fini(&gsp_v_w_i);
  gsp_fini(&gsp_xv_xw_i);

  gsp_fini(&gsp_xi_phi_f);
  gsp_fini(&gsp_xxi_xphi_f);

  gsp_fini(&gsp_xi_phi_i);
  gsp_fini(&gsp_xxi_xphi_i);

  if(evecs_eval != NULL) free(evecs_eval);

  if(evecs_loop_i  != NULL) { free(evecs_loop_i);  }
  if(evecs_loop_f  != NULL) { free(evecs_loop_f);  }
  if(sample_loop_i != NULL) { free(sample_loop_i); }
  if(sample_loop_f != NULL) { free(sample_loop_f); }

  if(sample_diag_f != NULL) {
    if(sample_diag_f[0] != NULL) free( sample_diag_f[0] );
    free(sample_diag_f);
  }

  if(sample_diag_i != NULL) { 
    if(sample_diag_i[0] != NULL) free(sample_diag_i[0]);
    free(sample_diag_i); 
  }

  if(buffer != NULL) { free(buffer); }

  if(io_proc == 2) {
    if(correlator_re_re != NULL) free( correlator_re_re );
    if(correlator_re_im != NULL) free( correlator_re_im );
    if(correlator_im_re != NULL) free( correlator_im_re );
    if(correlator_im_im != NULL) free( correlator_im_im );
  }

  free_geometry();

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_gsp_disc_full] %s# [test_gsp_disc_full] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_gsp_disc_full] %s# [test_gsp_disc_full] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

