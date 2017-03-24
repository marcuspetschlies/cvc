/****************************************************
 * test_gsp_oet.cpp
 *
 * Sat Jul  2 18:56:31 CEST 2016
 *
 * PURPOSE:
 * TODO:
 * - use BLAS for matrix multiplication?
 * DONE:
 * CHANGES:
 *
 *     0  1  2  3  id   5  0_5  1_5  2_5  3_5  0_1  0_2  0_3  1_2  1_3  2_3
 *     0  1  2  3   4   5    6    7    8    9   10   11   12   13   14   15
 * g5 g^+ g5
 *     1  1  1  1   1   1    1    1    1    1   -1   -1   -1   -1   -1   -1
 *
 * (g5 g)^+
 *    -1 -1 -1 -1   1   1    1    1    1    1   -1    -1   -1  -1   -1   -1
 *     
 *    
 *     
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

const int g_g5_g_dag_g5[16] = { 1,  1,  1,  1,   1,   1,   1,    1,    1,    1,   -1,   -1,   -1,   -1,   -1,   -1};

const int g_g5_g_dag[16] = { -1, -1, -1, -1 ,  1,   1,    1,    1,    1,    1,   -1,    -1,   -1,  -1,   -1,   -1 };

#define _SQR(_a) ((_a)*(_a))

int main(int argc, char **argv) {
  
  int pvec0[3] = {0,0,0};

  int c, status;
  int i, i2pt, io_proc=2;
  int i_gi, i_gf, i_si, i_sf, i_ti, i_tf, i_tfi, x0;
  int i_g5gi, i_g5gf;
  int filename_set = 0;
#ifdef HAVE_OPENMP
  int threadid, nthreads;
#endif

  int evecs_num=0, evecs_num_contract=0;
  double *evecs_eval = NULL;

  complex w, w2;

  double *****gsp_v_w   = NULL, *****gsp_xv_xw   = NULL;
  double *****gsp_v_v_i = NULL, *****gsp_xv_xv_i = NULL;
  double *****gsp_w_w_f = NULL, *****gsp_xw_xw_f = NULL;
#ifdef HAVE_MPI
  double *****gsp_t=NULL;
  int mrank, mcoords[4];
#endif
  double *correlator=NULL, *loops=NULL, *loops_i, *loops_f;
  double *buffer=NULL;
  double *buffer_re_re=NULL, *buffer_re_im=NULL, *buffer_im_re=NULL, *buffer_im_im=NULL;
  char gsp_tag[100];
  double tr_factor_re=0., tr_factor_im=0., mutilde;

  int verbose = 0;
  char filename[200];

  double ratime, retime;

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
  size_t items, bytes;


#ifdef HAVE_MPI
  MPI_Status mstatus;

  MPI_Init(&argc, &argv);
#endif


  while ((c = getopt(argc, argv, "h?vf:n:c:")) != -1) {
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
      fprintf(stdout, "# [test_gsp_oet] dimension of eigenspace set to%d\n", evecs_num);
      break;
    case 'c':
      evecs_num_contract = atoi(optarg);
      fprintf(stdout, "# [test_gsp_oet] use first %d eigenvectors for contraction\n", evecs_num_contract);
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
  if(g_cart_id==0) fprintf(stdout, "# [test_gsp_oet] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_gsp_oet] calling tmLQCD wrapper init functions\n");

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
    fprintf(stderr, "[test_gsp_oet] Error from init_geometry\n");
    EXIT(4);
  }

  geometry();

#if (defined HAVE_MPI) && ( (defined PARALLELTX) ||  (defined PARALLELTXY) ||  (defined PARALLELTXYZ) )
  fprintf(stderr, "[test_gsp_oet] Error, for now only 1-dimensional MPI-parallelization\n");
  EXIT(5);
#endif

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_oet] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private(nthreads,threadid)
{
  threadid = omp_get_thread_num();
  nthreads = omp_get_num_threads();
  fprintf(stdout, "# [test_gsp_oet] proc%.4d thread%.4d using %d threads\n", g_cart_id, threadid, nthreads);
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_gsp_oet] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if (evecs_num == 0) {
    if(g_cart_id==0) fprintf(stderr, "[test_gsp_oet] eigenspace dimension is 0\n");
    EXIT(6);
  }

  /***********************************************
   * read eigenvalues
   ***********************************************/
  sprintf(gsp_tag, "%s.%.4d", "gsp_eval", Nconf);
  status = gsp_read_eval(&evecs_eval, evecs_num, gsp_tag);
  if( status != 0) {
    fprintf(stderr, "[test_gsp_oet] Error from gsp_read_eval, status was %d\n", status);
    EXIT(7);
  }

  /***********************************************
   * inverse square root of diagonal elements
   ***********************************************/
  for(i=0; i<evecs_num; i++) {
    if(g_cart_id==0) fprintf(stdout, "# [test_gsp_oet] eval %4d %25.16e\n", i, evecs_eval[i]) ;
    evecs_eval[i] = 2.*g_kappa / sqrt(evecs_eval[i]);
  }

  /***********************************************
   * allocate gsp's
   ***********************************************/
  /* allocate */
  ratime = _GET_TIME;
  status = 0;
  status = status ||  gsp_init (&gsp_v_w,    1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_xv_xw,  1, 1, T, evecs_num);

  status = status ||  gsp_init (&gsp_v_v_i,   1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_xv_xv_i, 1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_w_w_f,   1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_xw_xw_f, 1, 1, T, evecs_num);
#ifdef HAVE_MPI
  status = status ||  gsp_init (&gsp_t,   1, 1, T, evecs_num);
#endif
  if(status) {
    fprintf(stderr, "[test_gsp_oet] Error from gsp_init\n");
    EXIT(8);
  }
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_oet] time to initialize gsp = %e seconds\n", retime - ratime);

  /***********************************************
   * correlator
   ***********************************************/
  correlator = (double*)malloc(2*T_global*sizeof(double));
  if(correlator == NULL) {
    fprintf(stderr, "[test_gsp_oet] Error from malloc\n");
    EXIT(9);
  }
  loops = (double*)malloc(4*T_global*sizeof(double));
  if(loops == NULL) {
    fprintf(stderr, "[test_gsp_oet] Error from malloc\n");
    EXIT(10);
  }
  loops_i = loops;
  loops_f = loops + 2*T_global;
 

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [test_gsp_oet] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [test_gsp_oet] proc%.4d is send process\n", g_cart_id);
    } else {
      io_proc = 0;
    }
  }
#else
  io_proc = 2;
#endif

  /***********************************************
   * output file
   ***********************************************/
  if(io_proc == 2) {
#ifdef HAVE_LHPC_AFF

    aff_status_str = (char*)aff_version();
    fprintf(stdout, "# [test_gsp_oet] using aff version %s\n", aff_status_str);

    sprintf(filename, "%s.%.4d.aff", "gsp_oet", Nconf);
    fprintf(stdout, "# [test_gsp_oet] writing gsp data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_gsp_oet] Error from aff_writer, status was %s\n", aff_status_str);
      EXIT(11);
    }

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[test_gsp_oet] Error, aff writer is not initialized\n");
      EXIT(12);
    }

    aff_buffer = (double _Complex*)malloc(T_global*sizeof(double _Complex));
    if(aff_buffer == NULL) {
      fprintf(stderr, "[test_gsp_oet] Error from malloc\n");
      EXIT(13);
    }
#else
    sprintf(filename, "%s.%.4d.%.4d", "gsp_oet", Nconf, evecs_num_contract);
    ofs = fopen(filename, "w");
    if(ofs == NULL) {
      fprintf(stderr, "[test_gsp_oet] Error, could not open file %s for writing\n", filename);
      EXIT(14);
    }
#endif
  }  /* of if io_proc == 2  */


  /***********************************************
   * allocate buffer
   ***********************************************/
  buffer = (double*)malloc(2*T_global*sizeof(double));
  if( buffer == NULL ) {
    fprintf(stderr, "[test_gsp_oet] Error from malloc\n");
    EXIT(15);
  }
  if(io_proc == 2) {
    buffer_re_re = (double*)malloc(T_global*sizeof(double));
    if( buffer_re_re == NULL ) {
      fprintf(stderr, "[test_gsp_oet] Error from malloc\n");
      EXIT(24);
    }

    buffer_re_im = (double*)malloc(T_global*sizeof(double));
    if( buffer_re_im == NULL ) {
      fprintf(stderr, "[test_gsp_oet] Error from malloc\n");
      EXIT(25);
    }

    buffer_im_re = (double*)malloc(T_global*sizeof(double));
    if( buffer_im_re == NULL ) {
      fprintf(stderr, "[test_gsp_oet] Error from malloc\n");
      EXIT(24);
    }

    buffer_im_im = (double*)malloc(T_global*sizeof(double));
    if( buffer_im_im == NULL ) {
      fprintf(stderr, "[test_gsp_oet] Error from malloc\n");
      EXIT(24);
    }
  }  /* end of if io_proc == 2 */

  /***********************************************
   * check, whether evecs_num_contract was set
   ***********************************************/
  if(evecs_num_contract == 0) {
    evecs_num_contract = evecs_num;
    if(g_cart_id == 0) fprintf(stdout, "[test_gsp_2pt] reset evecs_num_contract to default value %d\n", evecs_num_contract);
  }

  /***********************************************
   * set trace factor
   ***********************************************/
  mutilde = 2. * g_kappa * g_mu;
  tr_factor_re = 2*g_kappa / (1 + mutilde*mutilde );
  tr_factor_im = mutilde * tr_factor_re;


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
    i_gi = g_m_m_2pt_list[i2pt].gi;
    i_gf = g_m_m_2pt_list[i2pt].gf;
    i_g5gi = gamma_id_g5_ti_gamma[g_m_m_2pt_list[i2pt].gi];
    i_g5gf = gamma_id_g5_ti_gamma[g_m_m_2pt_list[i2pt].gf];
    /* gamma sign at source and sink */
    i_si = gamma_sign_g5_ti_gamma[g_m_m_2pt_list[i2pt].gi];
    i_sf = gamma_sign_g5_ti_gamma[g_m_m_2pt_list[i2pt].gf];

    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_oet] (%d, %d) to (%d, %d); (%d, %d)\n", i_gi, i_gf, i_g5gi, i_si, i_g5gf, i_sf);

    memset(loops, 0, 4*T_global*sizeof(double));

    /***********************************************
     * at source 
     ***********************************************/

    /* 1st part */

    /* (XV)^+ Gamma (XW) */
    sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeow", Nconf);
    status = gsp_read_node (gsp_xv_xw[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_oet] Error from gsp_read_node\n");
      EXIT(17);
    }
    
    for(i_ti=0; i_ti<T; i_ti++) {
      x0 = i_ti + g_proc_coords[0] * T;
      co_eq_tr_gsp (&w2, gsp_xv_xw[0][0][i_ti], evecs_eval, evecs_num_contract);
      loops_i[2*x0  ] = tr_factor_re * (1 + g_g5_g_dag_g5[i_gi]) * w2.re;
      loops_i[2*x0+1] = tr_factor_re * (1 - g_g5_g_dag_g5[i_gi]) * w2.im; 
    }

    /* (XV)^+ g5 Gamma (XW) */
    sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeow", Nconf);
    status = gsp_read_node (gsp_xv_xw[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_g5gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_oet] Error from gsp_read_node\n");
      EXIT(17);
    }
    
    for(i_ti=0; i_ti<T; i_ti++) {
      x0 = i_ti + g_proc_coords[0] * T;
      co_eq_tr_gsp (&w2, gsp_xv_xw[0][0][i_ti], evecs_eval, evecs_num_contract);
      loops_i[2*x0  ] +=  tr_factor_im * (1 - g_g5_g_dag[i_gi]) * w2.im * i_si;
      loops_i[2*x0+1] += -tr_factor_im * (1 + g_g5_g_dag[i_gi]) * w2.re * i_si;
    }

    ratime = _GET_TIME;
    /* 2nd part */

    sprintf(gsp_tag, "%s.%.4d", "gsp_v_v", Nconf);
    status = gsp_read_node (gsp_v_v_i[0][0], evecs_num, pvec0, 4, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_oet] Error from gsp_read_node\n");
      EXIT(16);
    }

    sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeobarv", Nconf);
    status = gsp_read_node (gsp_xv_xv_i[0][0], evecs_num, pvec0, 4, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_oet] Error from gsp_read_node\n");
      EXIT(17);
    }


    sprintf(gsp_tag, "%s.%.4d", "gsp_w_w", Nconf);
    status = gsp_read_node (gsp_w_w_f[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_g5gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_oet] Error from gsp_read_node\n");
      EXIT(18);
    }

    sprintf(gsp_tag, "%s.%.4d", "gsp_xeow_xeow", Nconf);
    status = gsp_read_node (gsp_xw_xw_f[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_g5gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_oet] Error from gsp_read_node\n");
      EXIT(19);
    }

    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_oet] time to read gsp nodes gsp = %e seconds\n", retime - ratime);

    /* reduce */
    ratime = _GET_TIME;

    /* local sum of V^+V over ti */
    for(i_ti=1; i_ti<T; i_ti++) {
      gsp_pl_eq_gsp((double _Complex**)gsp_v_v_i[0][0][0], (double _Complex**)gsp_v_v_i[0][0][i_ti], evecs_num);
    }   
#ifdef HAVE_MPI
    /* global sum */
    memcpy(gsp_t[0][0][0][0], gsp_v_v_i[0][0][0][0], 2*evecs_num*evecs_num*sizeof(double) );

    status = MPI_Allreduce(gsp_t[0][0][0][0], gsp_v_v_i[0][0][0][0], 2*evecs_num*evecs_num, MPI_DOUBLE, MPI_SUM, g_tr_comm);
    if(status != MPI_SUCCESS) {
      fprintf(stderr, "[test_gsp_oet] Error from MPI_Allreduce, status was %d\n", status);
      EXIT(12);
    }
#endif

    /* local sum of (XV)^+ (XV) over ti */
    for(i_ti=1; i_ti<T; i_ti++) {
      gsp_pl_eq_gsp( (double _Complex**)gsp_xv_xv_i[0][0][0], (double _Complex**)gsp_xv_xv_i[0][0][i_ti], evecs_num);
    }   
#ifdef HAVE_MPI
    /* global sum */
    memcpy(gsp_t[0][0][0][0], gsp_xv_xv_i[0][0][0][0], 2*evecs_num*evecs_num*sizeof(double) );

    status = MPI_Allreduce(gsp_t[0][0][0][0], gsp_xv_xv_i[0][0][0][0], 2*evecs_num*evecs_num, MPI_DOUBLE, MPI_SUM, g_tr_comm);
    if(status != MPI_SUCCESS) {
      fprintf(stderr, "[test_gsp_oet] Error from MPI_Allreduce, status was %d\n", status);
      EXIT(13);
    }
#endif

    for(i_ti = 0; i_ti<T; i_ti++) {
      x0 = i_ti + g_proc_coords[0] * T;

      _co_eq_zero(&w);
      /* accumulate four trace terms */
      co_eq_tr_gsp_ti_gsp (&w2, gsp_xv_xv_i[0][0][0], gsp_xw_xw_f[0][0][i_ti], evecs_eval, evecs_num_contract);
      _co_pl_eq_co(&w, &w2);

      co_eq_tr_gsp_ti_gsp (&w2, gsp_v_v_i[0][0][0], gsp_xw_xw_f[0][0][i_ti], evecs_eval, evecs_num_contract);
      _co_pl_eq_co(&w, &w2);


      co_eq_tr_gsp_ti_gsp (&w2, gsp_xv_xv_i[0][0][0], gsp_w_w_f[0][0][i_ti], evecs_eval, evecs_num_contract);
      _co_pl_eq_co(&w, &w2);

      co_eq_tr_gsp_ti_gsp (&w2, gsp_v_v_i[0][0][0], gsp_w_w_f[0][0][i_ti], evecs_eval, evecs_num_contract);
      _co_pl_eq_co(&w, &w2);

      _co_ti_eq_re(&w, i_si);
  
      loops_i[2*x0  ] += w.re;
      loops_i[2*x0+1] += w.im;

    }  /* end of loop on ti */
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_oet] time for souce loop reduction = %e seconds\n", retime - ratime);


    /***********************************************
     * at sink
     ***********************************************/

    /* 1st part */

    /* (XV)^+ Gamma (XW) */
    sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeow", Nconf);
    status = gsp_read_node (gsp_xv_xw[0][0], evecs_num, g_m_m_2pt_list[i2pt].pf, i_gf, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_oet] Error from gsp_read_node\n");
      EXIT(17);
    }
    
    for(i_ti=0; i_ti<T; i_ti++) {
      x0 = i_ti + g_proc_coords[0] * T;
      co_eq_tr_gsp (&w2, gsp_xv_xw[0][0][i_ti], evecs_eval, evecs_num_contract);
      loops_f[2*x0  ] = tr_factor_re * (1 + g_g5_g_dag_g5[i_gf]) * w2.re;
      loops_f[2*x0+1] = tr_factor_re * (1 - g_g5_g_dag_g5[i_gf]) * w2.im; 
    }

    /* (XV)^+ g5 Gamma (XW) */
    sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeow", Nconf);
    status = gsp_read_node (gsp_xv_xw[0][0], evecs_num, g_m_m_2pt_list[i2pt].pf, i_g5gf, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_oet] Error from gsp_read_node\n");
      EXIT(17);
    }
    
    for(i_ti=0; i_ti<T; i_ti++) {
      x0 = i_ti + g_proc_coords[0] * T;
      co_eq_tr_gsp (&w2, gsp_xv_xw[0][0][i_ti], evecs_eval, evecs_num_contract);
      loops_f[2*x0  ] +=  tr_factor_im * (1 - g_g5_g_dag[i_gf]) * w2.im * i_sf;
      loops_f[2*x0+1] += -tr_factor_im * (1 + g_g5_g_dag[i_gf]) * w2.re * i_sf;
    }

    ratime = _GET_TIME;
    /* 2nd part */

    sprintf(gsp_tag, "%s.%.4d", "gsp_w_w", Nconf);
    status = gsp_read_node (gsp_w_w_f[0][0], evecs_num, g_m_m_2pt_list[i2pt].pf, i_g5gf, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_oet] Error from gsp_read_node\n");
      EXIT(18);
    }

    sprintf(gsp_tag, "%s.%.4d", "gsp_xeow_xeow", Nconf);
    status = gsp_read_node (gsp_xw_xw_f[0][0], evecs_num, g_m_m_2pt_list[i2pt].pf, i_g5gf, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_oet] Error from gsp_read_node\n");
      EXIT(19);
    }

    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_oet] time to read gsp nodes gsp = %e seconds\n", retime - ratime);

    /* reduce */
    ratime = _GET_TIME;

    for(i_ti = 0; i_ti<T; i_ti++) {
      x0 = i_ti + g_proc_coords[0] * T;

      _co_eq_zero(&w);
      /* accumulate four trace terms */
      co_eq_tr_gsp_ti_gsp (&w2, gsp_xv_xv_i[0][0][0], gsp_xw_xw_f[0][0][i_ti], evecs_eval, evecs_num_contract);
      _co_pl_eq_co(&w, &w2);

      co_eq_tr_gsp_ti_gsp (&w2, gsp_v_v_i[0][0][0], gsp_xw_xw_f[0][0][i_ti], evecs_eval, evecs_num_contract);
      _co_pl_eq_co(&w, &w2);


      co_eq_tr_gsp_ti_gsp (&w2, gsp_xv_xv_i[0][0][0], gsp_w_w_f[0][0][i_ti], evecs_eval, evecs_num_contract);
      _co_pl_eq_co(&w, &w2);

      co_eq_tr_gsp_ti_gsp (&w2, gsp_v_v_i[0][0][0], gsp_w_w_f[0][0][i_ti], evecs_eval, evecs_num_contract);
      _co_pl_eq_co(&w, &w2);

      _co_ti_eq_re(&w, i_sf);
  
      loops_f[2*x0  ] += w.re;
      loops_f[2*x0+1] += w.im;

    }  /* end of loop on ti */
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_oet] time for sink loop reduction = %e seconds\n", retime - ratime);

    ratime = _GET_TIME;

#ifdef HAVE_MPI
    /***********************************************
     * reduce
     ***********************************************/
    ratime = _GET_TIME;
    items = 2 * (size_t)T_global;
    bytes = items * sizeof(double);

    memcpy(buffer, loops_i, bytes);
    status = MPI_Allreduce(buffer, loops_i, (2*T_global), MPI_DOUBLE, MPI_SUM, g_cart_grid);
    if (status != MPI_SUCCESS) {
      fprintf(stderr, "[test_gsp_oet] Error from MPI_Allreduce\n");
      EXIT(20);
    }

    memcpy(buffer, loops_f, bytes);
    status = MPI_Allreduce(buffer, loops_f, (2*T_global), MPI_DOUBLE, MPI_SUM, g_cart_grid);
    if (status != MPI_SUCCESS) {
      fprintf(stderr, "[test_gsp_oet] Error from MPI_Allreduce\n");
      EXIT(21);
    }

    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_oet] time for xchange = %e seconds\n", retime - ratime);
#endif

    /***********************************************
     * I / O
     ***********************************************/
    ratime = _GET_TIME;

    if(io_proc == 2) {

      /* real source - real sink */
      for(i_ti=0; i_ti<T_global; i_ti++) {

        for(i_tf = 0; i_tf < T_global; i_tf++) {
          i_tfi = ( i_tf - i_ti + T_global ) % T_global;

          buffer_re_re[i_tfi] = loops_i[2*i_ti  ] * loops_f[2*i_tf  ] * _SQR(2.*g_mu);
          buffer_re_im[i_tfi] = loops_i[2*i_ti  ] * loops_f[2*i_tf+1] * _SQR(2.*g_mu);
          buffer_im_re[i_tfi] = loops_i[2*i_ti+1] * loops_f[2*i_tf  ] * _SQR(2.*g_mu);
          buffer_im_im[i_tfi] = loops_i[2*i_ti+1] * loops_f[2*i_tf+1] * _SQR(2.*g_mu);
        }
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
        /* if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_oet] current aff path = %s\n", aff_buffer_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
        status = aff_node_put_double (affw, affdir, buffer_re_re, (uint32_t)items);
        if(status != 0) {
          fprintf(stderr, "[test_gsp_oet] Error from aff_node_put_double, status was %d\n", status);
          EXIT(26);
        }

        sprintf(aff_buffer_path, "/re-im/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        /* if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_oet] current aff path = %s\n", aff_buffer_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
        status = aff_node_put_double (affw, affdir, buffer_re_im, (uint32_t)items);
        if(status != 0) {
          fprintf(stderr, "[test_gsp_oet] Error from aff_node_put_double, status was %d\n", status);
          EXIT(27);
        }

        sprintf(aff_buffer_path, "/im-re/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        /* if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_oet] current aff path = %s\n", aff_buffer_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
        status = aff_node_put_double (affw, affdir, buffer_im_re, (uint32_t)items);
        if(status != 0) {
          fprintf(stderr, "[test_gsp_oet] Error from aff_node_put_double, status was %d\n", status);
          EXIT(28);
        }

        sprintf(aff_buffer_path, "/im-im/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        /* if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_oet] current aff path = %s\n", aff_buffer_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
        status = aff_node_put_double (affw, affdir, buffer_im_im, (uint32_t)items);
        if(status != 0) {
          fprintf(stderr, "[test_gsp_oet] Error from aff_node_put_double, status was %d\n", status);
          EXIT(29);
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
          fprintf(ofs, "\t%25.16e\n", buffer_re_re[i_tf] );
        }

        fprintf(ofs, "# /re-im/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d\n", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        for(i_tf = 0; i_tf < T_global; i_tf++) {
          fprintf(ofs, "\t%25.16e\n", buffer_re_im[i_tf] );
        }

        fprintf(ofs, "# /im-re/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d\n", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        for(i_tf = 0; i_tf < T_global; i_tf++) {
          fprintf(ofs, "\t%25.16e\n", buffer_im_re[i_tf] );
        }

        fprintf(ofs, "# /im-im/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d\n", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        for(i_tf = 0; i_tf < T_global; i_tf++) {
          fprintf(ofs, "\t%25.16e\n", buffer_im_im[i_tf] );
        }
#endif  /* of if def HAVE_LHPC_AFF */
      }  /* end of loop on i_ti */

    }  /* end of if io_proc == 2 */

    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_oet] time for writing = %e seconds\n", retime - ratime);

  }  /* end of loop on i2pt */

  /***********************************************
   * close output file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(io_proc == 2 ) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_gsp_oet] Error from aff_writer_close, status was %s\n", aff_status_str);
      EXIT(23);
    }
  }
#else
  fclose(ofs);
#endif


  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  gsp_fini(&gsp_v_w);
  gsp_fini(&gsp_xv_xw);

  gsp_fini(&gsp_v_v_i);
  gsp_fini(&gsp_xv_xv_i);
  gsp_fini(&gsp_w_w_f);
  gsp_fini(&gsp_xw_xw_f);

#ifdef HAVE_MPI
  gsp_fini(&gsp_t);
#endif
 
  if(evecs_eval != NULL) free(evecs_eval);
  if(correlator != NULL) {
    free(correlator);
  }
 if(loops!= NULL) free(loops);

  if(buffer != NULL) {
    free(buffer);
  }
  if(io_proc == 2) {
    if( buffer_re_re != NULL ) free( buffer_re_re );
    if( buffer_re_im != NULL ) free( buffer_re_im);
    if( buffer_im_re != NULL ) free( buffer_im_re );
    if( buffer_im_im != NULL ) free( buffer_im_im );
  }

  free_geometry();

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_gsp_oet] %s# [test_gsp_oet] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_gsp_oet] %s# [test_gsp_oet] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

