/****************************************************
 * test_gsp_2pt_full.cpp
 *
 * Thu Aug  4 17:12:55 CEST 2016
 *
 * PURPOSE:
 * TODO:
 * - use BLAS for matrix multiplication?
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
  int i, i2pt, iproc;
  int i_gi, i_gf, i_si, i_sf, i_ti, i_tf, i_tfi, x0;
  int filename_set = 0;
  int append_output = 0;
#ifdef HAVE_OPENMP
  int threadid, nthreads;
#endif

  int evecs_num=0, evecs_num_contract=0, sample_num_contract = 0;
  double *evecs_eval = NULL;

  complex w, w2;

  double *****gsp_w_w_f = NULL, *****gsp_v_w_f   = NULL, *****gsp_xw_xw_f = NULL, *****gsp_xv_xw_f = NULL;
  double *****gsp_v_v_i = NULL, *****gsp_xv_xv_i = NULL, *****gsp_v_w_i   = NULL, *****gsp_xv_xw_i = NULL;
#ifdef HAVE_MPI
  double *****gsp_t=NULL;
  int mrank, mcoords[4];
  int io_proc = 0;
#endif
  double **correlator=NULL, **correlator2=NULL;
  char gsp_tag[100];

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


  while ((c = getopt(argc, argv, "ah?vf:n:c:C:")) != -1) {
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
      fprintf(stdout, "# [test_gsp_2pt] dimension of eigenspace set to %d\n", evecs_num);
      break;
    case 'c':
      evecs_num_contract = atoi(optarg);
      fprintf(stdout, "# [test_gsp_2pt] use first %d eigenvectors for contraction\n", evecs_num_contract);
      break;
    case 'C':
      sample_num_contract = atoi(optarg);
      fprintf(stdout, "# [test_gsp_2pt] use first %d samples for contraction\n", sample_num_contract);
      break;
    case 'a':
      append_output = 1;
      fprintf(stdout, "# [test_gsp_2pt] will append output\n");
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
  if(g_cart_id==0) fprintf(stdout, "# [test_gsp_2pt] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_gsp_2pt] calling tmLQCD wrapper init functions\n");

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
    fprintf(stderr, "[test_gsp_2pt] Error from init_geometry\n");
    EXIT(101);
  }

  geometry();

#if (defined HAVE_MPI) && ( (defined PARALLELTX) ||  (defined PARALLELTXY) ||  (defined PARALLELTXYZ) )
  fprintf(stderr, "[test_gsp_2pt] Error, for now only 1-dimensional MPI-parallelization\n");
  EXIT(130);
#endif

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_2pt] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private(nthreads,threadid)
{
  threadid = omp_get_thread_num();
  nthreads = omp_get_num_threads();
  fprintf(stdout, "# [test_gsp_2pt] proc%.4d thread%.4d using %d threads\n", g_cart_id, threadid, nthreads);
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_gsp_2pt] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if (evecs_num == 0) {
    if(g_cart_id==0) fprintf(stderr, "[test_gsp_2pt] eigenspace dimension is 0\n");
    EXIT(1);
  }

  /***********************************************
   * read eigenvalues
   ***********************************************/
  sprintf(gsp_tag, "%s.%.4d", "gsp_eval", Nconf);
  status = gsp_read_eval(&evecs_eval, evecs_num, gsp_tag);
  if( status != 0) {
    fprintf(stderr, "[test_gsp_2pt] Error from gsp_read_eval, status was %d\n", status);
    EXIT(2);
  }

  /***********************************************
   * inverse square root of diagonal elements
   ***********************************************/
  for(i=0; i<evecs_num; i++) {
    if(g_cart_id==0) fprintf(stdout, "# [test_gsp_2pt] eval %4d %25.16e\n", i, evecs_eval[i]) ;
    evecs_eval[i] = 2.*g_kappa / sqrt(evecs_eval[i]);
  }

  /***********************************************
   * allocate gsp's
   ***********************************************/
  /* allocate */
  ratime = _GET_TIME;
  status = 0;
  status = status ||  gsp_init (&gsp_w_w_f,   1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_xw_xw_f, 1, 1, T, evecs_num);

  status = status ||  gsp_init (&gsp_v_v_i,   1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_xv_xv_i, 1, 1, T, evecs_num);

  status = status ||  gsp_init (&gsp_v_w_f,   1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_xv_xw_f, 1, 1, T, evecs_num);

  status = status ||  gsp_init (&gsp_v_w_i,   1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_xv_xw_i, 1, 1, T, evecs_num);
#ifdef HAVE_MPI
  status = status ||  gsp_init (&gsp_t,   1, 1, T, evecs_num);
#endif
  if(status) {
    fprintf(stderr, "[test_gsp_2pt] Error from gsp_init\n");
    EXIT(155);
  }
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_2pt] time to initialize gsp = %e seconds\n", retime - ratime);

  /***********************************************
   * correlator
   ***********************************************/
  correlator = (double**)malloc(T*sizeof(double*));
  if(correlator == NULL) {
    fprintf(stderr, "[test_gsp_2pt] Error from malloc\n");
    EXIT(163);
  }
  correlator[0] = (double*)malloc(2*T*T_global*sizeof(double));
  if(correlator[0] == NULL) {
    fprintf(stderr, "[test_gsp_2pt] Error from malloc\n");
    EXIT(164);
  }
  for(i=1; i<T; i++) correlator[i] = correlator[i-1] + 2 * T_global;
 
  correlator2 = (double**)malloc(T*sizeof(double*));
  if(correlator2 == NULL) {
    fprintf(stderr, "[test_gsp_2pt] Error from malloc\n");
    EXIT(163);
  }
  correlator2[0] = (double*)malloc(2*T*T_global*sizeof(double));
  if(correlator2[0] == NULL) {
    fprintf(stderr, "[test_gsp_2pt] Error from malloc\n");
    EXIT(164);
  }
  for(i=1; i<T; i++) correlator2[i] = correlator2[i-1] + 2 * T_global;
 

  /***********************************************
   * output file
   ***********************************************/
  if(g_cart_id == 0) {
#ifdef HAVE_LHPC_AFF

    aff_status_str = (char*)aff_version();
    fprintf(stdout, "# [test_gsp_2pt] using aff version %s\n", aff_status_str);

    sprintf(filename, "%s.%.4d.aff", "gsp_correlator", Nconf);
    fprintf(stdout, "# [test_gsp_2pt] writing gsp data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_gsp_2pt] Error from aff_writer, status was %s\n", aff_status_str);
      EXIT(173);
    }

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[test_gsp_2pt] Error, aff writer is not initialized\n");
      EXIT(174);
    }

    aff_buffer = (double _Complex*)malloc(T_global*sizeof(double _Complex));
    if(aff_buffer == NULL) {
      fprintf(stderr, "[test_gsp_2pt] Error from malloc\n");
      EXIT(175);
    }
#else
    sprintf(filename, "%s.%.4d.%.4d", "gsp_correlator", Nconf, evecs_num_contract);
    if(append_output == 1 ) {
      ofs = fopen(filename, "a");
    } else {
      ofs = fopen(filename, "w");
    }
    if(ofs == NULL) {
      fprintf(stderr, "[test_gsp_2pt] Error, could not open file %s for writing\n", filename);
      EXIT(177);
    }
#endif
  }  /* of if g_cart_id == 0  */

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [test_gsp_2pt] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [test_gsp_2pt] proc%.4d is send process\n", g_cart_id);
    } else {
      io_proc = 0;
    }
  }
#endif

  /***********************************************
   * check, whether evecs_num_contract was set
   ***********************************************/
  if(evecs_num_contract == 0) {
    evecs_num_contract = evecs_num;
    if(g_cart_id == 0) fprintf(stdout, "[test_gsp_2pt] reset evecs_num_contract to default value %d\n", evecs_num_contract);
  }

  /***********************************************
   * loops on configurations of momenta
   * at source and
   * sink and Gamma at source and sink
   *
   * at sink
   *   gsp_w_w_f
   *   gsp_xw_xw_f
   *   gsp_v_w_f
   *   gsp_xv_xw_f
   *
   * at source
   *   gsp_v_v_i
   *   gsp_xv_xv_i
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

    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_2pt] (%d, %d) to (%d, %d); (%d, %d)\n", g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].gf, i_gi, i_si, i_gf, i_sf);

    ratime = _GET_TIME;

    /* at source */
    sprintf(gsp_tag, "%s.%.4d", "gsp_v_v", Nconf);
    status = gsp_read_node (gsp_v_v_i[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_2pt] Error from gsp_read_node\n");
      EXIT(156);
    }
    /* TEST */
    /* gsp_printf (gsp_v_v_i[0][0], evecs_num, "gsp_v_v_i", stdout); */

    sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeobarv", Nconf);
    status = gsp_read_node (gsp_xv_xv_i[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_2pt] Error from gsp_read_node\n");
      EXIT(158);
    }
    /* TEST */
    /* gsp_printf (gsp_xv_xv_i[0][0], evecs_num, "gsp_xv_xv_i", stdout); */

    sprintf(gsp_tag, "%s.%.4d", "gsp_v_w", Nconf);
    status = gsp_read_node (gsp_v_w_i[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_2pt] Error from gsp_read_node\n");
      EXIT(160);
    }
    /* TEST */
    /* gsp_printf (gsp_v_w_i[0][0], evecs_num, "gsp_v_w_i", stdout); */

    sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeow", Nconf);
    status = gsp_read_node (gsp_xv_xw_i[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_2pt] Error from gsp_read_node\n");
      EXIT(161);
    }
    /* TEST */
    /* gsp_printf (gsp_xv_xw_i[0][0], evecs_num, "gsp_xv_xw_i", stdout); */

    /* at sink */
    sprintf(gsp_tag, "%s.%.4d", "gsp_w_w", Nconf);
    status = gsp_read_node (gsp_w_w_f[0][0], evecs_num, g_m_m_2pt_list[i2pt].pf, i_gf, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_2pt] Error from gsp_read_node\n");
      EXIT(140);
    }
    /* TEST */
    /* gsp_printf (gsp_w_w_f[0][0], evecs_num, "gsp_w_w_f", stdout); */

    sprintf(gsp_tag, "%s.%.4d", "gsp_xeow_xeow", Nconf);
    status = gsp_read_node (gsp_xw_xw_f[0][0], evecs_num, g_m_m_2pt_list[i2pt].pf, i_gf, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_2pt] Error from gsp_read_node\n");
      EXIT(141);
    }
    /* TEST */
    /* gsp_printf (gsp_xw_xw_f[0][0], evecs_num, "gsp_xw_xw_f", stdout); */

    sprintf(gsp_tag, "%s.%.4d", "gsp_v_w", Nconf);
    status = gsp_read_node (gsp_v_w_f[0][0], evecs_num, g_m_m_2pt_list[i2pt].pf, i_gf, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_2pt] Error from gsp_read_node\n");
      EXIT(142);
    }
    /* TEST */
    /* gsp_printf (gsp_v_w_f[0][0], evecs_num, "gsp_v_w_f", stdout); */

    sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeow", Nconf);
    status = gsp_read_node (gsp_xv_xw_f[0][0], evecs_num, g_m_m_2pt_list[i2pt].pf, i_gf, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_2pt] Error from gsp_read_node\n");
      EXIT(143);
    }
    /* TEST */
    /* gsp_printf (gsp_xv_xw_f[0][0], evecs_num, "gsp_xv_xw_f", stdout); */

    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_2pt] time to read gsp nodes gsp = %e seconds\n", retime - ratime);


    /* reduce (trace) matrix product 
     *   for all combinations of source and sink time
     */
    memset(correlator[0], 0, 2*T_global*T*sizeof(double));
    memset(correlator2[0], 0, 2*T_global*T*sizeof(double));

    for(iproc = 0; iproc < g_nproc_t; iproc++) {
      if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_2pt] processing sink time block %d\n", iproc);

      for(i_ti = 0; i_ti<T; i_ti++) {
        for(i_tf = 0; i_tf<T; i_tf++) {

          ratime = _GET_TIME;
          _co_eq_zero(&w);

          /* accumulate four trace terms */
          co_eq_tr_gsp_ti_gsp (&w2, gsp_xv_xv_i[0][0][i_ti], gsp_xw_xw_f[0][0][i_tf], evecs_eval, evecs_num_contract);
          _co_pl_eq_co(&w, &w2);

          /* TEST */
          /* fprintf(stdout, "# [test_gsp_2pt] ti=%2d tf=%2d xv_xv x xw_xw = %25.16e %25.16e\n", i_ti, i_tf, w2.re, w2.im); */

          co_eq_tr_gsp_ti_gsp (&w2, gsp_v_v_i[0][0][i_ti],   gsp_xw_xw_f[0][0][i_tf], evecs_eval, evecs_num_contract);
          _co_pl_eq_co(&w, &w2);

          /* TEST */
          /* fprintf(stdout, "# [test_gsp_2pt] ti=%2d tf=%2d v_v x xw_xw = %25.16e %25.16e\n", i_ti, i_tf, w2.re, w2.im); */

          co_eq_tr_gsp_ti_gsp (&w2, gsp_xv_xv_i[0][0][i_ti], gsp_w_w_f[0][0][i_tf],   evecs_eval, evecs_num_contract);
          _co_pl_eq_co(&w, &w2);

          /* TEST */
          /* fprintf(stdout, "# [test_gsp_2pt] ti=%2d tf=%2d xv_xv x w_w = %25.16e %25.16e\n", i_ti, i_tf, w2.re, w2.im); */

          co_eq_tr_gsp_ti_gsp (&w2, gsp_v_v_i[0][0][i_ti],   gsp_w_w_f[0][0][i_tf],   evecs_eval, evecs_num_contract);
          _co_pl_eq_co(&w, &w2);

          /* TEST */
          /* fprintf(stdout, "# [test_gsp_2pt] ti=%2d tf=%2d v_v x w_w = %25.16e %25.16e\n\n\n", i_ti, i_tf, w2.re, w2.im); */

          /* multiply with sign from source and sink */
          _co_ti_eq_re(&w, (i_si * i_sf) );
  
          i_tfi = ( i_tf + iproc * T - i_ti + T_global ) % T_global;

          correlator[i_ti][2*i_tfi  ] = w.re;
          correlator[i_ti][2*i_tfi+1] = w.im;

          retime = _GET_TIME;
          if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_2pt] time for flavor nonsinglet reduction = %e seconds\n", retime - ratime);

          ratime = _GET_TIME;
          _co_eq_zero(&w);

          /* accumulate four trace terms */
          co_eq_tr_gsp_ti_gsp (&w2, gsp_xv_xw_i[0][0][i_ti], gsp_xv_xw_f[0][0][i_tf], evecs_eval, evecs_num_contract);
          _co_pl_eq_co(&w, &w2);

          /* TEST */
          /* fprintf(stdout, "# [test_gsp_2pt] ti=%2d tf=%2d xv_xw x xv_xw = %25.16e %25.16e\n", i_ti, i_tf, w2.re, w2.im); */

          co_eq_tr_gsp_ti_gsp (&w2, gsp_v_w_i[0][0][i_ti],   gsp_xv_xw_f[0][0][i_tf], evecs_eval, evecs_num_contract);
          _co_pl_eq_co(&w, &w2);

          /* TEST */
          /* fprintf(stdout, "# [test_gsp_2pt] ti=%2d tf=%2d v_w x xv_xw = %25.16e %25.16e\n", i_ti, i_tf, w2.re, w2.im); */

          co_eq_tr_gsp_ti_gsp (&w2, gsp_xv_xw_i[0][0][i_ti], gsp_v_w_f[0][0][i_tf],   evecs_eval, evecs_num_contract);
          _co_pl_eq_co(&w, &w2);

          /* TEST */
          /* fprintf(stdout, "# [test_gsp_2pt] ti=%2d tf=%2d xv_xw x v_w = %25.16e %25.16e\n", i_ti, i_tf, w2.re, w2.im); */

          co_eq_tr_gsp_ti_gsp (&w2, gsp_v_w_i[0][0][i_ti],   gsp_v_w_f[0][0][i_tf],   evecs_eval, evecs_num_contract);
          _co_pl_eq_co(&w, &w2);

          /* TEST */
          /* fprintf(stdout, "# [test_gsp_2pt] ti=%2d tf=%2d v_w x v_w = %25.16e %25.16e\n\n\n", i_ti, i_tf, w2.re, w2.im); */

          /* multiply with sign from source and sink */
          _co_ti_eq_re(&w, (i_si * i_sf) );
  
          i_tfi = ( i_tf + iproc * T - i_ti + T_global ) % T_global;

          correlator2[i_ti][2*i_tfi  ] = w.re;
          correlator2[i_ti][2*i_tfi+1] = w.im;

          retime = _GET_TIME;
          if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_2pt] time for flavor singlet reduction = %e seconds\n", retime - ratime);


        }  /* end of loop on tf */
      }    /* end of loop on ti */

#ifdef HAVE_MPI
      items = 2 * T * (size_t)(evecs_num * evecs_num);
      bytes = items * sizeof(double);

      ratime = _GET_TIME;

      memcpy( gsp_t[0][0][0][0], gsp_w_w_f[0][0][0][0], bytes );
      status = MPI_Sendrecv(gsp_t[0][0][0][0], items, MPI_DOUBLE, g_nb_t_dn, 101, gsp_w_w_f[0][0][0][0],   items, MPI_DOUBLE, g_nb_t_up, 101, g_cart_grid, &mstatus);
      if (status != MPI_SUCCESS) {
        fprintf(stderr, "[test_gsp_2pt] Error from MPI_Send\n");
        EXIT(169);
      }

      memcpy( gsp_t[0][0][0][0], gsp_xw_xw_f[0][0][0][0], bytes );
      status = MPI_Sendrecv(gsp_t[0][0][0][0], items, MPI_DOUBLE, g_nb_t_dn, 103, gsp_xw_xw_f[0][0][0][0], items, MPI_DOUBLE, g_nb_t_up, 103, g_cart_grid, &mstatus);
      if (status != MPI_SUCCESS) {
        fprintf(stderr, "[test_gsp_2pt] Error from MPI_Send\n");
        EXIT(170);
      }

      memcpy( gsp_t[0][0][0][0], gsp_v_w_f[0][0][0][0], bytes );
      status = MPI_Sendrecv(gsp_t[0][0][0][0], items, MPI_DOUBLE, g_nb_t_dn, 105, gsp_v_w_f[0][0][0][0],   items, MPI_DOUBLE, g_nb_t_up, 105, g_cart_grid, &mstatus);
      if (status != MPI_SUCCESS) {
        fprintf(stderr, "[test_gsp_2pt] Error from MPI_Send\n");
        EXIT(171);
      }

      memcpy( gsp_t[0][0][0][0], gsp_xv_xw_f[0][0][0][0], bytes );
      status = MPI_Sendrecv(gsp_t[0][0][0][0], items, MPI_DOUBLE, g_nb_t_dn, 107, gsp_xv_xw_f[0][0][0][0], items, MPI_DOUBLE, g_nb_t_up, 107, g_cart_grid, &mstatus);
      if (status != MPI_SUCCESS) {
        fprintf(stderr, "[test_gsp_2pt] Error from MPI_Send\n");
        EXIT(172);
      }

      MPI_Barrier(g_cart_grid);


      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_2pt] time for xchange = %e seconds\n", retime - ratime);
#endif


    }  /* end of for iproc */

    /***********************************************
     * I / O
     ***********************************************/
    ratime = _GET_TIME;
    for(iproc=0; iproc<g_nproc_t; iproc++) {

#ifdef HAVE_MPI
      /***********************************************
       * exchange correlator nad correlator2
       ***********************************************/
      items = 2 * T_global * T;
      if(iproc > 0) {
        if(io_proc == 2)  {
          mcoords[0] = iproc; mcoords[1] = 0; mcoords[2] = 0; mcoords[3] = 0;
          MPI_Cart_rank(g_cart_grid, mcoords, &mrank);
          fprintf(stdout, "# [test_gsp_2pt] proc%.2d receiving from proc%.2d\n", g_cart_id, mrank);
          /* receive correlator with tag 2*iproc */
          MPI_Recv(correlator[0],  items, MPI_DOUBLE, mrank, 2*iproc,   g_cart_grid, &mstatus);
          /* receive correlator2 with tag 2*iproc */
          MPI_Recv(correlator2[0], items, MPI_DOUBLE, mrank, 2*iproc+1, g_cart_grid, &mstatus);
        } else {
          if(g_proc_coords[0] == iproc && io_proc == 1 ) {
            mcoords[0] = 0; mcoords[1] = 0; mcoords[2] = 0; mcoords[3] = 0;
            MPI_Cart_rank(g_cart_grid, mcoords, &mrank);
            fprintf(stdout, "# [test_gsp_2pt] proc%.2d sending to proc%.2d\n", g_cart_id, mrank);
            /* send correlator with tag 2*iproc */
            MPI_Send(correlator[0],  items, MPI_DOUBLE, mrank, 2*iproc,   g_cart_grid);
            /* send correlator2 with tag 2*iproc+1 */
            MPI_Send(correlator2[0], items, MPI_DOUBLE, mrank, 2*iproc+1, g_cart_grid);
          }
        }
      }  /* end of if iproc > 0 */
#endif

#ifdef HAVE_MPI
      if(io_proc == 2) {
#endif
#ifdef HAVE_LHPC_AFF
        for(i_ti=0; i_ti<T; i_ti++) {

          /***********************************************
           * write correlator
           ***********************************************/
          x0 = (i_ti + ( g_proc_coords[0] + iproc ) * T ) % T_global;
          sprintf(aff_buffer_path, "/u-d/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d", 
              g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
              g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
              x0);
          /* if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_2pt] current aff path = %s\n", aff_buffer_path); */

          affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
          items = T_global;
          memcpy(aff_buffer, correlator[i_ti], 2*items*sizeof(double));
          status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)items);
          if(status != 0) {
            fprintf(stderr, "[test_gsp_2pt] Error from aff_node_put_double, status was %d\n", status);
            EXIT(178);
          }

          /***********************************************
           * write correlator2
           ***********************************************/
          x0 = (i_ti + ( g_proc_coords[0] + iproc ) * T ) % T_global;
          sprintf(aff_buffer_path, "/u-u/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d", 
              g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
              g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
              x0);
          /* if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_2pt] current aff path = %s\n", aff_buffer_path); */

          affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
          items = T_global;
          memcpy(aff_buffer, correlator2[i_ti], 2*items*sizeof(double));
          status = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)items);
          if(status != 0) {
            fprintf(stderr, "[test_gsp_2pt] Error from aff_node_put_double, status was %d\n", status);
            EXIT(178);
          }

        }  /* end of loop on i_ti */
#else
        /***********************************************
         * write correlator
         ***********************************************/
        for(i_ti=0; i_ti<T; i_ti++) {
          x0 = (i_ti + ( g_proc_coords[0] + iproc ) * T ) % T_global;
          fprintf(ofs, "# /u-d/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d\n", 
              g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
              g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
              x0);
          for(i_tf = 0; i_tf < T_global; i_tf++) {
            fprintf(ofs, "\t%25.16e%25.16e\n", correlator[i_ti][2*i_tf], correlator[i_ti][2*i_tf+1]);
          }
        }  /* end of loop on i_ti */

        /***********************************************
         * write correlator2
         ***********************************************/
        for(i_ti=0; i_ti<T; i_ti++) {
          x0 = (i_ti + ( g_proc_coords[0] + iproc ) * T ) % T_global;
          fprintf(ofs, "# /u-u/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d\n", 
              g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
              g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
              x0);
          for(i_tf = 0; i_tf < T_global; i_tf++) {
            fprintf(ofs, "\t%25.16e%25.16e\n", correlator2[i_ti][2*i_tf], correlator2[i_ti][2*i_tf+1]);
          }
        }  /* end of loop on i_ti */
#endif

#ifdef HAVE_MPI
      }  /* end of if io_proc == 2 */
#endif
    }  /* end of loop on iproc */

#if 0
#endif  /* of if 0 */

    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_2pt] time for writing = %e seconds\n", retime - ratime);

  }  /* end of loop on i2pt */

  /***********************************************
   * close output file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(g_cart_id == 0) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_gsp_2pt] Error from aff_writer_close, status was %s\n", aff_status_str);
      EXIT(176);
    }
  }
#else
  fclose(ofs);
#endif


  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  gsp_fini(&gsp_w_w_f);
  gsp_fini(&gsp_xw_xw_f);
  gsp_fini(&gsp_v_w_f);
  gsp_fini(&gsp_xv_xw_f);

  gsp_fini(&gsp_v_v_i);
  gsp_fini(&gsp_xv_xv_i);
  gsp_fini(&gsp_v_w_i);
  gsp_fini(&gsp_xv_xw_i);

#ifdef HAVE_MPI
  gsp_fini(&gsp_t);
#endif
 
  if(evecs_eval != NULL) free(evecs_eval);
  if(correlator != NULL) {
    if(correlator[0] != NULL) {
      free(correlator[0]);
    }
    free(correlator);
  }
 if(correlator2 != NULL) {
   if(correlator2[0] != NULL) {
     free(correlator2[0]);
   }
   free(correlator2);
  }

  free_geometry();

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_gsp_2pt] %s# [test_gsp_2pt] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_gsp_2pt] %s# [test_gsp_2pt] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

