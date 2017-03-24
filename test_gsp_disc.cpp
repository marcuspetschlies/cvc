/****************************************************
 * test_gsp_disc.cpp
 *
 * Mi 22. Jun 12:51:35 CEST 2016
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
  int i, i2pt, io_proc;
  int i_gi, i_gf, i_si, i_sf, i_ti, i_tf, i_tfi, x0;
  int filename_set = 0;
#ifdef HAVE_OPENMP
  int threadid, nthreads;
#endif

  int evecs_num=0, evecs_num_contract=0;
  double *evecs_eval = NULL;

  complex w, w2;

  double *****gsp_v_w_f = NULL, *****gsp_xv_xw_f = NULL, *****gsp_v_w_i = NULL, *****gsp_xv_xw_i = NULL;
#ifdef HAVE_MPI
  double *****gsp_t=NULL;
  int mrank, mcoords[4];
#endif
  double *correlator=NULL, *correlator2=NULL;
  double *buffer=NULL;
  double *buffer_re_re=NULL, *buffer_re_im=NULL, *buffer_im_re=NULL, *buffer_im_im=NULL;
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
      fprintf(stdout, "# [test_gsp_disc] dimension of eigenspace set to%d\n", evecs_num);
      break;
    case 'c':
      evecs_num_contract = atoi(optarg);
      fprintf(stdout, "# [test_gsp_disc] use first %d eigenvectors for contraction\n", evecs_num_contract);
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
  if(g_cart_id==0) fprintf(stdout, "# [test_gsp_disc] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test_gsp_disc] calling tmLQCD wrapper init functions\n");

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
    fprintf(stderr, "[test_gsp_disc] Error from init_geometry\n");
    EXIT(4);
  }

  geometry();

#if (defined HAVE_MPI) && ( (defined PARALLELTX) ||  (defined PARALLELTXY) ||  (defined PARALLELTXYZ) )
  fprintf(stderr, "[test_gsp_disc] Error, for now only 1-dimensional MPI-parallelization\n");
  EXIT(5);
#endif

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private(nthreads,threadid)
{
  threadid = omp_get_thread_num();
  nthreads = omp_get_num_threads();
  fprintf(stdout, "# [test_gsp_disc] proc%.4d thread%.4d using %d threads\n", g_cart_id, threadid, nthreads);
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_gsp_disc] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if (evecs_num == 0) {
    if(g_cart_id==0) fprintf(stderr, "[test_gsp_disc] eigenspace dimension is 0\n");
    EXIT(6);
  }

  /***********************************************
   * read eigenvalues
   ***********************************************/
  sprintf(gsp_tag, "%s.%.4d", "gsp_eval", Nconf);
  status = gsp_read_eval(&evecs_eval, evecs_num, gsp_tag);
  if( status != 0) {
    fprintf(stderr, "[test_gsp_disc] Error from gsp_read_eval, status was %d\n", status);
    EXIT(7);
  }

  /***********************************************
   * inverse square root of diagonal elements
   ***********************************************/
  for(i=0; i<evecs_num; i++) {
    if(g_cart_id==0) fprintf(stdout, "# [test_gsp_disc] eval %4d %25.16e\n", i, evecs_eval[i]) ;
    evecs_eval[i] = 2.*g_kappa / sqrt(evecs_eval[i]);
  }

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
#ifdef HAVE_MPI
  status = status ||  gsp_init (&gsp_t,   1, 1, T, evecs_num);
#endif
  if(status) {
    fprintf(stderr, "[test_gsp_disc] Error from gsp_init\n");
    EXIT(8);
  }
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc] time to initialize gsp = %e seconds\n", retime - ratime);

  /***********************************************
   * correlator
   ***********************************************/
  correlator = (double*)malloc(2*T_global*sizeof(double));
  if(correlator == NULL) {
    fprintf(stderr, "[test_gsp_disc] Error from malloc\n");
    EXIT(9);
  }
  correlator2 = (double*)malloc(2*T_global*sizeof(double));
  if(correlator2 == NULL) {
    fprintf(stderr, "[test_gsp_disc] Error from malloc\n");
    EXIT(10);
  }
 

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [test_gsp_disc] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [test_gsp_disc] proc%.4d is send process\n", g_cart_id);
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
    fprintf(stdout, "# [test_gsp_disc] using aff version %s\n", aff_status_str);

    sprintf(filename, "%s.%.4d.aff", "gsp_correlator_disc", Nconf);
    fprintf(stdout, "# [test_gsp_disc] writing gsp data to file %s\n", filename);
    affw = aff_writer(filename);
    aff_status_str = (char*)aff_writer_errstr(affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_gsp_disc] Error from aff_writer, status was %s\n", aff_status_str);
      EXIT(11);
    }

    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[test_gsp_disc] Error, aff writer is not initialized\n");
      EXIT(12);
    }

    aff_buffer = (double _Complex*)malloc(T_global*sizeof(double _Complex));
    if(aff_buffer == NULL) {
      fprintf(stderr, "[test_gsp_disc] Error from malloc\n");
      EXIT(13);
    }
#else
    sprintf(filename, "%s.%.4d.%.4d", "gsp_correlator_disc", Nconf, evecs_num_contract);
    ofs = fopen(filename, "w");
    if(ofs == NULL) {
      fprintf(stderr, "[test_gsp_disc] Error, could not open file %s for writing\n", filename);
      EXIT(14);
    }
#endif
  }  /* of if io_proc == 0  */


  /***********************************************
   * allocate buffer
   ***********************************************/
  buffer = (double*)malloc(2*T_global*sizeof(double));
  if( buffer == NULL ) {
    fprintf(stderr, "[test_gsp_disc] Error from malloc\n");
    EXIT(15);
  }
  if(io_proc == 2) {
    buffer_re_re = (double*)malloc(T_global*sizeof(double));
    if( buffer_re_re == NULL ) {
      fprintf(stderr, "[test_gsp_disc] Error from malloc\n");
      EXIT(24);
    }

    buffer_re_im = (double*)malloc(T_global*sizeof(double));
    if( buffer_re_im == NULL ) {
      fprintf(stderr, "[test_gsp_disc] Error from malloc\n");
      EXIT(25);
    }

    buffer_im_re = (double*)malloc(T_global*sizeof(double));
    if( buffer_im_re == NULL ) {
      fprintf(stderr, "[test_gsp_disc] Error from malloc\n");
      EXIT(24);
    }

    buffer_im_im = (double*)malloc(T_global*sizeof(double));
    if( buffer_im_im == NULL ) {
      fprintf(stderr, "[test_gsp_disc] Error from malloc\n");
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

    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc] (%d, %d) to (%d, %d); (%d, %d)\n", g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].gf, i_gi, i_si, i_gf, i_sf);

    ratime = _GET_TIME;

    /* at source */
    sprintf(gsp_tag, "%s.%.4d", "gsp_v_w", Nconf);
    status = gsp_read_node (gsp_v_w_i[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_disc] Error from gsp_read_node\n");
      EXIT(16);
    }
    /* TEST */
    /* gsp_printf (gsp_v_w_i[0][0], evecs_num, "gsp_v_w_i", stdout); */

    sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeow", Nconf);
    status = gsp_read_node (gsp_xv_xw_i[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_disc] Error from gsp_read_node\n");
      EXIT(17);
    }
    /* TEST */
    /* gsp_printf (gsp_xv_xw_i[0][0], evecs_num, "gsp_xv_xw_i", stdout); */

    /* at sink */
    sprintf(gsp_tag, "%s.%.4d", "gsp_v_w", Nconf);
    status = gsp_read_node (gsp_v_w_f[0][0], evecs_num, g_m_m_2pt_list[i2pt].pf, i_gf, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_disc] Error from gsp_read_node\n");
      EXIT(18);
    }
    /* TEST */
    /* gsp_printf (gsp_v_w_f[0][0], evecs_num, "gsp_v_w_f", stdout); */

    sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeow", Nconf);
    status = gsp_read_node (gsp_xv_xw_f[0][0], evecs_num, g_m_m_2pt_list[i2pt].pf, i_gf, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_disc] Error from gsp_read_node\n");
      EXIT(19);
    }
    /* TEST */
    /* gsp_printf (gsp_xv_xw_f[0][0], evecs_num, "gsp_xv_xw_f", stdout); */

    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc] time to read gsp nodes gsp = %e seconds\n", retime - ratime);


    /* reduce (trace) matrix product 
     *   for all combinations of source and sink time
     */
    memset(correlator,  0, 2*T_global*sizeof(double));
    memset(correlator2, 0, 2*T_global*sizeof(double));

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
  
      correlator[2*x0  ] = w.re;
      correlator[2*x0+1] = w.im;
      /* TEST */
      /* fprintf(stdout, "# [test_gsp_disc] correlator[%2d] = %25.16e %25.16e\n", x0, correlator[2*x0  ], correlator[2*x0+1]); */

    }  /* end of loop on tf */
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc] time for souce loop reduction = %e seconds\n", retime - ratime);

    for(i_tf = 0; i_tf<T; i_tf++) {
      x0 = i_tf + g_proc_coords[0] * T;

      _co_eq_zero(&w);
      /* accumulate four trace terms */
      co_eq_tr_gsp (&w2, gsp_xv_xw_f[0][0][i_tf], evecs_eval, evecs_num_contract);
      _co_pl_eq_co(&w, &w2);

      co_eq_tr_gsp (&w2, gsp_v_w_f[0][0][i_tf],   evecs_eval, evecs_num_contract);
      _co_pl_eq_co(&w, &w2);

      _co_ti_eq_re(&w, i_sf);
  
      correlator2[2*x0  ] = w.re;
      correlator2[2*x0+1] = w.im;

      /* TEST */
      /* fprintf(stdout, "# [test_gsp_disc] correlator2[%2d] = %25.16e %25.16e\n", x0, correlator2[2*x0  ], correlator2[2*x0+1]); */
    }    /* end of loop on tf */
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc] time for sink loop reduction = %e seconds\n", retime - ratime);

#ifdef HAVE_MPI
    ratime = _GET_TIME;
    items = 2 * (size_t)T_global;
    bytes = items * sizeof(double);

    memcpy(buffer, correlator, bytes);
    status = MPI_Allreduce(buffer, correlator, (2*T_global), MPI_DOUBLE, MPI_SUM, g_cart_grid);
    if (status != MPI_SUCCESS) {
      fprintf(stderr, "[test_gsp_disc] Error from MPI_Allreduce\n");
      EXIT(20);
    }

    memcpy(buffer, correlator2, bytes);
    status = MPI_Allreduce(buffer, correlator2, (2*T_global), MPI_DOUBLE, MPI_SUM, g_cart_grid);
    if (status != MPI_SUCCESS) {
      fprintf(stderr, "[test_gsp_disc] Error from MPI_Allreduce\n");
      EXIT(21);
    }

    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc] time for xchange = %e seconds\n", retime - ratime);
#endif

    /***********************************************
     * I / O
     ***********************************************/
    ratime = _GET_TIME;
#ifdef HAVE_MPI
    if(io_proc == 2) {
#endif

      /* real source - real sink */
      for(i_ti=0; i_ti<T_global; i_ti++) {
        memset(buffer, 0, 2*T_global*sizeof(double));
        for(i_tf = 0; i_tf < T_global; i_tf++) {
          i_tfi = ( i_tf - i_ti + T_global ) % T_global;
          /* _co_eq_co_ti_co( (complex*)(&buffer[2*i_tfi]), (complex*)(&correlator[2*i_ti]), (complex*)(&correlator2[2*i_tf])); */
          buffer_re_re[i_tfi] = correlator[2*i_ti  ] * correlator2[2*i_tf  ];
          buffer_re_im[i_tfi] = correlator[2*i_ti  ] * correlator2[2*i_tf+1];
          buffer_im_re[i_tfi] = correlator[2*i_ti+1] * correlator2[2*i_tf  ];
          buffer_im_im[i_tfi] = correlator[2*i_ti+1] * correlator2[2*i_tf+1];
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
        /* if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc] current aff path = %s\n", aff_buffer_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
        status = aff_node_put_double (affw, affdir, buffer_re_re, (uint32_t)items);
        if(status != 0) {
          fprintf(stderr, "[test_gsp_disc] Error from aff_node_put_double, status was %d\n", status);
          EXIT(26);
        }

        sprintf(aff_buffer_path, "/re-im/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        /* if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc] current aff path = %s\n", aff_buffer_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
        status = aff_node_put_double (affw, affdir, buffer_re_im, (uint32_t)items);
        if(status != 0) {
          fprintf(stderr, "[test_gsp_disc] Error from aff_node_put_double, status was %d\n", status);
          EXIT(27);
        }

        sprintf(aff_buffer_path, "/im-re/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        /* if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc] current aff path = %s\n", aff_buffer_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
        status = aff_node_put_double (affw, affdir, buffer_im_re, (uint32_t)items);
        if(status != 0) {
          fprintf(stderr, "[test_gsp_disc] Error from aff_node_put_double, status was %d\n", status);
          EXIT(28);
        }

        sprintf(aff_buffer_path, "/im-im/gi%.2d/pix%.2dpiy%.2dpiz%.2d/gf%.2d/pfx%.2dpfy%.2dpfz%.2d/ti%.2d", 
            g_m_m_2pt_list[i2pt].gi, g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2],
            g_m_m_2pt_list[i2pt].gf, g_m_m_2pt_list[i2pt].pf[0], g_m_m_2pt_list[i2pt].pf[1], g_m_m_2pt_list[i2pt].pf[2],
            x0);
        /* if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc] current aff path = %s\n", aff_buffer_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
        status = aff_node_put_double (affw, affdir, buffer_im_im, (uint32_t)items);
        if(status != 0) {
          fprintf(stderr, "[test_gsp_disc] Error from aff_node_put_double, status was %d\n", status);
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

#ifdef HAVE_MPI
    }  /* end of if io_proc == 2 */
#endif

    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_disc] time for writing = %e seconds\n", retime - ratime);

  }  /* end of loop on i2pt */

  /***********************************************
   * close output file
   ***********************************************/
#ifdef HAVE_LHPC_AFF
  if(io_proc == 2 ) {
    aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[test_gsp_disc] Error from aff_writer_close, status was %s\n", aff_status_str);
      EXIT(23);
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

#ifdef HAVE_MPI
  gsp_fini(&gsp_t);
#endif
 
  if(evecs_eval != NULL) free(evecs_eval);
  if(correlator != NULL) {
    free(correlator);
  }
 if(correlator2 != NULL) {
   free(correlator2);
  }
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
    fprintf(stdout, "# [test_gsp_disc] %s# [test_gsp_disc] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_gsp_disc] %s# [test_gsp_disc] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

