/****************************************************
 * test_gsp_binary.cpp
 *
 * Do 30. Jun 21:48:24 CEST 2016
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
#ifdef HAVE_OPENMP
  int threadid, nthreads;
#endif

  int evecs_num=0;
  double *evecs_eval = NULL;

  complex w, w2;

  double *****gsp_w_w = NULL, *****gsp_v_w   = NULL, *****gsp_xw_xw= NULL, *****gsp_xv_xw = NULL;
  double *****gsp_v_v = NULL, *****gsp_xv_xv = NULL;
#ifdef HAVE_MPI
  double *****gsp_t=NULL;
  int mrank, mcoords[4];
  int io_proc = 0;
#endif
  char gsp_tag[100];

  int verbose = 0;
  char filename[200];

  double ratime, retime;

  FILE *ofs = NULL;
  size_t items, bytes;

#ifdef HAVE_MPI
  MPI_Status mstatus;
  MPI_Init(&argc, &argv);
#endif


  while ((c = getopt(argc, argv, "h?vf:n:")) != -1) {
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
      fprintf(stdout, "# [test_gsp_binary] dimension of eigenspace set to%d\n", evecs_num);
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
  if(g_cart_id==0) fprintf(stdout, "# [test_gsp_binary] Reading input from file %s\n", filename);
  read_input_parser(filename);


  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /* initialize geometry */
  if(init_geometry() != 0) {
    fprintf(stderr, "[test_gsp_binary] ERROR from init_geometry\n");
    EXIT(101);
  }

  geometry();

#if (defined HAVE_MPI) && ( (defined PARALLELTX) ||  (defined PARALLELTXY) ||  (defined PARALLELTXYZ) )
  fprintf(stderr, "[test_gsp_binary] Error, for now only 1-dimensional MPI-parallelization\n");
  EXIT(130);
#endif

#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_binary] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel private(nthreads,threadid)
{
  threadid = omp_get_thread_num();
  nthreads = omp_get_num_threads();
  fprintf(stdout, "# [test_gsp_binary] proc%.4d thread%.4d using %d threads\n", g_cart_id, threadid, nthreads);
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_gsp_binary] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if (evecs_num == 0) {
    if(g_cart_id==0) fprintf(stderr, "[test_gsp_binary] eigenspace dimension is 0\n");
    EXIT(1);
  }

  /***********************************************
   * allocate gsp's
   ***********************************************/
  /* allocate */
  ratime = _GET_TIME;
  status = 0;
  status = status ||  gsp_init (&gsp_w_w,   1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_xw_xw, 1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_v_v,   1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_xv_xv, 1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_v_w,   1, 1, T, evecs_num);
  status = status ||  gsp_init (&gsp_xv_xw, 1, 1, T, evecs_num);
#ifdef HAVE_MPI
  status = status ||  gsp_init (&gsp_t,   1, 1, T, evecs_num);
#endif
  if(status) {
    fprintf(stderr, "[test_gsp_binary] Error from gsp_init\n");
    EXIT(155);
  }
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_binary] time to initialize gsp = %e seconds\n", retime - ratime);


#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [test_gsp_binary] proc%.4d is io process\n", g_cart_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [test_gsp_binary] proc%.4d is send process\n", g_cart_id);
    } else {
      io_proc = 0;
    }
  }
#endif


  /***********************************************
   *
   *   gsp_w_w
   *   gsp_xw_xw
   *   gsp_v_w
   *   gsp_xv_xw
   *   gsp_v_v
   *   gsp_xv_xv
   ***********************************************/

  for(i2pt=0; i2pt < g_m_m_2pt_num; i2pt++) {

    i_gi = g_m_m_2pt_list[i2pt].gi;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_binary] %d\n", g_m_m_2pt_list[i2pt].gi);

    sprintf(filename, "test_gsp_binary.g%.2d.px%.2dpy%.2dpz%.2d.%.4d", i_gi, 
        g_m_m_2pt_list[i2pt].pi[0], g_m_m_2pt_list[i2pt].pi[1], g_m_m_2pt_list[i2pt].pi[2], g_cart_id);
    ofs = fopen(filename, "w");
    if(ofs == NULL) {
      EXIT(11);
    }

    ratime = _GET_TIME;

    sprintf(gsp_tag, "%s.%.4d", "gsp_v_v", Nconf);
    status = gsp_read_node (gsp_v_v[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_binary] Error from gsp_read_node\n");
      EXIT(156);
    }
    gsp_printf (gsp_v_v[0][0], evecs_num, "gsp_v_v", ofs);
#if 0
    sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeobarv", Nconf);
    status = gsp_read_node (gsp_xv_xv[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_binary] Error from gsp_read_node\n");
      EXIT(158);
    }
    gsp_printf (gsp_xv_xv[0][0], evecs_num, "gsp_xv_xv", ofs);

    sprintf(gsp_tag, "%s.%.4d", "gsp_v_w", Nconf);
    status = gsp_read_node (gsp_v_w[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_binary] Error from gsp_read_node\n");
      EXIT(160);
    }
    gsp_printf (gsp_v_w[0][0], evecs_num, "gsp_v_w", ofs);

    sprintf(gsp_tag, "%s.%.4d", "gsp_xeobarv_xeow", Nconf);
    status = gsp_read_node (gsp_xv_xw[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_binary] Error from gsp_read_node\n");
      EXIT(161);
    }
    gsp_printf (gsp_xv_xw[0][0], evecs_num, "gsp_xv_xw", ofs);

    sprintf(gsp_tag, "%s.%.4d", "gsp_w_w", Nconf);
    status = gsp_read_node (gsp_w_w[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_binary] Error from gsp_read_node\n");
      EXIT(140);
    }
    gsp_printf (gsp_w_w[0][0], evecs_num, "gsp_w_w", ofs);

    sprintf(gsp_tag, "%s.%.4d", "gsp_xeow_xeow", Nconf);
    status = gsp_read_node (gsp_xw_xw[0][0], evecs_num, g_m_m_2pt_list[i2pt].pi, i_gi, gsp_tag);
    if(status) {
      fprintf(stderr, "[test_gsp_binary] Error from gsp_read_node\n");
      EXIT(141);
    }
    gsp_printf (gsp_xw_xw[0][0], evecs_num, "gsp_xw_xw", ofs);
#endif  /* of if 0 */
    retime = _GET_TIME;
    if(g_cart_id == 0) fprintf(stdout, "# [test_gsp_binary] time to read gsp nodes gsp = %e seconds\n", retime - ratime);

    fclose(ofs);

  }  /* end of loop on i2pt */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/

  gsp_fini(&gsp_w_w);
  gsp_fini(&gsp_xw_xw);
  gsp_fini(&gsp_v_w);
  gsp_fini(&gsp_xv_xw);
  gsp_fini(&gsp_v_v);
  gsp_fini(&gsp_xv_xv);

#ifdef HAVE_MPI
  gsp_fini(&gsp_t);
#endif
 
  if(evecs_eval != NULL) free(evecs_eval);
  free_geometry();

  if (g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_gsp_binary] %s# [test_gsp_binary] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_gsp_binary] %s# [test_gsp_binary] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  return(0);
}

