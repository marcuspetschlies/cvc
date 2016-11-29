/****************************************************
 * test_momentum_projection.cpp
 * 
 * Tue Nov 29 12:29:37 CET 2016
 *
 * PURPOSE:
 * TODO:
 * DONE:
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef OPENMP
#include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "ranlxd.h"
#include "propagator_io.h"
#include "gauge_io.h"
#include "read_input_parser.h"
#include "matrix_init.h"
#include "project.h"

using namespace cvc;

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "test_momentum_projection";

  int c, i, j, k;
  int filename_set = 0;
  int exitstatus;
  int it, x0, x1, x2, x3, y0, y1, y2, y3;
  char filename[200];
  /* double ratime, retime; */
  unsigned int ix;
  unsigned int VOL3;
  spinor_propagator_type *connq=NULL;
  double ****connt = NULL;
  int gsx[4] = {1,2,3,4};


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
      fprintf(stdout, "# [test_momentum_projection] exit\n");
      break;
    }
  }

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[test_momentum_projection] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /******************************************************
   *
   ******************************************************/

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_momentum_projection] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  VOL3 = LX*LY*LZ;

  /***********************************************************
   * allocate memory for the contractions
   **********************************************************/
  connq = create_sp_field( (size_t)VOLUME );
  if(connq == NULL) {
    fprintf(stderr, "[test_momentum_projection] Error, could not alloc connq\n");
    EXIT(2);
  }

  /***********************************************************
   * initialize rng
   **********************************************************/
  sprintf(filename, "%s.rng", outfile_prefix);
  exitstatus = init_rng_stat_file (g_seed, filename);
  if(exitstatus != 0) {
    fprintf(stderr, "# [test_momentum_projection] Error from init_rng_stat_file, status was %d\n", exitstatus);
    EXIT(4);
  }

  /***********************************************************
   * fill connq
   ***********************************************************/
  k = VOLUME * 2 * g_sv_dim * g_sv_dim;
  ranlxd(connq[0][0], k);

  /* TEST */
  {
    for(i=0; i<g_sv_dim; i++) {
      for(j=0; j<g_sv_dim; j++) {
        for(k=0; k<g_nproc; k++) {
          FILE *ofs = NULL;
          sprintf(filename, "%s_x.%.2d_%.2d", outfile_prefix, i, j);
          ofs = k== 0 ? fopen(filename, "w") : fopen(filename, "a");
          if(ofs == NULL) {
            fprintf(stderr, "[test_momentum_projection] Error opening file %s\n", filename);
            EXIT(56);
          }

          if(k == g_cart_id) {
            if(g_cart_id == 0) fprintf(ofs, "connq <- array(dim=c(%d,%d,%d,%d))\n", T_global, LX_global, LY_global, LZ_global);
            for(x0=0; x0<T; x0++) {
              y0 = x0 + g_proc_coords[0]*T + 1;
            for(x1=0; x1<LX; x1++) {
              y1 = x1 + g_proc_coords[1]*LX + 1;
            for(x2=0; x2<LY; x2++) {
              y2 = x2 + g_proc_coords[2]*LY + 1;
            for(x3=0; x3<LZ; x3++) {
              y3 = x3 + g_proc_coords[3]*LZ + 1;
              ix = g_ipt[x0][x1][x2][x3];
              fprintf(ofs, "connq[%d,%d,%d,%d] <- %25.16e + %25.16e*1.i\n", y0, y1, y2, y3, connq[ix][i][2*j], connq[ix][i][2*j+1]);
            }}}}
          }
          fclose(ofs);
          MPI_Barrier(g_cart_grid);
        }
      }
    }
  }

  /***********************************************
   * momentum projections
   ***********************************************/
  init_4level_buffer(&connt, T, g_sink_momentum_number, g_sv_dim, 2*g_sv_dim);
  for(it=0; it<T; it++) {
    /* fprintf(stdout, "# [test_momentum_projection] proc%.4d momentum projection for t = %2d\n", g_cart_id, it); fflush(stdout); */
    exitstatus = momentum_projection2 (connq[it*VOL3][0], connt[it][0][0], g_sv_dim*g_sv_dim, g_sink_momentum_number, g_sink_momentum_list, &(gsx[1]) );
    /* exitstatus = momentum_projection2 (connq[it*VOL3][0], connt[it][0][0], g_sv_dim*g_sv_dim, g_sink_momentum_number, g_sink_momentum_list, NULL ); */
  }


  /* TEST */
  {
    sprintf(filename, "%s_tq.%.2d", outfile_prefix, g_cart_id);

    FILE *ofs = fopen(filename, "w");
    if(ofs == NULL) {
      fprintf(stderr, "[test_momentum_projection] Error opening file %s\n", filename);
      EXIT(56);
    }
    for(k=0; k<g_sink_momentum_number; k++) {
      for(it=0; it<T; it++) {
        fprintf(ofs, "# t = %2d p = (%d, %d, %d)\n", it+g_proc_coords[0]*T, g_sink_momentum_list[k][0], g_sink_momentum_list[k][1], g_sink_momentum_list[k][2]);
        for(i=0; i<g_sv_dim; i++) {
          for(j=0; j<g_sv_dim; j++) {
            fprintf(ofs, "%3d%3d%25.16e%25.16e\n", i, j, connt[it][k][i][2*j], connt[it][k][i][2*j+1]);
          }
        }
      }
    }
    fclose(ofs);
  }
  fini_4level_buffer(&connt);

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  free_geometry();
  free_sp_field(&connq);

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_momentum_projection] %s# [test_momentum_projection] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_momentum_projection] %s# [test_momentum_projection] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
