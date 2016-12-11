/****************************************************
 * test.cpp
 *
 * Fri Dec  9 17:34:16 CET 2016
 *
 * PURPOSE:
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
#include "ranlxd.h"

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
#include "io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "prepare_source.h"

using namespace cvc;

void usage() {
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
  
  int c, i, no_fields=0;
  int filename_set = 0;
  int have_source_flag = 0;
  int gsx[4];
  int x0, x1, x2, x3;
  /* int sx[4]; */
  int exitstatus;
  int source_proc_coords[4], source_proc_id = -1;
  unsigned int ix;
  char filename[100];
  /* double ratime, retime; */
  double plaq;
  double spinor1[24], spinor2[24], spinor3[24];
  double dtmp[2];
  /* double *phi=NULL, *chi=NULL; */
  /* complex w, w1; */
  FILE *ofs;

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
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1);
  if(exitstatus != 0) {
    exit(557);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    exit(558);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(exitstatus != 0) {
    exit(559);
  }
#endif

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[test] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  no_fields = 3;
  g_spinor_field = (double**)malloc(no_fields * sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&(g_spinor_field[i]), VOLUME);

  init_rng_stat_file (g_seed, NULL);

  ranlxd(g_spinor_field[0], _GSI(VOLUME));
  ranlxd(g_spinor_field[1], _GSI(VOLUME));

#if 0
  spinor_field_tm_rotation(g_spinor_field[1], g_spinor_field[0], +1, _TM_FERMION, VOLUME);
  spinor_field_tm_rotation(g_spinor_field[2], g_spinor_field[0], -1, _TM_FERMION, VOLUME);

  for(ix=0; ix<VOLUME; ix++) {
    fprintf(stdout, "# [] ix = %u\n", ix);

    for(i=0; i<12; i++) {
      fprintf(stdout, "%d %d %25.16e %25.16e \t %25.16e %25.16e \t %25.16e %25.16e\n", 
        i/3, i%3,
        g_spinor_field[0][_GSI(ix)+2*i], g_spinor_field[0][_GSI(ix)+2*i+1],
        g_spinor_field[1][_GSI(ix)+2*i], g_spinor_field[1][_GSI(ix)+2*i+1],
        g_spinor_field[2][_GSI(ix)+2*i], g_spinor_field[2][_GSI(ix)+2*i+1] );
    }
  }
#endif

  int iseq_mom = 0;
  memcpy(gsx, g_source_coords_list[0], 4*sizeof(int));
  fprintf(stdout, "# [test] source coords = (%d, %d, %d, %d)\n", gsx[0], gsx[1], gsx[2], gsx[3]);
  double *prop_list[2] = {g_spinor_field[0], g_spinor_field[1]};
  exitstatus = init_coherent_sequential_source(g_spinor_field[2], prop_list, gsx[0], g_coherent_source_number, g_seq_source_momentum_list[iseq_mom], 5);

  sprintf(filename, "coh.%d", g_cart_id);
  ofs = fopen(filename, "w");

  for(x0 = 0; x0 < T; x0++) {
  for(x1 = 0; x1 < LX; x1++) {
  for(x2 = 0; x2 < LY; x2++) {
  for(x3 = 0; x3 < LZ; x3++) {

    ix = g_ipt[x0][x1][x2][x3];
    fprintf(ofs, "# [] x %3d %3d %3d %3d \n", 
        x0+g_proc_coords[0]*T, x1+g_proc_coords[1]*LX, x2+g_proc_coords[2]*LY, x3+g_proc_coords[3]*LZ);

    for(i=0; i<12; i++) {
      fprintf(ofs, "%d %d %25.16e %25.16e \t %25.16e %25.16e \t %25.16e %25.16e\n", 
        i/3, i%3,
        g_spinor_field[0][_GSI(ix)+2*i], g_spinor_field[0][_GSI(ix)+2*i+1],
        g_spinor_field[1][_GSI(ix)+2*i], g_spinor_field[1][_GSI(ix)+2*i+1],
        g_spinor_field[2][_GSI(ix)+2*i], g_spinor_field[2][_GSI(ix)+2*i+1] );
    }
  }}}}

  fclose(ofs);

  /****************************************
   * free the allocated memory, finalize
   ****************************************/
  if(no_fields > 0 && g_spinor_field != NULL) { 
    for(i=0; i<no_fields; i++) free(g_spinor_field[i]);
    free(g_spinor_field);
  }

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif


#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test] %s# [test] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test] %s# [test] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
