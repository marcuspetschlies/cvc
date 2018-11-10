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
#include "complex.h"
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>
#include "ranlxd.h"

#if 0
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
#endif

#define MAIN_PROGRAM

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
#include "prepare_propagator.h"
#include "table_init_d.h"
#include "table_init_z.h"
#include "table_init_asym_z.h"
#include "rotations.h"

using namespace cvc;

void usage() {
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}


int main(int argc, char **argv) {
 
#if 0
  const int n_s = 4;
  const int n_c = 3;
#endif

  int c;
  int filename_set = 0;
  char filename[100];
#if 0
  int i, k, no_fields=0;
  /* int have_source_flag = 0; */
  // int gsx[4];
  int x0, x1, x2, x3;
  /* int sx[4]; */
  int exitstatus;
  /* int source_proc_coords[4], source_proc_id = -1; */
  unsigned int ix;
  /* double ratime, retime; */
  // double plaq;
  FILE *ofs;

  // in order to be able to initialise QMP if QPhiX is used, we need
  // to allow tmLQCD to intialise MPI via QMP
  // the input file has to be read 
#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_init_parallel_and_read_input(argc, argv, 1, "invert.input");
#else
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
#endif
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

  // g_the_time = time(NULL);


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /***********************************************************
   * set number of openmp threads
   ***********************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test] setting omp number of threads to %d\n", g_num_threads);
    omp_set_num_threads(g_num_threads);
#pragma omp parallel
  {
    fprintf(stdout, "# [test] thread%.4d using %d threads\n", omp_get_thread_num(), omp_get_num_threads());
  }
#else
  if(g_cart_id == 0) fprintf(stdout, "[hvp_lma_recombine] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  int const N = 97;

  double _Complex **A = init_2level_ztable ( (size_t)N, (size_t)N);
  double _Complex **B = init_2level_ztable ( (size_t)N, (size_t)N);
  double _Complex **C = init_2level_ztable ( (size_t)N, (size_t)N);
  double _Complex **D = init_2level_ztable ( (size_t)N, (size_t)N);

  ranlxd( (double*)(A[0]), 2*N*N );
  ranlxd( (double*)(B[0]), 2*N*N );

#define MAT_OP rot_mat_ti_mat_adj

  MAT_OP ( D, A, B, N );

  memcpy ( C[0], A[0], N*N*sizeof(double _Complex ) );
  MAT_OP ( C, C, B, N );
  fprintf ( stdout, "# [test] (1) |C-D| = %25.16e %25.16e\n", rot_mat_diff_norm ( C, D, N ), sqrt( rot_mat_norm2 (C, N ) ) );

  memcpy ( C[0], B[0], N*N*sizeof(double _Complex ) );
  MAT_OP  ( C, A, C, N );
  fprintf ( stdout, "# [test] (2) |C-D| = %25.16e %25.16e\n", rot_mat_diff_norm ( C, D, N ), sqrt( rot_mat_norm2 (C, N ) ) );

  MAT_OP( D, A, A, N );
  memcpy ( C[0], A[0], N*N*sizeof(double _Complex ) );
  MAT_OP  ( C, C, C, N );
  fprintf ( stdout, "# [test] (3) |C-D| = %25.16e %25.16e\n", rot_mat_diff_norm ( C, D, N ), sqrt( rot_mat_norm2 (C, N ) ) );

  fini_2level_ztable ( &A );
  fini_2level_ztable ( &B );
  fini_2level_ztable ( &C );
  fini_2level_ztable ( &D );


#if 0
  /******************************************************************
   * TEST co_eq_trace_mat_ti_mat_weight_re 
   ******************************************************************/
  int const N = 97;
  rlxd_init( 2, 12342343 );
  double _Complex **A = init_2level_ztable ( (size_t)N, (size_t)N);
  double _Complex **B = init_2level_ztable ( (size_t)N, (size_t)N);
  double * w  = init_1level_dtable ( (size_t)N );
  double * w2  = init_1level_dtable ( (size_t)N );

  ranlxd( (double*)(A[0]), 2*N*N );
  ranlxd( (double*)(B[0]), 2*N*N );
  ranlxd( w, N );
  ranlxd( w2, N );

  // double _Complex ztmp = co_eq_trace_mat_ti_mat_weight_re ( A, B, w, N );
  //
  double _Complex ztmp = co_eq_trace_mat_ti_weight_ti_mat_ti_weight_re ( A, w, B, w2, N );


  double _Complex ztmp2 = 0.;
  for ( int i = 0; i < N; i++ ) {
  for ( int k = 0; k < N; k++ ) {
    ztmp2 += A[i][k] * w[k] * B[k][i] * w2[i];
  }}
  double dtmp = cabs ( ztmp - ztmp2 );
  fprintf ( stdout, "# [test] z %25.16e %25.16e   z2 %25.16e  %25.16e  diff %25.16e\n", creal( ztmp ), cimag( ztmp ), creal( ztmp2 ), cimag( ztmp2 ), dtmp ); 

#if 0


  // double _Complex ztmp = rot_mat_trace ( A, N );
  double _Complex ztmp = rot_mat_trace_weight_re ( A, w, N );


  double _Complex ztmp2 = 0.;
  for ( int i = 0; i < N; i++ ) {
    // ztmp2 += A[i][i];
    ztmp2 += A[i][i] * w[i];
  }
  double dtmp = cabs ( ztmp - ztmp2 );
  fprintf ( stdout, "# [test] z %25.16e %25.16e   z2 %25.16e  %25.16e  diff %25.16e\n", creal( ztmp ), cimag( ztmp ), creal( ztmp2 ), cimag( ztmp2 ), dtmp ); 
#endif

  fini_2level_ztable ( &A );
  fini_2level_ztable ( &B );
  fini_1level_dtable ( &w );
  fini_1level_dtable ( &w2 );
#endif

#if 0
  /******************************************************************
   * TEST init_4level_ztable_asym and fini_4level_ztable_asym
   ******************************************************************/
  int const N0 =  7;
  int const N1 =  5;
  int const N2 =  3;
  int const dim[3] = {1, 2, 4};
  double _Complex **** a = init_4level_ztable_asym ( N0, N1, N2, (int *const )dim );
  for ( int i = 0; i < N0*N1* ( dim[0] + dim[1] + dim[2]); i++ )  {
    a[0][0][0][i] = 1./(i+1) + 1./(i+2.)*I;
  }

  int count = 0;
  for ( int i0 = 0; i0 < N0; i0++ ) {
  for ( int i1 = 0; i1 < N1; i1++ ) {
  for ( int i2 = 0; i2 < N2; i2++ ) {
    for ( int i3 = 0; i3 < dim[i2]; i3++ ) {
      fprintf ( stdout, " %2d %2d %2d %2d %6d   %25.16e %25.16e   %25.16e %25.16e\n", i0, i1, i2, i3, count,
          creal( a[i0][i1][i2][i3] ), cimag( a[i0][i1][i2][i3]),
          creal( a[0][0][0][count] ), cimag( a[0][0][0][count] ) );

      count++;
  }}}}
  fini_4level_ztable_asym ( &a );
#endif  // of if 0
 

#if 0

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [test] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1, 0);
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

#if 0
  no_fields = 5 * n_s*n_c;
  g_spinor_field = (double**)malloc(no_fields * sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&(g_spinor_field[i]), VOLUME);

  init_rng_stat_file (g_seed, NULL);

  for(i=0; i<3*n_s*n_c; i++) {
    ranlxd(g_spinor_field[i], _GSI(VOLUME));
  }

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

#if 0
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
#endif



  /* prepare_seqn_stochastic_vertex_propagator_sliced3d */
#if 0
  int gseq2 = 5;
  int g_nsample = n_s*n_c;
  double *pffii_list[n_s*n_c];
  double *stochastic_propagator_list[n_s*n_c], *stochastic_source_list[n_s*n_c];
  double *sequential_propagator_list[n_s*n_c];
  double *spinor_test[n_s*n_c];

  g_seq2_source_momentum_list[0][0] =  1;
  g_seq2_source_momentum_list[0][1] = -2;
  g_seq2_source_momentum_list[0][2] =  3;

  memcpy(gsx, g_source_coords_list[0], 4*sizeof(int));

  for(i=0; i<n_s*n_c; i++) {
    sequential_propagator_list[i] = g_spinor_field[            i];
    stochastic_propagator_list[i] = g_spinor_field[  n_s*n_c + i];
    stochastic_source_list[i]     = g_spinor_field[2*n_s*n_c + i];
    pffii_list[i]                 = g_spinor_field[3*n_s*n_c + i];
    spinor_test[i]                = g_spinor_field[4*n_s*n_c + i];
  }

  exitstatus = prepare_seqn_stochastic_vertex_propagator_sliced3d (pffii_list, stochastic_propagator_list, stochastic_source_list,
                           sequential_propagator_list, g_nsample, n_s*n_c, g_seq2_source_momentum_list[0], gseq2);


  /* calculation by hand */
  for(x0 = 0; x0 < T; x0++) {
    double **p = NULL;
    double sp1[24], sp2[24];
    init_2level_buffer(&p, n_s*n_c, 2*g_nsample);
    for(x1 = 0; x1 < LX; x1++) {
    for(x2 = 0; x2 < LY; x2++) {
    for(x3 = 0; x3 < LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      /* apply vertex, reduce */
      double phase = 2.*M_PI * ( 
            g_seq2_source_momentum_list[0][0] * (x1 + g_proc_coords[1]*LX)/(double)LX_global
          + g_seq2_source_momentum_list[0][1] * (x2 + g_proc_coords[2]*LY)/(double)LY_global
          + g_seq2_source_momentum_list[0][2] * (x3 + g_proc_coords[3]*LZ)/(double)LZ_global );
      complex w = {cos(phase), sin(phase)};
      for(i=0; i<n_s*n_c; i++) {
        for(k=0; k<g_nsample; k++) {
          _fv_eq_gamma_ti_fv(sp1, gseq2, sequential_propagator_list[i]+_GSI(ix));
          _fv_eq_fv_ti_co(sp2, sp1, &w);
          _co_pl_eq_fv_dag_ti_fv((complex*)(p[i]+2*k), stochastic_source_list[k]+_GSI(ix), sp2);
        }
      }

    }}}
#ifdef HAVE_MPI
    double *buffer = (double*)malloc(2*n_s*n_c*g_nsample*sizeof(double));
    memcpy(buffer, p[0], 2*n_s*n_c*g_nsample*sizeof(double));
    if( (exitstatus = MPI_Allreduce(buffer, p[0], 2*n_s*n_c*g_nsample, MPI_DOUBLE, MPI_SUM, g_ts_comm) ) != MPI_SUCCESS ) {
      fprintf(stderr, "[] Error from MPI_Allreduce, status was %d\n", exitstatus);
      EXIT(1);
    }
    free(buffer);
#endif

/*
    for(i=0; i<n_s*n_c; i++) {
      for(k=0; k<g_nsample; k++) {
        fprintf(stdout, "p proc%.2d t%.2d %2d %2d %25.16e %25.16e\n", g_cart_id, x0, i, k, p[i][2*k], p[i][2*k+1]);
      }
    }
*/

    /* expand */
    for(x1 = 0; x1 < LX; x1++) {
    for(x2 = 0; x2 < LY; x2++) {
    for(x3 = 0; x3 < LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      for(i=0; i<n_s*n_c; i++) {
        _fv_eq_zero( spinor_test[i]+_GSI(ix) );
        for(k=0; k<g_nsample; k++) {
          _fv_eq_fv_ti_co(sp1, stochastic_propagator_list[k]+_GSI(ix), (complex*)(p[i]+2*k) );
          _fv_pl_eq_fv(spinor_test[i]+_GSI(ix), sp1);
        }
      }
    }}}
    fini_2level_buffer(&p);

  }


  sprintf(filename, "vertex.%d", g_cart_id);
  ofs = fopen(filename, "w");

  for(x0 = 0; x0 < T; x0++) {
  for(x1 = 0; x1 < LX; x1++) {
  for(x2 = 0; x2 < LY; x2++) {
  for(x3 = 0; x3 < LZ; x3++) {

    ix = g_ipt[x0][x1][x2][x3];
    fprintf(ofs, "# [] x %3d %3d %3d %3d \n", 
        x0+g_proc_coords[0]*T, x1+g_proc_coords[1]*LX, x2+g_proc_coords[2]*LY, x3+g_proc_coords[3]*LZ);

    for(i=0; i<n_s*n_c; i++) {
      for(k=0; k<n_s*n_c; k++) {
        fprintf(ofs, "%2d %d %d %25.16e %25.16e \t %25.16e %25.16e\n", 
          i, k/n_c, k%n_c,
          pffii_list[i][ _GSI(ix)+2*k], pffii_list[i][ _GSI(ix)+2*k+1],
          spinor_test[i][_GSI(ix)+2*k], spinor_test[i][_GSI(ix)+2*k+1] );
      }
    }
  }}}}

  fclose(ofs);
#endif

  no_fields = 3 * n_s*n_c + 8;
  g_spinor_field = (double**)malloc(no_fields * sizeof(double*));
  for(i=0; i<no_fields; i++) alloc_spinor_field(&(g_spinor_field[i]), VOLUME);

  init_rng_stat_file (g_seed, NULL);

  int gseq = 5, gseq2 = 5;
  int g_nsample = n_s*n_c;
  double *pfifi_list[n_s*n_c];
  double *stochastic_propagator_list[8];
  double *propagator_list_up[n_s*n_c];
  double *spinor_test[n_s*n_c];

  g_seq2_source_momentum_list[0][0] =  1;
  g_seq2_source_momentum_list[0][1] = -2;
  g_seq2_source_momentum_list[0][2] =  3;

  for(i=0; i<n_s*n_c; i++) {
    propagator_list_up[i]         = g_spinor_field[            i];
    pfifi_list[i]                 = g_spinor_field[  n_s*n_c + i];
    spinor_test[i]                = g_spinor_field[2*n_s*n_c + i];
  }
  for(i=0; i<8; i++) {
    stochastic_propagator_list[i] = g_spinor_field[3*n_s*n_c + i];
  }

  for(i=0; i<n_s*n_c; i++) {
    ranlxd(propagator_list_up[i], _GSI(VOLUME));
  }

  for(i=0; i<8; i++) {
    ranlxd(stochastic_propagator_list[i], _GSI(VOLUME));
  }

  exitstatus = prepare_seqn_stochastic_vertex_propagator_sliced3d_oet (pfifi_list, stochastic_propagator_list, &(stochastic_propagator_list[4]),
                   propagator_list_up, g_seq2_source_momentum_list[0], gseq2, gseq);
  if( exitstatus != 0 ) {
    fprintf(stderr, "[] Error from prepare_seqn_stochastic_vertex_propagator_sliced3d_oet, status was %d\n", exitstatus);
    EXIT(45);
  }

  /* calculation by hand */
  for(x0 = 0; x0 < T; x0++) {
    double **p = NULL;
    double sp1[24], sp2[24];
    init_2level_buffer(&p, n_s*n_c, 8);
    for(x1 = 0; x1 < LX; x1++) {
    for(x2 = 0; x2 < LY; x2++) {
    for(x3 = 0; x3 < LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      /* apply vertex, reduce */
      double phase = 2.*M_PI * ( 
            g_seq2_source_momentum_list[0][0] * (x1 + g_proc_coords[1]*LX)/(double)LX_global
          + g_seq2_source_momentum_list[0][1] * (x2 + g_proc_coords[2]*LY)/(double)LY_global
          + g_seq2_source_momentum_list[0][2] * (x3 + g_proc_coords[3]*LZ)/(double)LZ_global );
      complex w = {cos(phase), sin(phase)};
      for(i=0; i<n_s*n_c; i++) {
        for(k=0; k<4; k++) {
          _fv_eq_gamma_ti_fv(sp1, gseq2, propagator_list_up[i]+_GSI(ix));
          _fv_eq_gamma_ti_fv(sp2, 5, sp1);
          _fv_eq_fv_ti_co(sp1, sp2, &w);
          _co_pl_eq_fv_dag_ti_fv((complex*)(p[i]+2*k), stochastic_propagator_list[k]+_GSI(ix), sp1);
        }
      }
    }}}
#ifdef HAVE_MPI
    double *buffer = (double*)malloc(2*n_s*n_c*4*sizeof(double));
    memcpy(buffer, p[0], 2*n_s*n_c*4*sizeof(double));
    if( (exitstatus = MPI_Allreduce(buffer, p[0], 2*n_s*n_c*4, MPI_DOUBLE, MPI_SUM, g_ts_comm) ) != MPI_SUCCESS ) {
      fprintf(stderr, "[] Error from MPI_Allreduce, status was %d\n", exitstatus);
      EXIT(1);
    }
    free(buffer);
#endif


    double **paux = NULL;
    init_2level_buffer(&paux, n_s*n_c, 8);

    memcpy(paux[0], p[0], 2*n_s*n_c*4*sizeof(double));
    int isimag = gamma_permutation[5][0]%2;
    for(i=0; i<n_s*n_c; i++) {
      for(k=0; k<4; k++) {
        p[i][2*k  ] = paux[i][2*(gamma_permutation[5][6*k]/6) +   isimag] * gamma_sign[5][6*k  ];
        // p[i][2*k+1] = paux[i][2*(gamma_permutation[5][6*k]/6) + 1-isimag] * gamma_sign[5][6*k+1];
        p[i][2*k+1] = (isimag ? -1 : 1) * paux[i][2*(gamma_permutation[5][6*k]/6) + 1-isimag] * gamma_sign[5][6*k];
      }
    }

    memcpy(paux[0], p[0], 2*n_s*n_c*4*sizeof(double));
    isimag = gamma_permutation[gseq][0]%2;
    for(i=0; i<n_s*n_c; i++) {
      for(k=0; k<4; k++) {
        p[i][2*k  ] = paux[i][2*(gamma_permutation[gseq][6*k]/6) +   isimag] * gamma_sign[gseq][6*k  ];
        // p[i][2*k+1] = paux[i][2*(gamma_permutation[gseq][6*k]/6) + 1-isimag] * gamma_sign[gseq][6*k+1];
        p[i][2*k+1] = (isimag ? -1 : 1) * paux[i][2*(gamma_permutation[gseq][6*k]/6) + 1-isimag] * gamma_sign[gseq][6*k];
      }
    }

    fini_2level_buffer(&paux);

    for(i=0; i<n_s*n_c; i++) {
      for(k=0; k<4; k++) {
        fprintf(stdout, "p  proc%.2d t%.2d %2d %2d %25.16e %25.16e\n", g_cart_id, x0, i, k, p[i][2*k], p[i][2*k+1]);
      }
    }

    /* expand */
    for(x1 = 0; x1 < LX; x1++) {
    for(x2 = 0; x2 < LY; x2++) {
    for(x3 = 0; x3 < LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      for(i=0; i<n_s*n_c; i++) {
        _fv_eq_zero( spinor_test[i]+_GSI(ix) );
        for(k=0; k<4; k++) {
          _fv_eq_fv_ti_co(sp1, stochastic_propagator_list[4+k]+_GSI(ix), (complex*)(p[i]+2*k) );
          _fv_pl_eq_fv(spinor_test[i]+_GSI(ix), sp1);
        }
      }
    }}}
    fini_2level_buffer(&p);

  }


  sprintf(filename, "vertex2.%d", g_cart_id);
  ofs = fopen(filename, "w");

  for(i=0; i<n_s*n_c; i++) {
    double norm = 0.;
    spinor_field_norm_diff (&norm, pfifi_list[i], spinor_test[i], VOLUME);
    if(g_cart_id == 0) fprintf(stdout, "# [test] norm %2d %25.16e\n", i, norm);
  }

#if 0
  for(x0 = 0; x0 < T; x0++) {
  for(x1 = 0; x1 < LX; x1++) {
  for(x2 = 0; x2 < LY; x2++) {
  for(x3 = 0; x3 < LZ; x3++) {

    ix = g_ipt[x0][x1][x2][x3];
    fprintf(ofs, "# [] x %3d %3d %3d %3d \n", 
        x0+g_proc_coords[0]*T, x1+g_proc_coords[1]*LX, x2+g_proc_coords[2]*LY, x3+g_proc_coords[3]*LZ);

    for(i=0; i<n_s*n_c; i++) {
      for(k=0; k<n_s*n_c; k++) {
        fprintf(ofs, "%2d %d %d %25.16e %25.16e \t %25.16e %25.16e\n", 
          i, k/n_c, k%n_c,
          pfifi_list[i][ _GSI(ix)+2*k], pfifi_list[i][ _GSI(ix)+2*k+1],
          spinor_test[i][_GSI(ix)+2*k], spinor_test[i][_GSI(ix)+2*k+1] );

      }
    }
  }}}}

  fclose(ofs);
#endif


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
#endif  // of if 0
  return(0);

}
