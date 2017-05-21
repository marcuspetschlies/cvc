/****************************************************
 * test_rotations.cpp
 *
 * Sat May 13 22:36:53 CEST 2017
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
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
#include "read_input_parser.h"
#include "matrix_init.h"
#include "rotations.h"
#include "ranlxd.h"
#include "prepare_source.h"
#include "Q_phi.h"

using namespace cvc;

void usage() {
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}

void print_sf (double*sf, char*name) {

  FILE *ofs = fopen(name, "w");
  for( int x0 = 0; x0 < T; x0++ ) {
  for( int x1 = 0; x1 < LX; x1++ ) {
  for( int x2 = 0; x2 < LY; x2++ ) {
  for( int x3 = 0; x3 < LZ; x3++ ) {
    unsigned int ix = g_ipt[x0][x1][x2][x3];
    fprintf(ofs, "# x %3d%3d%3d%3d\n", x0, x1, x2, x3);
    for( int mu=0; mu<12; mu++ ) {
      fprintf(ofs, "%3d %3d %25.16e %25.16e\n", mu/3, mu%3, sf[_GSI(ix)+2*mu], sf[_GSI(ix)+2*mu+1]);
    }
  }}}}
  fclose(ofs);
}  /* end of print_sf */


void print_fp_point_from_sf (double**sf, unsigned int ix, char*name, FILE*ofs) {
  fprintf(ofs, "%s <- array(dim=c(%d, %d))\n", name,12,12);
  for( int mu=0; mu<12; mu++ ) {
  for( int nu=0; nu<12; nu++ ) {
    fprintf(ofs, "%s[%2d,%2d] <- %25.16e +  %25.16e*1.i;\n", name, mu+1, nu+1,
        sf[nu][_GSI(ix)+2*mu], sf[nu][_GSI(ix)+2*mu+1]);
  }}
}  /* end of print_fp_point_from_sf */


int main(int argc, char **argv) {

  const double ONE_OVER_SQRT2 = 1. / sqrt(2.);
  const int Ndim = 3;

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;
  double norm;
  FILE *ofs = NULL;
  double _Complex **R = NULL;
  double _Complex **A = NULL;
  double _Complex **B = NULL;
  double _Complex **ASpin = NULL;
  char name[12];
  fermion_propagator_type fp1, fp2;


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
        fprintf(stdout, "# [test_rotations] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_rotations] Reading input from file %s\n", filename);
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1);
  if(exitstatus != 0) {
    EXIT(1);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    EXIT(2);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
    if(exitstatus != 0) {
    EXIT(3);
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
 if(g_cart_id == 0) fprintf(stdout, "# [p2gg] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_rotations] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  rot_init_rotation_table();

  rlxd_init(2, g_seed);


  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [p2gg] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [p2gg] initializing unit matrices\n");
    for( unsigned int ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  }
#ifdef HAVE_MPI
  xchange_gauge_field( g_gauge_field );
#endif

  fflush(stdout);
  fflush(stderr);
#ifdef HAVE_MPI
  MPI_Barrier(g_cart_grid);
#endif

  if (g_cart_id == 0 ) {
    fprintf(stdout, "# [test_rotations] plaquette reference value\n");
    fflush(stdout);
  }
  plaquetteria  ( g_gauge_field );

  double *gauge_field_rot = NULL;
  alloc_gauge_field(&gauge_field_rot, VOLUMEPLUSRAND);

  R = rot_init_rotation_matrix ( Ndim );
  A = rot_init_rotation_matrix ( Ndim );
  B = rot_init_rotation_matrix ( Ndim );

  no_fields = 25;
  g_spinor_field = (double**)malloc( no_fields * sizeof(double*));
  g_spinor_field[0] = (double*)malloc( no_fields * _GSI(VOLUME+RAND)*sizeof(double) );
  for( int i=1; i<no_fields; i++  ) {
    g_spinor_field[i] = g_spinor_field[i-1] +  _GSI(VOLUME+RAND);
  }

  ofs = fopen( "tmp", "w");

  /***********************************************************
   * loop on rotations
   ***********************************************************/
  // for(int irot=0; irot < 48; irot++ )
  for(int irot = 46; irot < 47; irot++ )
  // for(int irot = 0; irot < 1; irot++ )
  {

    if (g_cart_id == 0 ) {
      fprintf(stdout, "# [test_rotations] rotation no. %2d n = (%2d, %2d, %2d) w = %16.7e pi\n", irot,
          cubic_group_double_cover_rotations[irot].n[0], cubic_group_double_cover_rotations[irot].n[1], cubic_group_double_cover_rotations[irot].n[2],
          cubic_group_double_cover_rotations[irot].w);
    }

    rot_rotation_matrix_spherical_basis ( R, Ndim-1, cubic_group_double_cover_rotations[irot].n, cubic_group_double_cover_rotations[irot].w );
    rot_spherical2cartesian_3x3 (A, R);
    if ( rot_mat_check_is_real_int ( A, Ndim ) ) {
      if (g_cart_id == 0 )
        fprintf(stdout, "# [test_rotations] rot_mat_check_is_real_int matrix A rot %2d ok\n", irot);
    } else {
      EXIT(6);
    }

    sprintf(name, "A[%.2d]", irot);
    rot_printf_rint_matrix (A, Ndim, name, stdout );


    exitstatus = rot_gauge_field ( gauge_field_rot, g_gauge_field, A);
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[test_rotations] Error from rot_gauge_field, status was %d\n", exitstatus);
      EXIT(23);
    }

    xchange_gauge_field ( gauge_field_rot );

    if (g_cart_id == 0 ) {
      fprintf(stdout, "# [test_rotations] plaquette value rotation %2d\n", irot);
      fflush(stdout);
    }
    plaquetteria  ( gauge_field_rot );

    prepare_volume_source(g_spinor_field[0], VOLUME);

    xchange_field ( g_spinor_field[0] );

    ASpin = rot_bispinor_rotation_matrix_spherical_basis ( cubic_group_double_cover_rotations[irot].n, cubic_group_double_cover_rotations[irot].w );

    ranlxd( (double*)(ASpin[0]), 32);

    // rot_mat_adj(ASpin, ASpin, 4);

    sprintf(name, "ASpin%.2d", irot);
    rot_printf_matrix (ASpin, 4, name, ofs );

    create_fp(  &fp1 );
    create_fp(  &fp2 );

    // ranlxd( fp1[0], 288);

    ranlxd( g_spinor_field[0], _GSI(VOLUME)*12 );

    sprintf(name, "sfp%.2d", irot );
    // printf_fp( fp1, name, ofs);
    print_fp_point_from_sf ( &(g_spinor_field[0]), 11, name, ofs);

    // rot_bispinor_mat_ti_fp( fp2, ASpin, fp1 );

    // rot_fp_ti_bispinor_mat ( fp2, ASpin, fp1 );

    rot_fv_ti_bispinor_mat ( &(g_spinor_field[12]), ASpin, &(g_spinor_field[0]), 11 );

    sprintf(name, "sfp_rot%.2d", irot );
    // printf_fp( fp2, name, ofs);
    print_fp_point_from_sf ( &(g_spinor_field[12]), 11, name, ofs);

    free_fp( &fp1 );
    free_fp( &fp2 );

    rot_fini_rotation_matrix (&ASpin);

  }  /* end of loop on rotations */

  fclose( ofs );
  rot_fini_rotation_matrix ( &R );
  rot_fini_rotation_matrix ( &A );
  rot_fini_rotation_matrix ( &B );

  free ( g_spinor_field[0] );
  free ( g_spinor_field );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_rotations] %s# [test_rotations] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_rotations] %s# [test_rotations] end of run\n", ctime(&g_the_time));
  }

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
