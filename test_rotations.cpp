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
#include "scalar_products.h"

#define _SQR(_a) ((_a)*(_a))

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

int main(int argc, char **argv) {

  const double ONE_OVER_SQRT2 = 1. / sqrt(2.);
  const int Ndim = 3;

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;
  double norm, norm2;
  FILE *ofs = NULL;
  double _Complex **R = NULL;
  double _Complex **A = NULL;
  double _Complex **B = NULL;


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

/*
  int n[3] = {-1,1,1};
  double w = 4*M_PI/3.;
*/
/*
  int n[3] = {-1,1,0};
  double w = M_PI;
*/
/*
  int n[3] = {1,1,1};
  double w = 4*M_PI/3.;
*/

  rot_init_rotation_table();

  rlxd_init(2, g_seed);

#if 0
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if( strcmp(gaugefilename_prefix,"identity") == 0 ) {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [p2gg] initializing unit matrices\n");
    for( unsigned int ix=0;ix<VOLUME;ix++) {
      _cm_eq_id( g_gauge_field + _GGI(ix, 0) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 1) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 2) );
      _cm_eq_id( g_gauge_field + _GGI(ix, 3) );
    }
  } else if ( strcmp( gaugefilename_prefix, "random") == 0 ) {
    random_gauge_field( g_gauge_field, 1.);
  } else {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [p2gg] reading gauge field from file %s\n", filename);
    read_lime_gauge_field_doubleprec(filename);
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
#endif  /* of if 0 */

/*
  sprintf(filename, "gf_orig.%.2d", g_cart_id);
  ofs = fopen(filename, "w");
  for( int x0 = 0; x0 < T; x0++ ) {
  for( int x1 = 0; x1 < LX; x1++ ) {
  for( int x2 = 0; x2 < LY; x2++ ) {
  for( int x3 = 0; x3 < LZ; x3++ ) {
    unsigned int ix = g_ipt[x0][x1][x2][x3];
    fprintf(ofs, "# x %3d%3d%3d%3d\n", x0, x1, x2, x3);
    for( int mu=0; mu<36; mu++ ) {
      fprintf(ofs, "%3d %3d %25.16e %25.16e\n", mu/9, mu%9, 
          g_gauge_field[_GGI(ix,0)+2*mu], g_gauge_field[_GGI(ix,0)+2*mu+1]);
    }
  }}}}
  fclose(ofs);
*/

  double *gauge_field_rot = NULL;
  alloc_gauge_field(&gauge_field_rot, VOLUMEPLUSRAND);

  no_fields = 5;
  g_spinor_field = (double**)malloc( no_fields * sizeof(double*));
  g_spinor_field[0] = (double*)malloc( no_fields * _GSI(VOLUME+RAND)*sizeof(double) );
  for( int i=1; i<no_fields; i++  ) {
    g_spinor_field[i] = g_spinor_field[i-1] +  _GSI(VOLUME+RAND);
  }

  R = rot_init_rotation_matrix (Ndim);
  A = rot_init_rotation_matrix (Ndim);
  B = rot_init_rotation_matrix (Ndim);

  /***********************************************************
   * loop on rotations
   ***********************************************************/
  ofs = fopen("spin1_rotation_matrices", "w");

  fprintf( ofs, "A_spherical <- list()\n" );

  for(int irot=0; irot < 48; irot++ )
  // for(int irot = 46; irot < 47; irot++ )
  // for(int irot = 0; irot < 1; irot++ )
  {
    char name[20];

    if (g_cart_id == 0 ) {
      // fprintf( ofs, "\n# [test_rotations] rotation no. %2d n = (%2d, %2d, %2d) w = %16.7e pi\n", 
      fprintf( ofs, "\n\n# [test_rotations] rotation no.  %d n = ( %d,  %d,  %d) w = %16.6e pi\n", 
          irot+1,
          cubic_group_double_cover_rotations[irot].n[0], cubic_group_double_cover_rotations[irot].n[1], cubic_group_double_cover_rotations[irot].n[2],
          cubic_group_double_cover_rotations[irot].w / M_PI);
    }

    rot_rotation_matrix_spherical_basis ( R, Ndim-1, cubic_group_double_cover_rotations[irot].n, cubic_group_double_cover_rotations[irot].w );

    sprintf(name, "A_spherical[[%d]]", irot+1);
    fprintf( ofs, "%s <- array( dim=c(3,3) )\n", name, irot+1 );
    rot_printf_matrix ( R, Ndim, name, ofs );


    rot_spherical2cartesian_3x3 (A, R);
    if ( rot_mat_check_is_real_int ( A, Ndim ) ) {
      if (g_cart_id == 0 )
        fprintf(stdout, "# [test_rotations] rot_mat_check_is_real_int matrix A rot %2d ok\n", irot);
    } else {
      EXIT(6);
    }

    sprintf(name, "A_cartesian[[%d]]", irot+1);
    rot_printf_rint_matrix (A, Ndim, name, stdout );

    int nrot[3] = {0,0,0};
    
    rot_point ( nrot, cubic_group_double_cover_rotations[irot].n, A );
    double norm = sqrt(
      _SQR(nrot[0] - cubic_group_double_cover_rotations[irot].n[0]) + 
      _SQR(nrot[1] - cubic_group_double_cover_rotations[irot].n[1]) + 
      _SQR(nrot[2] - cubic_group_double_cover_rotations[irot].n[2]) );
    double norm2 = sqrt( _SQR(nrot[0]) + _SQR(nrot[1]) + _SQR(nrot[2]) );

    double norm3 = sqrt( 
        _SQR( cubic_group_double_cover_rotations[irot].n[0] ) + 
        _SQR( cubic_group_double_cover_rotations[irot].n[1] ) + 
        _SQR( cubic_group_double_cover_rotations[irot].n[2] ) );
    fprintf(stdout, "# [test_rotations] irot %2d || Rn - n || = %16.7e || n ||      = %16.7e || Rn ||     = %16.7e \n", irot+1, norm, norm2, norm3);

#if 0
    exitstatus = rot_gauge_field ( gauge_field_rot, g_gauge_field, A);
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[test_rotations] Error from rot_gauge_field, status was %d\n", exitstatus);
      EXIT(23);
    }
#endif  /* of if 0 */

/*
    sprintf(filename, "gf_rot.%.2d", g_cart_id);
    ofs = fopen(filename, "w");
    for( int x0 = 0; x0 < T; x0++ ) {
    for( int x1 = 0; x1 < LX; x1++ ) {
    for( int x2 = 0; x2 < LY; x2++ ) {
    for( int x3 = 0; x3 < LZ; x3++ ) {
      unsigned int ix = g_ipt[x0][x1][x2][x3];
      fprintf(ofs, "# x %3d%3d%3d%3d\n", x0, x1, x2, x3);
      for( int mu=0; mu<36; mu++ ) {
        fprintf(ofs, "%3d %3d %25.16e %25.16e\n", mu/9, mu%9, 
            gauge_field_rot[_GGI(ix,0)+2*mu], gauge_field_rot[_GGI(ix,0)+2*mu+1]);
      }
    }}}}
    fclose(ofs);
*/

#if 0

#ifdef HAVE_MPI
    xchange_gauge_field ( gauge_field_rot );
#endif
    if (g_cart_id == 0 ) {
      fprintf(stdout, "# [test_rotations] plaquette value rotation %2d\n", irot);
      fflush(stdout);
    }
    plaquetteria  ( gauge_field_rot );

#endif  /* of if 0 */

#if 0
    /* B = A^-1 */
    rot_mat_adj (B, A, Ndim);
    if ( rot_mat_check_is_real_int ( B, Ndim ) ) {
      if (g_cart_id == 0 )
        fprintf(stdout, "# [test_rotations] rot_mat_check_is_real_int matrix B rot %2d ok\n", irot);
    } else {
      EXIT(6);
    }
    sprintf(name, "B[%.2d]", irot);
    rot_printf_rint_matrix (B, Ndim, name, stdout );

/*
    rot_mat_ti_mat (R, A, B, Ndim);
    if ( rot_mat_check_is_real_int ( R, Ndim ) ) {
      if (g_cart_id == 0 )
        fprintf(stdout, "# [test_rotations] rot_mat_check_is_real_int matrix R = A B rot %2d ok\n", irot);
    } else {         
      EXIT(6);               
    }         
    sprintf(name, "AxB[%.2d]", irot);
    rot_printf_rint_matrix (R, Ndim, name, stdout );
*/


    prepare_volume_source(g_spinor_field[0], VOLUME);

#ifdef HAVE_MPI
    xchange_field ( g_spinor_field[0] );
#endif

#endif  /* of if 0 */


#if 0
    /**************************
     * direction dir
     **************************/
    int dir = -3, dir_rot = 0;
    int fbwd = dir/abs(dir);

    spinor_field_eq_gauge_field_fbwd_ti_spinor_field ( g_spinor_field[1], g_gauge_field, g_spinor_field[0], abs(dir), fbwd, VOLUME);

    //sprintf(filename, "sf.%d.%.2d", dir, g_cart_id);
    //print_sf( g_spinor_field[1], filename );

    rot_spinor_field ( g_spinor_field[2], g_spinor_field[1], A);
    sprintf(filename, "sf.%d.%.2d", dir, g_cart_id);
    // print_sf( g_spinor_field[2], filename );

    /*(2) V_0 psi ° R */
    rot_spinor_field ( g_spinor_field[1], g_spinor_field[0], A);
    // memcpy( g_spinor_field[1], g_spinor_field[0], _GSI(VOLUME)*sizeof(double));
#ifdef HAVE_MPI
    xchange_field ( g_spinor_field[1] );
#endif

    /* rotate direction dir */
    int d[3] = {0,0,0};
    int drot[3] = {0,0,0};
    d[ abs(dir)-1 ] = dir / abs(dir);
    /* inverse rotation */
    rot_point_inv ( drot, d, A);
    dir_rot =  drot[0] != 0 ? drot[0] : ( drot[1] != 0 ? 2*drot[1] : 3*drot[2] );
    if ( g_cart_id == 0 ) {
      fprintf(stdout, "# [test_rotations] dir %2d to dir rot %2d\n", dir, dir_rot);
    }

    int fbwd_rot = dir_rot / abs( dir_rot );
    spinor_field_eq_gauge_field_fbwd_ti_spinor_field ( g_spinor_field[3], gauge_field_rot, g_spinor_field[1], abs(dir_rot), fbwd_rot, VOLUME);

    sprintf(filename, "sf_rot.%d.%.2d", dir_rot, g_cart_id);
    // print_sf( g_spinor_field[3], filename );
    spinor_field_norm_diff( &norm, g_spinor_field[2], g_spinor_field[3], VOLUME);
    if ( g_cart_id == 0 ) {
      fprintf(stdout, "# [test_rotations] norm rot %2d dir %2d = %25.16e\n", irot, dir, norm);
    }
#endif  /* of if 0 */
#if 0
    initialize ASpin
    rot_bispinor_rotation_matrix_spherical_basis ( ASpin, cubic_group_double_cover_rotations[irot].n, cubic_group_double_cover_rotations[irot].w );

    rot_mat_adj(ASpin, ASpin, 4);
    /*
    sprintf(name, "ASpind[%.2d]", irot);
    rot_printf_matrix (ASpin, 4, name, stdout );
    */

    sprintf(name, "ASpin[%.2d]", irot);
    rot_printf_matrix (ASpin, 4, name, stdout );
/*
    int ispin = 0;
    unsigned int source_location = 0;
    memset( g_spinor_field[0], 0, _GSI(VOLUME)*sizeof(double) );
    g_spinor_field[0][_GSI(source_location)+2*(3*ispin+0)] = 1.;
    sprintf(filename, "sf.%d.%.2d", ispin, g_cart_id);
    print_sf( g_spinor_field[0], filename );
*/

    Q_phi( g_spinor_field[1], g_spinor_field[0], g_gauge_field, g_mu);
    rot_spinor_field ( g_spinor_field[2], g_spinor_field[1], A);
    // rot_bispinor_mat_ti_spinor_field ( g_spinor_field[2], ASpin, g_spinor_field[2], VOLUME);
    //sprintf(filename, "sf_RQ.%d.%.2d", ispin, g_cart_id);
    //print_sf( g_spinor_field[2], filename );
#if 0
#endif  /* of if 0 */

    rot_spinor_field ( g_spinor_field[1], g_spinor_field[0], A);
    rot_bispinor_mat_ti_spinor_field ( g_spinor_field[1], ASpin, g_spinor_field[1], VOLUME);
    Q_phi( g_spinor_field[3], g_spinor_field[1], gauge_field_rot, g_mu);
    rot_mat_adj(ASpin, ASpin, 4);
    rot_bispinor_mat_ti_spinor_field ( g_spinor_field[3], ASpin, g_spinor_field[3], VOLUME);
    //sprintf(filename, "sf_QR.%d.%.2d", ispin, g_cart_id);
    //print_sf( g_spinor_field[3], filename );

    // rot_bispinor_mat_ti_spinor_field ( g_spinor_field[3], ASpin, g_spinor_field[3], VOLUME);
    spinor_field_norm_diff( &norm, g_spinor_field[2], g_spinor_field[3], VOLUME);
    spinor_scalar_product_re ( &norm2, g_spinor_field[0], g_spinor_field[0], VOLUME);
    if ( g_cart_id == 0 ) {
      fprintf(stdout, "# [test_rotations] norm rot %2d %25.16e %25.16e\n", irot, norm2, norm / sqrt(norm2) );
    }

#if 0
    rot_bispinor_mat_ti_spinor_field ( g_spinor_field[1], ASpin, g_spinor_field[0], VOLUME);
    spinor_field_norm_diff( &norm, g_spinor_field[1], g_spinor_field[0], VOLUME);
    if ( g_cart_id == 0 ) {
      fprintf(stdout, "# [test_rotations] norm rot %2d = %25.16e\n", irot, norm);
    }

    rot_mat_adj(ASpin, ASpin, 4);
    rot_bispinor_mat_ti_spinor_field ( g_spinor_field[1], ASpin, g_spinor_field[1], VOLUME);
    spinor_field_norm_diff( &norm, g_spinor_field[1], g_spinor_field[0], VOLUME);
    if ( g_cart_id == 0 ) {
      fprintf(stdout, "# [test_rotations] norm id %2d = %25.16e\n", irot, norm);
    }
#endif  /* of if 0 */

    rot_fini_rotation_matrix (&ASpin);


#endif  /* of if 0 */
  }  /* end of loop on rotations */

  fclose ( ofs );

  rot_fini_rotation_matrix( &R );
  rot_fini_rotation_matrix( &A );
  rot_fini_rotation_matrix( &B );

  free( g_spinor_field[0] );
  free( g_spinor_field );

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
