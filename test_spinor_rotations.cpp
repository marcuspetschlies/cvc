/****************************************************
 * test_spinor_rotations.cpp
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

int main(int argc, char **argv) {

  const int Ndim = 7;

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;
  double norm, norm2;
  double _Complex **R = NULL;
  double _Complex *v = NULL;
  double _Complex *w = NULL;


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
        fprintf(stdout, "# [test_spinor_rotations] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_spinor_rotations] Reading input from file %s\n", filename);
  read_input_parser(filename);

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
    fprintf(stderr, "[test_spinor_rotations] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

  rot_init_rotation_table();

  rlxd_init(2, g_seed);

  R = rot_init_rotation_matrix (Ndim);
  

  /***********************************************************
   * 
   ***********************************************************/

  double r[3];
  ranlxd( r, 3);

  double a[3] = { 2*r[0] * M_PI, r[1] * M_PI, 2*r[2] * M_PI };

  rot_rotation_matrix_spherical_basis_Wigner_D ( R, Ndim-1, a );

  rot_printf_matrix ( R, Ndim, "R", stdout  );

  /***********************************************************/
  /***********************************************************/

  init_1level_zbuffer ( &v, Ndim );

  init_1level_zbuffer ( &w, Ndim );

  ranlxd( (double*)v, 2*Ndim );
  // memset( w, 0, Ndim*sizeof(double _Complex) );
  ranlxd( (double*)w, 2*Ndim );

  /***********************************************************/
  /***********************************************************/

  fprintf(stdout, "v <- c( %25.16e + %25.16e*1.i ", creal( v[0]) , cimag(v[0]));
  for ( int i = 1; i < Ndim; i++ )
    fprintf(stdout, ", %25.16e + %25.16e*1.i ", creal( v[i]) , cimag(v[i]));
  fprintf(stdout, " )\n");

#if 0
  rot_mat_ti_vec ( w, R, v, Ndim );

  fprintf(stdout, "w <- c( %25.16e + %25.16e*1.i ", creal( w[0]) , cimag(w[0]));
  for ( int i = 1; i < Ndim; i++ )
    fprintf(stdout, ", %25.16e + %25.16e*1.i ", creal( w[i]) , cimag(w[i]));
  fprintf(stdout, " )\n");

  rot_mat_transpose_ti_vec ( w, R, v, Ndim );

  fprintf(stdout, "wt <- c( %25.16e + %25.16e*1.i ", creal( w[0]) , cimag(w[0]));
  for ( int i = 1; i < Ndim; i++ )
    fprintf(stdout, ", %25.16e + %25.16e*1.i ", creal( w[i]) , cimag(w[i]));
  fprintf(stdout, " )\n");

  rot_mat_adjoint_ti_vec ( w, R, v, Ndim );

  fprintf(stdout, "wa <- c( %25.16e + %25.16e*1.i ", creal( w[0]) , cimag(w[0]));
  for ( int i = 1; i < Ndim; i++ )
    fprintf(stdout, ", %25.16e + %25.16e*1.i ", creal( w[i]) , cimag(w[i]));
  fprintf(stdout, " )\n");
#endif

  double _Complex b[2];
  ranlxd( (double*)b, 4 );
  fprintf(stdout, "bv <- %25.16e + %25.16e*1.i\n", creal(b[0]), cimag(b[0]));
  fprintf(stdout, "bw <- %25.16e + %25.16e*1.i\n", creal(b[1]), cimag(b[1]));

  fprintf(stdout, "wb <- c( %25.16e + %25.16e*1.i ", creal( w[0]) , cimag(w[0]));
  for ( int i = 1; i < Ndim; i++ )
    fprintf(stdout, ", %25.16e + %25.16e*1.i ", creal( w[i]) , cimag(w[i]));
  fprintf(stdout, " )\n");

  rot_vec_accum_vec_ti_co_pl_mat_ti_vec_ti_co ( w, R, v, b[0], b[1], Ndim );


  fprintf(stdout, "wa <- c( %25.16e + %25.16e*1.i ", creal( w[0]) , cimag(w[0]));
  for ( int i = 1; i < Ndim; i++ )
    fprintf(stdout, ", %25.16e + %25.16e*1.i ", creal( w[i]) , cimag(w[i]));
  fprintf(stdout, " )\n");

  /***********************************************************/
  /***********************************************************/

  rot_fini_rotation_matrix ( &R );
  fini_1level_zbuffer ( &v );
  fini_1level_zbuffer ( &w );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_spinor_rotations] %s# [test_spinor_rotations] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_spinor_rotations] %s# [test_spinor_rotations] end of run\n", ctime(&g_the_time));
  }

  free_geometry ();
#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
