/****************************************************
 * test_product_mat.cpp
 *
 * Mo 5. Feb 07:58:16 CET 2018
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
#include "group_projection.h"

using namespace cvc;

void usage() {
  EXIT(0);
}

int main(int argc, char **argv) {

  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;


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
        fprintf(stdout, "# [test_product_mat] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_product_mat] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
 if(g_cart_id == 0) fprintf(stdout, "# [test_product_mat] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_product_mat] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_product_mat] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /*********************************
   * set up geometry fields
   *********************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[test_product_mat] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();


  /******************************************************************
   * initialize rotation matrix
   * tables for cubic group and
   * its double cover
   ******************************************************************/
  rot_init_rotation_table();

  /******************************************************************/
  /******************************************************************/

  /******************************************************************
   * initialize random number * generator
   ******************************************************************/
  rlxd_init(2, g_seed);


  const int nspin = 4;
  int spin_list[nspin] = { 2, 4 ,6, 8};

  int *spin_dim = NULL;
  init_1level_ibuffer ( &spin_dim, nspin );
  for ( int i = 0; i < nspin; i++ ) spin_dim[i] = spin_list[i]+1;
  int pdim = 1;
  for ( int i = 0; i < nspin; i++ ) pdim *= spin_dim[i];


  /******************************************************************
   * initialize rotation tables, set and show
   ******************************************************************/
  rot_mat_table_type *rspin = (rot_mat_table_type*)malloc ( nspin * sizeof(rot_mat_table_type ));

  for ( int i = 0; i < nspin; i++ ) {
    init_rot_mat_table ( rspin+i );
    set_rot_mat_table_spin ( rspin+i, spin_list[i], 0 );

    char name[100];
    sprintf ( name, "rspin%d", spin_list[i] );
    rot_mat_table_printf ( rspin+i, name, stdout );
  }

#if 0
#endif

#if 0
  /******************************************************************
   * TEST
   *   ( product_vector_index2coords )
   *   product_vector_set_element
   *   product_vector_printf
   ******************************************************************/

  for ( int idx = 0; idx < pdim; idx++ ) {
    double _Complex *v = NULL;
    int *coords = NULL;
    init_1level_ibuffer ( &coords,   nspin );
    init_1level_zbuffer ( &v, pdim );

    product_vector_index2coords ( idx, coords, spin_dim, nspin );
    product_vector_set_element ( v, (double _Complex)idx+1, coords, spin_dim, nspin );

    char name[100];
    sprintf ( name, "v%d", idx+1 );
    product_vector_printf ( v, spin_dim, nspin, name, stdout );

    fini_1level_zbuffer ( &v);
    fini_1level_ibuffer ( &coords );
  }
#endif

#if 0
  /******************************************************************
   * TEST
   *   product_mat_printf
   ******************************************************************/
  double _Complex **R = rot_init_rotation_matrix ( pdim );

  for ( int idx = 0; idx < pdim; idx++ ) {
    for ( int kdx = 0; kdx < pdim; kdx++ ) {
      R[idx][kdx] = (double _Complex)(idx+1) + (double _Complex)(kdx+1)*I;
  }}

  product_mat_printf ( R, spin_dim, nspin, "mat", stdout );
  rot_fini_rotation_matrix ( &R );
#endif

#if 0
  /******************************************************************
   * TEST
   *   product_vector_project_accum
   ******************************************************************/

  double _Complex *v = NULL, *v0 = NULL;
  init_1level_zbuffer ( &v, pdim );
  init_1level_zbuffer ( &v0, pdim );

  ranlxd ( (double*)v, 2*pdim );
  ranlxd ( (double*)v0, 2*pdim );
  double _Complex ccoeff[2];
  ranlxd ( (double*)ccoeff, 4 );
  fprintf ( stdout, "c1 <- %25.16e + %25.16e*1.i\n", creal(ccoeff[0]), cimag(ccoeff[0]) );
  fprintf ( stdout, "c2 <- %25.16e + %25.16e*1.i\n", creal(ccoeff[1]), cimag(ccoeff[1]) );

  for ( int irot = 0; irot < 48; irot++ ) {
    char name[100];
    sprintf ( name, "v[[%d]]", irot+1 );
    product_vector_printf ( v, spin_dim, nspin, name, stdout );
    sprintf ( name, "u[[%d]]", irot+1 );
    product_vector_printf ( v0, spin_dim, nspin, name, stdout );

    product_vector_project_accum ( v, rspin, -1, irot, v0,  ccoeff[0], ccoeff[1], spin_dim , nspin );

    sprintf ( name, "z[[%d]]", irot+1 );
    product_vector_printf ( v, spin_dim, nspin, name, stdout );
  }

  fini_1level_zbuffer ( &v );
  fini_1level_zbuffer ( &v0 );


  fini_1level_ibuffer ( &spin_dim );

#endif

#if 0
  /******************************************************************
   * TEST
   *   product_mat_pl_eq_mat_ti_co
   ******************************************************************/


  double _Complex ccoeff;
  ranlxd ( (double*)&ccoeff, 2 );

  fprintf ( stdout, "c <- %25.16e + %25.16e*1.i\n", creal(ccoeff), cimag(ccoeff) );
  double _Complex **R = rot_init_rotation_matrix ( pdim );

  // ranlxd( (double*)R[0], 2*pdim*pdim );
  // memset  ( R[0], 0, 2*pdim*pdim*sizeof(double) );

  for ( int irot = 8; irot < 19; irot++ )
  {
    ranlxd( (double*)R[0], 2*pdim*pdim );
    char name[100];
    sprintf ( name, "V[[%d]]", irot+1 );
    product_mat_printf ( R, spin_dim, nspin, name, stdout );

    product_mat_pl_eq_mat_ti_co ( R, rspin, -1, irot, ccoeff, spin_dim, nspin );

    sprintf ( name, "Z[[%d]]", irot+1 );
    product_mat_printf ( R, spin_dim, nspin, name, stdout );
  }

  rot_fini_rotation_matrix ( &R );
#endif
#if 0
  const int nspin = 4;
  int *coords = NULL, *spin_dim = NULL;
  init_1level_ibuffer ( &coords,   nspin );
  init_1level_ibuffer ( &spin_dim, nspin );
  spin_dim[0] = 3;
  spin_dim[1] = 2;
  spin_dim[2] = 5;
  spin_dim[3] = 7;
  int pdim = 1;
  for ( int i = 0; i < nspin; i++ ) pdim *= spin_dim[i];
  fprintf ( stdout, "# [test_product_mat] pdim = %d\n", pdim );
  for ( int idx = 0; idx < pdim; idx++ )  {
    product_vector_index2coords ( idx, coords, spin_dim, nspin );
    int idy = product_vector_coords2index ( coords,  spin_dim, nspin );
      
    fprintf ( stdout, "idx %3d coords ", idx );
    for ( int k = 0; k < nspin; k++ ) fprintf ( stdout, " %2d", coords[k] );
    fprintf ( stdout, " %3d\n", idy);
  }

  fini_1level_ibuffer ( &coords );
  fini_1level_ibuffer ( &spin_dim );
#endif

  /******************************************************************
   * TEST
   *   rot_mat_eq_product_mat_ti_rot_mat
   ******************************************************************/

  rot_mat_table_type rspin2;
  rot_mat_table_eq_product_mat_table ( &rspin2, rspin, nspin );


  double _Complex **R  = rot_init_rotation_matrix ( pdim );
  double _Complex **R2 = rot_init_rotation_matrix ( pdim );
  double _Complex **S  = rot_init_rotation_matrix ( pdim );


  ranlxd ( (double*)(S[0]), 2*pdim*pdim );

  rot_printf_matrix ( S, pdim, "S", stdout );


  for ( int irot = 0; irot < 48 ; irot++ )
  {
    char name[100];

    rot_mat_eq_product_mat_ti_rot_mat ( R, rspin, -1, irot, S, nspin );
    sprintf ( name, "R[[%d]]", irot+1 );
    rot_printf_matrix ( R, pdim, name, stdout );

    rot_mat_ti_mat ( R2, rspin2.IR[irot] , S, pdim );
    sprintf ( name, "R2[[%d]]", irot+1 );
    rot_printf_matrix ( R2, pdim, name, stdout );

    double norm = rot_mat_norm_diff ( R, R2, pdim );
    fprintf ( stdout, "# [test_product_mat] irot %d norm diff %16.7e\n", irot, norm );

  }

  rot_fini_rotation_matrix ( &R );
  rot_fini_rotation_matrix ( &R2 );
  rot_fini_rotation_matrix ( &S );

  fini_rot_mat_table ( &rspin2 );

  for ( int i = 0; i < nspin; i++ ) fini_rot_mat_table ( rspin+i );
  free ( rspin );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_product_mat] %s# [test_product_mat] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_product_mat] %s# [test_product_mat] end of run\n", ctime(&g_the_time));
  }

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
