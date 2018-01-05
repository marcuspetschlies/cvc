/****************************************************
 * test_rot_mult_tab_lg.cpp
 *
 * Mo 11. Dez 16:25:05 CET 2017
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

  int Ndim;

  int c;
  int filename_set = 0;
  char filename[100];
  char name[12];
  int exitstatus;
  /* FILE *ofs = NULL; */
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
        fprintf(stdout, "# [test_rot_mult_tab_lg] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_rot_mult_tab_lg] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
 if(g_cart_id == 0) fprintf(stdout, "# [test_rot_mult_tab_lg] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_rot_mult_tab_lg] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_rot_mult_tab_lg] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /*********************************
   * set up geometry fields
   *********************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[test_rot_mult_tab_lg] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();


  /*********************************
   * initialize rotation matrix
   * tables for cubic group and
   * its double cover
   *********************************/
  rot_init_rotation_table();

  /*********************************
   * initialize random number 
   * generator
   *********************************/
  rlxd_init(2, g_seed);

  little_group_type *lg = NULL;
  int nlg = 0;

  if ( ( nlg = little_group_read_list ( &lg, "little_groups_2Oh.tab") ) <= 0 )
  {
    fprintf(stderr, "[test_rot_mult_tab_lg] Error from little_group_read_list, status was %d\n", nlg);
    EXIT(2);
  }
  fprintf(stdout, "# [test_rot_mult_tab_lg] number of little groups = %d\n", nlg);

  little_group_show ( lg, stdout, nlg );



#if 0
  /*********************************
   * 2 x spin + 1
   *********************************/
  Ndim = 4;

  R = rot_init_rotation_matrix (Ndim);
  A = rot_init_rotation_matrix (Ndim);
  B = rot_init_rotation_matrix (Ndim);


  /***********************************************************
   * loop on rotations
   ***********************************************************/
  for(int irot=0; irot < 48; irot++ )
  {

    if (g_cart_id == 0 ) {
      fprintf(stdout, "# [test_rot_mult_tab_lg] rotation no. %2d n = (%2d, %2d, %2d) w = %16.7e pi\n", irot,
          cubic_group_double_cover_rotations[irot].n[0], cubic_group_double_cover_rotations[irot].n[1], cubic_group_double_cover_rotations[irot].n[2],
          cubic_group_double_cover_rotations[irot].w);
    }

    rot_rotation_matrix_spherical_basis ( R, Ndim-1, cubic_group_double_cover_rotations[irot].n, cubic_group_double_cover_rotations[irot].w );

    sprintf(name, "Ashpe[%.2d]", irot);
    rot_printf_matrix ( R, Ndim, name, stdout );


    rot_spherical2cartesian_3x3 (A, R);
    if ( rot_mat_check_is_real_int ( A, Ndim ) ) {
      if (g_cart_id == 0 )
        fprintf(stdout, "# [test_rot_mult_tab_lg] rot_mat_check_is_real_int matrix A rot %2d ok\n", irot);
    } else {
      EXIT(6);
    }

    sprintf(name, "Akart[%.2d]", irot);
    rot_printf_rint_matrix (A, Ndim, name, stdout );

  }  /* end of loop on rotations */

#endif  /* of if 0 */


  for ( int ilg = 0; ilg < nlg; ilg++ )
  // for ( int ilg = 0; ilg <= 0; ilg++ )
  {

    for ( int iirrep = 0; iirrep < lg[ilg].nirrep; iirrep++ )
    // for ( int iirrep = 0; iirrep <= 0; iirrep++ )
    {

      fprintf(stdout, "# [test_rot_mult_tab_lg] lg %s irrep %s\n", lg[ilg].name, lg[ilg].lirrep[iirrep] );

      rot_mat_table_type rtab;
      init_rot_mat_table ( &rtab );

      if ( ( exitstatus = set_rot_mat_table_cubic_group_double_cover ( &rtab, lg[ilg].name, lg[ilg].lirrep[iirrep] ) ) != 0 ) {
        fprintf(stderr, "[test_rot_mult_tab_lg] Error from set_rot_mat_table_cubic_group_double_cover; status was %d\n", exitstatus );
        exit(1);
      }

      sprintf( filename, "%s.%s.mat", rtab.group, rtab.irrep );
      FILE *ofs = fopen ( filename, "w" );
      rot_mat_table_printf ( &rtab, "U", ofs );
      fclose ( ofs );
     
      double _Complex **rtab_characters = NULL;

      if ( ( exitstatus = rot_mat_table_character ( &rtab_characters, &rtab ) ) != 0 ) {
        fprintf(stderr, "[test_rot_mult_tab_lg] Error from rot_mat_table_character; status was %d\n", exitstatus );
        exit(1);
      }

      sprintf( filename, "%s.%s.character", rtab.group, rtab.irrep );
      ofs = fopen ( filename, "w" );
      for ( int i = 0; i < rtab.n; i++ ) {
        fprintf( ofs, "%s  %s   R%.2d  %f  %f\n", rtab.group, rtab.irrep, rtab.rid[i]+1, creal ( rtab_characters[0][i] ), cimag ( rtab_characters[0][i] ));
      }
      fprintf( ofs, "\n");
      for ( int i = 0; i < rtab.n; i++ ) {
        fprintf( ofs, "%s  %s  IR%.2d  %f  %f\n", rtab.group, rtab.irrep, rtab.rmid[i]+1, creal ( rtab_characters[1][i] ), cimag ( rtab_characters[1][i] ));
      }
      fclose ( ofs );

      fini_rot_mat_table ( &rtab );
      fini_2level_zbuffer ( &rtab_characters );

    }

  }  /* end of loop on irreps */
#if 0
#endif  /* of if 0 */


  little_group_fini ( &lg, nlg );


#if 0
  init_rot_mat_table ( &rtab2 );
  if ( ( exitstatus = set_rot_mat_table_spin ( &rtab2, 1, 0 ) ) != 0 ) {
    fprintf(stderr, "[test_rot_mult_tab_lg] Error from set_rot_mat_table_spin; status was %d\n", exitstatus );
    exit(1);
  }
#endif

#if 0
  /***********************************************************
   * check rotations for SU property
   ***********************************************************/
  for ( int i = 0; i < 48; i++ ) {
    int notsun = 1 - rot_mat_check_is_sun ( rtab.R[i], rtab.dim);
    /* fprintf ( stdout, "# [test_rot_mult_tab_lg] %2d not special unitary is %d\n", i, notsun ); */
    if ( notsun ) {
      rot_printf_matrix ( rtab.R[i], rtab.dim, "error", stderr );
      EXIT(2);
    }
#if 0
    /***********************************************************
     * special case for Ndim = 3, i.e. spin 1
     ***********************************************************/
    if ( Ndim == 3 ) {
      rot_spherical2cartesian_3x3 ( A, rtab.R[i] );
      int notrealint = 1 - rot_mat_check_is_real_int ( A, rtab.dim );

      if ( !notrealint ) {
        if (g_cart_id == 0 ) fprintf(stdout, "# [test_rot_mult_tab_lg] rot_mat_check_is_real_int matrix A rot %2d ok\n", i);
      } else {
        fprintf(stderr, "[test_rot_mult_tab_lg] rotation no. %2d not ok n = %d %d %d w %25.16e\n", i,
            cubic_group_double_cover_rotations[i].n[0],
            cubic_group_double_cover_rotations[i].n[1],
            cubic_group_double_cover_rotations[i].n[2],
            cubic_group_double_cover_rotations[i].w);
        rot_printf_rint_matrix ( A, rtab.dim, "error", stderr );

        EXIT(6);
      }

      /***********************************************************
       * print rotation matrix in Cartesian basis
       ***********************************************************/
      sprintf(name, "Akart[%.2d]", i );
      rot_printf_rint_matrix ( A, rtab.dim, name, stdout );
    }  /* end of if Ndim == 3 */
#endif
  }  /* end of loop on rotations */
#endif  /* of if 0 */

#if 0
  /* TEST */
  for ( int i = 0; i < 48; i++ ) {
    for ( int k = 0; k < 48; k++ ) {
        double diff_norm2 = rot_mat_diff_norm2 ( rtab.R[i], rtab.R[k], rtab.dim );
        if ( diff_norm2 < 5.e-15 ) {
          fprintf(stdout, "# [test_rot_mult_tab_lg]  match %2d is %2d %2d\n", i, i, k);
      }
    }
  }
  /* END OF TEST */
#endif  /* of if 0  */

  /* TEST */
/*
  for ( int i = 0; i < 24; i++ ) {
    int k0 = cubic_group_double_cover_identification_table[i][0];
    int k1 = cubic_group_double_cover_identification_table[i][1];
    double diff_norm2 = rot_mat_diff_norm2 ( rtab.R[k0], rtab.R[k1], rtab.dim );
    fprintf(stdout, "# [test_rot_mult_tab_lg] %2d pair %2d %2d matches at %e\n", i, k0, k1, diff_norm2);
  }
*/
  /* END OF TEST */


#if 0
  int **rtab_mult_table = NULL, **rtab2_mult_table = NULL;

  if ( ( exitstatus = rot_mat_mult_table ( &rtab_mult_table, &rtab ) ) != 0 ) {
    fprintf(stderr, "[test_rot_mult_tab_lg] Error from rot_mat_mult_table, status was %d\n", exitstatus);
    EXIT(1);
  }

  if ( ( exitstatus = rot_mat_mult_table ( &rtab2_mult_table, &rtab2 ) ) != 0 ) {
    fprintf(stderr, "[test_rot_mult_tab_lg] Error from rot_mat_mult_table, status was %d\n", exitstatus);
    EXIT(1);
  }
#endif

#if 0
  int dvec[3] = {0,0,0};
  if ( ! rot_mat_table_is_lg ( &rtab, dvec ) ) {
    fprintf(stderr, "[test_rot_mult_tab_lg] Error from rot_mat_table_is_lg\n");
    EXIT(1)
  } else {
    fprintf(stdout, "# [test_rot_mult_tab_lg] %s irrep %s is lg for d = %3d %3d %3d\n", rtab.irrep, rtab.group, dvec[0], dvec[1], dvec[2] );
  }
#endif

#if 0
  for ( int i = 0; i < rtab.n; i++ ) {
    for ( int k = 0; k < rtab.n; k++ ) {
      if ( rtab_mult_table[i][k] != rtab2_mult_table[i][k] ) {
        fprintf(stdout, "# [test_rot_mult_tab_lg] mult tab no match for %2d %2d   %2d %2d\n", i,k, rtab_mult_table[i][k], rtab2_mult_table[i][k] );
      } else {
        fprintf(stdout, "# [test_rot_mult_tab_lg] mult tab match for %2d %2d   %2d %2d\n", i,k, rtab_mult_table[i][k], rtab2_mult_table[i][k] );
      }
    }
  }

  fini_2level_ibuffer ( &rtab_mult_table );
  fini_2level_ibuffer ( &rtab2_mult_table );
#endif

  /* fini_rot_mat_table ( &rtab2 ); */

  /***********************************************************
   * finalize
   ***********************************************************/
#if 0
  rot_fini_rotation_matrix( &R );
  rot_fini_rotation_matrix( &A );
  rot_fini_rotation_matrix( &B );
#endif  /* of if 0 */

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_rot_mult_tab_lg] %s# [test_rot_mult_tab_lg] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [test_rot_mult_tab_lg] %s# [test_rot_mult_tab_lg] end of run\n", ctime(&g_the_time));
  }

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  return(0);

}
