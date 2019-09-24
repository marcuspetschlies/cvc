/****************************************************
 * piN2piN_diagram_sum
 * 
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
}
#endif

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_timer.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "read_input_parser.h"
#include "matrix_init.h"
#include "table_init_z.h"
#include "table_init_2pt.h"
#include "contract_diagrams.h"
#include "aff_key_conversion.h"
#include "zm4x4.h"
#include "gamma.h"
#include "twopoint_function_utils.h"
#include "rotations.h"
#include "group_projection.h"
#include "little_group_projector_set.h"

#define MAX_UDLI_NUM 10000


using namespace cvc;

/***********************************************************
 * main program
 ***********************************************************/
int main(int argc, char **argv) {
 
#define _ZCOEFF_EPS 8.e-12

#if defined CUBIC_GROUP_DOUBLE_COVER
  char const little_group_list_filename[] = "little_groups_2Oh.tab";
  /* int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_double_cover; */
#elif defined CUBIC_GROUP_SINGLE_COVER
  const char little_group_list_filename[] = "little_groups_Oh.tab";
  /* int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_single_cover; */
#endif


  int c;
  int filename_set = 0;
  int exitstatus;
  int check_reference_rotation = 0;
  char filename[200];
  double ratime, retime;
  FILE *ofs = NULL;

  int udli_count = 0;
  char udli_list[MAX_UDLI_NUM][500];
  char udli_name[500];
  twopoint_function_type *udli_ptr[MAX_UDLI_NUM];

  struct timeval ta, tb;
  struct timeval start_time, end_time;

  /***********************************************************
   * set Cg basis projection coefficients
   ***********************************************************/
  double const Cgamma_basis_matching_coeff[16] = {
    1.00,  /*  0 =  Cgy        */
   -1.00,  /*  1 =  Cgzg5      */
   -1.00,  /*  2 =  Cg0        */
    1.00,  /*  3 =  Cgxg5      */
    1.00,  /*  4 =  Cgyg0      */
   -1.00,  /*  5 =  Cgyg5g0    */
    1.00,  /*  6 =  Cgyg5      */
   -1.00,  /*  7 =  Cgz        */
    1.00,  /*  8 =  Cg5g0      */
    1.00,  /*  9 =  Cgx        */
    1.00,  /* 10 =  Cgzg5g0    */
    1.00,  /* 11 =  C          */
   -1.00,  /* 12 =  Cgxg5g0    */
   -1.00,  /* 13 =  Cgxg0      */
    1.00,  /* 14 =  Cg5        */
    1.00   /* 15 =  Cgzg0      */
  };

  /***********************************************************
   * initialize MPI if used
   ***********************************************************/
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  /***********************************************************
   * evaluate command line arguments
   ***********************************************************/
  while ((c = getopt(argc, argv, "ch?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_reference_rotation = 1;
      fprintf ( stdout, "# [piN2piN_diagram_sum] check_reference_rotation set to %d\n", check_reference_rotation );
      break;
    case 'h':
    case '?':
    default:
      exit(1);
      break;
    }
  }

  /***********************************************************
   * timer for total time
   ***********************************************************/
  gettimeofday ( &start_time, (struct timezone *)NULL );

  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  read_input_parser(filename);

#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[piN2piN_diagram_sum] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /***********************************************************
   * package-own initialization of MPI parameters
   ***********************************************************/
  mpi_init(argc, argv);

  /***********************************************************
   * report git version
   ***********************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [piN2piN_diagram_sum] git version = %s\n", g_gitversion);
  }

  /***********************************************************
   * set geometry
   ***********************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_diagram_sum] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }
  geometry();


  /***********************************************************
   * set io process
   ***********************************************************/
  int const io_proc = get_io_proc ();

  /****************************************************
   * set cubic group single/double cover
   * rotation tables
   ****************************************************/
  rot_init_rotation_table();

  /***********************************************************
   * initialize gamma matrix algebra and several
   * gamma basis matrices
   ***********************************************************/
  init_gamma_matrix ();

  /******************************************************
   * set gamma matrices
   *   tmLQCD counting
   ******************************************************/
  gamma_matrix_type gamma[16];
  for ( int i = 0; i < 16; i++ ) {
    gamma_matrix_set ( &(gamma[i]), i, 1. );
  }


  /******************************************************
   * check source coords list
   ******************************************************/
  for ( int i = 0; i < g_source_location_number; i++ ) {
    g_source_coords_list[i][0] = ( g_source_coords_list[i][0] +  T_global ) %  T_global;
    g_source_coords_list[i][1] = ( g_source_coords_list[i][1] + LX_global ) % LX_global;
    g_source_coords_list[i][2] = ( g_source_coords_list[i][2] + LY_global ) % LY_global;
    g_source_coords_list[i][3] = ( g_source_coords_list[i][3] + LZ_global ) % LZ_global;
  }

  /******************************************************
   * loop on source locations
   ******************************************************/
  for( int i_src = 0; i_src<g_source_location_number; i_src++) {

    int const t_base = g_source_coords_list[i_src][0];

    /******************************************************
     * loop on coherent source locations
     ******************************************************/
    for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      int const t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

      int source_proc_id, sx[4];
      int const gsx[4] = { t_coherent,
                    ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global,
                    ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global,
                    ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global };


      get_point_source_info (gsx, sx, &source_proc_id);

      /******************************************************
       * loop on 2-point functions
       ******************************************************/
      for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {

        if ( g_verbose > 4 ) {
          gettimeofday ( &ta, (struct timezone *)NULL );
          /******************************************************
           * print the 2-point function parameters
           ******************************************************/
          sprintf ( filename, "twopoint_function_%d.show", i2pt );
          if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
            fprintf ( stderr, "[piN2piN_diagram_sum] Error from fopen %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }
          twopoint_function_print ( &(g_twopoint_function_list[i2pt]), "TWPT", ofs );
          fclose ( ofs );

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "piN2piN_diagram_sum", "print-twopoint", io_proc == 2 );

        }  /* end of if g_verbose */

        /****************************************************
         * read little group parameters
         ****************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );

        little_group_type little_group;
        if ( ( exitstatus = little_group_read ( &little_group, g_twopoint_function_list[i2pt].group, little_group_list_filename ) ) != 0 ) {
          fprintf ( stderr, "[piN2piN_diagram_sum] Error from little_group_read, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(2);
        }

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum", "little_group_read", io_proc == 2 );

        if ( g_verbose > 4 ) {
          sprintf ( filename, "little_group_%d.show", i2pt );
          if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
            fprintf ( stderr, "[piN2piN_diagram_sum] Error from fopen %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }
          little_group_show ( &little_group, ofs, 1 );
          fclose ( ofs );
        }

        /******************************************************
         * set current source coords in 2pt function
         ******************************************************/
        g_twopoint_function_list[i2pt].source_coords[0] = gsx[0];
        g_twopoint_function_list[i2pt].source_coords[1] = gsx[1];
        g_twopoint_function_list[i2pt].source_coords[2] = gsx[2];
        g_twopoint_function_list[i2pt].source_coords[3] = gsx[3];

        /******************************************************
         * this is a temporary twopoint function struct to
         * store one term in the projection sum
         ******************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );

        twopoint_function_type tp;

        twopoint_function_init ( &tp );

        twopoint_function_copy ( &tp, &( g_twopoint_function_list[i2pt] ), 1 );

        if ( twopoint_function_allocate ( &tp ) == NULL ) {
          fprintf ( stderr, "[piN2piN_diagram_sum] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
          EXIT(123);
        }

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum", "init-copy-allocate-tp", io_proc == 2 );

        /******************************************************/
        /******************************************************/

        /******************************************************
         * fill the diagram with data
         ******************************************************/
        for ( int ids = 0; ids < tp.n; ids++ ) {
          if ( ( exitstatus = twopoint_function_data_location_identifier ( udli_name, &tp, filename_prefix, ids, "#" ) ) != 0 ) {
            fprintf ( stderr, "[piN2piN_diagram_sum] Error from twopoint_function_data_location_identifier, status was %d %s %d\n",
                exitstatus, __FILE__, __LINE__ );
            EXIT(212);
          }

          /******************************************************
           * check, whether udli_name exists in udli list
           ******************************************************/
          gettimeofday ( &ta, (struct timezone *)NULL );

          int udli_id = -1;
          for ( int i = 0; i < udli_count; i++ ) {
            if ( strcmp ( udli_name, udli_list[i] ) == 0 ) {
              udli_id  = i;
              break;
            }
          }

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "piN2piN_diagram_sum", "check-udli-entry", io_proc == 2 );

          if ( udli_id == -1 ) {
            fprintf ( stdout, "# [piN2piN_diagram_sum] could not find udli_name %s in udli_list\n", udli_name );

            gettimeofday ( &ta, (struct timezone *)NULL );

            /******************************************************
             * start new entry in udli list
             ******************************************************/

            /******************************************************
             * check, that number udlis is not exceeded
             ******************************************************/
            if ( udli_count == MAX_UDLI_NUM ) {
              fprintf ( stderr, "[piN2piN_diagram_sum] Error, maximal number of udli exceeded\n" );
              EXIT(111);
            } else {
              if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagram_sum] starting udli entry number %d\n", udli_count );
            }

            udli_ptr[udli_count] = ( twopoint_function_type *)malloc ( sizeof ( twopoint_function_type ) );
            if ( udli_ptr[udli_count] == NULL ) {
              fprintf ( stderr, "[piN2piN_diagram_sum] Error from malloc %s %d\n", __FILE__, __LINE__ );
              EXIT(211);
            }

            twopoint_function_init ( udli_ptr[udli_count] );

            twopoint_function_copy ( udli_ptr[udli_count], &tp, 0 );

            udli_ptr[udli_count]->n = 1;

            strcpy ( udli_ptr[udli_count]->name, udli_name );

            twopoint_function_allocate ( udli_ptr[udli_count] );

            /******************************************************
             * fill data from udli
             ******************************************************/
            if ( ( exitstatus = twopoint_function_fill_data_from_udli ( udli_ptr[udli_count] , udli_name , io_proc) ) != 0 ) {
              fprintf ( stderr, "[piN2piN_diagram_sum] Error from twopoint_function_fill_data_from_udli, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(212);
            }

            /******************************************************
             * set udli_id on new entry
             ******************************************************/
            udli_id = udli_count;

            /******************************************************
             * set entry in udli_list
             ******************************************************/
            strcpy ( udli_list[udli_id], udli_name );

            /******************************************************
             * count new entry
             ******************************************************/
            udli_count++;

            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "piN2piN_diagram_sum", "add-udli-entry", io_proc == 2 );

          } else {
            if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagram_sum] udli_name %s matches udli_list[%d] %s\n", udli_name, udli_id, udli_list[udli_id] );
          }

          /******************************************************
           * copy data from udli_id entry to current tp
           ******************************************************/
          memcpy ( tp.c[ids][0][0], udli_ptr[udli_id]->c[0][0][0], tp.T * tp.d * tp.d * sizeof(double _Complex ) );

        }  /* end of loop on data sets = diagrams */

        /******************************************************
         * apply diagram norm
         *
         * a little overhead, since this done for each tp,
         * not each udli_ptr only
         ******************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );
        if ( ( exitstatus = twopoint_function_apply_diagram_norm ( &tp ) ) != 0 ) {
          fprintf ( stderr, "[piN2piN_diagram_sum] Error from twopoint_function_apply_diagram_norm, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(213);
        }
        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum", "twopoint_function_apply_diagram_norm", io_proc == 2 );


        /******************************************************
         * sum up data sets in tp
         * - add data sets 1,...,tp.n-1 to data set 0
         ******************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );
        if ( ( exitstatus = twopoint_function_accum_diagrams ( tp.c[0], &tp ) ) != 0 ) {
          fprintf ( stderr, "[piN2piN_diagram_sum] Error from twopoint_function_accum_diagrams, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(216);
        }

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum", "twopoint_function_accum_diagrams", io_proc == 2 );

        /******************************************************
         * output of tp_project
         *
         * loop over individiual projection variants
         ******************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );

        for ( int itp = 0; itp < n_tp_project; itp++ ) {

          twopoint_function_type * tp_project_ptr = tp_project[0][0][0];

          /******************************************************
           * multiply group projection normalization
           ******************************************************/
          /* No, not this part, 
           * twopoint_function_get_correlator_phase ( &(tp_project_ptr[itp]) )
           *
           * this has been added in piN2piN_diagrams_complete via
           * via factor in zsign from function contract_diagram_get_correlator_phase */

          /* ztmp given by ( irrep_dim / number of little group members )^2
           *   factor of 4 because ...->n is the number of proper rotations only,
           *   so half the number group elements
           */



          double _Complex const ztmp = (double)( projector.rtarget->dim * projector.rtarget->dim ) / \
                                       ( 4. *    projector.rtarget->n   * projector.rtarget->n   );

          if ( g_verbose > 4 ) fprintf ( stdout, "# [piN2piN_diagram_sum] correlator norm = %25.16e %25.16e\n", creal( ztmp ), cimag( ztmp ) );

          exitstatus = contract_diagram_zm4x4_field_ti_eq_co ( tp_project_ptr[itp].c[0], ztmp, tp_project_ptr[itp].T );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[piN2piN_diagram_sum] Error from contract_diagram_zm4x4_field_ti_eq_co %s %d\n", __FILE__, __LINE__ );
            EXIT(217)
          }
 
          /******************************************************
           * write to disk
           ******************************************************/
          exitstatus = twopoint_function_write_data ( &( tp_project_ptr[itp] ) );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[piN2piN_diagram_sum] Error from twopoint_function_write_data, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(12);
          }

        }  /* end of loop on n_tp_project */

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum", "group-projection-norm-output", io_proc == 2 );

        /******************************************************
         * deallocate twopoint_function vars tp and tp_project
         ******************************************************/
        twopoint_function_fini ( &tp );
        for ( int i = 0; i < n_tp_project; i++ ) {
          twopoint_function_fini ( &(tp_project[0][0][0][i]) );
        }
        fini_4level_2pttable ( &tp_project );

      }  // end of loop on g_twopoint_function_number 2pt functions

      /******************************************************
       * reset udli_count and deallocate udli_ptr
       *
       * NOTE: too much memory ? free udli content
       *       ONLY after loop on 2-point functions
       *
       * new-readin definitely after change of total
       * momentum
       ******************************************************/
      for ( int i = 0; i < udli_count; i++ ) {
        twopoint_function_fini ( udli_ptr[i] );
        free ( udli_ptr[i] );
        udli_ptr[i] = NULL;
      }
      udli_count = 0;

    }  // end of loop on coherent source locations

  }  // end of loop on base source locations

  /******************************************************/
  /******************************************************/

  /******************************************************
   * finalize
   *
   * free the allocated memory, finalize
   ******************************************************/
  free_geometry();

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "piN2piN_diagram_sum", "total-time", io_proc == 2 );

  
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [piN2piN_diagram_sum] %s# [piN2piN_diagram_sum] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [piN2piN_diagram_sum] %s# [piN2piN_diagram_sum] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
