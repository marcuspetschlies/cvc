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
 * give reader id depending on diagram type 
 ***********************************************************/
static inline int diagram_name_to_reader_id ( char * name ) {
  char c = name[0];
  switch (c) {
    case 'm':  /* M-type  */
    case 'n':  /* N-type */
    case 'd':  /* D-type */
    case 't':  /* T-type, triangle */
    case 'b':  /* B-type  */
      return(0);
      break;
    case 'w':  /* W-type  */
      return(1);
      break;
    case 'z':  /* Z-type  */
      return(2);
      break;
    case 's':  /* S-type  */
      return(3);
      break;
    default:
      return(-1);
      break;
  }
  return(-1);
}  /* end of diagram_name_to_reader_id */


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


  char const diagram_name_list[5][20] = { "N-N", "D-D", "pixN-D", "pixN-pixN", "m-m" };
  int const diagram_name_number = 5;


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
   * loop on reference moving frames
   ******************************************************/
  for ( int ipref = 0; ipref < g_total_momentum_number; ipref++ ) {

    /******************************************************
     * loop diagram names
     ******************************************************/
    for ( int iname = 0; iname < diagram_name_number; iname++ ) {

      /******************************************************
       * open AFF files for name and pref
       * we know which here
       ******************************************************/


      /******************************************************
       * loop on 2-point functions
       ******************************************************/
      for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {

        if ( strcmp ( g_twopoint_function_list[i2pt].name , diagram_name_list[iname] ) != 0 ) {
          if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagram_sum] skip twopoint %6d %s %d\n", i2pt, __FILE__, __LINE__ );
          continue;
        }

        int ptot[3];
        if (   ( strcmp ( g_twopoint_function_list[i2pt].type , "b-b" ) == 0 ) 
            || ( strcmp ( g_twopoint_function_list[i2pt].type , "m-m" ) == 0 ) 
            || ( strcmp ( g_twopoint_function_list[i2pt].type , "mxb-m" ) == 0 ) ) {

          ptot[0] = g_twopoint_function_list[i2pt].pf1[0];
          ptot[1] = g_twopoint_function_list[i2pt].pf1[1];
          ptot[2] = g_twopoint_function_list[i2pt].pf1[2];

        } else if ( strcmp ( g_twopoint_function_list[i2pt].type , "mxb-mxb" ) == 0 ) {
          ptot[0] = g_twopoint_function_list[i2pt].pf1[0] + g_twopoint_function_list[i2pt].pf2[0];
          ptot[1] = g_twopoint_function_list[i2pt].pf1[1] + g_twopoint_function_list[i2pt].pf2[1];
          ptot[2] = g_twopoint_function_list[i2pt].pf1[2] + g_twopoint_function_list[i2pt].pf2[2];
        }

        int pref[3], &refframerot;
        exitstatus = get_reference_rotation ( pref, &refframerot, ptot );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[piN2piN_diagram_sum] Error from get_reference_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(4);
        } else if ( g_verbose > 1 ) {
          fprintf ( stdout, "# [piN2piN_diagram_sum] twopoint_function %3d ptot = %3d %3d %3d refframerot %2d for Pref = %3d %3d %3d\n", i2pt,
              ptot[0], ptot[1], ptot[2], refframerot, pref[0], pref[1], pref[2]);
        }

        if ( ( pref[0] != g_total_momentum_list[ipref][0] ) 
          || ( pref[1] != g_total_momentum_list[ipref][1] ) 
          || ( pref[2] != g_total_momentum_list[ipref][2] )  ) {
            if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagram_sum] skip twopoint %6d %s %d\n", i2pt, __FILE__, __LINE__ );
            continue;
        }

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

            int const gsx[4] = { t_coherent,
                        ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global,
                        ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global,
                        ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global };


            /******************************************************
             * loop on diagram in twopt
             ******************************************************/
            for ( int idiag 0; idiag < g_twopoint_function_list[i2pt].n ; idiag++ ) {

              char diagram_name[10];
              twopoint_function_get_diagram_name ( diagram_name, &(g_twopoint_function_list[i2pt]), idiag );

              int const affr_id = diagram_name_to_reader_id ( diagram_name );
              if ( affr_id == -1 ) {
                fprintf ( stderr, "[piN2piN_diagram_sum] Error from diagram_name_to_reader_id %s %d\n", __FILE__, __LINE__ );
                EXIT(127);
              }

              /******************************************************
               * read the twopoint function diagram items
               *
               * get which aff reader from diagram name
               ******************************************************/
              char diagram_key[500];
              int const nc = g_twopoint_function_list[i2pt].d * g_twopoint_function_list[i2pt].d;

              exitstatus = read_aff_contraction ( contr, affr[affr_id], NULL, diagram_key, nc );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[piN2piN_diagram_sum] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(129);
              }

            }  /* end of loop on diagrams */

          }  /* end of loop on coherent source locations */

        }  /* end of loop on base source locations */

        /******************************************************
         * average over source locations
         ******************************************************/

        /******************************************************
         * apply diagram norm
         ******************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );
        if ( ( exitstatus = twopoint_function_apply_diagram_norm ( &tp ) ) != 0 ) {
          fprintf ( stderr, "[piN2piN_diagram_sum] Error from twopoint_function_apply_diagram_norm, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(213);
        }
        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum", "twopoint_function_apply_diagram_norm", io_proc == 2 );


        /******************************************************
         * add up diagrams
         ******************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );
        if ( ( exitstatus = twopoint_function_accum_diagrams ( tp.c[0], &tp ) ) != 0 ) {
          fprintf ( stderr, "[piN2piN_diagram_sum] Error from twopoint_function_accum_diagrams, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(216);
        }
        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum", "twopoint_function_accum_diagrams", io_proc == 2 );

        /******************************************************
         * write to h5 file
         ******************************************************/
        exitstatus = twopoint_function_write_data ( &( tp_project_ptr[itp] ) );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[piN2piN_diagram_sum] Error from twopoint_function_write_data, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(12);
        }

      }  /* end of loop on 2-point functions */

      /******************************************************
       * close AFF readers
       ******************************************************/
      for ( int ir = 0; ir < affr_num; ir++ ) {
        aff_reader_close ( affr[ir] );
        affr[ir] = NULL;
      }

    }  /* end of loop on twopoint function names */

  }  /* end of loop on reference moving frames */

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


            twopoint_function_init ( udli_ptr[udli_count] );

            twopoint_function_copy ( udli_ptr[udli_count], &tp, 0 );

            udli_ptr[udli_count]->n = 1;

            strcpy ( udli_ptr[udli_count]->name, udli_name );

            twopoint_function_allocate ( udli_ptr[udli_count] );






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
