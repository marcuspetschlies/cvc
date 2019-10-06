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
#include "contractions_io.h"
#include "table_init_i.h"
#include "table_init_z.h"
#include "table_init_2pt.h"
#include "contract_diagrams.h"
#include "zm4x4.h"
#include "gamma.h"
#include "twopoint_function_utils.h"
#include "rotations.h"
#include "group_projection.h"

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


  char const twopt_name_list[5][20] = { "N-N", "D-D", "pixN-D", "pixN-pixN", "pi-pi" };
  int const twopt_name_number = 5;


  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[200];
  FILE *ofs = NULL;

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
  while ((c = getopt(argc, argv, "h?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
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

  /****************************************************
   * set cubic group single/double cover
   * rotation tables
   ****************************************************/
  rot_init_rotation_table();

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
   * total number of source locations, 
   * base x coherent
   *
   * fill a list of all source coordinates
   ******************************************************/
  int const source_location_number = g_source_location_number * g_coherent_source_number;
  int ** source_coords_list = init_2level_itable ( source_location_number, 4 );
  if( source_coords_list == NULL ) {
    fprintf ( stderr, "[] Error from init_2level_itable %s %d\n", __FILE__, __LINE__ );
    EXIT( 43 );
  }
  for ( int ib = 0; ib < g_source_location_number; ib++ ) {
    g_source_coords_list[ib][0] = ( g_source_coords_list[ib][0] +  T_global ) %  T_global;
    g_source_coords_list[ib][1] = ( g_source_coords_list[ib][1] + LX_global ) % LX_global;
    g_source_coords_list[ib][2] = ( g_source_coords_list[ib][2] + LY_global ) % LY_global;
    g_source_coords_list[ib][3] = ( g_source_coords_list[ib][3] + LZ_global ) % LZ_global;

    int const t_base = g_source_coords_list[ib][0];
    
    for ( int ic = 0; ic < g_coherent_source_number; ic++ ) {
      int const ibc = ib * g_coherent_source_number + ic;

      int const t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * ic ) % T_global;
      source_coords_list[ibc][0] = t_coherent;
      source_coords_list[ibc][1] = ( g_source_coords_list[ib][1] + (LX_global/2) * ic ) % LX_global;
      source_coords_list[ibc][2] = ( g_source_coords_list[ib][2] + (LY_global/2) * ic ) % LY_global;
      source_coords_list[ibc][3] = ( g_source_coords_list[ib][3] + (LZ_global/2) * ic ) % LZ_global;
    }
  }


  /******************************************************
   * loop on reference moving frames
   ******************************************************/
  for ( int ipref = 0; ipref < g_total_momentum_number; ipref++ ) {

    /******************************************************
     * loop diagram names
     ******************************************************/
    for ( int iname = 0; iname < twopt_name_number; iname++ ) {

      gettimeofday ( &ta, (struct timezone *)NULL );

      /******************************************************
       * AFF readers
       *
       * open AFF files for name and pref
       * we know which here
       * 12 is some upper limit
       ******************************************************/
      struct AffReader_s *** affr = NULL;
      int affr_diag_tag_num = 0;
      char affr_diag_tag_list[12];

      if (        strcmp( twopt_name_list[iname], "N-N"       ) == 0 )  {
        affr_diag_tag_num = 1;
        affr_diag_tag_list[0] = 'n';
      } else if ( strcmp( twopt_name_list[iname], "D-D"       ) == 0 )  {
        affr_diag_tag_num = 1;
        affr_diag_tag_list[0] = 'd';
      } else if ( strcmp( twopt_name_list[iname], "pixN-D"    ) == 0 )  {
        affr_diag_tag_num = 1;
        affr_diag_tag_list[0] = 't';
      } else if ( strcmp( twopt_name_list[iname], "pixN-pixN" ) == 0 )  {
        affr_diag_tag_num = 4;
        affr_diag_tag_list[0] = 'b';
        affr_diag_tag_list[1] = 'w';
        affr_diag_tag_list[2] = 'z';
        affr_diag_tag_list[3] = 's';
      } else if ( strcmp( twopt_name_list[iname], "pi-pi"     ) == 0 )  {
        affr_diag_tag_num = 1;
        affr_diag_tag_list[0] = 'm';
      } else {
        fprintf( stderr, "[piN2piN_diagram_sum] Error, unrecognized twopt name %s %s %d\n", twopt_name_list[iname], __FILE__, __LINE__ );
        EXIT(123);
      }

      /* total number of readers */
      int const affr_num = source_location_number * affr_diag_tag_num;
      affr = (struct AffReader_s *** )malloc ( affr_diag_tag_num * sizeof ( struct AffReader_s ** )) ;
      if ( affr == NULL ) {
        fprintf( stderr, "[piN2piN_diagram_sum] Error from malloc %s %d\n", __FILE__, __LINE__ );
        EXIT(124);
      }
      affr[0] = (struct AffReader_s ** )malloc ( affr_num * sizeof ( struct AffReader_s *)) ;
      if ( affr[0] == NULL ) {
        fprintf( stderr, "[piN2piN_diagram_sum] Error from malloc %s %d\n", __FILE__, __LINE__ );
        EXIT(124);
      }
      for( int i = 1; i< affr_diag_tag_num; i++ ) {
        affr[i] = affr[i-1] + source_location_number;
      }
      for ( int i = 0 ; i < affr_diag_tag_num; i++ ) {
        for ( int k = 0; k < source_location_number; k++ ) {
          sprintf ( filename, "%s.%c.PX%dPY%dPZ%d.%.4d.t%dx%dy%dz%d.aff", filename_prefix, affr_diag_tag_list[i],
              g_total_momentum_list[ipref][0], g_total_momentum_list[ipref][1], g_total_momentum_list[ipref][2],
              Nconf,
              source_coords_list[k][0], source_coords_list[k][1], source_coords_list[k][2], source_coords_list[k][3] );

          affr[i][k] = aff_reader ( filename );
          if ( const char * aff_status_str =  aff_reader_errstr ( affr[i][k] ) ) {
            fprintf(stderr, "[piN2piN_diagram_sum] Error from aff_reader for filename %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
            EXIT(45);
          } else {
            if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagram_sum] opened data file %s for reading %s %d\n", filename, __FILE__, __LINE__);
          }
        }
      }

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "piN2piN_diagram_sum", "open-reader-list", io_proc == 2 );

      /******************************************************
       * AFF writer
       ******************************************************/

      gettimeofday ( &ta, (struct timezone *)NULL );

      sprintf ( filename, "%s.PX%d_PY%d_PZ%d.aff", twopt_name_list[iname], 
          g_total_momentum_list[ipref][0], g_total_momentum_list[ipref][1], g_total_momentum_list[ipref][2] );
      
      struct AffWriter_s * affw = aff_writer(filename);
      if ( const char * aff_status_str = aff_writer_errstr ( affw ) ) {
        fprintf(stderr, "[piN2piN_diagram_sum] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(48);
      }

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "piN2piN_diagram_sum", "open-writer", io_proc == 2 );

      /******************************************************/
      /******************************************************/

      /******************************************************
       * loop on 2-point functions
       ******************************************************/
      for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {

        gettimeofday ( &ta, (struct timezone *)NULL );

        if ( strcmp ( g_twopoint_function_list[i2pt].name , twopt_name_list[iname] ) != 0 ) {
          if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagram_sum] skip twopoint %6d %s %d\n", i2pt, __FILE__, __LINE__ );
          continue;
        }

        int ptot[3];
        if (   ( strcmp ( g_twopoint_function_list[i2pt].type , "b-b" ) == 0 ) 
            || ( strcmp ( g_twopoint_function_list[i2pt].type , "m-m" ) == 0 ) 
            || ( strcmp ( g_twopoint_function_list[i2pt].type , "mxb-b" ) == 0 ) ) {

          ptot[0] = g_twopoint_function_list[i2pt].pf1[0];
          ptot[1] = g_twopoint_function_list[i2pt].pf1[1];
          ptot[2] = g_twopoint_function_list[i2pt].pf1[2];

        } else if ( strcmp ( g_twopoint_function_list[i2pt].type , "mxb-mxb" ) == 0 ) {
          ptot[0] = g_twopoint_function_list[i2pt].pf1[0] + g_twopoint_function_list[i2pt].pf2[0];
          ptot[1] = g_twopoint_function_list[i2pt].pf1[1] + g_twopoint_function_list[i2pt].pf2[1];
          ptot[2] = g_twopoint_function_list[i2pt].pf1[2] + g_twopoint_function_list[i2pt].pf2[2];
        }

        int pref[3], refframerot;
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

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum", "check-valid-twopt", io_proc == 2 );

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
         * allocate tp_sum
         ******************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );

        twopoint_function_type tp_sum;
        twopoint_function_type * tp = init_1level_2pttable ( source_location_number );

        twopoint_function_init ( &tp_sum );
        twopoint_function_copy ( &tp_sum, &( g_twopoint_function_list[i2pt]), 0 );
          
        if ( twopoint_function_allocate ( &tp_sum) == NULL ) {
          fprintf ( stderr, "[piN2piN_diagram_sum] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
          EXIT(131);
        }

        for ( int i = 0; i < source_location_number; i++ ) {
          twopoint_function_init ( &(tp[i]) );
          twopoint_function_copy ( &(tp[i]), &( g_twopoint_function_list[i2pt]), 0 );
          if ( twopoint_function_allocate ( &(tp[i]) ) == NULL ) {
            fprintf ( stderr, "[piN2piN_diagram_sum] Error from twopoint_function_allocate %s %d\n", __FILE__, __LINE__ );
            EXIT(125);
          }
          memcpy ( tp[i].source_coords , source_coords_list[i], 4 * sizeof ( int ) );
        }

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum", "init-copy-allocate-twopt", io_proc == 2 );

        /******************************************************
         * loop on diagram in twopt
         ******************************************************/
        for ( int idiag = 0; idiag < g_twopoint_function_list[i2pt].n ; idiag++ ) {

          char diagram_name[10];
          twopoint_function_get_diagram_name ( diagram_name, &tp_sum, idiag );

          int const affr_diag_id = diagram_name_to_reader_id ( diagram_name );
          if ( affr_diag_id == -1 ) {
            fprintf ( stderr, "[piN2piN_diagram_sum] Error from diagram_name_to_reader_id %s %d\n", __FILE__, __LINE__ );
            EXIT(127);
          }

          /******************************************************
           * loop on source locations
           ******************************************************/
 
          gettimeofday ( &ta, (struct timezone *)NULL );

          for( int isrc = 0; isrc < source_location_number; isrc++) {

            /******************************************************
             * read the twopoint function diagram items
             *
             * get which aff reader from diagram name
             ******************************************************/
            char key[500];
            char key_suffix[400];
            unsigned int const nc = tp_sum.d * tp_sum.d * tp_sum.T;

            exitstatus = contract_diagram_key_suffix_from_type ( key_suffix, &(tp[isrc]) );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[piN2piN_diagram_sum] Error from contract_diagram_key_suffix_from_type, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(1);
            }

            sprintf( key, "/%s/%s/%s/%s", tp[isrc].name, diagram_name, tp[isrc].fbwd, key_suffix );
            if ( g_verbose > 3 ) fprintf ( stdout, "# [piN2piN_diagram_sum] key = %s %s %d\n", key, __FILE__, __LINE__ );

            exitstatus = read_aff_contraction ( tp[isrc].c[idiag][0][0], affr[affr_diag_id][isrc], NULL, key, nc );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[piN2piN_diagram_sum] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(129);
            }

          }  /* end of loop on source locations */
        
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "piN2piN_diagram_sum", "read-diagram-all-src", io_proc == 2 );

        }  /* end of loop on diagrams */

        /******************************************************
         * average over source locations
         ******************************************************/
        for( int isrc = 0; isrc < source_location_number; isrc++) {

          double const norm = 1. / (double)source_location_number;
#pragma omp parallel for
          for ( int i = 0; i < tp_sum.n * tp_sum.d * tp_sum.d * tp_sum.T; i++ ) {
            tp_sum.c[0][0][0][i] += tp[isrc].c[0][0][0][i] * norm;
          }
        }

        /******************************************************
         * apply diagram norm
         ******************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );
        
        if ( ( exitstatus = twopoint_function_apply_diagram_norm ( &tp_sum ) ) != 0 ) {
          fprintf ( stderr, "[piN2piN_diagram_sum] Error from twopoint_function_apply_diagram_norm, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(213);
        }

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum", "twopoint_function_apply_diagram_norm", io_proc == 2 );


        /******************************************************
         * add up diagrams
         ******************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );

        if ( ( exitstatus = twopoint_function_accum_diagrams ( tp_sum.c[0], &tp_sum ) ) != 0 ) {
          fprintf ( stderr, "[piN2piN_diagram_sum] Error from twopoint_function_accum_diagrams, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(216);
        }

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum", "twopoint_function_accum_diagrams", io_proc == 2 );

        /******************************************************
         * write to h5 file
         ******************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );
        
        char key[500], key_suffix[400];

        /* key suffix */
#if 0
        exitstatus = contract_diagram_key_suffix ( key_suffix, tp_sum.gf2, tp_sum.pf2, tp_sum.gf1[0], tp_sum.gf1[1], tp_sum.pf1, tp_sum.gi2, tp_sum.pi2, tp_sum.gi1[0], tp_sum.gi1[1], tp_sum.pi1, NULL); */
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[piN2piN_diagram_sum] Error from contract_diagram_key_suffix, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(12);
        }
#endif  /* of if 0 */

        exitstatus = contract_diagram_key_suffix_from_type ( key_suffix, &tp_sum );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[piN2piN_diagram_sum] Error from contract_diagram_key_suffix_from_type, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(12);
        }


        /* full key */
        sprintf( key, "/%s/%s%s", tp_sum.name, tp_sum.fbwd, key_suffix );
        if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagram_sum] key = %s %s %d\n", key, __FILE__, __LINE__ );

        unsigned int const nc = tp_sum.d * tp_sum.d * tp_sum.T;
        exitstatus = write_aff_contraction ( tp_sum.c[0][0][0], affw, NULL, key, nc);
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[piN2piN_diagram_sum] Error from write_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(12);
        }
        
        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "piN2piN_diagram_sum", "write-key-to-file", io_proc == 2 );

        /******************************************************
         * deallocate tp_sum and tp list
         ******************************************************/
        for ( int i = 0; i < source_location_number; i++ ) {
          twopoint_function_fini ( &( tp[i] ) );
        }
        fini_1level_2pttable ( &tp );
        twopoint_function_fini ( &tp_sum );

      }  /* end of loop on 2-point functions */

      /******************************************************
       * close AFF readers
       ******************************************************/
      for ( int ir = 0; ir < affr_num; ir++ ) {
        aff_reader_close ( affr[0][ir] );
      }

      if ( const char * aff_status_str = aff_writer_close ( affw ) ) {
        fprintf(stderr, "[piN2piN_diagram_sum] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(46);
      }

    }  /* end of loop on twopoint function names */

  }  /* end of loop on reference moving frames */

  /******************************************************/
  /******************************************************/

  /******************************************************
   * finalize
   *
   * free the allocated memory, finalize
   ******************************************************/
  free_geometry();
  fini_2level_itable ( &source_coords_list  );

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
