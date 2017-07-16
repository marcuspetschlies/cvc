/****************************************************
 * piN2piN_correlators.cpp
 * 
 * Tue Jul 11 11:30:41 CEST 2017
 *
 * PURPOSE:
 *   originally copied from piN2piN_diagrams.cpp
 * TODO:
 * DONE:
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <time.h>
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
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "read_input_parser.h"
#include "matrix_init.h"
#include "contract_diagrams.h"
#include "aff_key_conversion.h"
#include "zm4x4.h"
#include "gamma.h"
#include "twopoint_function_utils.h"

using namespace cvc;

/***********************************************************
 * usage function
 ***********************************************************/
void usage() {
  EXIT(0);
}
  
  
/***********************************************************
 * main program
 ***********************************************************/
int main(int argc, char **argv) {
  
  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[200];
  char tag[20];
  int io_proc = -1;
  double ratime, retime;
  FILE *ofs = NULL;
#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  char * aff_status_str;
  struct AffNode_s *affn = NULL, *affdir = NULL;
  char aff_tag[200];
  // struct AffWriter_s *affw = NULL;
  // struct AffNode_s *affn2 = NULL;
#endif


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

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  read_input_parser(filename);

#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[piN2piN_correlators] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /******************************************************
   *
   ******************************************************/

  if(init_geometry() != 0) {
    fprintf(stderr, "[piN2piN_correlators] Error from init_geometry\n");
    EXIT(1);
  }
  geometry();

#ifdef HAVE_MPI
  /***********************************************
   * set io process
   ***********************************************/
  if( g_proc_coords[0] == 0 && g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
    io_proc = 2;
    fprintf(stdout, "# [piN2piN_factorized] proc%.4d tr%.4d is io process\n", g_cart_id, g_tr_id);
  } else {
    if( g_proc_coords[1] == 0 && g_proc_coords[2] == 0 && g_proc_coords[3] == 0) {
      io_proc = 1;
      fprintf(stdout, "# [piN2piN_factorized] proc%.4d tr%.4d is send process\n", g_cart_id, g_tr_id);
    } else {
      io_proc = 0;
    }
  }
#else
  io_proc = 2;
#endif

  /******************************************************
   * initialize gamma matrix algebra and several
   * gamma basis matrices
   ******************************************************/
  init_gamma_matrix ();

  /******************************************************
   * check source coords list
   ******************************************************/
  for ( int i = 0; i < g_source_location_number; i++ ) {
    g_source_coords_list[i][0] = ( g_source_coords_list[i][0] + T_global ) % T_global;
    g_source_coords_list[i][1] = ( g_source_coords_list[i][1] + LX_global ) % LX_global;
    g_source_coords_list[i][2] = ( g_source_coords_list[i][2] + LY_global ) % LY_global;
    g_source_coords_list[i][3] = ( g_source_coords_list[i][3] + LZ_global ) % LZ_global;
  }

  /******************************************************/
  /******************************************************/

  /* loop on source locations */
  for( int i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];

    if(io_proc == 2) {

      sprintf(filename, "%s.%.4d.tsrc%.2d", "piN_piN_correlator", Nconf, t_base );
      ofs = fopen( filename, "w");
      if( ofs == NULL ) {
        fprintf(stderr, "[piN2piN_correlators] Error from fopen \n");
        EXIT(4);
      }

      /* if( (affn2 = aff_writer_root( affw )) == NULL ) {
        fprintf(stderr, "[piN2piN_correlators] Error, aff writer is not initialized\n");
        EXIT(103);
      } */


    }  /* end of if io_proc == 2 */

    /*******************************************
     * loop on coherent source locations
     *******************************************/
    for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

      int source_proc_id, sx[4], gsx[4] = { t_coherent,
                    ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global,
                    ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global,
                    ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global };

      ratime = _GET_TIME;
      get_point_source_info (gsx, sx, &source_proc_id);

      /*******************************************/
      /*******************************************/

      /*******************************************
       * loop on 2-point functions
       *******************************************/
      for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {

        /* set the source coordinates if necessary */
        if ( g_twopoint_function_list[i2pt].source_coords[0] == -1 ) {
          g_twopoint_function_list[i2pt].source_coords[0] = gsx[0];
          g_twopoint_function_list[i2pt].source_coords[1] = gsx[1];
          g_twopoint_function_list[i2pt].source_coords[2] = gsx[2];
          g_twopoint_function_list[i2pt].source_coords[3] = gsx[3];
        }

        twopoint_function_print ( &(g_twopoint_function_list[i2pt]), "A2PT", stdout );

        /* if ( strcmp( g_twopoint_function_list[i2pt].type , "b-b") != 0 ) continue; */

        double _Complex *correlator = NULL, ***diagram = NULL, ***diagram_buffer = NULL;

        exitstatus= init_1level_zbuffer ( &correlator, T_global );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_correlators] Error from init_1level_zbuffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        exitstatus= init_3level_zbuffer ( &diagram, T_global, 4, 4 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_correlators] Error from init_3level_zbuffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        exitstatus= init_3level_zbuffer ( &diagram_buffer, T_global, 4, 4 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_correlators] Error from init_3level_zbuffer, status was %d\n", exitstatus);
          EXIT(47);
        }

        int pi1[3] = {
          g_twopoint_function_list[i2pt].pi1[0],
          g_twopoint_function_list[i2pt].pi1[1],
          g_twopoint_function_list[i2pt].pi1[2]
        };
        int pi2[3] = {
          g_twopoint_function_list[i2pt].pi2[0],
          g_twopoint_function_list[i2pt].pi2[1],
          g_twopoint_function_list[i2pt].pi2[2]
        };
        int pf1[3] = {
          g_twopoint_function_list[i2pt].pf1[0],
          g_twopoint_function_list[i2pt].pf1[1],
          g_twopoint_function_list[i2pt].pf1[2]
        };
        int pf2[3] = {
          g_twopoint_function_list[i2pt].pf2[0],
          g_twopoint_function_list[i2pt].pf2[1],
          g_twopoint_function_list[i2pt].pf2[2]
        };

        int gi1[2] = { g_twopoint_function_list[i2pt].gi1[0], g_twopoint_function_list[i2pt].gi1[1] };
        int gi2    = g_twopoint_function_list[i2pt].gi2;
  
        int gf1[2] = { g_twopoint_function_list[i2pt].gf1[0], g_twopoint_function_list[i2pt].gf1[1] };
        int gf2    = g_twopoint_function_list[i2pt].gf2;


        /* open file */
        char filename_prefix[100];
        twopoint_function_get_aff_filename_prefix ( filename_prefix, &(g_twopoint_function_list[i2pt]) );

        sprintf(filename, "%s.%.4d.tsrc%.2d.aff", filename_prefix, Nconf, t_base );

        fprintf(stdout, "# [piN2piN_correlators] reading data from file %s\n", filename);
        affr = aff_reader (filename);
        aff_status_str = (char*)aff_reader_errstr(affr);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[piN2piN_correlators] Error from aff_reader, status was %s\n", aff_status_str);
          EXIT(4);
        } else {
          fprintf(stdout, "# [piN2piN_correlators] reading data from aff file %s\n", filename);
        }

        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[piN2piN_correlators] Error, aff writer is not initialized\n");
          EXIT(103);
        }

        for ( int idiag = 0; idiag < g_twopoint_function_list[i2pt].n; idiag++ )
        // for ( int idiag = 0; idiag < 1; idiag++ )
        {

          // fprintf(stdout, "# [piN2piN_correlators] diagrams %d = %s\n", idiag, g_twopoint_function_list[i2pt].diagrams );

          if ( io_proc == 2 ) {
            char aff_tag[400];
            twopoint_function_print_diagram_key ( aff_tag, &(g_twopoint_function_list[i2pt]), idiag );

            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, diagram_buffer[0][0], T_global*16);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_correlators] Error from aff_node_get_complex, status was %d\n", exitstatus);
              EXIT(105);
            }

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
            for ( int it = 0; it < T_global; it++ ) {
              zm_pl_eq_zm_transposed_4x4_array ( diagram[it][0], diagram_buffer[it][0] );
            }

          }  /* end of if io_proc == 2 */
        }   /* end of loop on diagrams */

        /* TEST */
        /* for ( int it = 0; it < T; it++ ) {
          fprintf(stdout, "# initial correlator t = %2d\n", it );
          zm4x4_printf ( diagram[it], "c_i", stdout );
        } */

        /* reorder to absolute time */
        reorder_to_absolute_time (diagram, diagram, g_twopoint_function_list[i2pt].source_coords[0], g_twopoint_function_list[i2pt].reorder, T_global );


        /* source phase */
        exitstatus = correlator_add_source_phase ( diagram, pi1,  &(gsx[1]), T_global );

        /* TEST */
        /* for ( int it = 0; it < T; it++ ) {
          fprintf( stdout, "# with source phase t = %2d\n", it );
          zm4x4_printf ( diagram[it], "c_sp", stdout );
        } */
  
        /* boundary phase */
        exitstatus = correlator_add_baryon_boundary_phase ( diagram, gsx[0] );

        /* TEST */
        /* for ( int it = 0; it < T; it++ ) {
          fprintf( stdout, "# with boundary phase t = %2d\n", it );
          zm4x4_printf ( diagram[it], "c_sp", stdout );
        } */

        /* aplly gi1[1] and gf1[1] */
        gamma_matrix_type gf11, gi11;
        gamma_matrix_set ( &gi11,  gi1[1], 1. );
        gamma_matrix_set ( &gf11,  gf1[1], 1. );


#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
        for ( int it = 0; it < T_global; it++ ) {

          /* diagram_buffer <- diagram x Gamma_i1_1 */
          zm4x4_eq_zm4x4_ti_zm4x4 ( diagram_buffer[it], diagram[it],  gi11.m );
          /* diagram <- Gamma_f1_1 x diagram_buffer */
          zm4x4_eq_zm4x4_ti_zm4x4 ( diagram[it], gf11.m, diagram_buffer[it] );

          /* ??? spin / spin-parity projection ??? */
          correlator_spin_parity_projection ( diagram, diagram, (double)g_twopoint_function_list[i2pt].parity_project, T_global);


          /* spin-trace */
          co_eq_tr_zm4x4 ( correlator+it, diagram[it] );
        }

        /* multiply with phase factor per convention */
        twopoint_function_correlator_phase ( correlator, &(g_twopoint_function_list[i2pt]), T_global );


        /* write to file  */
        twopoint_function_print_correlator_data ( correlator,  &(g_twopoint_function_list[i2pt]), ofs );

        /* twopoint_function_print_correlator_key ( aff_tag, &(g_twopoint_function_list[i2pt]));
        fprintf(ofs, "# %s\n", aff_tag );
        for ( int it = 0; it < T_global; it++ ) {
          int ir = ( it + gsx[0] ) % T_global;
          fprintf(ofs, "  %25.16e %25.16e\n", creal(correlator[ir]), cimag(correlator[ir]));
        }*/

        fini_3level_zbuffer ( &diagram_buffer );
        fini_3level_zbuffer ( &diagram );
        fini_1level_zbuffer ( &correlator );
      }  /* end of loop on 2-point functions */

    }  /* end of loop on coherent source locations */

    if(io_proc == 2) {
      aff_reader_close (affr);

      /* aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from aff_writer_close, status was %s\n", aff_status_str);
        EXIT(111);
      } */
      fclose ( ofs );
    }  /* end of if io_proc == 2 */

  }  /* end of loop on base source locations */

  /*******************************************/
  /*******************************************/

#if 0
#endif  /* of if 0 */

  /*******************************************
   * finalize
   *******************************************/

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [piN2piN_correlators] %s# [piN2piN_correlators] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [piN2piN_correlators] %s# [piN2piN_correlators] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
