/****************************************************
 * contract_correlators.cpp
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

#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "read_input_parser.h"
#include "matrix_init.h"
#include "table_init_z.h"
#include "contract_diagrams.h"
#include "aff_key_conversion.h"
#include "zm4x4.h"
#include "gamma.h"
#include "twopoint_function_utils.h"

namespace cvc {

/***********************************************************
 * main program
 ***********************************************************/
int contract_correlator ( ) {
  
  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[200];
  double ratime, retime;
  FILE *ofs = NULL;
#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
#endif

  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  read_input_parser(filename);

#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[contract_correlators] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /***********************************************************
   * report git version
   ***********************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [contract_correlators] git version = %s\n", g_gitversion);
  }

  /***********************************************************
   * set geometry
   ***********************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0 ) {
    fprintf(stderr, "[contract_correlators] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }
  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  int const io_proc = get_io_proc ();

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * set number of timeslices
   ***********************************************************/
  int const nT = g_src_snk_time_separation + 1;
  if ( io_proc == 2 ) fprintf( stdout, "# [contract_correlators] number of timeslices (incl. src and snk) is %d\n", nT);


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
    g_source_coords_list[i][0] = ( g_source_coords_list[i][0] + T_global ) % T_global;
    g_source_coords_list[i][1] = ( g_source_coords_list[i][1] + LX_global ) % LX_global;
    g_source_coords_list[i][2] = ( g_source_coords_list[i][2] + LY_global ) % LY_global;
    g_source_coords_list[i][3] = ( g_source_coords_list[i][3] + LZ_global ) % LZ_global;
  }

  /******************************************************/
  /******************************************************/

  /******************************************************
   * loop on source locations
   ******************************************************/
  for( int i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];

    /******************************************************
     * loop on coherent source locations
     ******************************************************/
    for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

      int source_proc_id, sx[4], gsx[4] = { t_coherent,
                    ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global,
                    ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global,
                    ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global };


      get_point_source_info (gsx, sx, &source_proc_id);

      /******************************************************/
      /******************************************************/

      /******************************************************
       * open ASCII format output file
       ******************************************************/
      if(io_proc == 2) {
        sprintf(filename, "%s.%.4d.tsrc%.2d", "piN_piN_correlator", Nconf, t_coherent );
        ofs = fopen( filename, "w");
        if( ofs == NULL ) {
          fprintf(stderr, "[contract_correlators] Error from fopen %s %d\n", __FILE__, __LINE__ );
          EXIT(4);
        }
      }  // end of if io_proc == 2

      /******************************************************/
      /******************************************************/

      /******************************************************
       * loop on 2-point functions
       ******************************************************/
      for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {

        ratime = _GET_TIME;

        /******************************************************
         * set the source coordinates if necessary
         ******************************************************/
        // if ( g_twopoint_function_list[i2pt].source_coords[0] == -1 ) {
          g_twopoint_function_list[i2pt].source_coords[0] = gsx[0];
          g_twopoint_function_list[i2pt].source_coords[1] = gsx[1];
          g_twopoint_function_list[i2pt].source_coords[2] = gsx[2];
          g_twopoint_function_list[i2pt].source_coords[3] = gsx[3];
        // } 

        /******************************************************
         * print the 2-point function parameters
         ******************************************************/
        twopoint_function_print ( &(g_twopoint_function_list[i2pt]), "A2PT", stdout );

        /******************************************************
         * allocate correlator and diagrams
         ******************************************************/
        //double _Complex * correlator = init_1level_ztable ( nT );
        //if ( correlator == NULL ) {
        //  fprintf(stderr, "[contract_correlators] Error from init_1level_ztable %s %d\n", __FILE__, __LINE__ );
        //  EXIT(47);
        //}

        double _Complex *** diagram = init_3level_ztable ( nT, 4, 4 );
        if ( diagram == NULL ) {
          fprintf(stderr, "[contract_correlators] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
          EXIT(47);
        }

        /******************************************************
         * open AFF file
         ******************************************************/
        char filename_prefix[100];
        twopoint_function_get_aff_filename_prefix ( filename_prefix, &(g_twopoint_function_list[i2pt]) );

        sprintf(filename, "%s.%.4d.tsrc%.2d.aff", filename_prefix, Nconf, t_base );

        if ( io_proc == 2 ) {
          affr = aff_reader (filename);
          if ( const char * aff_status_str = aff_reader_errstr(affr) ) {
            fprintf(stderr, "[contract_correlators] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__ );
            EXIT(4);
          } else {
            fprintf(stdout, "# [contract_correlators] reading data from aff file %s\n", filename);
          }
        }

        /******************************************************
         * accumulate the diagrams for a twopoint_function
         ******************************************************/

        exitstatus = twopoint_function_accumulate_diagrams ( diagram, &(g_twopoint_function_list[i2pt]), nT, affr );
        if( exitstatus != 0 ) {
          fprintf(stderr, "[contract_correlators] Error from twopoint_function_accumulate_diagrams, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(4);
        }

        /******************************************************
         * reorder according to reorder entry in twopoint_function
         ******************************************************/
        if ( g_verbose > 4 && io_proc == 2 ) fprintf ( stdout, "# [contract_correlators] calling reorder_to_absolute_time %s %d\n", __FILE__, __LINE__ );
        reorder_to_absolute_time (diagram, diagram, g_twopoint_function_list[i2pt].source_coords[0], g_twopoint_function_list[i2pt].reorder, nT );


        /******************************************************
         * source phase
         ******************************************************/
        if ( g_verbose > 4 && io_proc == 2 ) fprintf ( stdout, "# [contract_correlators] calling correlator_add_source_phase  %s %d\n", __FILE__, __LINE__ );
        exitstatus = correlator_add_source_phase ( diagram, g_twopoint_function_list[i2pt].pi1,  &(gsx[1]), nT );


        /******************************************************
         * boundary phase
         ******************************************************/
        if ( g_verbose > 4 && io_proc == 2 ) fprintf ( stdout, "# [contract_correlators] calling correlator_add_baryon_boundary_phase  %s %d\n", __FILE__, __LINE__ );
        exitstatus = correlator_add_baryon_boundary_phase ( diagram, gsx[0], nT ); 


        /******************************************************
         * aplly gi1[1] and gf1[1]
         ******************************************************/
        if ( g_verbose > 4 && io_proc == 2 ) fprintf ( stdout, "# [contract_correlators] calling contract_diagram_zm4x4_field_mul_gamma_lr   %s %d\n", __FILE__, __LINE__ );
        exitstatus =  contract_diagram_zm4x4_field_mul_gamma_lr ( diagram, diagram, gamma[g_twopoint_function_list[i2pt].gf1[1]], gamma[g_twopoint_function_list[i2pt].gi1[1]], nT );


        /******************************************************
         * spin / spin-parity projection
         ******************************************************/
        if ( g_verbose > 4 && io_proc == 2 ) fprintf ( stdout, "# [contract_correlators] calling correlator_spin_parity_projection %s %d\n", __FILE__, __LINE__ );
        correlator_spin_parity_projection ( diagram, diagram, (double)g_twopoint_function_list[i2pt].parity_project, nT );
        //

        /******************************************************
         * multiply with phase factor per convention
         ******************************************************/
        // twopoint_function_correlator_phase ( correlator, &(g_twopoint_function_list[i2pt]), nT );

        if ( g_verbose > 4 && io_proc == 2 ) fprintf ( stdout, "# [contract_correlators] calling contract_diagram_zm4x4_field_ti_eq_co %s %d\n", __FILE__, __LINE__ );
        exitstatus = contract_diagram_zm4x4_field_ti_eq_co ( diagram, twopoint_function_get_correlator_phase ( &(g_twopoint_function_list[i2pt]) ), nT );

        /******************************************************
         * TEST
         *   print the correlator
         ******************************************************/
        if ( g_verbose > 5 ) {
          for ( int it = 0; it < nT; it++ ) {
            fprintf( stdout, "# with boundary phase t = %2d\n", it );
            zm4x4_printf ( diagram[it], "c_sp", stdout );
          }
        }


        /******************************************************
         * write to file
         ******************************************************/
        // twopoint_function_print_correlator_data ( correlator,  &(g_twopoint_function_list[i2pt]), ofs );

        char key[500];
        twopoint_function_print_correlator_key ( key, &(g_twopoint_function_list[i2pt]) );
        exitstatus = contract_diagram_write_fp ( diagram, ofs, key, 0, g_src_snk_time_separation, 0 );

        // twopoint_function_print_correlator_key ( key, &(g_twopoint_function_list[i2pt]));

        fini_3level_ztable ( &diagram );
        // fini_1level_ztable ( &correlator );

        /******************************************************/
        /******************************************************/

        /******************************************************
         * close AFF reader
         ******************************************************/
        if(io_proc == 2) { aff_reader_close (affr); }

        retime = _GET_TIME;
        if ( io_proc == 2 ) fprintf ( stdout, "# [contract_correlators] time for twopoint_function entry = %e seconds\n", retime-ratime );

      }  // end of loop on 2-point functions

      /******************************************************/
      /******************************************************/

      /******************************************************
       * close ofs file pointer
       ******************************************************/
      if(io_proc == 2) { fclose ( ofs ); }

    }  // end of loop on coherent source locations

  }  // end of loop on base source locations

  return(0);
}  // end of contract_correlator

}  // end of namespace cvc
