/****************************************************
 * piN2piN_projection.cpp
 * 
 * Fr 27. Jul 16:46:24 CEST 2018
 *
 * PURPOSE:
 *   originally copied from piN2piN_correlators.cpp
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
#include "table_init_z.h"
#include "contract_diagrams.h"
#include "aff_key_conversion.h"
#include "zm4x4.h"
#include "gamma.h"
#include "twopoint_function_utils.h"
#include "rotations.h"
#include "group_projection.h"
#include "little_group_projector_set.h"


using namespace cvc;

/***********************************************************
 * main program
 ***********************************************************/
int main(int argc, char **argv) {
 
#if defined CUBIC_GROUP_DOUBLE_COVER
  char const little_group_list_filename[] = "little_groups_2Oh.tab";
  int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_double_cover;
#elif defined CUBIC_GROUP_SINGLE_COVER
  const char little_group_list_filename[] = "little_groups_Oh.tab";
  int (* const set_rot_mat_table ) ( rot_mat_table_type*, const char*, const char*) = set_rot_mat_table_cubic_group_single_cover;
#endif


  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[200];
  double ratime, retime;
  FILE *ofs = NULL;
#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
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
      exit(1);
      break;
    }
  }

  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "cvc.input");
  read_input_parser(filename);

#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[piN2piN_projection] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /***********************************************************
   * report git version
   ***********************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [piN2piN_projection] git version = %s\n", g_gitversion);
  }

  /***********************************************************
   * set geometry
   ***********************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0 ) {
    fprintf(stderr, "[piN2piN_projection] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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
    g_source_coords_list[i][0] = ( g_source_coords_list[i][0] + T_global ) % T_global;
    g_source_coords_list[i][1] = ( g_source_coords_list[i][1] + LX_global ) % LX_global;
    g_source_coords_list[i][2] = ( g_source_coords_list[i][2] + LY_global ) % LY_global;
    g_source_coords_list[i][3] = ( g_source_coords_list[i][3] + LZ_global ) % LZ_global;
  }

  /****************************************************
   * read relevant little group lists with their
   * rotation lists and irreps from file
   ****************************************************/
  little_group_type *lg = NULL;
  int const nlg = little_group_read_list ( &lg, little_group_list_filename );
  if ( nlg <= 0 ) {
    fprintf(stderr, "[test_lg] Error from little_group_read_list, status was %d %s %d\n", nlg, __FILE__, __LINE__ );
    EXIT(2);
  }
  fprintf(stdout, "# [test_lg] number of little groups = %d\n", nlg);

  little_group_show ( lg, stdout, nlg );
   
  /******************************************************
   * loop on 2-point functions
   ******************************************************/
  for ( int i2pt = 0; i2pt < g_twopoint_function_number; i2pt++ ) {

    /******************************************************
     * print the 2-point function parameters
     ******************************************************/
    char twopoint_function_filename[100];
    sprintf ( filename, "twopoint_function_%d.show", i2pt );
    if ( ( ofs = fopen ( filename, "w" ) ) == NULL ) {
      fprintf ( stderr, "[piN2piN_projection] Error from fopen %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }
    twopoint_function_print ( &(g_twopoint_function_list[i2pt]), "TWPT", ofs );
    fclose ( ofs );

    /***********************************************************
     * set number of timeslices
     ***********************************************************/
    int const nT = g_twopoint_function_list[i2pt].T;
    if ( io_proc == 2 ) fprintf( stdout, "# [piN2piN_projection] number of timeslices (incl. src and snk) is %d\n", nT);

    /****************************************************
     * read little group parameters
     ****************************************************/
    little_group_type little_group;
    if ( ( exitstatus = little_group_read ( &little_group, g_twopoint_function_list[i2pt].group, little_group_list_filename ) ) != 0 ) {
      fprintf ( stderr, "[] Error from little_group_read, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(2);
    }
    
    /****************************************************
     * set projector for current little group and irrep
     ****************************************************/
    little_group_projector_type projector;
    int ref_row_target    = -1;     // no reference row for target irrep
    int * ref_row_spin    = NULL;   // no reference row for spin matrices
    int refframerot       = -1;     // reference frame rotation
    int row_target        = -1;     // no target row
    int cartesian_list[1] = { 0 }   // not cartesian
    int parity_list[1]    = { 1 };  // intrinsic parity is +1
    int ** momentum_list  = NULL;   // no momentum list given
    int bispinor_list[1]  = { 1 };  // bispinor yes
    int J2_list[1]        = { 1 };  // spin 1/2

    exitstatus = little_group_projector_set ( &projector, &little_group, g_twopoint_function_list[i2pt].irrep , row_target, 1,
        J2_list, momentum_list, bispinor_list, parity_list, cartesian_list, ref_row_target, ref_row_spin, g_twopoint_function_list[i2pt].type, refframerot );

    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[piN2piN_projection] Error from little_group_projector_set, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(3);
    }

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

        /******************************************************
         * set current source coords in 2pt function
         ******************************************************/
        g_twopoint_function_list[i2pt].source_coords[0] = gsx[0];
        g_twopoint_function_list[i2pt].source_coords[1] = gsx[1];
        g_twopoint_function_list[i2pt].source_coords[2] = gsx[2];
        g_twopoint_function_list[i2pt].source_coords[3] = gsx[3];


        int const nrot = projector.rtarget->n;

        twopoint_function_type tp;

        twopoint_function_copy ( &tp, &( g_twopoint_function_list[i2pt] ) );

        gamma_matrix_type gl, gr, gi11, gi12, gi2, gf11, gf12, gf2;
        gamma_matrix_init ( &gl );
        gamma_matrix_init ( &gr );

        
        /******************************************************
         * loop on little group elements --- rotations
         ******************************************************/

        for ( int irotl = 0; irotl < projector.rtarget->n; irotl ++ ) {

          rot_point ( tp.pf1, g_twopoint_function_list[i2pt].pf1, projector.rp->R[irotl] );
          rot_point ( tp.pf2, g_twopoint_function_list[i2pt].pf2, projector.rp->R[irotl] );

          memcpy ( gl.v, projector.rspin[0]->R[irotl], 16*sizeof(double _Complex) )

          gamma_matrix_set ( &gf11, g_twopoint_function_list[i2pt].gf1[0], 1. );
          gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gf11, &gl, 'C', &gf11, &gl, 'H' );

          gamma_matrix_set ( &gf12, g_twopoint_function_list[i2pt].gf1[1], 1. );
          gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gf12, &gl, 'N', &gf12, &gl, 'H' );

          gamma_matrix_set ( &gf2, g_twopoint_function_list[i2pt].gf2, 1. );
          gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gf2, &gl, 'N', &gf2, &gl, 'H' );

          tp.gf1[0] = gf11.id;
          tp.gf1[1] = gf12.id;
          tp.gf2    = gf2.id;

        for ( int irotr = 0; irotr < projector.rtarget->n; irotr ++ ) {

          rot_point ( tp.pi1, g_twopoint_function_list[i2pt].pi1, projector.rp->R[irotr] );
          rot_point ( tp.pi2, g_twopoint_function_list[i2pt].pi2, projector.rp->R[irotr] );

          memcpy ( gr.v, projector.rspin[0]->R[irotl], 16*sizeof(double _Complex) )

          gamma_matrix_set ( &gi11, g_twopoint_function_list[i2pt].gi1[0], 1. );
          gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gi11, &gr, 'N', &gi11, &gr, 'T' );

          gamma_matrix_set ( &gi12, g_twopoint_function_list[i2pt].gi1[1], 1. );
          gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gi12, &gr, 'N', &gi12, &gr, 'H' );

          gamma_matrix_set ( &gi2, g_twopoint_function_list[i2pt].gi2, 1. );
          gamma_eq_gamma_op_ti_gamma_matrix_ti_gamma_op ( &gi2, &gr, 'N', &gi2, &gr, 'H' );

          tp.gi1[0] = gi11.id;
          tp.gi1[1] = gi12.id;
          tp.gi2    = gi2.id;

          STOPPED HERE
          /******************************************************
           * AFF reader
           ******************************************************/
          sprintf(filename, "%s.B%d.%.4d.PX%dPY%dPZ%d.t%dx%dy%dz%d.aff", "piN_piN_diagrams", iperm+1, Nconf,
              g_total_momentum_list[iptot][0], g_total_momentum_list[iptot][1], g_total_momentum_list[iptot][2],
              gsx[0], gsx[1], gsx[2], gsx[3] );
          affw = aff_writer (filename);
          if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
            EXIT(4);
          } else {
            fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
          }



      /******************************************************/
      /******************************************************/



        /******************************************************
         * allocate correlator and diagrams
         ******************************************************/
        //double _Complex * correlator = init_1level_ztable ( nT );
        //if ( correlator == NULL ) {
        //  fprintf(stderr, "[piN2piN_projection] Error from init_1level_ztable %s %d\n", __FILE__, __LINE__ );
        //  EXIT(47);
        //}

        double _Complex *** diagram = init_3level_ztable ( nT, 4, 4 );
        if ( diagram == NULL ) {
          fprintf(stderr, "[piN2piN_projection] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
          EXIT(47);
        }

        /******************************************************
         * open AFF file
         ******************************************************/
        char aff_filename_prefix[100];
        twopoint_function_get_aff_filename_prefix ( aff_filename_prefix, &(g_twopoint_function_list[i2pt]) );

        sprintf(filename, "%s.%.4d.tsrc%.2d.aff", aff_filename_prefix, Nconf, t_base );

        if ( io_proc == 2 ) {
          affr = aff_reader (filename);
          if ( const char * aff_status_str = aff_reader_errstr(affr) ) {
            fprintf(stderr, "[piN2piN_projection] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__ );
            EXIT(4);
          } else {
            fprintf(stdout, "# [piN2piN_projection] reading data from aff file %s\n", filename);
          }
        }

        /******************************************************
         * accumulate the diagrams for a twopoint_function
         ******************************************************/

        exitstatus = twopoint_function_accumulate_diagrams ( diagram, &(g_twopoint_function_list[i2pt]), nT, affr );
        if( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_projection] Error from twopoint_function_accumulate_diagrams, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(4);
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
        if ( io_proc == 2 ) fprintf ( stdout, "# [piN2piN_projection] time for twopoint_function entry = %e seconds\n", retime-ratime );

      }  // end of loop on 2-point functions

      /******************************************************/
      /******************************************************/

      /******************************************************
       * close ofs file pointer
       ******************************************************/
      if(io_proc == 2) { fclose ( ofs ); }

    }  // end of loop on coherent source locations

  }  // end of loop on base source locations

  /******************************************************/
  /******************************************************/

  /******************************************************
   * finalize
   ******************************************************/

  /******************************************************
   * free the allocated memory, finalize
   ******************************************************/
  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [piN2piN_projection] %s# [piN2piN_projection] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [piN2piN_projection] %s# [piN2piN_projection] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
