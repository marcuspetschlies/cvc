/****************************************************
 * p2gg_analyse.c
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "Q_clover_phi.h"
#include "contract_cvc_tensor.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"

#include "clover.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse P-J-J correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "p2gg";

  int c;
  int filename_set = 0;
  int check_momentum_space_WI = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  char aff_tag[400];
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

  g_the_time = time(NULL);

  /* set the default values */
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [p2gg_analyse] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [p2gg_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  Vhalf                  = VOLUME / 2;
  sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);
  sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);

  /***********************************************
   * set io process
   ***********************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[p2gg_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [p2gg_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   ***********************************************************
   **
   ** loop on source locations
   **
   ***********************************************************
   ***********************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) {

    /***********************************************************
     * determine source coordinates
     ***********************************************************/
    int const gsx[4] = {
      ( g_source_coords_list[isource_location][0] +  T_global ) %  T_global,
      ( g_source_coords_list[isource_location][1] + LX_global ) % LX_global,
      ( g_source_coords_list[isource_location][2] + LY_global ) % LY_global,
      ( g_source_coords_list[isource_location][3] + LZ_global ) % LZ_global };

    int source_proc_id = -1, sx[4];
    exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_analyse] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

#ifdef HAVE_LHPC_AFF
    /***********************************************
     ***********************************************
     **
     ** writer for aff output file
     **
     ***********************************************
     ***********************************************/
    if(io_proc == 2) {
      sprintf ( filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
      fprintf(stdout, "# [p2gg_analyse] reading data to file %s\n", filename);
      affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#endif

    /**********************************************************
     **********************************************************
     **
     ** hvp analysis
     **
     ** different momenta, mu, nu
     **
     **********************************************************
     **********************************************************/

    /**********************************************************
     * loop on momenta
     **********************************************************/
    for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

      int const sink_momentum[3] = {
        g_sink_momentum_number[isink_momentum][0],
        g_sink_momentum_number[isink_momentum][1],
        g_sink_momentum_number[isink_momentum][2] };

      /**********************************************************
       * loop on shifts in directions mu, nu
       **********************************************************/
      for( int mu = 0; mu < 4; mu++) {
      for( int mu = 0; mu < 4; mu++) {

        /**********************************************************
         *
         **********************************************************/


      }  /* end of loop on shift direction mu */

    }  /* end of loop on sink momenta */



    /* allocate memory for contractions, initialize */
    cvc_tensor_eo = init_2level_dtable ( 2, 32 * (size_t)Vhalf );
    if( cvc_tensor_eo == NULL ) {
      fprintf(stderr, "[p2gg_analyse] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(24);
    }

    /***************************************************************************
     * full tensor
     ***************************************************************************/
    memset(cvc_tensor_eo[0], 0, 32*VOLUME*sizeof(double) );
    memset(contact_term[0], 0, 8*sizeof(double));
    /* contraction */
    contract_cvc_tensor_eo ( cvc_tensor_eo[0], cvc_tensor_eo[1], contact_term[0], &(eo_spinor_field[120]), &(eo_spinor_field[180]),
       &(eo_spinor_field[0]), &(eo_spinor_field[60]), gauge_field_with_phase );

    /* subtract contact term */
    cvc_tensor_eo_subtract_contact_term (cvc_tensor_eo, contact_term[0], gsx, (int)( source_proc_id == g_cart_id ) );

    /* momentum projections */
    exitstatus = cvc_tensor_eo_momentum_projection ( &cvc_tp, cvc_tensor_eo, g_sink_momentum_list, g_sink_momentum_number);
    if(exitstatus != 0) {
      fprintf(stderr, "[p2gg_analyse] Error from cvc_tensor_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(26);
    }
    /* write results to file */
    sprintf(aff_tag, "/hvp/u-cvc-u-cvc/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = cvc_tensor_tp_write_to_aff_file ( cvc_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
    if(exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_analyse] Error from cvc_tensor_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(45);
    }
    fini_3level_dtable ( &cvc_tp );

    /* check position space WI */
    if(check_position_space_WI) {
      exitstatus = cvc_tensor_eo_check_wi_position_space ( cvc_tensor_eo );
      if(exitstatus != 0) {
        fprintf(stderr, "[p2gg_analyse] Error from cvc_tensor_eo_check_wi_position_space for full, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }
    }

    fini_2level_dtable ( &cvc_tensor_eo );

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     * local - cvc 2-point
     ***************************************************************************/
    sprintf(aff_tag, "/local-cvc/u-gf-u-cvc/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_local_cvc_2pt_eo (
        &(eo_spinor_field[120]), &(eo_spinor_field[180]),
        &(eo_spinor_field[0]), &(eo_spinor_field[60]),
        g_sequential_source_gamma_id_list, g_sequential_source_gamma_id_number, g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_analyse] Error from contract_local_cvc_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     * local - local 2-point  u - u
     ***************************************************************************/
    sprintf(aff_tag, "/local-local/u-gf-u-gi/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[168]), &(eo_spinor_field[228]),
       &(eo_spinor_field[ 48]), &(eo_spinor_field[108]),
       g_source_gamma_id_list, g_source_gamma_id_number,
       g_source_gamma_id_list, g_source_gamma_id_number,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_analyse] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }


    sprintf(aff_tag, "/local-local/d-gf-u-gi/t%.2dx%.2dy%.2dz%.2d", gsx[0], gsx[1], gsx[2], gsx[3] );
    exitstatus = contract_local_local_2pt_eo (
       &(eo_spinor_field[48]), &(eo_spinor_field[108]),
       &(eo_spinor_field[48]), &(eo_spinor_field[108]),
       g_source_gamma_id_list, g_source_gamma_id_number,
       g_source_gamma_id_list, g_source_gamma_id_number,
       g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_analyse] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(1);
    }
    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     ***************************************************************************
     **
     ** P -> gamma gamma contractions
     **
     ***************************************************************************
     ***************************************************************************/

    /***************************************************************************
     * loop on sequential source gamma matrices
     ***************************************************************************/
    for ( int iseq_source_momentum = 0; iseq_source_momentum < g_seq_source_momentum_number; iseq_source_momentum++) {

      g_seq_source_momentum[0] = g_seq_source_momentum_list[iseq_source_momentum][0];
      g_seq_source_momentum[1] = g_seq_source_momentum_list[iseq_source_momentum][1];
      g_seq_source_momentum[2] = g_seq_source_momentum_list[iseq_source_momentum][2];

      if( g_verbose > 2 && g_cart_id == 0) fprintf(stdout, "# [p2gg_analyse] using sequential source momentum no. %2d = (%d, %d, %d)\n", iseq_source_momentum,
          g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2]);

      /***************************************************************************
       * loop on sequential source gamma matrices
       ***************************************************************************/
      for( int isequential_source_gamma_id = 0; isequential_source_gamma_id < g_sequential_source_gamma_id_number; isequential_source_gamma_id++) {

        int sequential_source_gamma_id = g_sequential_source_gamma_id_list[ isequential_source_gamma_id ];
        if( g_verbose > 2 && g_cart_id == 0) fprintf(stdout, "# [p2gg_analyse] using sequential source gamma id no. %2d = %d\n", isequential_source_gamma_id, sequential_source_gamma_id);

        /***************************************************************************
         * loop on sequential source time slices
         ***************************************************************************/
        for ( int isequential_source_timeslice = 0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++) {

          g_sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];
          /* shift sequential source timeslice by source timeslice gsx[0] */
          int g_shifted_sequential_source_timeslice = ( gsx[0] + g_sequential_source_timeslice + T_global ) % T_global;

          if( g_verbose > 2 && g_cart_id == 0) 
            fprintf(stdout, "# [p2gg_analyse] using sequential source timeslice %d / %d\n", g_sequential_source_timeslice, g_shifted_sequential_source_timeslice);

          /* allocate memory for contractions, initialize */
          cvc_tensor_eo = init_2level_dtable ( 2, 32 * (size_t)Vhalf);
          if( cvc_tensor_eo == NULL ) {
            fprintf(stderr, "[p2gg_analyse] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(24);
          }
          memset(contact_term[0], 0, 8*sizeof(double));
          memset(contact_term[1], 0, 8*sizeof(double));


          /***************************************************************************
           * loop on quark flavors
           ***************************************************************************/
          for( int iflavor = 1; iflavor >= 0; iflavor-- )
          {

            /* flavor-dependent sequential source momentum */
            int seq_source_momentum[3] = { (1 - 2*iflavor) * g_seq_source_momentum[0],
                                           (1 - 2*iflavor) * g_seq_source_momentum[1],
                                           (1 - 2*iflavor) * g_seq_source_momentum[2] };

            if( g_verbose > 2 && g_cart_id == 0)
              fprintf(stdout, "# [p2gg_analyse] using flavor-dependent sequential source momentum (%d, %d, %d)\n", seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2]);


            /***************************************************************************
             * prepare sequential sources
             ***************************************************************************/
            for( int is = 0; is < 60; is++ ) 
            {
              int eo_spinor_field_id_e     = iflavor * 120 + is;
              int eo_spinor_field_id_o     = eo_spinor_field_id_e + 60;
              int eo_seq_spinor_field_id_e = 240 + is;
              int eo_seq_spinor_field_id_o = eo_seq_spinor_field_id_e + 60;

              exitstatus = init_clover_eo_sequential_source(
                  eo_spinor_field[ eo_seq_spinor_field_id_e ], eo_spinor_field[ eo_seq_spinor_field_id_o ],
                  eo_spinor_field[ eo_spinor_field_id_e     ], eo_spinor_field[ eo_spinor_field_id_o     ] ,
                  g_shifted_sequential_source_timeslice, gauge_field_with_phase, mzzinv[iflavor][0],
                  seq_source_momentum, sequential_source_gamma_id, eo_spinor_work[0]);
              if(exitstatus != 0) {
                fprintf(stderr, "[p2gg_analyse] Error from init_clover_eo_sequential_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(25);
              }

              /***************************************************************************
               * invert
               ***************************************************************************/

              double *full_spinor_work[2] = { eo_spinor_work[0], eo_spinor_work[2] };

              memset ( full_spinor_work[1], 0, sizeof_spinor_field);
              /* eo-precon -> full */
              spinor_field_eo2lexic ( full_spinor_work[0], eo_spinor_field[eo_seq_spinor_field_id_e], eo_spinor_field[eo_seq_spinor_field_id_o] );

              /* full_spinor_work[1] = D^-1 full_spinor_work[0] */
              exitstatus = _TMLQCD_INVERT ( full_spinor_work[1], full_spinor_work[0], iflavor );
              if(exitstatus < 0) {
                fprintf(stderr, "[p2gg_analyse] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(19);
              }

              /* full -> eo-precon 
               * full_spinor_work[0] = eo_spinor_work[0,1] <- full_spinor_work[1]
               * */
              spinor_field_lexic2eo ( full_spinor_work[1], eo_spinor_work[0], eo_spinor_work[1] );
              
              /* check residuum */  
              exitstatus = check_residuum_eo ( 
                  &( eo_spinor_field[eo_seq_spinor_field_id_e]), &(eo_spinor_field[eo_seq_spinor_field_id_o]),
                  &( eo_spinor_work[0] ),                        &( eo_spinor_work[1] ),
                  gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1 );
 
              /* copy solution into place */
              memcpy ( eo_spinor_field[eo_seq_spinor_field_id_e], eo_spinor_work[0], sizeof_eo_spinor_field );
              memcpy ( eo_spinor_field[eo_seq_spinor_field_id_o], eo_spinor_work[1], sizeof_eo_spinor_field );

            }  /* end of loop on spin-color and shift direction */



            /***************************************************************************
             * P - cvc - cvc tensor
             ***************************************************************************/

            if ( iflavor == 1 ) {
              memset(cvc_tensor_eo[0], 0, 32*VOLUME*sizeof(double) );
              memset(contact_term[1], 0, 8*sizeof(double));
            } else if ( iflavor == 0 ) {
              /* note: 4 x 4 x 2 x VOLUME/2 COMPLEX elements in cvc_tensor_eo[0/1] */
              complex_field_eq_complex_field_conj_ti_re (cvc_tensor_eo[0], (double)sequential_source_gamma_id_sign[ sequential_source_gamma_id ], 16*VOLUME );
              memset( contact_term[0], 0, 8*sizeof(double) );
            }

            /* flavor-dependent aff tag  */
            sprintf(aff_tag, "/p-cvc-cvc/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/fl%d",
                                  gsx[0], gsx[1], gsx[2], gsx[3], 
                                  g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2],
                                  sequential_source_gamma_id, g_sequential_source_timeslice, iflavor);

            /***************************************************************************
             * contraction for P - CVC - CVC tensor
             ***************************************************************************/
            contract_cvc_tensor_eo ( 
                cvc_tensor_eo[0], cvc_tensor_eo[1], contact_term[0], 
                &(eo_spinor_field[ ( 1 - iflavor ) * 120]), &(eo_spinor_field[ ( 1 - iflavor ) * 120 + 60]),
                &(eo_spinor_field[240]), &(eo_spinor_field[300]),
                gauge_field_with_phase );

            /* write the contact term to file */
            exitstatus = cvc_tensor_eo_write_contact_term_to_aff_file (contact_term[0], affw, aff_tag, io_proc );
            if(exitstatus != 0) {
              fprintf(stderr, "[p2gg_analyse] Error from cvc_tensor_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(26);
            }


            /***************************************************************************
             * contraction for P - local - local tensor
             ***************************************************************************/
            sprintf(aff_tag, "/p-loc-loc/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/fl%d", gsx[0], gsx[1], gsx[2], gsx[3],
                g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2],
                sequential_source_gamma_id, g_sequential_source_timeslice, iflavor );

            exitstatus = contract_local_local_2pt_eo (
                &(eo_spinor_field[ ( 1 - iflavor ) * 120 + 48]), &(eo_spinor_field[ ( 1 - iflavor ) * 120 + 108]),
                &(eo_spinor_field[288]), &(eo_spinor_field[348]),
                g_source_gamma_id_list, g_source_gamma_id_number,
                g_source_gamma_id_list, g_source_gamma_id_number,
                g_sink_momentum_list, g_sink_momentum_number,  affw, aff_tag, io_proc );

            if( exitstatus != 0 ) {
              fprintf(stderr, "[p2gg_analyse] Error from contract_local_local_2pt_eo, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(1);
            }

          }  /* end of loop on flavor */

          /***************************************************************************/
          /***************************************************************************/


          /* subtract contact term */
          cvc_tensor_eo_subtract_contact_term (cvc_tensor_eo, contact_term[0], gsx, (int)( source_proc_id == g_cart_id ) );

          /* momentum projections */
          exitstatus = cvc_tensor_eo_momentum_projection ( &cvc_tp, cvc_tensor_eo, g_sink_momentum_list, g_sink_momentum_number);
          if(exitstatus != 0) {
            fprintf(stderr, "[p2gg_analyse] Error from cvc_tensor_eo_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(26);
          }
             
          /* flavor-dependent aff tag  */
          sprintf(aff_tag, "/p-cvc-cvc/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d",
                                gsx[0], gsx[1], gsx[2], gsx[3], 
                                g_seq_source_momentum[0], g_seq_source_momentum[1], g_seq_source_momentum[2],
                                sequential_source_gamma_id, g_sequential_source_timeslice);

          /* write results to file */
          exitstatus = cvc_tensor_tp_write_to_aff_file ( cvc_tp, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
          if(exitstatus != 0 ) {
            fprintf(stderr, "[p2gg_analyse] Error from cvc_tensor_tp_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(45);
          }
          fini_3level_dtable ( &cvc_tp );

          /* check position space WI */
          if(check_position_space_WI) {
            exitstatus = cvc_tensor_eo_check_wi_position_space ( cvc_tensor_eo );
            if(exitstatus != 0) {
              fprintf(stderr, "[p2gg_analyse] Error from cvc_tensor_eo_check_wi_position_space for full, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(38);
            }
          }

          fini_2level_dtable ( &cvc_tensor_eo );

          /***************************************************************************/
          /***************************************************************************/

        }  /* end of loop on sequential source timeslices */
      }  /* end of loop on sequential source gamma id */
    }  /* end of loop on sequential source momentum */

#ifdef HAVE_LHPC_AFF
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg_analyse] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */


  }  /* end of loop on source locations */



  /****************************************
   * free the allocated memory, finalize
   ****************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

  fini_2level_dtable ( &eo_spinor_field );
  fini_2level_dtable ( &eo_spinor_work );

  /* free clover matrix terms */
  fini_clover ();

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif


#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_analyse] %s# [p2gg_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_analyse] %s# [p2gg_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
