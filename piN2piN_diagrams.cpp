/****************************************************
 * piN2piN_diagrams.cpp
 * 
 * Fri Jul  7 15:17:51 CEST 2017
 *
 * PURPOSE:
 *   originally copied from test_diagrams.cpp
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
// #include "matrix_init.h"
#include "table_init_z.h"
#include "contract_diagrams.h"
#include "aff_key_conversion.h"
#include "gamma.h"
#include "zm4x4.h"

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

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL, *affr_oet = NULL;
  struct AffWriter_s *affw = NULL;
  char * aff_status_str;
  struct AffNode_s *affn = NULL, *affn_oet = NULL, *affdir = NULL;
  char aff_tag[500];
#endif

  // vertex i2, gamma_5 and id
  int const gamma_i2_number = 1;
  int const gamma_i2_list[gamma_i2_number]    = {  5 };
  // double const gamma_i2_sign[gamma_i2_number] = { +1 };

  // vertex f2, gamma_5 and id
  int const gamma_f2_number = 1;
  int const gamma_f2_list[gamma_f2_number]    = {  5 };
  double const gamma_f2_sign[gamma_f2_number] = { +1 };

  // vertex f1, nucleon type
  // this is identical to vertex i1
  int const gamma_f1_nucleon_number = 4;
  int const gamma_f1_nucleon_list[gamma_f1_nucleon_number]    = {  5,  4,  6,  0 };
  double const gamma_f1_nucleon_sign[gamma_f1_nucleon_number] = { +1, +1, +1, +1 };

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
  fprintf(stdout, "[piN2piN_diagrams] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

 /******************************************************
  * report git version
  ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [piN2piN_diagrams] git version = %s\n", g_gitversion);
  }

  /******************************************************
   * set initial timestamp
   * - con: this is after parsing the input file
   ******************************************************/
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [piN2piN_diagrams] %s# [piN2piN_diagrams] start of run\n", ctime(&g_the_time));
    fflush(stdout);
  }

  /******************************************************
   *
   ******************************************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[piN2piN_diagrams] Error from init_geometry\n");
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

  gamma_matrix_type gamma_C, gamma_C_gi1;
  gamma_matrix_type gamma[16];

  gamma_matrix_init ( &gamma_C_gi1 );

  for ( int i = 0; i < 16; i++ ) {
    gamma_matrix_set ( &gamma[i], i, 1 );
  }
  
  gamma_matrix_init ( &gamma_C );
  gamma_matrix_mult ( &gamma_C, &gamma[0], &gamma[2] );

  if ( g_verbose > 2 ) {
    gamma_matrix_printf (&gamma[0], "g0", stdout);
    gamma_matrix_printf (&gamma[2], "g2", stdout);
    gamma_matrix_printf (&gamma_C,  "C", stdout);
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

  /******************************************************/
  /******************************************************/

  /******************************************************
   * loop on source locations
   ******************************************************/
  for( int i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];

    /******************************************************
     * open AFF input and output files
     ******************************************************/
    if(io_proc == 2) {

      /******************************************************
       * AFF input files
       ******************************************************/
      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN", Nconf, t_base );
      affr = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr(affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(4);
      } else {
        fprintf(stdout, "# [piN2piN_diagrams] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
      }
      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        EXIT(103);
      }

      /******************************************************
       * AFF oet input files
       ******************************************************/
      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN_oet", Nconf, t_base );
      affr_oet = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr(affr_oet);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(4);
      } else {
        fprintf(stdout, "# [piN2piN_diagrams] reading oet data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
      }
      if( (affn_oet = aff_reader_root( affr_oet )) == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error, aff oet reader is not initialized %s %d\n", __FILE__, __LINE__);
        EXIT(103);
      }

      /******************************************************
       * AFF output file
       ******************************************************/
      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN_diagrams", Nconf, t_base );
      affw = aff_writer (filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(4);
      } else {
        fprintf(stdout, "# [piN2piN_diagrams] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
      }
    }  /* end of if io_proc == 2 */


    /*******************************************
     * v3 type b_1_xi
     *
     *   Note: only one gamma_f2, which is gamma_5
     *******************************************/
    double _Complex ****** b1xi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
    if ( b1xi == NULL ) {
      fprintf(stderr, "[piN2piN_diagrams] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__);
      EXIT(47);
    }

    strcpy ( tag, "b_1_xi" );
    if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams] tag = %s\n", tag);

    for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

      for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

        for ( int isample = 0; isample < g_nsample; isample++ ) {

          if ( io_proc == 2 ) {
            aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], NULL, g_seq2_source_momentum_list[ipf2], g_source_coords_list[i_src], -1, -1 );
            if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] key = \"%s\"\n", aff_tag);
 
            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, b1xi[ipi2][0][ipf2][isample][0], T_global*12);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }
          }

        }  /* end of loop on samples */

      }  /* end of loop on sink momentum pf2 */

    }  /* end of loop on sequential source momentum pi2 */

    /**************************************************************************************/
    /**************************************************************************************/

    /*******************************************
     * loop on coherent source locations
     *******************************************/
    for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

      int source_proc_id, sx[4], gsx[4] = { t_coherent,
                    ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global,
                    ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global,
                    ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global };

      get_point_source_info (gsx, sx, &source_proc_id);

      /*******************************************
       * v2 type b_1_phi
       *******************************************/
      double _Complex ***** b1phi = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 );
      if ( b1phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "b_1_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams] tag = %s\n", tag);

      for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

        for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

          for ( int isample = 0; isample < g_nsample; isample++ ) {

            if ( io_proc == 2 ) {
              aff_key_conversion ( aff_tag, tag, isample, NULL, g_sink_momentum_list[ipf1], NULL, gsx, gamma_f1_nucleon_list[igf1], -1 );
              if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] key = \"%s\"\n", aff_tag);
 
              affdir = aff_reader_chpath (affr, affn, aff_tag );
              exitstatus = aff_node_get_complex (affr, affdir, b1phi[igf1][ipf1][isample][0], T_global*192);
              if( exitstatus != 0 ) {
                fprintf(stderr, "[piN2piN_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }
            }

          }  /* end of loop on Gamma_f1 */

        }  /* end of loop on samples */

      }  /* end of loop on sink momentum pf1 */

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * loop on total momenta
       **************************************************************************************/
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int * sink_momentum_id =  get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, g_total_momentum_list[iptot], g_sink_momentum_list, g_sink_momentum_number );
        if ( sink_momentum_id == NULL ) {
          fprintf(stderr, "[piN2piN_diagrams] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        /**************************************************************************************
         * loop on pf2
         **************************************************************************************/
        for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

          int ipf1 = sink_momentum_id[ipf2];
          if ( ipf1 == -1 ) continue;
           
          /**************************************************************************************
           * loop on gf1
           **************************************************************************************/
          for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {
            int igf2 = 0;

            /**************************************************************************************
             * loop on pi2
             **************************************************************************************/
            for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

              /**************************************************************************************
               * loop on gi1
               **************************************************************************************/
              for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) {
                int igi2 = 0;

                char aff_tag_suffix[400];

                contract_diagram_key_suffix ( aff_tag_suffix, gamma_f2_list[igf2], g_seq2_source_momentum_list[ipf2],
                    gamma_f1_nucleon_list[igf1], g_sink_momentum_list[ipf1], gamma_i2_list[igi2], g_seq_source_momentum_list[ipi2],
                    gamma_f1_nucleon_list[igi1], gsx );


                /**************************************************************************************
                 * set inner gamma matrix structure for baryon at source
                 **************************************************************************************/

                gamma_matrix_type gi1;
                gamma_matrix_set ( &gi1, gamma_f1_nucleon_list[igi1], gamma_f1_nucleon_sign[igi1] );

                gamma_matrix_mult ( &gamma_C_gi1, &gamma_C, &gi1 );
                gamma_matrix_transposed ( &gamma_C_gi1, &gamma_C_gi1);
                if ( g_verbose > 2 ) {
                  char name[20];
                  sprintf(name, "C_g%.2d_transposed", gi1.id);
                  gamma_matrix_printf (&gamma_C_gi1, name, stdout);
                }

                /**************************************************************************************/
                /**************************************************************************************/

                /**************************************************************************************
                 * B diagrams
                 **************************************************************************************/
                int const perm[2][4] = { { 1, 0, 2, 3}, { 2, 0, 1, 3 } };

                for ( int iperm = 0; iperm < 2; iperm++ ) {

                  double _Complex ***diagram = init_3level_ztable ( T_global, 4, 4 );
                  if ( diagram == NULL ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
                    EXIT(47);
                  }

                  // reduce to diagram, average over stochastic samples
                  if ( ( exitstatus = contract_diagram_sample ( diagram, b1xi[ipi2][0][ipf2], b1phi[igf1][ipf1], g_nsample, perm[iperm], gamma_C_gi1, T_global ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  // AFF
                  sprintf(aff_tag, "/%s/b%d/fwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( g_verbose > 4 ) fprintf ( stdout, "# [] AFF tag = \"%s\" %s %d\n", aff_tag, __FILE__, __LINE__ );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(106);
                  }

                  // AFF
                  sprintf(aff_tag, "/%s/b%d/bwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( g_verbose > 4 ) fprintf ( stdout, "# [] AFF tag = \"%s\" %s %d\n", aff_tag, __FILE__, __LINE__ );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(106);
                  }

                  fini_3level_ztable ( &diagram );

                }  // end of loop on permutations

              }  // end of loop on Gamma_i1

            }  // end of loop on p_i2

          }  // end of loop on Gamma_f1

        }  // end of loop on p_f2

        free ( sink_momentum_id );

      }  // end of loop on p_tot

      fini_5level_ztable ( &b1phi );

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * v3 type w_1_xi
       **************************************************************************************/
      double _Complex ***** w1xi = init_5level_ztable ( gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
      if ( w1xi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(47);
      }

      strcpy ( tag, "w_1_xi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams] tag = %s\n", tag);

      for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

        for ( int isample = 0; isample < g_nsample; isample++ ) {

          if ( io_proc == 2 ) {
            aff_key_conversion ( aff_tag, tag, isample, NULL, NULL, g_seq2_source_momentum_list[ipf2], gsx, -1, -1 );
            if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] key = \"%s\"\n", aff_tag);
 
            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, w1xi[0][ipf2][isample][0], T_global*12);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }

          }  // end of if io proc == 2

        }  // end of loop on samples

      }  // end of loop on sink momentum pf2

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * v2 type w_1_phi
       **************************************************************************************/
      double _Complex ******w1phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( w1phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_1_phi" );
      fprintf(stdout, "\n\n# [piN2piN_diagrams] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, gamma_f1_nucleon_list[igf1], -1 );
                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w1phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }

              }  // end of if io proc == 2

            }  // end of loop on Gamma_f1

          }  // end of loop on samples

        }  // end of loop on sink momentum pf1

      }  // end of loop on seq source momentum pi2

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * w_3_phi
       **************************************************************************************/
      double _Complex ****** w3phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_3_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, gamma_f1_nucleon_list[igf1], -1 );
                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w3phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }

              }  // end of if io proc == 2

            }  // end of loop on samples

          }  // end of loop on sink momentum pf1

        }  // end of loop on Gamma_f1

      }  // end of loop on seq source momentum pi2

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * loop on total momentum
       **************************************************************************************/
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int * sink_momentum_id =  get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, g_total_momentum_list[iptot], g_sink_momentum_list, g_sink_momentum_number );
        if ( sink_momentum_id == NULL ) {
          fprintf(stderr, "[piN2piN_diagrams] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        /**************************************************************************************/
        /**************************************************************************************/

        /**************************************************************************************
         * loop on pf2
         **************************************************************************************/
        for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

          int ipf1 = sink_momentum_id[ipf2];
          if ( ipf1 == -1 ) continue;

          /**************************************************************************************/
          /**************************************************************************************/

          /**************************************************************************************
           * loop on gf1
           **************************************************************************************/
          for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {
            int igf2 = 0;

            /**************************************************************************************/
            /**************************************************************************************/

            /**************************************************************************************
             * loop on pi2
             **************************************************************************************/
            for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

              /**************************************************************************************/
              /**************************************************************************************/

              /**************************************************************************************
               * loop on gi1
               **************************************************************************************/
              for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) {
                int igi2 = 0;

                char aff_tag_suffix[400];

                contract_diagram_key_suffix ( aff_tag_suffix, gamma_f2_list[igf2], g_seq2_source_momentum_list[ipf2],
                    gamma_f1_nucleon_list[igf1], g_sink_momentum_list[ipf1], gamma_i2_list[igi2], g_seq_source_momentum_list[ipi2],
                    gamma_f1_nucleon_list[igi1], gsx );

                /**************************************************************************************/
                /**************************************************************************************/

                int const perm[4][4] = { { 1, 0, 2, 3}, { 3, 0, 2, 1 }, { 2, 0, 3, 1 }, { 2, 0, 1, 3 } };
                gamma_matrix_type gi1;
                gamma_matrix_set  ( &gi1, gamma_f1_nucleon_list[igi1], gamma_f1_nucleon_sign[igi1] );
                gamma_matrix_mult ( &gamma_C_gi1, &gamma_C, &gi1 );
                // no transpostion here with choice of perm above; cf. pdf
                // gamma_matrix_transposed ( &gamma_C_gi1, &gamma_C_gi1);
                if ( g_verbose > 2 ) {
                  char name[20];
                  sprintf(name, "C_g%.2d", gi1.id);
                  gamma_matrix_printf (&gamma_C_gi1, name, stdout);
                }

                /**************************************************************************************/
                /**************************************************************************************/

                /**************************************************************************************
                 * W diagrams 1, 2
                 **************************************************************************************/
               
                for ( int iperm = 0; iperm < 2; iperm++ ) {

                  double _Complex ***diagram = init_3level_ztable ( T_global, 4, 4 );
                  if ( diagram == NULL ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
                    EXIT(47);
                  }

                  // reduce to diagram, average over stochastic samples
                  if ( ( exitstatus = contract_diagram_sample ( diagram, w1xi[0][ipf2], w1phi[ipi2][igf1][ipf1], g_nsample, perm[iperm], gamma_C_gi1, T_global ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  // AFF key
                  sprintf(aff_tag, "/%s/w%d/fwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(106);
                  }

                  // AFF key
                  sprintf(aff_tag, "/%s/w%d/bwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(106);
                  }

                  fini_3level_ztable ( &diagram );

                }  // end of loop on permutations

                /**************************************************************************************/
                /**************************************************************************************/

                /**************************************************************************************
                 * W diagrams 3, 4
                 **************************************************************************************/

                for ( int iperm = 2; iperm < 4; iperm++ ) {

                  double _Complex ***diagram = init_3level_ztable ( T_global, 4, 4 );
                  if ( diagram == NULL ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
                    EXIT(47);
                  }

                  // reduce to diagram, average over stochastic samples
                  if ( ( exitstatus = contract_diagram_sample ( diagram, w1xi[0][ipf2], w3phi[ipi2][igf1][ipf1], g_nsample, perm[iperm], gamma_C_gi1, T_global ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  // AFF key
                  sprintf(aff_tag, "/%s/w%d/fwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(106);
                  }

                  // AFF key
                  sprintf(aff_tag, "/%s/w%d/bwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(106);
                  }

                  fini_3level_ztable ( &diagram );

                }  // end of loop on permutations

              }  // end of loop on Gamma_i1

            }  // end of loop on p_i2

          }  // end of loop on Gamma_f1

        }  // end of loop on p_f2

        free ( sink_momentum_id );

      }  // end of loop on p_tot

      fini_6level_ztable ( &w1phi );
      fini_6level_ztable ( &w3phi );

      fini_5level_ztable ( &w1xi );

    }  // end of loop on coherent source locations

    /**************************************************************************************/
    /**************************************************************************************/

    fini_6level_ztable ( &b1xi );

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * oet part
     **************************************************************************************/

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * loop on coherent source locations
     **************************************************************************************/
    for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

      int source_proc_id, sx[4], gsx[4] = { t_coherent,
                    ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global,
                    ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global,
                    ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global };

      get_point_source_info (gsx, sx, &source_proc_id);

      int zero_momentum[3] = {0,0,0};

      /**************************************************************************************/
      /**************************************************************************************/
    
      /**************************************************************************************
       * v3 type z_1_xi
       **************************************************************************************/
      double _Complex ***** z1xi = init_5level_ztable ( gamma_f2_number, g_seq2_source_momentum_number, 4, T_global, 12 );
      if ( z1xi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }
      strcpy ( tag, "z_1_xi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams] tag = %s\n", tag);

      /**************************************************************************************
       * loop on gf2
       **************************************************************************************/
      for ( int igf2 = 0; igf2 < gamma_f2_number; igf2++ ) {

        /**************************************************************************************
         * loop on pf2
         **************************************************************************************/
        for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

          /**************************************************************************************
           * loop on spin component
           **************************************************************************************/
          for ( int ispin = 0; ispin < 4; ispin++ ) {

            if ( io_proc == 2 ) {

              aff_key_conversion ( aff_tag, tag, 0, zero_momentum, NULL, g_seq2_source_momentum_list[ipf2], gsx, gamma_f2_list[igf2], ispin );
              if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] key = \"%s\"\n", aff_tag);
 
              affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
              exitstatus = aff_node_get_complex (affr_oet, affdir, z1xi[igf2][ipf2][ispin][0], T_global*12);
              if( exitstatus != 0 ) {
                fprintf(stderr, "[piN2piN_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }

            }  // end of if io proc == 2

          }  // end of loop on oet spin

        }  // end of loop on p_f2

      }  // end of loop on Gamma_f2

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * v2 type z_3_phi
       **************************************************************************************/
      double _Complex ****** z3phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
      if ( z3phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }
      strcpy ( tag, "z_3_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams] tag = %s\n", tag);

      /**************************************************************************************
       * loop on pi2
       **************************************************************************************/
      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        /**************************************************************************************
         * loop on gf1
         **************************************************************************************/
        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          /**************************************************************************************
           * loop on pf1
           **************************************************************************************/
          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            /**************************************************************************************
             * loop on spin components
             **************************************************************************************/
            for ( int ispin = 0; ispin < 4; ispin++ ) {

              if ( io_proc == 2 ) {

                aff_key_conversion ( aff_tag, tag, 0, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, gamma_f1_nucleon_list[igf1], ispin );
                if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
                exitstatus = aff_node_get_complex (affr_oet, affdir, z3phi[ipi2][igf1][ipf1][ispin][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }
                
              }  // end of if io proc == 2

            }  // end of loop on oet spin index */

          }  // end of loop on sink momentum p_f1 */

        }  // end of loop on Gamma_f1 */

      }  // end of loop on seq source momentum pi2 */

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * z_1_phi
       **************************************************************************************/
      double _Complex ****** z1phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
      if ( z1phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(47);
      }
      strcpy ( tag, "z_1_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams] tag = %s\n", tag);

      /**************************************************************************************
       * loop on pi2
       **************************************************************************************/
      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        /**************************************************************************************
         * loop on gf1
         **************************************************************************************/
        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          /**************************************************************************************
           * loop on pf1
           **************************************************************************************/
          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            /**************************************************************************************
             * loop on spin components
             **************************************************************************************/
            for ( int ispin = 0; ispin < 4; ispin++ ) {

              if ( io_proc == 2 ) {

                aff_key_conversion ( aff_tag, tag, 0, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, gamma_f1_nucleon_list[igf1], ispin );
                if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
                exitstatus = aff_node_get_complex (affr_oet, affdir, z1phi[ipi2][igf1][ipf1][ispin][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }
                
              }  // end of if io proc

            }  // end of loop on oet spin index

          }  // end of loop on sink momentum p_f1

        }  // end of loop on Gamma_f1

      }  // end of loop on seq source momentum pi2

      /**************************************************************************************/
      /**************************************************************************************/

      char name[20];
      gamma_matrix_type goet, gamma_5;
      gamma_matrix_set ( &goet, gamma_f2_list[0], gamma_f2_sign[0] );
      gamma_matrix_set ( &gamma_5, 5, 1 );
      gamma_matrix_mult ( &goet, &goet, &gamma_5 );
      if ( g_verbose > 2 ) {
        sprintf(name, "goet_g5" );
        gamma_matrix_printf (&goet, name, stdout);
      }

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * loop on total momenta
       **************************************************************************************/
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int * sink_momentum_id =  get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, g_total_momentum_list[iptot], g_sink_momentum_list, g_sink_momentum_number );
        if ( sink_momentum_id == NULL ) {
          fprintf(stderr, "[piN2piN_diagrams] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        /**************************************************************************************
         * loop on pf2
         **************************************************************************************/
        for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

          int ipf1 = sink_momentum_id[ipf2];
          if ( ipf1 == -1 ) continue;

          /**************************************************************************************/
          /**************************************************************************************/

          /**************************************************************************************
           * loop on gf1
           **************************************************************************************/
          for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {
            int igf2 = 0;

            /**************************************************************************************/
            /**************************************************************************************/

            /**************************************************************************************
             * loop on pi2
             **************************************************************************************/
            for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

              /**************************************************************************************/
              /**************************************************************************************/

              /**************************************************************************************
               * loop on gi1
               **************************************************************************************/
              for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) {
                int igi2 = 0;

                char aff_tag_suffix[400];

                contract_diagram_key_suffix ( aff_tag_suffix, gamma_f2_list[igf2], g_seq2_source_momentum_list[ipf2],
                    gamma_f1_nucleon_list[igf1], g_sink_momentum_list[ipf1], gamma_i2_list[igi2], g_seq_source_momentum_list[ipi2],
                    gamma_f1_nucleon_list[igi1], gsx );

                /**************************************************************************************/
                /**************************************************************************************/

                int const perm[4][4] = { { 0, 2, 1, 3 }, { 0, 2, 3, 1 }, { 2, 0, 3, 1 }, { 2, 0, 1, 3 } };

                gamma_matrix_type gi1;
                gamma_matrix_set  ( &gi1, gamma_f1_nucleon_list[igi1], gamma_f1_nucleon_sign[igi1] );
                gamma_matrix_mult ( &gamma_C_gi1, &gamma_C, &gi1 );
                // no transposition with above choice of permutation, cf. pdf
                // gamma_matrix_transposed ( &gamma_C_gi1, &gamma_C_gi1);
                if ( g_verbose > 2 ) {
                  sprintf(name, "C_g%.2d", gi1.id);
                  gamma_matrix_printf (&gamma_C_gi1, name, stdout);
                }

                /**************************************************************************************/
                /**************************************************************************************/

                /**************************************************************************************
                 * Z diagrams 1, 2
                 **************************************************************************************/
                for ( int iperm = 0; iperm < 2; iperm++ ) {

                  double _Complex *** diagram = init_3level_ztable ( T_global, 4, 4 );
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
                    EXIT(47);
                  }

                  if ( ( exitstatus = contract_diagram_sample_oet (diagram, z1xi[0][ipf2],  z1phi[ipi2][igf1][ipf1], goet, perm[iperm], gamma_C_gi1, T_global ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_sample_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(108);
                  }

                  // AFF tag
                  sprintf(aff_tag, "/%s/z%d/fwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(109);
                  }

                  // AFF tag
                  sprintf(aff_tag, "/%s/z%d/bwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(110);
                  }

                  fini_3level_ztable ( &diagram );

                } // end of loop on permutations


                /**************************************************************************************
                 * Z diagrams 3, 4
                 **************************************************************************************/
                for ( int iperm = 2; iperm < 4; iperm++ ) {

                  double _Complex *** diagram = init_3level_ztable ( T_global, 4, 4 );
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
                    EXIT(47);
                  }

                  if ( ( exitstatus = contract_diagram_sample_oet (diagram, z1xi[0][ipf2],  z3phi[ipi2][igf1][ipf1], goet, perm[iperm], gamma_C_gi1, T_global ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_sample_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  // AFF tag
                  sprintf(aff_tag, "/%s/z%d/fwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(106);
                  }

                  // AFF tag
                  sprintf(aff_tag, "/%s/z%d/bwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(107);
                  }

                  fini_3level_ztable ( &diagram );

                }  // end of loop on permutations

              }  // end of loop on Gamma_i1

            }  // end of loop on p_i2

          }  // end of loop on Gamma_f1

        }  // end of loop on p_f2

        free ( sink_momentum_id );

      }  // end of loop on p_tot

      /**************************************************************************************/
      /**************************************************************************************/

      fini_6level_ztable ( &z1phi );
      fini_6level_ztable ( &z3phi );
      fini_5level_ztable ( &z1xi );

    }  // end of loop on coherent source locations

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * direct diagrams
     **************************************************************************************/

    /* loop on coherent source locations */
    for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

      int source_proc_id, sx[4], gsx[4] = { t_coherent,
                    ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global,
                    ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global,
                    ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global };

      get_point_source_info (gsx, sx, &source_proc_id);

      /**************************************************************************************
       * bb_aux auxilliary field, which is 4x4 instead of 16
       **************************************************************************************/
      double _Complex *** bb_aux = init_3level_ztable ( T_global, 4, 4 );
      if ( bb_aux == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      /**************************************************************************************
       * bb
       **************************************************************************************/
      double _Complex ****** bb = init_6level_ztable ( gamma_f1_nucleon_number, gamma_f1_nucleon_number, g_sink_momentum_number, 2, T_global, 16 );
      if ( bb == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }
      strcpy ( tag, "N-N" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams] tag = %s\n", tag);

      /**************************************************************************************
       * loop on gi1
       **************************************************************************************/
      for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) {

        /**************************************************************************************
         * loop on gf1
         **************************************************************************************/
        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          /**************************************************************************************
           * loop on pf1
           **************************************************************************************/
          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            /**************************************************************************************
             * loop on diagrams
             **************************************************************************************/
            for ( int idiag = 0; idiag < 2; idiag++ ) {

              if ( io_proc == 2 ) {

                aff_key_conversion_diagram ( aff_tag, tag, NULL, NULL, g_sink_momentum_list[ipf1], NULL, gamma_f1_nucleon_list[igi1], -1, gamma_f1_nucleon_list[igf1], -1, gsx, "n", idiag+1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, bb[igi1][igf1][ipf1][idiag][0], T_global*16);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__ );
                  EXIT(105);
                }

              }  // end of if io proc == 2

            }  // end of loop on diagrams

          }  // end of loop on sink momentum

        }  // end of loop on Gamma_f1

      }  // end of loop on Gamma_i1

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * mm
       **************************************************************************************/
      double _Complex ***** mm = init_5level_ztable ( g_seq_source_momentum_number, gamma_f2_number, gamma_f2_number, g_sink_momentum_number, T_global );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }
      strcpy ( tag, "m-m" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams] tag = %s\n", tag);

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * loop on pi2
       **************************************************************************************/
      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        /**************************************************************************************
         * loop on gi2
         **************************************************************************************/
        for ( int igi2 = 0; igi2 < gamma_i2_number; igi2++ ) {

          /**************************************************************************************
           * loop on gf2
           **************************************************************************************/
          for ( int igf2 = 0; igf2 < gamma_f2_number; igf2++ ) {

            /**************************************************************************************
             * loop on gf2
             **************************************************************************************/
            for ( int ipf2 = 0; ipf2 < g_sink_momentum_number; ipf2++ ) {

              if ( io_proc == 2 ) {

                /* sprintf(aff_tag, "/%s/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/px%.2dpy%.2dpz%.2d", tag, gsx[0], gsx[1], gsx[2], gsx[3],
                    gamma_f2_list[igi2], gamma_f2_list[igf2],
                    g_sink_momentum_list[ipf2][0], g_sink_momentum_list[ipf2][1], g_sink_momentum_list[ipf2][2] ); */

                aff_key_conversion_diagram (  aff_tag, tag, NULL, g_seq_source_momentum_list[ipi2], NULL, g_sink_momentum_list[ipf2], -1, gamma_i2_list[igi2], -1, gamma_f2_list[igf2], gsx, NULL, 0  );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
                exitstatus = aff_node_get_complex (affr_oet, affdir, mm[ipi2][igi2][igf2][ipf2], T_global);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d\n", aff_tag, exitstatus);
                  EXIT(105);
                }
                if ( g_verbose > 4 ) {
                  for ( int it = 0; it < T_global; it++ ) {
                    fprintf(stdout, "# [piN2piN_diagrams] m-m %3d %25.16e %25.16e\n", it, 
                        creal( mm[ipi2][igi2][igf2][ipf2][it] ), cimag( mm[ipi2][igi2][igf2][ipf2][it] ));
                  }
                }

              }   // end of if io proc > 2 

            }  // end of loop on sink momentum

          }  // end of loop on Gamma_f2

        }  // end of loop on Gamma_i2

      }  // end of loop in pi2

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * loop on total momenta
       **************************************************************************************/
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int * sink_momentum_id =  get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, g_total_momentum_list[iptot], g_sink_momentum_list, g_sink_momentum_number );
        if ( sink_momentum_id == NULL ) {
          fprintf(stderr, "[piN2piN_diagrams] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        /**************************************************************************************
         * loop on pf2
         **************************************************************************************/
        for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

          int ipf1 = sink_momentum_id[ipf2];
          if ( ipf1 == -1 ) continue;

          /**************************************************************************************
           * loop on gf1
           **************************************************************************************/
          for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {
            int igf2 = 0;

            /**************************************************************************************
             * loop on pi2
             **************************************************************************************/
            for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

              /**************************************************************************************
               * loop on gi1
               **************************************************************************************/
              for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) { 
                int igi2 = 0;

                char aff_tag_suffix[400];

                contract_diagram_key_suffix ( aff_tag_suffix, gamma_f2_list[igf2], g_seq2_source_momentum_list[ipf2],
                    gamma_f1_nucleon_list[igf1], g_sink_momentum_list[ipf1], gamma_i2_list[igi2], g_seq_source_momentum_list[ipi2],
                    gamma_f1_nucleon_list[igi1], gsx );


                /**************************************************************************************/
                /**************************************************************************************/

                /**************************************************************************************
                 * loop on diagrams
                 **************************************************************************************/
                for ( int idiag = 0; idiag < 2; idiag++ ) {

                  double _Complex *** diagram = init_3level_ztable ( T_global, 4, 4 );
                  if ( diagram == NULL ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
                    EXIT(47);
                  }

                  memcpy( bb_aux[0][0], bb[igi1][igf1][ipf1][idiag][0], 16*T_global*sizeof(double _Complex) );

                  // multiply baryon 2-point function with meson 2-point function */
                  exitstatus = contract_diagram_zmx4x4_field_ti_co_field ( diagram, bb_aux,  mm[ipi2][igi2][igf2][ipf2], T_global );
                  // memcpy(diagram[0][0],  bb_aux[0][0], 16*T_global*sizeof(double _Complex) );

                  // transpose
                  exitstatus = contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( diagram, diagram, T_global );

                  // AFF
                  sprintf(aff_tag, "/%s/s%d/fwd/%s", "pixN-pixN", idiag+1, aff_tag_suffix );
                  if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] aff tag = %s\n", aff_tag);
                  if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d\n", exitstatus);
                    EXIT(105);
                  }
   
                  // AFF
                  sprintf(aff_tag, "/%s/s%d/bwd/%s", "pixN-pixN", idiag+1, aff_tag_suffix );
                  if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] aff tag = %s\n", aff_tag);
                  if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d\n", exitstatus);
                    EXIT(105);
                  }

                  fini_3level_ztable ( &diagram );

                }  // end of loop on diagrams

              }  // end of loop on Gamma_i1

            }  // end of loop on p_i2

          }  // end of loop on Gamma_f1

        }  // end of loop on p_f2

        free ( sink_momentum_id );

      }  // end of loop on p_tot

      /**************************************************************************************/
      /**************************************************************************************/

      fini_6level_ztable ( &bb );
      fini_5level_ztable ( &mm );
      fini_3level_ztable ( &bb_aux );


    }  // end of loop on coherent source locations

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * close AFF readers for input and output files
     **************************************************************************************/
    if(io_proc == 2) {
      aff_reader_close (affr);
      aff_reader_close (affr_oet);

      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__ );
        EXIT(111);
      }
    }  // end of if io_proc == 2

  }  // end of loop on base source locations

  /**************************************************************************************
   * free the allocated memory, finalize
   **************************************************************************************/
  free_geometry();

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [piN2piN_diagrams] %s# [piN2piN_diagrams] end of run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [piN2piN_diagrams] %s# [piN2piN_diagrams] end of run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
