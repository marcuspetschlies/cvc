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
#include "gamma.h"

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

  int C_gamma_to_gamma[16][2];

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  struct AffWriter_s *affw = NULL;
  char * aff_status_str;
  struct AffNode_s *affn = NULL, *affn2 = NULL, *affdir = NULL;
  char aff_tag[200];
#endif

  /* vertex f2, gamma_5 and id,  vector indices and pseudo-vector */
  const int gamma_f2_number = 1;
  int gamma_f2_list[gamma_f2_number]    = {  5 };
  double gamma_f2_sign[gamma_f2_number] = { +1 };
/*
  const int gamma_f1_nucleon_number = 4;
  int gamma_f1_nucleon_list[gamma_f1_nucleon_number]    = {  5,  4,  6,  0 };
  double gamma_f1_nucleon_sign[gamma_f1_nucleon_number] = { +1, +1, +1, +1 };
*/
  const int gamma_f1_nucleon_number = 3;
  int gamma_f1_nucleon_list[gamma_f1_nucleon_number]    = {  5,  4,  6 };
  double gamma_f1_nucleon_sign[gamma_f1_nucleon_number] = { +1, +1, +1 };

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
  fprintf(stdout, "[ piN2piN_correlators] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /******************************************************
   *
   ******************************************************/

  if(init_geometry() != 0) {
    fprintf(stderr, "[ piN2piN_correlators] Error from init_geometry\n");
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

  gamma_matrix_type gamma_0, gamma_2, gamma_C, C_gi1;

  gamma_matrix_init ( &C_gi1 );

  gamma_matrix_set ( &gamma_0, 0, 1 );
  gamma_matrix_set ( &gamma_2, 2, 1 );
  
  gamma_matrix_init ( &gamma_C );
  gamma_matrix_mult ( &gamma_C, &gamma_0, &gamma_2 );

  gamma_matrix_printf (&gamma_0, "g0", stdout);
  gamma_matrix_printf (&gamma_2, "g2", stdout);
  gamma_matrix_printf (&gamma_C, "C", stdout);

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
      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN", Nconf, t_base );
      fprintf(stdout, "# [ piN2piN_correlators] writing data to file %s\n", filename);
      affr = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr(affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[ piN2piN_correlators] Error from aff_reader, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [ piN2piN_correlators] reading data from aff file %s\n", filename);
      }

      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[ piN2piN_correlators] Error, aff writer is not initialized\n");
        EXIT(103);
      }

      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN_diagrams", Nconf, t_base );
      affw = aff_writer (filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[ piN2piN_correlators] Error from aff_writer, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [ piN2piN_correlators] writing data to file %s\n", filename);
      }

      if( (affn2 = aff_writer_root( affw )) == NULL ) {
        fprintf(stderr, "[ piN2piN_correlators] Error, aff writer is not initialized\n");
        EXIT(103);
      }
    }  /* end of if io_proc == 2 */

    double _Complex ******b1xi = NULL;
    double _Complex **diagram = NULL, **diagram_buffer = NULL;

    exitstatus= init_2level_zbuffer ( &diagram, T_global, 16 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[ piN2piN_correlators] Error from init_level_zbuffer, status was %d\n", exitstatus);
      EXIT(47);
    }

    exitstatus= init_2level_zbuffer ( &diagram_buffer, T_global, 16 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[ piN2piN_correlators] Error from init_level_zbuffer, status was %d\n", exitstatus);
      EXIT(47);
    }

    /*******************************************
     * b_1_xi
     *
     *   Note: only one gamma_f2, which is gamma_5
     *******************************************/
    exitstatus= init_6level_zbuffer ( &b1xi, g_seq_source_momentum_number, gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[ piN2piN_correlators] Error from init_6level_zbuffer, status was %d\n", exitstatus);
      EXIT(47);
    }

    strcpy ( tag, "b_1_xi" );
    fprintf(stdout, "\n\n# [ piN2piN_correlators] tag = %s\n", tag);

    for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {
      for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {
        for ( int isample = 0; isample < g_nsample; isample++ ) {

          if ( io_proc == 2 ) {
            aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], NULL, g_seq2_source_momentum_list[ipf2], g_source_coords_list[i_src], -1, -1 );
            if ( g_verbose > 2 ) fprintf(stdout, "# [ piN2piN_correlators] key = \"%s\"\n", aff_tag);
 
            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, b1xi[ipi2][0][ipf2][isample][0], T_global*12);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_get_complex, status was %d\n", exitstatus);
              EXIT(105);
            }
          }

        }  /* end of loop on samples */
      }  /* end of loop on sink momentum pf2 */
    }  /* end of loop on sequential source momentum pi2 */

    /*******************************************/
    /*******************************************/

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

      /*******************************************
       * b_1_phi
       *******************************************/
      double _Complex *****b1phi = NULL;
      exitstatus= init_5level_zbuffer ( &b1phi, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[ piN2piN_correlators] Error from init_5level_zbuffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      strcpy ( tag, "b_1_phi" );
      fprintf(stdout, "\n\n# [ piN2piN_correlators] tag = %s\n", tag);

      for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {
        for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {
          for ( int isample = 0; isample < g_nsample; isample++ ) {

            if ( io_proc == 2 ) {
              aff_key_conversion ( aff_tag, tag, isample, NULL, g_sink_momentum_list[ipf1], NULL, gsx, gamma_f1_nucleon_list[igf1], -1 );
              if ( g_verbose > 2 ) fprintf(stdout, "# [ piN2piN_correlators] key = \"%s\"\n", aff_tag);
 
              affdir = aff_reader_chpath (affr, affn, aff_tag );
              exitstatus = aff_node_get_complex (affr, affdir, b1phi[igf1][ipf1][isample][0], T_global*192);
              if( exitstatus != 0 ) {
                fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_get_complex, status was %d\n", exitstatus);
                EXIT(105);
              }
            }

          }  /* end of loop on Gamma_f1 */
        }  /* end of loop on samples */
      }  /* end of loop on sink momentum pf1 */

      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int **sink_momentum_list = NULL, **sink_momentum_list_all = NULL;
        exitstatus = init_2level_ibuffer ( &sink_momentum_list, g_seq2_source_momentum_number, 3 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[ piN2piN_correlators] Error from init_2level_ibuffer, status was %d\n", exitstatus );
          EXIT(1);
        }
        exitstatus = init_2level_ibuffer ( &sink_momentum_list_all, g_sink_momentum_number, 3 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[ piN2piN_correlators] Error from init_2level_ibuffer, status was %d\n", exitstatus );
          EXIT(1);
        }
        for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {
          sink_momentum_list_all[ipf1][0] = g_sink_momentum_list[ipf1][0];
          sink_momentum_list_all[ipf1][1] = g_sink_momentum_list[ipf1][1];
          sink_momentum_list_all[ipf1][2] = g_sink_momentum_list[ipf1][2];
        }
        for ( int ipf1 = 0; ipf1 < g_seq2_source_momentum_number; ipf1++ ) {
          sink_momentum_list[ipf1][0] = g_total_momentum_list[iptot][0] - g_seq2_source_momentum_list[ipf1][0];
          sink_momentum_list[ipf1][1] = g_total_momentum_list[iptot][1] - g_seq2_source_momentum_list[ipf1][1];
          sink_momentum_list[ipf1][2] = g_total_momentum_list[iptot][2] - g_seq2_source_momentum_list[ipf1][2];
        }
        int *sink_momentum_id = NULL;
        exitstatus = match_momentum_id ( &sink_momentum_id, sink_momentum_list, sink_momentum_list_all, g_seq2_source_momentum_number, g_sink_momentum_number );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[ piN2piN_correlators] Error from match_momentum_id, status was %d\n", exitstatus );
          EXIT(1);
        }
        fini_2level_ibuffer ( &sink_momentum_list );
        fini_2level_ibuffer ( &sink_momentum_list_all );

        for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

          int ipf1 = sink_momentum_id[ipf2];
          if ( ipf1 == -1 ) continue;
           
          for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

            for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

              for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) {

                char aff_tag_suffix[200];
                sprintf(aff_tag_suffix, "pi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
                    g_seq_source_momentum_list[ipi2][0], g_seq_source_momentum_list[ipi2][1], g_seq_source_momentum_list[ipi2][2],
                    g_sink_momentum_list[ipf1][0], g_sink_momentum_list[ipf1][1], g_sink_momentum_list[ipf1][2],
                    g_seq2_source_momentum_list[ipf2][0], g_seq2_source_momentum_list[ipf2][1], g_seq2_source_momentum_list[ipf2][2],
                    gsx[0], gsx[1], gsx[2], gsx[3],
                    gamma_f1_nucleon_list[igf1], gamma_f1_nucleon_list[igi1]);

                /* B_1 */
                int perm[4] = {1,0,2,3};
                gamma_matrix_type gi1;
                gamma_matrix_set ( &gi1, gamma_f1_nucleon_list[igi1], gamma_f1_nucleon_sign[igi1] );

                gamma_matrix_mult ( &C_gi1, &gamma_C, &gi1 );
                gamma_matrix_transposed ( &C_gi1, &C_gi1);
                if ( g_verbose > 2 ) {
                  char name[20];
                  sprintf(name, "C_g%.2d_transposed", gi1.id);
                  gamma_matrix_printf (&C_gi1, name, stdout);
                }

                for ( int isample = 0; isample < g_nsample; isample++ ) {
                  exitstatus = contract_diagram_v2_gamma_v3 ( diagram_buffer, b1phi[igf1][ipf1][isample], b1xi[ipi2][0][ipf2][isample], C_gi1, perm, T_global, (int)(isample==0) );
                }

                /* transpose */
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
                for ( int it = 0; it < T_global ; it++ ) {
                  zm_4x4_array_transposed ( diagram[it], diagram_buffer[it] );
                }


                /* sprintf(aff_tag, "/%s/%s", "B1", aff_tag_suffix ); */
                sprintf(aff_tag, "/%s/%s", "/pixN-pixN/diag0", aff_tag_suffix );

                affdir = aff_writer_mkpath (affw, affn2, aff_tag );
                exitstatus = aff_node_put_complex (affw, affdir, diagram[0], (uint32_t)T_global*16);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_put_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }

                /* B_2 */
                perm[0] = 2;
                perm[1] = 0;
                perm[2] = 1;
                perm[3] = 3;

                for ( int isample = 0; isample < g_nsample; isample++ ) {
                  exitstatus = contract_diagram_v2_gamma_v3 ( diagram_buffer, b1phi[igf1][ipf1][isample], b1xi[ipi2][0][ipf2][isample], C_gi1, perm, T_global, (int)(isample==0) );
                }

                /* transpose */
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
                for ( int it = 0; it < T_global ; it++ ) {
                  zm_4x4_array_transposed ( diagram[it], diagram_buffer[it] );
                }

                /* sprintf(aff_tag, "/%s/%s", "B2", aff_tag_suffix ); */
                sprintf(aff_tag, "/%s/%s", "/pixN-pixN/diag1", aff_tag_suffix );

                affdir = aff_writer_mkpath (affw, affn2, aff_tag );
                exitstatus = aff_node_put_complex (affw, affdir, diagram[0], (uint32_t)T_global*16);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_put_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }
              }  /* end of loop on Gamma_i1 */
            }  /* end of loop on p_i2 */
          }  /* end of loop on Gamma_f1 */
        }  /* end of loop on p_f2 */

        free ( sink_momentum_id );

      }  /* end of loop on p_tot */

      fini_5level_zbuffer ( &b1phi );


      /*******************************************
       * w_1_xi
       *******************************************/
      double _Complex *****w1xi = NULL, ******w1phi = NULL, ******w3phi = NULL;
      exitstatus= init_5level_zbuffer ( &w1xi, gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[ piN2piN_correlators] Error from init_5level_zbuffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      strcpy ( tag, "w_1_xi" );
      fprintf(stdout, "\n\n# [ piN2piN_correlators] tag = %s\n", tag);

      for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

        for ( int isample = 0; isample < g_nsample; isample++ ) {

          if ( io_proc == 2 ) {
            aff_key_conversion ( aff_tag, tag, isample, NULL, NULL, g_seq2_source_momentum_list[ipf2], gsx, -1, -1 );
            if ( g_verbose > 2 ) fprintf(stdout, "# [ piN2piN_correlators] key = \"%s\"\n", aff_tag);
 
            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, w1xi[0][ipf2][isample][0], T_global*12);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_get_complex, status was %d\n", exitstatus);
              EXIT(105);
            }
          }

        }  /* end of loop on samples */
      }  /* end of loop on sink momentum pf2 */

      /*******************************************
       * w_1_phi
       *******************************************/
      exitstatus= init_6level_zbuffer ( &w1phi, g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[ piN2piN_correlators] Error from init_6level_zbuffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      strcpy ( tag, "w_1_phi" );
      fprintf(stdout, "\n\n# [ piN2piN_correlators] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, gamma_f1_nucleon_list[igf1], -1 );
                if (g_verbose > 2 ) fprintf(stdout, "# [ piN2piN_correlators] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w1phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_get_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }
              }

            }  /* end of loop on Gamma_f1 */
          }  /* end of loop on samples */
        }  /* end of loop on sink momentum pf1 */
      }  /* end of loop on seq source momentum pi2 */

   
      /*******************************************
       * w_3_phi
       *******************************************/
      exitstatus= init_6level_zbuffer ( &w3phi, g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[ piN2piN_correlators] Error from init_6level_zbuffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      strcpy ( tag, "w_3_phi" );
      fprintf(stdout, "\n\n# [ piN2piN_correlators] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {
        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {
          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {
            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, gamma_f1_nucleon_list[igf1], -1 );
                if (g_verbose > 2 ) fprintf(stdout, "# [ piN2piN_correlators] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w3phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_get_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }
              }

            }  /* end of loop on samples */
          }  /* end of loop on sink momentum pf1 */
        }  /* end of loop on Gamma_f1 */
      }  /* end of loop on seq source momentum pi2 */

      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int **sink_momentum_list = NULL, **sink_momentum_list_all = NULL;
        exitstatus = init_2level_ibuffer ( &sink_momentum_list, g_seq2_source_momentum_number, 3 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[ piN2piN_correlators] Error from init_2level_ibuffer, status was %d\n", exitstatus );
          EXIT(1);
        }
        exitstatus = init_2level_ibuffer ( &sink_momentum_list_all, g_sink_momentum_number, 3 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[ piN2piN_correlators] Error from init_2level_ibuffer, status was %d\n", exitstatus );
          EXIT(1);
        }
        for ( int ipf1 = 0; ipf1 < g_seq2_source_momentum_number; ipf1++ ) {
          sink_momentum_list[ipf1][0] = g_total_momentum_list[iptot][0] - g_seq2_source_momentum_list[ipf1][0];
          sink_momentum_list[ipf1][1] = g_total_momentum_list[iptot][1] - g_seq2_source_momentum_list[ipf1][1];
          sink_momentum_list[ipf1][2] = g_total_momentum_list[iptot][2] - g_seq2_source_momentum_list[ipf1][2];
        }
        for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {
          sink_momentum_list_all[ipf1][0] = g_sink_momentum_list[ipf1][0];
          sink_momentum_list_all[ipf1][1] = g_sink_momentum_list[ipf1][1];
          sink_momentum_list_all[ipf1][2] = g_sink_momentum_list[ipf1][2];
        }
        int *sink_momentum_id = NULL;
        exitstatus = match_momentum_id ( &sink_momentum_id, sink_momentum_list, sink_momentum_list_all, g_seq2_source_momentum_number, g_sink_momentum_number );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[ piN2piN_correlators] Error from match_momentum_id, status was %d\n", exitstatus );
          EXIT(1);
        }
        fini_2level_ibuffer ( &sink_momentum_list );
        fini_2level_ibuffer ( &sink_momentum_list_all );

        for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

          int ipf1 = sink_momentum_id[ipf2];
          if ( ipf1 == -1 ) continue;

          for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

            for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

              for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) {

                char aff_tag_suffix[200];
                sprintf(aff_tag_suffix, "pi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
                    g_seq_source_momentum_list[ipi2][0],  g_seq_source_momentum_list[ipi2][1],  g_seq_source_momentum_list[ipi2][2],
                    g_sink_momentum_list[ipf1][0], g_sink_momentum_list[ipf1][1], g_sink_momentum_list[ipf1][2],
                    g_seq2_source_momentum_list[ipf2][0], g_seq2_source_momentum_list[ipf2][1], g_seq2_source_momentum_list[ipf2][2],
                    gsx[0], gsx[1], gsx[2], gsx[3],
                    gamma_f1_nucleon_list[igf1], gamma_f1_nucleon_list[igi1]);


                /* W_1 */
                int perm[4] = {1,0,2,3};
                gamma_matrix_type gi1;
                gamma_matrix_set  ( &gi1, gamma_f1_nucleon_list[igi1], gamma_f1_nucleon_sign[igi1] );
                gamma_matrix_mult ( &C_gi1, &gamma_C, &gi1 );
                /* gamma_matrix_transposed ( &C_gi1, &C_gi1); */
                if ( g_verbose > 2 ) {
                  char name[20];
                  sprintf(name, "C_g%.2d", gi1.id);
                  gamma_matrix_printf (&C_gi1, name, stdout);
                }

                for ( int isample = 0; isample < g_nsample; isample++ ) {
                  exitstatus = contract_diagram_v2_gamma_v3 ( diagram_buffer, w1phi[ipi2][igf1][ipf1][isample], w1xi[0][ipf2][isample], C_gi1, perm, T_global, (int)(isample==0) );
                }

                /* transpose */
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
                for ( int it = 0; it < T_global ; it++ ) {
                  zm_4x4_array_transposed ( diagram[it], diagram_buffer[it] );
                }

                /* sprintf(aff_tag, "/%s/%s", "W1", aff_tag_suffix ); */
                sprintf(aff_tag, "/%s/%s", "/pixN-pixN/diag2", aff_tag_suffix );

                affdir = aff_writer_mkpath (affw, affn2, aff_tag );
                exitstatus = aff_node_put_complex (affw, affdir, diagram[0], (uint32_t)T_global*16);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_put_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }

                /* W_2 */
                perm[0] = 3;
                perm[1] = 0;
                perm[2] = 2;
                perm[3] = 1;

                for ( int isample = 0; isample < g_nsample; isample++ ) {
                  exitstatus = contract_diagram_v2_gamma_v3 ( diagram_buffer, w1phi[ipi2][igf1][ipf1][isample], w1xi[0][ipf2][isample], C_gi1, perm, T_global, (int)(isample==0) );
                }

                /* transpose */
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
                for ( int it = 0; it < T_global ; it++ ) {
                  zm_4x4_array_transposed ( diagram[it], diagram_buffer[it] );
                }

                /* sprintf(aff_tag, "/%s/%s", "W2", aff_tag_suffix );  */
                sprintf(aff_tag, "/%s/%s", "/pixN-pixN/diag3", aff_tag_suffix );

                affdir = aff_writer_mkpath (affw, affn2, aff_tag );
                exitstatus = aff_node_put_complex (affw, affdir, diagram[0], (uint32_t)T_global*16);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_put_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }


                /* W_3 */
                perm[0] = 2;
                perm[1] = 0;
                perm[2] = 3;
                perm[3] = 1;

                for ( int isample = 0; isample < g_nsample; isample++ ) {
                  exitstatus = contract_diagram_v2_gamma_v3 ( diagram_buffer, w3phi[ipi2][igf1][ipf1][isample], w1xi[0][ipf2][isample], C_gi1, perm, T_global, (int)(isample==0) );
                }

                /* transpose */
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
                for ( int it = 0; it < T_global ; it++ ) {
                  zm_4x4_array_transposed ( diagram[it], diagram_buffer[it] );
                }

                /* sprintf(aff_tag, "/%s/%s", "W3", aff_tag_suffix ); */
                sprintf(aff_tag, "/%s/%s", "/pixN-pixN/diag4", aff_tag_suffix );

                affdir = aff_writer_mkpath (affw, affn2, aff_tag );
                exitstatus = aff_node_put_complex (affw, affdir, diagram[0], (uint32_t)T_global*16);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_put_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }

                /* W_4 */
                perm[0] = 2;
                perm[1] = 0;
                perm[2] = 1;
                perm[3] = 3;

                for ( int isample = 0; isample < g_nsample; isample++ ) {
                  exitstatus = contract_diagram_v2_gamma_v3 ( diagram_buffer, w3phi[ipi2][igf1][ipf1][isample], w1xi[0][ipf2][isample], C_gi1, perm, T_global, (int)(isample==0) );
                }

                /* transpose */
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
                for ( int it = 0; it < T_global ; it++ ) {
                  zm_4x4_array_transposed ( diagram[it], diagram_buffer[it] );
                }

                /* sprintf(aff_tag, "/%s/%s", "W4", aff_tag_suffix ); */
                sprintf(aff_tag, "/%s/%s", "/pixN-pixN/diag5", aff_tag_suffix );

                affdir = aff_writer_mkpath (affw, affn2, aff_tag );
                exitstatus = aff_node_put_complex (affw, affdir, diagram[0], (uint32_t)T_global*16);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_put_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }

              }  /* end of loop on Gamma_i1 */
            }  /* end of loop on p_i2 */
          }  /* end of loop on Gamma_f1 */

        }  /* end of loop on p_f2 */

        free ( sink_momentum_id );

      }  /* end of loop on p_tot */

      fini_6level_zbuffer ( &w1phi );
      fini_6level_zbuffer ( &w3phi );

      fini_5level_zbuffer ( &w1xi );
#if 0
#endif  /* of if 0 */

    }  /* end of loop on coherent source locations */

    if(io_proc == 2) {
      aff_reader_close (affr);

      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[ piN2piN_correlators] Error from aff_writer_close, status was %s\n", aff_status_str);
        EXIT(111);
      }

    }  /* end of if io_proc == 2 */

    fini_6level_zbuffer ( &b1xi );
    fini_2level_zbuffer ( &diagram );
    fini_2level_zbuffer ( &diagram_buffer );

  }  /* end of loop on base source locations */


  /*******************************************/
  /*******************************************/
   

  /*******************************************
   * oet part
   *******************************************/

  /* loop on source locations */
  for( int i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];

    if(io_proc == 2) {

      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN_oet", Nconf, t_base );
      affr = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr(affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[ piN2piN_correlators] Error from aff_reader, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [ piN2piN_correlators] reading data from aff file %s\n", filename);
      }

      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN_oet_diagrams", Nconf, t_base );
      affw = aff_writer (filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[ piN2piN_correlators] Error from aff_writer, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [ piN2piN_correlators] writing data to file %s\n", filename);
      }

      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[ piN2piN_correlators] Error, aff reader is not initialized\n");
        EXIT(103);
      }

      if( (affn2 = aff_writer_root( affw )) == NULL ) {
        fprintf(stderr, "[ piN2piN_correlators] Error, aff writer is not initialized\n");
        EXIT(103);
      }
    }  /* end of if io_proc == 2 */

    double _Complex **diagram = NULL, **diagram_buffer = NULL;
    exitstatus= init_2level_zbuffer ( &diagram, T_global, 16 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[ piN2piN_correlators] Error from init_2level_zbuffer, status was %d\n", exitstatus);
      EXIT(47);
    }

    exitstatus= init_2level_zbuffer ( &diagram_buffer, T_global, 16 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[ piN2piN_correlators] Error from init_2level_zbuffer, status was %d\n", exitstatus);
      EXIT(47);
    }

    /* loop on coherent source locations */
    for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

      int source_proc_id, sx[4], gsx[4] = { t_coherent,
                    ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global,
                    ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global,
                    ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global };

      ratime = _GET_TIME;
      get_point_source_info (gsx, sx, &source_proc_id);

      double _Complex *****z1xi = NULL, ******z1phi = NULL, ******z3phi = NULL;
      int zero_momentum[3] = {0,0,0};

      /*******************************************
       * z_1_xi
       *******************************************/
      exitstatus= init_5level_zbuffer ( &z1xi, gamma_f2_number, g_seq2_source_momentum_number, 4, T_global, 12 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[ piN2piN_correlators] Error from init_5level_zbuffer, status was %d\n", exitstatus);
        EXIT(47);
      }
      strcpy ( tag, "z_1_xi" );
      fprintf(stdout, "\n\n# [ piN2piN_correlators] tag = %s\n", tag);

      for ( int igf2 = 0; igf2 < gamma_f2_number; igf2++ ) {
        for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {
          for ( int ispin = 0; ispin < 4; ispin++ ) {

            if ( io_proc == 2 ) {

              aff_key_conversion ( aff_tag, tag, 0, zero_momentum, NULL, g_seq2_source_momentum_list[ipf2], gsx, gamma_f2_list[igf2], ispin );
              if (g_verbose > 2 ) fprintf(stdout, "# [ piN2piN_correlators] key = \"%s\"\n", aff_tag);
 
              affdir = aff_reader_chpath (affr, affn, aff_tag );
              exitstatus = aff_node_get_complex (affr, affdir, z1xi[igf2][ipf2][ispin][0], T_global*12);
              if( exitstatus != 0 ) {
                fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_get_complex, status was %d\n", exitstatus);
                EXIT(105);
              }
            }
          }  /* end of loop on oet spin */
        }  /* end of loop on p_f2 */
      }  /* end of loop on Gamma_f2 */


      /*******************************************
       * z_3_phi
       *******************************************/
      exitstatus= init_6level_zbuffer ( &z3phi, g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[ piN2piN_correlators] Error from init_6level_zbuffer, status was %d\n", exitstatus);
        EXIT(47);
      }
      strcpy ( tag, "z_3_phi" );
      fprintf(stdout, "\n\n# [ piN2piN_correlators] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {
        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {
          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {
            for ( int ispin = 0; ispin < 4; ispin++ ) {

              if ( io_proc == 2 ) {

                aff_key_conversion ( aff_tag, tag, 0, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, gamma_f1_nucleon_list[igf1], ispin );
                if ( g_verbose > 2 ) fprintf(stdout, "# [ piN2piN_correlators] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, z3phi[ipi2][igf1][ipf1][ispin][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_get_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }
                
              }
            }  /* end of loop on oet spin index */
          }  /* end of loop on sink momentum p_f1 */
        }  /* end of loop on Gamma_f1 */
      }  /* end of loop on seq source momentum pi2 */

      /*******************************************
       * z_1_phi
       *******************************************/
      exitstatus= init_6level_zbuffer ( &z1phi, g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[ piN2piN_correlators] Error from init_6level_zbuffer, status was %d\n", exitstatus);
        EXIT(47);
      }
      strcpy ( tag, "z_1_phi" );
      fprintf(stdout, "\n\n# [ piN2piN_correlators] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {
        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {
          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {
            for ( int ispin = 0; ispin < 4; ispin++ ) {

              if ( io_proc == 2 ) {

                aff_key_conversion ( aff_tag, tag, 0, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, gamma_f1_nucleon_list[igf1], ispin );
                if ( g_verbose > 2 ) fprintf(stdout, "# [ piN2piN_correlators] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, z1phi[ipi2][igf1][ipf1][ispin][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_get_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }
                
              }
            }  /* end of loop on oet spin index */
          }  /* end of loop on sink momentum p_f1 */
        }  /* end of loop on Gamma_f1 */
      }  /* end of loop on seq source momentum pi2 */


      char name[20];
      gamma_matrix_type goet, gamma_5;
      gamma_matrix_set ( &goet, gamma_f2_list[0], gamma_f2_sign[0] );
      gamma_matrix_set ( &gamma_5, 5, 1 );
      gamma_matrix_mult ( &goet, &goet, &gamma_5 );
      sprintf(name, "goet_g5" );
      gamma_matrix_printf (&goet, name, stdout);


      /*******************************************/
      /*******************************************/


      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int **sink_momentum_list = NULL, **sink_momentum_list_all = NULL;
        exitstatus = init_2level_ibuffer ( &sink_momentum_list, g_seq2_source_momentum_number, 3 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[ piN2piN_correlators] Error from init_2level_ibuffer, status was %d\n", exitstatus );
          EXIT(1);
        }
        exitstatus = init_2level_ibuffer ( &sink_momentum_list_all, g_sink_momentum_number, 3 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[ piN2piN_correlators] Error from init_2level_ibuffer, status was %d\n", exitstatus );
          EXIT(1);
        }
        for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {
          sink_momentum_list_all[ipf1][0] = g_sink_momentum_list[ipf1][0];
          sink_momentum_list_all[ipf1][1] = g_sink_momentum_list[ipf1][1];
          sink_momentum_list_all[ipf1][2] = g_sink_momentum_list[ipf1][2];
        }
        for ( int ipf1 = 0; ipf1 < g_seq2_source_momentum_number; ipf1++ ) {
          sink_momentum_list[ipf1][0] = g_total_momentum_list[iptot][0] - g_seq2_source_momentum_list[ipf1][0];
          sink_momentum_list[ipf1][1] = g_total_momentum_list[iptot][1] - g_seq2_source_momentum_list[ipf1][1];
          sink_momentum_list[ipf1][2] = g_total_momentum_list[iptot][2] - g_seq2_source_momentum_list[ipf1][2];
        }
        int *sink_momentum_id = NULL;
        exitstatus = match_momentum_id ( &sink_momentum_id, sink_momentum_list, sink_momentum_list_all, g_seq2_source_momentum_number, g_sink_momentum_number );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[ piN2piN_correlators] Error from match_momentum_id, status was %d\n", exitstatus );
          EXIT(1);
        }
        fini_2level_ibuffer ( &sink_momentum_list );
        fini_2level_ibuffer ( &sink_momentum_list_all );

        for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

          int ipf1 = sink_momentum_id[ipf2];
          if ( ipf1 == -1 ) continue;

          for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

            for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

              for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) {

                char aff_tag_suffix[200];
                sprintf(aff_tag_suffix, "pi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
                    g_seq_source_momentum_list[ipi2][0],  g_seq_source_momentum_list[ipi2][1],  g_seq_source_momentum_list[ipi2][2],
                    g_sink_momentum_list[ipf1][0], g_sink_momentum_list[ipf1][1], g_sink_momentum_list[ipf1][2],
                    g_seq2_source_momentum_list[ipf2][0], g_seq2_source_momentum_list[ipf2][1], g_seq2_source_momentum_list[ipf2][2],
                    gsx[0], gsx[1], gsx[2], gsx[3],
                    gamma_f1_nucleon_list[igf1], gamma_f1_nucleon_list[igi1]);


                int perm[4] = {0,2,1,3};
                gamma_matrix_type gi1;
                gamma_matrix_set  ( &gi1, gamma_f1_nucleon_list[igi1], gamma_f1_nucleon_sign[igi1] );
                gamma_matrix_mult ( &C_gi1, &gamma_C, &gi1 );
                /* gamma_matrix_transposed ( &C_gi1, &C_gi1); */
                if ( g_verbose > 2 ) {
                  sprintf(name, "C_g%.2d", gi1.id);
                  gamma_matrix_printf (&C_gi1, name, stdout);
                }

                /* Z_1 */
                perm[0] = 0;
                perm[1] = 2;
                perm[2] = 1;
                perm[3] = 3;
                exitstatus = contract_diagram_oet_v2_gamma_v3 ( diagram_buffer, z1phi[ipi2][igf1][ipf1], z1xi[0][ipf2], goet, C_gi1, perm, T_global, 1 );

                /* transpose */
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
                for ( int it = 0; it < T_global ; it++ ) {
                  zm_4x4_array_transposed ( diagram[it], diagram_buffer[it] );
                }

                /* sprintf(aff_tag, "/%s/%s", "Z1", aff_tag_suffix ); */
                sprintf(aff_tag, "/%s/%s", "/pixN-pixN/sample0/diag0", aff_tag_suffix );

                affdir = aff_writer_mkpath (affw, affn2, aff_tag );
                exitstatus = aff_node_put_complex (affw, affdir, diagram[0], (uint32_t)T_global*16);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_put_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }

                /* Z_2 */
                perm[0] = 0;
                perm[1] = 2;
                perm[2] = 3;
                perm[3] = 1;
                exitstatus = contract_diagram_oet_v2_gamma_v3 ( diagram_buffer, z1phi[ipi2][igf1][ipf1], z1xi[0][ipf2], goet, C_gi1, perm, T_global, 1 );

                /* transpose */
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
                for ( int it = 0; it < T_global ; it++ ) {
                  zm_4x4_array_transposed ( diagram[it], diagram_buffer[it] );
                }

                /* sprintf(aff_tag, "/%s/%s", "Z2", aff_tag_suffix ); */
                sprintf(aff_tag, "/%s/%s", "/pixN-pixN/sample0/diag1", aff_tag_suffix );

                affdir = aff_writer_mkpath (affw, affn2, aff_tag );
                exitstatus = aff_node_put_complex (affw, affdir, diagram[0], (uint32_t)T_global*16);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_put_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }

                /* Z_3 */
                perm[0] = 2;
                perm[1] = 0;
                perm[2] = 3;
                perm[3] = 1;
                memset ( diagram[0], 0, 16*T_global*sizeof(double _Complex) );
                exitstatus = contract_diagram_oet_v2_gamma_v3 ( diagram_buffer, z3phi[ipi2][igf1][ipf1], z1xi[0][ipf2], goet, C_gi1, perm, T_global, 1 );

                /* transpose */
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
                for ( int it = 0; it < T_global ; it++ ) {
                  zm_4x4_array_transposed ( diagram[it], diagram_buffer[it] );
                }

                /* sprintf(aff_tag, "/%s/%s", "Z3", aff_tag_suffix ); */
                sprintf(aff_tag, "/%s/%s", "/pixN-pixN/sample0/diag2", aff_tag_suffix );

                affdir = aff_writer_mkpath (affw, affn2, aff_tag );
                exitstatus = aff_node_put_complex (affw, affdir, diagram[0], (uint32_t)T_global*16);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_put_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }

                /* Z_4 */
                perm[0] = 2;
                perm[1] = 0;
                perm[2] = 1;
                perm[3] = 3;
                exitstatus = contract_diagram_oet_v2_gamma_v3 ( diagram_buffer, z3phi[ipi2][igf1][ipf1], z1xi[0][ipf2], goet, C_gi1, perm, T_global, 1 );

                /* transpose */
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
                for ( int it = 0; it < T_global ; it++ ) {
                  zm_4x4_array_transposed ( diagram[it], diagram_buffer[it] );
                }

                /* sprintf(aff_tag, "/%s/%s", "Z4", aff_tag_suffix ); */
                sprintf(aff_tag, "/%s/%s", "/pixN-pixN/sample0/diag3", aff_tag_suffix );

                affdir = aff_writer_mkpath (affw, affn2, aff_tag );
                exitstatus = aff_node_put_complex (affw, affdir, diagram[0], (uint32_t)T_global*16);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[ piN2piN_correlators] Error from aff_node_put_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }

              }  /* end of loop on Gamma_i1 */
            }  /* end of loop on p_i2 */
          }  /* end of loop on Gamma_f1 */
        }  /* end of loop on p_f2 */

        free ( sink_momentum_id );

      }  /* end of loop on p_tot */

      fini_6level_zbuffer ( &z1phi );
      fini_6level_zbuffer ( &z3phi );
      fini_5level_zbuffer ( &z1xi );

    }  /* end of loop on coherent source locations */

    if(io_proc == 2) {
      aff_reader_close (affr);

      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[ piN2piN_correlators] Error from aff_writer_close, status was %s\n", aff_status_str);
        EXIT(111);
      }
    }  /* end of if io_proc == 2 */

    fini_2level_zbuffer ( &diagram );
    fini_2level_zbuffer ( &diagram_buffer );

  }  /* end of loop on base source locations */
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
    fprintf(stdout, "# [ piN2piN_correlators] %s# [ piN2piN_correlators] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [ piN2piN_correlators] %s# [ piN2piN_correlators] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
