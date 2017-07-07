/****************************************************
 * test_conversion.cpp
 * 
 * Thu Jul  6 10:52:21 CEST 2017
 *
 * PURPOSE:
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

#  ifdef HAVE_TMLQCD_LIBWRAPPER
#    include "tmLQCD.h"
#  endif

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
#include "contract_factorized.h"
#include "ranlxd.h"
/*  #include "C_gamma_to_gamma.h" */
#include "aff_key_conversion.h"

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
  double *tmLQCD_gauge_field = NULL;
  double *gauge_field_smeared = NULL;
  int read_stochastic_source = 0;
  int write_stochastic_source = 0;
  double **spinor_work = NULL;
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

  const int gamma_f1_nucleon_number = 4;
  int gamma_f1_nucleon_list[gamma_f1_nucleon_number] = { 5, 4,  6,  0 };


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

#ifdef HAVE_TMLQCD_LIBWRAPPER

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  /* exitstatus = tmLQCD_invert_init(argc, argv, 1, 0); */
  exitstatus = tmLQCD_invert_init(argc, argv, 1);
  if(exitstatus != 0) {
    EXIT(14);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    EXIT(15);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(exitstatus != 0) {
    EXIT(16);
  }
#endif

#ifdef HAVE_OPENMP
  omp_set_num_threads(g_num_threads);
#else
  fprintf(stdout, "[test_conversion] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

  /******************************************************
   *
   ******************************************************/

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_conversion] Error from init_geometry\n");
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
   * check source coords list
   ******************************************************/
  for ( int i = 0; i < g_source_location_number; i++ ) {
    g_source_coords_list[i][0] = ( g_source_coords_list[i][0] + T_global ) % T_global;
    g_source_coords_list[i][1] = ( g_source_coords_list[i][1] + LX_global ) % LX_global;
    g_source_coords_list[i][2] = ( g_source_coords_list[i][2] + LY_global ) % LY_global;
    g_source_coords_list[i][3] = ( g_source_coords_list[i][3] + LZ_global ) % LZ_global;
  }

  /* init_aff_key_conversion(); */

#if 0

  /* loop on source locations */
  for( int i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];

    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN", Nconf, t_base );
      fprintf(stdout, "# [test_conversion] writing data to file %s\n", filename);
      affr = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr(affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[test_conversion] Error from aff_reader, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [test_conversion] reading data from aff file %s\n", filename);
      }

      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[test_conversion] Error, aff writer is not initialized\n");
        EXIT(103);
      }
    }  /* end of if io_proc == 2 */

    double _Complex **v2p = NULL, **v3p = NULL;
    exitstatus= init_2level_zbuffer ( &v3p, T_global, 12 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[test_conversion] Error from init_2level_buffer, status was %d\n", exitstatus);
      EXIT(47);
    }

    exitstatus= init_2level_zbuffer ( &v2p, T_global, 192 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[test_conversion] Error from init_2level_buffer, status was %d\n", exitstatus);
      EXIT(47);
    }
    /*******************************************
     * b_1_xi
     *******************************************/
    strcpy ( tag, "b_1_xi" );
      fprintf(stdout, "\n\n# [test_conversion] tag = %s\n", tag);

    for ( int iseqmom = 0; iseqmom < g_seq_source_momentum_number; iseqmom++ ) {
      for ( int isnkmom = 0; isnkmom < g_sink_momentum_number; isnkmom++ ) {
        for ( int isample = 0; isample < g_nsample; isample++ ) {

          if ( io_proc == 2 ) {
            aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[iseqmom], NULL, g_sink_momentum_list[isnkmom], g_source_coords_list[i_src], -1, -1 );
            fprintf(stdout, "# [test_conversion] key = \"%s\"\n", aff_tag);
 
            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, v3p[0], T_global*12);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[test_conversion] Error from aff_node_get_complex, status was %d\n", exitstatus);
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
      strcpy ( tag, "b_1_phi" );
      fprintf(stdout, "\n\n# [test_conversion] tag = %s\n", tag);

      for ( int isnkmom = 0; isnkmom < g_sink_momentum_number; isnkmom++ ) {
        for ( int isample = 0; isample < g_nsample; isample++ ) {
          for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

            if ( io_proc == 2 ) {
              aff_key_conversion ( aff_tag, tag, isample, NULL, g_sink_momentum_list[isnkmom], NULL, gsx, gamma_f1_nucleon_list[igf1], -1 );
              fprintf(stdout, "# [test_conversion] key = \"%s\"\n", aff_tag);
 
              affdir = aff_reader_chpath (affr, affn, aff_tag );
              exitstatus = aff_node_get_complex (affr, affdir, v2p[0], T_global*192);
              if( exitstatus != 0 ) {
                fprintf(stderr, "[test_conversion] Error from aff_node_get_complex, status was %d\n", exitstatus);
                EXIT(105);
              }
            }

          }  /* end of loop on Gamma_f1 */
        }  /* end of loop on samples */
      }  /* end of loop on sink momentum pf1 */

      /*******************************************
       * w_1_xi
       *******************************************/
      strcpy ( tag, "w_1_xi" );
      fprintf(stdout, "\n\n# [test_conversion] tag = %s\n", tag);

      for ( int isnkmom = 0; isnkmom < g_sink_momentum_number; isnkmom++ ) {
        for ( int isample = 0; isample < g_nsample; isample++ ) {

          if ( io_proc == 2 ) {
            aff_key_conversion ( aff_tag, tag, isample, NULL, NULL, g_sink_momentum_list[isnkmom], gsx, -1, -1 );
            fprintf(stdout, "# [test_conversion] key = \"%s\"\n", aff_tag);
 
            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, v3p[0], T_global*12);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[test_conversion] Error from aff_node_get_complex, status was %d\n", exitstatus);
              EXIT(105);
            }
          }

        }  /* end of loop on samples */
      }  /* end of loop on sink momentum pf2 */

      /*******************************************
       * w_1_phi
       *******************************************/
      strcpy ( tag, "w_1_phi" );
      fprintf(stdout, "\n\n# [test_conversion] tag = %s\n", tag);

      for ( int iseqmom = 0; iseqmom < g_seq_source_momentum_number; iseqmom++ ) {
        for ( int isnkmom = 0; isnkmom < g_sink_momentum_number; isnkmom++ ) {
          for ( int isample = 0; isample < g_nsample; isample++ ) {
            for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[iseqmom], g_sink_momentum_list[isnkmom], NULL, gsx, gamma_f1_nucleon_list[igf1], -1 );
                fprintf(stdout, "# [test_conversion] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, v2p[0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[test_conversion] Error from aff_node_get_complex, status was %d\n", exitstatus);
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
      strcpy ( tag, "w_3_phi" );
      fprintf(stdout, "\n\n# [test_conversion] tag = %s\n", tag);

      for ( int iseqmom = 0; iseqmom < g_seq_source_momentum_number; iseqmom++ ) {
        for ( int isnkmom = 0; isnkmom < g_sink_momentum_number; isnkmom++ ) {
          for ( int isample = 0; isample < g_nsample; isample++ ) {
            for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[iseqmom], g_sink_momentum_list[isnkmom], NULL, gsx, gamma_f1_nucleon_list[igf1], -1 );
                fprintf(stdout, "# [test_conversion] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, v2p[0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[test_conversion] Error from aff_node_get_complex, status was %d\n", exitstatus);
                  EXIT(105);
                }
              }

            }  /* end of loop on Gamma_f1 */
          }  /* end of loop on samples */
        }  /* end of loop on sink momentum pf1 */
      }  /* end of loop on seq source momentum pi2 */

    }  /* end of loop on coherent source locations */

    fini_2level_zbuffer ( &v3p );
    fini_2level_zbuffer ( &v2p );

    if(io_proc == 2) {
      aff_reader_close (affr);
    }  /* end of if io_proc == 2 */

  }  /* end of loop on base source locations */

#endif  /* of if 0 */

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
        fprintf(stderr, "[test_conversion] Error from aff_reader, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [test_conversion] reading data from aff file %s\n", filename);
      }

      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN_oet_sorted", Nconf, t_base );
      affw = aff_writer (filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[test_conversion] Error from aff_writer, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [test_conversion] writing data to file %s\n", filename);
      }

      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[test_conversion] Error, aff reader is not initialized\n");
        EXIT(103);
      }

      if( (affn2 = aff_writer_root( affw )) == NULL ) {
        fprintf(stderr, "[test_conversion] Error, aff writer is not initialized\n");
        EXIT(103);
      }
    }  /* end of if io_proc == 2 */

    double _Complex **v2p = NULL, **v3p = NULL;
    exitstatus= init_2level_zbuffer ( &v3p, T_global, 4*12 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[test_conversion] Error from init_2level_buffer, status was %d\n", exitstatus);
      EXIT(47);
    }

    exitstatus= init_2level_zbuffer ( &v2p, T_global, 4*192 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[test_conversion] Error from init_2level_buffer, status was %d\n", exitstatus);
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


      /*******************************************
       * z_1_phi
       *******************************************/
      strcpy ( tag, "z_1_phi" );
      fprintf(stdout, "\n\n# [test_conversion] tag = %s\n", tag);

      for ( int iseqmom = 0; iseqmom < g_seq_source_momentum_number; iseqmom++ ) {
        for ( int isnkmom = 0; isnkmom < g_sink_momentum_number; isnkmom++ ) {
          for ( int isample = 0; isample < g_nsample_oet; isample++ ) {
            for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {
              /* for ( int ispin = 3; ispin >= 0; ispin-- ) { */

                if ( io_proc == 2 ) {
#if 0
                  aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[iseqmom], g_sink_momentum_list[isnkmom], NULL, gsx, gamma_f1_nucleon_list[igf1], ispin );
                  fprintf(stdout, "# [test_conversion] key = \"%s\"\n", aff_tag);
 

                  affdir = aff_reader_chpath (affr, affn, aff_tag );
                  exitstatus = aff_node_get_complex (affr, affdir, v2p[0], T_global*192);
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[test_conversion] Error from aff_node_get_complex, status was %d\n", exitstatus);
                    EXIT(105);
                  }

                  int LL[4] = {4,4,4,3};
                  int perm[4] = {0,2,1,3};
                  exitstatus = v2_key_index_conversion ( v2p[0], perm, T_global, LL );
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[test_conversion] Error from v2_key_index_conversion, status was %d\n", exitstatus);
                    EXIT(105);
                  }
#endif  /* of if 0 */
                  exitstatus = vn_oet_read_key ( v2p[0], tag, isample, g_seq_source_momentum_list[iseqmom], g_sink_momentum_list[isnkmom], NULL, gsx, gamma_f1_nucleon_list[igf1], affr );
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[test_conversion] Error from v2_oet_read_key, status was %d\n", exitstatus);
                    EXIT(105);
                  }

                  /* /z_1_phi/sample00/pi2x00pi2y00pi2z-01/pf1x00pf1y00pf1z-01/t01x02y03z04/g06 */

                  sprintf(aff_tag, "%s/sample%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2d",
                      tag, isample, 
                      g_seq_source_momentum_list[iseqmom][0], g_seq_source_momentum_list[iseqmom][1], g_seq_source_momentum_list[iseqmom][2],
                      g_sink_momentum_list[isnkmom][0], g_sink_momentum_list[isnkmom][1], g_sink_momentum_list[isnkmom][2],
                      gsx[0], gsx[1], gsx[2], gsx[3], gamma_f1_nucleon_list[igf1]);

                  affdir = aff_writer_mkpath (affw, affn2, aff_tag );
                  exitstatus = aff_node_put_complex (affw, affdir, v2p[0], T_global*4*192);
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[test_conversion] Error from aff_node_put_complex, status was %d\n", exitstatus);
                    EXIT(105);
                  }

                }

              /* } */  /* end of loop on oet spin index */ 
            }  /* end of loop on Gamma_f1 */
          }  /* end of loop on samples */
        }  /* end of loop on sink momentum pf1 */
      }  /* end of loop on seq source momentum pi2 */


      /*******************************************
       * z_3_phi
       *******************************************/
      strcpy ( tag, "z_3_phi" );
      fprintf(stdout, "\n\n# [test_conversion] tag = %s\n", tag);

      for ( int iseqmom = 0; iseqmom < g_seq_source_momentum_number; iseqmom++ ) {
        for ( int isnkmom = 0; isnkmom < g_sink_momentum_number; isnkmom++ ) {
          for ( int isample = 0; isample < g_nsample_oet; isample++ ) {
            for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {
              /* for ( int ispin = 3; ispin >= 0; ispin-- ) { */

                if ( io_proc == 2 ) {
#if 0
                  aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[iseqmom], g_sink_momentum_list[isnkmom], NULL, gsx, gamma_f1_nucleon_list[igf1], ispin );
                  fprintf(stdout, "# [test_conversion] key = \"%s\"\n", aff_tag);
 
                  affdir = aff_reader_chpath (affr, affn, aff_tag );
                  exitstatus = aff_node_get_complex (affr, affdir, v2p[0], T_global*192);
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[test_conversion] Error from aff_node_get_complex, status was %d\n", exitstatus);
                    EXIT(105);
                  }
                  int LL[4] = {4,4,4,3};
                  int perm[4] = {2,1,0,3};
                  exitstatus = v2_key_index_conversion ( v2p[0], perm, T_global, LL );
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[test_conversion] Error from v2_key_index_conversion, status was %d\n", exitstatus);
                    EXIT(105);
                  }
#endif

                  exitstatus = vn_oet_read_key ( v2p[0], tag, isample, g_seq_source_momentum_list[iseqmom], g_sink_momentum_list[isnkmom], NULL, gsx, gamma_f1_nucleon_list[igf1], affr );
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[test_conversion] Error from v2_oet_read_key, status was %d\n", exitstatus);
                    EXIT(105);
                  }

                  /* /z_3_phi/sample00/pi2x00pi2y00pi2z-01/pf1x00pf1y00pf1z-01/t09x06y07z00/g06 */

                  sprintf(aff_tag, "%s/sample%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2d",
                      tag, isample, 
                      g_seq_source_momentum_list[iseqmom][0], g_seq_source_momentum_list[iseqmom][1], g_seq_source_momentum_list[iseqmom][2],
                      g_sink_momentum_list[isnkmom][0], g_sink_momentum_list[isnkmom][1], g_sink_momentum_list[isnkmom][2],
                      gsx[0], gsx[1], gsx[2], gsx[3], gamma_f1_nucleon_list[igf1]);

                  affdir = aff_writer_mkpath (affw, affn2, aff_tag );
                  exitstatus = aff_node_put_complex (affw, affdir, v2p[0], T_global*4*192);
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[test_conversion] Error from aff_node_put_complex, status was %d\n", exitstatus);
                    EXIT(105);
                  }

                }

              /* } */  /* end of loop on oet spin index */
            }  /* end of loop on Gamma_f1 */
          }  /* end of loop on samples */
        }  /* end of loop on sink momentum pf1 */
      }  /* end of loop on seq source momentum pi2 */


      /*******************************************
       * z_1_xi
       *******************************************/
      strcpy ( tag, "z_1_xi" );
        fprintf(stdout, "\n\n# [test_conversion] tag = %s\n", tag);

      for ( int isnkmom = 0; isnkmom < g_sink_momentum_number; isnkmom++ ) {
        for ( int isample = 0; isample < g_nsample_oet; isample++ ) {
          /* for ( int ispin = 0; ispin < 4; ispin++ ) { */

            if ( io_proc == 2 ) {
#if 0
              aff_key_conversion ( aff_tag, tag, isample, NULL, NULL, g_sink_momentum_list[isnkmom], gsx, -1, ispin );
              fprintf(stdout, "# [test_conversion] key = \"%s\"\n", aff_tag);
 
              affdir = aff_reader_chpath (affr, affn, aff_tag );
              exitstatus = aff_node_get_complex (affr, affdir, v3p[0], T_global*12);
              if( exitstatus != 0 ) {
                fprintf(stderr, "[test_conversion] Error from aff_node_get_complex, status was %d\n", exitstatus);
                EXIT(105);
              }
#endif

             exitstatus = vn_oet_read_key ( v3p[0], tag, isample, NULL, NULL, g_sink_momentum_list[isnkmom], gsx, -1, affr );
             if( exitstatus != 0 ) {
               fprintf(stderr, "[test_conversion] Error from v2_oet_read_key, status was %d\n", exitstatus);
               EXIT(105);
             }

             /* /z_1_xi/sample00/pf2x00pf2y00pf2z-01/t01x02y03z04 */

             sprintf(aff_tag, "%s/sample%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/t%.2dx%.2dy%.2dz%.2d",
               tag, isample,
               g_sink_momentum_list[isnkmom][0], g_sink_momentum_list[isnkmom][1], g_sink_momentum_list[isnkmom][2],
               gsx[0], gsx[1], gsx[2], gsx[3] );

               affdir = aff_writer_mkpath (affw, affn2, aff_tag );
               exitstatus = aff_node_put_complex (affw, affdir, v3p[0], T_global*4*12);
               if( exitstatus != 0 ) {
                 fprintf(stderr, "[test_conversion] Error from aff_node_put_complex, status was %d\n", exitstatus);
                 EXIT(105);
               }
            }

          /* } */  /* end of loop oet spin index */
        }  /* end of loop on samples */
      }  /* end of loop on sink momentum pf2 */

    }  /* end of loop on coherent source locations */

    fini_2level_zbuffer ( &v3p );
    fini_2level_zbuffer ( &v2p );

    if(io_proc == 2) {
      aff_reader_close (affr);

      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[test_conversion] Error from aff_writer_close, status was %s\n", aff_status_str);
        EXIT(111);
      }
    }  /* end of if io_proc == 2 */

  }  /* end of loop on base source locations */


  /*******************************************
   * finalize
   *******************************************/

  /***********************************************
   * free the allocated memory, finalize
   ***********************************************/
  free_geometry();

  if( g_gauge_field != NULL ) free( g_gauge_field );

  if( gauge_field_smeared != NULL ) free( gauge_field_smeared );
#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_conversion] %s# [test_conversion] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_conversion] %s# [test_conversion] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
