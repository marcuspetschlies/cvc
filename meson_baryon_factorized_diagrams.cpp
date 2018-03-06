/****************************************************
 * meson_baryon_factorized_diagrams.cpp
 * 
 * Di 6. Mär 09:04:43 CET 2018
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
#include "matrix_init.h"
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
  char aff_tag[200];
#endif

  // pion-type gamma list at vertex i2
  const int gamma_i2_number = 2;
  int gamma_i2_list[gamma_i2_number]    = { 15,  7 };
  double gamma_i2_sign[gamma_i2_number] = { +1, -1 };

  // pion-type gamma list at vertex f2
  const int gamma_f2_number = 4;
  int gamma_f2_list[gamma_f2_number]    = { 15,  0,  8,  7 };
  double gamma_f2_sign[gamma_f2_number] = { +1, +1, +1, -1 };

  // Nucleon-type gamma list at vertex f1
  const int gamma_f1_nucleon_number = 4;
  int gamma_f1_nucleon_list[gamma_f1_nucleon_number]    = { 10,  5,  2, 13 };
  double gamma_f1_nucleon_sign[gamma_f1_nucleon_number] = { -1, -1, -1, -1 };

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
  fprintf(stdout, "[meson_baryon_factorized_diagrams] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

 /******************************************************
  * report git version
  ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [meson_baryon_factorized_diagrams] git version = %s\n", g_gitversion);
  }

  /******************************************************
   * set initial timestamp
   * - con: this is after parsing the input file
   ******************************************************/
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [meson_baryon_factorized_diagrams] %s# [meson_baryon_factorized_diagrams] start of run\n", ctime(&g_the_time));
    fflush(stdout);
  }

  /******************************************************
   *
   ******************************************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_geometry\n");
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
   * initialize all 16 gamma matrices in qlua basis
   * with binary counting
   ******************************************************/
  init_gamma_matrix ();

  gamma_matrix_type gamma[16];

  for ( int i = 0; i < 16; i++ ) {
    gamma_matrix_qlua_binary ( &(gamma[i]), i );

    char name[10];
    sprintf( name, "g%.2d", i );
    gamma_matrix_printf ( &(gamma[i]), name, stdout);
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

  /* loop on source locations */
  for( int i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];

    /******************************************************
     * open AFF input and output files
     ******************************************************/
    if(io_proc == 2) {
      /* AFF output file */
      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN_diagrams", Nconf, t_base );
      affw = aff_writer (filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from aff_writer, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [meson_baryon_factorized_diagrams] writing data to file %s\n", filename);
      }
    }  /* end of if io_proc == 2 */

    /*******************************************
     * v2 factors
     *
     *******************************************/
    double _Complex ***** b_v2_factor = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 );
    if ( b_v2_factor == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(48);
    }

    for ( int isample = 0; isample < g_nsample; isample++ ) {
      if(io_proc == 2) {
        /* AFF input files */
        sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_v2_phi_light", Nconf, t_base, isample );
        affr = aff_reader (filename);
        aff_status_str = (char*)aff_reader_errstr(affr);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from aff_reader, status was %s\n", aff_status_str);
          EXIT(4);
        } else {
          fprintf(stdout, "# [meson_baryon_factorized_diagrams] reading data from aff file %s\n", filename);
        }
        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
          EXIT(103);
        }
      }

      for ( int igf = 0; igf < gamma_f1_nucleon_number; igf++ ) {

        for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

          if ( io_proc == 2 ) {

            exitstatus = contract_diagram_read_key_qlua ( b_v2_factor[igf][ipf][isample], "phil-gf-fl-fl", -1, NULL, g_source_coords_list[i_src], isample, 2, gamma_f1_nucleon_list[igf], g_sink_momentum_list[ipf], affr, 192);

            if ( exitstatus != 0 ) {
              fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(104);
            }

          }  // end of if io_proc == 2

        }  /* end of loop on sink momentum pf2 */
      }  /* end of loop on sequential source momentum pi2 */

      if(io_proc == 2) {
        aff_reader_close (affr);
        aff_reader_close (affr_oet);

        aff_status_str = (char*)aff_writer_close (affw);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from aff_writer_close, status was %s\n", aff_status_str);
          EXIT(111);
        }
      }  /* end of if io_proc == 2 */

    }  /* end of loop on samples */

    /**************************************************************************************/
    /**************************************************************************************/

    /*******************************************
     * v3 factors
     *
     *******************************************/
    double _Complex ******* b_v3_factor = init_7level_ztable ( gamma_i2_number, g_seq_source_momentum_number, g_f2_number, g_sink_momentum_number, g_nsample, T_global, 12 );
    if ( b_v3_factor == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_7level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(48);
    }

    if(io_proc == 2) {
      /* AFF input files */
      sprintf(filename, "%s.%.4d.tbase%.2d.aff", "contract_v3_xi_light", Nconf, t_base );
      affr = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr(affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from aff_reader, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [meson_baryon_factorized_diagrams] reading data from aff file %s\n", filename);
      }
      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        EXIT(103);
      }
    }

    for ( int igi = 0; igi < gamma_i2_number; igi++ ) {

      for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {

        for ( int igf = 0; igf < gamma_f2_number; igf++ ) {

          for ( int ipf = 0; ipf < g_seq2_source_momentum_number; ipf++ ) {
 
            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {

                exitstatus = contract_diagram_read_key_qlua ( b_v3_factor[igi][ipi][igf][ipf][isample], "phil-gf-fl-fl", gamma_i2_list[igi], g_seq_source_momentum_list[ipi],
                    g_source_coords_list[i_src], isample, 3, gamma_f2_list[igf], g_seq2_source_momentum_list[ipf], affr, 12);

                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(104);
                }

              }  // end of if io_proc == 2

            }  // end of loop on isample

          }  // end of loop on pf2

        }  // end of loop on gf2

      }  // end of loop on pi2

    }  // end of loop on gi2

    if(io_proc == 2) {
      aff_reader_close (affr);
      aff_reader_close (affr_oet);

      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from aff_writer_close, status was %s\n", aff_status_str);
        EXIT(111);
      }
    }  /* end of if io_proc == 2 */

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * B-type contractions
     **************************************************************************************/


    /**************************************************************************************
     * loop on total momenta
     **************************************************************************************/
    for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

      int **sink_momentum_list = NULL, **sink_momentum_list_all = NULL;
      exitstatus = init_2level_ibuffer ( &sink_momentum_list, g_seq2_source_momentum_number, 3 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_2level_ibuffer, status was %d\n", exitstatus );
        EXIT(1);
      }
      exitstatus = init_2level_ibuffer ( &sink_momentum_list_all, g_sink_momentum_number, 3 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_2level_ibuffer, status was %d\n", exitstatus );
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
        fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from match_momentum_id, status was %d\n", exitstatus );
        EXIT(1);
      }
      fini_2level_ibuffer ( &sink_momentum_list );
      fini_2level_ibuffer ( &sink_momentum_list_all );

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * loop on gf1
       **************************************************************************************/
      for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

        /**************************************************************************************/
        /**************************************************************************************/

        /**************************************************************************************
         * loop on gf2
         **************************************************************************************/
        for ( int igf2 = 0; igf2 < gamma_f2_number; igf2++ ) {

          /**************************************************************************************/
          /**************************************************************************************/

          /**************************************************************************************
           * loop on pf2
           **************************************************************************************/
          for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

            /**************************************************************************************
             * pf2 is looped over, pf1 is set via total momentum
             **************************************************************************************/
            int ipf1 = sink_momentum_id[ipf2];
            if ( ipf1 == -1 ) continue;
           
            /**************************************************************************************/
            /**************************************************************************************/

            /**************************************************************************************
             * loop on gi2
             **************************************************************************************/
            for ( int igi2 = 0; igi2 < gamma_i2_number; igi2++ ) {

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

                  char aff_tag_suffix[200];

                  contract_diagram_key_suffix ( aff_tag_suffix, gamma_f2_list[igf2], g_seq2_source_momentum_list[ipf2], gamma_f1_nucleon_list[igf1], g_sink_momentum_list[ipf1], gamma_i2_list[igi2], g_seq_source_momentum_list[ipi2], 
                     gamma_f1_nucleon_list[igi1] );

                  /**************************************************************************************/
                  /**************************************************************************************/

                  double _Complex ****diagram = init_4level_ztable ( 2, T_global, 4, 4 );
                  if ( diagram == NULL ) {
                    fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_4level_ztable %s %d\n", __FILE__, __LINE__ );
                    EXIT(47);
                  }

                  /**************************************************************************************/
                  /**************************************************************************************/

                  /**************************************************************************************
                   * diagram B1
                   **************************************************************************************/
                  int perm[4] = {1,0,2,3};

                  /* reduce to diagram, average over stochastic samples */
                  if ( ( exitstatus = contract_diagram_sample ( diagram[0], b_v3_factor[igi2][ipi2][igf2][ipf2], b_v2_factor[igf1][ipf1], g_nsample, perm, gamma[gamma_f1_nucleon_list[igi1]] , T_global ) ) != 0 ) {
                    fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  /**************************************************************************************
                   * diagram B2
                   **************************************************************************************/
                  perm[0] = 2;
                  perm[1] = 0;
                  perm[2] = 1;
                  perm[3] = 3;

                  /* reduce to diagram, average over stochastic samples */
                  if ( ( exitstatus = contract_diagram_sample ( diagram[1], b_v3_factor[igi2][ipi2][igf2][ipf2], b_v2_factor[igf1][ipf1], g_nsample, perm, gamma[gamma_f1_nucleon_list[igi1]], T_global ) ) != 0 ) {
                    fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

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

                    /**************************************************************************************/
                    /**************************************************************************************/

                    /* AFF */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/b1/fwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[0], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /* AFF */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/b1/bwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[0], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/b2/fwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[1], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/b2/bwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[1], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                  }  // end of loop on coherent source locations

                  fini_4level_ztable ( &diagram );

                }  // end of loop on Gamma_i1
              }  // end of loop on p_i2
            }  // end of loop on Gamma_i2
          }  // end of loop on p_f2
        }  // end of loop on Gamma_f2
      }  // end of loop on Gamma_f1

      free ( sink_momentum_id );
    }  // end of loop on p_tot

    fini_5level_ztable ( &b_v2_factor );
    fini_7level_ztable ( &b_v3_factor );

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * W diagrams
     **************************************************************************************/

    /**************************************************************************************
     * v3 factors
     **************************************************************************************/
    double _Complex *****w_v3_factor = init_5level_ztable ( &w_v3_factor, gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
    if ( w_v3_factor == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(47);
    }

    for ( int isample = 0; isample < g_nsample; isample++ ) {

      if(io_proc == 2) {
        /* AFF input files */
        sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_v3_phi_light", Nconf, t_base , isample );
        affr = aff_reader (filename);
        aff_status_str = (char*)aff_reader_errstr(affr);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from aff_reader, status was %s\n", aff_status_str);
          EXIT(4);
        } else {
          fprintf(stdout, "# [meson_baryon_factorized_diagrams] reading data from aff file %s\n", filename);
        }
        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
          EXIT(103);
        }
      } // end of if io_proc 


      for ( int igf = 0; igf < gamma_f2_number; igf++ ) {

        for ( int ipf = 0; ipf < g_seq2_source_momentum_number; ipf++ ) {

          if ( io_proc == 2 ) {

            exitstatus = contract_diagram_read_key_qlua ( w_v3_factor[igf][ipf][isample], "g5.phi-gf-fl", -1, NULL, g_source_coords_list[i_src], isample, 3, gamma_f2_list[igf], g_seq2_source_momentum_list[ipf], affr, 12);

            if ( exitstatus != 0 ) {
              fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(104);
            }

          }  // end of if io_proc == 2

        }  // end of loop on seq2 source mometnum pf2 */

      }  // end of loop on Gamma_f2
      
      if(io_proc == 2) {
        aff_reader_close (affr);
        aff_reader_close (affr_oet);

        aff_status_str = (char*)aff_writer_close (affw);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from aff_writer_close, status was %s\n", aff_status_str);
          EXIT(111);
        }
      }  // end of if io_proc == 2

    }  // end of loop on samples

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * v2 factor
     **************************************************************************************/
    double _Complex ******* w_v2_factor = init_7level_ztable ( g_i2_number, g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number,  g_nsample, T_global, 192 ) ;
    if ( w_v2_factor == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_7level_ztable %s %d\n", __FILE__, __LINE__ );
       EXIT(47);
    }

    for ( int igi = 0; igi < gamma_i2_number; igi++ ) {

      for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {

                exitstatus = contract_diagram_read_key_qlua ( w_v2_factor[igi][ipi][igf][ipf][isample], "g5.xi-gf-fl-sll", gamma_i2_list[igi], g_seq_source_momentum_list[ipi],
                    g_source_coords_list[i_src], isample, 2, gamma_f1_nucleon_list[igf], g_sink_momentum_list[ipf], affr, 192);

                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(104);
                }

              }  // end of if io_proc == 2

            }  /* end of loop on samples */

          }  /* end of loop on sink momentum pf1 */

        }  /* end of loop on Gamma_f1 */

      }  /* end of loop on seq source momentum pi2 */

    }  /* end of loop on Gamma_i2 */

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * v2 factor, switched U and T
     **************************************************************************************/
    double _Complex ******* w_v2_factor2 = init_7level_ztable ( g_i2_number, g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number,  g_nsample, T_global, 192 ) ;
    if ( w_v2_factor2 == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_7level_ztable %s %d\n", __FILE__, __LINE__ );
       EXIT(47);
    }

    for ( int igi = 0; igi < gamma_i2_number; igi++ ) {

      for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {

                exitstatus = contract_diagram_read_key_qlua ( w_v2_factor2[igi][ipi][igf][ipf][isample], "g5.xi-gf-sll-fl", gamma_i2_list[igi], g_seq_source_momentum_list[ipi],
                    g_source_coords_list[i_src], isample, 2, gamma_f1_nucleon_list[igf], g_sink_momentum_list[ipf], affr, 192);

                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(104);
                }

              }  // end of if io_proc == 2

            }  /* end of loop on samples */

          }  /* end of loop on sink momentum pf1 */

        }  /* end of loop on Gamma_f1 */

      }  /* end of loop on seq source momentum pi2 */

    }  /* end of loop on Gamma_i2 */

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * contractions for W1 - W4
     **************************************************************************************/

    /**************************************************************************************
     * loop on total momentum
     **************************************************************************************/
    for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

      int **sink_momentum_list = NULL, **sink_momentum_list_all = NULL;
      exitstatus = init_2level_ibuffer ( &sink_momentum_list, g_seq2_source_momentum_number, 3 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_2level_ibuffer, status was %d\n", exitstatus );
        EXIT(1);
      }
      exitstatus = init_2level_ibuffer ( &sink_momentum_list_all, g_sink_momentum_number, 3 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_2level_ibuffer, status was %d\n", exitstatus );
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
        fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from match_momentum_id, status was %d\n", exitstatus );
        EXIT(1);
      }
      fini_2level_ibuffer ( &sink_momentum_list );
      fini_2level_ibuffer ( &sink_momentum_list_all );

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * loop on gf1
       **************************************************************************************/
      for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

        /**************************************************************************************/
        /**************************************************************************************/

        /**************************************************************************************
         * loop on gf2
         **************************************************************************************/
        for ( int igf2 = 0; igf2 < gamma_f2_number; igf2++ ) {

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
             * loop on gf2
             **************************************************************************************/
            for ( int igi2 = 0; igi2 < gamma_i2_number; igi2++ ) {

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

                  char aff_tag_suffix[200];

                  contract_diagram_key_suffix ( aff_tag_suffix, gamma_f2_list[igf2], g_seq2_source_momentum_list[ipf2], gamma_f1_nucleon_list[igf1], g_sink_momentum_list[ipf1], gamma_i2_list[igi2], g_seq_source_momentum_list[ipi2], 
                     gamma_f1_nucleon_list[igi1] );

                  /**************************************************************************************/
                  /**************************************************************************************/

                  double _Complex ****diagram = init_4level_ztable ( 4, T_global, 4, 4 );
                  if ( diagram == NULL ) {
                    fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_4level_ztable %s %d\n", __FILE__, __LINE__ );
                    EXIT(47);
                  }

                  /**************************************************************************************/
                  /**************************************************************************************/

                  /**************************************************************************************
                   * diagram W1
                   **************************************************************************************/
                  int perm[4] = {1,0,2,3};

                  /* reduce to diagram, average over stochastic samples */
                  if ( ( exitstatus = contract_diagram_sample ( diagram[0], w_v3_factor[igf2][ipf2], w_v2_factor[igi2][ipi2][igf1][ipf1], g_nsample, perm, gamma[gamma_f1_nucleon_list[igf1]], T_global ) ) != 0 ) {
                    fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  /**************************************************************************************
                   * W_2
                   **************************************************************************************/
                  perm[0] = 3;
                  perm[1] = 0;
                  perm[2] = 2;
                  perm[3] = 1;

                  /* reduce to diagram, average over stochastic samples */
                  if ( ( exitstatus = contract_diagram_sample ( diagram[1], w_v3_factor[igf2][ipf2], w_v2_factor[igi2][ipi2][igf1][ipf1], g_nsample, perm, gamma[gamma_f1_nucleon_list[igf1]], T_global ) ) != 0 ) {
                    fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  /**************************************************************************************
                   * W_3
                   **************************************************************************************/
                  perm[0] = 2;
                  perm[1] = 0;
                  perm[2] = 3;
                  perm[3] = 1;

                  /* reduce to diagram, average over stochastic samples */
                  if ( ( exitstatus = contract_diagram_sample ( diagram[2], w_v3_factor[igf2][ipf2], w_v2_factor2[igi2][ipi2][igf1][ipf1], g_nsample, perm, gamma[gamma_f1_nucleon_list[igf1]], T_global ) ) != 0 ) {
                    fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  /**************************************************************************************
                   * W_4
                   **************************************************************************************/
                  perm[0] = 2;
                  perm[1] = 0;
                  perm[2] = 1;
                  perm[3] = 3;

                  /* reduce to diagram, average over stochastic samples */
                  if ( ( exitstatus = contract_diagram_sample ( diagram[3], w_v3_factor[igf2][ipf2], w_v2_factor[igi2][ipi2][igf1][ipf1], g_nsample, perm, gamma[gamma_f1_nucleon_list[igf1]], T_global ) ) != 0 ) {
                    fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }


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

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w1/fwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[0], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w1/bwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[0], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /**************************************************************************************/
                    /**************************************************************************************/

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w2/fwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[1], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w2/bwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[1], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /**************************************************************************************/
                    /**************************************************************************************/

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w3/fwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[2], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w3/bwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[2], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /**************************************************************************************/
                    /**************************************************************************************/

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w4/fwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[3], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w4/bwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[3], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                  }  // end of loop on coherent source locations

                  fini_4level_ztable ( &diagram );

                }  /* end of loop on Gamma_i1 */

              }  /* end of loop on p_i2 */

            }  /* end of loop on Gamma_i2 */

          }  /* end of loop on p_f2 */

        }  /* end of loop on Gamma_f2 */

      }  /* end of loop on Gamma_f1 */

      free ( sink_momentum_id );

    }  /* end of loop on p_tot */

    fini_7level_ztable ( &w_v2_factor  );
    fini_7level_ztable ( &w_v2_factor2 );
    fini_5level_ztable ( &w_v3_factor  );

    /**************************************************************************************/
    /**************************************************************************************/


    /**************************************************************************************
     * oet part
     *
     * Z diagrams
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

      double _Complex *****z1xi = NULL, ******z1phi = NULL, ******z3phi = NULL;
      int zero_momentum[3] = {0,0,0};

      /**************************************************************************************/
      /**************************************************************************************/
    
      STOPPED HERE
      /**************************************************************************************
       * loop on oet sampls
       **************************************************************************************/
      for ( int isample = 0; isample < g_nsample_oet; isample++ ) {


        /**************************************************************************************
         * z_1_xi
         **************************************************************************************/
      exitstatus= init_5level_zbuffer ( &z1xi, gamma_f2_number, g_seq2_source_momentum_number, 4, T_global, 12 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_5level_zbuffer, status was %d\n", exitstatus);
        EXIT(47);
      }
      strcpy ( tag, "z_1_xi" );
      fprintf(stdout, "\n\n# [meson_baryon_factorized_diagrams] tag = %s\n", tag);

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
              if (g_verbose > 2 ) fprintf(stdout, "# [meson_baryon_factorized_diagrams] key = \"%s\"\n", aff_tag);
 
              affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
              exitstatus = aff_node_get_complex (affr_oet, affdir, z1xi[igf2][ipf2][ispin][0], T_global*12);
              if( exitstatus != 0 ) {
                fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d\n", aff_tag, exitstatus);
                EXIT(105);
              }
            }
          }  /* end of loop on oet spin */
        }  /* end of loop on p_f2 */
      }  /* end of loop on Gamma_f2 */

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * z_3_phi
       **************************************************************************************/
      exitstatus= init_6level_zbuffer ( &z3phi, g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_6level_zbuffer, status was %d\n", exitstatus);
        EXIT(47);
      }
      strcpy ( tag, "z_3_phi" );
      fprintf(stdout, "\n\n# [meson_baryon_factorized_diagrams] tag = %s\n", tag);

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
                if ( g_verbose > 2 ) fprintf(stdout, "# [meson_baryon_factorized_diagrams] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
                exitstatus = aff_node_get_complex (affr_oet, affdir, z3phi[ipi2][igf1][ipf1][ispin][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d\n", aff_tag, exitstatus);
                  EXIT(105);
                }
                
              }
            }  /* end of loop on oet spin index */
          }  /* end of loop on sink momentum p_f1 */
        }  /* end of loop on Gamma_f1 */
      }  /* end of loop on seq source momentum pi2 */

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * z_1_phi
       **************************************************************************************/
      exitstatus= init_6level_zbuffer ( &z1phi, g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_6level_zbuffer, status was %d\n", exitstatus);
        EXIT(47);
      }
      strcpy ( tag, "z_1_phi" );
      fprintf(stdout, "\n\n# [meson_baryon_factorized_diagrams] tag = %s\n", tag);

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
                if ( g_verbose > 2 ) fprintf(stdout, "# [meson_baryon_factorized_diagrams] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
                exitstatus = aff_node_get_complex (affr_oet, affdir, z1phi[ipi2][igf1][ipf1][ispin][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d\n", aff_tag, exitstatus);
                  EXIT(105);
                }
                
              }
            }  /* end of loop on oet spin index */
          }  /* end of loop on sink momentum p_f1 */
        }  /* end of loop on Gamma_f1 */
      }  /* end of loop on seq source momentum pi2 */

      /**************************************************************************************/
      /**************************************************************************************/

      char name[20];
      gamma_matrix_type goet, gamma_5;
      gamma_matrix_set ( &goet, gamma_f2_list[0], gamma_f2_sign[0] );
      gamma_matrix_set ( &gamma_5, 5, 1 );
      gamma_matrix_mult ( &goet, &goet, &gamma_5 );
      sprintf(name, "goet_g5" );
      gamma_matrix_printf (&goet, name, stdout);

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * loop on total momenta
       **************************************************************************************/
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int **sink_momentum_list = NULL, **sink_momentum_list_all = NULL;
        exitstatus = init_2level_ibuffer ( &sink_momentum_list, g_seq2_source_momentum_number, 3 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_2level_ibuffer, status was %d\n", exitstatus );
          EXIT(1);
        }
        exitstatus = init_2level_ibuffer ( &sink_momentum_list_all, g_sink_momentum_number, 3 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from init_2level_ibuffer, status was %d\n", exitstatus );
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
          fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from match_momentum_id, status was %d\n", exitstatus );
          EXIT(1);
        }
        fini_2level_ibuffer ( &sink_momentum_list );
        fini_2level_ibuffer ( &sink_momentum_list_all );

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

                char aff_tag_suffix[200];
                sprintf(aff_tag_suffix, "pi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
                    g_seq_source_momentum_list[ipi2][0],  g_seq_source_momentum_list[ipi2][1],  g_seq_source_momentum_list[ipi2][2],
                    g_sink_momentum_list[ipf1][0], g_sink_momentum_list[ipf1][1], g_sink_momentum_list[ipf1][2],
                    g_seq2_source_momentum_list[ipf2][0], g_seq2_source_momentum_list[ipf2][1], g_seq2_source_momentum_list[ipf2][2],
                    gsx[0], gsx[1], gsx[2], gsx[3],
                    gamma_f1_nucleon_list[igf1], gamma_f1_nucleon_list[igi1]);

                /**************************************************************************************/
                /**************************************************************************************/

                int perm[4] = {0,2,1,3};
                gamma_matrix_type gi1;
                gamma_matrix_set  ( &gi1, gamma_f1_nucleon_list[igi1], gamma_f1_nucleon_sign[igi1] );
                gamma_matrix_mult ( &C_gi1, &gamma_C, &gi1 );
                /* gamma_matrix_transposed ( &C_gi1, &C_gi1); */
                if ( g_verbose > 2 ) {
                  sprintf(name, "C_g%.2d", gi1.id);
                  gamma_matrix_printf (&C_gi1, name, stdout);
                }

                /**************************************************************************************/
                /**************************************************************************************/

                /**************************************************************************************
                 * Z_1
                 **************************************************************************************/
                perm[0] = 0;
                perm[1] = 2;
                perm[2] = 1;
                perm[3] = 3;

                if ( ( exitstatus = contract_diagram_sample_oet (diagram, z1xi[0][ipf2],  z1phi[ipi2][igf1][ipf1], goet, perm, C_gi1, T_global ) ) != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_sample_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }

                /* AFF tag */
                sprintf(aff_tag, "/%s/%s", "pixN-pixN/z1/fwd", aff_tag_suffix );
                if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(106);
                }

                /* AFF tag */
                sprintf(aff_tag, "/%s/%s", "pixN-pixN/z1/bwd", aff_tag_suffix );
                if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(106);
                }

                /**************************************************************************************/
                /**************************************************************************************/

                /**************************************************************************************
                 * Z_2
                 **************************************************************************************/
                perm[0] = 0;
                perm[1] = 2;
                perm[2] = 3;
                perm[3] = 1;

                if ( ( exitstatus = contract_diagram_sample_oet (diagram, z1xi[0][ipf2],  z1phi[ipi2][igf1][ipf1], goet, perm, C_gi1, T_global ) ) != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_sample_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }

                /* AFF tag */
                sprintf(aff_tag, "/%s/%s", "pixN-pixN/z2/fwd", aff_tag_suffix );
                if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(106);
                }

                /* AFF tag */
                sprintf(aff_tag, "/%s/%s", "pixN-pixN/z2/bwd", aff_tag_suffix );
                if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(106);
                }

                /**************************************************************************************/
                /**************************************************************************************/

                /**************************************************************************************
                 * Z_3
                 **************************************************************************************/
                perm[0] = 2;
                perm[1] = 0;
                perm[2] = 3;
                perm[3] = 1;

                if ( ( exitstatus = contract_diagram_sample_oet (diagram, z1xi[0][ipf2],  z3phi[ipi2][igf1][ipf1], goet, perm, C_gi1, T_global ) ) != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_sample_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }

                /* AFF tag */
                sprintf(aff_tag, "/%s/%s", "pixN-pixN/z3/fwd", aff_tag_suffix );
                if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(106);
                }

                /* AFF tag */
                sprintf(aff_tag, "/%s/%s", "pixN-pixN/z3/bwd", aff_tag_suffix );
                if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(106);
                }

                /**************************************************************************************/
                /**************************************************************************************/

                /**************************************************************************************
                 * Z_4
                 **************************************************************************************/
                perm[0] = 2;
                perm[1] = 0;
                perm[2] = 1;
                perm[3] = 3;

                if ( ( exitstatus = contract_diagram_sample_oet (diagram, z1xi[0][ipf2],  z3phi[ipi2][igf1][ipf1], goet, perm, C_gi1, T_global ) ) != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_sample_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }

                /* AFF tag */
                sprintf(aff_tag, "/%s/%s", "pixN-pixN/z4/fwd", aff_tag_suffix );
                if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(106);
                }

                /* AFF tag */
                sprintf(aff_tag, "/%s/%s", "pixN-pixN/z4/bwd", aff_tag_suffix );
                if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(106);
                }

              }  /* end of loop on Gamma_i1 */
            }  /* end of loop on p_i2 */
          }  /* end of loop on Gamma_f1 */
        }  /* end of loop on p_f2 */

        free ( sink_momentum_id );

      }  /* end of loop on p_tot */

      /**************************************************************************************/
      /**************************************************************************************/

      fini_6level_zbuffer ( &z1phi );
      fini_6level_zbuffer ( &z3phi );
      fini_5level_zbuffer ( &z1xi );

    }  /* end of loop on coherent source locations */

    fini_3level_zbuffer ( &diagram );

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * direct diagrams
     **************************************************************************************/

    exitstatus= init_3level_zbuffer ( &diagram, T_global, 4, 4 );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[piN2piN_diagrams] Error from init_3level_zbuffer, status was %d\n", exitstatus);
      EXIT(47);
    }

    /* loop on coherent source locations */
    for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

      int source_proc_id, sx[4], gsx[4] = { t_coherent,
                    ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global,
                    ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global,
                    ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global };

      get_point_source_info (gsx, sx, &source_proc_id);

      double _Complex ******bb = NULL, *****mm = NULL, ***bb_aux = NULL;

      /**************************************************************************************
       * bb_aux
       **************************************************************************************/
      exitstatus= init_3level_zbuffer ( &bb_aux, T_global, 4, 4 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_3level_zbuffer, status was %d\n", exitstatus);
        EXIT(47);
      }

      /**************************************************************************************
       * bb
       **************************************************************************************/
      exitstatus= init_6level_zbuffer ( &bb, gamma_f1_nucleon_number, gamma_f1_nucleon_number, g_sink_momentum_number, 2, T_global, 16 );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_6level_zbuffer, status was %d\n", exitstatus);
        EXIT(47);
      }
      strcpy ( tag, "N-N" );
      fprintf(stdout, "\n\n# [piN2piN_diagrams] tag = %s\n", tag);

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
                  fprintf(stderr, "[piN2piN_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d\n", aff_tag, exitstatus);
                  EXIT(105);
                }
              }
            }  /* end of loop on diagrams */
          }  /* end of loop on sink momentum */
        }  /* end of loop on Gamma_f1 */
      }  /* end of loop on Gamma_i1 */

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * mm
       **************************************************************************************/
      exitstatus= init_5level_zbuffer ( &mm, g_seq_source_momentum_number, gamma_f2_number, gamma_f2_number, g_sink_momentum_number, T_global );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_5level_zbuffer, status was %d\n", exitstatus);
        EXIT(47);
      }
      strcpy ( tag, "m-m" );
      fprintf(stdout, "\n\n# [piN2piN_diagrams] tag = %s\n", tag);

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * loop on pi2
       **************************************************************************************/
      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        /**************************************************************************************
         * loop on gi2
         **************************************************************************************/
        for ( int igi2 = 0; igi2 < gamma_f2_number; igi2++ ) {

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

                aff_key_conversion_diagram (  aff_tag, tag, NULL, g_seq_source_momentum_list[ipi2], NULL, g_sink_momentum_list[ipf2], -1, gamma_f2_list[igi2], -1, gamma_f2_list[igf2], gsx, NULL, 0  );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
                exitstatus = aff_node_get_complex (affr_oet, affdir, mm[ipi2][igi2][igf2][ipf2], T_global);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams] Error from aff_node_get_complex for key \"%s\", status was %d\n", aff_tag, exitstatus);
                  EXIT(105);
                }
                if ( g_verbose > 3 ) {
                  for ( int it = 0; it < T_global; it++ ) {
                    fprintf(stdout, "# [piN2piN_diagrams] m-m %3d %25.16e %25.16e\n", it, 
                        creal( mm[ipi2][igi2][igf2][ipf2][it] ), cimag( mm[ipi2][igi2][igf2][ipf2][it] ));
                  }
                }
              }  
            }  /* end of loop on sink momentum */
          }  /* end of loop on Gamma_f2 */
        }  /* end of loop on Gamma_i2 */
      }  /* end of loop in pi2 */

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * loop on total momenta
       **************************************************************************************/
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int **sink_momentum_list = NULL, **sink_momentum_list_all = NULL;
        exitstatus = init_2level_ibuffer ( &sink_momentum_list, g_seq2_source_momentum_number, 3 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_diagrams] Error from init_2level_ibuffer, status was %d\n", exitstatus );
          EXIT(1);
        }
        exitstatus = init_2level_ibuffer ( &sink_momentum_list_all, g_sink_momentum_number, 3 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_diagrams] Error from init_2level_ibuffer, status was %d\n", exitstatus );
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
          fprintf(stderr, "[piN2piN_diagrams] Error from match_momentum_id, status was %d\n", exitstatus );
          EXIT(1);
        }
        fini_2level_ibuffer ( &sink_momentum_list );
        fini_2level_ibuffer ( &sink_momentum_list_all );

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

                char aff_tag_suffix[200];
                sprintf(aff_tag_suffix, "pi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/t%.2dx%.2dy%.2dz%.2d/g%.2dg%.2d",
                    g_seq_source_momentum_list[ipi2][0],  g_seq_source_momentum_list[ipi2][1],  g_seq_source_momentum_list[ipi2][2],
                    g_sink_momentum_list[ipf1][0], g_sink_momentum_list[ipf1][1], g_sink_momentum_list[ipf1][2],
                    g_seq2_source_momentum_list[ipf2][0], g_seq2_source_momentum_list[ipf2][1], g_seq2_source_momentum_list[ipf2][2],
                    gsx[0], gsx[1], gsx[2], gsx[3],
                    gamma_f1_nucleon_list[igf1], gamma_f1_nucleon_list[igi1]);


                /**************************************************************************************/
                /**************************************************************************************/

                /**************************************************************************************
                 * loop on diagrams
                 **************************************************************************************/
                for ( int idiag = 0; idiag < 2; idiag++ ) {

                  memcpy( bb_aux[0][0], bb[igi1][igf1][ipf1][idiag][0], 16*T_global*sizeof(double _Complex) );

                  /* multiply baryon 2-point function with meson 2-point function */
                  exitstatus = contract_diagram_zmx4x4_field_ti_co_field ( diagram, bb_aux,  mm[ipi2][igi2][igf2][ipf2], T_global );
                  // memcpy(diagram[0][0],  bb_aux[0][0], 16*T_global*sizeof(double _Complex) );

                  /* transpose */
                  exitstatus = contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( diagram, diagram, T_global );

                  /* AFF */
                  sprintf(aff_tag, "/%s/s%d/fwd/%s", "pixN-pixN", idiag+1, aff_tag_suffix );
                  if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] aff tag = %s\n", aff_tag);
                  if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d\n", exitstatus);
                    EXIT(105);
                  }
   
                  /* AFF */
                  sprintf(aff_tag, "/%s/s%d/bwd/%s", "pixN-pixN", idiag+1, aff_tag_suffix );
                  if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams] aff tag = %s\n", aff_tag);
                  if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d\n", exitstatus);
                    EXIT(105);
                  }

                }  /* end of loop on diagrams */
              }  /* end of loop on Gamma_i1 */
            }  /* end of loop on p_i2 */
          }  /* end of loop on Gamma_f1 */
        }  /* end of loop on p_f2 */

        free ( sink_momentum_id );

      }  /* end of loop on p_tot */

      /**************************************************************************************/
      /**************************************************************************************/

      fini_6level_zbuffer ( &bb );
      fini_5level_zbuffer ( &mm );
      fini_3level_zbuffer ( &bb_aux );


    }  /* end of loop on coherent source locations */

    fini_3level_zbuffer ( &diagram );

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
        fprintf(stderr, "[piN2piN_diagrams] Error from aff_writer_close, status was %s\n", aff_status_str);
        EXIT(111);
      }
    }  /* end of if io_proc == 2 */

  }  /* end of loop on base source locations */

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
