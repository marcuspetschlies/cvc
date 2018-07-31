/****************************************************
 * meson_baryon_factorized_diagrams_2pt.cpp
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
  int io_proc = -1;

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  struct AffWriter_s *affw = NULL;
  char * aff_status_str;
  struct AffNode_s *affn = NULL;
#endif

  // pion-type gamma list at vertex i2
  const int gamma_i2_number = 2;
  int gamma_i2_list[gamma_i2_number]    = { 15,  7 };
  // double gamma_i2_sign[gamma_i2_number] = { +1, -1 };

  // pion-type gamma list at vertex f2
  // const int gamma_f2_number = 4;
  // int gamma_f2_list[gamma_f2_number]    = { 15,  0,  8,  7 };
  const int gamma_f2_number = 2;
  int gamma_f2_list[gamma_f2_number]    = { 15,  7 };
  // double gamma_f2_sign[gamma_f2_number] = { +1, -1 };

  // Nucleon-type gamma list at vertex f1
  const int gamma_f1_nucleon_number = 4;
  int gamma_f1_nucleon_list[gamma_f1_nucleon_number]    = { 10,  5,  2, 13 };
  // double gamma_f1_nucleon_sign[gamma_f1_nucleon_number] = { -1, -1, -1, -1 };

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
  fprintf(stdout, "[meson_baryon_factorized_diagrams_2pt] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

 /******************************************************
  * report git version
  ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [meson_baryon_factorized_diagrams_2pt] git version = %s\n", g_gitversion);
  }

  /******************************************************
   * set initial timestamp
   * - con: this is after parsing the input file
   ******************************************************/
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [meson_baryon_factorized_diagrams_2pt] %s# [meson_baryon_factorized_diagrams_2pt] start of run\n", ctime(&g_the_time));
    fflush(stdout);
  }

  /******************************************************
   *
   ******************************************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from init_geometry\n");
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

  /**************************************************************************************/
  /**************************************************************************************/

  /* loop on source locations */
  for( int i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];

    /******************************************************
     * open AFF input and output files
     ******************************************************/
    if(io_proc == 2) {
      /* AFF output file */
      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "meson_baryon_factorized_diagrams_2pt", Nconf, t_base );
      affw = aff_writer (filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from aff_writer, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [meson_baryon_factorized_diagrams_2pt] writing data to file %s\n", filename);
      }
    }  /* end of if io_proc == 2 */

    /**************************************************************************************/
    /**************************************************************************************/

#if 0

    /**************************************************************************************
     * B diagrams
     **************************************************************************************/

    /**************************************************************************************/
    /**************************************************************************************/

    /*******************************************
     * v2 factors
     *
     *******************************************/
    double _Complex ***** b_v2_factor = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 );
    if ( b_v2_factor == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(48);
    }

    for ( int isample = 0; isample < g_nsample; isample++ ) {
      if(io_proc == 2) {
        /* AFF input files */
        sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_v2_phi_light", Nconf, t_base, isample );
        affr = aff_reader (filename);
        aff_status_str = (char*)aff_reader_errstr(affr);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from aff_reader, status was %s\n", aff_status_str);
          EXIT(4);
        } else {
          fprintf(stdout, "# [meson_baryon_factorized_diagrams_2pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
        }
        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
          EXIT(103);
        }
      }

      for ( int igf = 0; igf < gamma_f1_nucleon_number; igf++ ) {

        for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

          if ( io_proc == 2 ) {

            exitstatus = contract_diagram_read_key_qlua ( b_v2_factor[igf][ipf][isample], "phil-gf-fl-fl", -1, NULL, g_source_coords_list[i_src], isample, "v2", gamma_f1_nucleon_list[igf], g_sink_momentum_list[ipf], affr, T_global, 192);

            if ( exitstatus != 0 ) {
              fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(104);
            }

          }  // end of if io_proc == 2

        }  /* end of loop on sink momentum pf2 */
      }  /* end of loop on sequential source momentum pi2 */

      if(io_proc == 2) {
        aff_reader_close (affr);
      }  /* end of if io_proc == 2 */

    }  /* end of loop on samples */

    /**************************************************************************************/
    /**************************************************************************************/

    /*******************************************
     * v3 factors
     *
     *******************************************/
    double _Complex ******* b_v3_factor = init_7level_ztable ( gamma_i2_number, g_seq_source_momentum_number, gamma_f2_number, g_sink_momentum_number, g_nsample, T_global, 12 );
    if ( b_v3_factor == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from init_7level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(48);
    }

    if(io_proc == 2) {
      /* AFF input files */
      sprintf(filename, "%s.%.4d.tbase%.2d.aff", "contract_v3_xi_light", Nconf, t_base );
      affr = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr(affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from aff_reader, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [meson_baryon_factorized_diagrams_2pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
      }
      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        EXIT(103);
      }
    }

    for ( int igi = 0; igi < gamma_i2_number; igi++ ) {

      for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {

        for ( int igf = 0; igf < gamma_f2_number; igf++ ) {

          for ( int ipf = 0; ipf < g_seq2_source_momentum_number; ipf++ ) {
 
            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {

                exitstatus = contract_diagram_read_key_qlua ( b_v3_factor[igi][ipi][igf][ipf][isample], "xil-gf-sll", gamma_i2_list[igi], g_seq_source_momentum_list[ipi],
                    g_source_coords_list[i_src], isample, "v3", gamma_f2_list[igf], g_seq2_source_momentum_list[ipf], affr, T_global, 12);

                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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

      int * sink_momentum_id =  get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, g_total_momentum_list[iptot], g_sink_momentum_list, g_sink_momentum_number );
      if ( sink_momentum_id == NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      }

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
                    fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from init_4level_ztable %s %d\n", __FILE__, __LINE__ );
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
                    fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
                    fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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

                    char aff_tag[500];

                    /* AFF */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/b1/fwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[0], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /* AFF */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/b1/bwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[0], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/b2/fwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[1], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/b2/bwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[1], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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

#endif  // of if 0

    /**************************************************************************************/
    /**************************************************************************************/

#if 0

    /**************************************************************************************
     * W diagrams
     **************************************************************************************/

    /**************************************************************************************
     * v3 factors
     **************************************************************************************/
    double _Complex *****w_v3_factor = init_5level_ztable ( gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
    if ( w_v3_factor == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(47);
    }

    for ( int isample = 0; isample < g_nsample; isample++ ) {

      if(io_proc == 2) {
        /* AFF input files */
        sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_v3_phi_light", Nconf, t_base , isample );
        affr = aff_reader (filename);
        aff_status_str = (char*)aff_reader_errstr(affr);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from aff_reader, status was %s\n", aff_status_str);
          EXIT(4);
        } else {
          fprintf(stdout, "# [meson_baryon_factorized_diagrams_2pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
        }
        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
          EXIT(103);
        }
      } // end of if io_proc 


      for ( int igf = 0; igf < gamma_f2_number; igf++ ) {

        for ( int ipf = 0; ipf < g_seq2_source_momentum_number; ipf++ ) {

          if ( io_proc == 2 ) {

            exitstatus = contract_diagram_read_key_qlua ( w_v3_factor[igf][ipf][isample], "g5.phil-gf-fl", -1, NULL, g_source_coords_list[i_src], isample, "v3", gamma_f2_list[igf], g_seq2_source_momentum_list[ipf], affr, T_global, 12);

            if ( exitstatus != 0 ) {
              fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(104);
            }

          }  // end of if io_proc == 2

        }  // end of loop on seq2 source mometnum pf2 */

      }  // end of loop on Gamma_f2
      
      if(io_proc == 2) {
        aff_reader_close (affr);
      }  // end of if io_proc == 2

    }  // end of loop on samples

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * v2 factor
     **************************************************************************************/
    double _Complex ******* w_v2_factor = init_7level_ztable ( gamma_i2_number, g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number,  g_nsample, T_global, 192 ) ;
    if ( w_v2_factor == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from init_7level_ztable %s %d\n", __FILE__, __LINE__ );
       EXIT(47);
    }

    if(io_proc == 2) {
      /* AFF input files */
      sprintf(filename, "%s.%.4d.tbase%.2d.aff", "contract_v2_xi_light", Nconf, t_base );
      affr = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr(affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from aff_reader, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [meson_baryon_factorized_diagrams_2pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
      }
      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        EXIT(103);
      }
    } // end of if io_proc 


    for ( int igi = 0; igi < gamma_i2_number; igi++ ) {

      for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {

        for ( int igf = 0; igf < gamma_f1_nucleon_number; igf++ ) {

          for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {

                exitstatus = contract_diagram_read_key_qlua ( w_v2_factor[igi][ipi][igf][ipf][isample], "g5.xil-gf-fl-sll", gamma_i2_list[igi], g_seq_source_momentum_list[ipi],
                    g_source_coords_list[i_src], isample, "v2", gamma_f1_nucleon_list[igf], g_sink_momentum_list[ipf], affr, T_global, 192);

                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
    double _Complex ******* w_v2_factor2 = init_7level_ztable ( gamma_i2_number, g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number,  g_nsample, T_global, 192 ) ;
    if ( w_v2_factor2 == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from init_7level_ztable %s %d\n", __FILE__, __LINE__ );
       EXIT(47);
    }

    for ( int igi = 0; igi < gamma_i2_number; igi++ ) {

      for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {

        for ( int igf = 0; igf < gamma_f1_nucleon_number; igf++ ) {

          for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {

                exitstatus = contract_diagram_read_key_qlua ( w_v2_factor2[igi][ipi][igf][ipf][isample], "g5.xil-gf-sll-fl", gamma_i2_list[igi], g_seq_source_momentum_list[ipi],
                    g_source_coords_list[i_src], isample, "v2", gamma_f1_nucleon_list[igf], g_sink_momentum_list[ipf], affr, T_global, 192);

                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(104);
                }

              }  // end of if io_proc == 2

            }  /* end of loop on samples */

          }  /* end of loop on sink momentum pf1 */

        }  /* end of loop on Gamma_f1 */

      }  /* end of loop on seq source momentum pi2 */

    }  /* end of loop on Gamma_i2 */

    if(io_proc == 2) {
      aff_reader_close (affr);
    }  // end of if io_proc == 2

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * contractions for W1 - W4
     **************************************************************************************/

    /**************************************************************************************
     * loop on total momentum
     **************************************************************************************/
    for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

      int * sink_momentum_id =  get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, g_total_momentum_list[iptot], g_sink_momentum_list, g_sink_momentum_number );
      if ( sink_momentum_id == NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      }

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
                    fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from init_4level_ztable %s %d\n", __FILE__, __LINE__ );
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
                    fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
                    fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
                    fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
                  if ( ( exitstatus = contract_diagram_sample ( diagram[3], w_v3_factor[igf2][ipf2], w_v2_factor2[igi2][ipi2][igf1][ipf1], g_nsample, perm, gamma[gamma_f1_nucleon_list[igf1]], T_global ) ) != 0 ) {
                    fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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

                    char aff_tag[500];

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w1/fwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[0], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w1/bwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[0], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /**************************************************************************************/
                    /**************************************************************************************/

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w2/fwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[1], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w2/bwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[1], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /**************************************************************************************/
                    /**************************************************************************************/

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w3/fwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[2], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w3/bwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[2], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /**************************************************************************************/
                    /**************************************************************************************/

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w4/fwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[3], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(106);
                    }

                    /* AFF key */
                    sprintf(aff_tag, "/%s/%s", "pixN-pixN/w4/bwd", aff_tag_suffix );
                    if ( ( exitstatus = contract_diagram_write_aff ( diagram[3], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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

#endif  // of if 0

    /**************************************************************************************/
    /**************************************************************************************/

#if 0

    /**************************************************************************************
     * oet part
     **************************************************************************************/

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
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

      /**************************************************************************************/
      /**************************************************************************************/
  
      /**************************************************************************************
       * loop on oet samples
       *
       * we loop on samples here, because we likely will only have one
       **************************************************************************************/
      for ( int isample = 0; isample < g_nsample_oet; isample++ ) {


        /**************************************************************************************
         * read v3_factor
         **************************************************************************************/
        double _Complex  ******z_v3_factor = init_6level_ztable ( g_seq_source_momentum_number, gamma_f2_number, g_seq2_source_momentum_number, 4, T_global, 12 );
        if ( z_v3_factor == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from init_6level_ztable, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(47);
        }

        if(io_proc == 2) {
          /* AFF input files */
          sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_oet_v3_light", Nconf, t_base , isample );
          affr = aff_reader (filename);
          aff_status_str = (char*)aff_reader_errstr(affr);
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from aff_reader, status was %s\n", aff_status_str);
            EXIT(4);
          } else {
            fprintf(stdout, "# [meson_baryon_factorized_diagrams_2pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
          }
          if( (affn = aff_reader_root( affr )) == NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
            EXIT(103);
          }
        } // end of if io_proc == 2


        /**************************************************************************************/
        /**************************************************************************************/

        /**************************************************************************************
         * loop on pi2
         **************************************************************************************/
        for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {

          /**************************************************************************************
           * loop on gf2
           **************************************************************************************/
          for ( int igf = 0; igf < gamma_f2_number; igf++ ) {

            /**************************************************************************************
             * loop on pf2
             **************************************************************************************/
            for ( int ipf = 0; ipf < g_seq2_source_momentum_number; ipf++ ) {

              if ( io_proc == 2 ) {
 
                exitstatus = contract_diagram_read_oet_key_qlua ( z_v3_factor[ipi][igf][ipf], "phil-gf-fl", g_seq_source_momentum_list[ipi], gsx, "v3", gamma_f2_list[igf], g_seq2_source_momentum_list[ipf], affr, T_global, 12 );

                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_read_oet_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(50);
                }

              }  // end of if io_proc == 0

            }  /* end of loop on p_f2 */

          }  /* end of loop on Gamma_f2 */

        }  /* end of loop on pi2 */

        if(io_proc == 2) {
          aff_reader_close (affr);
        }  // end of if io_proc == 2

        /**************************************************************************************/
        /**************************************************************************************/

        /**************************************************************************************
         * read v2 factor
         **************************************************************************************/
        double _Complex ***** z_v2_factor = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
        if ( z_v2_factor == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from init_5level_ztable, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(47);
        }
 
        if(io_proc == 2) {
          /* AFF input files */
          sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_oet_v2_light", Nconf, t_base , isample );
          affr = aff_reader (filename);
          aff_status_str = (char*)aff_reader_errstr(affr);
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from aff_reader, status was %s\n", aff_status_str);
            EXIT(4);
          } else {
            fprintf(stdout, "# [meson_baryon_factorized_diagrams_2pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
          }
          if( (affn = aff_reader_root( affr )) == NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
            EXIT(103);
          }
        } // end of if io_proc == 2

        /**************************************************************************************
         * loop on gf1
         **************************************************************************************/
        for ( int igf = 0; igf < gamma_f1_nucleon_number; igf++ ) {

          /**************************************************************************************
           * loop on pf1
           **************************************************************************************/
          for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

            if ( io_proc == 2 ) {

              exitstatus = contract_diagram_read_oet_key_qlua ( z_v2_factor[igf][ipf], "phil-gf-fl-fl", NULL, gsx, "v2", gamma_f1_nucleon_list[igf], g_sink_momentum_list[ipf], affr, T_global, 192 );

              if ( exitstatus != 0 ) {
                fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_read_oet_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(51);
              }
              
            }   // end of if io_proc == 2

          }  /* end of loop on sink momentum p_f1 */

        }  /* end of loop on Gamma_f1 */

        if(io_proc == 2) {
          aff_reader_close (affr);
        }  // end of if io_proc == 2

        /**************************************************************************************/
        /**************************************************************************************/
 
        /**************************************************************************************
         * read v4 factor 
         **************************************************************************************/
        double _Complex ***** z_v4_factor = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
        if ( z_v4_factor == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from init_5level_ztable, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(47);
        }

        if(io_proc == 2) {
          /* AFF input files */
          sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_oet_v4_light", Nconf, t_base , isample );
          affr = aff_reader (filename);
          aff_status_str = (char*)aff_reader_errstr(affr);
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from aff_reader, status was %s\n", aff_status_str);
            EXIT(4);
          } else {
            fprintf(stdout, "# [meson_baryon_factorized_diagrams_2pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
          }
          if( (affn = aff_reader_root( affr )) == NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
            EXIT(103);
          }
        } // end of if io_proc == 2

        /**************************************************************************************
         * loop on gf1
         **************************************************************************************/
        for ( int igf = 0; igf < gamma_f1_nucleon_number; igf++ ) {

          /**************************************************************************************
           * loop on pf1
           **************************************************************************************/
          for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

            if ( io_proc == 2 ) {

              exitstatus = contract_diagram_read_oet_key_qlua ( z_v4_factor[igf][ipf], "phil-gf-fl-fl", NULL, gsx, "v4", gamma_f1_nucleon_list[igf], g_sink_momentum_list[ipf], affr, T_global, 192 );

              if ( exitstatus != 0 ) {
                fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_read_oet_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(51);
              }
              
            }  // end of if io_proc == 2

          }  /* end of loop on sink momentum p_f1 */

        }  /* end of loop on Gamma_f1 */

        if(io_proc == 2) {
          aff_reader_close (affr);
        }  // end of if io_proc == 2


        /**************************************************************************************/
        /**************************************************************************************/

        /**************************************************************************************
         * contractions for Z diagrams
         **************************************************************************************/

        /**************************************************************************************/
        /**************************************************************************************/

        /**************************************************************************************
         * loop on total momenta
         **************************************************************************************/
        for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

          int * sink_momentum_id =  get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, g_total_momentum_list[iptot], g_sink_momentum_list, g_sink_momentum_number );
          if ( sink_momentum_id == NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }

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
               * loop on gf1
               **************************************************************************************/
              for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

                /**************************************************************************************/
                /**************************************************************************************/

                /**************************************************************************************
                 * loop on gi2
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
  
                      int perm[4];
  
                      double _Complex ****diagram = init_4level_ztable ( 4, T_global, 4, 4 );
                      if ( diagram == NULL ) {
                        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from init_4level_ztable %s %d\n", __FILE__, __LINE__ );
                        EXIT(47);
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
  
                      if ( ( exitstatus = contract_diagram_sample_oet (diagram[0], z_v3_factor[ipi2][igf2][ipf2],  z_v4_factor[igf1][ipf1], gamma[gamma_i2_list[igi2]], perm, gamma[gamma_f1_nucleon_list[igi1]], T_global ) ) != 0 ) {
                        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_sample_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(105);
                      }
  
                      /**************************************************************************************
                       * Z_2
                       **************************************************************************************/
                      perm[0] = 0;
                      perm[1] = 2;
                      perm[2] = 3;
                      perm[3] = 1;
  
                      if ( ( exitstatus = contract_diagram_sample_oet (diagram[1], z_v3_factor[ipi2][igf2][ipf2],  z_v4_factor[igf1][ipf1], gamma[gamma_i2_list[igi2]], perm, gamma[gamma_f1_nucleon_list[igi1]], T_global ) ) != 0 ) {
                        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_sample_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(105);
                      }
  
                      /**************************************************************************************
                       * Z_3
                       **************************************************************************************/
                      perm[0] = 2;
                      perm[1] = 0;
                      perm[2] = 3;
                      perm[3] = 1;
  
                      if ( ( exitstatus = contract_diagram_sample_oet (diagram[2], z_v3_factor[ipi2][igf2][ipf2], z_v2_factor[igf1][ipf1], gamma[gamma_i2_list[igi2]], perm, gamma[gamma_f1_nucleon_list[igi1]], T_global ) ) != 0 ) {
                        fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_sample_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(105);
                      }
  
                      /**************************************************************************************
                       * Z_4
                       **************************************************************************************/
                      perm[0] = 2;
                      perm[1] = 0;
                      perm[2] = 1;
                      perm[3] = 3;
  
                      if ( ( exitstatus = contract_diagram_sample_oet (diagram[3], z_v3_factor[ipi2][igf2][ipf2], z_v2_factor[igf1][ipf1], gamma[gamma_i2_list[igi2]], perm, gamma[gamma_f1_nucleon_list[igi1]], T_global ) ) != 0 ) {
                        fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_sample_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(105);
                      }
  
                      /**************************************************************************************/
                      /**************************************************************************************/
  
                      char aff_tag[500];

                      /* AFF tag */
                      sprintf(aff_tag, "/%s/%s", "pixN-pixN/z1/fwd", aff_tag_suffix );
                      if ( ( exitstatus = contract_diagram_write_aff ( diagram[0], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(106);
                      }
  
                      /* AFF tag */
                      sprintf(aff_tag, "/%s/%s", "pixN-pixN/z1/bwd", aff_tag_suffix );
                      if ( ( exitstatus = contract_diagram_write_aff ( diagram[0], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(106);
                      }
  
                      /**************************************************************************************/
                      /**************************************************************************************/
  
                      /* AFF tag */
                      sprintf(aff_tag, "/%s/%s", "pixN-pixN/z2/fwd", aff_tag_suffix );
                      if ( ( exitstatus = contract_diagram_write_aff ( diagram[1], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(106);
                      }
  
                      /* AFF tag */
                      sprintf(aff_tag, "/%s/%s", "pixN-pixN/z2/bwd", aff_tag_suffix );
                      if ( ( exitstatus = contract_diagram_write_aff ( diagram[1], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                        fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(106);
                      }
  
                      /**************************************************************************************/
                      /**************************************************************************************/
  
                      /* AFF tag */
                      sprintf(aff_tag, "/%s/%s", "pixN-pixN/z3/fwd", aff_tag_suffix );
                      if ( ( exitstatus = contract_diagram_write_aff ( diagram[2], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                        fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(106);
                      }
  
                     /* AFF tag */
                      sprintf(aff_tag, "/%s/%s", "pixN-pixN/z3/bwd", aff_tag_suffix );
                      if ( ( exitstatus = contract_diagram_write_aff ( diagram[2], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                        fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(106);
                      }
  
                      /**************************************************************************************/
                      /**************************************************************************************/
  
                      /* AFF tag */
                      sprintf(aff_tag, "/%s/%s", "pixN-pixN/z4/fwd", aff_tag_suffix );
                      if ( ( exitstatus = contract_diagram_write_aff ( diagram[3], affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                        fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(106);
                      }
  
                      /* AFF tag */
                      sprintf(aff_tag, "/%s/%s", "pixN-pixN/z4/bwd", aff_tag_suffix );
                      if ( ( exitstatus = contract_diagram_write_aff ( diagram[3], affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                        fprintf(stderr, "[piN2piN_diagrams] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                        EXIT(106);
                      }

                      /**************************************************************************************/
                      /**************************************************************************************/

                      fini_4level_ztable ( &diagram );
  
                    }  // end of loop on Gamma_i1

                  }  // end of loop on p_i2
 
                }  // end of loop on Gamma_i2

              }  // end of loop on Gamma_f1
  
            }  // end of loop on p_f2

          }  // end of loop on Gamma_f2

          free ( sink_momentum_id );

        }  // end of loop on p_tot



        /**************************************************************************************/
        /**************************************************************************************/

        fini_6level_ztable ( &z_v3_factor );
        fini_5level_ztable ( &z_v2_factor );
        fini_5level_ztable ( &z_v4_factor );

      }  // end of loop on oet sampels

    }  /* end of loop on coherent source locations */
    
#endif  // of if 0

    /**************************************************************************************/
    /**************************************************************************************/


    /**************************************************************************************
     * direct diagrams
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

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * read baryon 2-point factors
       **************************************************************************************/

      double _Complex ***** bb_t1_factor = init_5level_ztable ( gamma_f1_nucleon_number, gamma_f1_nucleon_number, g_sink_momentum_number, T_global, 16 );
      if ( bb_t1_factor == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(52);
      }

      double _Complex ***** bb_t2_factor = init_5level_ztable ( gamma_f1_nucleon_number, gamma_f1_nucleon_number, g_sink_momentum_number, T_global, 16 );
      if ( bb_t2_factor == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(53);
      }

      if(io_proc == 2) {
        /* AFF input files */
        sprintf(filename, "%s.%.4d.tbase%.2d.aff", "contract_2pt_light", Nconf, t_base );
        affr = aff_reader (filename);
        aff_status_str = (char*)aff_reader_errstr(affr);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from aff_reader, status was %s\n", aff_status_str);
          EXIT(4);
        } else {
          fprintf(stdout, "# [meson_baryon_factorized_diagrams_2pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
        }
        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
          EXIT(103);
        }
      } // end of if io_proc == 2

      /**************************************************************************************
       * loop on gf1
       **************************************************************************************/
      for ( int igf = 0; igf < gamma_f1_nucleon_number; igf++ ) {

        /**************************************************************************************
         * loop on pf1
         **************************************************************************************/
        for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

          /**************************************************************************************
           * loop on gi1
           **************************************************************************************/
          for ( int igi = 0; igi < gamma_f1_nucleon_number; igi++ ) {

            if ( io_proc == 2 ) {

              exitstatus = contract_diagram_read_key_qlua ( bb_t1_factor[igi][igf][ipf], "fl-fl", gamma_f1_nucleon_list[igi], NULL, gsx, -1, "t1", gamma_f1_nucleon_list[igf], g_sink_momentum_list[ipf], affr, T_global, 16);
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(114);
              }

              exitstatus = contract_diagram_read_key_qlua ( bb_t2_factor[igi][igf][ipf], "fl-fl", gamma_f1_nucleon_list[igi], NULL, gsx, -1, "t2", gamma_f1_nucleon_list[igf], g_sink_momentum_list[ipf], affr, T_global, 16);
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(115);
              }

            }  // end of if io_proc == 2

          }  /* end of loop on Gamma_i1 */

        }  /* end of loop on sink momentum */

      }  /* end of loop on Gamma_f1 */

      if(io_proc == 2) {
        aff_reader_close (affr);
      }  // end of if io_proc == 2

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * loop on stochastic oet samples
       **************************************************************************************/
      for ( int isample = 0; isample < g_nsample_oet; isample++ ) {


        /**************************************************************************************
         * read mm factor
         **************************************************************************************/
        double _Complex ****** mm_m1_factor =  init_6level_ztable ( gamma_i2_number, g_seq_source_momentum_number, gamma_f2_number, g_seq2_source_momentum_number, T_global , 1 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[piN2piN_diagrams] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__);
          EXIT(147);
        }
  
        if(io_proc == 2) {
          /* AFF input files */
          sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_oet_m_m_2pt_light", Nconf, t_base, isample );
          affr = aff_reader (filename);
          aff_status_str = (char*)aff_reader_errstr(affr);
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from aff_reader, status was %s\n", aff_status_str);
            EXIT(4);
          } else {
            fprintf(stdout, "# [meson_baryon_factorized_diagrams_2pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
          }
          if( (affn = aff_reader_root( affr )) == NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
            EXIT(103);
          }
        } // end of if io_proc == 2

        /**************************************************************************************/
        /**************************************************************************************/

        /**************************************************************************************
         * loop on gi2
         **************************************************************************************/
        for ( int igi = 0; igi < gamma_i2_number; igi++ ) {

          /**************************************************************************************
           * loop on pi2
           **************************************************************************************/
          for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {


            /**************************************************************************************
             * loop on gf2
             **************************************************************************************/
            for ( int igf = 0; igf < gamma_f2_number; igf++ ) {

              /**************************************************************************************
               * loop on gf2
               **************************************************************************************/
              for ( int ipf = 0; ipf < g_seq2_source_momentum_number; ipf++ ) {

                if ( io_proc == 2 ) {

                  exitstatus = contract_diagram_read_key_qlua ( mm_m1_factor[igi][ipi][igf][ipf], "fl-fl", gamma_i2_list[igi], g_seq_source_momentum_list[ipi], gsx, 
                      -1, "m1", gamma_f2_list[igf], g_seq2_source_momentum_list[ipf], affr, T_global, 1);
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(114);
                  }

                }  // end of if io_proc == 2

              }  // end of loop on pf2

            }  /* end of loop on Gamma_f2 */

          }  /* end of loop in pi2 */

        }  /* end of loop on Gamma_i2 */

        if(io_proc == 2) {
          aff_reader_close (affr);
        }  // end of if io_proc == 2

        /**************************************************************************************/
        /**************************************************************************************/

        /**************************************************************************************
         * contractions for direct diagrams
         **************************************************************************************/

        /**************************************************************************************/
        /**************************************************************************************/

        /**************************************************************************************
         * loop on total momenta
         **************************************************************************************/
        for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

          /**************************************************************************************
           * sink momentum ids
           **************************************************************************************/
          int *sink_momentum_id = get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, g_total_momentum_list[iptot], g_sink_momentum_list, g_sink_momentum_number );
          if ( sink_momentum_id == NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from get_minus_momentum_id %s %d\n", __FILE__, __LINE__ );
            EXIT(146);
          }

          /**************************************************************************************
           * loop on gf2
           **************************************************************************************/
          for ( int igf2 = 0; igf2 < gamma_f2_number; igf2++ ) {

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

                /**************************************************************************************
                 * loop on gi2
                 **************************************************************************************/
                for ( int igi2 = 0; igi2 < gamma_i2_number; igi2++ ) {

                  /**************************************************************************************
                   * loop on pi2
                   **************************************************************************************/
                  for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

                    /**************************************************************************************
                     * loop on gi1
                     **************************************************************************************/
                    for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) { 

                      char aff_tag_suffix[200];

                      contract_diagram_key_suffix ( aff_tag_suffix, gamma_f2_list[igf2], g_seq2_source_momentum_list[ipf2], gamma_f1_nucleon_list[igf1], g_sink_momentum_list[ipf1], gamma_i2_list[igi2], g_seq_source_momentum_list[ipi2],
                          gamma_f1_nucleon_list[igi1] );


                      /**************************************************************************************/
                      /**************************************************************************************/

                      double _Complex ***diagram = init_3level_ztable ( T_global, 4, 4 );
                      if ( diagram == NULL ) {
                        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
                        EXIT(142);
                      }

                      double _Complex ***bb_aux = init_3level_ztable ( T_global, 4, 4 );
                      if ( bb_aux == NULL ) {
                        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
                        EXIT(147);
                      }

                      char aff_tag[500];

                      /**************************************************************************************
                       * S1
                       **************************************************************************************/
                      memcpy( bb_aux[0][0], bb_t1_factor[igi1][igf1][ipf1][0], 16*T_global*sizeof(double _Complex) );

                      // multiply baryon 2-point function with meson 2-point function
                      exitstatus = contract_diagram_zm4x4_field_ti_co_field ( diagram, bb_aux, mm_m1_factor[igi2][ipi2][igf2][ipf2][0], T_global );

                      // transpose
                      exitstatus = contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( diagram, diagram, T_global );

                      /* AFF */
                      sprintf(aff_tag, "/%s/s1/fwd/%s", "pixN-pixN", aff_tag_suffix );
                      if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d\n", exitstatus);
                        EXIT(154);
                      }
   
                      /* AFF */
                      sprintf(aff_tag, "/%s/s1/bwd/%s", "pixN-pixN", aff_tag_suffix );
                      if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d\n", exitstatus);
                        EXIT(155);
                      }


                      /**************************************************************************************
                       * S2
                       **************************************************************************************/
                      memcpy( bb_aux[0][0], bb_t2_factor[igi1][igf1][ipf1][0], 16*T_global*sizeof(double _Complex) );

                      // multiply baryon 2-point function with meson 2-point function
                      exitstatus = contract_diagram_zm4x4_field_ti_co_field ( diagram, bb_aux, mm_m1_factor[igi2][ipi2][igf2][ipf2][0], T_global );

                      // transpose
                      exitstatus = contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( diagram, diagram, T_global );

                      /* AFF */
                      sprintf(aff_tag, "/%s/s2/fwd/%s", "pixN-pixN", aff_tag_suffix );
                      if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d\n", exitstatus);
                        EXIT(154);
                      }
   
                      /* AFF */
                      sprintf(aff_tag, "/%s/s2/bwd/%s", "pixN-pixN", aff_tag_suffix );
                      if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                        fprintf(stderr, "[meson_baryon_factorized_diagrams_2pt] Error from contract_diagram_write_aff, status was %d\n", exitstatus);
                        EXIT(155);
                      }

                      fini_3level_ztable ( &diagram );
                      fini_3level_ztable ( &bb_aux );

                    }  // end of loop on Gamma_i1

                  }  // end of loop on pi2

                }  // end of loop on Gamma_i2

              }  // end of loop on Gamma_f1

            }  // end of loop on pf2

          }  // end of loop on Gamma_f2

          free ( sink_momentum_id );

        }  /* end of loop on p_tot */
#if 0
#endif  // of if 0
        /**************************************************************************************/
        /**************************************************************************************/

        fini_5level_ztable ( &bb_t1_factor );
        fini_5level_ztable ( &bb_t2_factor );
        fini_6level_ztable ( &mm_m1_factor );

      }  // end of loop on stochastic oet samples

    }  // end of loop coherent source timeslices


    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * close AFF readers for input and output files
     **************************************************************************************/
    if(io_proc == 2) {
      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from aff_writer_close, status was %s\n", aff_status_str);
        EXIT(171);
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
