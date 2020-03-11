/****************************************************
 * meson_baryon_factorized_diagrams_3pt
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

#define MESON_BARYON_FACTORIZED_DIAGRAMS_2PT

#undef MESON_BARYON_FACTORIZED_DIAGRAMS_3PT


#ifdef MESON_BARYON_FACTORIZED_DIAGRAMS_2PT
#warning "using MESON_BARYON_FACTORIZED_DIAGRAMS_2PT"
#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT
#warning "using MESON_BARYON_FACTORIZED_DIAGRAMS_3PT"
#else
#error "need either MESON_BARYON_FACTORIZED_DIAGRAMS_2PT or MESON_BARYON_FACTORIZED_DIAGRAMS_3PT defined"
#endif

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
  int const gamma_i2_number = 1;
  //                                              g5   g5 gt ~ gx gy gz
  // int const gamma_i2_list[gamma_i2_number] = { 15,  7 };
  int const gamma_i2_list[gamma_i2_number]    = { 0 };

#if defined MESON_BARYON_FACTORIZED_DIAGRAMS_2PT
  // pion-type gamma list at vertex f2
  int const gamma_f2_number = 1;
  //                                           g5   1  gt gtg5 
  // int const gamma_f2_list[gamma_f2_number] = { 15 };
  // int const gamma_f2_list[gamma_f2_number] = { 15,  0,  8,   7 };
  int const gamma_f2_list[gamma_f2_number] = { 0 };

#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT
  // current vertex
  int const gamma_f2_number = 6;
  int const gamma_f2_list[gamma_f2_number] = { 1, 2, 4, 14, 13, 11 } ;
#else
#error "need MESON_BARYON_FACTORIZED_DIAGRAMS_<2/3>PT defined" 
#endif

  // Nucleon-type gamma list at vertex f1
  int const gamma_f1_number = 2;
  //                                                  C,  g5    Cg5, 1      Cgt, g5    Cg5gt, 1
  // int const gamma_f1_list[gamma_f1_number][2] = { {10, 15}, { 5,  0},  { 2,   15}, {13,    0} };
  int const gamma_f1_list[gamma_f1_number][2]    = {           { 14,  4} , {11,    4} };

  // list of permutations for recombinations
  int const permutation_list[6][4] =  {
    {0, 2, 3, 1},
    {2, 0, 3, 1},
    {2, 3, 0, 1},
    {0, 1, 2, 3},
    {0, 2, 1, 3},
    {2, 0, 1, 3} };


   int const b_v3_factor_number = 1;
#if defined MESON_BARYON_FACTORIZED_DIAGRAMS_2PT
   char const b_v3_factor_list[b_v3_factor_number][20] = { "xil-gf-sll" };
#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT
   char const b_v3_factor_list[b_v3_factor_number][20] = { "xil-gc-sll" };
#endif

   int const b_v2_factor_number = 1;
   char const b_v2_factor_list[b_v2_factor_number][20] = { "phil-gf-fl-fl" };

   int const b_v4_factor_number = 1;
   char const b_v4_factor_list[b_v4_factor_number][20] = { "phil-gf-fl-fl" };

   int const w_v3_factor_number = 1;
#if defined MESON_BARYON_FACTORIZED_DIAGRAMS_2PT
   char const w_v3_factor_list[w_v3_factor_number][20] = { "g5.phil-gf-fl" };
#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT
   char const w_v3_factor_list[w_v3_factor_number][20] = { "g5.phil-gc-fl" };
#endif

   int const w_v2_factor_number = 2;
   char const w_v2_factor_list[w_v2_factor_number][20] = { "g5.xil-gf-fl-sll", "g5.xil-gf-sll-fl" };

   int const w_v4_factor_number = 2;
   char const w_v4_factor_list[w_v4_factor_number][20] = { "g5.xil-gf-fl-sll", "g5.xil-gf-sll-fl" };

   int const z_v3_factor_number = 1;
#if defined MESON_BARYON_FACTORIZED_DIAGRAMS_2PT
   char const z_v3_factor_list[z_v3_factor_number][20] = { "phil-gf-fl" };
#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT
   char const z_v3_factor_list[z_v3_factor_number][20] = { "phil-gc-fl" };
#endif

   int const z_v2_factor_number = 1;
   char const z_v2_factor_list[z_v2_factor_number][20] = { "phil-gf-fl-fl" };

   int const z_v4_factor_number = 1;
   char const z_v4_factor_list[z_v4_factor_number][20] = { "phil-gf-fl-fl" };

   int const bb_t1_factor_number = 1;
   char const bb_t1_factor_list[bb_t1_factor_number][20] = { "fl-fl" };

   int const bb_t2_factor_number = 1;
   char const bb_t2_factor_list[bb_t2_factor_number][20] = { "fl-fl" };

   int const mm_m1_factor_number = 1;
#if defined MESON_BARYON_FACTORIZED_DIAGRAMS_2PT
   char const mm_m1_factor_list[mm_m1_factor_number][20] = { "fl-fl" };
#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT
   char const mm_m1_factor_list[mm_m1_factor_number][20] = { "fl-gc-fl-gi" };
#endif


#if defined MESON_BARYON_FACTORIZED_DIAGRAMS_2PT
   char diagram_type[] = "mxb-mxb";
   int const npt_mode = 2;
#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT
   char diagram_type[] = "mxb-J-b";
   int const npt_mode = 3;
#endif

   /* int const gamma_basis_conversion_to_cvc[16] =  { 
      4, //  0 = 1
      1, //  1 = x
      2, //  2 = y 
     13, //  3 = xy 
      3, //  4 = z
     14, //  6 = xz
     15, //  7 = yz 
      6, //  8 = xyz
      0, //  9 = t
     10, //  0 = tx
     11, // 10 = ty
      9, // 11 = txy 
     12, // 12 = tz
      8, // 13 = txz
      7, // 14 = tyz
      5  // 15 = txyz 
   };*/

   int const gamma_basis_conversion_to_cvc[16] =  { 
      0, //  0 = t
      1, //  1 = x
      2, //  2 = y 
      3, //  3 = z 
      4, //  4 = id
      5, //  6 = 5
      6, //  7 = t5 
      7, //  8 = x5
      8, //  9 = y5
      9, //  0 = z5
     10, // 10 = tx
     11, // 11 = ty 
     12, // 12 = tz
     13, // 13 = xy
     14, // 14 = xz
     15  // 15 = yz 
   };

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
/*
    case 'n':
      npt_mode = atoi ( optarg );
      fprintf ( stdout, "# [meson_baryon_factorized_diagrams_3pt] n-pt mode set to %d\n", npt_mode );
      break;
*/
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
  fprintf(stdout, "[meson_baryon_factorized_diagrams_3pt] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

 /******************************************************
  * report git version
  ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [meson_baryon_factorized_diagrams_3pt] git version = %s\n", g_gitversion);
  }

  /******************************************************
   * set initial timestamp
   * - con: this is after parsing the input file
   ******************************************************/
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [meson_baryon_factorized_diagrams_3pt] %s# [meson_baryon_factorized_diagrams_3pt] start of run\n", ctime(&g_the_time));
    fflush(stdout);
  }

  /******************************************************
   *
   ******************************************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_geometry\n");
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
    g_source_coords_list[i][0] = ( g_source_coords_list[i][0] +  T_global ) %  T_global;
    g_source_coords_list[i][1] = ( g_source_coords_list[i][1] + LX_global ) % LX_global;
    g_source_coords_list[i][2] = ( g_source_coords_list[i][2] + LY_global ) % LY_global;
    g_source_coords_list[i][3] = ( g_source_coords_list[i][3] + LZ_global ) % LZ_global;
  }

  /**************************************************************************************/
  /**************************************************************************************/

  /**************************************************************************************
   * loop on source locations
   **************************************************************************************/
  for( int i_src = 0; i_src<g_source_location_number; i_src++) {
    int t_base = g_source_coords_list[i_src][0];

    /******************************************************
     * open AFF output files
     ******************************************************/
    if(io_proc == 2) {
      /* AFF output file */
      if ( npt_mode == 2 ) {
        sprintf(filename, "%s_2pt.%.4d.tsrc%.2d.aff", g_outfile_prefix, Nconf, t_base );
      } else if ( npt_mode == 3 ) {
        sprintf(filename, "%s_3pt.%.4d.tsrc%.2d.dt%d.aff", g_outfile_prefix, Nconf, t_base, g_src_snk_time_separation );
      } else {
        fprintf ( stderr, "[meson_baryon_factorized_diagrams_3pt] Error, unrecognized npt_mode value %s %d\n", __FILE__, __LINE__ );
        EXIT(125);
      }

      affw = aff_writer (filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from aff_writer, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [meson_baryon_factorized_diagrams_3pt] writing data to file %s\n", filename);
      }
    }  /* end of if io_proc == 2 */

    /**************************************************************************************/
    /**************************************************************************************/


    /**************************************************************************************
     * B diagrams
     **************************************************************************************/

    /**************************************************************************************/
    /**************************************************************************************/

    /*******************************************
     * v2 factors
     *
     *******************************************/
    double _Complex ****** b_v2_factor = init_6level_ztable ( b_v2_factor_number, gamma_f1_number, g_sink_momentum_number, g_nsample, T_global, 192 );
    if ( b_v2_factor == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(48);
    }

    for ( int isample = 0; isample < g_nsample; isample++ ) {
      if(io_proc == 2) {
        /* AFF input files */
        if ( npt_mode == 2 ) {
          sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_v2_phi_light", Nconf, t_base, isample );
        } else if ( npt_mode == 3 ) {
          sprintf(filename, "%s.%.4d.tbase%.2d.dt%.2d.%.5d.aff", "contract_v2_phi_light", Nconf, t_base, g_src_snk_time_separation, isample );
        }
        affr = aff_reader (filename);
        aff_status_str = (char*)aff_reader_errstr(affr);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from aff_reader for %s, status was %s %s %d\n", filename, aff_status_str, __FILE__ , __LINE__ );
          EXIT(4);
        } else {
          fprintf(stdout, "# [meson_baryon_factorized_diagrams_3pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
        }
        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
          EXIT(103);
        }
      }

      for ( int ifac = 0; ifac < b_v2_factor_number; ifac++ ) {

        for ( int igf = 0; igf < gamma_f1_number; igf++ ) {
 
         for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

            if ( io_proc == 2 ) {

              exitstatus = contract_diagram_read_key_qlua ( b_v2_factor[ifac][igf][ipf][isample], b_v2_factor_list[ifac], 
                  -1, NULL, g_source_coords_list[i_src], isample, "v2", gamma_f1_list[igf][0], g_sink_momentum_list[ipf], affr, T_global, 192);
 
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(104);
              }

            }  // end of if io_proc == 2

          }  // end of loop on sink momentum pf2

        }  // end of loop on sequential source momentum pi2

      }  // end of loop on factors

      if(io_proc == 2) aff_reader_close (affr);


    }  /* end of loop on samples */

    /**************************************************************************************/
    /**************************************************************************************/

    /*******************************************
     * v4 factors
     *
     *******************************************/
    double _Complex ****** b_v4_factor = init_6level_ztable ( b_v4_factor_number, gamma_f1_number, g_sink_momentum_number, g_nsample, T_global, 192 );
    if ( b_v2_factor == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(48);
    }

    for ( int isample = 0; isample < g_nsample; isample++ ) {
      if(io_proc == 2) {
        /* AFF input files */
        if ( npt_mode == 2 ) {
          sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_v4_phi_light", Nconf, t_base, isample );
        } else if ( npt_mode == 3 ) {
          sprintf(filename, "%s.%.4d.tbase%.2d.dt%.2d.%.5d.aff", "contract_v4_phi_light", Nconf, t_base, g_src_snk_time_separation, isample );
        }
        affr = aff_reader (filename);
        aff_status_str = (char*)aff_reader_errstr(affr);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from aff_reader for %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__ );
          EXIT(4);
        } else {
          fprintf(stdout, "# [meson_baryon_factorized_diagrams_3pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
        }
        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
          EXIT(103);
        }
      }

      for ( int ifac = 0; ifac < b_v4_factor_number; ifac++ ) {

        for ( int igf = 0; igf < gamma_f1_number; igf++ ) {

          for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

            if ( io_proc == 2 ) {
 
              exitstatus = contract_diagram_read_key_qlua ( b_v4_factor[ifac][igf][ipf][isample], b_v4_factor_list[ifac],
                  -1, NULL, g_source_coords_list[i_src], isample, "v4", gamma_f1_list[igf][0], g_sink_momentum_list[ipf], affr, T_global, 192);
 
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(104);
              }

            }  // end of if io_proc == 2

          }  // end of loop on sink momentum pf1

        }  // end of loop on Gamma_f1

      }  // end of loop on ifac

      if(io_proc == 2) aff_reader_close (affr);

    }  // end of loop on samples

    /**************************************************************************************/
    /**************************************************************************************/

    /*******************************************
     * v3 factors
     *
     *******************************************/
    double _Complex ******** b_v3_factor = init_8level_ztable ( b_v3_factor_number, gamma_i2_number, g_seq_source_momentum_number, gamma_f2_number, g_sink_momentum_number, g_nsample, T_global, 12 );
    if ( b_v3_factor == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_8level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(48);
    }

    if(io_proc == 2) {
      /* AFF input files */
      sprintf(filename, "%s.%.4d.tbase%.2d.aff", "contract_v3_xi_light", Nconf, t_base );
      affr = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr(affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from aff_reader for %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__ );
        EXIT(4);
      } else {
        fprintf(stdout, "# [meson_baryon_factorized_diagrams_3pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
      }
      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        EXIT(103);
      }
    }

    for ( int ifac = 0; ifac < b_v3_factor_number; ifac++ ) {

      for ( int igi = 0; igi < gamma_i2_number; igi++ ) {

        for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {

          for ( int igf = 0; igf < gamma_f2_number; igf++ ) {

            for ( int ipf = 0; ipf < g_seq2_source_momentum_number; ipf++ ) {
 
              for ( int isample = 0; isample < g_nsample; isample++ ) {

                if ( io_proc == 2 ) {

                  exitstatus = contract_diagram_read_key_qlua ( b_v3_factor[ifac][igi][ipi][igf][ipf][isample], b_v3_factor_list[ifac],
                      gamma_i2_list[igi], g_seq_source_momentum_list[ipi],
                      g_source_coords_list[i_src], isample, "v3", gamma_f2_list[igf], g_seq2_source_momentum_list[ipf], affr, T_global, 12);
 
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(104);
                  }

                }  // end of if io_proc == 2

              }  // end of loop on isample

            }  // end of loop on pf2

          }  // end of loop on gf2

        }  // end of loop on pi2

      }  // end of loop on gi2

    }  // end of loop factors

    if(io_proc == 2) aff_reader_close (affr);

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
        fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      }

      /**************************************************************************************
       * loop on gf1
       **************************************************************************************/
      for ( int igf1 = 0; igf1 < gamma_f1_number; igf1++ ) {

        /**************************************************************************************
         * loop on gf2
         **************************************************************************************/
        for ( int igf2 = 0; igf2 < gamma_f2_number; igf2++ ) {

          /**************************************************************************************
           * loop on pf2
           **************************************************************************************/
          for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

            /**************************************************************************************
             * pf2 is looped over, pf1 is set via total momentum
             **************************************************************************************/
            int ipf1 = sink_momentum_id[ipf2];
            if ( ipf1 == -1 ) continue;
           
            /**************************************************************************************
             * loop on gi2
             **************************************************************************************/
            for ( int igi2 = 0; igi2 < gamma_i2_number; igi2++ ) {

              /**************************************************************************************
               * loop on pi2
               **************************************************************************************/
              for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

                int const pi1[3] = {
                  -( g_total_momentum_list[iptot][0] + g_seq_source_momentum_list[ipi2][0] ),
                  -( g_total_momentum_list[iptot][1] + g_seq_source_momentum_list[ipi2][1] ),
                  -( g_total_momentum_list[iptot][2] + g_seq_source_momentum_list[ipi2][2] ) };

                /**************************************************************************************
                 * loop on gi1
                 **************************************************************************************/
                for ( int igi1 = 0; igi1 < gamma_f1_number; igi1++ ) {

                  char aff_tag_suffix[200];

                  contract_diagram_key_suffix ( aff_tag_suffix, gamma_f2_list[igf2], g_seq2_source_momentum_list[ipf2],
                      gamma_f1_list[igf1][0], gamma_f1_list[igf1][1], g_sink_momentum_list[ipf1], 
                      gamma_i2_list[igi2], g_seq_source_momentum_list[ipi2], 
                      gamma_f1_list[igi1][0], gamma_f1_list[igi1][1], NULL, NULL );


                  /**************************************************************************************
                   * loop on v3 factors
                   **************************************************************************************/
                  for ( int iv3 = 0; iv3 < b_v3_factor_number; iv3++ ) {

                    /**************************************************************************************
                     * loop on v2 factors
                     **************************************************************************************/
                    for ( int iv2 = 0; iv2 < b_v2_factor_number; iv2++ ) {

                      /**************************************************************************************
                       * loop on permutations
                       **************************************************************************************/
                      for ( int ip = 0; ip < 6; ip++ ) 
                      {

                        double _Complex ***diagram = init_3level_ztable ( T_global, 4, 4 );
                        if ( diagram == NULL ) {
                          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
                          EXIT(47);
                        }

                        /**************************************************************************************
                         * diagram B
                         **************************************************************************************/

                        /* reduce to diagram, average over stochastic samples */
                        if ( ( exitstatus = contract_diagram_sample ( diagram, b_v3_factor[iv3][igi2][ipi2][igf2][ipf2], b_v2_factor[iv2][igf1][ipf1],
                                g_nsample, permutation_list[ip], gamma[gamma_f1_list[igi1][0]] , T_global ) ) != 0 ) {
                          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
  
                          /*******************************************
                           * add to diagram (in place)
                           * - baryon boundary phase
                           * - source phase
                           * - outer gamma matrices
                           * - overall phase from conventions
                           *
                           * NOTE: SIGN for GF12 CHOSEN AS +1 for now
                           *******************************************/
                          exitstatus = contract_diagram_finalize ( diagram, diagram_type, gsx, pi1, 
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][1]], +1, gamma_basis_conversion_to_cvc[gamma_f2_list[igf2]],
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][1]], +1, gamma_basis_conversion_to_cvc[gamma_i2_list[igi2]],
                              T_global );

                          /**************************************************************************************/
                          /**************************************************************************************/
  
                          char aff_tag[500];
   
                          /* AFF */
                          sprintf(aff_tag, "/v3/%s/v2/%s/p%d%d%d%d/%s/%s", b_v3_factor_list[iv3], b_v2_factor_list[iv2],  
                              permutation_list[ip][0], permutation_list[ip][1], permutation_list[ip][2], permutation_list[ip][3], "fwd", aff_tag_suffix );
  
                          if ( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                            EXIT(106);
                          } 
   
                          /* AFF */
                          sprintf(aff_tag, "/v3/%s/v2/%s/p%d%d%d%d/%s/%s", b_v3_factor_list[iv3], b_v2_factor_list[iv2],  
                              permutation_list[ip][0], permutation_list[ip][1], permutation_list[ip][2], permutation_list[ip][3], "bwd", aff_tag_suffix );
  
                          if ( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                            EXIT(106);
                          }
  
                        }  // end of loop on coherent source locations

                        fini_3level_ztable ( &diagram );

                      }  // end of loop on permutations
                    }  // end of loop on v2 vactors

                    /**************************************************************************************
                     * loop on v4 factors
                     **************************************************************************************/
                    for ( int iv4 = 0; iv4 < b_v4_factor_number; iv4++ ) {

                      /**************************************************************************************
                       * loop on permutations
                       **************************************************************************************/
                      for ( int ip = 0; ip < 6; ip++ ) {

                        double _Complex ***diagram = init_3level_ztable ( T_global, 4, 4 );
                        if ( diagram == NULL ) {
                          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
                          EXIT(47);
                        }

                        /**************************************************************************************
                         * diagram B
                         **************************************************************************************/

                        /* reduce to diagram, average over stochastic samples */
                        if ( ( exitstatus = contract_diagram_sample ( diagram, b_v3_factor[iv3][igi2][ipi2][igf2][ipf2], b_v4_factor[iv4][igf1][ipf1],
                                g_nsample, permutation_list[ip], gamma[gamma_f1_list[igi1][0]] , T_global ) ) != 0 ) {
                          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
  
                          /*******************************************
                           * finalize diagram
                           *******************************************/
                          exitstatus = contract_diagram_finalize ( diagram, diagram_type, gsx, pi1, 
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][1]], +1, gamma_basis_conversion_to_cvc[gamma_f2_list[igf2]],
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][1]], +1, gamma_basis_conversion_to_cvc[gamma_i2_list[igi2]],
                              T_global );

                          /**************************************************************************************/
                          /**************************************************************************************/
  
                          char aff_tag[500];
   
                          /* AFF */
                          sprintf(aff_tag, "/v3/%s/v4/%s/p%d%d%d%d/%s/%s", b_v3_factor_list[iv3], b_v4_factor_list[iv4],  
                              permutation_list[ip][0], permutation_list[ip][1], permutation_list[ip][2], permutation_list[ip][3], "fwd", aff_tag_suffix );
  
                          if ( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                            EXIT(106);
                          } 
   
                          /* AFF */
                          sprintf(aff_tag, "/v3/%s/v4/%s/p%d%d%d%d/%s/%s", b_v3_factor_list[iv3], b_v4_factor_list[iv4],  
                              permutation_list[ip][0], permutation_list[ip][1], permutation_list[ip][2], permutation_list[ip][3], "bwd", aff_tag_suffix );
  
                          if ( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                            EXIT(106);
                          }


  
                        }  // end of loop on coherent source locations

                        fini_3level_ztable ( &diagram );

                      }  // end of loop on permutations
                    }  // end of loop on v4 vactors
                  }  // end of loop on v3 factors
                }  // end of loop on Gamma_i1
              }  // end of loop on p_i2
            }  // end of loop on Gamma_i2
          }  // end of loop on p_f2
        }  // end of loop on Gamma_f2
      }  // end of loop on Gamma_f1

      free ( sink_momentum_id );
    }  // end of loop on p_tot

    fini_6level_ztable ( &b_v2_factor );
    fini_8level_ztable ( &b_v3_factor );
#if 0
#endif  /* of if 0  */

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * W diagrams
     **************************************************************************************/

    /**************************************************************************************
     * v3 factors
     **************************************************************************************/
    double _Complex ******w_v3_factor = init_6level_ztable ( w_v3_factor_number, gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
    if ( w_v3_factor == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(47);
    }

    for ( int isample = 0; isample < g_nsample; isample++ ) {

      if(io_proc == 2) {
        /* AFF input files */
        if ( npt_mode == 2 ) {
          sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_v3_phi_light", Nconf, t_base , isample );
        } else if ( npt_mode == 3 ) {
          sprintf(filename, "%s.%.4d.tbase%.2d.dt%.2d.%.5d.aff", "contract_v3_phi_light", Nconf, t_base , g_src_snk_time_separation, isample );
        }
        affr = aff_reader (filename);
        aff_status_str = (char*)aff_reader_errstr(affr);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from aff_reader for %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__ );
          EXIT(4);
        } else {
          fprintf(stdout, "# [meson_baryon_factorized_diagrams_3pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
        }
        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
          EXIT(103);
        }
      } // end of if io_proc 


      for ( int ifac = 0; ifac < w_v3_factor_number; ifac++ ) {

        for ( int igf = 0; igf < gamma_f2_number; igf++ ) {

          for ( int ipf = 0; ipf < g_seq2_source_momentum_number; ipf++ ) {

            if ( io_proc == 2 ) {

              exitstatus = contract_diagram_read_key_qlua ( w_v3_factor[ifac][igf][ipf][isample], w_v3_factor_list[ifac],
                  -1, NULL, g_source_coords_list[i_src], isample, "v3", gamma_f2_list[igf], g_seq2_source_momentum_list[ipf], affr, T_global, 12);

              if ( exitstatus != 0 ) {
                fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(104);
              }

            }  // end of if io_proc == 2

          }  // end of loop on seq2 source mometnum pf2 */

        }  // end of loop on Gamma_f2

      }  // end of loop on factors
      
      if(io_proc == 2) aff_reader_close (affr);

    }  // end of loop on samples

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * v2 factor
     **************************************************************************************/
    double _Complex ******** w_v2_factor = init_8level_ztable ( w_v2_factor_number, gamma_i2_number, g_seq_source_momentum_number, gamma_f1_number, g_sink_momentum_number,  g_nsample, T_global, 192 ) ;
    if ( w_v2_factor == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_8level_ztable %s %d\n", __FILE__, __LINE__ );
       EXIT(47);
    }

    if(io_proc == 2) {
      /* AFF input files */
      sprintf(filename, "%s.%.4d.tbase%.2d.aff", "contract_v2_xi_light", Nconf, t_base );
      affr = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr(affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from aff_reader for %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__ );
        EXIT(4);
      } else {
        fprintf(stdout, "# [meson_baryon_factorized_diagrams_3pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
      }
      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        EXIT(103);
      }
    } // end of if io_proc 


    for ( int ifac = 0; ifac < w_v2_factor_number; ifac++ ) {

      for ( int igi = 0; igi < gamma_i2_number; igi++ ) {

        for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {

          for ( int igf = 0; igf < gamma_f1_number; igf++ ) {

            for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

              for ( int isample = 0; isample < g_nsample; isample++ ) {

                if ( io_proc == 2 ) {

                  exitstatus = contract_diagram_read_key_qlua ( w_v2_factor[ifac][igi][ipi][igf][ipf][isample], w_v2_factor_list[ifac], gamma_i2_list[igi], g_seq_source_momentum_list[ipi],
                    g_source_coords_list[i_src], isample, "v2", gamma_f1_list[igf][0], g_sink_momentum_list[ipf], affr, T_global, 192);

                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(104);
                  }

                }  // end of if io_proc == 2

              }  // end of loop on samples

            }  // end of loop on sink momentum pf1

          }  // end of loop on Gamma_f1

        }  // end of loop on seq source momentum pi2

      }  // end of loop on Gamma_i2

    }  // end of loop factors

    if(io_proc == 2) aff_reader_close (affr);

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * v4 factor
     **************************************************************************************/
    double _Complex ******** w_v4_factor = init_8level_ztable ( w_v4_factor_number, gamma_i2_number, g_seq_source_momentum_number, gamma_f1_number, g_sink_momentum_number,  g_nsample, T_global, 192 ) ;
    if ( w_v4_factor == NULL ) {
      fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_8level_ztable %s %d\n", __FILE__, __LINE__ );
       EXIT(47);
    }

    if(io_proc == 2) {
      /* AFF input files */
      sprintf(filename, "%s.%.4d.tbase%.2d.aff", "contract_v4_xi_light", Nconf, t_base );
      affr = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr(affr);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from aff_reader for %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__ );
        EXIT(4);
      } else {
        fprintf(stdout, "# [meson_baryon_factorized_diagrams_3pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
      }
      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        EXIT(103);
      }
    } // end of if io_proc 

    for ( int ifac = 0; ifac < w_v4_factor_number; ifac++ ) {

      for ( int igi = 0; igi < gamma_i2_number; igi++ ) {

        for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {

          for ( int igf = 0; igf < gamma_f1_number; igf++ ) {

            for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

              for ( int isample = 0; isample < g_nsample; isample++ ) {

                if ( io_proc == 2 ) {

                  exitstatus = contract_diagram_read_key_qlua ( w_v4_factor[ifac][igi][ipi][igf][ipf][isample], w_v4_factor_list[ifac], gamma_i2_list[igi], g_seq_source_momentum_list[ipi],
                      g_source_coords_list[i_src], isample, "v4", gamma_f1_list[igf][0], g_sink_momentum_list[ipf], affr, T_global, 192);

                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(104);
                  }

                }  // end of if io_proc == 2

              }  // end of loop on samples

            }  //* end of loop on sink momentum pf1

          }  // end of loop on Gamma_f1

        }  // end of loop on seq source momentum pi2

      }  // end of loop on Gamma_i2

    }  // end of loop on factors

    if(io_proc == 2) aff_reader_close (affr);

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * contractions for W
     **************************************************************************************/

    /**************************************************************************************
     * loop on total momentum
     **************************************************************************************/
    for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

      int * sink_momentum_id =  get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, g_total_momentum_list[iptot], g_sink_momentum_list, g_sink_momentum_number );
      if ( sink_momentum_id == NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      }

      /**************************************************************************************
       * loop on gf1
       **************************************************************************************/
      for ( int igf1 = 0; igf1 < gamma_f1_number; igf1++ ) {

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
             * loop on gf2
             **************************************************************************************/
            for ( int igi2 = 0; igi2 < gamma_i2_number; igi2++ ) {

              /**************************************************************************************
               * loop on pi2
               **************************************************************************************/
              for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

                int const pi1[3] = {
                  -( g_total_momentum_list[iptot][0] + g_seq_source_momentum_list[ipi2][0] ),
                  -( g_total_momentum_list[iptot][1] + g_seq_source_momentum_list[ipi2][1] ),
                  -( g_total_momentum_list[iptot][2] + g_seq_source_momentum_list[ipi2][2] ) };

                /**************************************************************************************
                 * loop on gi1
                 **************************************************************************************/
                for ( int igi1 = 0; igi1 < gamma_f1_number; igi1++ ) {

                  char aff_tag_suffix[200];

                  contract_diagram_key_suffix ( aff_tag_suffix, gamma_f2_list[igf2], g_seq2_source_momentum_list[ipf2],
                      gamma_f1_list[igf1][0], gamma_f1_list[igf1][1], g_sink_momentum_list[ipf1],
                      gamma_i2_list[igi2], g_seq_source_momentum_list[ipi2], 
                      gamma_f1_list[igi1][0], gamma_f1_list[igi1][1], NULL, NULL );

                  /**************************************************************************************
                   * loop on v3 factors
                   **************************************************************************************/
                  for ( int iv3 = 0; iv3 < w_v3_factor_number; iv3++ ) {

                    /**************************************************************************************
                     * loop on v2 factors
                     **************************************************************************************/
                    for ( int iv2 = 0; iv2 < w_v2_factor_number; iv2++ ) {

                      /**************************************************************************************
                       * loop on permutations
                       **************************************************************************************/
                      for ( int ip = 0; ip < 6; ip++ ) {

                        double _Complex ***diagram = init_3level_ztable ( T_global, 4, 4 );
                        if ( diagram == NULL ) {
                          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
                          EXIT(47);
                        }

                        /**************************************************************************************
                         * W diagrams
                         **************************************************************************************/

                        /* reduce to diagram, average over stochastic samples */
                        if ( ( exitstatus = contract_diagram_sample ( diagram, w_v3_factor[iv3][igf2][ipf2], w_v2_factor[iv2][igi2][ipi2][igf1][ipf1], g_nsample, permutation_list[ip],
                                gamma[gamma_f1_list[igi1][0]], T_global ) ) != 0 ) {
                          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
      
                          /*******************************************
                           * finalize diagram
                           *******************************************/
                          exitstatus = contract_diagram_finalize ( diagram, diagram_type, gsx, pi1,
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][1]], +1, gamma_basis_conversion_to_cvc[gamma_f2_list[igf2]],
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][1]], +1, gamma_basis_conversion_to_cvc[gamma_i2_list[igi2]],
                              T_global );

                          /*******************************************/
                          /*******************************************/

                          char aff_tag[500];
      
                          /* AFF key */
                          sprintf(aff_tag, "/v3/%s/v2/%s/p%d%d%d%d/%s/%s", w_v3_factor_list[iv3], w_v2_factor_list[iv2],  
                              permutation_list[ip][0], permutation_list[ip][1], permutation_list[ip][2], permutation_list[ip][3], "fwd", aff_tag_suffix );

                          if ( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                            EXIT(106);
                          }
      
                          /* AFF key */
                          sprintf(aff_tag, "/v3/%s/v2/%s/p%d%d%d%d/%s/%s", w_v3_factor_list[iv3], w_v2_factor_list[iv2],  
                              permutation_list[ip][0], permutation_list[ip][1], permutation_list[ip][2], permutation_list[ip][3], "bwd", aff_tag_suffix );

                          if ( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                            EXIT(106);
                          }
      
                        }  // end of loop on coherent source locations
      
                        fini_3level_ztable ( &diagram );

                      }  // end of loop on permutations
                    }  // end of loop on v2 factors

                    /**************************************************************************************
                     * loop on v4 factors
                     **************************************************************************************/
                    for ( int iv4 = 0; iv4 < w_v4_factor_number; iv4++ ) {

                      /**************************************************************************************
                       * loop on permutations
                       **************************************************************************************/
                      for ( int ip = 0; ip < 6; ip++ ) {

                        double _Complex ***diagram = init_3level_ztable ( T_global, 4, 4 );
                        if ( diagram == NULL ) {
                          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
                          EXIT(47);
                        }

                        /**************************************************************************************
                         * W diagrams
                         **************************************************************************************/

                        /* reduce to diagram, average over stochastic samples */
                        if ( ( exitstatus = contract_diagram_sample ( diagram, w_v3_factor[iv3][igf2][ipf2], w_v4_factor[iv4][igi2][ipi2][igf1][ipf1], g_nsample, permutation_list[ip],
                                gamma[gamma_f1_list[igi1][0]], T_global ) ) != 0 ) {
                          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
      
                          /*******************************************
                           * finalize diagram
                           *******************************************/
                          exitstatus = contract_diagram_finalize ( diagram, diagram_type, gsx, pi1,
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][1]], +1, gamma_basis_conversion_to_cvc[gamma_f2_list[igf2]],
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][1]], +1, gamma_basis_conversion_to_cvc[gamma_i2_list[igi2]],
                              T_global );

                          /*******************************************/
                          /*******************************************/

                          char aff_tag[500];
      
                          /* AFF key */
                          sprintf(aff_tag, "/v3/%s/v4/%s/p%d%d%d%d/%s/%s", w_v3_factor_list[iv3], w_v4_factor_list[iv4],  
                              permutation_list[ip][0], permutation_list[ip][1], permutation_list[ip][2], permutation_list[ip][3], "fwd", aff_tag_suffix );

                          if ( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                            EXIT(106);
                          }
      
                          /* AFF key */
                          sprintf(aff_tag, "/v3/%s/v4/%s/p%d%d%d%d/%s/%s", w_v3_factor_list[iv3], w_v4_factor_list[iv4],  
                              permutation_list[ip][0], permutation_list[ip][1], permutation_list[ip][2], permutation_list[ip][3], "bwd", aff_tag_suffix );

                          if ( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                            EXIT(106);
                          }
      
                        }  // end of loop on coherent source locations
      
                        fini_3level_ztable ( &diagram );

                      }  // end of loop on permutations
                    }  // end of loop on v4 factors

                  }  // end of loop on v3 factors
                }  /* end of loop on Gamma_i1 */
              }  /* end of loop on p_i2 */
            }  /* end of loop on Gamma_i2 */
          }  /* end of loop on p_f2 */
        }  /* end of loop on Gamma_f2 */
      }  /* end of loop on Gamma_f1 */

      free ( sink_momentum_id );

    }  /* end of loop on p_tot */

    fini_8level_ztable ( &w_v2_factor );
    fini_8level_ztable ( &w_v4_factor );
    fini_6level_ztable ( &w_v3_factor );
#if 0
#endif  // of if 0

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
        double _Complex  *******z_v3_factor = init_7level_ztable ( z_v3_factor_number, g_seq_source_momentum_number, gamma_f2_number, g_seq2_source_momentum_number, 4, T_global, 12 );
        if ( z_v3_factor == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_7level_ztable, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(47);
        }

        if(io_proc == 2) {
          /* AFF input files */
          sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_oet_v3_light", Nconf, t_base , isample );
          affr = aff_reader (filename);
          aff_status_str = (char*)aff_reader_errstr(affr);
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from aff_reader for %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__ );
            EXIT(4);
          } else {
            fprintf(stdout, "# [meson_baryon_factorized_diagrams_3pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
          }
          if( (affn = aff_reader_root( affr )) == NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
            EXIT(103);
          }
        } // end of if io_proc == 2


        /**************************************************************************************/
        /**************************************************************************************/

        for ( int ifac = 0; ifac < z_v3_factor_number; ifac++ ) {

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
 
                  exitstatus = contract_diagram_read_oet_key_qlua ( z_v3_factor[ifac][ipi][igf][ipf], z_v3_factor_list[ifac], g_seq_source_momentum_list[ipi], gsx,
                      "v3", gamma_f2_list[igf], g_seq2_source_momentum_list[ipf], affr, T_global, 12 );

                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_read_oet_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(50);
                  }

                }  // end of if io_proc == 0

              }  // end of loop on p_f2

            }  // end of loop on Gamma_f2

          }  // end of loop on pi2

        }  // end of loop on factors

        if(io_proc == 2) aff_reader_close (affr);

        /**************************************************************************************/
        /**************************************************************************************/

        /**************************************************************************************
         * read v2 factor
         **************************************************************************************/
        double _Complex ****** z_v2_factor = init_6level_ztable ( z_v2_factor_number,  gamma_f1_number, g_sink_momentum_number, 4, T_global, 192 );
        if ( z_v2_factor == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_6level_ztable, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(47);
        }
 
        if(io_proc == 2) {
          /* AFF input files */
          sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_oet_v2_light", Nconf, t_base , isample );
          affr = aff_reader (filename);
          aff_status_str = (char*)aff_reader_errstr(affr);
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from aff_reader for %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__ );
            EXIT(4);
          } else {
            fprintf(stdout, "# [meson_baryon_factorized_diagrams_3pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
          }
          if( (affn = aff_reader_root( affr )) == NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
            EXIT(103);
          }
        } // end of if io_proc == 2

        for ( int ifac = 0; ifac < z_v2_factor_number; ifac++ ) {

          /**************************************************************************************
           * loop on gf1
           **************************************************************************************/
          for ( int igf = 0; igf < gamma_f1_number; igf++ ) {

            /**************************************************************************************
             * loop on pf1
             **************************************************************************************/
            for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

              if ( io_proc == 2 ) {

                exitstatus = contract_diagram_read_oet_key_qlua ( z_v2_factor[ifac][igf][ipf], z_v2_factor_list[ifac], NULL, gsx,
                    "v2", gamma_f1_list[igf][0], g_sink_momentum_list[ipf], affr, T_global, 192 );

                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_read_oet_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(51);
                }
              
              }   // end of if io_proc == 2

            }  // end of loop on sink momentum p_f1

          }  // end of loop on Gamma_f1

        }  // end of loop on factors

        if(io_proc == 2) aff_reader_close (affr);

        /**************************************************************************************/
        /**************************************************************************************/
 
        /**************************************************************************************
         * read v4 factor 
         **************************************************************************************/
        double _Complex ****** z_v4_factor = init_6level_ztable ( z_v4_factor_number, gamma_f1_number, g_sink_momentum_number, 4, T_global, 192 );
        if ( z_v4_factor == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_6level_ztable, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(47);
        }

        if(io_proc == 2) {
          /* AFF input files */
          sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_oet_v4_light", Nconf, t_base , isample );
          affr = aff_reader (filename);
          aff_status_str = (char*)aff_reader_errstr(affr);
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from aff_reader for %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__ );
            EXIT(4);
          } else {
            fprintf(stdout, "# [meson_baryon_factorized_diagrams_3pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
          }
          if( (affn = aff_reader_root( affr )) == NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
            EXIT(103);
          }
        } // end of if io_proc == 2

        for ( int ifac = 0; ifac < z_v4_factor_number; ifac++ ) {

          /**************************************************************************************
           * loop on gf1
           **************************************************************************************/
          for ( int igf = 0; igf < gamma_f1_number; igf++ ) {

            /**************************************************************************************
             * loop on pf1
             **************************************************************************************/
            for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

              if ( io_proc == 2 ) {

                exitstatus = contract_diagram_read_oet_key_qlua ( z_v4_factor[ifac][igf][ipf], z_v4_factor_list[ifac], NULL, gsx,
                    "v4", gamma_f1_list[igf][0], g_sink_momentum_list[ipf], affr, T_global, 192 );

                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_read_oet_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(51);
                }
              
              }  // end of if io_proc == 2

            }  // end of loop on sink momentum p_f1

          }  // end of loop on Gamma_f1

        }  // end of loop on factors

        if(io_proc == 2) aff_reader_close (affr);

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
            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
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
              for ( int igf1 = 0; igf1 < gamma_f1_number; igf1++ ) {

                /**************************************************************************************
                 * loop on gi2
                 **************************************************************************************/
                for ( int igi2 = 0; igi2 < gamma_i2_number; igi2++ ) {

                  /**************************************************************************************
                   * loop on pi2
                   **************************************************************************************/
                  for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

                    int const pi1[3] = {
                      -( g_total_momentum_list[iptot][0] + g_seq_source_momentum_list[ipi2][0] ),
                      -( g_total_momentum_list[iptot][1] + g_seq_source_momentum_list[ipi2][1] ),
                      -( g_total_momentum_list[iptot][2] + g_seq_source_momentum_list[ipi2][2] ) };

                    /**************************************************************************************
                     * loop on gi1
                     **************************************************************************************/
                    for ( int igi1 = 0; igi1 < gamma_f1_number; igi1++ ) {

                      char aff_tag_suffix[200];

                      contract_diagram_key_suffix ( aff_tag_suffix, gamma_f2_list[igf2], g_seq2_source_momentum_list[ipf2],
                          gamma_f1_list[igf1][0], gamma_f1_list[igf1][1], g_sink_momentum_list[ipf1], 
                          gamma_i2_list[igi2], g_seq_source_momentum_list[ipi2],
                          gamma_f1_list[igi1][0], gamma_f1_list[igi1][1], NULL, NULL );


                      /**************************************************************************************
                       * loop on v3 factors
                       **************************************************************************************/
                      for ( int iv3 = 0; iv3 < z_v3_factor_number; iv3++ ) {
  
                        /**************************************************************************************
                         * loop on v2 factors
                         **************************************************************************************/
                        for ( int iv2 = 0; iv2 < z_v2_factor_number; iv2++ ) {
  
                          /**************************************************************************************
                           * loop on permutations
                           **************************************************************************************/
                          for ( int ip = 0; ip < 6; ip++ ) {
  
                            double _Complex ***diagram = init_3level_ztable ( T_global, 4, 4 );
                            if ( diagram == NULL ) {
                              fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
                              EXIT(47);
                            }
        
                            /**************************************************************************************
                             * Z diagrams
                             **************************************************************************************/

                            double _Complex *** z_v2_buffer = init_3level_ztable (  4, T_global, 192 );
#if defined MESON_BARYON_FACTORIZED_DIAGRAMS_2PT

                            memcpy ( z_v2_buffer[0][0], z_v2_factor[iv2][igf1][ipf1][0][0], 4*T_global*192*sizeof(double _Complex ) );

#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT

                            for ( int ia = 0; ia < 4; ia++ ) {
                              int tsnk = ( gsx[0] + g_src_snk_time_separation + T_global ) % T_global;

                              for ( int it = 0; it <= g_src_snk_time_separation; it++ ) {
                                int const tcur = ( it + gsx[0] + T_global ) % T_global;
                                memcpy ( z_v2_buffer[ia][tcur] , z_v2_factor[iv2][igf1][ipf1][ia][tsnk], 192 * sizeof( double _Complex ) );
                              }

                              tsnk = ( gsx[0] - g_src_snk_time_separation + T_global ) % T_global;
                              for ( int it = 1; it <= g_src_snk_time_separation; it++ ) {
                                int const tcur = ( -it + gsx[0] + T_global ) % T_global;
                                memcpy ( z_v2_buffer[ia][tcur] , z_v2_factor[iv2][igf1][ipf1][ia][tsnk], 192 * sizeof( double _Complex ) );
                              }
                            }

#endif

                            if ( ( exitstatus = contract_diagram_sample_oet (diagram, z_v3_factor[iv3][ipi2][igf2][ipf2],  z_v2_buffer, gamma[ gamma_i2_list[igi2] ],
                                    permutation_list[ip], gamma[gamma_f1_list[igi1][0]], T_global ) ) != 0 ) {
                              fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_sample_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                              EXIT(105);
                            }

                            fini_3level_ztable ( &z_v2_buffer );

                            /*******************************************/
                            /*******************************************/

                            /*******************************************
                             * finalize diagram
                             *******************************************/
                            exitstatus = contract_diagram_finalize ( diagram, diagram_type, gsx, pi1,
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][1]], +1, gamma_basis_conversion_to_cvc[gamma_f2_list[igf2]],
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][1]], +1, gamma_basis_conversion_to_cvc[gamma_i2_list[igi2]],
                              T_global );

                            /**************************************************************************************/
                            /**************************************************************************************/
        
                            char aff_tag[500];
      
                            /* AFF tag */
                            sprintf(aff_tag, "/v3/%s/v2/%s/p%d%d%d%d/%s/%s", z_v3_factor_list[iv3], z_v2_factor_list[iv2],  
                                permutation_list[ip][0], permutation_list[ip][1], permutation_list[ip][2], permutation_list[ip][3], "fwd", aff_tag_suffix );

                            if ( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                              fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                              EXIT(106);
                            }
        
                            /* AFF tag */
                            sprintf(aff_tag, "/v3/%s/v2/%s/p%d%d%d%d/%s/%s", z_v3_factor_list[iv3], z_v2_factor_list[iv2],  
                                permutation_list[ip][0], permutation_list[ip][1], permutation_list[ip][2], permutation_list[ip][3], "bwd", aff_tag_suffix );

                            if ( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                              fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                              EXIT(106);
                            }
        
                            /**************************************************************************************/
                            /**************************************************************************************/
      
                            fini_3level_ztable ( &diagram );

                          }  // end of loop on permutations
                        }  // end of loop on v2 factors

                        /**************************************************************************************
                         * loop on v4 factors
                         **************************************************************************************/
                        for ( int iv4 = 0; iv4 < z_v4_factor_number; iv4++ ) {
  
                          /**************************************************************************************
                           * loop on permutations
                           **************************************************************************************/
                          for ( int ip = 0; ip < 6; ip++ ) {
  
                            double _Complex ***diagram = init_3level_ztable ( T_global, 4, 4 );
                            if ( diagram == NULL ) {
                              fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
                              EXIT(47);
                            }
        
                            /**************************************************************************************
                             * Z diagrams
                             **************************************************************************************/

                            double _Complex *** z_v4_buffer = init_3level_ztable ( 4, T_global, 192  );
#if defined MESON_BARYON_FACTORIZED_DIAGRAMS_2PT

                            memcpy ( z_v4_buffer[0][0], z_v4_factor[iv4][igf1][ipf1][0][0], 4*T_global*192*sizeof(double _Complex ) );

#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT

                            for ( int ia = 0; ia < 4; ia++ ) {
                              int tsnk = ( gsx[0] + g_src_snk_time_separation + T_global ) % T_global;

                              for ( int it = 0; it <= g_src_snk_time_separation; it++ ) {
                                int const tcur = ( it + gsx[0] + T_global ) % T_global;
                                memcpy ( z_v4_buffer[ia][tcur] , z_v4_factor[iv4][igf1][ipf1][ia][tsnk], 192 * sizeof ( double _Complex ) );
                              }

                              tsnk = ( gsx[0] - g_src_snk_time_separation + T_global ) % T_global;
                              for ( int it = 1; it <= g_src_snk_time_separation; it++ ) {
                                int const tcur = ( -it + gsx[0] + T_global ) % T_global;
                                memcpy ( z_v4_buffer[ia][tcur] , z_v4_factor[iv4][igf1][ipf1][ia][tsnk], 192 * sizeof ( double _Complex ) );
                              }
                            }
#endif

                            if ( ( exitstatus = contract_diagram_sample_oet (diagram, z_v3_factor[iv3][ipi2][igf2][ipf2],  z_v4_buffer, gamma[ gamma_i2_list[igi2] ],
                                    permutation_list[ip], gamma[gamma_f1_list[igi1][0]], T_global ) ) != 0 ) {
                              fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_sample_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                              EXIT(105);
                            }

                            fini_3level_ztable ( &z_v4_buffer );

                            /*******************************************/
                            /*******************************************/

                            /*******************************************
                             * finalize diagram
                             *******************************************/
                            exitstatus = contract_diagram_finalize ( diagram, diagram_type, gsx, pi1,
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][1]], +1, gamma_basis_conversion_to_cvc[gamma_f2_list[igf2]],
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][1]], +1, gamma_basis_conversion_to_cvc[gamma_i2_list[igi2]],
                              T_global );

                            /**************************************************************************************/
                            /**************************************************************************************/
        
                            char aff_tag[500];
      
                            /* AFF tag */
                            sprintf(aff_tag, "/v3/%s/v4/%s/p%d%d%d%d/%s/%s", z_v3_factor_list[iv3], z_v4_factor_list[iv4],  
                                permutation_list[ip][0], permutation_list[ip][1], permutation_list[ip][2], permutation_list[ip][3], "fwd", aff_tag_suffix );

                            if ( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                              fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                              EXIT(106);
                            }
        
                            /* AFF tag */
                            sprintf(aff_tag, "/v3/%s/v4/%s/p%d%d%d%d/%s/%s", z_v3_factor_list[iv3], z_v4_factor_list[iv4],  
                                permutation_list[ip][0], permutation_list[ip][1], permutation_list[ip][2], permutation_list[ip][3], "bwd", aff_tag_suffix );

                            if ( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                              fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                              EXIT(106);
                            }
        
                            /**************************************************************************************/
                            /**************************************************************************************/
      
                            fini_3level_ztable ( &diagram );

                          }  // end of loop on permutations
                        }  // end of loop on v4 factors
                      }  // end of loop on v3 factors
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

        fini_7level_ztable ( &z_v3_factor );
        fini_6level_ztable ( &z_v2_factor );
        fini_6level_ztable ( &z_v4_factor );

      }  // end of loop on oet sampels

    }  /* end of loop on coherent source locations */
#if 0
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

      double _Complex ****** bb_t1_factor = init_6level_ztable ( bb_t1_factor_number, gamma_f1_number, gamma_f1_number, g_sink_momentum_number, T_global, 16 );
      if ( bb_t1_factor == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(52);
      }

      double _Complex ****** bb_t2_factor = init_6level_ztable ( bb_t2_factor_number, gamma_f1_number, gamma_f1_number, g_sink_momentum_number, T_global, 16 );
      if ( bb_t2_factor == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(53);
      }

      if(io_proc == 2) {
        /* AFF input files */
        sprintf(filename, "%s.%.4d.tbase%.2d.aff", "contract_2pt_light", Nconf, t_base );
        affr = aff_reader (filename);
        aff_status_str = (char*)aff_reader_errstr(affr);
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from aff_reader for %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__ );
          EXIT(4);
        } else {
          fprintf(stdout, "# [meson_baryon_factorized_diagrams_3pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
        }
        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
          EXIT(103);
        }
      } // end of if io_proc == 2


      /**************************************************************************************
       * loop on gf1
       **************************************************************************************/
      for ( int igf = 0; igf < gamma_f1_number; igf++ ) {

        /**************************************************************************************
         * loop on pf1
         **************************************************************************************/
        for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

          /**************************************************************************************
           * loop on gi1
           **************************************************************************************/
          for ( int igi = 0; igi < gamma_f1_number; igi++ ) {

            for ( int ifac = 0; ifac < bb_t1_factor_number; ifac++ ) {

              if ( io_proc == 2 ) {

                exitstatus = contract_diagram_read_key_qlua ( bb_t1_factor[ifac][igi][igf][ipf], bb_t1_factor_list[ifac], gamma_f1_list[igi][0], NULL, gsx, -1, "t1", gamma_f1_list[igf][0], g_sink_momentum_list[ipf], affr, T_global, 16);
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(114);
                }

              }  // end of if io_proc == 2

            }  // end of loop on factors

            for ( int ifac = 0; ifac < bb_t2_factor_number; ifac++ ) {

              if ( io_proc == 2 ) {

                exitstatus = contract_diagram_read_key_qlua ( bb_t2_factor[ifac][igi][igf][ipf], bb_t2_factor_list[ifac], gamma_f1_list[igi][0], NULL, gsx, -1, "t2", gamma_f1_list[igf][0], g_sink_momentum_list[ipf], affr, T_global, 16);
                if ( exitstatus != 0 ) {
                  fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(115);
                }

              }

            }  // end of loop on factors

          }  /* end of loop on Gamma_i1 */

        }  /* end of loop on sink momentum */

      }  /* end of loop on Gamma_f1 */

      if(io_proc == 2) aff_reader_close (affr);

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * loop on stochastic oet samples
       **************************************************************************************/
      for ( int isample = 0; isample < g_nsample_oet; isample++ ) {

        /**************************************************************************************
         * read mm factor
         **************************************************************************************/
        double _Complex ******* mm_m1_factor =  init_7level_ztable ( mm_m1_factor_number, gamma_i2_number, g_seq_source_momentum_number, gamma_f2_number, g_seq2_source_momentum_number, T_global , 1 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_7level_ztable %s %d\n", __FILE__, __LINE__);
          EXIT(147);
        }
  
        if(io_proc == 2) {
          /* AFF input files */
          sprintf(filename, "%s.%.4d.tbase%.2d.%.5d.aff", "contract_oet_m_m_2pt_light", Nconf, t_base, isample );
          affr = aff_reader (filename);
          aff_status_str = (char*)aff_reader_errstr(affr);
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from aff_reader for %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__ );
            EXIT(4);
          } else {
            fprintf(stdout, "# [meson_baryon_factorized_diagrams_3pt] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
          }
          if( (affn = aff_reader_root( affr )) == NULL ) {
            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
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

                for ( int ifac = 0; ifac < mm_m1_factor_number; ifac++ ) {

                  if ( io_proc == 2 ) {

                    exitstatus = contract_diagram_read_key_qlua ( mm_m1_factor[ifac][igi][ipi][igf][ipf], mm_m1_factor_list[ifac], gamma_i2_list[igi], g_seq_source_momentum_list[ipi], gsx, 
                        -1, "m1", gamma_f2_list[igf], g_seq2_source_momentum_list[ipf], affr, T_global, 1);
                    if ( exitstatus != 0 ) {
                      fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_read_key_qlua, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(114);
                    }

                  }  // end of loop on factors

                }  // end of if io_proc == 2

              }  // end of loop on pf2

            }  /* end of loop on Gamma_f2 */

          }  /* end of loop in pi2 */

        }  /* end of loop on Gamma_i2 */

        if(io_proc == 2) aff_reader_close (affr);

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
            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from get_minus_momentum_id %s %d\n", __FILE__, __LINE__ );
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
              for ( int igf1 = 0; igf1 < gamma_f1_number; igf1++ ) {

                /**************************************************************************************
                 * loop on gi2
                 **************************************************************************************/
                for ( int igi2 = 0; igi2 < gamma_i2_number; igi2++ ) {

                  /**************************************************************************************
                   * loop on pi2
                   **************************************************************************************/
                  for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

                    int const pi1[3] = {
                      -( g_total_momentum_list[iptot][0] + g_seq_source_momentum_list[ipi2][0] ),
                      -( g_total_momentum_list[iptot][1] + g_seq_source_momentum_list[ipi2][1] ),
                      -( g_total_momentum_list[iptot][2] + g_seq_source_momentum_list[ipi2][2] ) };

                    /**************************************************************************************
                     * loop on gi1
                     **************************************************************************************/
                    for ( int igi1 = 0; igi1 < gamma_f1_number; igi1++ ) { 

                      char aff_tag_suffix[200];

                      contract_diagram_key_suffix ( aff_tag_suffix, gamma_f2_list[igf2], g_seq2_source_momentum_list[ipf2],
                          gamma_f1_list[igf1][0], gamma_f1_list[igf1][1], g_sink_momentum_list[ipf1],
                          gamma_i2_list[igi2], g_seq_source_momentum_list[ipi2],
                          gamma_f1_list[igi1][0], gamma_f1_list[igi1][1], NULL, NULL );


                      /**************************************************************************************
                       * loop on mm m1 factors
                       **************************************************************************************/
                      for ( int im1 = 0; im1 < mm_m1_factor_number; im1++ ) {

                        /**************************************************************************************
                         * loop on bb t1 factors
                         **************************************************************************************/
                        for ( int it1 = 0; it1 < bb_t1_factor_number; it1++ ) {


                          double _Complex ***diagram = init_3level_ztable ( T_global, 4, 4 );
                          if ( diagram == NULL ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
                            EXIT(142);
                          }

                          double _Complex ***bb_aux = init_3level_ztable ( T_global, 4, 4 );
                          if ( bb_aux == NULL ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
                            EXIT(147);
                          }

                          char aff_tag[500];

                          /**************************************************************************************
                           * S diagrams
                           **************************************************************************************/
                          memcpy( bb_aux[0][0], bb_t1_factor[it1][igi1][igf1][ipf1][0], 16*T_global*sizeof(double _Complex) );

                          // multiply baryon 2-point function with meson 2-point function
#if defined MESON_BARYON_FACTORIZED_DIAGRAMS_2PT

                          exitstatus = contract_diagram_zm4x4_field_ti_co_field ( diagram, bb_aux, mm_m1_factor[im1][igi2][ipi2][igf2][ipf2][0], T_global );

#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT

                          /* fwd direction */
                          int isnk = ( gsx[0] + g_src_snk_time_separation + T_global ) % T_global;
                          for ( int it = 0; it <= g_src_snk_time_separation; it++ ) {
                            int const icur = ( it + gsx[0] + T_global ) % T_global;
                            zm4x4_eq_zm4x4_ti_co ( diagram[icur], bb_aux[isnk],  mm_m1_factor[im1][igi2][ipi2][igf2][ipf2][icur][0] );

                            // zm4x4_eq_zm4x4_ti_co ( diagram[icur], bb_aux[isnk],  1. );
                            //zm4x4_eq_unity ( diagram[icur] );
                            //zm4x4_ti_eq_co ( diagram[icur], mm_m1_factor[im1][igi2][ipi2][igf2][ipf2][icur][0] );
                          }

                          /* bwd direction 
                           *
                           * keep the fwd value at source time, so start with it = 1
                           */
                          isnk = ( gsx[0] - g_src_snk_time_separation + T_global ) % T_global;
                          for ( int it = 1; it <= g_src_snk_time_separation; it++ ) {
                            int const icur = ( -it + gsx[0] + T_global ) % T_global;
                            zm4x4_eq_zm4x4_ti_co ( diagram[icur], bb_aux[isnk],  mm_m1_factor[im1][igi2][ipi2][igf2][ipf2][icur][0] );

                            // zm4x4_eq_zm4x4_ti_co ( diagram[icur], bb_aux[isnk],  1. );
                            //zm4x4_eq_unity ( diagram[icur] );
                            //zm4x4_ti_eq_co ( diagram[icur], mm_m1_factor[im1][igi2][ipi2][igf2][ipf2][icur][0] );
                          }
#endif
                          /* multiply factor -1 from pi-pi Wick contraction closed fermion loop */
                          exitstatus = contract_diagram_zm4x4_field_ti_eq_re ( diagram, -1., T_global );

#if 0
                          NOT needed for qlua code
                          // transpose
                          exitstatus = contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( diagram, diagram, T_global );
#endif  /* of if 0 */

                          /*******************************************/
                          /*******************************************/

                          /*******************************************
                           * finalize diagram
                           *******************************************/
                          exitstatus = contract_diagram_finalize ( diagram, diagram_type, gsx, pi1,
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][1]], +1, gamma_basis_conversion_to_cvc[gamma_f2_list[igf2]],
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][1]], +1, gamma_basis_conversion_to_cvc[gamma_i2_list[igi2]],
                              T_global );

                          /* AFF */
                          sprintf(aff_tag, "/m1/%s/t1/%s/p0000/fwd/%s", mm_m1_factor_list[im1], bb_t1_factor_list[it1], aff_tag_suffix );
                          if( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d\n", exitstatus);
                            EXIT(154);
                          }
   
                          /* AFF */
                          sprintf(aff_tag, "/m1/%s/t1/%s/p0000/bwd/%s", mm_m1_factor_list[im1], bb_t1_factor_list[it1], aff_tag_suffix );
                          if( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d\n", exitstatus);
                            EXIT(155);
                          }


                          fini_3level_ztable ( &diagram );
                          fini_3level_ztable ( &bb_aux );

                        }  // end of loop on t1 factors

                        /**************************************************************************************
                         * loop on bb t2 factors
                         **************************************************************************************/
                        for ( int it2 = 0; it2 < bb_t2_factor_number; it2++ ) {


                          double _Complex ***diagram = init_3level_ztable ( T_global, 4, 4 );
                          if ( diagram == NULL ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
                            EXIT(142);
                          }

                          double _Complex ***bb_aux = init_3level_ztable ( T_global, 4, 4 );
                          if ( bb_aux == NULL ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
                            EXIT(147);
                          }

                          char aff_tag[500];

                          /**************************************************************************************
                           * S diagrams
                           **************************************************************************************/
                          memcpy( bb_aux[0][0], bb_t2_factor[it2][igi1][igf1][ipf1][0], 16*T_global*sizeof(double _Complex) );

                          // multiply baryon 2-point function with meson 2-point function
#if defined MESON_BARYON_FACTORIZED_DIAGRAMS_2PT

                          exitstatus = contract_diagram_zm4x4_field_ti_co_field ( diagram, bb_aux, mm_m1_factor[im1][igi2][ipi2][igf2][ipf2][0], T_global );

#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT

                          /* fwd direction */
                          int isnk = ( gsx[0] + g_src_snk_time_separation + T_global ) % T_global;
                          for ( int it = 0; it <= g_src_snk_time_separation; it++ ) {
                            int const icur = ( it + gsx[0] + T_global ) % T_global;
                            zm4x4_eq_zm4x4_ti_co ( diagram[icur], bb_aux[isnk],  mm_m1_factor[im1][igi2][ipi2][igf2][ipf2][icur][0] );

                            // zm4x4_eq_zm4x4_ti_co ( diagram[icur], bb_aux[isnk],  1. );
                            //zm4x4_eq_unity ( diagram[icur] );
                            //zm4x4_ti_eq_co ( diagram[icur], mm_m1_factor[im1][igi2][ipi2][igf2][ipf2][icur][0] );
                          }

                          /* bwd direction 
                           *
                           * keep the fwd value at source time, so start with it = 1
                           */
                          isnk = ( gsx[0] - g_src_snk_time_separation + T_global ) % T_global;
                          for ( int it = 1; it <= g_src_snk_time_separation; it++ ) {
                            int const icur = ( -it + gsx[0] + T_global ) % T_global;
                            zm4x4_eq_zm4x4_ti_co ( diagram[icur], bb_aux[isnk],  mm_m1_factor[im1][igi2][ipi2][igf2][ipf2][icur][0] );

                            // zm4x4_eq_zm4x4_ti_co ( diagram[icur], bb_aux[isnk], 1. );
                            //zm4x4_eq_unity ( diagram[icur] );
                            //zm4x4_ti_eq_co ( diagram[icur], mm_m1_factor[im1][igi2][ipi2][igf2][ipf2][icur][0] );
                          }
#endif


                          /* multiply factor -1 from pi-pi Wick contraction closed fermion loop */
                          exitstatus = contract_diagram_zm4x4_field_ti_eq_re ( diagram, -1., T_global );
#if 0
                          NOT needed for qlua code
                          // transpose
                          exitstatus = contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( diagram, diagram, T_global );
#endif  /* of if 0 */

                          /*******************************************/
                          /*******************************************/

                          /*******************************************
                           * finalize diagram
                           *******************************************/
                          exitstatus = contract_diagram_finalize ( diagram, diagram_type, gsx, pi1,
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igf1][1]], +1, gamma_basis_conversion_to_cvc[gamma_f2_list[igf2]],
                              gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][0]], gamma_basis_conversion_to_cvc[gamma_f1_list[igi1][1]], +1, gamma_basis_conversion_to_cvc[gamma_i2_list[igi2]],
                              T_global );

                          /* AFF */
                          sprintf(aff_tag, "/m1/%s/t2/%s/p0000/fwd/%s", mm_m1_factor_list[im1], bb_t2_factor_list[it2], aff_tag_suffix );
                          if( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d\n", exitstatus);
                            EXIT(154);
                          }
   
                          /* AFF */
                          sprintf(aff_tag, "/m1/%s/t2/%s/p0000/bwd/%s", mm_m1_factor_list[im1], bb_t2_factor_list[it2], aff_tag_suffix );
                          if( ( exitstatus = contract_diagram_write_aff_sst ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                            fprintf(stderr, "[meson_baryon_factorized_diagrams_3pt] Error from contract_diagram_write_aff_sst, status was %d\n", exitstatus);
                            EXIT(155);
                          }

                          fini_3level_ztable ( &diagram );
                          fini_3level_ztable ( &bb_aux );

                        }  // end of loop on t2 factors

                      }  // end of loop on m1 factors
                    }  // end of loop on Gamma_i1
                  }  // end of loop on pi2
                }  // end of loop on Gamma_i2
              }  // end of loop on Gamma_f1
            }  // end of loop on pf2
          }  // end of loop on Gamma_f2

          free ( sink_momentum_id );

        }  /* end of loop on p_tot */

        /**************************************************************************************/
        /**************************************************************************************/

        fini_6level_ztable ( &bb_t1_factor );
        fini_6level_ztable ( &bb_t2_factor );
        fini_7level_ztable ( &mm_m1_factor );

      }  // end of loop on stochastic oet samples

    }  // end of loop coherent source timeslices
#if 0
#endif  // of if 0

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
