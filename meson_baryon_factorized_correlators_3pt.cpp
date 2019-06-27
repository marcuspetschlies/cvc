/****************************************************
 * meson_baryon_factorized_correlators_3pt
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
#include "contractions_io.h"
#include "gamma.h"
#include "zm4x4.h"

using namespace cvc;

#define MAX_DIAGRAM_NUM 16

typedef struct {
  char flavor_name[20];

  int diagram_num;

  char OPi1_list[MAX_DIAGRAM_NUM];
  char OPf1_list[MAX_DIAGRAM_NUM];
  char OPD_list[MAX_DIAGRAM_NUM];

  char RN1_list[MAX_DIAGRAM_NUM][3];
  char RN2_list[MAX_DIAGRAM_NUM][3];

  char PXGP1_list[MAX_DIAGRAM_NUM][20];
  char PXGP2_list[MAX_DIAGRAM_NUM][20];

  int permutation_list[MAX_DIAGRAM_NUM][4];

} fccl_struct;


/* #define MESON_BARYON_FACTORIZED_DIAGRAMS_2PT */

#define MESON_BARYON_FACTORIZED_DIAGRAMS_3PT


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
 
int const gamma_basis_conversion_qlua_bin_to_cvc[16] =  { 
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
};

char const gamma_qlua_baryon_tag[16][12] = {
  "Cgygt",
  "Cgzg5",
  "Cgt",  
  "Cgxgt",
  "Cgxg5",
  "Cg5",
  "Cgzgt",
  "Cgyg5",
  "Cgy",
  "Cgzgtg5",
  "C",
  "Cgx",
  "Cgxgtg5",
  "Cg5gt",
  "Cgz",
  "Cgygtg5" };

int const gamma_qlua_transpose_sign[16] = { 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1 };


/***********************************************************
 * main program
 ***********************************************************/
int main(int argc, char **argv) {
 
  char const infile_prefix[] = "meson_baryon_factorized_diagrams";
  char const outfile_prefix[] = "meson_baryon_factorized_correlators";

  char const fbwd_str[2][4] = { "fwd", "bwd" };

  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[200];

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  struct AffWriter_s *affw = NULL;
  char * aff_status_str;
#endif

  fccl_struct * fccl = NULL;

  // pion-type gamma list at vertex i2
  int const gamma_i2_number = 1;
  //                                              g5   g5 gt ~ gx gy gz
  int const gamma_i2_list[gamma_i2_number]    = { 15 };

#if defined MESON_BARYON_FACTORIZED_DIAGRAMS_2PT
  // pion-type gamma list at vertex f2
  int const gamma_f2_number = 1;
  //                                           g5   1  gt gtg5 
  int const gamma_f2_list[gamma_f2_number] = { 15 };
  //
#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT
  // current vertex
  int const gamma_f2_number = 6;
  int const gamma_f2_list[gamma_f2_number] = { 1, 2, 4, 14, 13, 11 } ;
#else
#error "need MESON_BARYON_FACTORIZED_DIAGRAMS_<2/3>PT defined" 
#endif

  // Nucleon-type gamma list at vertex f1
  int const gamma_f1_number = 1;
  //                                                  C,  g5    Cg5, 1      Cgt, g5    Cg5gt, 1
  // int const gamma_f1_list[gamma_f1_number][2] = { {10, 15}, { 5,  0},  { 2,   15}, {13,    0} };
  int const gamma_f1_list[gamma_f1_number][2]    = {           { 5,  0} };


#if defined MESON_BARYON_FACTORIZED_DIAGRAMS_2PT
   /* char diagram_type[] = "mxb-mxb"; */
   int const npt_mode = 2;
#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT
   /* char diagram_type[] = "mxb-J-b"; */
   int const npt_mode = 3;
#endif


#if defined MESON_BARYON_FACTORIZED_DIAGRAMS_2PT
   int const fcc_num = 1;
#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT
   int const fcc_num = 6;
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
  fprintf(stdout, "[meson_baryon_factorized_correlators_3pt] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

 /******************************************************
  * report git version
  ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [meson_baryon_factorized_correlators_3pt] git version = %s\n", g_gitversion);
  }

  /******************************************************
   * set initial timestamp
   * - con: this is after parsing the input file
   ******************************************************/
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [meson_baryon_factorized_correlators_3pt] %s# [meson_baryon_factorized_correlators_3pt] start of run\n", ctime(&g_the_time));
    fflush(stdout);
  }

  /******************************************************
   *
   ******************************************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[meson_baryon_factorized_correlators_3pt] Error from init_geometry\n");
    EXIT(1);
  }
  geometry();

  /***********************************************
   * set io process
   ***********************************************/
  int const io_proc = get_io_proc();

  /******************************************************
   * the fcc list
   ******************************************************/
  fccl = (fccl_struct * ) malloc ( fcc_num * sizeof ( fccl_struct ) );
  if( fccl == NULL ) {
    fprintf( stderr, "[meson_baryon_factorized_correlators_3pt] Error from malloc %s %d\n", __FILE__, __LINE__ );
    EXIT(23);
  }

  /******************************************************
   * initialize
   ******************************************************/
  for ( int i = 0; i < fcc_num; i++ ) {
    strcpy ( fccl[i].flavor_name, "NA" );
    fccl[i].diagram_num = 0;
  }

#if defined MESON_BARYON_FACTORIZED_DIAGRAMS_2PT
  sprintf( filename, "%s_2pt.input", outfile_prefix );
#elif defined MESON_BARYON_FACTORIZED_DIAGRAMS_3PT
  sprintf( filename, "%s_3pt.input", outfile_prefix );
#endif
  
  FILE * ifs = fopen ( filename, "r" );
  if ( ifs == NULL ) {
    fprintf ( stderr, "[meson_baryon_factorized_correlators_3pt] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
    EXIT(24);
  }


  for ( int i = 0; i < fcc_num; i++ ) {

    fscanf ( ifs, "%s\n", fccl[i].flavor_name );
    fscanf ( ifs, "%d\n", &(fccl[i].diagram_num) );
    if ( fccl[i].diagram_num > MAX_DIAGRAM_NUM ) {
      fprintf ( stderr, "[meson_baryon_factorized_correlators_3pt] Error , diagram num %d too large\n", fccl[i].diagram_num );
      EXIT(23);
    } else {
      fprintf ( stdout, "# [meson_baryon_factorized_correlators_3pt] reading %s with %d diagrams\n", fccl[i].flavor_name, fccl[i].diagram_num ); 
    }
    for ( int idiag = 0; idiag < fccl[i].diagram_num; idiag++ ) {
      fscanf ( ifs, "%s %s %s %s %c %c %c %1d%1d%1d%1d\n", 
          fccl[i].RN2_list[idiag],
          fccl[i].PXGP2_list[idiag],
          fccl[i].RN1_list[idiag],
          fccl[i].PXGP1_list[idiag],
          fccl[i].OPi1_list+idiag,
          fccl[i].OPf1_list+idiag,
          fccl[i].OPD_list+idiag,
          fccl[i].permutation_list[idiag]+0,
          fccl[i].permutation_list[idiag]+1,
          fccl[i].permutation_list[idiag]+2,
          fccl[i].permutation_list[idiag]+3  );

      if ( g_verbose > 2 ) {
        fprintf ( stdout, "  %3s %20s %3s %20s %c %c %c %2d%2d%2d%2d\n", 
            fccl[i].RN2_list[idiag],
            fccl[i].PXGP2_list[idiag],
            fccl[i].RN1_list[idiag],
            fccl[i].PXGP1_list[idiag],
            fccl[i].OPi1_list[idiag],
            fccl[i].OPf1_list[idiag],
            fccl[i].OPD_list[idiag],
            fccl[i].permutation_list[idiag][0],
            fccl[i].permutation_list[idiag][1],
            fccl[i].permutation_list[idiag][2],
            fccl[i].permutation_list[idiag][3]  );

      }
    }
  }

  fclose ( ifs );


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

    int const t_base = g_source_coords_list[i_src][0];

    /******************************************************
     * open AFF output files
     ******************************************************/
    if(io_proc == 2) {
      /* AFF output file */
      if ( npt_mode == 2 ) {
        sprintf(filename, "%s_2pt.%.4d.tsrc%.2d.aff", outfile_prefix, Nconf, t_base );
      } else if ( npt_mode == 3 ) {
        sprintf(filename, "%s_3pt.%.4d.tsrc%.2d.aff", outfile_prefix, Nconf, t_base );
      } else {
        fprintf ( stderr, "[meson_baryon_factorized_correlators_3pt] Error, unrecognized npt_mode value %s %d\n", __FILE__, __LINE__ );
        EXIT(125);
      }

      affw = aff_writer (filename);
      aff_status_str = (char*)aff_writer_errstr(affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_correlators_3pt] Error from aff_writer, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [meson_baryon_factorized_correlators_3pt] writing data to file %s\n", filename);
      }

      /* aff input file */

      if ( npt_mode == 2 ) {
        sprintf(filename, "%s_2pt.%.4d.tsrc%.2d.aff", infile_prefix, Nconf, t_base );
      } else if ( npt_mode == 3 ) {
        sprintf(filename, "%s_3pt.%.4d.tsrc%.2d.aff", infile_prefix, Nconf, t_base );
      } else {
        fprintf ( stderr, "[meson_baryon_factorized_correlators_3pt] Error, unrecognized npt_mode value %s %d\n", __FILE__, __LINE__ );
        EXIT(125);
      }

      affr = aff_reader (filename);
      aff_status_str = (char*)aff_reader_errstr( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_correlators_3pt] Error from aff_reader, status was %s\n", aff_status_str);
        EXIT(4);
      } else {
        fprintf(stdout, "# [meson_baryon_factorized_correlators_3pt] reading data from file %s\n", filename);
      }
    }  /* end of if io_proc == 2 */

    /**************************************************************************************/
    /**************************************************************************************/

    for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {

      int const t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

      int source_proc_id, sx[4], gsx[4] = { t_coherent,
          ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global,
          ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global,
          ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global };

      get_point_source_info (gsx, sx, &source_proc_id);

      int const nT = g_src_snk_time_separation + 1;


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
  
#if 0
                  int const pi1[3] = {
                    -( g_total_momentum_list[iptot][0] + g_seq_source_momentum_list[ipi2][0] ),
                    -( g_total_momentum_list[iptot][1] + g_seq_source_momentum_list[ipi2][1] ),
                    -( g_total_momentum_list[iptot][2] + g_seq_source_momentum_list[ipi2][2] ) };
#endif  /* of if 0 */

                  /**************************************************************************************
                   * loop on gi1
                   **************************************************************************************/
                  for ( int igi1 = 0; igi1 < gamma_f1_number; igi1++ ) {

                    /**************************************************************************************
                     * loop on fwd, bwd
                     **************************************************************************************/
                    for ( int ifbwd = 0; ifbwd < 2; ifbwd++ ) {

                      /**************************************************************************************
                       * loop on flavor component correlator list
                       **************************************************************************************/
                      for ( int ifcc = 0; ifcc < fcc_num; ifcc++ ) {

                        fccl_struct * const fcc_ptr = fccl + ifcc;

                        /**************************************************************************************
                         * read the diagrams
                         **************************************************************************************/

                        double _Complex **** diagram = init_4level_ztable ( fcc_ptr->diagram_num, nT, 4, 4 );
                        if ( diagram == NULL ) {
                          fprintf ( stderr, "[meson_baryon_factorized_correlators_3pt] Error from init_4level_ztable %s %d\n", __FILE__, __LINE__ );
                          EXIT(2);
                        }

                        double _Complex *** buffer = init_3level_ztable ( 4, 4, nT );
                        if ( buffer == NULL ) {
                          fprintf ( stderr, "[meson_baryon_factorized_correlators_3pt] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
                          EXIT(2);
                        }

                        for ( int idiag = 0; idiag < fcc_ptr->diagram_num; idiag++ )
                        {

                          char diagram_key[1000];

                          sprintf ( diagram_key, "/%s/%s/%s/%s/p%d%d%d%d/%s/gf2%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/gf1%.2d_%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/gi2%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/gi1%.2d_%.2d",
                              fcc_ptr->RN1_list[idiag], fcc_ptr->PXGP1_list[idiag], fcc_ptr->RN2_list[idiag], fcc_ptr->PXGP2_list[idiag],
                              fcc_ptr->permutation_list[idiag][0]-1, fcc_ptr->permutation_list[idiag][1]-1, fcc_ptr->permutation_list[idiag][2]-1, fcc_ptr->permutation_list[idiag][3]-1,
                              fbwd_str[ifbwd], gamma_f2_list[igf2], 
                              g_seq2_source_momentum_list[ipf2][0], g_seq2_source_momentum_list[ipf2][1], g_seq2_source_momentum_list[ipf2][2],
                              gamma_f1_list[igf1][0], gamma_f1_list[igf1][1],
                              g_sink_momentum_list[ipf1][0], g_sink_momentum_list[ipf1][1], g_sink_momentum_list[ipf1][2],
                              gamma_i2_list[igi2],
                              g_seq_source_momentum_list[ipi2][0],
                              g_seq_source_momentum_list[ipi2][1],
                              g_seq_source_momentum_list[ipi2][2],
                              gamma_f1_list[igi1][0], gamma_f1_list[igi1][1] );

                          fprintf ( stdout, "# [meson_baryon_factorized_correlators_3pt] diagram_key = %s %s %d\n", diagram_key, __FILE__, __LINE__ );

                          exitstatus = read_aff_contraction ( buffer[0][0], affr, NULL, diagram_key, nT * 16 );
                          if ( exitstatus != 0 ) {
                            fprintf(stderr, "[meson_baryon_factorized_correlators_3pt] Error fro read_aff_contraction, status was %d %s %d\n", exitstatus,  __FILE__, __LINE__);
                            EXIT(3);
                          }
                          for ( int ia = 0; ia < 4; ia++ ) {
                          for ( int ib = 0; ib < 4; ib++ ) {
                            for ( int it = 0; it < nT; it++ ) {
                              diagram[idiag][it][ia][ib] = buffer[ia][ib][it];
                            }
                          }}

                          /**************************************************************************************
                           * determine the sign from tables
                           **************************************************************************************/

                          int const sign_OPi1 = fcc_ptr->OPi1_list[idiag] == 'T' ? gamma_qlua_transpose_sign[ gamma_f1_list[igi1][0] ] : 1.;

                          int const sign_OPf1 = fcc_ptr->OPf1_list[idiag] == 'T' ? gamma_qlua_transpose_sign[ gamma_f1_list[igf1][0] ] : 1.;

                          exitstatus = contract_diagram_zm4x4_field_ti_eq_re ( diagram[idiag], sign_OPi1 * sign_OPf1, nT );

                          /**************************************************************************************
                           * transpose the diagram if so listed
                           **************************************************************************************/
                          if ( fcc_ptr->OPD_list[idiag] == 'T' ) {
                            exitstatus = contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( diagram[idiag], diagram[idiag], nT );
                          }

                          for ( int ia = 0; ia < 4; ia++ ) {
                          for ( int ib = 0; ib < 4; ib++ ) {
                            for ( int it = 0; it < nT; it++ ) {
                              buffer[ia][ib][it] = diagram[idiag][it][ia][ib];
                            }
                          }}

                          char correlator_key[1000];
                          sprintf ( correlator_key, "/%s%s", fcc_ptr->flavor_name, diagram_key );

                          exitstatus = write_aff_contraction ( buffer[0][0], affw, NULL, correlator_key, 16*nT );

                        }  /* end of loop on diagrams */

                        fini_4level_ztable ( &diagram );
                        fini_3level_ztable ( &buffer );

                      }  /* end of loop on fccl */

                    }  /* end of loop on fwd / bwd */
                  }  /* end of loop on gi1 */
                }  /* end of loop on pi2 */
              }  /* end of loop on gi2 */
            }  /* end of loop on pf2 */
          }  /* end of loop on gf2 */
        }  /* end of loop on gf1 */
      }  /* end of loop on ptot */

      /**************************************************************************************/
      /**************************************************************************************/

    }  /* end of loop on coherent sources */

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * close AFF readers for input and output files
     **************************************************************************************/
    if(io_proc == 2) {
      aff_reader_close (affr);

      aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[meson_baryon_factorized_correlators_3pt] Error from aff_writer_close, status was %s\n", aff_status_str);
        EXIT(171);
      }
    }  /* end of if io_proc == 2 */

  }  /* end of loop on base sources */
#if 0
#endif  /* of if 0 */

  /**************************************************************************************
   * free the allocated memory, finalize
   **************************************************************************************/
  free ( fccl );
  free_geometry();

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [meson_baryon_factorized_correlators_3pt] %s# [meson_baryon_factorized_correlators_3pt] end of run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [meson_baryon_factorized_correlators_3pt] %s# [meson_baryon_factorized_correlators_3pt] end of run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
