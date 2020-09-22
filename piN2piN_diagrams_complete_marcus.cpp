/****************************************************
 * piN2piN_diagrams_complete.cpp
 * 
 * So 22. Jul 18:19:35 CEST 2018
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
// #include "matrix_init.h"
#include "table_init_z.h"
#include "table_init_i.h"
#include "contract_diagrams.h"
#include "aff_key_conversion.h"
#include "gamma.h"
#include "zm4x4.h"

#define _Z_V3_OET_MOM

#define _ADD_BOUNDARY_PHASE
#define _USE_GET_CONSERVED_MOMENTUM_ID

using namespace cvc;

/***********************************************************
 * usage function
 ***********************************************************/
static inline void pvec_to_pref ( int * const pref, int * const pvec ) {
  int q[3] = { abs( pvec[0] ), abs( pvec[1] ), abs( pvec[2] ) };
  if ( q[1] < q[0] ) {
    int a = q[0];
    q[0] = q[1];
    q[1] = a;
  }
  if ( q[2] < q[1] ) {
    int a = q[1];
    q[1] = q[2];
    q[2] = a;
  }
  if ( q[1] < q[0] ) {
    int a = q[0];
    q[0] = q[1];
    q[1] = a;
  }
  memcpy ( pref, q, 3*sizeof(int) );
}  /* end of pvec_to_pref */

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
 /*  int const gamma_i2_list[gamma_i2_number]    = {  0 }; */
  // double const gamma_i2_sign[gamma_i2_number] = { +1 };

  // vertex f2, gamma_5 and id
  int const gamma_f2_number = 1;
  int const gamma_f2_list[gamma_f2_number]    = {  5 }; 
  /*int const gamma_f2_list[gamma_f2_number]    = {  0 };*/
  double const gamma_f2_sign[gamma_f2_number] = { +1 };

  // vertex f1, nucleon type
  int const gamma_f1_nucleon_number = 2;

  /* int gamma_f1_nucleon_list[gamma_f1_nucleon_number][2]    = { { 14,  4}, { 11,  5}, {  8,  4}, {  2,  5} };
  double gamma_f1_nucleon_sign[gamma_f1_nucleon_number][2] = { { +1, +1}, { +1, +1}, { -1, +1}, { -1, +1} }; */

  int gamma_f1_nucleon_list[gamma_f1_nucleon_number][2]    = { { 14,  4}, { 11,  5} };
  double gamma_f1_nucleon_sign[gamma_f1_nucleon_number][2] = { { +1, +1}, { +1, +1} };

  // int const gamma_f1_nucleon_list[gamma_f1_nucleon_number][2]    = {  { 5,  4 },  { 4,  5 },  { 6,  4 }, { 0,  5 } };
  // double const gamma_f1_nucleon_sign[gamma_f1_nucleon_number][2] = {  {+1, +1 },  {+1, +1 },  {+1, +1 }, {+1, +1 } };

  // vertex f1, Delta-type operators
  int const gamma_f1_delta_number                         = 3;
  // int const gamma_f1_delta_list[gamma_f1_delta_number][2] = { {  1,  4},  { 2,  4}, {  3,  4}, { 10,  4}, { 11,  4}, { 12,  4} };

  //                                                        C gx,      C gy       C gz       C gx gt    C gy gt    C gz gt
  int gamma_f1_delta_list[gamma_f1_delta_number][2]    = { {  9,  4}, {  0,  4}, {  7,  4} };
  double gamma_f1_delta_sign[gamma_f1_delta_number][2] = { { +1, +1}, { +1, +1}, { -1, +1} }; 

  //int gamma_f1_delta_list[gamma_f1_delta_number][2]    = { {  12,  4}, {  5,  4}, {  10,  4} }; 
  //double gamma_f1_delta_sign[gamma_f1_delta_number][2] = { {  +1, +1}, { +1, +1}, {  -1, +1} };

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
  fprintf(stdout, "[piN2piN_diagrams_complete] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /* initialize MPI parameters */
  mpi_init(argc, argv);

 /******************************************************
  * report git version
  ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [piN2piN_diagrams_complete] git version = %s\n", g_gitversion);
  }

  /******************************************************
   * set initial timestamp
   * - con: this is after parsing the input file
   ******************************************************/
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [piN2piN_diagrams_complete] %s# [piN2piN_diagrams_complete] start of run\n", ctime(&g_the_time));
    fflush(stdout);
  }

  /******************************************************
   *
   ******************************************************/
  if(init_geometry() != 0) {
    fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_geometry\n");
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

  gamma_matrix_type gamma_C, gamma_g5g0, gamma_g0g5;
  gamma_matrix_type gamma[16];

  for ( int i = 0; i < 16; i++ ) {
    gamma_matrix_set ( &gamma[i], i, 1 );
  }
  
  gamma_matrix_init ( &gamma_C );
  gamma_matrix_mult ( &gamma_C, &gamma[2], &gamma[0] );

  gamma_matrix_init ( &gamma_g0g5 );
  gamma_matrix_mult ( &gamma_g0g5, &gamma[0], &gamma[5] );

  gamma_matrix_init ( &gamma_g5g0 );
  gamma_matrix_mult ( &gamma_g5g0, &gamma[5], &gamma[0] );

  if ( g_verbose > 2 ) {
    gamma_matrix_printf (&gamma[0], "g0", stdout);
    gamma_matrix_printf (&gamma[2], "g2", stdout);
    gamma_matrix_printf (&gamma_C,  "C", stdout);
    gamma_matrix_printf (&gamma_g5g0,  "g5g0", stdout);
    gamma_matrix_printf (&gamma_g0g5,  "g0g5", stdout);
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
  * total momentum classes
  ******************************************************/
  int ptot_nclass = 0;
  int *ptot_nmem = NULL;
  int *** ptot_class = NULL;

  exitstatus = init_momentum_classes ( &ptot_class, &ptot_nmem, &ptot_nclass );
  if ( exitstatus  != 0 ) {
    fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_momentum_classes, status was %s %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(4);
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
       * AFF input file for base source
       ******************************************************/
      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN", Nconf, t_base );
      affr = aff_reader (filename);
      if( const char * aff_status_str = aff_reader_errstr(affr) ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(4);
      } else {
        fprintf(stdout, "# [piN2piN_diagrams_complete] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
      }
      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        EXIT(103);
      }

      /******************************************************
       * AFF oet input file for base source
       ******************************************************/
      sprintf(filename, "%s.%.4d.tsrc%.2d.aff", "piN_piN_oet", Nconf, t_base );
      affr_oet = aff_reader (filename);
      if ( const char * aff_status_str = aff_reader_errstr( affr_oet ) ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(4);
      } else {
        fprintf(stdout, "# [piN2piN_diagrams_complete] reading oet data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
      }
      if( (affn_oet = aff_reader_root( affr_oet )) == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error, aff oet reader is not initialized %s %d\n", __FILE__, __LINE__);
        EXIT(103);
      }

    }  // end of if io_proc == 2


    /**************************************************************************************
     * v3 type b_1_xi
     *
     *   Note: only one gamma_f2, which is gamma_5
     **************************************************************************************/
    double _Complex ****** b1xi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
    if ( b1xi == NULL ) {
      fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__);
      EXIT(47);
    }

    strcpy ( tag, "b_1_xi" );
    if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

    int igf2 = 0;

    for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

      for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

        for ( int isample = 0; isample < g_nsample; isample++ ) {

          if ( io_proc == 2 ) {
            aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], NULL, g_seq2_source_momentum_list[ipf2], g_source_coords_list[i_src], -1, gamma_f2_list[igf2], -1 );

            if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);
 
            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, b1xi[ipi2][0][ipf2][isample][0], T_global*12);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }
          }

        }  /* end of loop on samples */

      }  // end of loop on sink momentum pf2

    }  // end of loop on sequential source momentum pi2

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * v3 type b_3_xi
     *
     *   Note: only one gamma_f2, which is gamma_5
     **************************************************************************************/
    double _Complex ******b3xi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
    if ( b3xi == NULL ) {
      fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__);
      EXIT(47);
    }

    strcpy ( tag, "b_3_xi" );
    if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);
    

    for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

      for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

        for ( int isample = 0; isample < g_nsample; isample++ ) {

          if ( io_proc == 2 ) {
            aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], NULL, g_seq2_source_momentum_list[ipf2], g_source_coords_list[i_src], -1, gamma_f2_list[igf2], -1 );

            if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, b3xi[ipi2][0][ipf2][isample][0], T_global*12);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }
          }

        }  /* end of loop on samples */

      }  // end of loop on sink momentum pf2

    }  // end of loop on sequential source momentum pi2

    /**************************************************************************************/
    /**************************************************************************************/


    /**************************************************************************************
     * v3 type b_7_xi
     *
     *   Note: only one gamma_f2, which is gamma_5
     **************************************************************************************/
    double _Complex ****** b7xi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
    if ( b7xi == NULL ) {
      fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__);
      EXIT(47);
    }

    strcpy ( tag, "b_7_xi" );
    if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);


    for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

      for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

        for ( int isample = 0; isample < g_nsample; isample++ ) {

          if ( io_proc == 2 ) {
            aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], NULL, g_seq2_source_momentum_list[ipf2], g_source_coords_list[i_src], -1, gamma_f2_list[igf2], -1 );

            if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, b7xi[ipi2][0][ipf2][isample][0], T_global*12);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }
          }

        }  /* end of loop on samples */

      }  // end of loop on sink momentum pf2

    }  // end of loop on sequential source momentum pi2

    /**************************************************************************************/
    /**************************************************************************************/


    /**************************************************************************************
     * v3 type b_9_xi
     *
     *   Note: only one gamma_f2, which is gamma_5
     **************************************************************************************/
    double _Complex ****** b9xi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
    if ( b9xi == NULL ) {
      fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__);
      EXIT(47);
    }

    strcpy ( tag, "b_9_xi" );
    if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);


    for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

      for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

        for ( int isample = 0; isample < g_nsample; isample++ ) {

          if ( io_proc == 2 ) {
            aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], NULL, g_seq2_source_momentum_list[ipf2], g_source_coords_list[i_src], -1, gamma_f2_list[igf2], -1 );

            if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, b9xi[ipi2][0][ipf2][isample][0], T_global*12);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }
          }

        }  /* end of loop on samples */

      }  // end of loop on sink momentum pf2

    }  // end of loop on sequential source momentum pi2

    /**************************************************************************************/
    /**************************************************************************************/


    /**************************************************************************************
     * v3 type b_1_xi
     *
     *   Note: only one gamma_f2, which is gamma_5
     **************************************************************************************/
    double _Complex ****** b17xi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
    if ( b17xi == NULL ) {
      fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__);
      EXIT(47);
    }

    strcpy ( tag, "b_17_xi" );
    if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);


    for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

      for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

        for ( int isample = 0; isample < g_nsample; isample++ ) {

          if ( io_proc == 2 ) {
            aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], NULL, g_seq2_source_momentum_list[ipf2], g_source_coords_list[i_src], -1, gamma_f2_list[igf2], -1 );

            if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, b17xi[ipi2][0][ipf2][isample][0], T_global*12);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }
          }

        }  /* end of loop on samples */

      }  // end of loop on sink momentum pf2

    }  // end of loop on sequential source momentum pi2

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


      /**************************************************************************************
       * v2 type b_1_phi
       **************************************************************************************/
      double _Complex ***** b1phi = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 );
      if ( b1phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "b_1_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

        for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

          for ( int isample = 0; isample < g_nsample; isample++ ) {

            if ( io_proc == 2 ) {
              aff_key_conversion ( aff_tag, tag, isample, NULL, g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );
              if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

              affdir = aff_reader_chpath (affr, affn, aff_tag );
              exitstatus = aff_node_get_complex (affr, affdir, b1phi[igf1][ipf1][isample][0], T_global*192);
              if( exitstatus != 0 ) {
                fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }
            }

          }  /* end of loop on Gamma_f1 */

        }  /* end of loop on samples */

      }  /* end of loop on sink momentum pf1 */

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * v2 type b_3_phi
       **************************************************************************************/
      double _Complex ***** b3phi = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 );
      if ( b3phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "b_3_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

        for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

          for ( int isample = 0; isample < g_nsample; isample++ ) {

            if ( io_proc == 2 ) {
              aff_key_conversion ( aff_tag, tag, isample, NULL, g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );
              if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

              affdir = aff_reader_chpath (affr, affn, aff_tag );
              exitstatus = aff_node_get_complex (affr, affdir, b3phi[igf1][ipf1][isample][0], T_global*192);
              if( exitstatus != 0 ) {
                fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }
            }

          }  /* end of loop on Gamma_f1 */

        }  /* end of loop on samples */

      }  /* end of loop on sink momentum pf1 */

      /**************************************************************************************/
      /**************************************************************************************/


      /**************************************************************************************
       * v2 type b_4_phi
       **************************************************************************************/
      double _Complex ***** b4phi = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 );
      if ( b4phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "b_4_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

        for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

          for ( int isample = 0; isample < g_nsample; isample++ ) {

            if ( io_proc == 2 ) {
              aff_key_conversion ( aff_tag, tag, isample, NULL, g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );
              if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

              affdir = aff_reader_chpath (affr, affn, aff_tag );
              exitstatus = aff_node_get_complex (affr, affdir, b4phi[igf1][ipf1][isample][0], T_global*192);
              if( exitstatus != 0 ) {
                fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }
            }

          }  /* end of loop on Gamma_f1 */

        }  /* end of loop on samples */

      }  /* end of loop on sink momentum pf1 */

      /**************************************************************************************/
      /**************************************************************************************/


      /**************************************************************************************
       * v2 type b_13_phi
       **************************************************************************************/
      double _Complex ***** b13phi = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 );
      if ( b13phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "b_13_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

        for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

          for ( int isample = 0; isample < g_nsample; isample++ ) {

            if ( io_proc == 2 ) {
              aff_key_conversion ( aff_tag, tag, isample, NULL, g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );
              if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

              affdir = aff_reader_chpath (affr, affn, aff_tag );
              exitstatus = aff_node_get_complex (affr, affdir, b13phi[igf1][ipf1][isample][0], T_global*192);
              if( exitstatus != 0 ) {
                fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }
            }

          }  /* end of loop on Gamma_f1 */

        }  /* end of loop on samples */

      }  /* end of loop on sink momentum pf1 */

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * v2 type b_15_phi
       **************************************************************************************/
      double _Complex ***** b15phi = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 );
      if ( b15phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "b_15_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

        for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

          for ( int isample = 0; isample < g_nsample; isample++ ) {

            if ( io_proc == 2 ) {
              aff_key_conversion ( aff_tag, tag, isample, NULL, g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );
              if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

              affdir = aff_reader_chpath (affr, affn, aff_tag );
              exitstatus = aff_node_get_complex (affr, affdir, b15phi[igf1][ipf1][isample][0], T_global*192);
              if( exitstatus != 0 ) {
                fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
      for ( int iclass = 0; iclass  < ptot_nclass; iclass++ ) {

        int pref[3] = { ptot_class[iclass][0][0], ptot_class[iclass][0][1], ptot_class[iclass][0][2] };


        /******************************************************
         * AFF output file
         ******************************************************/
        if ( io_proc == 2 ) {
          sprintf(filename, "%s.b.PX%dPY%dPZ%d.%.4d.t%dx%dy%dz%d.aff", "piN_piN_diagrams",
              pref[0], pref[1], pref[2], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
          affw = aff_writer (filename);
          if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
            EXIT(4);
          } else {
            fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
          }
        }  // end of if io_proc == 2

      for ( int imem = 0; imem < ptot_nmem[iclass]; imem++ ) {

        int ptot[3]  = { ptot_class[iclass][imem][0], ptot_class[iclass][imem][1], ptot_class[iclass][imem][2] };

        int mptot[3] = { -ptot[0], -ptot[1], -ptot[2] };

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
        int * sink_momentum_id   =  get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, ptot,  g_sink_momentum_list, g_sink_momentum_number );
        if ( sink_momentum_id == NULL ) {
          // fprintf(stderr, "[piN2piN_diagrams_complete] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
          // EXIT(1);
          fprintf(stdout, "# [piN2piN_diagrams_complete] sink_momentum_id empty; continue %s %d\n", __FILE__, __LINE__ );
          continue;
        }

        int * source_momentum_id =  get_conserved_momentum_id ( g_seq_source_momentum_list,  g_seq_source_momentum_number,  mptot, g_sink_momentum_list, g_sink_momentum_number );
        if ( source_momentum_id == NULL ) {
          // fprintf(stderr, "[piN2piN_diagrams_complete] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
          // EXIT(1);
          fprintf(stdout, "# [piN2piN_diagrams_complete] source_momentum_id empty; continue %s %d\n", __FILE__, __LINE__ );
          continue;
        }

#endif  /* of if _USE_GET_CONSERVED_MOMENTUM_ID */

        /**************************************************************************************
         * loop on B diagrams 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
         **************************************************************************************/
        int const perm[20][4] = { { 1, 0, 2, 3 }, //B1
				  { 2, 0, 1, 3 }, //B2
                                  { 3, 0, 1, 2 }, //B3
                                  { 0, 1, 3, 2 }, //B4
                                  { 3, 0, 2, 1 }, //B5
                                  { 0, 2, 3, 1 }, //B6
                                  { 1, 0, 2, 3 }, //B7
                                  { 2, 0, 1, 3 }, //B8
                                  { 1, 0, 3, 2 }, //B9
                                  { 0, 3, 1, 2 }, //B10
                                  { 2, 0, 3, 1 }, //B11
                                  { 0, 3, 2, 1 }, //B12
                                  { 0, 3, 1, 2 }, //B13
                                  { 0, 3, 2, 1 }, //B14
                                  { 3, 0, 1, 2 }, //B15
                                  { 3, 0, 2, 1 }, //B16
                                  { 0, 1, 3, 2 }, //B17
                                  { 0, 2, 3, 1 }, //B18
                                  { 1, 0, 3, 2 }, //B19
                                  { 2, 0, 3, 1 } }; //B20

        for ( int iperm = 0; iperm < 20; iperm++ ) {
          double _Complex ****** bxi = NULL, *****bphi = NULL;

          if ( iperm == 0  ) {
            bxi  = b1xi;
            bphi = b1phi;
          } else if ( iperm == 1  ) {
            bxi  = b1xi;
            bphi = b1phi;
          } else if ( iperm == 2  ) {
            bxi  = b3xi;
            bphi = b3phi;
          } else if ( iperm == 3  ) {
            bxi  = b3xi;
            bphi = b4phi;
          } else if ( iperm == 4  ) {
            bxi  = b3xi;
            bphi = b3phi;
          } else if ( iperm == 5  ) {
            bxi  = b3xi;
            bphi = b4phi;
          } else if ( iperm == 6  ) {
            bxi  = b7xi;
            bphi = b1phi;
          } else if ( iperm == 7  ) {
            bxi  = b7xi;
            bphi = b1phi;
          } else if ( iperm == 8  ) {
            bxi  = b9xi;
            bphi = b3phi;
          } else if ( iperm == 9  ) {
            bxi  = b9xi;
            bphi = b4phi;
          } else if ( iperm == 10  ) {
            bxi  = b9xi;
            bphi = b3phi;
          } else if ( iperm == 11  ) {
            bxi  = b9xi;
            bphi = b4phi;
          } else if ( iperm == 12  ) {
            bxi  = b1xi;
            bphi = b13phi;
          } else if ( iperm == 13  ) {
            bxi  = b1xi;
            bphi = b13phi;
          } else if ( iperm == 14  ) {
            bxi  = b1xi;
            bphi = b15phi;
          } else if ( iperm == 15  ) {
            bxi  = b1xi;
            bphi = b15phi;
          } else if ( iperm == 16  ) {
            bxi  = b17xi;
            bphi = b13phi;
          } else if ( iperm == 17  ) {
            bxi  = b17xi;
            bphi = b13phi;
          } else if ( iperm == 18  ) {
            bxi  = b17xi;
            bphi = b15phi;
          } else if ( iperm == 19  ) {
            bxi  = b17xi;
            bphi = b15phi;
          }

#if 0
          /******************************************************
           * AFF output file
           ******************************************************/
          if ( io_proc == 2 ) {
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
          }  // end of if io_proc == 2
#endif  // of if 0

          /**************************************************************************************
           * loop on pf2
           **************************************************************************************/
          for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

            int pf2[3] = {
              g_seq2_source_momentum_list[ipf2][0],
              g_seq2_source_momentum_list[ipf2][1],
              g_seq2_source_momentum_list[ipf2][2] 
            };

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
            int ipf1 = sink_momentum_id[ipf2];
            if ( ipf1 == -1 ) {
              if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagrams_complete] Warning, skipping seq2 momentum no %d  (  %3d, %3d, %3d )\n", ipf2,
                  g_seq2_source_momentum_list[ipf2][0], g_seq2_source_momentum_list[ipf2][1], g_seq2_source_momentum_list[ipf2][2] );
              continue;
            }
#else
          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

#endif  /* of _USE_GET_CONSERVED_MOMENTUM_ID  */

            int pf1[3] = {
              g_sink_momentum_list[ipf1][0],
              g_sink_momentum_list[ipf1][1],
              g_sink_momentum_list[ipf1][2] 
            };
           
            /**************************************************************************************
             * loop on gf1
             **************************************************************************************/
            for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {
              int igf2 = 0;

              gamma_matrix_type gf12;
              gamma_matrix_set ( &gf12, gamma_f1_nucleon_list[igf1][1], gamma_f1_nucleon_sign[igf1][1] );

              /**************************************************************************************
               * loop on pi2
               **************************************************************************************/
              for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

                int pi2[3] =  {
                    g_seq_source_momentum_list[ipi2][0],
                    g_seq_source_momentum_list[ipi2][1],
                    g_seq_source_momentum_list[ipi2][2]
                };


#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
                int ipi1 = source_momentum_id[ipi2];
                if ( ipi1 == -1 ) {
                  if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagrams_complete] Warning, skipping seq momentum no %d  (  %3d, %3d, %3d ) for ptot ( %3d, %3d, %3d )\n", ipi2, 
                      g_seq_source_momentum_list[ipi2][0], g_seq_source_momentum_list[ipi2][1], g_seq_source_momentum_list[ipi2][2],               
                      ptot[0], ptot[1], ptot[2] );
                  continue;
                }
                int pi1[3] = {
                    g_sink_momentum_list[ipi1][0],
                    g_sink_momentum_list[ipi1][1],
                    g_sink_momentum_list[ipi1][2]
                };

#else
                int pi1[3] = {
                  - ( pf1[0] + pf2[0] + pi2[0] ),
                  - ( pf1[1] + pf2[1] + pi2[1] ),
                  - ( pf1[2] + pf2[2] + pi2[2] ) 
                };

#endif  /* of _USE_GET_CONSERVED_MOMENTUM_ID */

                /**************************************************************************************
                 * loop on gi1
                 **************************************************************************************/
                for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) {
                  int igi2 = 0;

                  char aff_tag_suffix[400];

                  contract_diagram_key_suffix ( aff_tag_suffix, 
                      gamma_f2_list[igf2],                                            pf2,
                      gamma_f1_nucleon_list[igf1][0], gamma_f1_nucleon_list[igf1][1], pf1,
                      gamma_i2_list[igi2],                                            pi2,
                      gamma_f1_nucleon_list[igi1][0], gamma_f1_nucleon_list[igi1][1], pi1, 
                      gsx );


                  /**************************************************************************************
                   * set inner gamma matrix structure for baryon at source
                   **************************************************************************************/

                  gamma_matrix_type gi11, gi12;
                  gamma_matrix_set ( &gi11, gamma_f1_nucleon_list[igi1][0], gamma_f1_nucleon_sign[igi1][0] );
                  gamma_matrix_set ( &gi12, gamma_f1_nucleon_list[igi1][1], gamma_f1_nucleon_sign[igi1][1] );

                  gamma_matrix_transposed ( &gi11, &gi11 );
                  if ( g_verbose > 2 ) {
                    char name[20];
                    sprintf(name, "G%.2d_transposed", gi11.id);
                    gamma_matrix_printf (&gi11, name, stdout);
                  }

                  /**************************************************************************************/
                  /**************************************************************************************/

                  double _Complex ***diagram = init_3level_ztable ( T_global, 4, 4 );
                  if ( diagram == NULL ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
                    EXIT(104);
                  }

                  // reduce to diagram, average over stochastic samples
                  if ( ( exitstatus = contract_diagram_sample ( diagram, bxi[ipi2][0][ipf2], bphi[igf1][ipf1], g_nsample, perm[iperm], gi11, T_global ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  // add boundary phase
#ifdef _ADD_BOUNDARY_PHASE
                  if ( ( exitstatus = correlator_add_baryon_boundary_phase ( diagram, gsx[0], +1, T_global ) ) != 0 ) {
                    fprintf( stderr, "[piN2piN_diagrams_complete] Error from correlator_add_baryon_boundary_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(106);
                  }
#endif
                  // add source phase
                  if ( ( exitstatus = correlator_add_source_phase ( diagram, pi1, &(gsx[1]), T_global ) ) != 0 ) {
                    fprintf( stderr, "[piN2piN_diagrams_complete] Error from correlator_add_source_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(107);
                  }

                  // add outer gamma matrices
                  if ( ( exitstatus =  contract_diagram_zm4x4_field_mul_gamma_lr ( diagram, diagram, gf12, gi12, T_global ) ) != 0 ) {
                    fprintf( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_mul_gamma_lr, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(108);
                  }

                  // add phase from phase convention
                  double _Complex const zsign = contract_diagram_get_correlator_phase ( "mxb-mxb", 
                      gamma_f1_nucleon_list[igi1][0], 
                      gamma_f1_nucleon_list[igi1][1], 
                      gamma_i2_list[igi2], 
                      gamma_f1_nucleon_list[igf1][0], 
                      gamma_f1_nucleon_list[igf1][1], 
                      gamma_f2_list[igf2] );

                  if ( ( exitstatus = contract_diagram_zm4x4_field_ti_eq_co ( diagram, zsign, T_global ) ) != 0 ) {
                    fprintf ( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_ti_eq_co, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                    EXIT(109)
                  }


                  // write AFF fwd
                  sprintf(aff_tag, "/%s/b%d/fwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( g_verbose > 4 ) fprintf ( stdout, "# [piN2piN_diagrams_complete] AFF tag = \"%s\" %s %d\n", aff_tag, __FILE__, __LINE__ );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(110);
                  }

                  // write AFF bwd
                  sprintf(aff_tag, "/%s/b%d/bwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( g_verbose > 4 ) fprintf ( stdout, "# [piN2piN_diagrams_complete] AFF tag = \"%s\" %s %d\n", aff_tag, __FILE__, __LINE__ );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(111);
                  }

                  fini_3level_ztable ( &diagram );

                }  // end of loop on Gamma_i1

              }  // end of loop on p_i2

            }  // end of loop on Gamma_f1
#ifndef _USE_GET_CONSERVED_MOMENTUM_ID
          }  // end of loop on p_f1
#endif  /* of _USE_GET_CONSERVED_MOMENTUM_ID */

          }  // end of loop on p_f2


        }  // end of loop on permutations for B diagrams

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
        free ( sink_momentum_id );
        free ( source_momentum_id );
#endif

      }  // end of loop on ptot_nmem

        if ( io_proc == 2 ) {
          if ( const char * aff_status_str = aff_writer_close (affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__ );
            EXIT(111);
          }
        }

      }  // end of loop on ptot_nclass

      fini_5level_ztable ( &b1phi );
      fini_5level_ztable ( &b4phi );
      fini_5level_ztable ( &b3phi );
      fini_5level_ztable ( &b13phi );
      fini_5level_ztable ( &b15phi );


      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * v3 type w_1_xi
       **************************************************************************************/
      double _Complex ***** w1xi = init_5level_ztable ( gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
      if ( w1xi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(47);
      }

      strcpy ( tag, "w_1_xi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      int igf2 = 0;

      for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

        for ( int isample = 0; isample < g_nsample; isample++ ) {

          if ( io_proc == 2 ) {
            aff_key_conversion ( aff_tag, tag, isample, NULL, NULL, g_seq2_source_momentum_list[ipf2], gsx, -1, gamma_f2_list[igf2], -1 );
            if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);
 
            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, w1xi[0][ipf2][isample][0], T_global*12);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }

          }  // end of if io proc == 2

        }  // end of loop on samples

      }  // end of loop on sink momentum pf2

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * v3 type w_5_xi
       **************************************************************************************/
      double _Complex ***** w5xi = init_5level_ztable ( gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
      if ( w5xi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(47);
      }

      strcpy ( tag, "w_5_xi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);


      for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

        for ( int isample = 0; isample < g_nsample; isample++ ) {

          if ( io_proc == 2 ) {
            aff_key_conversion ( aff_tag, tag, isample, NULL, NULL, g_seq2_source_momentum_list[ipf2], gsx, -1, gamma_f2_list[igf2], -1 );
            if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, w5xi[0][ipf2][isample][0], T_global*12);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }

          }  // end of if io proc == 2

        }  // end of loop on samples

      }  // end of loop on sink momentum pf2

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * v3 type w_13_xi
       **************************************************************************************/
      double _Complex ***** w13xi = init_5level_ztable ( gamma_f2_number, g_seq2_source_momentum_number, g_nsample, T_global, 12 );
      if ( w13xi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(47);
      }

      strcpy ( tag, "w_13_xi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);


      for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

        for ( int isample = 0; isample < g_nsample; isample++ ) {

          if ( io_proc == 2 ) {
            aff_key_conversion ( aff_tag, tag, isample, NULL, NULL, g_seq2_source_momentum_list[ipf2], gsx, -1, gamma_f2_list[igf2], -1 );
            if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

            affdir = aff_reader_chpath (affr, affn, aff_tag );
            exitstatus = aff_node_get_complex (affr, affdir, w13xi[0][ipf2][isample][0], T_global*12);
            if( exitstatus != 0 ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_1_phi" );
      fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w1phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_3_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w3phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
       * w_5_phi
       **************************************************************************************/
      double _Complex ****** w5phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_5_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w5phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
       * w_6_phi
       **************************************************************************************/
      double _Complex ****** w6phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_6_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w6phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
       * w_9_phi
       **************************************************************************************/
      double _Complex ****** w9phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_9_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w9phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
       * w_10_phi
       **************************************************************************************/
      double _Complex ****** w10phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_10_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w10phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
       * w_13_phi
       **************************************************************************************/
      double _Complex ****** w13phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_13_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w13phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
       * w_14_phi
       **************************************************************************************/
      double _Complex ****** w14phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_14_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w14phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
       * w_17_phi
       **************************************************************************************/
      double _Complex ****** w17phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_17_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w17phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
       * w_18_phi
       **************************************************************************************/
      double _Complex ****** w18phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_18_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w18phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
       * w_25_phi
       **************************************************************************************/
      double _Complex ****** w25phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_25_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w25phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
       * w_27_phi
       **************************************************************************************/
      double _Complex ****** w27phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_27_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w27phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
       * w_29_phi
       **************************************************************************************/
      double _Complex ****** w29phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_29_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w29phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
       * w_31_phi
       **************************************************************************************/
      double _Complex ****** w31phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_31_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w31phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
       * w_33_phi
       **************************************************************************************/
      double _Complex ****** w33phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_33_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w33phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
       * w_35_phi
       **************************************************************************************/
      double _Complex ****** w35phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, g_nsample, T_global, 192 ) ;
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "w_35_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            for ( int isample = 0; isample < g_nsample; isample++ ) {

              if ( io_proc == 2 ) {
                aff_key_conversion ( aff_tag, tag, isample, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], -1 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, w35phi[ipi2][igf1][ipf1][isample][0], T_global*192);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
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
      for ( int iclass = 0; iclass  < ptot_nclass; iclass++ ) {

        int pref[3] = { ptot_class[iclass][0][0], ptot_class[iclass][0][1], ptot_class[iclass][0][2] };

        /******************************************************
         * AFF output file
         ******************************************************/
        if ( io_proc == 2 ) {
          sprintf(filename, "%s.w.PX%dPY%dPZ%d.%.4d.t%dx%dy%dz%d.aff", "piN_piN_diagrams", 
              pref[0], pref[1], pref[2], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
          affw = aff_writer (filename);
          if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
            EXIT(4);
          } else {
            fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
          }
        }  // end of if io_proc == 2


      for ( int imem = 0; imem < ptot_nmem[iclass]; imem++ ) {

        int ptot[3]  = { ptot_class[iclass][imem][0], ptot_class[iclass][imem][1], ptot_class[iclass][imem][2] };

        int mptot[3] = { -ptot[0], -ptot[1], -ptot[2] };

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
        int * sink_momentum_id   =  get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, ptot,  g_sink_momentum_list, g_sink_momentum_number );
        if ( sink_momentum_id == NULL ) {
          // fprintf(stderr, "[piN2piN_diagrams_complete] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
          // EXIT(1);
          fprintf(stdout, "# [piN2piN_diagrams_complete] sink_momentum_id empty; continue %s %d\n", __FILE__, __LINE__ );
          continue;
        }

        int * source_momentum_id =  get_conserved_momentum_id ( g_seq_source_momentum_list,  g_seq_source_momentum_number,  mptot, g_sink_momentum_list, g_sink_momentum_number );
        if ( source_momentum_id == NULL ) {
          // fprintf(stderr, "[piN2piN_diagrams_complete] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
          // EXIT(1);
          fprintf(stdout, "# [piN2piN_diagrams_complete] source_momentum_id empty; continue %s %d\n", __FILE__, __LINE__ );
          continue;
        }
#endif  /* of _USE_GET_CONSERVED_MOMENTUM_ID */

        /**************************************************************************************
         * W diagrams 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
         *            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36
         **************************************************************************************/
        int const perm[36][4] = { { 1, 0, 2, 3 }, //W1
                                 { 3, 0, 2, 1 }, //W2
                                 { 2, 0, 3, 1 }, //W3
                                 { 2, 0, 1, 3 }, //W4
                                 { 0, 2, 3, 1 }, //w5
                                 { 2, 0, 3, 1 }, //W6
                                 { 0, 2, 1, 3 }, //W7
                                 { 2, 0, 1, 3 }, //W8
                                 { 2, 0, 1, 3 }, //W9
                                 { 0, 2, 1, 3 }, //W10
                                 { 2, 0, 3, 1 }, //W11
                                 { 0, 2, 3, 1 }, //W12
                                 { 3, 0, 1, 2 }, //W13
                                 { 1, 0, 3, 2 }, //W14
                                 { 1, 0, 3, 2 }, //W15
                                 { 3, 0, 1, 2 }, //W16
                                 { 0, 1, 3, 2 }, //W17
                                 { 1, 0, 3, 2 }, //W18
                                 { 0, 3, 1, 2 }, //W19
                                 { 3, 0, 1, 2 }, //W20
                                 { 3, 0, 2, 1 }, //W21
                                 { 2, 0, 3, 1 }, //W22
                                 { 1, 0, 2, 3 }, //W23
                                 { 2, 0, 1, 3 }, //W24
                                 { 0, 3, 1, 2 }, //W25
                                 { 0, 1, 3, 2 }, //W26
                                 { 3, 0, 1, 2 }, //W27
                                 { 1, 0, 3, 2 }, //W28
                                 { 0, 3, 2, 1 }, //W29
                                 { 0, 1, 2, 3 }, //W30
                                 { 3, 0, 2, 1 }, //W31
                                 { 1, 0, 2, 3 }, //W32
                                 { 0, 1, 2, 3 }, //W33
                                 { 0, 3, 2, 1 }, //W34
                                 { 1, 0, 2, 3 }, //W35
                                 { 3, 0, 2, 1 }};//W36  
        for ( int iperm = 0; iperm < 36; iperm++ ) {

          double _Complex ***** wxi = NULL, ******wphi = NULL;

          if ( ( iperm == 0 ) || ( iperm == 1 ) ) {
            wxi  = w1xi;
            wphi = w1phi;
          } else if ( ( iperm == 2 ) || ( iperm == 3 ) ) {
            wxi  = w1xi;
            wphi = w3phi;
          }
          else if ( iperm == 4 ){
            wxi = w5xi;
            wphi= w5phi; 
          }
          else if ( iperm == 5 ){
            wxi = w5xi;
            wphi= w6phi;
          }
          else if ( iperm == 6 ){
            wxi = w5xi;
            wphi= w5phi;
          }
          else if ( iperm == 7 ){
            wxi = w5xi;
            wphi= w6phi;
          }
          else if ( iperm == 8 ){
            wxi = w5xi;
            wphi= w9phi;
          }
          else if ( iperm == 9 ){
            wxi = w5xi;
            wphi= w10phi;
          }
          else if ( iperm == 10 ){
            wxi = w5xi;
            wphi= w9phi;
          }
          else if ( iperm == 11 ){
            wxi = w5xi;
            wphi= w10phi;
          }
          else if ( iperm == 12 ){
            wxi = w13xi;
            wphi= w13phi;
          }
          else if ( iperm == 13 ){
            wxi = w13xi;
            wphi= w14phi;
          }
          else if ( iperm == 14 ){
            wxi = w13xi;
            wphi= w13phi;
          }
          else if ( iperm == 15 ){
            wxi = w13xi;
            wphi= w14phi;
          }
          else if ( iperm == 16 ){
            wxi = w5xi;
            wphi= w17phi;
          }
          else if ( iperm == 17 ){
            wxi = w5xi;
            wphi= w18phi;
          }
          else if ( iperm == 18 ){
            wxi = w5xi;
            wphi= w17phi;
          }
          else if ( iperm == 19 ){
            wxi = w5xi;
            wphi= w18phi;
          }
          else if ( iperm == 20 ){
            wxi = w13xi;
            wphi= w3phi;
          }
          else if ( iperm == 21 ){
            wxi = w13xi;
            wphi= w1phi;
          }
          else if ( iperm == 22 ){
            wxi = w13xi;
            wphi= w3phi;
          }
          else if ( iperm == 23 ){
            wxi = w13xi;
            wphi= w1phi;
          }
          else if ( iperm == 24 ){
            wxi = w1xi;
            wphi= w25phi;
          }
          else if ( iperm == 25 ){
            wxi = w1xi;
            wphi= w25phi;
          }
          else if ( iperm == 26 ){
            wxi = w1xi;
            wphi= w27phi;
          }
          else if ( iperm == 27 ){
            wxi = w1xi;
            wphi= w27phi;
          }
          else if ( iperm == 28 ){
            wxi = w1xi;
            wphi= w29phi;
          }
          else if ( iperm == 29 ){
            wxi = w1xi;
            wphi= w29phi;
          }
          else if ( iperm == 30 ){
            wxi = w1xi;
            wphi= w31phi;
          }
          else if ( iperm == 31 ){
            wxi = w1xi;
            wphi= w31phi;
          }
          else if ( iperm == 32 ){
            wxi = w1xi;
            wphi= w33phi;
          }
          else if ( iperm == 33 ){
            wxi = w1xi;
            wphi= w33phi;
          }
          else if ( iperm == 34 ){
            wxi = w1xi;
            wphi= w35phi;
          }
          else if ( iperm == 35 ){
            wxi = w1xi;
            wphi= w35phi;
          }

       

#if 0
          /******************************************************
           * AFF output file
           ******************************************************/
          if ( io_proc == 2 ) {
            sprintf(filename, "%s.W%d.%.4d.PX%dPY%dPZ%d.t%dx%dy%dz%d.aff", "piN_piN_diagrams", iperm+1, Nconf, 
                g_total_momentum_list[iptot][0], g_total_momentum_list[iptot][1], g_total_momentum_list[iptot][2],
                gsx[0], gsx[1], gsx[2], gsx[3] );
            affw = aff_writer (filename);
            if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
              EXIT(4);
            } else {
              fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
            }
          }  // end of if io_proc == 2
#endif  // of if 0

          /**************************************************************************************/
          /**************************************************************************************/

          /**************************************************************************************
           * loop on pf2
           **************************************************************************************/
          for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

            int pf2[3] = { g_seq2_source_momentum_list[ipf2][0], g_seq2_source_momentum_list[ipf2][1], g_seq2_source_momentum_list[ipf2][2] };

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
            int ipf1 = sink_momentum_id[ipf2];
            if ( ipf1 == -1 ) {
              if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagrams_complete] Warning, skipping seq2 momentum no %d  (  %3d, %3d, %3d )\n", ipf2,
                  g_seq2_source_momentum_list[ipf2][0], g_seq2_source_momentum_list[ipf2][1], g_seq2_source_momentum_list[ipf2][2] );
              continue;
            }
#else
          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

#endif  /* _USE_GET_CONSERVED_MOMENTUM_ID */

            int pf1[3] = {
              g_sink_momentum_list[ipf1][0],
              g_sink_momentum_list[ipf1][1],
              g_sink_momentum_list[ipf1][2]
            };

            /**************************************************************************************/
            /**************************************************************************************/

            /**************************************************************************************
             * loop on gf1
             **************************************************************************************/
            for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {
              int igf2 = 0;

              /**************************************************************************************
               * make gamma matrix gf12 for later left-multiplication
               **************************************************************************************/
              gamma_matrix_type gf12;
              gamma_matrix_set ( &gf12, gamma_f1_nucleon_list[igf1][1], gamma_f1_nucleon_sign[igf1][1] );


              /**************************************************************************************
               * loop on pi2
               **************************************************************************************/
              for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

                int pi2[3] = { g_seq_source_momentum_list[ipi2][0], g_seq_source_momentum_list[ipi2][1], g_seq_source_momentum_list[ipi2][2] };

                /**************************************************************************************/
                /**************************************************************************************/
#ifdef _USE_GET_CONSERVED_MOMENTUM_ID

                int ipi1 = source_momentum_id[ipi2];
                if ( ipi1 == -1 ) {
                  if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagrams_complete] Warning, skipping seq momentum no %d  (  %3d, %3d, %3d ) for ptot ( %3d, %3d, %3d )\n", ipi2, 
                      g_seq_source_momentum_list[ipi2][0], g_seq_source_momentum_list[ipi2][1], g_seq_source_momentum_list[ipi2][2],
                      ptot[0], ptot[1], ptot[2] );
                  continue;
                }
                int pi1[3] = {
                    g_sink_momentum_list[ipi1][0],
                    g_sink_momentum_list[ipi1][1],
                    g_sink_momentum_list[ipi1][2]
                };

#else
                int pi1[3] = {
                  - ( pf1[0] + pf2[0] + pi2[0] ),
                  - ( pf1[1] + pf2[1] + pi2[1] ),
                  - ( pf1[2] + pf2[2] + pi2[2] )
                };
#endif  /* of _USE_GET_CONSERVED_MOMENTUM_ID */

                /**************************************************************************************
                 * loop on gi1
                 **************************************************************************************/
                for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) {
                  int igi2 = 0;


                  char aff_tag_suffix[400];

                  contract_diagram_key_suffix ( aff_tag_suffix,
                      gamma_f2_list[igf2],                                            pf2,
                      gamma_f1_nucleon_list[igf1][0], gamma_f1_nucleon_list[igf1][1], pf1,
                      gamma_i2_list[igi2],                                            pi2,
                      gamma_f1_nucleon_list[igi1][0], gamma_f1_nucleon_list[igi1][1], pi1,
                      gsx );


                  /**************************************************************************************/
                  /**************************************************************************************/

                  gamma_matrix_type gi11, gi12;
                  gamma_matrix_set ( &gi11, gamma_f1_nucleon_list[igi1][0], gamma_f1_nucleon_sign[igi1][0] );
                  gamma_matrix_set ( &gi12, gamma_f1_nucleon_list[igi1][1], gamma_f1_nucleon_sign[igi1][1] );


                  // no transpostion here with choice of perm above; cf. pdf
                  // gamma_matrix_transposed ( &gi11, &gi11 );
                  if ( g_verbose > 2 ) {
                    char name[20];
                    sprintf(name, "G%.2d", gi11.id);
                    gamma_matrix_printf (&gi11, name, stdout);
                  }

                  /**************************************************************************************/
                  /**************************************************************************************/

                  double _Complex ***diagram = init_3level_ztable ( T_global, 4, 4 );
                  if ( diagram == NULL ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
                    EXIT(47);
                  }

                  // reduce to diagram, average over stochastic samples
                  if ( ( exitstatus = contract_diagram_sample ( diagram, wxi[0][ipf2], wphi[ipi2][igf1][ipf1], g_nsample, perm[iperm], gi11, T_global ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_sample, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  // add boundary phase
#ifdef _ADD_BOUNDARY_PHASE
                  if ( ( exitstatus = correlator_add_baryon_boundary_phase ( diagram, gsx[0], +1, T_global ) ) != 0 ) {
                    fprintf( stderr, "[piN2piN_diagrams_complete] Error from correlator_add_baryon_boundary_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(106);
                  }
#endif
                  // add source phase
                  if ( ( exitstatus = correlator_add_source_phase ( diagram, pi1, &(gsx[1]), T_global ) ) != 0 ) {
                    fprintf( stderr, "[piN2piN_diagrams_complete] Error from correlator_add_source_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(107);
                  }
  
                  // add outer gamma matrices
                  if ( ( exitstatus =  contract_diagram_zm4x4_field_mul_gamma_lr ( diagram, diagram, gf12, gi12, T_global ) ) != 0 ) {
                    fprintf( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_mul_gamma_lr, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(108);
                  }
  
                  // add phase from phase convention
                  double _Complex const zsign = contract_diagram_get_correlator_phase ( "mxb-mxb",
                      gamma_f1_nucleon_list[igi1][0],
                      gamma_f1_nucleon_list[igi1][1],
                      gamma_i2_list[igi2],
                      gamma_f1_nucleon_list[igf1][0],
                      gamma_f1_nucleon_list[igf1][1],
                      gamma_f2_list[igf2] );
  
                  if ( ( exitstatus = contract_diagram_zm4x4_field_ti_eq_co ( diagram, zsign, T_global ) ) != 0 ) {
                    fprintf ( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_ti_eq_co, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                    EXIT(109)
                  }
  
                  // AFF write fwd
                  sprintf(aff_tag, "/%s/w%d/fwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(106);
                  }

                  // AFF write bwd
                  sprintf(aff_tag, "/%s/w%d/bwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(106);
                  }

                  fini_3level_ztable ( &diagram );

                }  // end of loop on Gamma_i1

              }  // end of loop on p_i2

            }  // end of loop on Gamma_f1
#ifndef _USE_GET_CONSERVED_MOMENTUM_ID
          }  // end of loop on p_f1
#endif
          }  // end of loop on p_f2

        }  // end of loop on permutations for W diagrams

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
        free ( sink_momentum_id );
        free ( source_momentum_id );
#endif

      }  // end of loop on ptot_nmem

        if ( io_proc == 2 ) {
          if ( const char * aff_status_str = aff_writer_close (affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__ );
            EXIT(111);
          }
        }

      }  // end of loop on ptot_nclass

      fini_6level_ztable ( &w1phi );
      fini_6level_ztable ( &w3phi );

      fini_5level_ztable ( &w1xi );

    }  // end of loop on coherent source locations

    /**************************************************************************************/
    /**************************************************************************************/

    fini_6level_ztable ( &b1xi );
    fini_6level_ztable ( &b3xi );
    fini_6level_ztable ( &b7xi );
    fini_6level_ztable ( &b9xi );
    fini_6level_ztable ( &b17xi );


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
      * v3 type z_9_xi
      **************************************************************************************/
#ifndef _Z_V3_OET_MOM
      double _Complex ***** z9xi = init_5level_ztable ( gamma_f2_number, g_seq2_source_momentum_number, 4, T_global, 12 );
#else
      double _Complex ****** z9xi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f2_number, g_seq2_source_momentum_number, 4, T_global, 12 );
#endif
      if ( z9xi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "z_9_xi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);
#ifdef _Z_V3_OET_MOM
      /**************************************************************************************
       * loop on pf2
       **************************************************************************************/
      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {
        int pi2[3] = {
          g_seq_source_momentum_list[ipi2][0],
          g_seq_source_momentum_list[ipi2][1],
          g_seq_source_momentum_list[ipi2][2]
        };
#endif

      /**************************************************************************************
       * loop on gf2
       **************************************************************************************/
      for ( int igf2 = 0; igf2 < gamma_f2_number; igf2++ ) {

        /**************************************************************************************
         * loop on pf2
         **************************************************************************************/
        for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

          int pf2[3] = {
              g_seq2_source_momentum_list[ipf2][0],
              g_seq2_source_momentum_list[ipf2][1],
              g_seq2_source_momentum_list[ipf2][2] };

          /**************************************************************************************
           * loop on spin component
           **************************************************************************************/
          for ( int ispin = 0; ispin < 4; ispin++ ) {

            if ( io_proc == 2 ) {
#ifdef _Z_V3_OET_MOM
              aff_key_conversion ( aff_tag, tag, 0, pi2,           NULL, pf2, gsx, -1, gamma_f2_list[igf2], ispin );
#else
              aff_key_conversion ( aff_tag, tag, 0, zero_momentum, NULL, pf2, gsx, -1, gamma_f2_list[igf2], ispin );
#endif

              if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

              affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
#ifdef _Z_V3_OET_MOM
              exitstatus = aff_node_get_complex (affr_oet, affdir, z9xi[ipi2][igf2][ipf2][ispin][0], T_global*12);
#else
              exitstatus = aff_node_get_complex (affr_oet, affdir, z9xi[igf2][ipf2][ispin][0], T_global*12);
#endif
              if( exitstatus != 0 ) {
                fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }

            }  // end of if io proc == 2

          }  // end of loop on oet spin

        }  // end of loop on p_f2

      }  // end of loop on Gamma_f2
#ifdef _Z_V3_OET_MOM
      }  // end of loop on p_i2
#endif
      /**************************************************************************************/
      /**************************************************************************************/

  
    /**************************************************************************************
    * v3 type z_5_xi
    **************************************************************************************/
#ifndef _Z_V3_OET_MOM
      double _Complex ***** z5xi = init_5level_ztable ( gamma_f2_number, g_seq2_source_momentum_number, 4, T_global, 12 );
#else
      double _Complex ****** z5xi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f2_number, g_seq2_source_momentum_number, 4, T_global, 12 );
#endif
      if ( z5xi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "z_5_xi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);
#ifdef _Z_V3_OET_MOM
      /**************************************************************************************
       * loop on pf2
       **************************************************************************************/
      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {
        int pi2[3] = {
          g_seq_source_momentum_list[ipi2][0],
          g_seq_source_momentum_list[ipi2][1],
          g_seq_source_momentum_list[ipi2][2]
        };
#endif

      /**************************************************************************************
       * loop on gf2
       **************************************************************************************/
      for ( int igf2 = 0; igf2 < gamma_f2_number; igf2++ ) {

        /**************************************************************************************
         * loop on pf2
         **************************************************************************************/
        for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

          int pf2[3] = {
              g_seq2_source_momentum_list[ipf2][0],
              g_seq2_source_momentum_list[ipf2][1],
              g_seq2_source_momentum_list[ipf2][2] };

          /**************************************************************************************
           * loop on spin component
           **************************************************************************************/
          for ( int ispin = 0; ispin < 4; ispin++ ) {

            if ( io_proc == 2 ) {
#ifdef _Z_V3_OET_MOM
              aff_key_conversion ( aff_tag, tag, 0, pi2,           NULL, pf2, gsx, -1, gamma_f2_list[igf2], ispin );
#else
              aff_key_conversion ( aff_tag, tag, 0, zero_momentum, NULL, pf2, gsx, -1, gamma_f2_list[igf2], ispin );
#endif

              if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

              affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
#ifdef _Z_V3_OET_MOM
              exitstatus = aff_node_get_complex (affr_oet, affdir, z5xi[ipi2][igf2][ipf2][ispin][0], T_global*12);
#else
              exitstatus = aff_node_get_complex (affr_oet, affdir, z5xi[igf2][ipf2][ispin][0], T_global*12);
#endif
              if( exitstatus != 0 ) {
                fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }

            }  // end of if io proc == 2

          }  // end of loop on oet spin

        }  // end of loop on p_f2

      }  // end of loop on Gamma_f2
#ifdef _Z_V3_OET_MOM
      }  // end of loop on p_i2
#endif
      /**************************************************************************************/
      /**************************************************************************************/

    
      /**************************************************************************************
       * v3 type z_1_xi
       **************************************************************************************/
#ifndef _Z_V3_OET_MOM
      double _Complex ***** z1xi = init_5level_ztable ( gamma_f2_number, g_seq2_source_momentum_number, 4, T_global, 12 );
#else
      double _Complex ****** z1xi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f2_number, g_seq2_source_momentum_number, 4, T_global, 12 );
#endif
      if ( z1xi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "z_1_xi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);
#ifdef _Z_V3_OET_MOM
      /**************************************************************************************
       * loop on pf2
       **************************************************************************************/
      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {
        int pi2[3] = { 
          g_seq_source_momentum_list[ipi2][0],
          g_seq_source_momentum_list[ipi2][1],
          g_seq_source_momentum_list[ipi2][2]
        };
#endif

      /**************************************************************************************
       * loop on gf2
       **************************************************************************************/
      for ( int igf2 = 0; igf2 < gamma_f2_number; igf2++ ) {

        /**************************************************************************************
         * loop on pf2
         **************************************************************************************/
        for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

          int pf2[3] = {
              g_seq2_source_momentum_list[ipf2][0],
              g_seq2_source_momentum_list[ipf2][1],
              g_seq2_source_momentum_list[ipf2][2] };

          /**************************************************************************************
           * loop on spin component
           **************************************************************************************/
          for ( int ispin = 0; ispin < 4; ispin++ ) {

            if ( io_proc == 2 ) {
#ifdef _Z_V3_OET_MOM
              aff_key_conversion ( aff_tag, tag, 0, pi2,           NULL, pf2, gsx, -1, gamma_f2_list[igf2], ispin );
#else
              aff_key_conversion ( aff_tag, tag, 0, zero_momentum, NULL, pf2, gsx, -1, gamma_f2_list[igf2], ispin );
#endif

              if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);
 
              affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
#ifdef _Z_V3_OET_MOM
              exitstatus = aff_node_get_complex (affr_oet, affdir, z1xi[ipi2][igf2][ipf2][ispin][0], T_global*12);
#else
              exitstatus = aff_node_get_complex (affr_oet, affdir, z1xi[igf2][ipf2][ispin][0], T_global*12);
#endif
              if( exitstatus != 0 ) {
                fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }

            }  // end of if io proc == 2

          }  // end of loop on oet spin

        }  // end of loop on p_f2

      }  // end of loop on Gamma_f2
#ifdef _Z_V3_OET_MOM
      }  // end of loop on p_i2
#endif
      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * z_19_phi
       **************************************************************************************/
#ifndef _Z_V3_OET_MOM
      double _Complex ****** z19phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
#else
      double _Complex *****  z19phi = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
#endif
      if ( z19phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(47);
      }
      strcpy ( tag, "z_19_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

#ifndef _Z_V3_OET_MOM
      /**************************************************************************************
       * loop on pi2
       **************************************************************************************/
      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {
        int pi2[3] = {
            g_seq_source_momentum_list[ipi2][0],
            g_seq_source_momentum_list[ipi2][1],
            g_seq_source_momentum_list[ipi2][2] };
#endif

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
#ifndef _Z_V3_OET_MOM
                aff_key_conversion ( aff_tag, tag, 0, pi2,           g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], ispin );
#else
                aff_key_conversion ( aff_tag, tag, 0, zero_momentum, g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], ispin );
#endif
                if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
#ifndef _Z_V3_OET_MOM
                exitstatus = aff_node_get_complex (affr_oet, affdir, z19phi[ipi2][igf1][ipf1][ispin][0], T_global*192);
#else
                exitstatus = aff_node_get_complex (affr_oet, affdir, z19phi[igf1][ipf1][ispin][0], T_global*192);
#endif
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }

              }  // end of if io proc

            }  // end of loop on oet spin index

          }  // end of loop on sink momentum p_f1

        }  // end of loop on Gamma_f1
#ifndef _Z_V3_OET_MOM
      }  // end of loop on seq source momentum pi2
#endif

      /**************************************************************************************/
      /**************************************************************************************/



      /**************************************************************************************
       * z_17_phi
       **************************************************************************************/
#ifndef _Z_V3_OET_MOM
      double _Complex ****** z17phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
#else
      double _Complex *****  z17phi = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
#endif
      if ( z17phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(47);
      }
      strcpy ( tag, "z_17_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

#ifndef _Z_V3_OET_MOM
      /**************************************************************************************
       * loop on pi2
       **************************************************************************************/
      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {
        int pi2[3] = {
            g_seq_source_momentum_list[ipi2][0],
            g_seq_source_momentum_list[ipi2][1],
            g_seq_source_momentum_list[ipi2][2] };
#endif

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
#ifndef _Z_V3_OET_MOM
                aff_key_conversion ( aff_tag, tag, 0, pi2,           g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], ispin );
#else
                aff_key_conversion ( aff_tag, tag, 0, zero_momentum, g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], ispin );
#endif
                if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
#ifndef _Z_V3_OET_MOM
                exitstatus = aff_node_get_complex (affr_oet, affdir, z17phi[ipi2][igf1][ipf1][ispin][0], T_global*192);
#else
                exitstatus = aff_node_get_complex (affr_oet, affdir, z17phi[igf1][ipf1][ispin][0], T_global*192);
#endif
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }

              }  // end of if io proc

            }  // end of loop on oet spin index

          }  // end of loop on sink momentum p_f1

        }  // end of loop on Gamma_f1
#ifndef _Z_V3_OET_MOM
      }  // end of loop on seq source momentum pi2
#endif

      /**************************************************************************************/
      /**************************************************************************************/



      /**************************************************************************************
       * z_15_phi
       **************************************************************************************/
#ifndef _Z_V3_OET_MOM
      double _Complex ****** z15phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
#else
      double _Complex *****  z15phi = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
#endif
      if ( z15phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(47);
      }
      strcpy ( tag, "z_15_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

#ifndef _Z_V3_OET_MOM
      /**************************************************************************************
       * loop on pi2
       **************************************************************************************/
      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {
        int pi2[3] = {
            g_seq_source_momentum_list[ipi2][0],
            g_seq_source_momentum_list[ipi2][1],
            g_seq_source_momentum_list[ipi2][2] };
#endif

        /**************************************************************************************
         * loop on gf1
         **************************************************************************************/
        for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

        /************************************************************************************
         * loop on pf1
         **************************************************************************************/
          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            /**************************************************************************************
             * loop on spin components
             **************************************************************************************/
            for ( int ispin = 0; ispin < 4; ispin++ ) {

              if ( io_proc == 2 ) {
#ifndef _Z_V3_OET_MOM
                aff_key_conversion ( aff_tag, tag, 0, pi2,           g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], ispin );
#else
                aff_key_conversion ( aff_tag, tag, 0, zero_momentum, g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], ispin );
#endif
                if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
#ifndef _Z_V3_OET_MOM
                exitstatus = aff_node_get_complex (affr_oet, affdir, z15phi[ipi2][igf1][ipf1][ispin][0], T_global*192);
#else
                exitstatus = aff_node_get_complex (affr_oet, affdir, z15phi[igf1][ipf1][ispin][0], T_global*192);
#endif
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }

              }  // end of if io proc

            }  // end of loop on oet spin index

          }  // end of loop on sink momentum p_f1

        }  // end of loop on Gamma_f1
#ifndef _Z_V3_OET_MOM
      }  // end of loop on seq source momentum pi2
#endif

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * z_9_phi
       **************************************************************************************/
#ifndef _Z_V3_OET_MOM
      double _Complex ****** z9phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
#else
      double _Complex *****  z9phi = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
#endif
      if ( z9phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(47);
      }
      strcpy ( tag, "z_9_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

#ifndef _Z_V3_OET_MOM
      /**************************************************************************************
       * loop on pi2
       **************************************************************************************/
      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {
        int pi2[3] = {
            g_seq_source_momentum_list[ipi2][0],
            g_seq_source_momentum_list[ipi2][1],
            g_seq_source_momentum_list[ipi2][2] };
#endif

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
#ifndef _Z_V3_OET_MOM
                aff_key_conversion ( aff_tag, tag, 0, pi2,           g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], ispin );
#else
                aff_key_conversion ( aff_tag, tag, 0, zero_momentum, g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], ispin );
#endif
                if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
#ifndef _Z_V3_OET_MOM
                exitstatus = aff_node_get_complex (affr_oet, affdir, z9phi[ipi2][igf1][ipf1][ispin][0], T_global*192);
#else
                exitstatus = aff_node_get_complex (affr_oet, affdir, z9phi[igf1][ipf1][ispin][0], T_global*192);
#endif
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }

              }  // end of if io proc

            }  // end of loop on oet spin index

          }  // end of loop on sink momentum p_f1

        }  // end of loop on Gamma_f1
#ifndef _Z_V3_OET_MOM
      }  // end of loop on seq source momentum pi2
#endif

      /**************************************************************************************/
      /**************************************************************************************/





      /**************************************************************************************
       * v2 type z_3_phi
       **************************************************************************************/
#ifndef _Z_V3_OET_MOM
      double _Complex ****** z3phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
#else
      double _Complex *****  z3phi = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
#endif
      if ( z3phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }
      strcpy ( tag, "z_3_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

#ifndef _Z_V3_OET_MOM
      /**************************************************************************************
       * loop on pi2
       **************************************************************************************/
      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {
        int pi2[3] = {
            g_seq_source_momentum_list[ipi2][0],
            g_seq_source_momentum_list[ipi2][1],
            g_seq_source_momentum_list[ipi2][2] };
#endif

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
#ifndef _Z_V3_OET_MOM
                aff_key_conversion ( aff_tag, tag, 0, pi2,           g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], ispin );
#else
                aff_key_conversion ( aff_tag, tag, 0, zero_momentum, g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], ispin );
#endif
                if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
#ifndef _Z_V3_OET_MOM
                exitstatus = aff_node_get_complex (affr_oet, affdir, z3phi[ipi2][igf1][ipf1][ispin][0], T_global*192);
#else
                exitstatus = aff_node_get_complex (affr_oet, affdir, z3phi[igf1][ipf1][ispin][0], T_global*192);
#endif
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }
                
              }  // end of if io proc == 2

            }  // end of loop on oet spin index */

          }  // end of loop on sink momentum p_f1 */

        }  // end of loop on Gamma_f1 */
#ifndef _Z_V3_OET_MOM
      }  // end of loop on seq source momentum pi2 */
#endif

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * z_1_phi
       **************************************************************************************/
#ifndef _Z_V3_OET_MOM
      double _Complex ****** z1phi = init_6level_ztable ( g_seq_source_momentum_number, gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
#else
      double _Complex *****  z1phi = init_5level_ztable ( gamma_f1_nucleon_number, g_sink_momentum_number, 4, T_global, 192 );
#endif
      if ( z1phi == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(47);
      }
      strcpy ( tag, "z_1_phi" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

#ifndef _Z_V3_OET_MOM
      /**************************************************************************************
       * loop on pi2
       **************************************************************************************/
      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {
        int pi2[3] = {
            g_seq_source_momentum_list[ipi2][0],
            g_seq_source_momentum_list[ipi2][1],
            g_seq_source_momentum_list[ipi2][2] };
#endif

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
#ifndef _Z_V3_OET_MOM
                aff_key_conversion ( aff_tag, tag, 0, pi2,           g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], ispin );
#else
                aff_key_conversion ( aff_tag, tag, 0, zero_momentum, g_sink_momentum_list[ipf1], NULL, gsx, -1, gamma_f1_nucleon_list[igf1][0], ispin );
#endif
                if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
#ifndef _Z_V3_OET_MOM
                exitstatus = aff_node_get_complex (affr_oet, affdir, z1phi[ipi2][igf1][ipf1][ispin][0], T_global*192);
#else
                exitstatus = aff_node_get_complex (affr_oet, affdir, z1phi[igf1][ipf1][ispin][0], T_global*192);
#endif
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }
                
              }  // end of if io proc

            }  // end of loop on oet spin index

          }  // end of loop on sink momentum p_f1

        }  // end of loop on Gamma_f1
#ifndef _Z_V3_OET_MOM
      }  // end of loop on seq source momentum pi2
#endif

      /**************************************************************************************/
      /**************************************************************************************/

      char name[20];
      
      gamma_matrix_type gamma_5;
      /* gamma_matrix_set ( &gamma_5, 5, 1 ); */
      /* TEST PLEGMA COMPARISON */
      gamma_matrix_set ( &gamma_5, 5, 1 );

      gamma_matrix_type goet;
      gamma_matrix_set ( &goet, gamma_f2_list[0], gamma_f2_sign[0] );
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
      for ( int iclass = 0; iclass  < ptot_nclass; iclass++ ) {

        int pref[3] = { ptot_class[iclass][0][0], ptot_class[iclass][0][1], ptot_class[iclass][0][2] };

        /**************************************************************************************
         * AFF output file
         **************************************************************************************/
        if ( io_proc == 2 ) {
          sprintf( filename, "%s.z.PX%dPY%dPZ%d.%.4d.t%dx%dy%dz%d.aff", "piN_piN_diagrams",
              pref[0], pref[1], pref[2], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );

          affw = aff_writer (filename);
          if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
            EXIT(4);
          } else {
            fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
          }
        }

      for ( int imem = 0; imem < ptot_nmem[iclass]; imem++ ) {

        int ptot[3]  = { ptot_class[iclass][imem][0], ptot_class[iclass][imem][1], ptot_class[iclass][imem][2] };

        int mptot[3] = { -ptot[0], -ptot[1], -ptot[2] };
#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
        int * sink_momentum_id =   get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, ptot,  g_sink_momentum_list, g_sink_momentum_number );
        if ( sink_momentum_id == NULL ) {
          // fprintf(stderr, "[piN2piN_diagrams_complete] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
          // EXIT(1);
          fprintf(stdout, "# [piN2piN_diagrams_complete] sink_momentum_id empty; continue %s %d\n", __FILE__, __LINE__ );
          continue;
        }

        int * source_momentum_id = get_conserved_momentum_id ( g_seq_source_momentum_list,  g_seq_source_momentum_number,  mptot, g_sink_momentum_list, g_sink_momentum_number );
        if ( source_momentum_id == NULL ) {
          // fprintf(stderr, "[piN2piN_diagrams_complete] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
          // EXIT(1);
          fprintf(stdout, "# [piN2piN_diagrams_complete] source_momentum_id empty; continue %s %d\n", __FILE__, __LINE__ );
          continue;
        }
#endif
        /**************************************************************************************
         * Z diagrams 1, 2
         **************************************************************************************/
        int const perm[20][4] = { { 0, 2, 1, 3 }, //Z1
				  { 0, 2, 3, 1 }, //Z2
	                          { 2, 0, 3, 1 }, //Z3
                                  { 2, 0, 1, 3 }, //Z4
                                  { 2, 0, 1, 3 }, //Z5
				  { 0, 2, 1, 3 }, //Z6
                                  { 2, 0, 3, 1 }, //Z7
                                  { 0, 2, 3, 1 }, //Z8
                                  { 1, 0, 3, 2 }, //Z9
                                  { 3, 0, 1, 2 }, //Z10
				  { 1, 0, 2, 3 }, //Z11
                                  { 0, 1, 3, 2 }, //Z12
                                  { 3, 0, 2, 1 }, //Z13
                                  { 0, 2, 3, 1 }, //Z14
                                  { 3, 0, 1, 2 }, //Z15
                                  { 1, 0, 3, 2 }, //Z16
                                  { 1, 0, 2, 3 }, //Z17
                                  { 3, 0, 2, 1 }, //Z18 
                                  { 0, 1, 2, 3 }, //Z19
                                  { 0, 3, 2, 1 }}; //Z20
        for ( int iperm = 0; iperm < 20; iperm++ ) {

#ifndef _Z_V3_OET_MOM
          double _Complex ***** zxi = NULL, ****** zphi = NULL; 
#else
          double _Complex ****** zxi = NULL, ***** zphi = NULL; 
#endif

          if ( ( iperm == 0 ) || ( iperm == 1 ) ) {
            zxi  = z1xi;
            zphi = z1phi;
          } else if ( ( iperm == 2 ) || ( iperm == 3 ) ) {
            zxi  = z1xi;
            zphi = z3phi;
          } else if ( iperm == 4 ) {
            zxi = z5xi;
            zphi= z3phi;
          } else if ( iperm == 5) {
            zxi = z5xi; 
            zphi= z1phi;
          } else if ( iperm == 6) {
            zxi = z5xi;
            zphi= z3phi;
          } else if ( iperm == 7) {
            zxi = z5xi;
            zphi= z1phi;
          } else if ( iperm == 8) {
            zxi = z9xi;
            zphi= z9phi;
          } else if ( iperm == 9) {
            zxi = z9xi;
            zphi= z9phi;
          } else if ( iperm == 10) {
            zxi = z9xi;
            zphi= z3phi;
          } else if ( iperm == 11) {
            zxi = z9xi;
            zphi= z1phi;
          } else if ( iperm == 12) {
            zxi = z9xi;
            zphi= z3phi;
          } else if ( iperm == 13) {
            zxi = z9xi;
            zphi= z1phi;
          } else if ( iperm == 14) {
            zxi = z1xi;
            zphi= z15phi;
          } else if ( iperm == 15) {
            zxi = z1xi;
            zphi= z15phi;
          } else if ( iperm == 16) {
            zxi = z1xi;
            zphi= z17phi;
          } else if ( iperm == 17) {
            zxi = z1xi;
            zphi= z17phi;
          } else if ( iperm == 18) {
            zxi = z1xi;
            zphi= z19phi;
          } else if ( iperm == 19) {
            zxi = z1xi;
            zphi= z19phi;
          }

#if 0
          /**************************************************************************************
           * AFF output file
           **************************************************************************************/
          if ( io_proc == 2 ) {
            sprintf( filename, "%s.Z%d.%.4d.PX%dPY%dPZ%d.t%dx%dy%dz%d.aff", "piN_piN_diagrams", iperm+1, Nconf, 
                g_total_momentum_list[iptot][0], g_total_momentum_list[iptot][1], g_total_momentum_list[iptot][2],
                gsx[0], gsx[1], gsx[2], gsx[3] );
            affw = aff_writer (filename);
            if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
              EXIT(4);
            } else {
              fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
            }
          }  // end of if io_proc == 2
#endif  // of if 0

          /**************************************************************************************
           * loop on pf2
           **************************************************************************************/
          for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {
            int pf2[3] = { g_seq2_source_momentum_list[ipf2][0], g_seq2_source_momentum_list[ipf2][1], g_seq2_source_momentum_list[ipf2][2] };

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
            int ipf1 = sink_momentum_id[ipf2];
            if ( ipf1 == -1 ) {
              if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagrams_complete] Warning, skipping seq2 momentum no %d  (  %3d, %3d, %3d )\n", ipf2, 
                  g_seq2_source_momentum_list[ipf2][0], g_seq2_source_momentum_list[ipf2][1], g_seq2_source_momentum_list[ipf2][2] );
              continue;
            }
#else
          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {
#endif
            int pf1[3] = {
              g_sink_momentum_list[ipf1][0],
              g_sink_momentum_list[ipf1][1],
              g_sink_momentum_list[ipf1][2]
            };


            /**************************************************************************************
             * loop on gf1
             **************************************************************************************/
            for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {
              int igf2 = 0;

              gamma_matrix_type gf12;
              gamma_matrix_set  ( &gf12, gamma_f1_nucleon_list[igf1][1], gamma_f1_nucleon_sign[igf1][1] );

              /**************************************************************************************
               * loop on pi2
               **************************************************************************************/
              for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

                int pi2[3] = { g_seq_source_momentum_list[ipi2][0], g_seq_source_momentum_list[ipi2][1], g_seq_source_momentum_list[ipi2][2] };

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
                int ipi1 = source_momentum_id[ipi2];
                if ( ipi1 == -1 ) {
                  if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagrams_complete] Warning, skipping seq momentum no %d  (  %3d, %3d, %3d ) for ptot ( %3d, %3d, %3d )\n", ipi2, 
                      g_seq_source_momentum_list[ipi2][0], g_seq_source_momentum_list[ipi2][1], g_seq_source_momentum_list[ipi2][2],
                      ptot[0], ptot[1], ptot[2] );
                  continue;
                }
                int pi1[3] = {
                    g_sink_momentum_list[ipi1][0],
                    g_sink_momentum_list[ipi1][1],
                    g_sink_momentum_list[ipi1][2]
                };
#else
                int pi1[3] = {
                  - ( pf1[0] + pf2[0] + pi2[0] ),
                  - ( pf1[1] + pf2[1] + pi2[1] ),
                  - ( pf1[2] + pf2[2] + pi2[2] )
                };
#endif

                /**************************************************************************************
                 * loop on gi1
                 **************************************************************************************/
                for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) {
                  int igi2 = 0;

                  char aff_tag_suffix[400];

                  contract_diagram_key_suffix ( aff_tag_suffix,
                      gamma_f2_list[igf2],                                            pf2,
                      gamma_f1_nucleon_list[igf1][0], gamma_f1_nucleon_list[igf1][1], pf1,
                      gamma_i2_list[igi2],                                            pi2,
                      gamma_f1_nucleon_list[igi1][0], gamma_f1_nucleon_list[igi1][1], pi1,
                      gsx );

                  /**************************************************************************************/
                  /**************************************************************************************/

                  gamma_matrix_type gi11, gi12;
                  gamma_matrix_set  ( &gi11, gamma_f1_nucleon_list[igi1][0], gamma_f1_nucleon_sign[igi1][0] );
                  gamma_matrix_set  ( &gi12, gamma_f1_nucleon_list[igi1][1], gamma_f1_nucleon_sign[igi1][1] );

                  // no transposition with above choice of permutation, cf. pdf
                  // gamma_matrix_transposed ( &gi11, &gi11 );
                  if ( g_verbose > 2 ) {
                    sprintf(name, "G%.2d", gi11.id);
                    gamma_matrix_printf (&gi11, name, stdout);
                  }

                  /**************************************************************************************/
                  /**************************************************************************************/

                  double _Complex *** diagram = init_3level_ztable ( T_global, 4, 4 );
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
                    EXIT(47);
                  }
#ifndef _Z_V3_OET_MOM
                  exitstatus = contract_diagram_sample_oet (diagram, zxi[0][ipf2],  zphi[ipi2][igf1][ipf1], goet, perm[iperm], gi11, T_global );
#else
                  exitstatus = contract_diagram_sample_oet (diagram, zxi[ipi2][0][ipf2],  zphi[igf1][ipf1], goet, perm[iperm], gi11, T_global );
#endif

                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_sample_oet, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(108);
                  }

                  // add boundary phase
#ifdef _ADD_BOUNDARY_PHASE
                  if ( ( exitstatus = correlator_add_baryon_boundary_phase ( diagram, gsx[0], +1, T_global ) ) != 0 ) {
                    fprintf( stderr, "[piN2piN_diagrams_complete] Error from correlator_add_baryon_boundary_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(106);
                  }
#endif
                  // add source phase
                  if ( ( exitstatus = correlator_add_source_phase ( diagram, pi1, &(gsx[1]), T_global ) ) != 0 ) {
                    fprintf( stderr, "[piN2piN_diagrams_complete] Error from correlator_add_source_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(107);
                  }
  
                  // add outer gamma matrices
                  if ( ( exitstatus =  contract_diagram_zm4x4_field_mul_gamma_lr ( diagram, diagram, gf12, gi12, T_global ) ) != 0 ) {
                    fprintf( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_mul_gamma_lr, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(108);
                  }
  
                  // add phase from phase convention
                  double _Complex const zsign = contract_diagram_get_correlator_phase ( "mxb-mxb",
                      gamma_f1_nucleon_list[igi1][0],
                      gamma_f1_nucleon_list[igi1][1],
                      gamma_i2_list[igi2],
                      gamma_f1_nucleon_list[igf1][0],
                      gamma_f1_nucleon_list[igf1][1],
                      gamma_f2_list[igf2] );
  
                  if ( ( exitstatus = contract_diagram_zm4x4_field_ti_eq_co ( diagram, zsign, T_global ) ) != 0 ) {
                    fprintf ( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_ti_eq_co, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                    EXIT(109)
                  }


                  // AFF write fwd
                  sprintf(aff_tag, "/%s/z%d/fwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(109);
                  }

                  // AFF write bwd
                  sprintf(aff_tag, "/%s/z%d/bwd/%s", "pixN-pixN", iperm+1, aff_tag_suffix );
                  if ( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(110);
                  }

                  fini_3level_ztable ( &diagram );

                }  // end of loop on Gamma_i1

              }  // end of loop on p_i2

            }  // end of loop on Gamma_f1
#ifndef _USE_GET_CONSERVED_MOMENTUM_ID
          }  // end of loop on p_f1
#endif
          }  // end of loop on p_f2

        }  // end of loop on permutations

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
        free ( sink_momentum_id );
        free ( source_momentum_id );
#endif

      }  // end of loop on ptot_nmem

        if ( io_proc == 2 ) {
          if ( const char * aff_status_str = aff_writer_close (affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__ );
            EXIT(111);
          }
        }  // end of if io_proc == 2

      }  // end of loop on ptot_nclass

      /**************************************************************************************/
      /**************************************************************************************/
#ifndef _Z_V3_OET_MOM
      fini_6level_ztable ( &z1phi );
      fini_6level_ztable ( &z3phi );
      fini_5level_ztable ( &z1xi );
#else
      fini_5level_ztable ( &z1phi );
      fini_5level_ztable ( &z3phi );
      fini_6level_ztable ( &z1xi );
#endif

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
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      /**************************************************************************************
       * bb
       **************************************************************************************/
      double _Complex ****** bb = init_6level_ztable ( gamma_f1_nucleon_number, gamma_f1_nucleon_number, g_sink_momentum_number, 2, T_global, 16 );
      if ( bb == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }
      strcpy ( tag, "N-N" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

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

                aff_key_conversion_diagram ( aff_tag, tag, NULL, NULL, g_sink_momentum_list[ipf1], NULL, gamma_f1_nucleon_list[igi1][0], -1, gamma_f1_nucleon_list[igf1][0], -1, gsx, "n", idiag+1 , 0 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, bb[igi1][igf1][ipf1][idiag][0], T_global*16);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__ );
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
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }
      strcpy ( tag, "m-m" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

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

                aff_key_conversion_diagram (  aff_tag, tag, NULL, g_seq_source_momentum_list[ipi2], NULL, g_sink_momentum_list[ipf2], -1, gamma_i2_list[igi2], -1, gamma_f2_list[igf2], gsx, NULL, 0 , 0 );

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
                exitstatus = aff_node_get_complex (affr_oet, affdir, mm[ipi2][igi2][igf2][ipf2], T_global);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d\n", aff_tag, exitstatus);
                  EXIT(105);
                }
                if ( g_verbose > 4 ) {
                  for ( int it = 0; it < T_global; it++ ) {
                    fprintf(stdout, "# [piN2piN_diagrams_complete] m-m %3d %25.16e %25.16e\n", it, 
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
       * loop on total momentum classes and members
       **************************************************************************************/
      for ( int iclass = 0; iclass  < ptot_nclass; iclass++ ) {

        int pref[3] = { ptot_class[iclass][0][0], ptot_class[iclass][0][1], ptot_class[iclass][0][2] };

        /**************************************************************************************
         * AFF output file
         **************************************************************************************/
        if ( io_proc == 2 ) {
          sprintf( filename, "%s.s.PX%dPY%dPZ%d.%.4d.t%dx%dy%dz%d.aff", "piN_piN_diagrams",
              pref[0], pref[1], pref[2], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
          affw = aff_writer (filename);
          if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
            EXIT(4);
          } else {
            fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
          }
        }  // end of if io_proc == 2

      for ( int imem = 0; imem < ptot_nmem[iclass]; imem++ ) {

        int ptot[3]  = { ptot_class[iclass][imem][0], ptot_class[iclass][imem][1], ptot_class[iclass][imem][2] };

        int mptot[3] = { -ptot[0], -ptot[1], -ptot[2] };

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
        int * sink_momentum_id =   get_conserved_momentum_id ( g_seq2_source_momentum_list, g_seq2_source_momentum_number, ptot,  g_sink_momentum_list, g_sink_momentum_number );
        if ( sink_momentum_id == NULL ) {
          // fprintf(stderr, "[piN2piN_diagrams_complete] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
          // EXIT(1);
          fprintf(stdout, "# [piN2piN_diagrams_complete] sink_momentum_id empty; continue %s %d\n", __FILE__, __LINE__ );
          continue;
        }

        int * source_momentum_id = get_conserved_momentum_id ( g_seq_source_momentum_list,  g_seq_source_momentum_number,  mptot, g_sink_momentum_list, g_sink_momentum_number );
        if ( source_momentum_id == NULL ) {
          // fprintf(stderr, "[piN2piN_diagrams_complete] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
          // EXIT(1);
          fprintf(stdout, "# [piN2piN_diagrams_complete] source_momentum_id empty; continue %s %d\n", __FILE__, __LINE__ );
          continue;
        }
#endif

        /**************************************************************************************
         * loop on diagrams
         **************************************************************************************/
        for ( int idiag = 0; idiag < 2; idiag++ ) {

#if 0
          /**************************************************************************************
           * AFF output file
           **************************************************************************************/
          if ( io_proc == 2 ) {
            sprintf( filename, "%s.S%d.%.4d.PX%dPY%dPZ%d.t%dx%dy%dz%d.aff", "piN_piN_diagrams", idiag+1, Nconf, 
                g_total_momentum_list[iptot][0], g_total_momentum_list[iptot][1], g_total_momentum_list[iptot][2],
                gsx[0], gsx[1], gsx[2], gsx[3] );
            affw = aff_writer (filename);
            if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
              EXIT(4);
            } else {
              fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
            }
          }  // end of if io_proc == 2
#endif  // of if 0

          /**************************************************************************************
           * loop on pf2
           **************************************************************************************/
          for ( int ipf2 = 0; ipf2 < g_seq2_source_momentum_number; ipf2++ ) {

            int pf2[3] = { g_seq2_source_momentum_list[ipf2][0], g_seq2_source_momentum_list[ipf2][1], g_seq2_source_momentum_list[ipf2][2] };

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
            int ipf1 = sink_momentum_id[ipf2];
            if ( ipf1 == -1 ) {
              if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagrams_complete] Warning, skipping seq2 momentum no %d  (  %3d, %3d, %3d )\n", ipf2,
                  g_seq2_source_momentum_list[ipf2][0], g_seq2_source_momentum_list[ipf2][1], g_seq2_source_momentum_list[ipf2][2] );
              continue;
            }
#else
          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {
#endif

            int pf1[3] = {
              g_sink_momentum_list[ipf1][0],
              g_sink_momentum_list[ipf1][1],
              g_sink_momentum_list[ipf1][2]
            };

            /**************************************************************************************
             * loop on gf1
             **************************************************************************************/
            for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {
              int igf2 = 0;

              gamma_matrix_type gf12;
              gamma_matrix_set  ( &gf12, gamma_f1_nucleon_list[igf1][1], gamma_f1_nucleon_sign[igf1][1] );


              /**************************************************************************************
               * loop on pi2
               **************************************************************************************/
              for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

                int pi2[3] = { g_seq_source_momentum_list[ipi2][0], g_seq_source_momentum_list[ipi2][1], g_seq_source_momentum_list[ipi2][2] };

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
                int ipi1 = source_momentum_id[ipi2];
                if ( ipi1 == -1 ) {
                  if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagrams_complete] Warning, skipping seq momentum no %d  (  %3d, %3d, %3d ) for ptot ( %3d, %3d, %3d )\n", ipi2,
                      g_seq_source_momentum_list[ipi2][0], g_seq_source_momentum_list[ipi2][1], g_seq_source_momentum_list[ipi2][2],
                      ptot[0], ptot[1], ptot[2] );
                  continue;
                }
                int pi1[3] = {
                    g_sink_momentum_list[ipi1][0],
                    g_sink_momentum_list[ipi1][1],
                    g_sink_momentum_list[ipi1][2]
                };
#else
                int pi1[3] = {
                  - ( pf1[0] + pf2[0] + pi2[0] ),
                  - ( pf1[1] + pf2[1] + pi2[1] ),
                  - ( pf1[2] + pf2[2] + pi2[2] )
                };
#endif

                /**************************************************************************************
                 * loop on gi1
                 **************************************************************************************/
                for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) { 
                  int igi2 = 0;

                  gamma_matrix_type gi12;
                  gamma_matrix_set  ( &gi12, gamma_f1_nucleon_list[igi1][1], gamma_f1_nucleon_sign[igi1][1] );

                  char aff_tag_suffix[400];

                  contract_diagram_key_suffix ( aff_tag_suffix,
                      gamma_f2_list[igf2],                                            pf2,
                      gamma_f1_nucleon_list[igf1][0], gamma_f1_nucleon_list[igf1][1], pf1,
                      gamma_i2_list[igi2],                                            pi2,
                      gamma_f1_nucleon_list[igi1][0], gamma_f1_nucleon_list[igi1][1], pi1,
                      gsx );


                  /**************************************************************************************/
                  /**************************************************************************************/

                  double _Complex *** diagram = init_3level_ztable ( T_global, 4, 4 );
                  if ( diagram == NULL ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
                    EXIT(104);
                  }

                  memcpy( bb_aux[0][0], bb[igi1][igf1][ipf1][idiag][0], 16*T_global*sizeof(double _Complex) );

                  // multiply baryon 2-point function with meson 2-point function */
                  exitstatus = contract_diagram_zm4x4_field_ti_co_field ( diagram, bb_aux,  mm[ipi2][igi2][igf2][ipf2], T_global );
                  // memcpy(diagram[0][0],  bb_aux[0][0], 16*T_global*sizeof(double _Complex) );

                  // transpose
                  if ( ( exitstatus = contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( diagram, diagram, T_global ) ) != 0 ) {
                    fprintf( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_eq_zm4x4_field_transposed, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  // add boundary phase
#ifdef _ADD_BOUNDARY_PHASE
                  if ( ( exitstatus = correlator_add_baryon_boundary_phase ( diagram, gsx[0], +1, T_global ) ) != 0 ) {
                    fprintf( stderr, "[piN2piN_diagrams_complete] Error from correlator_add_baryon_boundary_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(106);
                  }
#endif
                  // add source phase
                  if ( ( exitstatus = correlator_add_source_phase ( diagram, pi1, &(gsx[1]), T_global ) ) != 0 ) {
                    fprintf( stderr, "[piN2piN_diagrams_complete] Error from correlator_add_source_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(107);
                  }
  
                  // add outer gamma matrices
                  if ( ( exitstatus =  contract_diagram_zm4x4_field_mul_gamma_lr ( diagram, diagram, gf12, gi12, T_global ) ) != 0 ) {
                    fprintf( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_mul_gamma_lr, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(108);
                  }
  
                  // add phase from phase convention
                  
                  /**************************************************************************************
                   **************************************************************************************
                   **
                   ** NOTE: extra minus sign, since meson loop is - trace ( ... )
                   ** 
                   ** that -1 is NOT given in -contract_diagram_get_correlator_phase
                   ** and has not been included in piN2piN_factorized's contract_vn_momentum_projection
                   **
                   **************************************************************************************
                   **************************************************************************************/
                  double _Complex const zsign = -contract_diagram_get_correlator_phase ( "mxb-mxb",
                      gamma_f1_nucleon_list[igi1][0],
                      gamma_f1_nucleon_list[igi1][1],
                      gamma_i2_list[igi2],
                      gamma_f1_nucleon_list[igf1][0],
                      gamma_f1_nucleon_list[igf1][1],
                      gamma_f2_list[igf2] );
  
                  if ( ( exitstatus = contract_diagram_zm4x4_field_ti_eq_co ( diagram, zsign, T_global ) ) != 0 ) {
                    fprintf ( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_ti_eq_co, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                    EXIT(109)
                  }

                  // AFF write fwd
                  sprintf(aff_tag, "/%s/s%d/fwd/%s", "pixN-pixN", idiag+1, aff_tag_suffix );
                  if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] aff tag = %s\n", aff_tag);
                  if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d\n", exitstatus);
                    EXIT(105);
                  }
   
                  // AFF write bwd
                  sprintf(aff_tag, "/%s/s%d/bwd/%s", "pixN-pixN", idiag+1, aff_tag_suffix );
                  if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] aff tag = %s\n", aff_tag);
                  if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d\n", exitstatus);
                    EXIT(105);
                  }

                  fini_3level_ztable ( &diagram );

                }  // end of loop on Gamma_i1

              }  // end of loop on p_i2

            }  // end of loop on Gamma_f1

#ifndef _USE_GET_CONSERVED_MOMENTUM_ID
          }  // end of loop on p_f1
#endif
          }  // end of loop on p_f2

        }  // end of loop on diagrams

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
        free ( sink_momentum_id );
        free ( source_momentum_id );
#endif

      }  // end of loop on ptot_nmem

        if ( io_proc == 2 ) {
          if ( const char * aff_status_str = aff_writer_close (affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__ );
            EXIT(111);
          }
        }  // end of if io_proc == 2

      }  // end of loop on ptot_nclass

      /**************************************************************************************/
      /**************************************************************************************/

      fini_6level_ztable ( &bb );
      fini_5level_ztable ( &mm );
      fini_3level_ztable ( &bb_aux );

    }  // end of loop on coherent source locations

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * 2-point functions
     *
     * /piN-D/t09x06y07z00/pi2x00pi2y00pi2z01/gi02/gf04/t2/px00py00pz00
     * /D-D/t09x06y07z00/gi15/gf07/d1/px00py-01pz00
     * /N-N/t01x02y03z04/gi08/gf14/n2/px00py-01pz00
     **************************************************************************************/

    // loop on coherent source locations
    for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {
      int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

      int source_proc_id, sx[4];
      int const gsx[4] = { t_coherent, 
          ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global,
          ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global,
          ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global };

      get_point_source_info (gsx, sx, &source_proc_id);

      /**************************************************************************************
       * bb
       **************************************************************************************/
      double _Complex ****** bb = init_6level_ztable ( gamma_f1_nucleon_number, gamma_f1_nucleon_number, g_sink_momentum_number, 2, T_global, 16 );
      if ( bb == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * N - N
       **************************************************************************************/

      strcpy ( tag, "N-N" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

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

                aff_key_conversion_diagram ( aff_tag, tag, NULL, NULL, g_sink_momentum_list[ipf1], NULL, gamma_f1_nucleon_list[igi1][0], -1, gamma_f1_nucleon_list[igf1][0], -1, gsx, "n", idiag+1 , 0);

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, bb[igi1][igf1][ipf1][idiag][0], T_global*16);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__ );
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
       * loop on total momenta
       **************************************************************************************/
      for ( int iclass = 0; iclass  < ptot_nclass; iclass++ ) {

        int pref[3] = { ptot_class[iclass][0][0], ptot_class[iclass][0][1], ptot_class[iclass][0][2] };

        /**************************************************************************************
         * AFF output file
         **************************************************************************************/
        if ( io_proc == 2 ) {
          sprintf( filename, "%s.n.PX%dPY%dPZ%d.%.4d.t%dx%dy%dz%d.aff", "piN_piN_diagrams",
              pref[0], pref[1], pref[2], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
          affw = aff_writer (filename);
          if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
            EXIT(4);
          } else {
            fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
          }
        }  // end of if io_proc == 2


      for ( int imem = 0; imem < ptot_nmem[iclass]; imem++ ) {

        int ptot[3]  = { ptot_class[iclass][imem][0], ptot_class[iclass][imem][1], ptot_class[iclass][imem][2] };

        int mptot[3] = { -ptot[0], -ptot[1], -ptot[2] };

        int iptot =  get_momentum_id ( ptot, g_sink_momentum_list, g_sink_momentum_number );
        if ( iptot == -1 ) continue;


        /**************************************************************************************
         * loop on diagrams
         **************************************************************************************/
        for ( int idiag = 0; idiag < 2; idiag++ ) {

#if 0
          /**************************************************************************************
           * AFF output file
           **************************************************************************************/
          if ( io_proc == 2 ) {
            sprintf( filename, "%s.N%d.%.4d.PX%dPY%dPZ%d.t%dx%dy%dz%d.aff", "piN_piN_diagrams", idiag+1, Nconf, 
                ptot[0], ptot[1], ptot[2], gsx[0], gsx[1], gsx[2], gsx[3] );
            affw = aff_writer (filename);
            if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
              EXIT(4);
            } else {
              fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
            }
          }  // end of if io_proc == 2
#endif  // of if 0

          /**************************************************************************************
           * loop on gf1
           **************************************************************************************/
          for ( int igf1 = 0; igf1 < gamma_f1_nucleon_number; igf1++ ) {

            gamma_matrix_type gf12;
            gamma_matrix_set  ( &gf12, gamma_f1_nucleon_list[igf1][1], gamma_f1_nucleon_sign[igf1][1] );

            /**************************************************************************************
             * loop on gi1
             **************************************************************************************/
            for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) { 

              gamma_matrix_type gi12;
              gamma_matrix_set  ( &gi12, gamma_f1_nucleon_list[igi1][1], gamma_f1_nucleon_sign[igi1][1] );

              char aff_tag_suffix[400];

              contract_diagram_key_suffix ( aff_tag_suffix,
                  -1, NULL,
                  gamma_f1_nucleon_list[igf1][0], gamma_f1_nucleon_list[igf1][1], ptot,
                  -1, NULL,
                  gamma_f1_nucleon_list[igi1][0], gamma_f1_nucleon_list[igi1][1], mptot,
                  gsx );

              /**************************************************************************************/
              /**************************************************************************************/

              double _Complex *** diagram = init_3level_ztable ( T_global, 4, 4 );
              if ( diagram == NULL ) {
                fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
                EXIT(101);
              }

              // copy current bb configuration to T x 4x4 diagram
              memcpy( diagram[0][0], bb[igi1][igf1][iptot][idiag][0], 16*T_global*sizeof(double _Complex) );

              // transpose
              if ( ( exitstatus = contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( diagram, diagram, T_global ) ) != 0 ) {
                fprintf( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_eq_zm4x4_field_transposed, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(102);
              }

              // add boundary phase
#ifdef _ADD_BOUNDARY_PHASE
              if ( ( exitstatus = correlator_add_baryon_boundary_phase ( diagram, gsx[0], +1, T_global ) ) != 0 ) {
                fprintf( stderr, "[piN2piN_diagrams_complete] Error from correlator_add_baryon_boundary_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(103);
              }
#endif
              // add source phase
              if ( ( exitstatus = correlator_add_source_phase ( diagram, mptot, &(gsx[1]), T_global ) ) != 0 ) {
                fprintf( stderr, "[piN2piN_diagrams_complete] Error from correlator_add_source_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(104);
              }
  
              // add outer gamma matrices
              if ( ( exitstatus =  contract_diagram_zm4x4_field_mul_gamma_lr ( diagram, diagram, gf12, gi12, T_global ) ) != 0 ) {
                fprintf( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_mul_gamma_lr, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }
  
              // add phase from phase convention
              double _Complex const zsign = contract_diagram_get_correlator_phase ( "b-b",
                  gamma_f1_nucleon_list[igi1][0], gamma_f1_nucleon_list[igi1][1], -1,
                  gamma_f1_nucleon_list[igf1][0], gamma_f1_nucleon_list[igf1][1], -1 );
  
              if ( ( exitstatus = contract_diagram_zm4x4_field_ti_eq_co ( diagram, zsign, T_global ) ) != 0 ) {
                fprintf ( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_ti_eq_co, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(106)
              }

              // AFF write fwd
              sprintf(aff_tag, "/%s/n%d/fwd/%s", "N-N", idiag+1, aff_tag_suffix );
              if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] aff tag = %s\n", aff_tag);
              if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(107);
              }
   
              // AFF write bwd
              sprintf(aff_tag, "/%s/n%d/bwd/%s", "N-N", idiag+1, aff_tag_suffix );
              if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] aff tag = %s\n", aff_tag);
              if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(108);
              }

              fini_3level_ztable ( &diagram );


            }  // end of loop on Gamma_i1

          }  // end of loop on Gamma_f1

        }  // end of loop on diagrams

      }  // end of loop on ptot_nmem

        if ( io_proc == 2 ) {
          if ( const char * aff_status_str = aff_writer_close (affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__ );
            EXIT(111);
          }
        }  // end of if io_proc == 2

      }  // end of loop on ptot_nclass

      fini_6level_ztable ( &bb );

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * D - D
       **************************************************************************************/

      bb = init_6level_ztable ( gamma_f1_delta_number, gamma_f1_delta_number, g_sink_momentum_number, 6, T_global, 16 );
      if ( bb == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      strcpy ( tag, "D-D" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      /**************************************************************************************
       * loop on gi1
       **************************************************************************************/
      for ( int igi1 = 0; igi1 < gamma_f1_delta_number; igi1++ ) {

        /**************************************************************************************
         * loop on gf1
         **************************************************************************************/
        for ( int igf1 = 0; igf1 < gamma_f1_delta_number; igf1++ ) {

          /**************************************************************************************
           * loop on pf1
           **************************************************************************************/
          for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

            /**************************************************************************************
             * loop on diagrams
             **************************************************************************************/
            for ( int idiag = 0; idiag < 6; idiag++ ) {

              if ( io_proc == 2 ) {

                aff_key_conversion_diagram ( aff_tag, tag, NULL, NULL, g_sink_momentum_list[ipf1], NULL, gamma_f1_delta_list[igi1][0], -1, gamma_f1_delta_list[igf1][0], -1, gsx, "d", idiag+1 , 0);

                if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);
 
                affdir = aff_reader_chpath (affr, affn, aff_tag );
                exitstatus = aff_node_get_complex (affr, affdir, bb[igi1][igf1][ipf1][idiag][0], T_global*16);
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__ );
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
       * loop on total momenta
       **************************************************************************************/
      for ( int iclass = 0; iclass  < ptot_nclass; iclass++ ) {

        int pref[3] = { ptot_class[iclass][0][0], ptot_class[iclass][0][1], ptot_class[iclass][0][2] };

        /**************************************************************************************
         * AFF output file
         **************************************************************************************/
        if ( io_proc == 2 ) {
          sprintf( filename, "%s.d.PX%dPY%dPZ%d.%.4d.t%dx%dy%dz%d.aff", "piN_piN_diagrams",
              pref[0], pref[1], pref[2], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
          affw = aff_writer (filename);
          if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
            EXIT(4);
          } else {
            fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
          }
        }  // end of if io_proc == 2


      for ( int imem = 0; imem < ptot_nmem[iclass]; imem++ ) {

        int ptot[3] = { ptot_class[iclass][imem][0], ptot_class[iclass][imem][1], ptot_class[iclass][imem][2] };

        int mptot[3] = { -ptot[0], -ptot[1], -ptot[2] };

        int iptot =  get_momentum_id ( ptot, g_sink_momentum_list, g_sink_momentum_number );
        if ( iptot == -1 ) continue;


        /**************************************************************************************
         * loop on diagrams
         **************************************************************************************/
        for ( int idiag = 0; idiag < 6; idiag++ ) {
#if 0
          /**************************************************************************************
           * AFF output file
           **************************************************************************************/
          if ( io_proc == 2 ) {
            sprintf( filename, "%s.D%d.%.4d.PX%dPY%dPZ%d.t%dx%dy%dz%d.aff", "piN_piN_diagrams", idiag+1, Nconf, 
                ptot[0], ptot[1], ptot[2],
                gsx[0], gsx[1], gsx[2], gsx[3] );
            affw = aff_writer (filename);
            if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
              EXIT(4);
            } else {
              fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
            }
          }  // end of if io_proc == 2
#endif  // of if 0

          /**************************************************************************************
           * loop on gf1
           **************************************************************************************/
          for ( int igf1 = 0; igf1 < gamma_f1_delta_number; igf1++ ) {

            gamma_matrix_type gf12;
            gamma_matrix_set  ( &gf12, gamma_f1_delta_list[igf1][1], gamma_f1_delta_sign[igf1][1] );

            /**************************************************************************************
             * loop on gi1
             **************************************************************************************/
            for ( int igi1 = 0; igi1 < gamma_f1_delta_number; igi1++ ) { 

              gamma_matrix_type gi12;
              gamma_matrix_set  ( &gi12, gamma_f1_delta_list[igi1][1], gamma_f1_delta_sign[igi1][1] );

              char aff_tag_suffix[400];

              contract_diagram_key_suffix ( aff_tag_suffix,
                  -1, NULL,
                  gamma_f1_delta_list[igf1][0], gamma_f1_delta_list[igf1][1], ptot,
                  -1, NULL,
                  gamma_f1_delta_list[igi1][0], gamma_f1_delta_list[igi1][1], mptot,
                  gsx );

              /**************************************************************************************/
              /**************************************************************************************/

              double _Complex *** diagram = init_3level_ztable ( T_global, 4, 4 );
              if ( diagram == NULL ) {
                fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
                EXIT(101);
              }

              // copy current bb configuration to T x 4x4 diagram
              memcpy( diagram[0][0], bb[igi1][igf1][iptot][idiag][0], 16*T_global*sizeof(double _Complex) );

              // transpose
              if ( ( exitstatus = contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( diagram, diagram, T_global ) ) != 0 ) {
                fprintf( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_eq_zm4x4_field_transposed, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(102);
              }

              // add boundary phase
#ifdef _ADD_BOUNDARY_PHASE
              if ( ( exitstatus = correlator_add_baryon_boundary_phase ( diagram, gsx[0], +1, T_global ) ) != 0 ) {
                fprintf( stderr, "[piN2piN_diagrams_complete] Error from correlator_add_baryon_boundary_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(103);
              }
#endif 
              // add source phase
              if ( ( exitstatus = correlator_add_source_phase ( diagram, mptot, &(gsx[1]), T_global ) ) != 0 ) {
                fprintf( stderr, "[piN2piN_diagrams_complete] Error from correlator_add_source_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(104);
              }
  
              // add outer gamma matrices
              if ( ( exitstatus =  contract_diagram_zm4x4_field_mul_gamma_lr ( diagram, diagram, gf12, gi12, T_global ) ) != 0 ) {
                fprintf( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_mul_gamma_lr, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }
  
              // add phase from phase convention
              double _Complex const zsign = contract_diagram_get_correlator_phase ( "b-b",
                  gamma_f1_delta_list[igi1][0], gamma_f1_delta_list[igi1][1], -1,
                  gamma_f1_delta_list[igf1][0], gamma_f1_delta_list[igf1][1], -1 );
  
              if ( ( exitstatus = contract_diagram_zm4x4_field_ti_eq_co ( diagram, zsign, T_global ) ) != 0 ) {
                fprintf ( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_ti_eq_co, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(106)
              }

              // AFF write fwd
              sprintf(aff_tag, "/%s/d%d/fwd/%s", "D-D", idiag+1, aff_tag_suffix );
              if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] aff tag = %s\n", aff_tag);
              if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(107);
              }
   
              // AFF write bwd
              sprintf(aff_tag, "/%s/d%d/bwd/%s", "D-D", idiag+1, aff_tag_suffix );
              if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] aff tag = %s\n", aff_tag);
              if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(108);
              }

              fini_3level_ztable ( &diagram );

            }  // end of loop on Gamma_i1

          }  // end of loop on Gamma_f1

        }  // end of loop on diagrams

      }  // end of loop on ptot_nmem

        if ( io_proc == 2 ) {
          if ( const char * aff_status_str = aff_writer_close (affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__ );
            EXIT(111);
          }
        }  // end of if io_proc == 2

      }  // end of loop on ptot_nclass

      /**************************************************************************************/
      /**************************************************************************************/

      fini_6level_ztable ( &bb );

      /**************************************************************************************/
      /**************************************************************************************/

      double _Complex ******* mbb = init_7level_ztable ( g_seq_source_momentum_number, gamma_f1_delta_number, gamma_f1_nucleon_number, g_sink_momentum_number, 6, T_global, 16 );
      if ( mbb == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_7level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }

      /**************************************************************************************
       * piN-D
       **************************************************************************************/

      strcpy ( tag, "piN-D" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      /**************************************************************************************
       * loop on pi2
       **************************************************************************************/
      for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

        /**************************************************************************************
         * loop on gi1
         **************************************************************************************/
        for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) {

          /**************************************************************************************
           * loop on gf1
           **************************************************************************************/
          for ( int igf1 = 0; igf1 < gamma_f1_delta_number; igf1++ ) {

            /**************************************************************************************
             * loop on pf1
             **************************************************************************************/
            for ( int ipf1 = 0; ipf1 < g_sink_momentum_number; ipf1++ ) {

              /**************************************************************************************
               * loop on diagrams
               **************************************************************************************/
              for ( int idiag = 0; idiag < 6; idiag++ ) {

                if ( io_proc == 2 ) {

                  aff_key_conversion_diagram ( aff_tag, tag, NULL, g_seq_source_momentum_list[ipi2], g_sink_momentum_list[ipf1], NULL, gamma_f1_nucleon_list[igi1][0], -1, gamma_f1_delta_list[igf1][0], -1, gsx, "t", idiag+1 , 0 );

                  if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);

                  affdir = aff_reader_chpath (affr, affn, aff_tag );
                  exitstatus = aff_node_get_complex (affr, affdir, mbb[ipi2][igi1][igf1][ipf1][idiag][0], T_global*16);
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__ );
                    EXIT(105);
                  }

                }  // end of if io proc == 2

              }  // end of loop on diagrams

            }  // end of loop on sink momentum

         }  // end of loop on Gamma_f1

        }  // end of loop on Gamma_i1

      }  // end of loop on sequential source momenta pi2

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * loop on total momenta
       **************************************************************************************/
      for ( int iclass = 0; iclass  < ptot_nclass; iclass++ ) {

        int pref[3] = { ptot_class[iclass][0][0], ptot_class[iclass][0][1], ptot_class[iclass][0][2] };

        /**************************************************************************************
         * AFF output file
         **************************************************************************************/
        if ( io_proc == 2 ) {
          sprintf( filename, "%s.t.PX%dPY%dPZ%d.%.4d.t%dx%dy%dz%d.aff", "piN_piN_diagrams",
              pref[0], pref[1], pref[2], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
          affw = aff_writer (filename);
          if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
            EXIT(4);
          } else {
            fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
          }
        }  // end of if io_proc == 2


      for ( int imem = 0; imem < ptot_nmem[iclass]; imem++ ) {

        int ptot[3]  = { ptot_class[iclass][imem][0], ptot_class[iclass][imem][1], ptot_class[iclass][imem][2] };

        int mptot[3] = { -ptot[0], -ptot[1], -ptot[2] };

        int iptot = get_momentum_id ( ptot, g_sink_momentum_list, g_sink_momentum_number );
        if ( iptot == -1 ) continue;

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
        int * source_momentum_id =  get_conserved_momentum_id ( g_seq_source_momentum_list,  g_seq_source_momentum_number,  mptot, g_sink_momentum_list, g_sink_momentum_number );
        if ( source_momentum_id == NULL ) {
          // fprintf(stderr, "[piN2piN_diagrams_complete] Error from get_conserved_momentum_id %s %d\n", __FILE__, __LINE__ );
          // EXIT(1);
          fprintf(stdout, "# [piN2piN_diagrams_complete] source_momentum_id empty; continue %s %d\n", __FILE__, __LINE__ );
          continue;
        }
        // TEST
        //for ( int i = 0; i < g_seq_source_momentum_number; i++ ) {
        //  fprintf ( stdout, "# [check_source_momentum_id] P %3d %3d% 3d   source_momentum_id %d  %d\n",
        //      ptot[0], ptot[1], ptot[2], i, source_momentum_id[i] );
        //}
#endif

        /**************************************************************************************
         * loop on diagrams
         **************************************************************************************/
        for ( int idiag = 0; idiag < 6; idiag++ ) {

#if 0
          /**************************************************************************************
           * AFF output file
           **************************************************************************************/
          if ( io_proc == 2 ) {
            sprintf( filename, "%s.T%d.%.4d.PX%dPY%dPZ%d.t%dx%dy%dz%d.aff", "piN_piN_diagrams", idiag+1, Nconf, 
                ptot[0], ptot[1], ptot[2],
                gsx[0], gsx[1], gsx[2], gsx[3] );
            affw = aff_writer (filename);
            if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
              EXIT(4);
            } else {
              fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
            }
          }  // end of if io_proc == 2
#endif  // of if 0

          /**************************************************************************************
           * loop on total momenta
           **************************************************************************************/
          for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

            int pi2[3] = { g_seq_source_momentum_list[ipi2][0], g_seq_source_momentum_list[ipi2][1], g_seq_source_momentum_list[ipi2][2] };

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
            int ipi1 = source_momentum_id[ipi2];
            if ( ipi1 == -1 ) {
              if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagrams_complete] Warning, skipping seq momentum no %d  (  %3d, %3d, %3d ) for ptot (%3d, %3d, %3d)\n", ipi2, 
                  g_seq_source_momentum_list[ipi2][0], g_seq_source_momentum_list[ipi2][1], g_seq_source_momentum_list[ipi2][2],
                  ptot[0], ptot[1], ptot[2] );
              continue;
            //} else {
            //  if ( g_verbose > 2 ) fprintf ( stdout, "# [piN2piN_diagrams_complete] using pi1 ( %3d %3d %3d )   for seq momentum no %d  (  %3d, %3d, %3d ) for ptot (%3d, %3d, %3d)\n", 
            //      g_sink_momentum_list[ipi1][0], g_sink_momentum_list[ipi1][1], g_sink_momentum_list[ipi1][2],
            //      ipi2, 
            //      g_seq_source_momentum_list[ipi2][0], g_seq_source_momentum_list[ipi2][1], g_seq_source_momentum_list[ipi2][2],
            //      ptot[0], ptot[1], ptot[2] );
            }

            int pi1[3] = {
              g_sink_momentum_list[ipi1][0],
              g_sink_momentum_list[ipi1][1],
              g_sink_momentum_list[ipi1][2]
            };
#else
            int pi1[3] = {
              -( ptot[0] + pi2[0] ),
              -( ptot[1] + pi2[1] ),
              -( ptot[2] + pi2[2] )
            };
#endif

            /**************************************************************************************
             * loop on gf1
             **************************************************************************************/
            for ( int igf1 = 0; igf1 < gamma_f1_delta_number; igf1++ ) {

              gamma_matrix_type gf12;
              gamma_matrix_set  ( &gf12, gamma_f1_delta_list[igf1][1], gamma_f1_delta_sign[igf1][1] );

              /**************************************************************************************
               * loop on gi1
               **************************************************************************************/
              for ( int igi1 = 0; igi1 < gamma_f1_nucleon_number; igi1++ ) {
                // sequential source vertex always gamma_5
                int const igi2 = 0;

                gamma_matrix_type gi12;
                gamma_matrix_set  ( &gi12, gamma_f1_nucleon_list[igi1][1], gamma_f1_nucleon_sign[igi1][1] );


                char aff_tag_suffix[400];

                contract_diagram_key_suffix ( aff_tag_suffix,
                    -1, NULL,
                    gamma_f1_delta_list[igf1][0],   gamma_f1_delta_list[igf1][1],   ptot,
                    gamma_i2_list[igi2],                                            pi2,
                    gamma_f1_nucleon_list[igi1][0], gamma_f1_nucleon_list[igi1][1], pi1,
                    gsx );

  
                /**************************************************************************************/
                /**************************************************************************************/

                double _Complex *** diagram = init_3level_ztable ( T_global, 4, 4 );
                if ( diagram == NULL ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
                  EXIT(101);
                }

                // copy current bb configuration to T x 4x4 diagram
                memcpy( diagram[0][0], mbb[ipi2][igi1][igf1][iptot][idiag][0], 16*T_global*sizeof(double _Complex) );

                // transpose
                if ( ( exitstatus = contract_diagram_zm4x4_field_eq_zm4x4_field_transposed ( diagram, diagram, T_global ) ) != 0 ) {
                  fprintf( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_eq_zm4x4_field_transposed, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(102);
                }

                // add boundary phase
#ifdef _ADD_BOUNDARY_PHASE
                if ( ( exitstatus = correlator_add_baryon_boundary_phase ( diagram, gsx[0], +1, T_global ) ) != 0 ) {
                  fprintf( stderr, "[piN2piN_diagrams_complete] Error from correlator_add_baryon_boundary_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(103);
                }
#endif 
                // add source phase
                if ( ( exitstatus = correlator_add_source_phase ( diagram, pi1, &(gsx[1]), T_global ) ) != 0 ) {
                  fprintf( stderr, "[piN2piN_diagrams_complete] Error from correlator_add_source_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(104);
                }
  
                // add outer gamma matrices
                if ( ( exitstatus =  contract_diagram_zm4x4_field_mul_gamma_lr ( diagram, diagram, gf12, gi12, T_global ) ) != 0 ) {
                  fprintf( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_mul_gamma_lr, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }
  
                // add phase from phase convention
                double _Complex const zsign = contract_diagram_get_correlator_phase ( "mxb-b",
                    gamma_f1_nucleon_list[igi1][0],
                    gamma_f1_nucleon_list[igi1][1],
                    gamma_i2_list[igi2],
                    gamma_f1_delta_list[igf1][0],
                    gamma_f1_delta_list[igf1][1],
                    -1 );
  
                if ( ( exitstatus = contract_diagram_zm4x4_field_ti_eq_co ( diagram, zsign, T_global ) ) != 0 ) {
                  fprintf ( stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_zm4x4_field_ti_eq_co, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(106)
                }

                // AFF write fwd
                sprintf(aff_tag, "/%s/t%d/fwd/%s", "pixN-D", idiag+1, aff_tag_suffix );
                if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] aff tag = %s\n", aff_tag);
                if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, +1, io_proc ) ) != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(107);
                }
   
                // AFF write bwd
                sprintf(aff_tag, "/%s/t%d/bwd/%s", "pixN-D", idiag+1, aff_tag_suffix );
                if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] aff tag = %s\n", aff_tag);
                if( ( exitstatus = contract_diagram_write_aff ( diagram, affw, aff_tag, gsx[0], g_src_snk_time_separation, -1, io_proc ) ) != 0 ) {
                  fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(108);
                }

                fini_3level_ztable ( &diagram );

              }  // end of loop on Gamma_i1

            }  // end of loop on Gamma_f1

          }  // end of loop on pi2

        }  // end of loop on diagrams

#ifdef _USE_GET_CONSERVED_MOMENTUM_ID
        free ( source_momentum_id );
#endif

      }  // end of loop on ptot_nmem

        if ( io_proc == 2 ) {
          if ( const char * aff_status_str = aff_writer_close (affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__ );
            EXIT(111);
          }
        }  // end of if io_proc == 2

      }  // end of loop on ptot_nclass

      /**************************************************************************************/
      /**************************************************************************************/

      fini_7level_ztable ( &mbb );

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * pi - pi
       * /m-m/t09/pi2x00pi2y00pi2z-01/sample00/gi05/gf05/px00py-01pz00
       **************************************************************************************/

      double _Complex * mm = init_1level_ztable ( T_global );
      if ( mm == NULL ) {
        fprintf(stderr, "[piN2piN_diagrams_complete] Error from init_1level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(47);
      }
      strcpy ( tag, "m-m" );
      if ( g_verbose > 2 ) fprintf(stdout, "\n\n# [piN2piN_diagrams_complete] tag = %s\n", tag);

      /**************************************************************************************/
      /**************************************************************************************/

      /**************************************************************************************
       * loop on pf2
       **************************************************************************************/
      for ( int iclass = 0; iclass  < ptot_nclass; iclass++ ) {

        int pref[3] = { ptot_class[iclass][0][0], ptot_class[iclass][0][1], ptot_class[iclass][0][2] };

        /**************************************************************************************
         * AFF output file
         **************************************************************************************/
        if ( io_proc == 2 ) {
          sprintf( filename, "%s.m.PX%dPY%dPZ%d.%.4d.t%dx%dy%dz%d.aff", "piN_piN_diagrams",
              pref[0], pref[1], pref[2], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
          affw = aff_writer (filename);
          if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
            EXIT(4);
          } else {
            fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
          }
        }  // end of if io_proc == 2

      
        for ( int imem = 0; imem < ptot_nmem[iclass]; imem++ ) {

          int pf2[3] = { ptot_class[iclass][imem][0], ptot_class[iclass][imem][1], ptot_class[iclass][imem][2] };

#if 0
          /**************************************************************************************
           * AFF output file
           **************************************************************************************/
          if ( io_proc == 2 ) {
            sprintf( filename, "%s.M%d.%.4d.PX%dPY%dPZ%d.t%dx%dy%dz%d.aff", "piN_piN_diagrams", 1, Nconf, 
                pf2[0], pf2[1], pf2[2],
                gsx[0], gsx[1], gsx[2], gsx[3] );
            affw = aff_writer (filename);
            if ( const char * aff_status_str =  aff_writer_errstr(affw) ) {
              fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
              EXIT(4);
            } else {
              fprintf(stdout, "# [piN2piN_diagrams_complete] writing data to file %s %s %d\n", filename, __FILE__, __LINE__);
            }
          }  // end of if io_proc == 2
#endif  // of if 0

          for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ ) {

            int pi2[3] = {
              g_seq_source_momentum_list[ipi2][0],
              g_seq_source_momentum_list[ipi2][1],
              g_seq_source_momentum_list[ipi2][2] 
            };

            /**************************************************************************************
             * loop on gi2
            **************************************************************************************/
            for ( int igi2 = 0; igi2 < gamma_i2_number; igi2++ ) {

              /**************************************************************************************
               * loop on gf2
               **************************************************************************************/
              for ( int igf2 = 0; igf2 < gamma_f2_number; igf2++ ) {

                if ( io_proc == 2 ) {

                  aff_key_conversion_diagram (  aff_tag, tag, NULL, pi2, NULL, pf2, -1, gamma_i2_list[igi2], -1, gamma_f2_list[igf2], gsx, NULL, 0 , 0 );

                  if (g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] key = \"%s\"\n", aff_tag);
 
                  affdir = aff_reader_chpath (affr_oet, affn_oet, aff_tag );
                  exitstatus = aff_node_get_complex (affr_oet, affdir, mm, T_global);
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_node_get_complex for key \"%s\", status was %d %s %d\n", aff_tag, exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  char aff_tag_suffix[400];

                  contract_diagram_key_suffix ( aff_tag_suffix, gamma_f2_list[igf2], pf2, -1, -1, NULL, gamma_i2_list[igi2], pi2, -1, -1, NULL, gsx );

                  // AFF write fwd
                  sprintf(aff_tag, "/%s/m1/fwd/%s", "pixN-pixN", aff_tag_suffix );
                  if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] aff tag = %s\n", aff_tag);
                  if( ( exitstatus = contract_diagram_write_scalar_aff ( mm, affw, aff_tag, gsx[0], T_global, +1, io_proc ) ) != 0 ) {
                    fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d\n", exitstatus);
                    EXIT(105);
                  }
 
                  // AFF write bwd
                  //sprintf(aff_tag, "/%s/m1/bwd/%s", "pixN-pixN", aff_tag_suffix );
                  //if ( g_verbose > 2 ) fprintf(stdout, "# [piN2piN_diagrams_complete] aff tag = %s\n", aff_tag);
                  //if( ( exitstatus = contract_diagram_write_scalar_aff ( mm, affw, aff_tag, gsx[0], T_global, -1, io_proc ) ) != 0 ) {
                  //  fprintf(stderr, "[piN2piN_diagrams_complete] Error from contract_diagram_write_aff, status was %d\n", exitstatus);
                  //  EXIT(105);
                  //}

                }   // end of if io proc > 2 

              }  // end of loop on Gamma_f2

            }  // end of loop on Gamma_i2

          }  // end of loop on pi2

        }  // end of loop on members
          
        if ( io_proc == 2 ) {
          if ( const char * aff_status_str = aff_writer_close (affw) ) {
            fprintf(stderr, "[piN2piN_diagrams_complete] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__ );
            EXIT(111);
          }
        }  // end of if io_proc == 2

      }  /* end of loop on classes */

      /**************************************************************************************/
      /**************************************************************************************/

      fini_1level_ztable ( &mm );

    }  // end of loop on coherent source locations

    /**************************************************************************************/
    /**************************************************************************************/

    /**************************************************************************************
     * close AFF readers for input and output files
     **************************************************************************************/
    if(io_proc == 2) {
      aff_reader_close (affr);
      aff_reader_close (affr_oet);

    }  // end of if io_proc == 2

  }  // end of loop on base source locations

  /**************************************************************************************
   * free the allocated memory, finalize
   **************************************************************************************/

  fini_momentum_classes ( &ptot_class, &ptot_nmem, &ptot_nclass );

  free_geometry();

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [piN2piN_diagrams_complete] %s# [piN2piN_diagrams_complete] end of run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [piN2piN_diagrams_complete] %s# [piN2piN_diagrams_complete] end of run\n", ctime(&g_the_time));
    fflush(stderr);
  }

  return(0);
}
