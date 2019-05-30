/****************************************************
 * twopt_analyse
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
#include <complex.h>
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
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "contract_diagrams.h"
#include "twopoint_function_utils.h"
#include "zm4x4.h"
#include "gamma.h"



#include "clover.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1


using namespace cvc;

void usage() {
  fprintf(stdout, "Code form 2-pt function\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "twopt";

  int c;
  int filename_set = 0;
  int gsx[4], sx[4];
  int exitstatus;
  int io_proc = -1;
  char filename[100];

  char const gamma_id_to_Cg_ascii[16][10] = {
    "Cgy",
    "Cgzg5",
    "Cgt",
    "Cgxg5",
    "Cgygt",
    "Cgyg5gt",
    "Cgyg5",
    "Cgz",
    "Cg5gt",
    "Cgx",
    "Cgzg5gt",
    "C",
    "Cgxg5gt",
    "Cgxgt",
    "Cg5",
    "Cgzgt"
  };

  char const fbwd_str[2][10] = { "fwd", "bwd" };
#if 0
  const int gamma_f1_number                                = 1;
  int gamma_f1_list[gamma_f1_number]               = { 14 };
  /* double gamma_f1_sign[gamma_f1_number]            = { +1 }; */
  /* double gamma_f1_transposed_sign[gamma_f1_number] = { -1 }; */
#endif

  const int gamma_f1_number            = 3;
  /*                                       C gx C gy C gz */
  int gamma_f1_list[gamma_f1_number]   = {    9,   0,   7 };

#ifdef HAVE_LHPC_AFF
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
  if(filename_set==0) strcpy(filename, "twopt.input");
  /* fprintf(stdout, "# [twopt_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [twopt_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [twopt_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [twopt_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[twopt_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[twopt_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[twopt_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [twopt_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***********************************************************
   * initialize gamma matrix algebra and several
   * gamma basis matrices
   ***********************************************************/
  init_gamma_matrix ();

  /***********************************************************
   * set gamma matrices
   *   tmLQCD counting
   ***********************************************************/
  gamma_matrix_type gamma[16];
  for ( int i = 0; i < 16; i++ ) {
    gamma_matrix_set ( &(gamma[i]), i, 1. );
  }

  /***************************************************************************
   ***************************************************************************
   **
   ** point-to-all version
   **
   ***************************************************************************
   ***************************************************************************/

  /***************************************************************************
   * init a twopoint function
   ***************************************************************************/
  twopoint_function_type tp;

  twopoint_function_init ( &tp );
#if 0
  strcpy ( tp.type,     "b-b" );
  strcpy ( tp.name,     "N-N");
  strcpy ( tp.diagrams, "n1,n2" );
  strcpy ( tp.norm,     "1.0,1.0");
  strcpy ( tp.fbwd,     "fwd" );
  tp.n              = 2;
  tp.reorder        = +1;
  tp.parity_project = +1;
  tp.T              = T;
  tp.d              = 4;
#endif

  strcpy ( tp.type,     "b-b" );
  strcpy ( tp.name,     "omega-omega");
  strcpy ( tp.diagrams, "d1,d6" );
  strcpy ( tp.norm,     "4.0,2.0");
  tp.n              = 2;
  tp.reorder        = +1;
  tp.T              = T;
  tp.d              = 4;

  twopoint_function_allocate ( &tp );

  /***********************************************************
   * loop on source locations
   ***********************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) {

    /***********************************************************
     * determine source coordinates, find out, if source_location is in this process
     ***********************************************************/
    gsx[0] = ( g_source_coords_list[isource_location][0] +  T_global ) %  T_global;
    gsx[1] = ( g_source_coords_list[isource_location][1] + LX_global ) % LX_global;
    gsx[2] = ( g_source_coords_list[isource_location][2] + LY_global ) % LY_global;
    gsx[3] = ( g_source_coords_list[isource_location][3] + LZ_global ) % LZ_global;

    int source_proc_id = -1;
    exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[twopt_analyse] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    memcpy ( tp.source_coords, gsx, 4*sizeof(int) );

#ifdef HAVE_LHPC_AFF
    /***********************************************
     * writer for aff output file
     ***********************************************/
    sprintf(filename, "%s.%.4d.t%dx%dy%dz%d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
   
    struct AffReader_s * affr = aff_reader ( filename );
    const char * aff_status_str = aff_reader_errstr ( affr );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[twopt_analyse] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    } else {
      if ( g_verbose > 1 ) fprintf ( stdout, "# [twopt_analyse] Reading data from file %s\n", filename );
    }
#endif

    /***********************************************************
     * loop on fwd, bwd
     ***********************************************************/
    for ( int ifbwd = 0; ifbwd <= 1 ; ifbwd++ ) {

      strcpy ( tp.fbwd, fbwd_str[ifbwd] );

      if ( strcmp ( tp.fbwd, "fwd" ) == 0 ) {
        tp.parity_project = +1;
      } else if ( strcmp ( tp.fbwd, "bwd" ) == 0 ) {
        tp.parity_project = -1;
      }

      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

        tp.pi1[0] = -g_sink_momentum_list[imom][0];
        tp.pi1[1] = -g_sink_momentum_list[imom][1];
        tp.pi1[2] = -g_sink_momentum_list[imom][2];

        tp.pf1[0] =  g_sink_momentum_list[imom][0];
        tp.pf1[1] =  g_sink_momentum_list[imom][1];
        tp.pf1[2] =  g_sink_momentum_list[imom][2];

        /***********************************************************
         * twopoint function to hold matrix of correlators
         ***********************************************************/

        twopoint_function_type tp_tensor;

        twopoint_function_init ( &tp_tensor );

        strcpy ( tp_tensor.type,     tp.type );
        strcpy ( tp_tensor.name,     tp.name);
        strcpy ( tp_tensor.fbwd,     tp.fbwd );
        tp_tensor.n              = gamma_f1_number * gamma_f1_number;
        tp_tensor.T              = T;
        tp_tensor.d              = 4;
        tp_tensor.parity_project = tp.parity_project;
        memcpy ( tp_tensor.pi1, tp.pi1, 3*sizeof(int) );
        memcpy ( tp_tensor.pf1, tp.pf1, 3*sizeof(int) );

        twopoint_function_allocate ( &tp_tensor );

        /***********************************************************
         * loop on inner vertices at source
         ***********************************************************/
        for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {

          tp.gi1[0] = gamma_f1_list[if1];
          tp.gi1[1] = 4;

        /***********************************************************
         * loop on inner vertices at sink
         ***********************************************************/
        for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {

          tp.gf1[0] = gamma_f1_list[if2];
          tp.gf1[1] = 4;

          /***********************************************************
           * show the 2-point function content
           ***********************************************************/
          if ( g_verbose > 3 ) twopoint_function_print ( &tp, tp.name, stdout );

          /***********************************************************
           * loop on diagrams
           ***********************************************************/
          for ( int idiag = 0; idiag < tp.n; idiag++ ) {

            char diagram_name[100];
            twopoint_function_get_diagram_name ( diagram_name, &tp, idiag );

            /***********************************************************
             * read the diagram data set
             ***********************************************************/
            sprintf(aff_tag, "/%s/t%.2dx%.2dy%.2dz%.2d/m%10.8f/gi%.2d/gf%.2d/%s/px%.2dpy%.2dpz%.2d", tp.name,
                gsx[0], gsx[1], gsx[2], gsx[3], g_mu,
                gamma_f1_list[if1], gamma_f1_list[if2], diagram_name,
                tp.pf1[0], tp.pf1[1], tp.pf1[2] );

            exitstatus = read_aff_contraction ( (void*)(tp.c[idiag][0][0]), affr, NULL, aff_tag, T*16 );
            if(exitstatus != 0) {
              fprintf(stderr, "[twopt_analyse] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(12);
            }
          }  /* end of loop on dagrams to read from file */

          for ( int idiag = 1; idiag < tp.n; idiag++ ) {

            /***********************************************************
             * add up diagrams
             ***********************************************************/
            exitstatus = contract_diagram_zm4x4_field_pl_eq_zm4x4_field ( tp.c[0], tp.c[idiag], tp.T );
          }

          /***********************************************************
           * reorder relative to source
           ***********************************************************/
          /* reorder_to_absolute_time ( tp.c[0], tp.c[0], tp.source_coords[0], tp.reorder, tp.T ); */
          reorder_to_relative_time ( tp.c[0], tp.c[0], tp.source_coords[0], tp.reorder, tp.T );

          /***********************************************************
           * add source phase
           ***********************************************************/
          exitstatus = correlator_add_source_phase ( tp.c[0], tp.pi1,  &(tp.source_coords[1]), tp.T );

          /***********************************************************
           * add boundary phase
           ***********************************************************/
          if ( strcmp ( tp.fbwd, "fwd" ) == 0 ) {
            exitstatus = correlator_add_baryon_boundary_phase ( tp.c[0], 0, +1, tp.T );
          } else if ( strcmp ( tp.fbwd, "bwd" ) == 0 ) {
            exitstatus = correlator_add_baryon_boundary_phase ( tp.c[0], 0, -1, tp.T );
          }

          /******************************************************
           * apply gi1[1] and gf1[1]
           ******************************************************/
          if ( ( tp.gi1[1] != -1 ) && ( tp.gf1[1] != -1 ) ) {
            exitstatus = contract_diagram_zm4x4_field_mul_gamma_lr ( tp.c[0], tp.c[0], gamma[tp.gf1[1]], gamma[tp.gi1[1]], tp.T );
          }

          /******************************************************
           * spin-parity projection
           ******************************************************/
          correlator_spin_parity_projection ( tp_tensor.c[if2 * gamma_f1_number + if1], tp.c[0], (double)tp.parity_project, tp.T );
          /* correlator_spin_parity_projection ( tp.c[0], tp.c[0], (double)tp.parity_project, tp.T ); */

          /******************************************************
           * correlator phase
           ******************************************************/
          double _Complex const zsign = contract_diagram_get_correlator_phase ( tp.type,
              tp.gi1[0], tp.gi1[1], tp.gi2,
              tp.gf1[0], tp.gf1[1], tp.gf2 );

          /* exitstatus = contract_diagram_zm4x4_field_ti_eq_co ( tp.c[0], zsign, tp.T  ); */
          exitstatus = contract_diagram_zm4x4_field_ti_eq_co ( tp_tensor.c[if2 * gamma_f1_number + if1], zsign, tp.T  );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[twopt_invert_contract] Error from contract_diagram_zm4x4_field_ti_eq_co, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(109)
          }


          /******************************************************
           * trace
           ******************************************************/
          double _Complex * corr_tr = init_1level_ztable ( tp_tensor.T );
#pragma omp parallel for
          for ( int i = 0; i < tp_tensor.T; i++ )  {
            co_eq_tr_zm4x4 ( corr_tr+i, tp_tensor.c[if2 * gamma_f1_number + if1 ][i] );
          }

          sprintf ( filename, "%s_px%dpy%dpz%d_%s_%s_%s_parity%d_n%.4d", tp.name,
              tp.pf1[0], tp.pf1[1], tp.pf1[2],
              gamma_id_to_Cg_ascii[tp.gf1[0]], gamma_id_to_Cg_ascii[tp.gi1[0]], tp.fbwd, tp.parity_project, Nconf );

          FILE * ofs = ( isource_location == 0 ) ? fopen ( filename, "w" ) : fopen ( filename, "a" );
          fprintf ( ofs, "# t%.2dx%.2dy%.2dz%.2d\n", gsx[0], gsx[1], gsx[2], gsx[3] );
          for ( int i = 0; i < tp_tensor.T; i++ )  {
            fprintf ( ofs , "%3d %25.16e %25.16e\n", i, creal( corr_tr[i] ), cimag ( corr_tr[i] ) );
          }
          fclose ( ofs );

          fini_1level_ztable ( &corr_tr );


        }}  /* end of loop on if2 and if1 */

        /******************************************************
         * spin 3/2 projection
         ******************************************************/

        double _Complex *** c_aux = init_3level_ztable ( tp.T, tp.d, tp.d );

        double _Complex **** c_proj = init_4level_ztable ( gamma_f1_number * gamma_f1_number, tp_tensor.T, tp_tensor.d, tp_tensor.d );
        
        for ( int i = 0; i < 3; i++ ) {
          for ( int k = 0; k < 3; k++ ) {

            for ( int l = 0; l < 3; l++ ) {
              correlator_spin_projection ( c_aux, tp_tensor.c[3*l+k], i+1, l+1, (double)(i==l), -1., tp_tensor.T );

              contract_diagram_zm4x4_field_pl_eq_zm4x4_field ( c_proj[3*i+k], c_aux, tp_tensor.T );
            }

          }  /* end of loop on k */

        } /* end of loop on i */

        memcpy ( tp_tensor.c[0][0][0], c_proj[0][0][0], 9 * tp_tensor.T * tp_tensor.d * tp_tensor.d * sizeof ( double _Complex ) );

        fini_3level_ztable ( &c_aux );
        fini_4level_ztable ( &c_proj );


        for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {

        for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {

          /******************************************************
           * trace
           ******************************************************/
          double _Complex * corr_tr = init_1level_ztable ( tp_tensor.T );
#pragma omp parallel for
          for ( int i = 0; i < tp_tensor.T; i++ )  {
            co_eq_tr_zm4x4 ( corr_tr+i, tp_tensor.c[if1 * gamma_f1_number + if2 ][i] );
          }

          sprintf ( filename, "%s_spinprojected_px%dpy%dpz%d_%s_%s_%s_parity%d_n%.4d", tp.name,
              tp.pf1[0], tp.pf1[1], tp.pf1[2],
              gamma_id_to_Cg_ascii[gamma_f1_list[if1]], gamma_id_to_Cg_ascii[gamma_f1_list[if2]], tp_tensor.fbwd, tp_tensor.parity_project, Nconf );

          FILE * ofs = ( isource_location == 0 ) ? fopen ( filename, "w" ) : fopen ( filename, "a" );
          fprintf ( ofs, "# t%.2dx%.2dy%.2dz%.2d\n", gsx[0], gsx[1], gsx[2], gsx[3] );
          for ( int i = 0; i < T; i++ )  {
            fprintf ( ofs , "%3d %25.16e %25.16e\n", i, creal( corr_tr[i] ), cimag ( corr_tr[i] ) );
          }
          fclose ( ofs );

          fini_1level_ztable ( &corr_tr );

        }}


        twopoint_function_fini ( &tp_tensor );

      }  /* end of loop on momenta */

    }  /* end of loop on fwd, bwd */

    /***************************************************************************/
    /***************************************************************************/

#ifdef HAVE_LHPC_AFF
    /***************************************************************************
     * close AFF reader
     ***************************************************************************/
    aff_reader_close ( affr );
#endif  /* of ifdef HAVE_LHPC_AFF */

  }  /* end of loop on source locations */

#if 0
  /***************************************************************************
   ***************************************************************************
   **
   ** stochastic propagator version
   **
   ***************************************************************************
   ***************************************************************************/

  /***************************************************************************
   * loop on source locations
   ***************************************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) {
  
    /***********************************************************
     * determine source coordinates, find out, if source_location is in this process
     ***********************************************************/
    int const gts  = ( g_source_coords_list[isource_location][0] +  T_global ) %  T_global;

    int const source_proc_id = ( gts / T == g_proc_coords[0] ) ? g_cart_id : -1;

    int const source_timeslice = ( source_proc_id == g_cart_id ) ? gts % T : -1;

    tp.source_coords[0] = gts;
    tp.source_coords[1] = -1;
    tp.source_coords[2] = -1;
    tp.source_coords[3] = -1;
  
    /***************************************************************************
     * loop on oet samples
     ***************************************************************************/
    for ( int isample = 0; isample < g_nsample_oet; isample++ ) {

#ifdef HAVE_LHPC_AFF
      /***********************************************
       * writer for aff output file
       ***********************************************/
      sprintf(filename, "%s_oet.%.4d.s%d.t%d.aff", outfile_prefix, Nconf, isample, gts );
     
      struct AffReader_s * affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[twopt_analyse] Error from aff_reader, status was %s for filename %s %s %d\n", aff_status_str, filename, __FILE__, __LINE__);
        EXIT(15);
      }
#endif
  
      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
  
        tp.pi1[0] = -g_sink_momentum_list[imom][0];
        tp.pi1[1] = -g_sink_momentum_list[imom][1];
        tp.pi1[2] = -g_sink_momentum_list[imom][2];
  
        tp.pf1[0] =  g_sink_momentum_list[imom][0];
        tp.pf1[1] =  g_sink_momentum_list[imom][1];
        tp.pf1[2] =  g_sink_momentum_list[imom][2];
  
        for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {
        for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {
  
          tp.gi1[0] = gamma_f1_list[if1];
          tp.gi1[1] = 4;
  
          tp.gf1[0] = gamma_f1_list[if2];
          tp.gf1[1] = 4;
  
          /***********************************************************
           * show the 2-point function content
           ***********************************************************/
          twopoint_function_print ( &tp, tp.name, stdout );
  
          /***********************************************************
           * read the n1 data set
           ***********************************************************/
          sprintf(aff_tag, "/N-N/s%d/t%.2d/gi%.2d/gf%.2d/n1/px%.2dpy%.2dpz%.2d", isample,
              gts, gamma_f1_list[if1], gamma_f1_list[if2],
              tp.pf1[0], tp.pf1[1], tp.pf1[2] );

          exitstatus = read_aff_contraction ( (void*)(tp.c[0][0][0]), affr, NULL, aff_tag, T*16 );
          if(exitstatus != 0) {
            fprintf(stderr, "[twopt_analyse] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(12);
          }
  
          /***********************************************************
           * read the n2 data set
           ***********************************************************/
          sprintf(aff_tag, "/N-N/s%d/t%.2d/gi%.2d/gf%.2d/n2/px%.2dpy%.2dpz%.2d", isample,
              gts, gamma_f1_list[if1], gamma_f1_list[if2],
              tp.pf1[0], tp.pf1[1], tp.pf1[2] );

          exitstatus = read_aff_contraction ( (void*)(tp.c[1][0][0]), affr, NULL, aff_tag, T*16 );
          if(exitstatus != 0) {
            fprintf(stderr, "[twopt_analyse] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(12);
          }
  
          /***********************************************************
           * add up diagrams
           ***********************************************************/
          exitstatus = contract_diagram_zm4x4_field_pl_eq_zm4x4_field ( tp.c[0], tp.c[1], tp.T );
  
          /***********************************************************
           * reorder relative to source
           ***********************************************************/
          /* reorder_to_absolute_time ( tp.c[0], tp.c[0], tp.source_coords[0], tp.reorder, tp.T ); */
          reorder_to_relative_time ( tp.c[0], tp.c[0], tp.source_coords[0], tp.reorder, tp.T );
  
          /***********************************************************
           * add source phase
           ***********************************************************/
          /* exitstatus = correlator_add_source_phase ( tp.c[0], tp.pi1,  &(tp.source_coords[1]), tp.T ); */
  
          /***********************************************************
           * add boundary phase
           ***********************************************************/
          if ( strcmp ( tp.fbwd, "fwd" ) == 0 ) {
            exitstatus = correlator_add_baryon_boundary_phase ( tp.c[0], 0, +1, tp.T );
          } else if ( strcmp ( tp.fbwd, "bwd" ) == 0 ) {
            exitstatus = correlator_add_baryon_boundary_phase ( tp.c[0], 0, -1, tp.T );
          }
  
          /******************************************************
           * apply gi1[1] and gf1[1]
           ******************************************************/
          if ( ( tp.gi1[1] != -1 ) && ( tp.gf1[1] != -1 ) ) {
            exitstatus =  contract_diagram_zm4x4_field_mul_gamma_lr ( tp.c[0], tp.c[0], gamma[tp.gf1[1]], gamma[tp.gi1[1]],tp.T );
          }
  
          /******************************************************
           * spin / spin-parity projection
           ******************************************************/
          correlator_spin_parity_projection ( tp.c[0], tp.c[0], (double)tp.parity_project, tp.T );
  
          /******************************************************
           * trace
           ******************************************************/
          double _Complex * corr_tr = init_1level_ztable ( T );
#pragma omp parallel for
          for ( int i = 0; i < T; i++ )  {
            co_eq_tr_zm4x4 ( corr_tr+i, tp.c[0][i] );
          }
  
          sprintf ( filename, "%s_oet_px%dpy%dpz%d_gi1%d_gi2%d_%s_parity%d_n%.4d", tp.name,
              tp.pf1[0], tp.pf1[1], tp.pf1[2],
              tp.gi1[0], tp.gf1[0], tp.fbwd, tp.parity_project, Nconf );

          FILE * ofs = ( isample == 0 && isource_location == 0 ) ? fopen ( filename, "w" ) : fopen ( filename, "a" );

          fprintf ( ofs, "# s%d t%d\n", isample, gts );
          for ( int i = 0; i < T; i++ )  {
            fprintf ( ofs , "%3d %25.16e %25.16e\n", i, creal( corr_tr[i] ), cimag ( corr_tr[i] ) );
          }
          fclose ( ofs );

          fini_1level_ztable ( &corr_tr );

        }}  /* end of loops on gi1[0] and gf1[0] */
  
      }  /* end of loop on momenta */
  
      /***************************************************************************/
      /***************************************************************************/
  
#ifdef HAVE_LHPC_AFF
      /***************************************************************************
       * close AFF reader
       ***************************************************************************/
      aff_reader_close ( affr );
#endif  /* of ifdef HAVE_LHPC_AFF */
  
    }  /* end of loop on source locations */

  }  /* end of loop on samples */ 


#endif  /* of if 0 */

  twopoint_function_fini ( &tp );

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [twopt_analyse] %s# [twopt_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [twopt_analyse] %s# [twopt_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
