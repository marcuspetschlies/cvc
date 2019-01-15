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

  const int gamma_f1_nucleon_number                                = 1;
  int gamma_f1_nucleon_list[gamma_f1_nucleon_number]               = { 14 };
  /* double gamma_f1_nucleon_sign[gamma_f1_nucleon_number]            = { +1 }; */
  /* double gamma_f1_nucleon_transposed_sign[gamma_f1_nucleon_number] = { -1 }; */

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
      fprintf(stderr, "[twopt_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    } else {
      if ( g_verbose > 1 ) fprintf ( stdout, "# [twopt_analyse] Reading data from file %s\n", filename );
    }
#endif

    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

      tp.pi1[0] = -g_sink_momentum_list[imom][0];
      tp.pi1[1] = -g_sink_momentum_list[imom][1];
      tp.pi1[2] = -g_sink_momentum_list[imom][2];

      tp.pf1[0] =  g_sink_momentum_list[imom][0];
      tp.pf1[1] =  g_sink_momentum_list[imom][1];
      tp.pf1[2] =  g_sink_momentum_list[imom][2];

      for ( int if1 = 0; if1 < gamma_f1_nucleon_number; if1++ ) {
      for ( int if2 = 0; if2 < gamma_f1_nucleon_number; if2++ ) {

        tp.gi1[0] = gamma_f1_nucleon_list[if1];
        tp.gi1[1] = 4;

        tp.gf1[0] = gamma_f1_nucleon_list[if2];
        tp.gf1[1] = 4;

        /***********************************************************
         * show the 2-point function content
         ***********************************************************/
        twopoint_function_print ( &tp, tp.name, stdout );

        /***********************************************************
         * read the n1 data set
         ***********************************************************/
        sprintf(aff_tag, "/N-N/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/n1/px%.2dpy%.2dpz%.2d",
            gsx[0], gsx[1], gsx[2], gsx[3],
            gamma_f1_nucleon_list[if1], gamma_f1_nucleon_list[if2],
            tp.pf1[0], tp.pf1[1], tp.pf1[2] );

        exitstatus = read_aff_contraction ( (void*)(tp.c[0][0][0]), affr, NULL, aff_tag, T*16 );
        if(exitstatus != 0) {
          fprintf(stderr, "[twopt_analyse] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(12);
        }

        /***********************************************************
         * read the n2 data set
         ***********************************************************/
        sprintf(aff_tag, "/N-N/t%.2dx%.2dy%.2dz%.2d/gi%.2d/gf%.2d/n2/px%.2dpy%.2dpz%.2d",
            gsx[0], gsx[1], gsx[2], gsx[3],
            gamma_f1_nucleon_list[if1], gamma_f1_nucleon_list[if2],
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
          exitstatus =  contract_diagram_zm4x4_field_mul_gamma_lr ( tp.c[0], tp.c[0], gamma[tp.gf1[1]], gamma[tp.gi1[1]], tp.T );
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


        sprintf ( filename, "%s_px%dpy%dpz%d_gi1%d_gi2%d_%s_parity%d_n%.4d", tp.name,
            tp.pf1[0], tp.pf1[1], tp.pf1[2],
            tp.gi1[0], tp.gf1[0], tp.fbwd, tp.parity_project, Nconf );

        FILE * ofs = ( isource_location == 0 ) ? fopen ( filename, "w" ) : fopen ( filename, "a" );
        fprintf ( ofs, "# t%.2dx%.2dy%.2dz%.2d\n", gsx[0], gsx[1], gsx[2], gsx[3] );
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
  
        for ( int if1 = 0; if1 < gamma_f1_nucleon_number; if1++ ) {
        for ( int if2 = 0; if2 < gamma_f1_nucleon_number; if2++ ) {
  
          tp.gi1[0] = gamma_f1_nucleon_list[if1];
          tp.gi1[1] = 4;
  
          tp.gf1[0] = gamma_f1_nucleon_list[if2];
          tp.gf1[1] = 4;
  
          /***********************************************************
           * show the 2-point function content
           ***********************************************************/
          twopoint_function_print ( &tp, tp.name, stdout );
  
          /***********************************************************
           * read the n1 data set
           ***********************************************************/
          sprintf(aff_tag, "/N-N/s%d/t%.2d/gi%.2d/gf%.2d/n1/px%.2dpy%.2dpz%.2d", isample,
              gts, gamma_f1_nucleon_list[if1], gamma_f1_nucleon_list[if2],
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
              gts, gamma_f1_nucleon_list[if1], gamma_f1_nucleon_list[if2],
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
