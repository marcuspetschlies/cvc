/****************************************************
 * htpp_analyse
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
  
  char const gamma_bin_to_name[16][8] = { "id", "gx", "gy", "gxgy", "gz", "gxgz", "gygz", "gtg5", "gt", "gxgt", "gygt", "gzg5", "gzgt", "gyg5", "gxg5", "g5" };

  int c;
  int filename_set = 0;
  int gsx[4], sx[4];
  int exitstatus;
  int io_proc = -1;
  char filename[100];

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
  /* fprintf(stdout, "# [htpp_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [htpp_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [htpp_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [htpp_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[htpp_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[htpp_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[htpp_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [htpp_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   ***************************************************************************
   **
   ** point-to-all version
   **
   ***************************************************************************
   ***************************************************************************/

  /***************************************************************************
   * loop on twopoint functions
   ***************************************************************************/
  for ( int i_2pt = 0; i_2pt < g_twopoint_function_number; i_2pt++ ) {

    twopoint_function_type * tp = &(g_twopoint_function_list[i_2pt]);

    for ( int i_diag = 0; i_diag < tp->n; i_diag++ ) {

      char diagram_name[500];

      twopoint_function_get_diagram_name ( diagram_name,  tp, i_diag );

      char key[500];

      sprintf ( key, "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/pi2x%dpi2y%dpi2z%d/gi2_%s/g1_%s/g2_%s/PX%d_PY%d_PZ%d",
          diagram_name, 
          tp->pf1[0], tp->pf1[1], tp->pf1[2], gamma_bin_to_name[tp->gf1[0]], g_src_snk_time_separation, 
          tp->pi2[0], tp->pi2[1], tp->pi2[2], gamma_bin_to_name[tp->gi2], 
          gamma_bin_to_name[tp->gf2], gamma_bin_to_name[tp->gi1[0]],
          tp->pf2[0], tp->pf2[1], tp->pf2[2] );

      if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_analyse] key = %s\n", key );

      sprintf ( filename, "%s_pfx%dpfy%dpfz%d_gf_%s_dt%d_pi2x%dpi2y%dpi2z%d_gi2_%s_g1_%s_g2_%s_PX%d_PY%d_PZ%d",
          diagram_name, 
          tp->pf1[0], tp->pf1[1], tp->pf1[2], gamma_bin_to_name[tp->gf1[0]], g_src_snk_time_separation, 
          tp->pi2[0], tp->pi2[1], tp->pi2[2], gamma_bin_to_name[tp->gi2], 
          gamma_bin_to_name[tp->gf2], gamma_bin_to_name[tp->gi1[0]],
          tp->pf2[0], tp->pf2[1], tp->pf2[2] );

      if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_analyse] filename = %s\n", filename );

      FILE * ofs = fopen ( filename, "a" );

      /***********************************************************
       * loop on source locations
       ***********************************************************/
      for( int i_src = 0; i_src<g_source_location_number; i_src++) {

        int const t_base = g_source_coords_list[i_src][0];

        /***********************************************************
         * determine source coordinates, find out, if source_location is in this process
         ***********************************************************/
        gsx[0] = ( g_source_coords_list[i_src][0] +  T_global ) %  T_global;
        gsx[1] = ( g_source_coords_list[i_src][1] + LX_global ) % LX_global;
        gsx[2] = ( g_source_coords_list[i_src][2] + LY_global ) % LY_global;
        gsx[3] = ( g_source_coords_list[i_src][3] + LZ_global ) % LZ_global;

        int source_proc_id = -1;
        exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
        if( exitstatus != 0 ) {
          fprintf(stderr, "[htpp_analyse] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(123);
        }

#ifdef HAVE_LHPC_AFF
        /***********************************************
         * writer for aff output file
         ***********************************************/
        sprintf ( filename, "contract_3pt_hl_seq.%.4d.tbase%.2d.aff", Nconf, t_base );
   
        struct AffReader_s * affr = aff_reader ( filename );
        const char * aff_status_str = aff_reader_errstr ( affr );
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[htpp_analyse] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
          //EXIT(15);
        } else {
          if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_analyse] Reading data from file %s\n", filename );
        }
#endif

#if 0
        exitstatus = read_aff_contraction ( (void*)(tp->c[i_diag][0][0]), affr, NULL, key, T * tp->d * tp->d );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[htpp_analyse] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(12);
        }
#endif

        for( int i_coherent=0; i_coherent<g_coherent_source_number; i_coherent++) {

          /* coherent source timeslice */
          int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * i_coherent ) % T_global;

          int const csx[4] = { t_coherent ,
                               ( g_source_coords_list[i_src][1] + (LX_global/2) * i_coherent ) % LX_global,
                               ( g_source_coords_list[i_src][2] + (LY_global/2) * i_coherent ) % LY_global,
                               ( g_source_coords_list[i_src][3] + (LZ_global/2) * i_coherent ) % LZ_global };

          get_point_source_info ( csx, sx, &source_proc_id);


          fprintf ( ofs, "# %s/conf%d/t%d_x%d_y%d_z%d\n", key, Nconf, csx[0], csx[1], csx[2], csx[3]  );

          double _Complex ephase = cexp ( 2. * M_PI * ( tp->pi1[0] * csx[1] + tp->pi1[1] * csx[2] + tp->pi1[2] * csx[3] ) * I );

          int const n_tc = g_src_snk_time_separation + 1;

          for ( int it = 0; it < n_tc; it++ ) {
            int const tt = ( csx[0] + it ) % tp->T; 
            double _Complex const zbuffer = tp->c[i_diag][tt][0][0] * ephase;
            fprintf ( ofs, "%4d %25.16e %25.16e\n", tt, creal ( zbuffer ), cimag ( zbuffer ) );
          }

        }  /* end of loop on coherent sources */

#ifdef HAVE_LHPC_AFF
        aff_reader_close ( affr );
#endif
      }  /* end of loop on base sources */

      fclose ( ofs );

    }  /* end of loop on diagrams */

  }  /* end of loop on 2-point functions */

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
    fprintf(stdout, "# [htpp_analyse] %s# [htpp_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [htpp_analyse] %s# [htpp_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
