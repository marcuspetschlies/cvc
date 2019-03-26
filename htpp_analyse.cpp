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
#include "read_input_parser.h"
#include "contractions_io.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "contract_diagrams.h"
#include "twopoint_function_utils.h"
#include "gamma.h"
#include "uwerr.h"

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
  int exitstatus;
  int io_proc = -1;
  char ensemble_name[100] = "c13";
  char filename[100];
  int num_conf = 0, num_src_per_conf = 0;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:S:N:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [htpp_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [htpp_analyse] number of sources per config = %d\n", num_src_per_conf );
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


  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf);
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[htpp_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 5 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[htpp_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [htpp_analyse] comment %s\n", line );
      continue;
    }


    /***********************************************************
     * QLUA source coords files have ordering 
     * stream conf x y z t
     ***********************************************************/
    char streamc;
    sscanf( line, "%c %d %d %d %d %d", &streamc,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf],
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+1,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+2,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+3,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+4 );

    count++;
  }

  fclose ( ofs );

  /***********************************************************
   * show all configs and source locations
   ***********************************************************/
  if ( g_verbose > 4 ) {
    fprintf ( stdout, "# [htpp_analyse] conf_src_list conf t x y z\n" );
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        fprintf ( stdout, "  %6d %3d %3d %3d %3d\n", 
            conf_src_list[iconf][isrc][0],
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4] );
      }
    }
  }

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

    twopoint_function_allocate ( tp );

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

      char output_filename[400];
      sprintf ( output_filename, "%s_pfx%dpfy%dpfz%d_gf_%s_dt%d_pi2x%dpi2y%dpi2z%d_gi2_%s_g1_%s_g2_%s_PX%d_PY%d_PZ%d",
          diagram_name, 
          tp->pf1[0], tp->pf1[1], tp->pf1[2], gamma_bin_to_name[tp->gf1[0]], g_src_snk_time_separation, 
          tp->pi2[0], tp->pi2[1], tp->pi2[2], gamma_bin_to_name[tp->gi2], 
          gamma_bin_to_name[tp->gf2], gamma_bin_to_name[tp->gi1[0]],
          tp->pf2[0], tp->pf2[1], tp->pf2[2] );

      if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_analyse] output_filename = %s\n", output_filename );

      FILE * ofs = fopen ( output_filename, "a" );

      /***********************************************************
       * field to store all data
       ***********************************************************/
      int const n_tc = g_src_snk_time_separation + 1;

      double *** corr = init_3level_dtable ( num_conf, num_src_per_conf * g_coherent_source_number, 2 * n_tc );
     
      /***********************************************************
       * loop on configs 
       ***********************************************************/
      for( int iconf = 0; iconf < num_conf; iconf++ ) {
          
          int const Nconf = conf_src_list[iconf][0][0];

        /***********************************************************
         * loop on sources per config
         ***********************************************************/
        for( int isrc = 0; isrc < num_src_per_conf; isrc++) {

          int const t_base = conf_src_list[iconf][isrc][1];

          /***********************************************************
           * store the source coordinates
           ***********************************************************/
          int const gsx[4] = {
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4] };

#ifdef HAVE_LHPC_AFF
          /***********************************************************
           * writer for aff output file
           ***********************************************************/
          /* sprintf ( filename, "%d/contract_3pt_hl_seq.%.4d.tbase%.2d.aff", Nconf, Nconf, t_base ); */
          sprintf ( filename, "%d/contract_3pt_hl_seq.%.4d.t%d_x%d_y%d_z%d.aff", Nconf, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
   
          struct AffReader_s * affr = aff_reader ( filename );
          const char * aff_status_str = aff_reader_errstr ( affr );
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[htpp_analyse] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
            EXIT(5);
          } else {
            if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_analyse] Reading data from file %s\n", filename );
            fflush ( stdout );
          }
#endif

          exitstatus = read_aff_contraction ( (void*)(tp->c[i_diag][0][0]), affr, NULL, key, T * tp->d * tp->d );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[htpp_analyse] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(12);
          }

          for( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {

            /* coherent source timeslice */
            int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * icoh ) % T_global;

            int const csx[4] = { t_coherent ,
                               ( gsx[1] + (LX_global / g_coherent_source_number ) * icoh) % LX_global,
                               ( gsx[2] + (LY_global / g_coherent_source_number ) * icoh) % LY_global,
                               ( gsx[3] + (LZ_global / g_coherent_source_number ) * icoh) % LZ_global };

            /***********************************************************
             * output key
             ***********************************************************/
            fprintf ( ofs, "# %s/conf%d/t%d_x%d_y%d_z%d\n", key, Nconf, csx[0], csx[1], csx[2], csx[3]  );

            /***********************************************************
             * source phase factor
             ***********************************************************/
            double _Complex const ephase = cexp ( 2. * M_PI * ( 
                  tp->pi1[0] * csx[1] / (double)LX_global 
                + tp->pi1[1] * csx[2] / (double)LY_global 
                + tp->pi1[2] * csx[3] / (double)LZ_global ) * I );
            if ( g_verbose > 4 ) fprintf ( stdout, "# [htpp_analyse] pi1 = %3d %3d %3d csx = %3d %3d %3d  ephase = %16.7e %16.7e\n",
                tp->pi1[0], tp->pi1[1], tp->pi1[2],
                csx[1], csx[2], csx[3],
                creal( ephase ), cimag( ephase ) );

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
            for ( int it = 0; it < n_tc; it++ ) {
              /* order from source */
              int const tt = ( csx[0] + it ) % tp->T; 
              double _Complex const zbuffer = tp->c[i_diag][tt][0][0] * ephase;
              
              corr[iconf][isrc*g_coherent_source_number+icoh][2*it  ] = creal ( zbuffer );
              corr[iconf][isrc*g_coherent_source_number+icoh][2*it+1] = cimag ( zbuffer );
            }

            for ( int it = 0; it < n_tc; it++ ) {
              int const tt = ( csx[0] + it ) % tp->T; 
              fprintf ( ofs, "%4d %25.16e %25.16e\n", tt,
                  corr[iconf][isrc*g_coherent_source_number+icoh][2*it  ],
                  corr[iconf][isrc*g_coherent_source_number+icoh][2*it+1] );
            } 

            fflush ( ofs );

          }  /* end of loop on coherent sources */

#if 0
#endif  /* of if 0 */


#ifdef HAVE_LHPC_AFF
          aff_reader_close ( affr );
#endif
        }  /* end of loop on base sources */

      }  /* end of loop on configs */
      

      /***************************************************************************
       ***************************************************************************
       **
       ** UWerr statistical analysis for corr
       **
       ***************************************************************************
       ***************************************************************************/
      double *** res = init_3level_dtable ( 2, n_tc, 5 );

      uwerr ustat;
      for ( int it = 0; it < 2*n_tc; it++ )
      {
          uwerr_init ( &ustat );
    
          ustat.nalpha   = 2 * n_tc;  /* real and imaginary part */
          ustat.nreplica = 1;
          for (  int i = 0; i < ustat.nreplica; i++) ustat.n_r[i] = num_conf * num_src_per_conf * g_coherent_source_number / ustat.nreplica;
          ustat.s_tau = 1.5;
          sprintf ( ustat.obsname, "corr_t%d", it  );
    
          ustat.ipo = it + 1;
          /* ustat.func  = NULL;
          ustat.dfunc = NULL;
          ustat.para  = NULL; */

          exitstatus = uwerr_analysis ( corr[0][0], &ustat );
          if ( exitstatus == 0 ) {
            res[it%2][it/2][0] = ustat.value;
            res[it%2][it/2][1] = ustat.dvalue;
            res[it%2][it/2][2] = ustat.ddvalue;
            res[it%2][it/2][3] = ustat.tauint;
            res[it%2][it/2][4] = ustat.dtauint;
          } else {
            fprintf ( stderr, "[hptt_analyse] Warning return status from uwerr_analysis was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          }
   
          uwerr_free ( &ustat );
      }  /* end of loop on t */
    
      sprintf ( filename, "%s.uwerr", output_filename );
      FILE * uwerr_ofs = fopen ( filename, "w" );
   
      fprintf ( uwerr_ofs, "# nalpha   = %llu\n", ustat.nalpha );
      fprintf ( uwerr_ofs, "# nreplica = %llu\n", ustat.nreplica );
      for (  int i = 0; i < ustat.nreplica; i++) fprintf( uwerr_ofs, "# nr[%d] = %llu\n", i, ustat.n_r[i] );
      fprintf ( uwerr_ofs, "#\n" );
    
      for ( int it = 0; it < n_tc; it++ )
      {
         fprintf ( uwerr_ofs, "%3d %16.7e %16.7e %16.7e %16.7e %16.7e     %16.7e %16.7e %16.7e %16.7e %16.7e\n",
             it,
             res[0][it][0], res[0][it][1], res[0][it][2], res[0][it][3], res[0][it][4],
             res[1][it][0], res[1][it][1], res[1][it][2], res[1][it][3], res[1][it][4] );
      }
    
      fclose( uwerr_ofs );
      fini_3level_dtable ( &res );
#if 0
#endif  /* of if 0 */
      /***************************************************************************/
      /***************************************************************************/

      /***************************************************************************
       * free the correlator field
       ***************************************************************************/
      fini_3level_dtable ( &corr );

      /***************************************************************************
       * close output file
       ***************************************************************************/
      fclose ( ofs );

    }  /* end of loop on diagrams */

    twopoint_function_fini ( tp );

  }  /* end of loop on 2-point functions */

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/
  fini_3level_itable ( &conf_src_list );

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
