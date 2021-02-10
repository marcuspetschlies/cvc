/****************************************************
 * htpp_compact_twop
 *
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
#include "derived_quantities.h"
#include "cvc_timer.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code form 2-pt function\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  EXIT(0);
}


char const gamma_bin_to_name[16][8] = { "id", "gx", "gy", "gxgy", "gz", "gxgz", "gygz", "gtg5", "gt", "gxgt", "gygt", "gzg5", "gzgt", "gyg5", "gxg5", "g5" };

/***********************************************************
 *
 ***********************************************************/
int main(int argc, char **argv) {
  
  int const gamma_parity_sign[16] = {       1,   -1,   -1,      1,   -1,      1,      1,     -1,    1,     -1,     -1,      1,     -1,      1,      1,   -1 };

  int const gamma_chargeconjugation_sign[16] = {
                                            1,   -1,   -1,     -1,   -1,     -1,     -1,      1,   -1,     -1,     -1,      1,     -1,      1,      1,    1 };

  int const gamma_g5herm_sign[16] = {       1,   -1,   -1,     -1,   -1,     -1,     -1,      1,   -1,     -1,     -1,      1,     -1,      1,      1,    1 };

  int const gamma_timereversal_sign[16] = { 1,    1,    1,      1,    1,      1,      1,      1,   -1,     -1,     -1,     -1,     -1,     -1,     -1,   -1 };
  
  char const reim_str[2][3]  = { "re", "im" };
  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char ensemble_name[100] = "NA";
  char filename[100];
  int num_conf = 0, num_src_per_conf = 0;

  struct timeval ta, tb;
  struct timeval start_time, end_time;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:S:N:E:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [htpp_compact_twop] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [htpp_compact_twop] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [htpp_compact_twop] ensemble name set to %s\n", ensemble_name );
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  /***********************************************************
   * timer for total time
   ***********************************************************/
  gettimeofday ( &start_time, (struct timezone *)NULL );


  /* set the default values */
  if(filename_set==0) strcpy(filename, "twopt.input");
  /* fprintf(stdout, "# [htpp_compact_twop] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [htpp_compact_twop] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [htpp_compact_twop] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [htpp_compact_twop] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[htpp_compact_twop] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[htpp_compact_twop] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[htpp_compact_twop] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [htpp_compact_twop] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[htpp_compact_twop] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[htpp_compact_twop] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [htpp_compact_twop] comment %s\n", line );
      continue;
    }


    /***********************************************************
     * QLUA source coords files have ordering 
     * stream conf x y z t
     ***********************************************************/
    sscanf( line, "%c %d %d %d %d %d",
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf],
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+1,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+2,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+3,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+4,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+5 );

    count++;
  }

  fclose ( ofs );

  /***********************************************************
   * show all configs and source locations
   ***********************************************************/
  if ( g_verbose > 4 ) {
    fprintf ( stdout, "# [htpp_compact_twop] conf_src_list conf t x y z\n" );
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        fprintf ( stdout, "%c  %6d %3d %3d %3d %3d\n", 
            conf_src_list[iconf][isrc][0],
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4],
            conf_src_list[iconf][isrc][5] );
      }
    }
  }

  /***********************************************************
   * field to store all data
   ***********************************************************/
  int const n_tc = T_global;
  int const num_meas_per_conf = num_src_per_conf * g_coherent_source_number;
  
  double * corr = init_1level_dtable (2 * n_tc );
  if ( corr == NULL ) {
    fprintf( stderr, "[htpp_compact_twop] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(2);
  }
     
  double * corr_buffer = init_1level_ztable ( T_global );
  if ( corr_buffer == NULL ) {
    fprintf( stderr, "[htpp_compact_twop] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(2);
  }

  /***********************************************************
   * ORIGINAL DATA INPUT
   *
   * loop on configs 
   ***********************************************************/
  for( int iconf = 0; iconf < num_conf; iconf++ ) {
          
    /***********************************************************
     * loop on sources per config
     ***********************************************************/
    for( int isrc = 0; isrc < num_src_per_conf; isrc++) {

      char const stream = conf_src_list[iconf][isrc][0];
      int const Nconf   = conf_src_list[iconf][isrc][1];

      /***********************************************************
       * store the source coordinates
       ***********************************************************/
      int const gsx[4] = {
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4],
            conf_src_list[iconf][isrc][5] };

#ifdef HAVE_LHPC_AFF
      /***********************************************************
       * reader for aff output file
       ***********************************************************/
      if ( ( strcmp ( g_twopoint_function_list[0].type , "m-m" ) == 0 )) {
        sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.t%d_x%d_y%d_z%d.aff", filename_prefix, stream, Nconf, filename_prefix2, Nconf, 
            gsx[0], gsx[1], gsx[2], gsx[3] );
      } else {
        continue;
      }
   
      struct AffReader_s * affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[htpp_compact_twop] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
        EXIT(5);
      } else {
        if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact_twop] Reading data from file %s\n", filename );
      }
#endif
      
      for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

        int pf[3] = {
              g_sink_momentum_list[ipf][0],
              g_sink_momentum_list[ipf][1],
              g_sink_momentum_list[ipf][2] };

        int pi2[3] = { -pf[0], -pf[1], -pf[2]};

        sprintf ( filename, "%s.px%d_py%d_pz%d.h5", g_outfile_prefix, pf[0], pf[1], pf[2] );

        /***************************************************************************
         * loop on twopoint functions
         ***************************************************************************/
        for ( int i_2pt = 0; i_2pt < g_twopoint_function_number; i_2pt++ ) {

          twopoint_function_type * tp = &(g_twopoint_function_list[i_2pt]);

          /* twopoint_function_allocate ( tp ); */

          char diagram_name[500];
          char key[500];

          twopoint_function_get_diagram_name ( diagram_name,  tp, i_diag );
        
          for ( int icoh = 0; ico < g_coherent_source_number; icoh++ ) {
          
            int const csx[4] = { ( gsx[0] + ( T_global  / g_coherent_source_number ) * icoh ) % T_global,
                                 ( gsx[1] + ( LX_global / g_coherent_source_number ) * icoh ) % LX_global,
                                 ( gsx[2] + ( LY_global / g_coherent_source_number ) * icoh ) % LY_global,
                                 ( gsx[3] + ( LZ_global / g_coherent_source_number ) * icoh ) % LZ_global };
            
            if ( strcmp ( tp->type , "m-m" ) == 0 ) {
              sprintf ( key,
                    /* /fs-fc/t70x02y11z20/gf02_gi02/PX0_PY0_PZ0 */
                    "/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d_gi%.2d/PX%d_PY%d_PZ%d",
                    diagram_name, csx[0], csx[1], csx[2], csx[3],
                    tp->gf1[0], tp->gi1[0],
                    pf[0], pf[1], pf[2] );
            } else {
              continue;
            }

            if ( g_verbose > 2 ) {
              fprintf ( stdout, "# [htpp_compact_twop] key = %s %s %d\n", key , __FILE__, __LINE__ );
            }
        
            gettimeofday ( &ta, (struct timezone *)NULL );

            exitstatus = read_aff_contraction ( corr_buffer, affr, NULL, key, T_global );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[htpp_compact_twop] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(12);
            }

            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "htpp_compact_twop", "read_aff_contraction", io_proc == 2 );

            /***********************************************************
             * source phase factor
             ***********************************************************/
            double _Complex const ephase = cexp ( 2. * M_PI * ( 
                          pi1[0] * csx[1] / (double)LX_global 
                        + pi1[1] * csx[2] / (double)LY_global 
                        + pi1[2] * csx[3] / (double)LZ_global ) * I );
            
            if ( g_verbose > 4 ) fprintf ( stdout, "# [htpp_compact_twop] pi1 = %3d %3d %3d csx = %3d %3d %3d  ephase = %16.7e %16.7e\n",
                        pi1[0], pi1[1], pi1[2],
                        csx[1], csx[2], csx[3],
                        creal( ephase ), cimag( ephase ) );

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
            for ( int it = 0; it < n_tc; it++ ) {
              /* order from source */
              int const tt = ( csx[0] + it ) % tp->T; 
              double _Complex const zbuffer = corr_buffer[tt] * ephase;
              
              corr[2*it  ] = creal ( zbuffer );
              corr[2*it+1] = cimag ( zbuffer );

            }


            int const cdim[1] = { 2 * n_tc };
                 
            if ( strcmp ( tp->type , "m-j-m" ) == 0 ) {
              sprintf ( key, "/%s/gf_%s/gi_%s/s%c/c%d/t%dx%dy%dz%d",
                        tp->name,
                        gamma_bin_to_name[tp->gf1[0]],
                        gamma_bin_to_name[tp->gi1[0]],
                        stream, Nconf, csx[0], csx[1], csx[2], csx[3] );

            }

            exitstatus = write_h5_contraction ( corr, NULL, filename, key, "double", 1, cdim );
            if ( exitstatus != 0) {
              fprintf ( stderr, "[http_compact] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(123);
            }

          }  /* end of loop on coherent sources */
        
          twopoint_function_fini ( tp );
        }  /* end of loop on twopoint functions */
      }  /* end of loop on sink momenta */


#ifdef HAVE_LHPC_AFF
      aff_reader_close ( affr );
#endif
    }  /* end of loop on base sources */

  }  /* end of loop on configs */
   
  /***************************************************************************/
  /***************************************************************************/
#if 0
  /***************************************************************************
   * COMPACT OUTPUT
   *
   * write to h5 file
   ***************************************************************************/
  for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

    int ptot[3] = {
        g_total_momentum_list[iptot][0],
        g_total_momentum_list[iptot][1],
        g_total_momentum_list[iptot][2] };



    for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

      int pf[3] = {
          g_sink_momentum_list[ipf][0],
          g_sink_momentum_list[ipf][1],
          g_sink_momentum_list[ipf][2] };

      int pc[3] = {
          ptot[0] - pf[0],
          ptot[1] - pf[1],
          ptot[2] - pf[2] };


      for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {

        int pi1[3] = {
            g_source_momentum_list[ipi][0],
            g_source_momentum_list[ipi][1],
            g_source_momentum_list[ipi][2] };

        int pi2[3] = {
            -( ptot[0] + pi1[0] ),
            -( ptot[1] + pi1[1] ),
            -( ptot[2] + pi1[2] )};

        for ( int i_2pt = 0; i_2pt < g_twopoint_function_number; i_2pt++ ) {

          twopoint_function_type * tp = &(g_twopoint_function_list[i_2pt]);

          /***********************************************************
           * set twop function momenta an filter
           ***********************************************************/
          memcpy( tp->pi1, pi1, 3*sizeof(int) );
          memcpy( tp->pi2, pi2, 3*sizeof(int) );
          memcpy( tp->pf1, pf,  3*sizeof(int) );
          memcpy( tp->pf2, pc,  3*sizeof(int) );

          if ( ! momentum_filter ( tp ) ) continue;


          /***************************************************************************
           * key prefix
           ***************************************************************************/
          char key_prefix[400];

          if ( strcmp ( tp->type , "m-j-m" ) == 0 ) {
            sprintf ( key,
                "/%s/pfx%dpfy%dpfz%d/gc_%s/gi_%s",
                tp->name,
                pf[0], pf[1], pf[2],
                gamma_bin_to_name[tp->gf2],
                gamma_bin_to_name[tp->gi1[0]] );

          } else if ( strcmp ( tp->type , "mxm-j-m" ) == 0 ) {

            sprintf ( key,
                "/%s/pfx%dpfy%dpfz%d/pi1x%dpi1y%dpi1z%d/gc_%s"
                tp->name,
                pf[0], pf[1], pf[2],
                pi1[0], pi1[1], pi1[2],
                gamma_bin_to_name[tp->gf2] );

          } else {
            continue;
          } 

          if ( g_verbose > 2 ) {
            fprintf ( stdout, "# [htpp_compact_twop] h5 key = %s %s %d\n", key , __FILE__, __LINE__ );
          }

          /***********************************************************
           * loop on configs 
           ***********************************************************/
          for( int iconf = 0; iconf < num_conf; iconf++ ) {

            /***********************************************************
             * loop on sources per config
             ***********************************************************/
            for( int isrc = 0; isrc < num_src_per_conf; isrc++) {

              char const stream = conf_src_list[iconf][isrc][0];
              int const Nconf   = conf_src_list[iconf][isrc][1];
              int const t_base  = conf_src_list[iconf][isrc][2];

              /***********************************************************
               * store the source coordinates
               ***********************************************************/
              int const gsx[4] = {
                  conf_src_list[iconf][isrc][2],
                  conf_src_list[iconf][isrc][3],
                  conf_src_list[iconf][isrc][4],
                  conf_src_list[iconf][isrc][5] };

              for( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {

                /* coherent source timeslice */
                int t_coherent = ( t_base + ( T_global / g_coherent_source_number ) * icoh ) % T_global;

                int const csx[4] = { t_coherent ,
                                   ( gsx[1] + (LX_global / g_coherent_source_number ) * icoh) % LX_global,
                                   ( gsx[2] + (LY_global / g_coherent_source_number ) * icoh) % LY_global,
                                   ( gsx[3] + (LZ_global / g_coherent_source_number ) * icoh) % LZ_global };

                char key[500];

              }  /* end of loop on coherent sources */

            }  /* end of loop on base sources */
         
          }  /* end of loop on configs */

        }  /* end of loop on 2pt functions */
      }  /* end if loop on source momentum / pi1 */
    }  /* end of loop on sink momentum */
  }  /* end of loop on total momentum */
#endif

  /***************************************************************************
   * free the correlator field
   ***************************************************************************/
  fini_1level_dtable ( &corr );
  fini_1level_ztable ( &corr_buffer );

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

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "htpp_compact_twop", "total-time", io_proc == 2 );

  return(0);

}
