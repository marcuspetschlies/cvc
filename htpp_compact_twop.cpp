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
#include "cvc_timer.h"

#ifndef TIMERS
#define TIMERS 0
#endif

#ifndef M_M
#define M_M 0
#endif

#ifndef MXM_M
#define MXM_M 0
#endif

#ifndef MXM_MXM
#define MXM_MXM 0
#endif

#ifndef MXM_MXM_OET
#define MXM_MXM_OET 1
#endif

using namespace cvc;

void usage() {
  fprintf(stdout, "Code form 2-pt function\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  EXIT(0);
}


char const gamma_bin_to_name[16][8] = { "id", "gx", "gy", "gxgy", "gz", "gxgz", "gygz", "gtg5", "gt", "gxgt", "gygt", "gzg5", "gzgt", "gyg5", "gxg5", "g5" };

/***************************************************************************
 *
 ***************************************************************************/
int main(int argc, char **argv) {
  
  int const max_single_particle_momentum_squared = 3;


  int const gamma_v_number = 4;
  int const gamma_v_list[4] = { 1, 2, 4, 8  };

  int const gamma_rho_number = 6;
  int const gamma_rho_list[6] = { 1, 2, 4, 9, 10, 12 };
  
  int const gamma_a_number = 4;
  int const gamma_a_list[4] = { 14, 13, 11, 7 };

  int const gamma_p_number = 1;
  int const gamma_p_list[1] = { 15 };


  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char ensemble_name[100] = "NA";
  char filename[100];
  int num_conf = 0, num_src_per_conf = 0;
  char diagram_name[100], correlator_name[100];

  struct timeval ta, tb;
  struct timeval start_time, end_time;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:S:N:E:D:C:")) != -1) {
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
    case 'D':
      strcpy ( diagram_name, optarg );
      fprintf ( stdout, "# [htpp_compact_twop] diagram_name set to %s\n", diagram_name );
      break;
    case 'C':
      strcpy ( correlator_name, optarg );
      fprintf ( stdout, "# [htpp_compact_twop] correlator_name set to %s\n", correlator_name );
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

  int * momentum_number = init_1level_itable ( g_total_momentum_number );
  int **** momentum_list = (int****) malloc ( g_total_momentum_number * sizeof ( int*** ) );

  double _Complex * corr_buffer = init_1level_ztable ( T_global );
  if ( corr_buffer == NULL ) {
    fprintf( stderr, "[htpp_compact_twop] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(2);
  }

#if M_M
  int const n_tc = T_global;

  double ***** corr_m_m = (double*****) malloc ( g_total_momentum_number * sizeof( double****) );

  /***********************************************************
   * loop on configs 
   ***********************************************************/
  for( int iconf = 0; iconf < num_conf; iconf++ ) {
 
    char const stream = conf_src_list[iconf][0][0];
    int const Nconf   = conf_src_list[iconf][0][1];
          
    /***********************************************************
     * loop on sources per config
     ***********************************************************/
    for( int isrc = 0; isrc < num_src_per_conf; isrc++) {

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
      // sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.t%d_x%d_y%d_z%d.aff", filename_prefix, stream, Nconf, filename_prefix2, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
      sprintf ( filename, "%s/stream_%c/%d/%s.aff", filename_prefix, stream, Nconf, filename_prefix2 );
   
      struct AffReader_s * affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[htpp_compact_twop] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
        EXIT(5);
      } else {
        if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact_twop] Reading data from file %s\n", filename );
      }
#endif

      /***********************************************************
       * loop on source momenta
       ***********************************************************/
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int ptot[3] = {
                 g_total_momentum_list[iptot][0],
                 g_total_momentum_list[iptot][1],
                 g_total_momentum_list[iptot][2] };

        int pf[3] = {
            ptot[0],
            ptot[1],
            ptot[2] };

        int pi1[3] = {
            -ptot[0],
            -ptot[1],
            -ptot[2] };

        corr_m_m[iptot] = init_4level_dtable ( g_coherent_source_number, g_source_gamma_id_number,  g_sink_gamma_id_number, 2 * n_tc );
        if ( corr_m_m[iptot] == NULL ) {
          fprintf( stderr, "[htpp_compact_twop] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(2);
        }

        /***************************************************************************
         * loop on rho components
         ***************************************************************************/
        for ( int igi = 0; igi < g_source_gamma_id_number; igi++ ) {

          /***************************************************************************
           * loop on vector components
           ***************************************************************************/
          for ( int igf = 0; igf < g_sink_gamma_id_number; igf++ ) {

            for( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {
 
#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif
              /* coherent source coords */
  
              int const csx[4] = {
                             ( gsx[0] + ( T_global / g_coherent_source_number ) * icoh ) % T_global,
                             ( gsx[1] + (LX_global / g_coherent_source_number ) * icoh ) % LX_global,
                             ( gsx[2] + (LY_global / g_coherent_source_number ) * icoh ) % LY_global,
                             ( gsx[3] + (LZ_global / g_coherent_source_number ) * icoh ) % LZ_global };


              /***********************************************************
               * aff key for reading data
               ***********************************************************/

              char key[500];
#if 0
              sprintf ( key,
                        "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/g1_%s/g2_%s/PX%d_PY%d_PZ%d",
                        /* "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/g1_%s/g2_%s/x%d_y%d_z%d", */
                        diagram_name,
                        pf[0], pf[1], pf[2],
                        gamma_bin_to_name[gamma_p_list[0]], g_src_snk_time_separation,
                        gamma_bin_to_name[gamma_v_list[iv]], gamma_bin_to_name[gamma_rho_list[irho]],
                        pc[0], pc[1], pc[2] );


                  sprintf ( key,
                        "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/pi2x%dpi2y%dpi2z%d/gi2_%s/g1_%s/g2_%s/PX%d_PY%d_PZ%d",
                        /* "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/pi2x%dpi2y%dpi2z%d/gi2_%s/g1_%s/g2_%s/x%d_y%d_z%d", */
                        diagram_name,
                        pf[0], pf[1], pf[2],
                        gamma_bin_to_name[tp->gf1[0]], g_src_snk_time_separation,
                        pi2[0], pi2[1], pi2[2], gamma_bin_to_name[tp->gi2],
                        gamma_bin_to_name[tp->gf2], gamma_bin_to_name[tp->gi1[0]],
                        pc[0], pc[1], pc[2] );

#endif
              sprintf ( key,
                        "/%s/gf%.2d_gi%.2d/px%.2dpy%.2dpz%.2d/x%.2dy%.2dz%.2dt%.2d",
                        diagram_name, g_sink_gamma_id_list[igf], g_source_gamma_id_list[igi],
                        pf[0], pf[1], pf[2],
                        csx[1], csx[2], csx[3], csx[0] );

              if ( g_verbose > 2 ) {
                fprintf ( stdout, "# [htpp_compact_twop] key = %s %s %d\n", key , __FILE__, __LINE__ );
              }

              /***********************************************************/
              /***********************************************************/


              exitstatus = read_aff_contraction (  corr_buffer, affr, NULL, key, T_global );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[htpp_compact_twop] Error from read_aff_contraction for file %s key %s, status was %d %s %d\n", 
                    filename, key, exitstatus, __FILE__, __LINE__ );
                EXIT(12);
              }

#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact_twop", "key-read_aff_contraction", io_proc == 2 );
#endif


                /***********************************************************
                 * source phase factor
                 * and order from source
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
                  /* it = time relative to source
                   * tt = lattice timeslice
                   */
                  int const tt = ( csx[0] + it ) % T_global; 
                  double _Complex const zbuffer = corr_buffer[tt] * ephase;
              
                  corr_m_m[iptot][icoh][igi][igf][2*it  ] = creal ( zbuffer );
                  corr_m_m[iptot][icoh][igi][igf][2*it+1] = cimag ( zbuffer );

                }

#if TIMERS
                gettimeofday ( &tb, (struct timezone *)NULL );
                show_time ( &ta, &tb, "htpp_compact_twop", "source-phase-and-reorder", io_proc == 2 );
#endif
              }  /* coherent sources */
          }  /* gf */
        }  /* gi */

      } /* total momenta */

#ifdef HAVE_LHPC_AFF
      aff_reader_close ( affr );
#endif

      /***************************************************************************
       * write to h5 file
       ***************************************************************************/
            
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int ptot[3] = {
                 g_total_momentum_list[iptot][0],
                 g_total_momentum_list[iptot][1],
                 g_total_momentum_list[iptot][2] };

        /***********************************************************
         * output filename
         ***********************************************************/
        sprintf ( filename, "%s.px%d_py%d_pz%d.h5", correlator_name, ptot[0], ptot[1], ptot[2] );

        if ( iconf == 0 && isrc == 0 ) {

          char gamma_tag[200];
          sprintf( gamma_tag, "/gamma_source" );
          exitstatus = write_h5_contraction ( (void*)(g_source_gamma_id_list), NULL, filename, gamma_tag, "int", 1, &g_source_gamma_id_number );
          if( exitstatus != 0 ) {
            fprintf ( stderr, "[htpp_compact_twop] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
  
          sprintf( gamma_tag, "/gamma_sink" );
          exitstatus = write_h5_contraction ( (void*)(g_sink_gamma_id_list), NULL, filename, gamma_tag, "int", 1, &g_sink_gamma_id_number );
          if( exitstatus != 0 ) {
            fprintf ( stderr, "[htpp_compact_twop] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
        }

#if TIMERS
        gettimeofday ( &ta, (struct timezone *)NULL );
#endif
        for( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {

          int const csx[4] = {
              ( gsx[0] + ( T_global / g_coherent_source_number ) * icoh ) % T_global,
              ( gsx[1] + (LX_global / g_coherent_source_number ) * icoh ) % LX_global,
              ( gsx[2] + (LY_global / g_coherent_source_number ) * icoh ) % LY_global,
              ( gsx[3] + (LZ_global / g_coherent_source_number ) * icoh ) % LZ_global };
 
          char key[500];

          sprintf ( key, "/s%c/c%d/t%dx%dy%dz%d", stream, Nconf, csx[0], csx[1], csx[2], csx[3] );

#if 0
                    sprintf ( key, "/%s/pfx%dpfy%dpfz%d/pi1x%dpi1y%dpi1z%d/gc_%s/s%c/c%d/t%dx%dy%dz%d",
                        tp->name,
                        pf[0], pf[1], pf[2],
                        pi1[0], pi1[1], pi1[2],
                        gamma_bin_to_name[tp->gf2],
                        stream, Nconf, csx[0], csx[1], csx[2], csx[3] );
#endif

          if ( g_verbose > 2 ) {
            fprintf ( stdout, "# [htpp_compact_twop] h5 key = %s %s %d\n", key , __FILE__, __LINE__ );
          }

          int const ndim = 3;
          int const cdim[3] = { g_source_gamma_id_number, g_sink_gamma_id_number, 2 * n_tc };

          exitstatus = write_h5_contraction ( corr_m_m[iptot][icoh][0][0], NULL, filename, key, "double", ndim, cdim );
          if ( exitstatus != 0) {
            fprintf ( stderr, "[http_compact] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(123);
          }
#if TIMERS
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "htpp_compact_twop", "write_h5_contraction", io_proc == 2 );
#endif
        }  /* end of loop on coherent sources */

        fini_4level_dtable ( &( corr_m_m[iptot] ) );

      }  /* end of loop on sink momenta */

    }  /* end of loop on base sources */
  }  /* end of loop on configs */
  free ( corr_m_m );
#endif  /* of if M_M */


#if MXM_M
  int const n_tc = T_global;

  double ***** corr_mxm_m = (double*****) malloc ( g_total_momentum_number * sizeof( double****) );

  /***********************************************************
   * loop on configs 
   ***********************************************************/
  for( int iconf = 0; iconf < num_conf; iconf++ ) {

    struct timeval conf_stime, conf_etime;

    gettimeofday ( &conf_stime, (struct timezone *)NULL );
 
    char const stream = conf_src_list[iconf][0][0];
    int const Nconf   = conf_src_list[iconf][0][1];
          
    /***********************************************************
     * loop on sources per config
     ***********************************************************/
    for( int isrc = 0; isrc < num_src_per_conf; isrc++) {

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
      // sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.t%d_x%d_y%d_z%d.aff", filename_prefix, stream, Nconf, filename_prefix2, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
      sprintf ( filename, "%s/stream_%c/%d/%s.aff", filename_prefix, stream, Nconf, filename_prefix2 );
   
      struct AffReader_s * affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[htpp_compact_twop] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
        EXIT(5);
      } else {
        if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact_twop] Reading data from file %s\n", filename );
      }
#endif

      /***********************************************************
       * loop on source momenta
       ***********************************************************/
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int ptot[3] = {
                 g_total_momentum_list[iptot][0],
                 g_total_momentum_list[iptot][1],
                 g_total_momentum_list[iptot][2] };

        if( iconf == 0 && isrc == 0 ) {

          momentum_number[iptot] = 0;

          /***********************************************************
           * loop on sink momenta
           ***********************************************************/

            int pf[3] = {
                ptot[0],
                ptot[1],
                ptot[2] };

            for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {

              int pi1[3] = {
                g_source_momentum_list[ipi][0],
                g_source_momentum_list[ipi][1],
                g_source_momentum_list[ipi][2] };

              int pi2[3] = {
                  -ptot[0] - pi1[0],
                  -ptot[1] - pi1[1],
                  -ptot[2] - pi1[2] };

              if ( momentum_filter ( pf, NULL, pi1, pi2 , max_single_particle_momentum_squared ) ) {
                momentum_number[iptot]++;
                if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_compact_twop] mom pi1 (%d %d %d) pi2 (%d %d %d) pf (%d %d %d) passed\n",
                    pi1[0], pi1[1], pi1[2],
                    pi2[0], pi2[1], pi2[2],
                    pf[0], pf[1], pf[2]  );
                    
              } else {
                if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_compact_twop] mom pi1 (%d %d %d) pi2 (%d %d %d) pf (%d %d %d) filtered\n",
                    pi1[0], pi1[1], pi1[2],
                    pi2[0], pi2[1], pi2[2],
                    pf[0], pf[1], pf[2]  );
              }
            }
 
          fprintf ( stdout, "# [htpp_compact_twop] number of momentum combinations = %d %s %d\n", momentum_number[iptot], __FILE__, __LINE__ );
          momentum_list[iptot] = init_3level_itable ( momentum_number[iptot], 2, 3 );
        }

        corr_mxm_m[iptot] = init_4level_dtable ( g_coherent_source_number, g_sink_gamma_id_number, momentum_number[iptot],  2 * n_tc );
        if ( corr_mxm_m[iptot] == NULL ) {
          fprintf( stderr, "[htpp_compact_twop] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(2);
        }

        /***************************************************************************
         * loop on vector components
         ***************************************************************************/
        for ( int iv = 0; iv < g_sink_gamma_id_number; iv++ ) {

           int momentum_config_counter = -1;
            /***********************************************************
             * loop on sink momenta
             ***********************************************************/
              int pf[3] = {
                ptot[0],
                ptot[1],
                ptot[2]
              };
  
            for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {

              int pi1[3] = {
                g_source_momentum_list[ipi][0],
                g_source_momentum_list[ipi][1],
                g_source_momentum_list[ipi][2] };

              int pi2[3] = {
                  -ptot[0] - pi1[0],
                  -ptot[1] - pi1[1],
                  -ptot[2] - pi1[2] };
#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif
              /***********************************************************
               * set twop function momenta an filter
               ***********************************************************/
              if ( ! momentum_filter ( pf, NULL, pi1, pi2 , max_single_particle_momentum_squared ) ) continue;
              momentum_config_counter++;

              /***********************************************************
               * add momentum config to list
               ***********************************************************/
              if ( iconf == 0 && isrc == 0 && iv == 0 ) {
                memcpy ( momentum_list[iptot][momentum_config_counter][0], pf, 3*sizeof(int) );
                memcpy ( momentum_list[iptot][momentum_config_counter][1], pi1, 3*sizeof(int) );
              }

              for( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {
 
#if TIMERS
                gettimeofday ( &ta, (struct timezone *)NULL );
#endif
                /* coherent source coords */
  
                int const csx[4] = {
                               ( gsx[0] + ( T_global / g_coherent_source_number ) * icoh ) % T_global,
                               ( gsx[1] + (LX_global / g_coherent_source_number ) * icoh ) % LX_global,
                               ( gsx[2] + (LY_global / g_coherent_source_number ) * icoh ) % LY_global,
                               ( gsx[3] + (LZ_global / g_coherent_source_number ) * icoh ) % LZ_global };
                
                /***********************************************************
                 * aff key for reading data
                 ***********************************************************/

                char key[500];

                sprintf ( key,
                        "/%s/g%.2d/px%.2dpy%.2dpz%.2d/kx%.2dky%.2dkz%.2d/x%.2dy%.2dz%.2dt%.2d",
                        /* "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/pi2x%dpi2y%dpi2z%d/gi2_%s/g1_%s/g2_%s/x%d_y%d_z%d", */
                        diagram_name,
                        g_sink_gamma_id_list[iv], 
                        pf[0], pf[1], pf[2],
                        pi2[0], pi2[1], pi2[2],
                        csx[1], csx[2], csx[3], csx[0] );

#if 0
              sprintf ( key,
                        "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/pi2x%dpi2y%dpi2z%d/gi2_%s/g1_%s/g2_%s/PX%d_PY%d_PZ%d",
                        /* "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/pi2x%dpi2y%dpi2z%d/gi2_%s/g1_%s/g2_%s/x%d_y%d_z%d", */
                        diagram_name,
                        pf[0], pf[1], pf[2],
                        gamma_bin_to_name[gamma_p_list[0]], g_src_snk_time_separation,
                        pi2[0], pi2[1], pi2[2], gamma_bin_to_name[gamma_p_list[0]],
                        gamma_bin_to_name[gamma_v_list[0]], gamma_bin_to_name[gamma_p_list[0]],
                        pc[0], pc[1], pc[2] );
#endif
              if ( g_verbose > 2 ) {
                fprintf ( stdout, "# [htpp_compact_twop] key = %s %s %d\n", key , __FILE__, __LINE__ );
              }
#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact_twop", "momentum-filter-and-key", io_proc == 2 );
#endif
              /***********************************************************/
              /***********************************************************/
#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif

              exitstatus = read_aff_contraction (  corr_buffer, affr, NULL, key, T_global );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[htpp_compact_twop] Error from read_aff_contraction for file %s key %s, status was %d %s %d\n",
                    filename, key, exitstatus, __FILE__, __LINE__ );
                EXIT(12);
              }

#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact_twop", "read_aff_contraction", io_proc == 2 );
#endif


#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif
                /***********************************************************
                 * source phase factor
                 * and order from source
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
                  /* it = time relative to source
                   * tt = lattice timeslice
                   */
                  int const tt = ( csx[0] + it ) % T_global; 
                  double _Complex const zbuffer = corr_buffer[tt] * ephase;
              
                  corr_mxm_m[iptot][icoh][iv][momentum_config_counter][2*it  ] = creal ( zbuffer );
                  corr_mxm_m[iptot][icoh][iv][momentum_config_counter][2*it+1] = cimag ( zbuffer );

                }

#if TIMERS
                gettimeofday ( &tb, (struct timezone *)NULL );
                show_time ( &ta, &tb, "htpp_compact_twop", "source-phase-and-reorder", io_proc == 2 );
#endif
              }  /* coherent sources */
            }  /* end of loop on pi1 */
        }  /* vector insertion */

      } /* total momenta */

#ifdef HAVE_LHPC_AFF
      aff_reader_close ( affr );
#endif

      /***************************************************************************
       * write to h5 file
       ***************************************************************************/
            
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int ptot[3] = {
                 g_total_momentum_list[iptot][0],
                 g_total_momentum_list[iptot][1],
                 g_total_momentum_list[iptot][2] };

        /***********************************************************
         * output filename
         ***********************************************************/
        sprintf ( filename, "%s.px%d_py%d_pz%d.h5", correlator_name, ptot[0], ptot[1], ptot[2] );

        if ( iconf == 0 && isrc == 0 ) {
          /***************************************************************************
           * write momentum list to h5 file
           ***************************************************************************/
          int const momentum_ndim = 3;
          int const momentum_cdim[3] = {momentum_number[iptot], 2, 3 };
          char momentum_tag[200];
          sprintf( momentum_tag, "/mom" );
          exitstatus = write_h5_contraction ( (void*)(momentum_list[iptot][0][0]), NULL, filename, momentum_tag, "int", momentum_ndim, momentum_cdim );
          if( exitstatus != 0 ) {
            fprintf ( stderr, "[htpp_compact_twop] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }

          char gamma_tag[200];
          sprintf( gamma_tag, "/gamma_sink" );
          exitstatus = write_h5_contraction ( (void*)(g_sink_gamma_id_list), NULL, filename, gamma_tag, "int", 1, &g_sink_gamma_id_number );
          if( exitstatus != 0 ) {
            fprintf ( stderr, "[htpp_compact_twop] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
        }

#if TIMERS
        gettimeofday ( &ta, (struct timezone *)NULL );
#endif
        for( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {

          int const csx[4] = {
              ( gsx[0] + ( T_global / g_coherent_source_number ) * icoh ) % T_global,
              ( gsx[1] + (LX_global / g_coherent_source_number ) * icoh ) % LX_global,
              ( gsx[2] + (LY_global / g_coherent_source_number ) * icoh ) % LY_global,
              ( gsx[3] + (LZ_global / g_coherent_source_number ) * icoh ) % LZ_global };
 
          char key[500];

          sprintf ( key, "/s%c/c%d/t%dx%dy%dz%d", stream, Nconf, csx[0], csx[1], csx[2], csx[3] );

          if ( g_verbose > 2 ) {
            fprintf ( stdout, "# [htpp_compact_twop] h5 key = %s %s %d\n", key , __FILE__, __LINE__ );
          }

          int const ndim = 3;
          int const cdim[3] = { g_sink_gamma_id_number, momentum_number[iptot],  2 * n_tc };

          exitstatus = write_h5_contraction ( corr_mxm_m[iptot][icoh][0][0], NULL, filename, key, "double", ndim, cdim );
          if ( exitstatus != 0) {
            fprintf ( stderr, "[http_compact] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(123);
          }
#if TIMERS
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "htpp_compact_twop", "write_h5_contraction", io_proc == 2 );
#endif
        }  /* end of loop on coherent sources */

        fini_4level_dtable ( &( corr_mxm_m[iptot] ) );
        fini_3level_itable ( &( momentum_list[iptot] ) );

      }  /* end of loop on sink momenta */

    }  /* end of loop on base sources */

    gettimeofday ( &conf_etime, (struct timezone *)NULL );
    show_time ( &conf_stime, &conf_etime, "htpp_compact_twop", "time-per-conf", io_proc == 2 );
  }  /* end of loop on configs */


  free ( corr_mxm_m );

#endif  /* of if MXM_M */


#if MXM_MXM

  int const n_tc = g_src_snk_time_separation + 1;
  double **** corr_mxm_mxm = (double****) malloc ( g_total_momentum_number * sizeof( double***) );

  /***********************************************************
   * initialize
   ***********************************************************/
  for ( int i = 0; i < g_total_momentum_number; i++ ) {
    momentum_list[i] = NULL;
    corr_mxm_mxm[i] = NULL;
    momentum_number[i] = -1;
  }

  /***********************************************************
   * loop on configs 
   ***********************************************************/
  for( int iconf = 0; iconf < num_conf; iconf++ ) {

    struct timeval conf_stime, conf_etime;

    gettimeofday ( &conf_stime, (struct timezone *)NULL );
 
    char const stream = conf_src_list[iconf][0][0];
    int const Nconf   = conf_src_list[iconf][0][1];
          
    /***********************************************************
     * loop on sources per config
     ***********************************************************/
    for( int isrc = 0; isrc < num_src_per_conf; isrc++) {

      /***********************************************************
       * store the source coordinates
       ***********************************************************/
      int const gsx[4] = {
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4],
            conf_src_list[iconf][isrc][5] };




      for ( int isample = 0; isample < g_nsample; isample++ ) {

#ifdef HAVE_LHPC_AFF
        /***********************************************************
         * reader for aff file
         ***********************************************************/
        sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.%.2d.%.5d" , filename_prefix, stream, Nconf, filename_prefix2, Nconf, gsx[0], isample );
   
        struct AffReader_s * affr = aff_reader ( filename );
        const char * aff_status_str = aff_reader_errstr ( affr );
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[htpp_compact_twop] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
          EXIT(5);
        } else {
          if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact_twop] Reading data from file %s\n", filename );
        }
#else
#error "Need AFF library"
#endif

      /***********************************************************
       * loop on source momenta
       ***********************************************************/
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int ptot[3] = {
                 g_total_momentum_list[iptot][0],
                 g_total_momentum_list[iptot][1],
                 g_total_momentum_list[iptot][2] };

        if( momentum_number[iptot] == -1 ) {

          momentum_number[iptot] = 0;

          /***********************************************************
           * loop on sink momenta
           ***********************************************************/

          for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

            int pf1[3] = {
              g_sink_momentum_list[ipf][0],
              g_sink_momentum_list[ipf][1],
              g_sink_momentum_list[ipf][2] };

            int pf2[3] = {
              ptot[0] - pf1[0],
              ptot[1] - pf1[1],
              ptot[2] - pf1[2] };

            for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {

              int pi1[3] = {
                g_source_momentum_list[ipi][0],
                g_source_momentum_list[ipi][1],
                g_source_momentum_list[ipi][2] };

              int pi2[3] = {
                  -ptot[0] - pi1[0],
                  -ptot[1] - pi1[1],
                  -ptot[2] - pi1[2] };

              if ( momentum_filter ( pf1, pf2, pi1, pi2 , max_single_particle_momentum_squared ) ) {
                momentum_number[iptot]++;
                if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_compact_twop] mom pi1 (%d %d %d) pi2 (%d %d %d) pf1 (%d %d %d) pf2 (%d %d %d) passed\n",
                    pi1[0], pi1[1], pi1[2],
                    pi2[0], pi2[1], pi2[2],
                    pf1[0], pf1[1], pf1[2],
                    pf2[0], pf2[1], pf2[2] );
                    
              } else {
                if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_compact_twop] mom pi1 (%d %d %d) pi2 (%d %d %d) pf1 (%d %d %d) pf2 (%d %d %d) filtered\n",
                    pi1[0], pi1[1], pi1[2],
                    pi2[0], pi2[1], pi2[2],
                    pf1[0], pf1[1], pf1[2],
                    pf2[0], pf2[1], pf2[2] );
              }
            }
 
          }

          fprintf ( stdout, "# [htpp_compact_twop] number of momentum combinations = %d %s %d\n", momentum_number[iptot], __FILE__, __LINE__ );
        
        }  /* end of if not initialized */

        if ( momentum_list[iptot] == NULL ) {
          momentum_list[iptot] = init_3level_itable ( momentum_number[iptot], 2, 3 );
          if ( momentum_list[iptot] == NULL ) {
            fprintf( stderr, "[htpp_compact_twop] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(2);
          }
        }

        if ( corr_mxm_mxm[iptot] == NULL ) {
          corr_mxm_mxm[iptot] = init_3level_dtable ( g_coherent_source_number, momentum_number[iptot],  2 * n_tc );
          if ( corr_mxm_mxm[iptot] == NULL ) {
            fprintf( stderr, "[htpp_compact_twop] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(2);
          }
        }

        int momentum_config_counter = -1;

        /***********************************************************
         * loop on sink momenta
         ***********************************************************/
        for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

          int pf1[3] = {
              g_sink_momentum_list[ipf][0],
              g_sink_momentum_list[ipf][1],
              g_sink_momentum_list[ipf][2] };

          int pf2[3] = {
              ptot[0] - pf1[0],
              ptot[1] - pf1[1],
              ptot[2] - pf1[2] };

          for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {

            int pi1[3] = {
              g_source_momentum_list[ipi][0],
              g_source_momentum_list[ipi][1],
              g_source_momentum_list[ipi][2] };

            int pi2[3] = {
                -ptot[0] - pi1[0],
                -ptot[1] - pi1[1],
                -ptot[2] - pi1[2] };
#if TIMERS
            gettimeofday ( &ta, (struct timezone *)NULL );
#endif
            /***********************************************************
             * set twop function momenta and filter
             ***********************************************************/
            if ( ! momentum_filter ( pf1, pf2, pi1, pi2 , max_single_particle_momentum_squared ) ) continue;
            momentum_config_counter++;

            /***********************************************************
             * add momentum config to list
             ***********************************************************/
            if ( iconf == 0 && isrc == 0 && isample == 0 ) {
              memcpy ( momentum_list[iptot][momentum_config_counter][0], pf1, 3*sizeof(int) );
              memcpy ( momentum_list[iptot][momentum_config_counter][1], pi1, 3*sizeof(int) );
            }
#if TIMERS
            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "htpp_compact_twop", "momentum-filter", io_proc == 2 );
#endif

            for( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {
 
#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif
              /* coherent source coords */
  
              int const csx[4] = {
                               ( gsx[0] + ( T_global / g_coherent_source_number ) * icoh ) % T_global,
                               ( gsx[1] + (LX_global / g_coherent_source_number ) * icoh ) % LX_global,
                               ( gsx[2] + (LY_global / g_coherent_source_number ) * icoh ) % LY_global,
                               ( gsx[3] + (LZ_global / g_coherent_source_number ) * icoh ) % LZ_global };
                
              /***********************************************************
               * aff key for reading data
               ***********************************************************/

              char key[500];

              /* /mxm-mxm/sample00/pi2x-01pi2y-01pi2z-01/pf1x-01pf1y-01pf1z-01/pf2x-01pf2y-01pf2z01/x30y00z21t03 */

              sprintf ( key,
                      "/%s/sample%.2d/pi2x%.2dpi2y%.2dpi2z%.2d/pf1x%.2dpf1y%.2dpf1z%.2d/pf2x%.2dpf2y%.2dpf2z%.2d/x%.2dy%.2dz%.2dt%.2d",
                      diagram_name, isample,
                      pi2[0], pi2[1], pi2[2],
                      pf1[0], pf1[1], pf1[2],
                      pf2[0], pf2[1], pf2[2],
                      csx[1], csx[2], csx[3], csx[0] );

              if ( g_verbose > 2 ) {
                fprintf ( stdout, "# [htpp_compact_twop] key = %s %s %d\n", key , __FILE__, __LINE__ );
              }
#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact_twop", "aff-key", io_proc == 2 );
#endif
              /***********************************************************/
              /***********************************************************/
#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif

              exitstatus = read_aff_contraction (  corr_buffer, affr, NULL, key, T_global );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[htpp_compact_twop] Error from read_aff_contraction for file %s key %s, status was %d %s %d\n",
                    filename, key, exitstatus, __FILE__, __LINE__ );
                EXIT(12);
              }

#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact_twop", "read_aff_contraction", io_proc == 2 );
#endif


#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif
              /***********************************************************
               * source phase factor
               * and order from source
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
                /* it = time relative to source
                 * tt = lattice timeslice
                 */
                /* int const tt = ( csx[0] + it ) % T_global;  */
                int const tt = it;
                double _Complex const zbuffer = corr_buffer[tt] * ephase;
              
                corr_mxm_mxm[iptot][icoh][momentum_config_counter][2*it  ] += creal ( zbuffer );
                corr_mxm_mxm[iptot][icoh][momentum_config_counter][2*it+1] += cimag ( zbuffer );

              }

#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact_twop", "source-phase-and-reorder", io_proc == 2 );
#endif
            }  /* coherent sources */
          }  /* end of loop on pi1 */
        }  /* end of loop on pf1 */
      } /* total momenta */
    
#ifdef HAVE_LHPC_AFF
      aff_reader_close ( affr );
#endif
      }  /* samples */

      /***************************************************************************
       * write to h5 file
       ***************************************************************************/
            
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int ptot[3] = {
                 g_total_momentum_list[iptot][0],
                 g_total_momentum_list[iptot][1],
                 g_total_momentum_list[iptot][2] };

        /***********************************************************
         * output filename
         ***********************************************************/
        sprintf ( filename, "%s.px%d_py%d_pz%d.h5", correlator_name, ptot[0], ptot[1], ptot[2] );

        if ( iconf == 0 && isrc == 0 ) {
          /***************************************************************************
           * write momentum list to h5 file
           ***************************************************************************/
          int const momentum_ndim = 3;
          int const momentum_cdim[3] = {momentum_number[iptot], 2, 3 };
          char momentum_tag[200];
          sprintf( momentum_tag, "/mom" );
          exitstatus = write_h5_contraction ( (void*)(momentum_list[iptot][0][0]), NULL, filename, momentum_tag, "int", momentum_ndim, momentum_cdim );
          if( exitstatus != 0 ) {
            fprintf ( stderr, "[htpp_compact_twop] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }

        }


        /***********************************************************
         * normalize with number of stochastic samples
         ***********************************************************/
        for ( int  i = 0; i < g_coherent_source_number * momentum_number[iptot] * 2 * n_tc ; i++ ) {
          corr_mxm_mxm[iptot][0][0][i] /= g_nsample;
        }

#if TIMERS
        gettimeofday ( &ta, (struct timezone *)NULL );
#endif
        for( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {

          int const csx[4] = {
              ( gsx[0] + ( T_global / g_coherent_source_number ) * icoh ) % T_global,
              ( gsx[1] + (LX_global / g_coherent_source_number ) * icoh ) % LX_global,
              ( gsx[2] + (LY_global / g_coherent_source_number ) * icoh ) % LY_global,
              ( gsx[3] + (LZ_global / g_coherent_source_number ) * icoh ) % LZ_global };
 
          char key[500];

          sprintf ( key, "/s%c/c%d/t%dx%dy%dz%d", stream, Nconf, csx[0], csx[1], csx[2], csx[3] );

          if ( g_verbose > 2 ) {
            fprintf ( stdout, "# [htpp_compact_twop] h5 key = %s %s %d\n", key , __FILE__, __LINE__ );
          }

          int const ndim = 2;
          int const cdim[2] = { momentum_number[iptot],  2 * n_tc };

          exitstatus = write_h5_contraction ( corr_mxm_mxm[iptot][icoh][0], NULL, filename, key, "double", ndim, cdim );
          if ( exitstatus != 0) {
            fprintf ( stderr, "[http_compact] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(123);
          }
#if TIMERS
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "htpp_compact_twop", "write_h5_contraction", io_proc == 2 );
#endif
        }  /* end of loop on coherent sources */

        fini_3level_dtable ( &( corr_mxm_mxm[iptot] ) );
        corr_mxm_mxm[iptot] = NULL;
        fini_3level_itable ( &( momentum_list[iptot] ) );
        momentum_list[iptot] = NULL;

      }  /* end of loop on total momenta */

    }  /* end of loop on base sources */

    gettimeofday ( &conf_etime, (struct timezone *)NULL );
    show_time ( &conf_stime, &conf_etime, "htpp_compact_twop", "time-per-conf", io_proc == 2 );

  }  /* end of loop on configs */


  free ( corr_mxm_mxm );

#endif  /* of if MXM_MXM */









#if MXM_MXM_OET

  double _Complex * corr_buffer_oet = init_1level_ztable ( T_global );
  if ( corr_buffer_oet == NULL ) {
    fprintf( stderr, "[htpp_compact_twop] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(2);
  }

  int const n_tc = g_src_snk_time_separation + 1;
  double **** corr_mxm_mxm = (double****) malloc ( g_total_momentum_number * sizeof( double***) );

  /***********************************************************
   * initialize
   ***********************************************************/
  for ( int i = 0; i < g_total_momentum_number; i++ ) {
    momentum_list[i] = NULL;
    corr_mxm_mxm[i] = NULL;
    momentum_number[i] = -1;
  }

  /***********************************************************
   * loop on configs 
   ***********************************************************/
  for( int iconf = 0; iconf < num_conf; iconf++ ) {

    struct timeval conf_stime, conf_etime;

    gettimeofday ( &conf_stime, (struct timezone *)NULL );
 
    char const stream = conf_src_list[iconf][0][0];
    int const Nconf   = conf_src_list[iconf][0][1];
          
    /***********************************************************
     * loop on sources per config
     ***********************************************************/
    for( int isrc = 0; isrc < num_src_per_conf; isrc++) {

      /***********************************************************
       * store the source coordinates
       ***********************************************************/
      int const gsx[4] = {
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4],
            conf_src_list[iconf][isrc][5] };

      for ( int isample = 0; isample < g_nsample_oet; isample++ ) {

#ifdef HAVE_LHPC_AFF
        /***********************************************************
         * reader for aff file
         ***********************************************************/
        sprintf ( filename, "%s/stream_%c/%d/%s.aff" , filename_prefix, stream, Nconf, filename_prefix2 );
   
        struct AffReader_s * affr = aff_reader ( filename );
        char * aff_status_str = (char*)aff_reader_errstr ( affr );
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[htpp_compact_twop] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
          EXIT(5);
        } else {
          if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact_twop] Reading data from file %s\n", filename );
        }

        sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.t%.2d" , filename_prefix, stream, Nconf, filename_prefix3, Nconf, gsx[0] );
   
        struct AffReader_s * affr_oet = aff_reader ( filename );
        aff_status_str = (char*)aff_reader_errstr ( affr_oet );
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[htpp_compact_twop] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
          EXIT(5);
        } else {
          if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact_twop] Reading data from file %s\n", filename );
        }
#else
#error "Need AFF library"
#endif

      /***********************************************************
       * loop on source momenta
       ***********************************************************/
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int ptot[3] = {
                 g_total_momentum_list[iptot][0],
                 g_total_momentum_list[iptot][1],
                 g_total_momentum_list[iptot][2] };

        if( momentum_number[iptot] == -1 ) {

          momentum_number[iptot] = 0;

          /***********************************************************
           * loop on sink momenta
           ***********************************************************/

          for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

            int pf1[3] = {
              g_sink_momentum_list[ipf][0],
              g_sink_momentum_list[ipf][1],
              g_sink_momentum_list[ipf][2] };

            int pf2[3] = {
              ptot[0] - pf1[0],
              ptot[1] - pf1[1],
              ptot[2] - pf1[2] };

            for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {

              int pi1[3] = {
                g_source_momentum_list[ipi][0],
                g_source_momentum_list[ipi][1],
                g_source_momentum_list[ipi][2] };

              int pi2[3] = {
                  -ptot[0] - pi1[0],
                  -ptot[1] - pi1[1],
                  -ptot[2] - pi1[2] };

              if ( momentum_filter ( pf1, pf2, pi1, pi2 , max_single_particle_momentum_squared ) ) {
                momentum_number[iptot]++;
                if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_compact_twop] mom pi1 (%d %d %d) pi2 (%d %d %d) pf1 (%d %d %d) pf2 (%d %d %d) passed\n",
                    pi1[0], pi1[1], pi1[2],
                    pi2[0], pi2[1], pi2[2],
                    pf1[0], pf1[1], pf1[2],
                    pf2[0], pf2[1], pf2[2] );
                    
              } else {
                if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_compact_twop] mom pi1 (%d %d %d) pi2 (%d %d %d) pf1 (%d %d %d) pf2 (%d %d %d) filtered\n",
                    pi1[0], pi1[1], pi1[2],
                    pi2[0], pi2[1], pi2[2],
                    pf1[0], pf1[1], pf1[2],
                    pf2[0], pf2[1], pf2[2] );
              }
            }
 
          }

          fprintf ( stdout, "# [htpp_compact_twop] number of momentum combinations = %d %s %d\n", momentum_number[iptot], __FILE__, __LINE__ );
        
        }  /* end of if not initialized */

        if ( momentum_list[iptot] == NULL ) {
          momentum_list[iptot] = init_3level_itable ( momentum_number[iptot], 2, 3 );
          if ( momentum_list[iptot] == NULL ) {
            fprintf( stderr, "[htpp_compact_twop] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(2);
          }
        }

        if ( corr_mxm_mxm[iptot]  == NULL ) {
          corr_mxm_mxm[iptot] = init_3level_dtable ( g_coherent_source_number, momentum_number[iptot],  2 * n_tc );
          if ( corr_mxm_mxm[iptot] == NULL ) {
            fprintf( stderr, "[htpp_compact_twop] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(2);
          }
        }

        int momentum_config_counter = -1;

        /***********************************************************
         * loop on sink momenta
         ***********************************************************/
        for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

          int pf1[3] = {
              g_sink_momentum_list[ipf][0],
              g_sink_momentum_list[ipf][1],
              g_sink_momentum_list[ipf][2] };

          int pf2[3] = {
              ptot[0] - pf1[0],
              ptot[1] - pf1[1],
              ptot[2] - pf1[2] };

          for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) {

            int pi1[3] = {
              g_source_momentum_list[ipi][0],
              g_source_momentum_list[ipi][1],
              g_source_momentum_list[ipi][2] };

            int pi2[3] = {
                -ptot[0] - pi1[0],
                -ptot[1] - pi1[1],
                -ptot[2] - pi1[2] };
#if TIMERS
            gettimeofday ( &ta, (struct timezone *)NULL );
#endif
            /***********************************************************
             * set twop function momenta and filter
             ***********************************************************/
            if ( ! momentum_filter ( pf1, pf2, pi1, pi2 , max_single_particle_momentum_squared ) ) continue;
            momentum_config_counter++;

            /***********************************************************
             * add momentum config to list
             ***********************************************************/
            if ( iconf == 0 && isrc == 0 && isample == 0 ) {
              memcpy ( momentum_list[iptot][momentum_config_counter][0], pf1, 3*sizeof(int) );
              memcpy ( momentum_list[iptot][momentum_config_counter][1], pi1, 3*sizeof(int) );
            }
#if TIMERS
            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "htpp_compact_twop", "momentum-filter", io_proc == 2 );
#endif

            for( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {
 
#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif
              /* coherent source coords */
  
              int const csx[4] = {
                               ( gsx[0] + ( T_global / g_coherent_source_number ) * icoh ) % T_global,
                               ( gsx[1] + (LX_global / g_coherent_source_number ) * icoh ) % LX_global,
                               ( gsx[2] + (LY_global / g_coherent_source_number ) * icoh ) % LY_global,
                               ( gsx[3] + (LZ_global / g_coherent_source_number ) * icoh ) % LZ_global };
                
              /***********************************************************
               * aff key for reading data
               ***********************************************************/

              char key[500], key_oet[500];

              /* /m-m-oet/sample00/gf01_gi01/px00py00pz01/kx00ky00kz00/t03 */

              /* /m-m/gf01_gi10/px01py00pz-01/x07y03z12t15 */

              sprintf ( key, "/m-m/gf%.2d_gi%.2d/px%.2dpy%.2dpz%.2d/x%.2dy%.2dz%.2dt%.2d",
                  gamma_p_list[0], gamma_p_list[0],
                   pf1[0], pf1[1], pf1[2],
                   csx[1], csx[2], csx[3], csx[0] );

              sprintf ( key_oet, "/m-m-oet/sample%.2d/gf%.2d_gi%.2d/px%.2dpy%.2dpz%.2d/kx%.2dky%.2dkz%.2d/t%.2d",
                  isample, gamma_p_list[0], gamma_p_list[0],
                  pf2[0], pf2[1], pf2[2],
                  pi2[0], pi2[1], pi2[2],
                  csx[0] );

              if ( g_verbose > 2 ) {
                fprintf ( stdout, "# [htpp_compact_twop] key     = %s %s %d\n", key     , __FILE__, __LINE__ );
                fprintf ( stdout, "# [htpp_compact_twop] key_oet = %s %s %d\n", key_oet , __FILE__, __LINE__ );
              }
#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact_twop", "aff-key", io_proc == 2 );
#endif
              /***********************************************************/
              /***********************************************************/
#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif

              exitstatus = read_aff_contraction (  corr_buffer, affr, NULL, key, T_global );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[htpp_compact_twop] Error from read_aff_contraction for key %s, status was %d %s %d\n",
                    key, exitstatus, __FILE__, __LINE__ );
                EXIT(12);
              }

              exitstatus = read_aff_contraction (  corr_buffer_oet, affr_oet, NULL, key_oet, T_global );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[htpp_compact_twop] Error from read_aff_contraction for key %s, status was %d %s %d\n",
                    key_oet, exitstatus, __FILE__, __LINE__ );
                EXIT(12);
              }

#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact_twop", "read_aff_contraction", io_proc == 2 );
#endif


#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif
              /***********************************************************
               * source phase factor
               * and order from source
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
                /* it = time relative to source
                 * tt = lattice timeslice
                 */
                int const tt = ( csx[0] + it ) % T_global;
                /***********************************************************
                 * -1 factor for 2 fermion loops
                 *  (relative to MXM_MXM)
                 ***********************************************************/
                double _Complex const zbuffer = -corr_buffer[tt] * corr_buffer_oet[it] * ephase;
              
                corr_mxm_mxm[iptot][icoh][momentum_config_counter][2*it  ] += creal ( zbuffer );
                corr_mxm_mxm[iptot][icoh][momentum_config_counter][2*it+1] += cimag ( zbuffer );

              }

#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact_twop", "source-phase-and-reorder", io_proc == 2 );
#endif
            }  /* coherent sources */
          }  /* end of loop on pi1 */
        }  /* end of loop on pf1 */
      } /* total momenta */
    
#ifdef HAVE_LHPC_AFF
      aff_reader_close ( affr );
      aff_reader_close ( affr_oet );
#endif
      }  /* samples */

      /***************************************************************************
       * write to h5 file
       ***************************************************************************/
            
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) {

        int ptot[3] = {
                 g_total_momentum_list[iptot][0],
                 g_total_momentum_list[iptot][1],
                 g_total_momentum_list[iptot][2] };

        /***********************************************************
         * output filename
         ***********************************************************/
        sprintf ( filename, "%s.px%d_py%d_pz%d.h5", correlator_name, ptot[0], ptot[1], ptot[2] );

        if ( iconf == 0 && isrc == 0 ) {
          /***************************************************************************
           * write momentum list to h5 file
           ***************************************************************************/
          int const momentum_ndim = 3;
          int const momentum_cdim[3] = {momentum_number[iptot], 2, 3 };
          char momentum_tag[200];
          sprintf( momentum_tag, "/mom" );
          exitstatus = write_h5_contraction ( (void*)(momentum_list[iptot][0][0]), NULL, filename, momentum_tag, "int", momentum_ndim, momentum_cdim );
          if( exitstatus != 0 ) {
            fprintf ( stderr, "[htpp_compact_twop] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }

        }


        /***********************************************************
         * normalize with number of stochastic samples
         ***********************************************************/
        for ( int  i = 0; i < g_coherent_source_number * momentum_number[iptot] * 2 * n_tc ; i++ ) {
          corr_mxm_mxm[iptot][0][0][i] /= g_nsample_oet;
        }

#if TIMERS
        gettimeofday ( &ta, (struct timezone *)NULL );
#endif
        for( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {

          int const csx[4] = {
              ( gsx[0] + ( T_global / g_coherent_source_number ) * icoh ) % T_global,
              ( gsx[1] + (LX_global / g_coherent_source_number ) * icoh ) % LX_global,
              ( gsx[2] + (LY_global / g_coherent_source_number ) * icoh ) % LY_global,
              ( gsx[3] + (LZ_global / g_coherent_source_number ) * icoh ) % LZ_global };
 
          char key[500];

          sprintf ( key, "/s%c/c%d/t%dx%dy%dz%d", stream, Nconf, csx[0], csx[1], csx[2], csx[3] );

          if ( g_verbose > 2 ) {
            fprintf ( stdout, "# [htpp_compact_twop] h5 key = %s %s %d\n", key , __FILE__, __LINE__ );
          }

          int const ndim = 2;
          int const cdim[2] = { momentum_number[iptot],  2 * n_tc };

          exitstatus = write_h5_contraction ( corr_mxm_mxm[iptot][icoh][0], NULL, filename, key, "double", ndim, cdim );
          if ( exitstatus != 0) {
            fprintf ( stderr, "[http_compact] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(123);
          }
#if TIMERS
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "htpp_compact_twop", "write_h5_contraction", io_proc == 2 );
#endif
        }  /* end of loop on coherent sources */

        fini_3level_dtable ( &( corr_mxm_mxm[iptot] ) );
        fini_3level_itable ( &( momentum_list[iptot] ) );

      }  /* end of loop on total momenta */

    }  /* end of loop on base sources */

    gettimeofday ( &conf_etime, (struct timezone *)NULL );
    show_time ( &conf_stime, &conf_etime, "htpp_compact_twop", "time-per-conf", io_proc == 2 );

  }  /* end of loop on configs */


  free ( corr_mxm_mxm );
  fini_1level_ztable ( &corr_buffer_oet );

#endif  /* of if MXM_MXM_OET */

  /***************************************************************************
   * free the correl fields
   ***************************************************************************/
  free ( momentum_list );
  fini_1level_itable ( &momentum_number );
  fini_1level_ztable ( &corr_buffer );
   
  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/
  fini_3level_itable ( &conf_src_list );

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "htpp_compact_twop", "total-time", io_proc == 2 );

  return(0);

}
