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
int get_momentum_id ( int * const p, int ** const p_list, int const p_num ) {
  for ( int i = 0; i < p_num; i++ ) {
    if (  ( p[0] == p_list[i][0] ) && ( p[1] == p_list[i][1] ) && ( p[2] == p_list[i][2] ) ) {
      return ( i );
    }
  }
  return ( -1 );
}  /* end of get_momentum_id */


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
  char diagram_name[100], diagram_name2[100], correlator_name[100];
  char mode[12] = "NA";

  struct timeval ta, tb;
  struct timeval start_time, end_time;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:S:N:E:D:d:C:M:")) != -1) {
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
    case 'd':
      strcpy ( diagram_name2, optarg );
      fprintf ( stdout, "# [htpp_compact_twop] diagram_name2 set to %s\n", diagram_name2 );
      break;
    case 'C':
      strcpy ( correlator_name, optarg );
      fprintf ( stdout, "# [htpp_compact_twop] correlator_name set to %s\n", correlator_name );
      break;
    case 'M':
      strcpy ( mode, optarg );
      fprintf ( stdout, "# [htpp_compact_twop] mode set to %s\n", mode );
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

if ( strcmp ( mode, "M_M" ) == 0 ) 
{
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
      sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.t%d_x%d_y%d_z%d.aff", filename_prefix, stream, Nconf, filename_prefix2, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
      // sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.tbase%.2d.aff", filename_prefix, stream, Nconf, filename_prefix2, Nconf, gsx[0] );
      // sprintf ( filename, "%s/stream_%c/%d/%s.aff", filename_prefix, stream, Nconf, filename_prefix2 );
   
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

              /* /fb-fl/t60x47y20z39/g1_gz/g2_gz/PX0_PY2_PZ1 */
              sprintf ( key, "/%s/t%.2dx%.2dy%.2dz%.2d/g1_%s/g2_%s/PX%d_PY%d_PZ%d",
                        diagram_name, csx[0], csx[1], csx[2], csx[3], 
                        gamma_bin_to_name[g_sink_gamma_id_list[igf]], gamma_bin_to_name[g_source_gamma_id_list[igi]],
                        pf[0], pf[1], pf[2]);


              /* /fl-fl/t05x02y21z18/gf01_gi01/PX0_PY0_PZ0 */
              /* sprintf ( key, "/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d_gi%.2d/PX%d_PY%d_PZ%d",
                        diagram_name, csx[0], csx[1], csx[2], csx[3], 
                        g_sink_gamma_id_list[igf], g_source_gamma_id_list[igi],
                        pf[0], pf[1], pf[2]); */

              /* /m-m/gf01_gi01/px00py00pz01/x20y06z00t65 */
              /* sprintf ( key, "/%s/gf%.2d_gi%.2d/px%.2dpy%.2dpz%.2d/x%.2dy%.2dz%.2dt%.2d",
                        diagram_name, 
                        g_sink_gamma_id_list[igf], g_source_gamma_id_list[igi],
                        pf[0], pf[1], pf[2],
                        csx[1], csx[2], csx[3], csx[0] ); */

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
            
                /* double _Complex const ephase = 1.; */

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
                  // int const tt = it;
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
        sprintf ( filename, "%s.px%d_py%d_pz%d.h5", g_outfile_prefix,  ptot[0], ptot[1], ptot[2] );

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

          /* sprintf ( key, "/s%c/c%d/t%dx%dy%dz%d", stream, Nconf, csx[0], csx[1], csx[2], csx[3] ); */
          
#if 0          
          for ( int igi = 0; igi < g_source_gamma_id_number; igi++ ) 
          {
          for ( int igf = 0; igf < g_sink_gamma_id_number; igf++ ) 
          {
#endif

#if 0
            /* /bu-ub/gf_g5/gi_g5/s0/c2108/t18x12y26z1 */
            sprintf ( key, "/%s/gf_%s/gi_%s/s%c/c%d/t%dx%dy%dz%d", correlator_name,
                gamma_bin_to_name[g_sink_gamma_id_list[igf]],
                gamma_bin_to_name[g_source_gamma_id_list[igi]],
                stream, Nconf, csx[0], csx[1], csx[2], csx[3] );


            sprintf ( key, "/%s/pfx%dpfy%dpfz%d/pi1x%dpi1y%dpi1z%d/gc_%s/s%c/c%d/t%dx%dy%dz%d",
                tp->name,
                pf[0], pf[1], pf[2],
                pi1[0], pi1[1], pi1[2],
                gamma_bin_to_name[tp->gf2],
                stream, Nconf, csx[0], csx[1], csx[2], csx[3] );
#endif
            sprintf ( key, "/s%c/c%d/t%dx%dy%dz%d", stream, Nconf, csx[0], csx[1], csx[2], csx[3] );


            if ( g_verbose > 2 ) {
              fprintf ( stdout, "# [htpp_compact_twop] h5 key = %s %s %d\n", key , __FILE__, __LINE__ );
            }

            int const ndim = 3;
            int const cdim[3] = { g_source_gamma_id_number, g_sink_gamma_id_number, 2 * n_tc };

            exitstatus = write_h5_contraction ( corr_m_m[iptot][icoh][0][0], NULL, filename, key, "double", ndim, cdim );
#if 0
            int const ndim = 1;
            int const cdim[1] = { 2 * n_tc };

            exitstatus = write_h5_contraction ( corr_m_m[iptot][icoh][igi][igf], NULL, filename, key, "double", ndim, cdim );
#endif
            if ( exitstatus != 0) {
             fprintf ( stderr, "[http_compact] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
               EXIT(123);
            }
#if 0
          }}
#endif

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

} /* end of of if M_M */

  /***********************************************************/
  /***********************************************************/

if ( strcmp ( mode, "MXM_M" ) == 0 )
{

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
      // sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.tbase%.2d.aff", filename_prefix, stream, Nconf, filename_prefix2, Nconf, gsx[0] );
      sprintf ( filename, "%s/stream_%c/%d/%s.aff", filename_prefix, stream, Nconf, filename_prefix2 );

      struct AffReader_s * affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[htpp_compact_twop] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
        // EXIT(5);
        sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.tbase%.2d.aff", filename_prefix, stream, Nconf, filename_prefix3, Nconf, gsx[0] );

        fprintf ( stdout, "# [http_compact_twop] Warning, try open alternative data file %s %s %d\n", filename, __FILE__, __LINE__ );
        affr = aff_reader ( filename );
      
        const char * aff_status_str2 = aff_reader_errstr ( affr );
        if( aff_status_str2 != NULL ) {
          fprintf(stderr, "[htpp_compact_twop] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str2, __FILE__, __LINE__);
          EXIT(5);
        }
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

            /* for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ )  */
            for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) 
            {

              /* int pi1[3] = {
                g_source_momentum_list[ipi][0],
                g_source_momentum_list[ipi][1],
                g_source_momentum_list[ipi][2] };
                */

              int pi2[3] = {
                g_seq_source_momentum_list[ipi][0],
                g_seq_source_momentum_list[ipi][1],
                g_seq_source_momentum_list[ipi][2] };

              int pi1[3] = {
                  -ptot[0] - pi2[0],
                  -ptot[1] - pi2[1],
                  -ptot[2] - pi2[2] };

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

        corr_mxm_m[iptot] = init_4level_dtable ( g_coherent_source_number, g_sink_gamma_id_number, momentum_number[iptot],  2 * T_global );
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
  
            /* for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ )  */
            for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ )
            {
/*
              int pi1[3] = {
                g_source_momentum_list[ipi][0],
                g_source_momentum_list[ipi][1],
                g_source_momentum_list[ipi][2] };
*/
              int pi2[3] = {
                g_seq_source_momentum_list[ipi2][0],
                g_seq_source_momentum_list[ipi2][1],
                g_seq_source_momentum_list[ipi2][2] };

              int pi1[3] = {
                  -ptot[0] - pi2[0],
                  -ptot[1] - pi2[1],
                  -ptot[2] - pi2[2] };
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

                
              /***********************************************************
               * aff key for reading data
               ***********************************************************/
              char key[500];

#if 0
                sprintf ( key,
                        "/%s/g%.2d/px%.2dpy%.2dpz%.2d/kx%.2dky%.2dkz%.2d/x%.2dy%.2dz%.2dt%.2d",
                        /* "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/pi2x%dpi2y%dpi2z%d/gi2_%s/g1_%s/g2_%s/x%d_y%d_z%d", */
                        diagram_name,
                        g_sink_gamma_id_list[iv], 
                        pf[0], pf[1], pf[2],
                        pi2[0], pi2[1], pi2[2],
                        csx[1], csx[2], csx[3], csx[0] );

              sprintf ( key,
                        "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/pi2x%dpi2y%dpi2z%d/gi2_%s/g1_%s/g2_%s/PX%d_PY%d_PZ%d",
                        /* "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/pi2x%dpi2y%dpi2z%d/gi2_%s/g1_%s/g2_%s/x%d_y%d_z%d", */
                        diagram_name,
                        pf[0], pf[1], pf[2],
                        gamma_bin_to_name[gamma_p_list[0]], g_src_snk_time_separation,
                        pi2[0], pi2[1], pi2[2], gamma_bin_to_name[gamma_p_list[0]],
                        gamma_bin_to_name[gamma_v_list[0]], gamma_bin_to_name[gamma_p_list[0]],
                        pc[0], pc[1], pc[2] );


              /* /fl-gf-sll/pi2x0pi2y0pi2z0/gi2_g5/g1_gy/g2_g5/PX0_PY1_PZ-1 */
              sprintf ( key,
                  "/%s/pi2x%dpi2y%dpi2z%d/gi2_%s/g1_%s/g2_%s/PX%d_PY%d_PZ%d",
                  diagram_name,
                  pi2[0], pi2[1], pi2[2],
                  gamma_bin_to_name[g_sequential_source_gamma_id],
                  gamma_bin_to_name[g_sink_gamma_id_list[iv]], 
                  gamma_bin_to_name[g_source_gamma_id_list[0]],
                  pf[0], pf[1], pf[2] );
#endif


              /* /fl-gf-sll/g01/kx00ky01kz01/PX0_PY0_PZ0 */
              /* sprintf ( key,
                  "/%s/g%.2d/kx%.2dky%.2dkz%.2d/PX%d_PY%d_PZ%d",
                  diagram_name,
                  g_sink_gamma_id_list[iv],
                  pi2[0], pi2[1], pi2[2],
                  pf[0], pf[1], pf[2] ); */

              /* /mxm-m/g01/px00py00pz00/kx00ky00kz01/x08y16z31t39 */
              sprintf ( key,
                  "/%s/g%.2d/px%.2dpy%.2dpz%.2d/kx%.2dky%.2dkz%.2d/x%.2dy%.2dz%.2dt%.2d",
                  diagram_name,
                  g_sink_gamma_id_list[iv],
                  pf[0], pf[1], pf[2],
                  pi2[0], pi2[1], pi2[2],
                  gsx[1], gsx[2], gsx[3], gsx[0] );


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

              /***********************************************************
               * loop on coherent source positions
               ***********************************************************/
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

#if 0
                int const n_tc = T_global / ( 2 * g_coherent_source_number );
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
                for ( int it = -n_tc + 1; it <= n_tc; it++ ) 
                {
                  /* it = time relative to source
                   * tt = lattice timeslice
                   */
                  int const tt  = ( csx[0] + it + T_global ) % T_global; 
                  int const it2 = (          it + T_global ) % T_global; 
                  double _Complex const zbuffer = corr_buffer[tt] * ephase;
              
                  corr_mxm_m[iptot][icoh][iv][momentum_config_counter][2*it2  ] = creal ( zbuffer );
                  corr_mxm_m[iptot][icoh][iv][momentum_config_counter][2*it2+1] = cimag ( zbuffer );

                }
#endif

                int const n_tc = T_global;
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
                for ( int it = 0; it < n_tc; it++ ) {
                  /* it = time relative to source
                   * tt = lattice timeslice
                   */
                  // int const tt = ( csx[0] + it ) % T_global;
                  int const tt = it;
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
        sprintf ( filename, "%s.px%d_py%d_pz%d.h5", g_outfile_prefix, ptot[0], ptot[1], ptot[2] );

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
          int const cdim[3] = { g_sink_gamma_id_number, momentum_number[iptot],  2 * T_global };

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

} /* end of if MXM_M */


  /***********************************************************/
  /***********************************************************/

if ( strcmp ( mode, "MXM_MXM_BOX" ) == 0 )
{
  int const n_tc = g_src_snk_time_separation + 1;
  fprintf ( stdout, "# [htpp_compact_twop] n_tc = %d   %s %d\n", n_tc, __FILE__, __LINE__ );

  double **** corr_mxm_mxm = (double****) malloc ( g_total_momentum_number * sizeof( double***) );

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
    for( int isrc = 0; isrc < num_src_per_conf; isrc++) 
    {

      /***********************************************************
       * store the source coordinates
       ***********************************************************/
      int const gsx[4] = {
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4],
            conf_src_list[iconf][isrc][5] };

      /***********************************************************
       * loop on source momenta
       ***********************************************************/
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ )
      {

        int ptot[3] = {
                   g_total_momentum_list[iptot][0],
                   g_total_momentum_list[iptot][1],
                   g_total_momentum_list[iptot][2] };

        if( iconf == 0 && isrc == 0 ) 
        {

          momentum_number[iptot] = 0;

          /***********************************************************
           * loop on sink momenta
           ***********************************************************/
          for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ )
          {

            int pf1[3] = {
              g_sink_momentum_list[ipf][0],
              g_sink_momentum_list[ipf][1],
              g_sink_momentum_list[ipf][2] };

            int pf2[3] = {
                ptot[0] - pf1[0],
                ptot[1] - pf1[1],
                ptot[2] - pf1[2] };

            int ipf2 = get_momentum_id ( pf2, g_seq2_source_momentum_list, g_seq2_source_momentum_number );
            if ( g_verbose > 2 ) {
              fprintf ( stdout, "# [htpp_compact_twop] pf2 %3d %3d %3d  seq2_source_momentum %2d   %s %d\n",
                  pf2[0], pf2[1], pf2[2], ipf2 , __FILE__, __LINE__ );
            }
            if ( ipf2 == -1 ) continue;

            for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) 
            {

              int pi2[3] = {
                g_seq_source_momentum_list[ipi][0],
                g_seq_source_momentum_list[ipi][1],
                g_seq_source_momentum_list[ipi][2] };

              int pi1[3] = {
                  -ptot[0] - pi2[0],
                  -ptot[1] - pi2[1],
                  -ptot[2] - pi2[2] };

              if ( momentum_filter ( pf1, pf2, pi1, pi2 , max_single_particle_momentum_squared ) ) {
                momentum_number[iptot]++;
                if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_compact_twop] mom pi1 (%d %d %d) pi2 (%d %d %d) pf1 (%d %d %d) pf2 (%d %d %d) passed\n",
                    pi1[0], pi1[1], pi1[2],
                    pi2[0], pi2[1], pi2[2],
                    pf1[0], pf1[1], pf1[2],
                    pf2[0], pf2[1], pf2[2]);
                  
              } else {
                if ( g_verbose > 2 ) fprintf ( stdout, "# [htpp_compact_twop] mom pi1 (%d %d %d) pi2 (%d %d %d) pf1 (%d %d %d) pf2 (%d %d %d) filtered\n",
                    pi1[0], pi1[1], pi1[2],
                    pi2[0], pi2[1], pi2[2],
                    pf1[0], pf1[1], pf1[2],
                    pf2[0], pf2[1], pf2[2]);
              }
            }
 
          }  /* end of loop on sink momentum */
          
          fprintf ( stdout, "# [htpp_compact_twop] number of momentum combinations = %d %s %d\n", momentum_number[iptot], __FILE__, __LINE__ );

        }  /* end of if iconf = 0 and isrc = 0 */

        momentum_list[iptot] = init_3level_itable ( momentum_number[iptot], 2, 3 );
        if ( momentum_list[iptot] == NULL ) {
          fprintf( stderr, "[htpp_compact_twop] Error from init_Xlevel_itable %s %d\n", __FILE__, __LINE__ );
          EXIT(2);
        }

        corr_mxm_mxm[iptot] = init_3level_dtable ( g_coherent_source_number, momentum_number[iptot],  2 * n_tc );
        if ( corr_mxm_mxm[iptot] == NULL ) {
          fprintf( stderr, "[htpp_compact_twop] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(2);
        }


        for ( int isample = 0; isample < g_nsample; isample++ )
        {

#ifdef HAVE_LHPC_AFF
          /***********************************************************
           * reader for aff output file
           ***********************************************************/
          sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.%.2d.%.5d", filename_prefix, stream, Nconf, filename_prefix2, Nconf, gsx[0], isample );

          struct AffReader_s * affr = aff_reader ( filename );
          const char * aff_status_str = aff_reader_errstr ( affr );
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[htpp_compact_twop] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
            EXIT(5);
          } else {
            if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact_twop] Reading data from file %s\n", filename );
          }
#endif
          int momentum_config_counter = -1;


          /***********************************************************
           * loop on sink momenta
           ***********************************************************/

          for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ )
          {

            int pf1[3] = {
                g_sink_momentum_list[ipf][0],
                g_sink_momentum_list[ipf][1],
                g_sink_momentum_list[ipf][2] };

            int pf2[3] = {
                  ptot[0] - pf1[0],
                  ptot[1] - pf1[1],
                  ptot[2] - pf1[2] };

            int ipf2 = get_momentum_id ( pf2, g_seq2_source_momentum_list, g_seq2_source_momentum_number );
            if ( g_verbose > 2 ) {
              fprintf ( stdout, "# [htpp_compact_twop] pf2 %3d %3d %3d  seq2_source_momentum %2d   %s %d\n",
                  pf2[0], pf2[1], pf2[2], ipf2 , __FILE__, __LINE__ );
            }
            if ( ipf2 == -1 ) continue;



            for ( int ipi2 = 0; ipi2 < g_seq_source_momentum_number; ipi2++ )
            {

              int pi2[3] = {
                  g_seq_source_momentum_list[ipi2][0],
                  g_seq_source_momentum_list[ipi2][1],
                  g_seq_source_momentum_list[ipi2][2] };

              int pi1[3] = {
                    -ptot[0] - pi2[0],
                    -ptot[1] - pi2[1],
                    -ptot[2] - pi2[2] };
#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif


              /***********************************************************
               * set twop function momenta an filter
               ***********************************************************/
              if ( ! momentum_filter ( pf1, pf2, pi1, pi2 , max_single_particle_momentum_squared ) ) continue;
              momentum_config_counter++;

              /***********************************************************
               * add momentum config to list
               ***********************************************************/
              if ( iconf == 0 && isrc == 0 )
              {
                memcpy ( momentum_list[iptot][momentum_config_counter][0], pf1, 3*sizeof(int) );
                memcpy ( momentum_list[iptot][momentum_config_counter][1], pi1, 3*sizeof(int) );
              }

              /***********************************************************
               * loop on coherent source positions
               ***********************************************************/
              for( int icoh = 0; icoh < g_coherent_source_number; icoh++ )
              {
 
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

                  /* /mxm-mxm/sample00/pi2x00pi2y01pi2z-01/pf1x00pf1y01pf1z01/pf2x01pf2y00pf2z-01/x14y16z05t51 */
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
                  show_time ( &ta, &tb, "htpp_compact_twop", "momentum-filter-and-key", io_proc == 2 );
#endif
                  /***********************************************************/
                  /***********************************************************/
#if TIMERS
                  gettimeofday ( &ta, (struct timezone *)NULL );
#endif

                  exitstatus = read_aff_contraction (  corr_buffer, affr, NULL, key, n_tc );
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
                    // int const tt = ( csx[0] + it ) % T_global;
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

            }  /* end of loop on pi2 */

          }  /* end of loop on pf1 */


#ifdef HAVE_LHPC_AFF
          aff_reader_close ( affr );
#endif

        }  /* end of loop on samples */


        /* normalize by 1  / number of samples */
        for ( int i = 0; i < momentum_number[iptot] * g_coherent_source_number * n_tc * 2; i++ )
        {
          corr_mxm_mxm[iptot][0][0][i] /= (double)g_nsample;
        }


      } /* total momenta */

      /***************************************************************************
       * write to h5 file
       ***************************************************************************/
            
      for ( int iptot = 0; iptot < g_total_momentum_number; iptot++ ) 
      {


        int ptot[3] = {
                 g_total_momentum_list[iptot][0],
                 g_total_momentum_list[iptot][1],
                 g_total_momentum_list[iptot][2] };

        /***********************************************************
         * output filename
         ***********************************************************/
        sprintf ( filename, "%s.px%d_py%d_pz%d.h5", g_outfile_prefix, ptot[0], ptot[1], ptot[2] );

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

#if TIMERS
        gettimeofday ( &ta, (struct timezone *)NULL );
#endif
        for( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) 
        {

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


    }  /* end of loop on base sources isrc */

    gettimeofday ( &conf_etime, (struct timezone *)NULL );
    show_time ( &conf_stime, &conf_etime, "htpp_compact_twop", "time-per-conf", io_proc == 2 );

  }  /* end of loop on configs */


  free ( corr_mxm_mxm );

} /* end of if MXM_MXM_BOX */

  /***********************************************************/
  /***********************************************************/

if ( strcmp ( mode, "MXM_MXM" ) == 0 )
{


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

      double _Complex ***** eta_xi = init_5level_ztable ( g_seq_source_momentum_number, g_nsample, 12, g_seq2_source_momentum_number, T_global );
      if ( eta_xi == NULL ) {
        fprintf(stderr, "[htpp_compact_twop] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(5);
      }

      sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.t%d_x%d_y%d_z%d.aff" , filename_prefix, stream, Nconf, filename_prefix2, Nconf, 
          gsx[0], gsx[1], gsx[2], gsx[3]);

      /* eta_xi_light.0576.tbase23.aff */
      /* sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.tbase%.2d.aff" , filename_prefix, stream, Nconf, filename_prefix2, Nconf, gsx[0]); */
   
      struct AffReader_s * affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[htpp_compact_twop] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
        EXIT(5);
      } else {
        if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact_twop] Reading data from file %s\n", filename );
      }
      char key[500];

      for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {
        for ( int isample = 0; isample < g_nsample; isample++ ) {
          for ( int icol = 0; icol < 12; icol++ ) {
            for ( int ipf = 0; ipf < g_seq2_source_momentum_number; ipf++ ) {

              /* /xil-gf-sll/pi2x00pi2y00pi2z00/sample01/gf15/c06/PX0_PY0_PZ0 */
              sprintf ( key, "/%s/pi2x%.2dpi2y%.2dpi2z%.2d/sample%.2d/gf%.2d/c%.2d/PX%d_PY%d_PZ%d", 
                  diagram_name,  
                  g_seq_source_momentum_list[ipi][0], g_seq_source_momentum_list[ipi][1], g_seq_source_momentum_list[ipi][2],
                  isample, g_sequential_source_gamma_id_list[1], icol, 
                  g_seq2_source_momentum_list[ipf][0], g_seq2_source_momentum_list[ipf][1], g_seq2_source_momentum_list[ipf][2]); 

              if ( g_verbose > 2 ) {
                fprintf ( stdout, "# [htpp_compact_twop] key = %s %s %d\n", key , __FILE__, __LINE__ );
              }

              exitstatus = read_aff_contraction ( eta_xi[ipi][isample][icol][ipf], affr, NULL, key, T_global );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[htpp_compact_twop] Error from read_aff_contraction for file %s key %s, status was %d %s %d\n",
                    filename, key, exitstatus, __FILE__, __LINE__ );
                EXIT(12);
              }

            }
          }
        }
      }

      aff_reader_close ( affr );

      double _Complex **** eta_phi = init_4level_ztable ( g_nsample, 12, g_sink_momentum_number, T_global );
      if ( eta_phi == NULL ) {
        fprintf(stderr, "[htpp_compact_twop] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(5);
      }

      for ( int isample = 0; isample < g_nsample; isample++ ) {

        sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.t%d_x%d_y%d_z%d.%.5d.aff" , filename_prefix, stream, Nconf, filename_prefix3, Nconf,
            gsx[0], gsx[1], gsx[2], gsx[3], isample );

        /* sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.tbase%.2d.%.5d.aff" , filename_prefix, stream, Nconf, filename_prefix3, Nconf, gsx[0], isample ); */

        affr = aff_reader ( filename );
        aff_status_str = aff_reader_errstr ( affr );

        if( aff_status_str != NULL ) {
          fprintf(stderr, "[htpp_compact_twop] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
          EXIT(5);
        } else {
          if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact_twop] Reading data from file %s\n", filename );
        }

        for ( int icol = 0; icol < 12; icol++ ) {
          for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

            /* /fl-gf-phil/sample10/gf15/c06/PX0_PY0_PZ-1 */
            sprintf ( key, "/%s/sample%.2d/gf%.2d/c%.2d/PX%d_PY%d_PZ%d", 
                  diagram_name2, isample, g_sink_gamma_id_list[0], icol, 
                  g_sink_momentum_list[ipf][0], g_sink_momentum_list[ipf][1], g_sink_momentum_list[ipf][2]); 

            if ( g_verbose > 2 ) {
              fprintf ( stdout, "# [htpp_compact_twop] key = %s %s %d\n", key , __FILE__, __LINE__ );
            }

            exitstatus = read_aff_contraction (  eta_phi[isample][icol][ipf], affr, NULL, key, T_global );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[htpp_compact_twop] Error from read_aff_contraction for file %s key %s, status was %d %s %d\n",
                  filename, key, exitstatus, __FILE__, __LINE__ );
              EXIT(12);
            }

          }
        }
      
        aff_reader_close ( affr );
      }



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

            int ipf2 = get_momentum_id ( pf2, g_seq2_source_momentum_list, g_seq2_source_momentum_number );
            if ( g_verbose > 2 ) {
              fprintf ( stdout, "# [htpp_compact_twop] pf2 %3d %3d %3d  seq2_source_momentum %2d   %s %d\n",  
                  pf2[0], pf2[1], pf2[2], ipf2 , __FILE__, __LINE__ );
            }
            if ( ipf2 == -1 ) continue;

            /* for ( int ipi = 0; ipi < g_source_momentum_number; ipi++ ) */
            for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ )
            {

              /* int pi1[3] = {
                g_source_momentum_list[ipi][0],
                g_source_momentum_list[ipi][1],
                g_source_momentum_list[ipi][2] };
                */

              int pi2[3] = {
                g_seq_source_momentum_list[ipi][0],
                g_seq_source_momentum_list[ipi][1],
                g_seq_source_momentum_list[ipi][2] };

              int pi1[3] = {
                  -ptot[0] - pi2[0],
                  -ptot[1] - pi2[1],
                  -ptot[2] - pi2[2] };

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
          corr_mxm_mxm[iptot] = init_3level_dtable ( g_coherent_source_number, momentum_number[iptot],  2 *  T_global );
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

          int ipf2 = get_momentum_id ( pf2, g_seq2_source_momentum_list, g_seq2_source_momentum_number );
          if ( g_verbose > 2 ) {
            fprintf ( stdout, "# [htpp_compact_twop] pf2 %3d %3d %3d  seq2_source_momentum %2d   %s %d\n",  
                pf2[0], pf2[1], pf2[2], ipf2, __FILE__, __LINE__ );
          }
          if ( ipf2 == -1 ) continue;


          for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {

            /* int pi1[3] = {
              g_source_momentum_list[ipi][0],
              g_source_momentum_list[ipi][1],
              g_source_momentum_list[ipi][2] };
              */

            int pi2[3] = {
              g_seq_source_momentum_list[ipi][0],
              g_seq_source_momentum_list[ipi][1],
              g_seq_source_momentum_list[ipi][2] };

            int pi1[3] = {
                -ptot[0] - pi2[0],
                -ptot[1] - pi2[1],
                -ptot[2] - pi2[2] };
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
            if ( iconf == 0 && isrc == 0 ) {
              memcpy ( momentum_list[iptot][momentum_config_counter][0], pf1, 3*sizeof(int) );
              memcpy ( momentum_list[iptot][momentum_config_counter][1], pi1, 3*sizeof(int) );
            }
#if TIMERS
            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "htpp_compact_twop", "momentum-filter", io_proc == 2 );
#endif

            /***********************************************************
             * 
             ***********************************************************/
            gamma_matrix_type gi1;
            gamma_matrix_qlua_binary ( &gi1,  g_source_gamma_id_list[0] );
            if ( g_verbose > 2 ) gamma_matrix_printf ( &gi1, "g5-qlua", stdout );


            memset ( corr_buffer, 0, T_global * sizeof ( double _Complex ) );
 
            for ( int isample = 0; isample < g_nsample; isample++ ) {

              for ( int i = 0; i < 4; i++ ) {
              for ( int k = 0; k < 4; k++ ) {
                for ( int a = 0; a < 3; a++ ) {

                  int const icol_xi  = 4 * a + i;  /* leading color */
                  int const icol_phi = 4 * a + k;  /* leading color */

                  for ( int it = 0; it < T_global; it++ ) 
                  {
                    corr_buffer[it] += eta_xi[ipi][isample][ icol_xi ][ipf2][it] * eta_phi[isample][ icol_phi ][ipf][it] * gi1.m[i][k];
                  }
                }
              }}
            }

            /***********************************************************
             * normalize with number of stochastic samples
             ***********************************************************/
            for ( int it = 0; it < T_global; it++ ) 
            {
              corr_buffer[it] /= (double)g_nsample;
            }

            /***********************************************************/
            /***********************************************************/

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

              int const n_tc = T_global / ( 2 * g_coherent_source_number );

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
              for ( int it = -n_tc +1 ; it <= n_tc; it++ ) {

                int const tt  = ( csx[0] + it + T_global ) % T_global;
                int const it2 = (          it + T_global ) % T_global;
                double _Complex const zbuffer = corr_buffer[tt] * ephase;
              
                corr_mxm_mxm[iptot][icoh][momentum_config_counter][2*it2  ] += creal ( zbuffer );
                corr_mxm_mxm[iptot][icoh][momentum_config_counter][2*it2+1] += cimag ( zbuffer );
              }
#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact_twop", "source-phase-and-reorder", io_proc == 2 );
#endif
            }  /* coherent sources */

          }  /* end of loop on pi2 */

        }  /* end of loop on pf1 */

      } /* total momenta */
    
      fini_4level_ztable ( &eta_phi );
      fini_5level_ztable ( &eta_xi );


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
        sprintf ( filename, "%s.px%d_py%d_pz%d.h5", g_outfile_prefix, ptot[0], ptot[1], ptot[2] );

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
          int const cdim[2] = { momentum_number[iptot],  2 * T_global };

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

}  /* end of if MXM_MXM */

  /***********************************************************/
  /***********************************************************/

if ( strcmp ( mode, "MXM_MXM_OET" ) == 0 )
{


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

      for( int icoh = 0; icoh < g_coherent_source_number; icoh++ ) {

        int const csx[4] = {
            ( gsx[0] + ( T_global / g_coherent_source_number ) * icoh ) % T_global,
            ( gsx[1] + (LX_global / g_coherent_source_number ) * icoh ) % LX_global,
            ( gsx[2] + (LY_global / g_coherent_source_number ) * icoh ) % LY_global,
            ( gsx[3] + (LZ_global / g_coherent_source_number ) * icoh ) % LZ_global };


        double _Complex **** corr_oet = init_4level_ztable ( g_nsample_oet, g_seq_source_momentum_number, g_seq2_source_momentum_number, T_global );
        if ( corr_oet == NULL ) {
          fprintf(stderr, "[htpp_compact_twop] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__);
          EXIT(5);
        }

        /* sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.t%d_x%d_y%d_z%d.aff" , filename_prefix, stream, Nconf, filename_prefix2, Nconf, 
          csx[0], csx[1], csx[2], csx[3]); */
        /* sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.tsrc%.2d.aff" , filename_prefix, stream, Nconf, filename_prefix2, Nconf, csx[0] ); */
        sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.t%.2d" , filename_prefix, stream, Nconf, filename_prefix2, Nconf, csx[0] );

        struct AffReader_s * affr = aff_reader ( filename );
        const char * aff_status_str = aff_reader_errstr ( affr );
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[htpp_compact_twop] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
          EXIT(5);
        } else {
          if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact_twop] Reading data from file %s\n", filename );
        }
        char key[500];

        for ( int isample = 0; isample < g_nsample_oet; isample++ ) {
          for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {
            for ( int ipf = 0; ipf < g_seq2_source_momentum_number; ipf++ ) {

              /* /fl-fl/sample00/tsrc12/gf02_gi00/kx00ky-01kz01/PX1_PY0_PZ-1 */
              /* sprintf ( key, "/%s/sample%.2d/tsrc%.2d/gf%.2d_gi%.2d/kx%.2dky%.2dkz%.2d/PX%d_PY%d_PZ%d", 
                    diagram_name, isample, csx[0],
                    g_sequential_source_gamma_id_list[1], g_sequential_source_gamma_id_list[0],
                    g_seq_source_momentum_list[ipi][0], g_seq_source_momentum_list[ipi][1], g_seq_source_momentum_list[ipi][2],
                    g_seq2_source_momentum_list[ipf][0], g_seq2_source_momentum_list[ipf][1], g_seq2_source_momentum_list[ipf][2]); */

              /* /m-m-oet/sample00/gf02_gi12/px-01py00pz01/kx00ky-01kz00/t03 */
              /* sprintf ( key, "/%s/sample%.2d/gf%.2d_gi%.2d/px%.2dpy%.2dpz%.2d/kx%.2dky%.2dkz%.2d/t%.2d", 
                    diagram_name, isample, 
                    g_sequential_source_gamma_id_list[1], g_sequential_source_gamma_id_list[0],
                    g_seq2_source_momentum_list[ipf][0], g_seq2_source_momentum_list[ipf][1], g_seq2_source_momentum_list[ipf][2],
                    g_seq_source_momentum_list[ipi][0], g_seq_source_momentum_list[ipi][1], g_seq_source_momentum_list[ipi][2],
                    csx[0]); */


              /* /m-m-oet/sample00/gf02_gi12/px-01py00pz01/kx00ky-01kz00/t03
               * TEST with k to minus k
               */
              sprintf ( key, "/%s/sample%.2d/gf%.2d_gi%.2d/px%.2dpy%.2dpz%.2d/kx%.2dky%.2dkz%.2d/t%.2d", 
                    diagram_name, isample, 
                    g_sequential_source_gamma_id_list[1], g_sequential_source_gamma_id_list[0],
                    g_seq2_source_momentum_list[ipf][0], g_seq2_source_momentum_list[ipf][1], g_seq2_source_momentum_list[ipf][2],
                    -g_seq_source_momentum_list[ipi][0], -g_seq_source_momentum_list[ipi][1], -g_seq_source_momentum_list[ipi][2],
                    csx[0]); 
              if ( g_verbose > 2 ) {
                fprintf ( stdout, "# [htpp_compact_twop] key = %s %s %d\n", key , __FILE__, __LINE__ );
              }

              exitstatus = read_aff_contraction ( corr_oet[isample][ipi][ipf], affr, NULL, key, T_global );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[htpp_compact_twop] Error from read_aff_contraction for file %s key %s, status was %d %s %d\n",
                    filename, key, exitstatus, __FILE__, __LINE__ );
                EXIT(12);
              }
            }
          }
        }

        aff_reader_close ( affr );

        double _Complex ** corr_pta = init_2level_ztable ( g_sink_momentum_number, T_global );
        if ( corr_pta == NULL ) {
          fprintf(stderr, "[htpp_compact_twop] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__);
          EXIT(5);
        }

        /* sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.t%d_x%d_y%d_z%d.aff" , filename_prefix, stream, Nconf, filename_prefix3, Nconf,
            gsx[0], gsx[1], gsx[2], gsx[3] ); */

        /* sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.tbase%.2d.aff" , filename_prefix, stream, Nconf, filename_prefix3, Nconf, gsx[0] ); */

        sprintf ( filename, "%s/stream_%c/%d/%s.aff" , filename_prefix, stream, Nconf, filename_prefix3 );

        affr = aff_reader ( filename );
        aff_status_str = aff_reader_errstr ( affr );

        if( aff_status_str != NULL ) {
          fprintf(stderr, "[htpp_compact_twop] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
          EXIT(5);
        } else {
          if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact_twop] Reading data from file %s\n", filename );
        }

        for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

            /* /fl-fl/t60x19y22z32/g1_g5/g2_g5/PX1_PY0_PZ-1 */
            /* sprintf ( key, "/%s/t%.2dx%.2dy%.2dz%.2d/g1_%s/g2_%s/PX%d_PY%d_PZ%d", 
                  diagram_name2, csx[0], csx[1], csx[2], csx[3],
                  gamma_bin_to_name[g_sink_gamma_id_list[0]], gamma_bin_to_name[g_source_gamma_id_list[0]],
                  g_sink_momentum_list[ipf][0], g_sink_momentum_list[ipf][1], g_sink_momentum_list[ipf][2]); */

            /* /fl-fl/t23x21y19z16/gf04_gi02/PX1_PY1_PZ0 */
            /* sprintf ( key, "/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d_gi%.2d/PX%d_PY%d_PZ%d", 
                  diagram_name2, csx[0], csx[1], csx[2], csx[3],
                  g_sink_gamma_id_list[0], g_source_gamma_id_list[0],
                  g_sink_momentum_list[ipf][0], g_sink_momentum_list[ipf][1], g_sink_momentum_list[ipf][2]);  */


            /* /m-m/gf01_gi01/px00py00pz01/x20y06z00t65 */
            sprintf ( key, "/%s/gf%.2d_gi%.2d/px%.2dpy%.2dpz%.2d/x%.2dy%.2dz%.2dt%.2d",
                diagram_name2,
                g_sink_gamma_id_list[0], g_source_gamma_id_list[0],
                g_sink_momentum_list[ipf][0], g_sink_momentum_list[ipf][1], g_sink_momentum_list[ipf][2],
                csx[1], csx[2], csx[3], csx[0] );


            if ( g_verbose > 2 ) {
              fprintf ( stdout, "# [htpp_compact_twop] key = %s %s %d\n", key , __FILE__, __LINE__ );
            }

            exitstatus = read_aff_contraction (  corr_pta[ipf], affr, NULL, key, T_global );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[htpp_compact_twop] Error from read_aff_contraction for file %s key %s, status was %d %s %d\n",
                  filename, key, exitstatus, __FILE__, __LINE__ );
              EXIT(12);
            }

        }

        aff_reader_close ( affr );

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

              int ipf2 = get_momentum_id ( pf2, g_seq2_source_momentum_list, g_seq2_source_momentum_number );
              if ( g_verbose > 2 ) {
                fprintf ( stdout, "# [htpp_compact_twop] pf2 %3d %3d %3d  seq2_source_momentum %2d   %s %d\n",  
                    pf2[0], pf2[1], pf2[2], ipf2 , __FILE__, __LINE__ );
              }
              if ( ipf2 == -1 ) continue;

              for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ )
              {

                int pi2[3] = {
                  g_seq_source_momentum_list[ipi][0],
                  g_seq_source_momentum_list[ipi][1],
                  g_seq_source_momentum_list[ipi][2] };

                int pi1[3] = {
                    -ptot[0] - pi2[0],
                    -ptot[1] - pi2[1],
                    -ptot[2] - pi2[2] };

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
            corr_mxm_mxm[iptot] = init_3level_dtable ( g_coherent_source_number, momentum_number[iptot],  2 *  T_global );
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

            int ipf2 = get_momentum_id ( pf2, g_seq2_source_momentum_list, g_seq2_source_momentum_number );
            if ( g_verbose > 2 ) {
              fprintf ( stdout, "# [htpp_compact_twop] pf2 %3d %3d %3d  seq2_source_momentum %2d   %s %d\n",  
                  pf2[0], pf2[1], pf2[2], ipf2 , __FILE__, __LINE__ );
            }
            if ( ipf2 == -1 ) continue;


            for ( int ipi = 0; ipi < g_seq_source_momentum_number; ipi++ ) {
 
              int pi2[3] = {
                g_seq_source_momentum_list[ipi][0],
                g_seq_source_momentum_list[ipi][1],
                g_seq_source_momentum_list[ipi][2] };

              int pi1[3] = {
                  -ptot[0] - pi2[0],
                  -ptot[1] - pi2[1],
                  -ptot[2] - pi2[2] };
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
              if ( iconf == 0 && isrc == 0 ) {
                memcpy ( momentum_list[iptot][momentum_config_counter][0], pf1, 3*sizeof(int) );
                memcpy ( momentum_list[iptot][momentum_config_counter][1], pi1, 3*sizeof(int) );
              }
#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact_twop", "momentum-filter", io_proc == 2 );
#endif

              /***********************************************************
               * 
               ***********************************************************/
              memset ( corr_buffer, 0, T_global * sizeof ( double _Complex ) );
   
              for ( int isample = 0; isample < g_nsample_oet; isample++ ) {

                /* for ( int i = 0; i < 4; i++ ) {
                for ( int k = 0; k < 4; k++ ) {
                  for ( int a = 0; a < 3; a++ ) { */
                    for ( int it = 0; it < T_global; it++ ) 
                    {
                      corr_buffer[it] += corr_pta[ipf][it] * corr_oet[isample][ipi][ipf2][it];
                    }
                  /* }
                }} */
              }

              /***********************************************************
               * normalize with number of stochastic samples
               ***********************************************************/
              for ( int it = 0; it < T_global; it++ ) 
              {
                corr_buffer[it] /= (double)g_nsample_oet;
              }

              /***********************************************************/
              /***********************************************************/

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
              for ( int it = 0; it < T_global; it++ ) {

                /* int const tt  = ( csx[0] + it + T_global ) % T_global; */
                int const tt  = it;
                double _Complex const zbuffer = corr_buffer[tt] * ephase;
              
                corr_mxm_mxm[iptot][icoh][momentum_config_counter][2*it  ] += creal ( zbuffer );
                corr_mxm_mxm[iptot][icoh][momentum_config_counter][2*it+1] += cimag ( zbuffer );
              }
#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact_twop", "source-phase-and-reorder", io_proc == 2 );
#endif

            }  /* end of loop on pi2 */
 
          }  /* end of loop on pf1 */

        } /* total momenta */
    
        fini_2level_ztable ( &corr_pta );
        fini_4level_ztable ( &corr_oet );

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
          sprintf ( filename, "%s.px%d_py%d_pz%d.h5", g_outfile_prefix, ptot[0], ptot[1], ptot[2] );

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
            int const cdim[2] = { momentum_number[iptot],  2 * T_global };

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

      }  /* coherent sources */
    }  /* end of loop on base sources */

    gettimeofday ( &conf_etime, (struct timezone *)NULL );
    show_time ( &conf_stime, &conf_etime, "htpp_compact_twop", "time-per-conf", io_proc == 2 );

  }  /* end of loop on configs */


  free ( corr_mxm_mxm );


}  /* end of if MXM_MXM_OET */



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
