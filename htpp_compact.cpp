/****************************************************
 * htpp_compact
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

#ifndef M_J_M
#define M_J_M 0
#endif

#ifndef MXM_J_M
#define MXM_J_M 1
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
 * momentum filter
 ***************************************************************************/
inline int momentum_filter ( int * const pf, int * const pc, int * const pi1, int * const pi2, int const pp_max ) {

  /* check mometnum conservation  */
  if ( pf == NULL || pc == NULL ) return ( 1 == 0 );

  if ( pi2 == NULL && pc == NULL ) {

    int const is_conserved = ( pi1[0] + pf[0] == 0 ) && ( pi1[1] + pf[1] == 0 ) && ( pi1[2] + pf[2] == 0 );

    int const is_lessequal = \
            ( pi1[0] * pi1[0] + pi1[1] * pi1[1] + pi1[2] * pi1[2] <= pp_max ) \
        &&  ( pf[0]  * pf[0]  + pf[1]  * pf[1]  + pf[2]  * pf[2]  <= pp_max );

    return ( is_conserved && is_lessequal );

  } else if ( pc != NULL ) {
    if ( pi2 == NULL ) {
  
      int const is_conserved = ( pi1[0] + pf[0] + pc[0] == 0 ) && \
                               ( pi1[1] + pf[1] + pc[1] == 0 ) && \
                               ( pi1[2] + pf[2] + pc[2] == 0 );

      int const is_lessequal = \
            ( pi1[0] * pi1[0] + pi1[1] * pi1[1] + pi1[2] * pi1[2]  <= pp_max ) \
         && ( pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]  <= pp_max ) \
         && ( pf[0] * pf[0] + pf[1] * pf[1] + pf[2] * pf[2]  <= pp_max );

      return ( is_conserved && is_lessequal );

    } else {

      int const is_conserved = ( pi1[0] + pi2[0] + pf[0] + pc[0] == 0 ) &&
                             ( pi1[1] + pi2[1] + pf[1] + pc[1] == 0 ) &&
                             ( pi1[2] + pi2[2] + pf[2] + pc[2] == 0 );

      int const is_lessequal = \
           ( pi1[0] * pi1[0] + pi1[1] * pi1[1] + pi1[2] * pi1[2]  <= pp_max ) \
        && ( pi2[0] * pi2[0] + pi2[1] * pi2[1] + pi2[2] * pi2[2]  <= pp_max ) \
        && ( pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]  <= pp_max ) \
        && ( pf[0] * pf[0] + pf[1] * pf[1] + pf[2] * pf[2]  <= pp_max );

      return ( is_conserved && is_lessequal );
    }
  } else {
    return ( 1 == 0 );
  }

} /* end of mometnum_filter */

/***********************************************************
 *
 ***********************************************************/
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
      fprintf ( stdout, "# [htpp_compact] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [htpp_compact] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [htpp_compact] ensemble name set to %s\n", ensemble_name );
      break;
    case 'D':
      strcpy ( diagram_name, optarg );
      fprintf ( stdout, "# [htpp_compact] diagram_name set to %s\n", diagram_name );
      break;
    case 'C':
      strcpy ( correlator_name, optarg );
      fprintf ( stdout, "# [htpp_compact] correlator_name set to %s\n", correlator_name );
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
  /* fprintf(stdout, "# [htpp_compact] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [htpp_compact] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [htpp_compact] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [htpp_compact] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[htpp_compact] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[htpp_compact] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[htpp_compact] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [htpp_compact] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[htpp_compact] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[htpp_compact] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [htpp_compact] comment %s\n", line );
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
    fprintf ( stdout, "# [htpp_compact] conf_src_list conf t x y z\n" );
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
  int const n_tc = g_src_snk_time_separation + 1;

  int * momentum_number = init_1level_itable ( g_total_momentum_number );
  int **** momentum_list = (int****) malloc ( g_total_momentum_number * sizeof ( int*** ) );

  double _Complex * corr_buffer = init_1level_ztable ( T_global );
  if ( corr_buffer == NULL ) {
    fprintf( stderr, "[htpp_compact] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(2);
  }

#if M_J_M
  double ****** corr_mjm = (double******) malloc ( g_total_momentum_number * sizeof( double*****) );

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
   
      struct AffReader_s * affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[htpp_compact] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
        EXIT(5);
      } else {
        if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact] Reading data from file %s\n", filename );
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

        int pi1[3] = {
            -ptot[0],
            -ptot[1],
            -ptot[2] };



        if( iconf == 0 && isrc == 0 ) {

          momentum_number[iptot] = 0;

          /***********************************************************
           * loop on sink momenta
           ***********************************************************/
          for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

            int pf[3] = {
                g_sink_momentum_list[ipf][0],
                g_sink_momentum_list[ipf][1],
                g_sink_momentum_list[ipf][2] };

            int pc[3] = {
                  ptot[0] - pf[0],
                  ptot[1] - pf[1],
                  ptot[2] - pf[2] };
 
            if ( momentum_filter ( pf, pc, pi1, NULL , max_single_particle_momentum_squared ) ) momentum_number[iptot]++;
          }
 
          fprintf ( stdout, "# [htpp_compact] number of momentum combinations = %d %s %d\n", momentum_number[iptot], __FILE__, __LINE__ );
          momentum_list[iptot] = init_3level_itable ( momentum_number[iptot], 2, 3 );
        }

        corr_mjm[iptot] = init_5level_dtable ( g_coherent_source_number, gamma_rho_number,  gamma_v_number, momentum_number[iptot],  2 * n_tc );
        if ( corr_mjm[iptot] == NULL ) {
          fprintf( stderr, "[htpp_compact] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(2);
        }

        /***************************************************************************
         * loop on rho components
         ***************************************************************************/
        for ( int irho = 0; irho < gamma_rho_number; irho++ ) {

          /***************************************************************************
           * loop on vector components
           ***************************************************************************/
          for ( int iv = 0; iv < gamma_v_number; iv++ ) {

           int momentum_config_counter = -1;
            /***********************************************************
             * loop on sink momenta
             ***********************************************************/
            for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

              int pf[3] = {
                  g_sink_momentum_list[ipf][0],
                  g_sink_momentum_list[ipf][1],
                  g_sink_momentum_list[ipf][2] };
  
              int pc[3] = {
                    ptot[0] - pf[0],
                    ptot[1] - pf[1],
                    ptot[2] - pf[2] };
 
#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif
              /***********************************************************
               * set twop function momenta an filter
               ***********************************************************/
              if ( ! momentum_filter ( pf, pc, pi1, NULL , max_single_particle_momentum_squared ) ) continue;
              momentum_config_counter++;

              /***********************************************************
               * add momentum config to list
               ***********************************************************/
              if ( iconf == 0 && isrc == 0 && iv == 0 && irho == 0 ) {
                memcpy ( momentum_list[iptot][momentum_config_counter][0], pf, 3*sizeof(int) );
                memcpy ( momentum_list[iptot][momentum_config_counter][1], pi1, 3*sizeof(int) );
              }

                
              /***********************************************************
               * aff key for reading data
               ***********************************************************/

              char key[500];
              sprintf ( key,
                        "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/g1_%s/g2_%s/PX%d_PY%d_PZ%d",
                        /* "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/g1_%s/g2_%s/x%d_y%d_z%d", */
                        diagram_name,
                        pf[0], pf[1], pf[2],
                        gamma_bin_to_name[gamma_p_list[0]], g_src_snk_time_separation,
                        gamma_bin_to_name[gamma_v_list[iv]], gamma_bin_to_name[gamma_rho_list[irho]],
                        pc[0], pc[1], pc[2] );
#if 0

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
              if ( g_verbose > 2 ) {
                fprintf ( stdout, "# [htpp_compact] key = %s %s %d\n", key , __FILE__, __LINE__ );
              }
#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact", "momentum-filter-and-key", io_proc == 2 );
#endif
              /***********************************************************/
              /***********************************************************/
#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif

              exitstatus = read_aff_contraction (  corr_buffer, affr, NULL, key, T_global );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[htpp_compact] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(12);
              }

#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact", "read_aff_contraction", io_proc == 2 );
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
                 * source phase factor
                 * and order from source
                 ***********************************************************/
                double _Complex const ephase = cexp ( 2. * M_PI * ( 
                        pi1[0] * csx[1] / (double)LX_global 
                      + pi1[1] * csx[2] / (double)LY_global 
                      + pi1[2] * csx[3] / (double)LZ_global ) * I );
            
                if ( g_verbose > 4 ) fprintf ( stdout, "# [htpp_compact] pi1 = %3d %3d %3d csx = %3d %3d %3d  ephase = %16.7e %16.7e\n",
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
              
                  corr_mjm[iptot][icoh][irho][iv][momentum_config_counter][2*it  ] = creal ( zbuffer );
                  corr_mjm[iptot][icoh][irho][iv][momentum_config_counter][2*it+1] = cimag ( zbuffer );

                }

#if TIMERS
                gettimeofday ( &tb, (struct timezone *)NULL );
                show_time ( &ta, &tb, "htpp_compact", "source-phase-and-reorder", io_proc == 2 );
#endif
              }  /* coherent sources */
            }  /* end of loop on pf */
          }  /* vector insertion */
        }  /* rho */

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
        sprintf ( filename, "%s.dt%d.px%d_py%d_pz%d.h5", correlator_name, g_src_snk_time_separation, ptot[0], ptot[1], ptot[2] );

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
            fprintf ( stderr, "[htpp_compact] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }

          char gamma_tag[200];
          sprintf( gamma_tag, "/gamma_rho" );
          exitstatus = write_h5_contraction ( (void*)(gamma_rho_list), NULL, filename, gamma_tag, "int", 1, &gamma_rho_number );
          if( exitstatus != 0 ) {
            fprintf ( stderr, "[htpp_compact] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
  
          sprintf( gamma_tag, "/gamma_v" );
          exitstatus = write_h5_contraction ( (void*)(gamma_v_list), NULL, filename, gamma_tag, "int", 1, &gamma_v_number );
          if( exitstatus != 0 ) {
            fprintf ( stderr, "[htpp_compact] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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
            fprintf ( stdout, "# [htpp_compact] h5 key = %s %s %d\n", key , __FILE__, __LINE__ );
          }

          int const ndim = 4;
          int const cdim[4] = { gamma_rho_number, gamma_v_number, momentum_number[iptot],  2 * n_tc };

          exitstatus = write_h5_contraction ( corr_mjm[iptot][icoh][0][0][0], NULL, filename, key, "double", ndim, cdim );
          if ( exitstatus != 0) {
            fprintf ( stderr, "[http_compact] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(123);
          }
#if TIMERS
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "htpp_compact", "write_h5_contraction", io_proc == 2 );
#endif
        }  /* end of loop on coherent sources */

        fini_5level_dtable ( &( corr_mjm[iptot] ) );
        fini_3level_itable ( &( momentum_list[iptot] ) );

      }  /* end of loop on sink momenta */

    }  /* end of loop on base sources */
  }  /* end of loop on configs */
  free ( corr_mjm );
#endif  /* of if M_J_M */                 


#if MXM_J_M

  double ***** corr_mxmjm = (double*****) malloc ( g_total_momentum_number * sizeof( double****) );

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
      sprintf ( filename, "%s/stream_%c/%d/%s.%.4d.t%d_x%d_y%d_z%d.aff", filename_prefix, stream, Nconf, filename_prefix2, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
   
      struct AffReader_s * affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[htpp_compact] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
        EXIT(5);
      } else {
        if ( g_verbose > 1 ) fprintf ( stdout, "# [htpp_compact] Reading data from file %s\n", filename );
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
                  -ptot[0] - pi1[0],
                  -ptot[1] - pi1[1],
                  -ptot[2] - pi1[2] };

              if ( momentum_filter ( pf, pc, pi1, pi2 , max_single_particle_momentum_squared ) ) momentum_number[iptot]++;
            }
          }
 
          fprintf ( stdout, "# [htpp_compact] number of momentum combinations = %d %s %d\n", momentum_number[iptot], __FILE__, __LINE__ );
          momentum_list[iptot] = init_3level_itable ( momentum_number[iptot], 2, 3 );
        }

        corr_mxmjm[iptot] = init_4level_dtable ( g_coherent_source_number, gamma_v_number, momentum_number[iptot],  2 * n_tc );
        if ( corr_mxmjm[iptot] == NULL ) {
          fprintf( stderr, "[htpp_compact] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(2);
        }

        /***************************************************************************
         * loop on vector components
         ***************************************************************************/
        for ( int iv = 0; iv < gamma_v_number; iv++ ) {

           int momentum_config_counter = -1;
            /***********************************************************
             * loop on sink momenta
             ***********************************************************/
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
                  -ptot[0] - pi1[0],
                  -ptot[1] - pi1[1],
                  -ptot[2] - pi1[2] };
#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif
              /***********************************************************
               * set twop function momenta an filter
               ***********************************************************/
              if ( ! momentum_filter ( pf, pc, pi1, pi2 , max_single_particle_momentum_squared ) ) continue;
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

              sprintf ( key,
                        "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/pi2x%dpi2y%dpi2z%d/gi2_%s/g1_%s/g2_%s/PX%d_PY%d_PZ%d",
                        /* "/%s/pfx%dpfy%dpfz%d/gf_%s/dt%d/pi2x%dpi2y%dpi2z%d/gi2_%s/g1_%s/g2_%s/x%d_y%d_z%d", */
                        diagram_name,
                        pf[0], pf[1], pf[2],
                        gamma_bin_to_name[gamma_p_list[0]], g_src_snk_time_separation,
                        pi2[0], pi2[1], pi2[2], gamma_bin_to_name[gamma_p_list[0]],
                        gamma_bin_to_name[gamma_v_list[0]], gamma_bin_to_name[gamma_p_list[0]],
                        pc[0], pc[1], pc[2] );

              if ( g_verbose > 2 ) {
                fprintf ( stdout, "# [htpp_compact] key = %s %s %d\n", key , __FILE__, __LINE__ );
              }
#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact", "momentum-filter-and-key", io_proc == 2 );
#endif
              /***********************************************************/
              /***********************************************************/
#if TIMERS
              gettimeofday ( &ta, (struct timezone *)NULL );
#endif

              exitstatus = read_aff_contraction (  corr_buffer, affr, NULL, key, T_global );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[htpp_compact] Error from read_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(12);
              }

#if TIMERS
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "htpp_compact", "read_aff_contraction", io_proc == 2 );
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
                 * source phase factor
                 * and order from source
                 ***********************************************************/
                double _Complex const ephase = cexp ( 2. * M_PI * ( 
                        pi1[0] * csx[1] / (double)LX_global 
                      + pi1[1] * csx[2] / (double)LY_global 
                      + pi1[2] * csx[3] / (double)LZ_global ) * I );
            
                if ( g_verbose > 4 ) fprintf ( stdout, "# [htpp_compact] pi1 = %3d %3d %3d csx = %3d %3d %3d  ephase = %16.7e %16.7e\n",
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
              
                  corr_mxmjm[iptot][icoh][iv][momentum_config_counter][2*it  ] = creal ( zbuffer );
                  corr_mxmjm[iptot][icoh][iv][momentum_config_counter][2*it+1] = cimag ( zbuffer );

                }

#if TIMERS
                gettimeofday ( &tb, (struct timezone *)NULL );
                show_time ( &ta, &tb, "htpp_compact", "source-phase-and-reorder", io_proc == 2 );
#endif
              }  /* coherent sources */
            }  /* end of loop on pi1 */
            }  /* end of loop on pf */
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
        sprintf ( filename, "%s.dt%d.px%d_py%d_pz%d.h5", correlator_name, g_src_snk_time_separation, ptot[0], ptot[1], ptot[2] );

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
            fprintf ( stderr, "[htpp_compact] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }

          char gamma_tag[200];
          sprintf( gamma_tag, "/gamma_v" );
          exitstatus = write_h5_contraction ( (void*)(gamma_v_list), NULL, filename, gamma_tag, "int", 1, &gamma_v_number );
          if( exitstatus != 0 ) {
            fprintf ( stderr, "[htpp_compact] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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
            fprintf ( stdout, "# [htpp_compact] h5 key = %s %s %d\n", key , __FILE__, __LINE__ );
          }

          int const ndim = 3;
          int const cdim[3] = { gamma_v_number, momentum_number[iptot],  2 * n_tc };

          exitstatus = write_h5_contraction ( corr_mxmjm[iptot][icoh][0][0], NULL, filename, key, "double", ndim, cdim );
          if ( exitstatus != 0) {
            fprintf ( stderr, "[http_compact] Error from write_h5_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(123);
          }
#if TIMERS
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "htpp_compact", "write_h5_contraction", io_proc == 2 );
#endif
        }  /* end of loop on coherent sources */

        fini_4level_dtable ( &( corr_mxmjm[iptot] ) );
        fini_3level_itable ( &( momentum_list[iptot] ) );

      }  /* end of loop on sink momenta */

    }  /* end of loop on base sources */

    gettimeofday ( &conf_etime, (struct timezone *)NULL );
    show_time ( &conf_stime, &conf_etime, "htpp_compact", "time-per-conf", io_proc == 2 );
  }  /* end of loop on configs */


  free ( corr_mxmjm );

#endif  /* of if MXM_J_M */                 

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
  show_time ( &start_time, &end_time, "htpp_compact", "total-time", io_proc == 2 );

  return(0);

}
