/****************************************************
 * twop_analyse.c
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
#include <sys/time.h>
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
#include "contract_cvc_tensor.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "table_init_i.h"
#include "uwerr.h"
#include "derived_quantities.h"

using namespace cvc;

inline int get_momentum_id ( int const q[3], int ** const p, int const n )
{
  int id = -1;
  for ( int i = 0; i < n; i++ ) {
    if ( ( q[0] == p[i][0] ) && ( q[1] == p[i][1] ) && ( q[2] == p[i][2] )  )
    {
      id = i;
      break;
    }
  }

  if ( id == -1 ) {
    fprintf(stderr, "[get_momentum_id] Error, momentum %3d %3d %3d not found   %s %d\n", q[0], q[1], q[2], __FILE__, __LINE__);
  } else if (g_verbose > 4 ) {
    fprintf( stdout, "# [get_momentum_id] momentum %3d %3d %3d id %2d    %s %d\n", q[0], q[1], q[2], id, __FILE__, __LINE__);
  }

  return(id);
}

void usage() {
  fprintf(stdout, "Code to analyse P-J-J correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  EXIT(0);
}

#define _TWOP_AFF_SINGLE 0
#define _TWOP_AFF_MULT   0
#define _TWOP_H5_SINGLE  0
#define _TWOP_H5_BLOCK   0
#define _TWOP_AFF_OET    0
#define _CVC_H5          0
#define _TWOP_H5_OET     1

int main(int argc, char **argv) {
  
  double const TWO_MPI = 2. * M_PI;

  char const reim_str[2][3] = {"re" , "im"};

  char const twop_correlator_prefix[3][20] = { "local-local" , "neutral", "charged" };

#if _TWOP_H5_OET
  char const twop_flavor_tag[4][20]        = { "d+-g-u-g" , "u+-g-u-g" , "l-gf-l-gi" , "s-gf-s-gi" };
#else
  char const twop_flavor_tag[5][20]        = { "u-gf-u-gi" , "d-gf-u-gi" , "u-gf-d-gi" , "d-gf-d-gi" , "u-v-u-v" };
#endif

#if _TWOP_H5_OET
  char const twop_flavor_tag[2][20]        = { "d+-g-u-g" , "u+-g-u-g" };
#else
  char const twop_flavor_tag[5][20]        = { "u-gf-u-gi" , "d-gf-u-gi" , "u-gf-d-gi" , "d-gf-d-gi" , "u-v-u-v" };
#endif

  char const gamma_id_to_ascii[16][10] = {
    "gt",
    "gx",
    "gy",
    "gz",
    "id",
    "g5",
    "gtg5",
    "gxg5",
    "gyg5",
    "gzg5",
    "gtgx",
    "gtgy",
    "gtgz",
    "gxgy",
    "gxgz",
    "gygz" 
  };



  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[500];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "cA211a.30.32";
  int fold_correlator= 0;
  struct timeval ta, tb;
  int correlator_type = -1;
  int flavor_type = -1;
  int write_data = 0;
  double twop_operator_norm[2] = {1., 1.};
  double muval[2] = {0.,0.};

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:F:c:s:E:w:n:m:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [twop_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [twop_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'F':
      fold_correlator = atoi ( optarg );
      fprintf ( stdout, "# [twop_analyse] fold_correlator set to %d\n", fold_correlator );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [twop_analyse] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'c':
      correlator_type = atoi ( optarg );
      fprintf ( stdout, "# [twop_analyse] correlator_type set to %d\n", correlator_type );
      break;
    case 's':
      flavor_type = atoi ( optarg );
      fprintf ( stdout, "# [twop_analyse] flavor_type set to %d\n", flavor_type );
      break;
    case 'w':
      write_data = atoi( optarg );
      fprintf ( stdout, "# [twop_analyse] write_data set to %d\n", write_data );
      break;
    case 'n':
      sscanf ( optarg, "%lf,%lf", twop_operator_norm, twop_operator_norm+1 );
      fprintf ( stdout, "# [twop_analyse] twop_operator_norm set to %e  %e\n", twop_operator_norm[0], twop_operator_norm[1] );
      break;
    case 'm':
      sscanf ( optarg, "%lf,%lf", muval, muval+1 );
      fprintf ( stdout, "# [twop_analyse] muval set to %e  %e\n", muval[0], muval[1] );
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
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [twop_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [twop_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [twop_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [twop_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[twop_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[twop_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[twop_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [twop_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[twop_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  read_source_coords_list ( conf_src_list, num_conf, num_src_per_conf, ensemble_name );

  /***********************************************************
   ***********************************************************
   **
   ** TWOPT
   **
   ***********************************************************
   ***********************************************************/
#if _CVC_H5
  
  sprintf ( filename, "stream_%c/%s/%d/%s.%.4d.t%.2dx%.2dy%.2dz%.2d.h5", 
          conf_src_list[0][0][0],
          filename_prefix, 
          conf_src_list[0][0][1],
          filename_prefix2,
          conf_src_list[0][0][1], 
          conf_src_list[0][0][2], conf_src_list[0][0][3], conf_src_list[0][0][4], conf_src_list[0][0][5] );

  fprintf(stdout, "# [twop_analyse] reading data from file %s\n", filename);

  char momentum_tag[12] = "/mom_snk";
  int * momentum_buffer = NULL;
  size_t * momentum_cdim = NULL, momentum_ncdim = 0;

  exitstatus = read_from_h5_file_varsize ( (void**)&momentum_buffer, filename, momentum_tag,  "int", &momentum_ncdim, &momentum_cdim,  io_proc );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[twop_analyse] Error from read_from_h5_file_varsize for file %s key %s   %s %d\n", filename, momentum_tag, __FILE__, __LINE__);
    EXIT(15);
  }

  if ( momentum_ncdim != 2 || momentum_cdim[1] != 3 ) {
    fprintf ( stderr, "[twop_analyse] Error from read_from_h5_file_varsize for file data %s %d\n", __FILE__, __LINE__ );
    EXIT(129);
  }

  int const momentum_number = (int)(momentum_cdim[0]);
  if ( g_verbose > 4 ) fprintf ( stdout, "# [p2gg_analyse] read %d momenta %s %d\n", momentum_number, __FILE__, __LINE__ );
  int ** momentum_list = init_2level_itable ( momentum_number, 3 );
  memcpy ( momentum_list[0], momentum_buffer, momentum_number * 3 * sizeof ( int ) );
  free ( momentum_buffer );
  free ( momentum_cdim );

  char gamma_tag[12] = "/gamma_v";
  int * gamma_buffer = NULL;
  size_t * gamma_cdim = NULL, gamma_ncdim = 0;

  exitstatus = read_from_h5_file_varsize ( (void**)&gamma_buffer, filename, gamma_tag,  "int", &gamma_ncdim, &gamma_cdim,  io_proc );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[twop_analyse] Error from read_from_h5_file_varsize for file %s key %s   %s %d\n", filename, gamma_tag, __FILE__, __LINE__);
    EXIT(15);
  }

  if ( gamma_ncdim != 1 ) {
    fprintf ( stderr, "[twop_analyse] Error from read_from_h5_file_varsize for file data %s %d\n", __FILE__, __LINE__ );
    EXIT(129);
  }

  int const gamma_v_number = (int)(gamma_cdim[0]);
  if ( g_verbose > 4 ) fprintf ( stdout, "# [p2gg_analyse] read %d gamma_v %s %d\n", gamma_v_number, __FILE__, __LINE__ );
  for ( int i = 0; i < gamma_v_number; i++ ) {
    g_sink_gamma_id_list[i]   = gamma_buffer[i];
    g_source_gamma_id_list[i] = gamma_buffer[i];
  }
  g_sink_gamma_id_number = gamma_v_number;
  g_source_gamma_id_number = gamma_v_number;
  free ( gamma_buffer );
  free ( gamma_cdim );
 
#endif  /* of _CVC_H5 */

  double ****** corr = init_6level_dtable ( num_conf, num_src_per_conf, g_sink_momentum_number, g_sink_gamma_id_number, g_source_gamma_id_number, 2 * T_global );
  if ( corr == NULL ) {
    fprintf ( stderr, "[twop_analyse] Error from init_Xlevel_dtable   %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }

#if _TWOP_AFF_MULT
  /***********************************************************
   * loop on configs and source locations per config
   ***********************************************************/
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      Nconf = conf_src_list[iconf][isrc][1];

      /***********************************************************
       * copy source coordinates
       ***********************************************************/
      int const gsx[4] = {
          conf_src_list[iconf][isrc][2],
          conf_src_list[iconf][isrc][3],
          conf_src_list[iconf][isrc][4],
          conf_src_list[iconf][isrc][5] };

      /***********************************************
       * reader for aff input file
       ***********************************************/

      gettimeofday ( &ta, (struct timezone *)NULL );
      struct AffNode_s *affn = NULL, *affdir = NULL;

      /* sprintf ( filename, "%d/%s.%.4d.t%d.aff", Nconf, g_outfile_prefix, Nconf, gsx[0] ); */
      sprintf ( filename, "stream_%c/%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", conf_src_list[iconf][isrc][0], g_outfile_prefix, Nconf, 
          gsx[0], gsx[1], gsx[2], gsx[3] );

      fprintf(stdout, "# [twop_analyse] reading data from file %s\n", filename);

      struct AffReader_s *affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[twop_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }

      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[twop_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        return(103);
      }

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "twop_analyse", "open-init-aff-reader", g_cart_id == 0 );

      for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ ) {

        int const sink_gamma_id = g_sink_gamma_id_list[ isink_gamma ];

      for ( int isource_gamma = 0; isource_gamma < g_source_gamma_id_number; isource_gamma++ ) {

        int const source_gamma_id = g_source_gamma_id_list[ isource_gamma ];

        /**********************************************************
         * loop on momenta
         **********************************************************/
        for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

          memset( corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma], 0, 2 * T_global * sizeof(double) );

          double * buffer = init_1level_dtable( 2 * T );
          if( buffer == NULL ) {
            fprintf(stderr, "[twop_analyse] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(15);
          }

          int const parity_num = ( flavor_type == 0 || flavor_type == 3 ) ? 1 : 2;
          double const parity_norm = 1. / (double)parity_num;
          if ( g_verbose > 4 ) fprintf ( stdout, "# [twop_analyse] parity_num = %d parity_norm = %e     %s %d\n", parity_num,
             parity_norm, __FILE__, __LINE__);

          for ( int ip = 0; ip < parity_num; ip++ )  {

            int const parity_sign = 1 - 2*ip;

            int ifl = -1;
            if  ( flavor_type == 0 || flavor_type == 3 ) {
              ifl = flavor_type;
            } else if  ( flavor_type == 1 )  { 
              ifl = flavor_type + ip;
            } else if  ( flavor_type == 2 )  { 
              ifl = flavor_type - ip;
            }

            int const sink_momentum[3] = {
              parity_sign * g_sink_momentum_list[isink_momentum][0],
              parity_sign * g_sink_momentum_list[isink_momentum][1],
              parity_sign * g_sink_momentum_list[isink_momentum][2] };

            gettimeofday ( &ta, (struct timezone *)NULL );

            /* sprintf ( key , "/%s/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d/gi%.2d/px%dpy%dpz%d", twop_correlator_prefix[ correlator_type ], twop_flavor_tag[ flavor_type ],
                gsx[0], gsx[1], gsx[2], gsx[3], sink_gamma_id, source_gamma_id, sink_momentum[0], sink_momentum[1], sink_momentum[2] ); */

            sprintf ( key , "/%s/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d/gi%.2d/px%dpy%dpz%d", twop_correlator_prefix[ correlator_type ], twop_flavor_tag[ ifl ],
                gsx[0], gsx[1], gsx[2], gsx[3], sink_gamma_id, source_gamma_id, sink_momentum[0], sink_momentum[1], sink_momentum[2] );

            if ( g_verbose > 2 ) fprintf ( stdout, "# [twop_analyse] key = %s\n", key );

            affdir = aff_reader_chpath (affr, affn, key );
            uint32_t uitems = (uint32_t)T_global;
            exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer), uitems );
            if( exitstatus != 0 ) {
              fprintf(stderr, "[twop_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }

            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "twop_analyse", "read-aff-key", g_cart_id == 0 );

            /**********************************************************
             * source phase
             **********************************************************/
            double const p[3] = {
                TWO_MPI * (double)sink_momentum[0] / (double)LX_global,
                TWO_MPI * (double)sink_momentum[1] / (double)LY_global,
                TWO_MPI * (double)sink_momentum[2] / (double)LZ_global };

            double const phase = -( p[0] * gsx[1] + p[1] * gsx[2] + p[2] * gsx[3] ) ;

            double _Complex const ephase = cexp ( phase * I );

            /**********************************************************
             * sort data from buffer into hvp,
             * add source phase
             **********************************************************/
#pragma omp parallel for
            for ( int it = 0; it < T; it++ ) {
              int const tt = ( it - gsx[0] + T_global ) % T_global; 

              double _Complex ztmp = ( buffer[2*it] +  buffer[2*it+1] * I ) * ephase;

              corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt  ] += creal( ztmp ) * twop_operator_norm[0] * twop_operator_norm[1] * parity_norm;
              corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt+1] += cimag( ztmp ) * twop_operator_norm[0] * twop_operator_norm[1] * parity_norm;
            }

          }  /* end of loop on ip */
          
          fini_1level_dtable( &buffer );

        }  /* end of loop on sink momenta */

      }  /* end of loop on source gamma id */

      }  /* end of loop on sink gamma id */

      aff_reader_close ( affr );

    }  /* end of loop on source locations */

  }   /* end of loop on configurations */

#endif  /* of _TWOP_AFF_MULT */

#if _TWOP_AFF_SINGLE
  for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ ) {

    int const sink_gamma_id = g_sink_gamma_id_list[ isink_gamma ];

    for ( int isource_gamma = 0; isource_gamma < g_source_gamma_id_number; isource_gamma++ ) {

      int const source_gamma_id = g_source_gamma_id_list[ isource_gamma ];

      /**********************************************************
        * loop on momenta
       **********************************************************/
      for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

        for ( int ip = 0; ip <=1; ip++ ) {

          int const parity_sign = 1 - 2 * ip;

          int const sink_momentum[3] = {
              parity_sign * g_sink_momentum_list[isink_momentum][0],
              parity_sign * g_sink_momentum_list[isink_momentum][1],
              parity_sign * g_sink_momentum_list[isink_momentum][2] };

          int ifl = -1;
          if  ( flavor_type == 0 || flavor_type == 3 ) {
            ifl = flavor_type;
          } else if  ( flavor_type == 1 )  {
            ifl = flavor_type + ip;
          } else if  ( flavor_type == 2 )  {
            ifl = flavor_type - ip;
          }

          gettimeofday ( &ta, (struct timezone *)NULL );

          sprintf ( filename, "%s/%s.%s.gf%.2d.gi%.2d.px%dpy%dpz%d.aff", filename_prefix, twop_correlator_prefix[correlator_type],
              twop_flavor_tag[ ifl ], sink_gamma_id, source_gamma_id, sink_momentum[0], sink_momentum[1], sink_momentum[2] );

          /***********************************************
           * reader for aff input file
           ***********************************************/

          struct AffNode_s *affn = NULL, *affdir = NULL;
  
          printf(stdout, "# [twop_analyse] reading data from file %s\n", filename);
          struct AffReader_s * affr = aff_reader ( filename );
          const char * aff_status_str = aff_reader_errstr ( affr );
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[twop_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
            EXIT(15);
          }

          if( (affn = aff_reader_root( affr )) == NULL ) {
            fprintf(stderr, "[twop_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
            return(103);
          }

          double * buffer = init_1level_dtable( 2 * T );
          if( buffer == NULL ) {
            fprintf(stderr, "[twop_analyse] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(15);
          }

          /***********************************************************
           * loop on configs and source locations per config
           ***********************************************************/
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

              /***********************************************************
               * copy source coordinates
               ***********************************************************/
              int const gsx[4] = {
                  conf_src_list[iconf][isrc][2],
                  conf_src_list[iconf][isrc][3],
                  conf_src_list[iconf][isrc][4],
                  conf_src_list[iconf][isrc][5] };
  
              sprintf ( key , "/stream_%c/conf_%d/t%.2dx%.2dy%.2dz%.2d",conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1],
                  conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );
  
              if ( g_verbose > 2 ) fprintf ( stdout, "# [twop_analyse] key = %s\n", key );
  
              affdir = aff_reader_chpath (affr, affn, key );
              if( affdir == NULL ) {
                fprintf(stderr, "[twop_analyse] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
                EXIT(105);
              }
             uint32_t uitems = (uint32_t)T_global;
              exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer), uitems );
              if( exitstatus != 0 ) {
                fprintf(stderr, "[twop_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }
  
              /**********************************************************
               * source phase
               **********************************************************/
              double const p[3] = {
                  TWO_MPI * (double)sink_momentum[0] / (double)LX_global,
                  TWO_MPI * (double)sink_momentum[1] / (double)LY_global,
                  TWO_MPI * (double)sink_momentum[2] / (double)LZ_global };
  
              double const phase = -( p[0] * gsx[1] + p[1] * gsx[2] + p[2] * gsx[3] ) ;
  
              double _Complex const ephase = cexp ( phase * I );
  
              /**********************************************************
               * sort data from buffer into hvp,
               * add source phase
               **********************************************************/
#pragma omp parallel for
              for ( int it = 0; it < T; it++ ) {
                int const tt = ( it - gsx[0] + T_global ) % T_global; 

                double _Complex ztmp = ( buffer[2*it] +  buffer[2*it+1] * I ) * ephase;
  
                corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt  ] += creal( ztmp ) * twop_operator_norm[0] * twop_operator_norm[1];
                corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt+1] += cimag( ztmp ) * twop_operator_norm[0] * twop_operator_norm[1];
              }
        
            }  /* end of loop on source locations */
          }   /* end of loop on configurations */

          aff_reader_close ( affr );
          fini_1level_dtable( &buffer );

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "twop_analyse", "open-init-aff-reader", g_cart_id == 0 );

        }  /* end of loop on parity */

      }  /* end of loop on sink momenta */

    }  /* end of loop on source gamma id */

  }  /* end of loop on sink gamma id */

#endif  /* of _TWOP_AFF_SINGLE */

#if _TWOP_H5_SINGLE
  for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ ) {

    int const sink_gamma_id = g_sink_gamma_id_list[ isink_gamma ];

    for ( int isource_gamma = 0; isource_gamma < g_source_gamma_id_number; isource_gamma++ ) {

      int const source_gamma_id = g_source_gamma_id_list[ isource_gamma ];

      /**********************************************************
        * loop on momenta
       **********************************************************/
      for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

        for ( int ip = 0; ip <=1; ip++ ) {

          int const parity_sign = 1 - 2 * ip;

          int const sink_momentum[3] = {
              parity_sign * g_sink_momentum_list[isink_momentum][0],
              parity_sign * g_sink_momentum_list[isink_momentum][1],
              parity_sign * g_sink_momentum_list[isink_momentum][2] };

          int ifl = -1;
          if  ( flavor_type == 0 || flavor_type == 3 ) {
            ifl = flavor_type;
          } else if  ( flavor_type == 1 )  {
            ifl = flavor_type + ip;
          } else if  ( flavor_type == 2 )  {
            ifl = flavor_type - ip;
          }

          gettimeofday ( &ta, (struct timezone *)NULL );

          sprintf ( filename, "%s/%s.%s.gf%.2d.gi%.2d.px%dpy%dpz%d.h5", filename_prefix, twop_correlator_prefix[correlator_type],
              twop_flavor_tag[ ifl ], sink_gamma_id, source_gamma_id, sink_momentum[0], sink_momentum[1], sink_momentum[2] );

          fprintf(stdout, "# [twop_analyse] reading data from file %s\n", filename);

          double * buffer = init_1level_dtable( 2 * T );
          if( buffer == NULL ) {
            fprintf(stderr, "[twop_analyse] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(15);
          }

          /***********************************************************
           * loop on configs and source locations per config
           ***********************************************************/
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

              /***********************************************************
               * copy source coordinates
               ***********************************************************/
              int const gsx[4] = {
                  conf_src_list[iconf][isrc][2],
                  conf_src_list[iconf][isrc][3],
                  conf_src_list[iconf][isrc][4],
                  conf_src_list[iconf][isrc][5] };
  
              sprintf ( key , "/stream_%c/conf_%d/t%.2dx%.2dy%.2dz%.2d",conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1],
                  conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );
  
              if ( g_verbose > 2 ) fprintf ( stdout, "# [twop_analyse] key = %s\n", key );
  
              exitstatus = read_from_h5_file ( buffer, filename, key, io_proc );
              if( exitstatus != 0 ) {
                fprintf(stderr, "[twop_analyse] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }

              /**********************************************************
               * source phase
               **********************************************************/
              double const p[3] = {
                  TWO_MPI * (double)sink_momentum[0] / (double)LX_global,
                  TWO_MPI * (double)sink_momentum[1] / (double)LY_global,
                  TWO_MPI * (double)sink_momentum[2] / (double)LZ_global };
  
              double const phase = -( p[0] * gsx[1] + p[1] * gsx[2] + p[2] * gsx[3] ) ;
  
              double _Complex const ephase = cexp ( phase * I );
  
              /**********************************************************
               * sort data from buffer into hvp,
               * add source phase
               **********************************************************/
#pragma omp parallel for
              for ( int it = 0; it < T; it++ ) {
                int const tt = ( it - gsx[0] + T_global ) % T_global; 

                double _Complex ztmp = ( buffer[2*it] +  buffer[2*it+1] * I ) * ephase;
  
                corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt  ] += creal( ztmp ) * twop_operator_norm[0] * twop_operator_norm[1];
                corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt+1] += cimag( ztmp ) * twop_operator_norm[0] * twop_operator_norm[1];
              }
        
            }  /* end of loop on source locations */
          }   /* end of loop on configurations */

          fini_1level_dtable( &buffer );

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "twop_analyse", "h5-read-all-src-conf", g_cart_id == 0 );

        }  /* end of loop on parity */

      }  /* end of loop on sink momenta */

    }  /* end of loop on source gamma id */

  }  /* end of loop on sink gamma id */

#endif  /* of H5 single file */

  /***********************************************************/
  /***********************************************************/


#if _TWOP_H5_BLOCK
  /**********************************************************
    * loop on momenta
   **********************************************************/
  for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

    for ( int ip = 0; ip <1; ip++ ) {

      int const parity_sign = 1 - 2 * ip;

      int const sink_momentum[3] = {
          parity_sign * g_sink_momentum_list[isink_momentum][0],
          parity_sign * g_sink_momentum_list[isink_momentum][1],
          parity_sign * g_sink_momentum_list[isink_momentum][2] };

      int ifl = -1;
      if  ( flavor_type == 0 || flavor_type == 3 ) {
        ifl = flavor_type;
      } else if  ( flavor_type == 1 )  {
        ifl = flavor_type + ip;
      } else if  ( flavor_type == 2 )  {
        ifl = flavor_type - ip;
      }

      gettimeofday ( &ta, (struct timezone *)NULL );

      sprintf ( filename, "%s/%s.%s.%s.px%d_py%d_pz%d.h5", filename_prefix, filename_prefix2, 
          twop_correlator_prefix[correlator_type], twop_flavor_tag[ ifl ], sink_momentum[0], sink_momentum[1], sink_momentum[2] );

      fprintf(stdout, "# [twop_analyse] reading data from file %s\n", filename);

      double *** buffer = init_3level_dtable( g_sink_gamma_id_number, g_source_gamma_id_number, 2 * T );
      if( buffer == NULL ) {
        fprintf(stderr, "[twop_analyse] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(15);
      }

      /***********************************************************
       * loop on configs and source locations per config
       ***********************************************************/
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

          /***********************************************************
           * copy source coordinates
           ***********************************************************/
          int const gsx[4] = {
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4],
            conf_src_list[iconf][isrc][5] };
  
          sprintf ( key , "/stream_%c/conf_%d/t%d_x%d_y%d_z%d",conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1],
              conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );
  
          if ( g_verbose > 2 ) fprintf ( stdout, "# [twop_analyse] key = %s\n", key );
  
          exitstatus = read_from_h5_file ( buffer[0][0], filename, key, "double", io_proc );

          if( exitstatus != 0 ) {
            fprintf(stderr, "[twop_analyse] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(105);
          }

          /**********************************************************
           * source phase
           **********************************************************/
          double const p[3] = {
              TWO_MPI * (double)sink_momentum[0] / (double)LX_global,
              TWO_MPI * (double)sink_momentum[1] / (double)LY_global,
              TWO_MPI * (double)sink_momentum[2] / (double)LZ_global };
  
          double const phase = -( p[0] * gsx[1] + p[1] * gsx[2] + p[2] * gsx[3] ) ;
  
          double _Complex const ephase = cexp ( phase * I );
  
          for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ ) {

            for ( int isource_gamma = 0; isource_gamma < g_source_gamma_id_number; isource_gamma++ ) {

              /**********************************************************
               * sort data from buffer into hvp,
               * add source phase
               **********************************************************/
#pragma omp parallel for
              for ( int it = 0; it < T; it++ ) {
                int const tt = ( it - gsx[0] + T_global ) % T_global; 

                double _Complex ztmp = ( buffer[isink_gamma][isource_gamma][2*it] +  buffer[isink_gamma][isource_gamma][2*it+1] * I ) * ephase;
  
                corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt  ] += creal( ztmp ) * twop_operator_norm[0] * twop_operator_norm[1];
                corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt+1] += cimag( ztmp ) * twop_operator_norm[0] * twop_operator_norm[1];
              }
    
            }  /* end of loop on source gamma id */

          }  /* end of loop on sink gamma id */
        
        }  /* end of loop on source locations */
      }   /* end of loop on configurations */

      fini_3level_dtable( &buffer );

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "twop_analyse", "h5-read-all-src-conf", g_cart_id == 0 );

    }  /* end of loop on parity */

  }  /* end of loop on sink momenta */

#endif  /* of H5 single block */

  /***********************************************************/
  /***********************************************************/

#if _TWOP_AFF_OET
  /***********************************************************
   * loop on configs and source locations per config
   ***********************************************************/
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      Nconf = conf_src_list[iconf][isrc][1];

      /***********************************************************
       * copy source coordinates
       ***********************************************************/
      int const gsx[4] = {
          conf_src_list[iconf][isrc][2],
          conf_src_list[iconf][isrc][3],
          conf_src_list[iconf][isrc][4],
          conf_src_list[iconf][isrc][5] };

      /***********************************************
       * reader for aff input file
       ***********************************************/

      gettimeofday ( &ta, (struct timezone *)NULL );
      struct AffNode_s *affn = NULL, *affdir = NULL;

      sprintf ( filename, "%s.%.4d.t%d.noise%d.aff", filename_prefix, Nconf, gsx[0], g_noise_type );

      fprintf(stdout, "# [twop_analyse] reading data from file %s\n", filename);

      struct AffReader_s *affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[twop_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }

      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[twop_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        return(103);
      }

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "twop_analyse", "open-init-aff-reader", g_cart_id == 0 );

      double * buffer = init_1level_dtable ( 2 * T_global );

      for ( int isample = 0; isample < g_nsample_oet; isample++ ) {

        for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ ) {

          int const sink_gamma_id = g_sink_gamma_id_list[ isink_gamma ];

          for ( int isource_gamma = 0; isource_gamma < g_source_gamma_id_number; isource_gamma++ ) {

            int const source_gamma_id = g_source_gamma_id_list[ isource_gamma ];

          /**********************************************************
           * loop on momenta
           **********************************************************/
          for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

            for ( int ip = 0; ip < 1; ip++ )  {

              int const parity_sign = 1 - 2*ip;
 
              int ifl = -1;
              if  ( flavor_type == 0 || flavor_type == 3 ) {
                ifl = flavor_type;
              } else if  ( flavor_type == 1 )  { 
                ifl = flavor_type + ip;
              } else if  ( flavor_type == 2 )  { 
                ifl = flavor_type - ip;
              }

              int const sink_momentum[3] = {
                parity_sign * g_sink_momentum_list[isink_momentum][0],
                parity_sign * g_sink_momentum_list[isink_momentum][1],
                parity_sign * g_sink_momentum_list[isink_momentum][2] };

              gettimeofday ( &ta, (struct timezone *)NULL );

              sprintf ( key , "/%s/t%d/s%d/gf%d/gi%d/pix%dpiy%dpiz%d/px%dpy%dpz%d", 
                  twop_flavor_tag[ ifl ],
                  gsx[0], isample, sink_gamma_id, source_gamma_id, 
                  -sink_momentum[0], -sink_momentum[1], -sink_momentum[2],
                   sink_momentum[0],  sink_momentum[1],  sink_momentum[2] );

              if ( g_verbose > 2 ) fprintf ( stdout, "# [twop_analyse] key = %s\n", key );
  
              affdir = aff_reader_chpath (affr, affn, key );
              uint32_t uitems = (uint32_t)T_global;
              exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer), uitems );
              if( exitstatus != 0 ) {
                fprintf(stderr, "[twop_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }

#pragma omp parallel for
              for ( int it = 0; it < T_global; it++ ) {
                int const itt = ( it + conf_src_list[iconf][isrc][2] + T_global ) % T_global;
                corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*it  ] += buffer[2*itt  ];
                corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*it+1] += buffer[2*itt+1];
              }

              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "twop_analyse", "read-aff-key", g_cart_id == 0 );

              /**********************************************************
               * NO source phase
               **********************************************************/

              /**********************************************************
               * NO sort data w.r.t. source time
               **********************************************************/

            }  /* end of loop on ip */
          
          }  /* end of loop on sink momenta */

        }  /* end of loop on source gamma id */

        }  /* end of loop on sink gamma id */

      }  /* end of loop on samples */

      double const norm = 1. / g_nsample_oet;
      for ( int it = 0; it < 2 * g_sink_momentum_number * g_sink_gamma_id_number * g_source_gamma_id_number * T_global; it++ ) {
        corr[iconf][isrc][0][0][0][it] *= norm;
      }

      aff_reader_close ( affr );

    }  /* end of loop on source locations */

  }   /* end of loop on configurations */

#endif  /* of TWOP_AFF_OET */

  /***********************************************************/
  /***********************************************************/


#if _TWOP_H5_OET
  for ( int iconf = 0; iconf < num_conf; iconf++ ) 
  {

    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) 
    {


      /* sprintf ( filename, "stream_%c/%s/%s.%.4d.t%d.h5",
          conf_src_list[iconf][isrc][0], 
          filename_prefix,
          filename_prefix2,
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][2] ); */

      sprintf ( filename, "stream_%c/%s/%d/%s.%.4d.t%d.s%d.h5",
          conf_src_list[iconf][isrc][0], 
          filename_prefix,
          conf_src_list[iconf][isrc][1],
          filename_prefix2,
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][2],
          conf_src_list[iconf][isrc][3]);

      double * buffer = init_1level_dtable ( 2 * T_global );


      if ( g_verbose > 1 ) fprintf( stdout, "# [twop_analyse] filename %s %s %d\n", filename, __FILE__, __LINE__ );

      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) 
      {
        for ( int igf = 0; igf < g_sink_gamma_id_number; igf++ ) 
        {
          for ( int igi = 0; igi < g_source_gamma_id_number; igi++ ) 
          {

            char key[500];
            /* /d+-g-u-g/t7/gf4/pfx0pfy0pfz0/gi4//pix0piy0piz0 */
            /* sprintf( key, "%s/t%d/gf%d/pfx%dpfy%dpfz%d/gi%d/pix%dpiy%dpiz%d", twop_flavor_tag[flavor_type], conf_src_list[iconf][isrc][2],
                g_sink_gamma_id_list[igf], 
                g_sink_momentum_list[imom][0],
                g_sink_momentum_list[imom][1],
                g_sink_momentum_list[imom][2],
                g_source_gamma_id_list[igi],
                -g_sink_momentum_list[imom][0],
                -g_sink_momentum_list[imom][1],
                -g_sink_momentum_list[imom][2] ); */

            /* /l-gf-l-gi/mu-0.0006/mu0.0006/t136/s10/gf0/gi6/pix0piy0piz0/px0py0pz0 */
            sprintf( key, "%s/mu%6.4f/mu%6.4f/t%d/s%d/gf%d/gi%d/pix%dpiy%dpiz%d/px%dpy%dpz%d", 
                twop_flavor_tag[flavor_type], 
                muval[0], muval[1],
                conf_src_list[iconf][isrc][2],
                conf_src_list[iconf][isrc][4],
                g_sink_gamma_id_list[igf], 
                g_source_gamma_id_list[igi],
                -g_sink_momentum_list[imom][0],
                -g_sink_momentum_list[imom][1],
                -g_sink_momentum_list[imom][2],
                g_sink_momentum_list[imom][0],
                g_sink_momentum_list[imom][1],
                g_sink_momentum_list[imom][2] );

            fprintf( stdout, "# [twop_analyse] using key %s   %s %d\n", key, __FILE__, __LINE__ );

            exitstatus = read_from_h5_file ( buffer, filename, key, "double", io_proc );
            if ( exitstatus != 0 )
            {
              fprintf ( stderr, "[twop_analyse] Error for file %s key %s   %s %d\n", filename, key, __FILE__, __LINE__ );
              EXIT(12);
            }
            for ( int it = 0; it < T_global; it++ ) 
            {
              int const itt = ( it + conf_src_list[iconf][isrc][2] ) % T_global;
              corr[iconf][isrc][imom][igf][igi][2*it  ] = buffer[2*itt  ];
              corr[iconf][isrc][imom][igf][igi][2*it+1] = buffer[2*itt+1];
            }

          }  /* gamma_i */
        }  /* gamma_f */
      
      }  /* end of loop on momenta  */

      fini_1level_dtable ( &buffer );

    }  /* end of loop on source timeslices */

  }  /* end of loop on configs */

#endif  /* of _TWOP_H5_OET */

  /***********************************************************/
  /***********************************************************/

#if _CVC_H5

  /***********************************************
   * reader for aff input file
   ***********************************************/
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      int const gsx[4] = {
              conf_src_list[iconf][isrc][2],
              conf_src_list[iconf][isrc][3],
              conf_src_list[iconf][isrc][4],
              conf_src_list[iconf][isrc][5] };


      sprintf ( filename, "stream_%c/%s/%d/%s.%.4d.t%.2dx%.2dy%.2dz%.2d.h5", 
          conf_src_list[iconf][isrc][0],
          filename_prefix, 
          conf_src_list[iconf][isrc][1],
          filename_prefix2,
          conf_src_list[iconf][isrc][1], gsx[0], gsx[1], gsx[2], gsx[3] );

      fprintf(stdout, "# [p2gg_analyse_wdisc] reading data from file %s\n", filename);

      double **** buffer = init_4level_dtable( T, g_sink_gamma_id_number, g_source_gamma_id_number, 2 * momentum_number );
      if( buffer == NULL ) {
        fprintf(stderr, "[p2gg_analyse_wdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(15);
      }
 
      gettimeofday ( &ta, (struct timezone *)NULL );

      /**********************************************************
       * neutral case
       **********************************************************/
      sprintf ( key , "/local-local/u-v-u-v" );

      exitstatus = read_from_h5_file (  buffer[0][0][0], filename, key,  "double", io_proc );
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[p2gg_analyse] Error from read_from_h5_file for file %s key %s    %s %d\n",
           filename, key, __FILE__, __LINE__);
        EXIT(15);
      }
              
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "p2gg_analyse", "read-pgg-flavor-tensor-components-neutral-h5", g_cart_id == 0 );
      
      /**********************************************************
       * loop on momenta
       **********************************************************/
      for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

        int const sink_momentum[3] = {
                  g_sink_momentum_list[isink_momentum][0],
                  g_sink_momentum_list[isink_momentum][1],
                  g_sink_momentum_list[isink_momentum][2] };

        int const msink_momentum[3] = {
                  -g_sink_momentum_list[isink_momentum][0],
                  -g_sink_momentum_list[isink_momentum][1],
                  -g_sink_momentum_list[isink_momentum][2] };

        int const  sink_momentum_id = get_momentum_id (  sink_momentum, momentum_list, momentum_number );
        int const msink_momentum_id = get_momentum_id ( msink_momentum, momentum_list, momentum_number );

        if ( sink_momentum_id == -1 || msink_momentum_id == -1 ) EXIT(127);

        double const p[4] = {
            0.,
            TWO_MPI * (double)sink_momentum[0] / (double)LX_global,
            TWO_MPI * (double)sink_momentum[1] / (double)LY_global,
            TWO_MPI * (double)sink_momentum[2] / (double)LZ_global };

        double const phase =  - ( p[0] * gsx[0] + p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] );
        double _Complex const ephase = cexp ( phase * I );

#pragma omp parallel for
        for ( int it = 0; it < T_global; it++ ) {
          int const tt = ( it + gsx[0] ) % T_global;
        
          for( int mu = 0; mu < g_sink_gamma_id_number; mu++) {
            double const parity_mu_sign = ( mu == 0 ) ? +1. : -1.;
          for( int nu = 0; nu < g_source_gamma_id_number; nu++) {
            double const parity_nu_sign = ( nu == 0 ) ? +1. : -1.;

            double const parity_sign = parity_mu_sign * parity_nu_sign;
            if ( g_verbose > 4 ) fprintf ( stdout, "# [twop_analyse] mu %d nu %d  parity_sign %e\n", mu, nu, parity_sign );

            double _Complex const ztmp  = ( buffer[tt][mu][nu][2 *  sink_momentum_id] + buffer[tt][mu][nu][2 *  sink_momentum_id+1] * I ) * ephase;
            double _Complex const ztmp2 = ( buffer[tt][mu][nu][2 * msink_momentum_id] + buffer[tt][mu][nu][2 * msink_momentum_id+1] * I ) * conj (ephase );

            corr[iconf][isrc][isink_momentum][mu][nu][2*it  ] = creal( ztmp + parity_sign * ztmp2 ) * twop_operator_norm[0] * twop_operator_norm[1] * 0.5;
            corr[iconf][isrc][isink_momentum][mu][nu][2*it+1] = cimag( ztmp + parity_sign * ztmp2 ) * twop_operator_norm[0] * twop_operator_norm[1] * 0.5;

          }}
        }
      
      }

      fini_4level_dtable( &buffer );

    }  /* end of loop on source */
  }
      
  fini_2level_itable ( &momentum_list );

#endif  /* end of if _CVC_H5 */


  /****************************************
   * show all data
   ****************************************/
  if ( g_verbose > 5 ) {
    gettimeofday ( &ta, (struct timezone *)NULL );
    FILE *ofs = fopen ( "twop_analyse.data", "w" );


    for ( int iconf = 0; iconf < num_conf; iconf++ )
    {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
      {
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
        {
          for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ )
          {
            for ( int isource_gamma = 0; isource_gamma < g_source_gamma_id_number; isource_gamma++ )
            {
              for ( int it = 0; it < T; it++ )
              {
                fprintf ( ofs, "c %6d s %3d p %3d %3d %3d gf %d gi %d  corr %3d  %25.16e %25.16e\n", iconf, isrc, 
                    g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], 
                    g_sink_gamma_id_list[ isink_gamma ], g_source_gamma_id_list[ isource_gamma ], it, 
                    corr[iconf][isrc][imom][isink_gamma][isource_gamma][2*it], corr[iconf][isrc][imom][isink_gamma][isource_gamma][2*it+1] );
              }
            }
          }
        }
      }
    }
    fclose ( ofs );
    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "twop_analyse", "show-all-data", g_cart_id == 0 );
  }

  /****************************************
   * write src-avg. correlator to file
   ****************************************/
  if ( write_data == 1 ) 
  {
    for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ ) 
    {
      int const sink_gamma_id = g_sink_gamma_id_list[ isink_gamma ];

      for ( int isource_gamma = 0; isource_gamma < g_source_gamma_id_number; isource_gamma++ ) 
      {
        int const source_gamma_id = g_source_gamma_id_list[ isource_gamma ];

        for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
        {

          char filename[500];
          sprintf ( filename, "%s.%s.%s.gf_%s.gi_%s.px%d_py%d_pz%d.corr", g_outfile_prefix, twop_flavor_tag[ flavor_type],
              twop_correlator_prefix[ correlator_type ],
              gamma_id_to_ascii[sink_gamma_id], gamma_id_to_ascii[source_gamma_id],
              g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] );

          FILE * fs = fopen( filename, "w" );

          for ( int iconf = 0; iconf < num_conf; iconf++ ) 
          {
            for ( int it = 0; it < T_global; it++ ) 
            {
              double dtmp[2] = {0.,0.};
              for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
              {
                dtmp[0] += corr[iconf][isrc][imom][isink_gamma][isource_gamma][2*it+0];
                dtmp[1] += corr[iconf][isrc][imom][isink_gamma][isource_gamma][2*it+1];
              }
              fprintf ( fs, "%3d %25.16e %25.16e %c %6d\n", it,
                  dtmp[0] / (double)num_src_per_conf, 
                  dtmp[1] / (double)num_src_per_conf, 
                  conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );

            }
          }


          fclose( fs );
        }
      }
    }
  }



  /****************************************
   * STATISTICAL ANALYSIS
   ****************************************/

  for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ ) {

    int const sink_gamma_id = g_sink_gamma_id_list[ isink_gamma ];

  for ( int isource_gamma = 0; isource_gamma < g_source_gamma_id_number; isource_gamma++ ) {

    int const source_gamma_id = g_source_gamma_id_list[ isource_gamma ];

    for ( int ireim = 0; ireim < 2; ireim++ )
    {

      double ** data = init_2level_dtable ( num_conf, T_global );

      /* fill data array */
#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int it = 0; it < T_global; it++ ) {
          data[iconf][it] = 0.;
          for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              data[iconf][it] += corr[iconf][isrc][imom][isink_gamma][isource_gamma][2*it+ireim];
            }
          }
          data[iconf][it] /= (double)num_src_per_conf * (double)g_sink_momentum_number;
        }
      }

      if ( fold_correlator ) {
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 1; it < T_global/2; it++ ) {
            data[iconf][it] += data[iconf][T_global - it];
            data[iconf][it] *= 0.5;
            data[iconf][T_global - it] = data[iconf][it];
          }
        }
      }

     /****************************************
      * STATISTICAL ANALYSIS of real and
      * imaginary part
      ****************************************/

      char obs_name[100];
#if _TWOP_AFF_OET
      sprintf ( obs_name, "%s.%s.%s.gf_%s.gi_%s.px%d_py%d_pz%d.noise%d.%s", g_outfile_prefix, twop_correlator_prefix[correlator_type], twop_flavor_tag[ flavor_type],
          gamma_id_to_ascii[sink_gamma_id], gamma_id_to_ascii[source_gamma_id], g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2],
         g_noise_type, reim_str[ireim] );
#else
      sprintf ( obs_name, "%s.%s.%s.gf_%s.gi_%s.px%d_py%d_pz%d.%s", g_outfile_prefix, twop_correlator_prefix[correlator_type], twop_flavor_tag[ flavor_type],
          gamma_id_to_ascii[sink_gamma_id], gamma_id_to_ascii[source_gamma_id],
          g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

#endif


#if 0
      if ( write_data == 1 ) {
        sprintf ( filename, "%s.corr" , obs_name );
        FILE * fs = fopen( filename, "w" );

        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            fprintf ( fs, "%3d %25.16e %c %6d\n", it, data[iconf][it], conf_src_list[iconf][0][0],  conf_src_list[iconf][0][1] );

            /* for matching Konstantin's required layout */
            /* fprintf ( fs, "%3d %25.16e %6d\n", it, data[iconf][it], iconf * g_gauge_step ); */
          }
        }


        fclose( fs );
      }
#endif
      if ( num_conf >= 6 ) {

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[twop_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }
    
#if 0
        /****************************************
         * STATISTICAL ANALYSIS of effective
         * mass from time-split acosh ratio
         ****************************************/
        for ( int itau = 1; itau < T_global/16; itau++ )
        {

          char obs_name2[200];
          sprintf( obs_name2, "%s.acoshratio.tau%d", obs_name, itau );

          int arg_first[3]  = { 0, 2*itau, itau };
          int arg_stride[3] = {1, 1, 1};

          exitstatus = apply_uwerr_func ( data[0], num_conf, T_global, T_global/2-itau, 3, arg_first, arg_stride, obs_name2, acosh_ratio, dacosh_ratio );

          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[twop_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
        }
#endif
      }

      fini_2level_dtable ( &data );

    }  /* end of loop on re / im */
  }}  /* end of loop on source and sink gamma id */

  /**********************************************************
   * free corr field
   **********************************************************/
  fini_6level_dtable ( &corr );

  /**********************************************************
   * free the allocated memory, finalize
   **********************************************************/

  fini_3level_itable ( &conf_src_list );

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif


#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [twop_analyse] %s# [twop_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [twop_analyse] %s# [twop_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
