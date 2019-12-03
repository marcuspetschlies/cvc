/****************************************************
 * cpff_threep_analyse
 *
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

#define TWOP_STATS
#define THREEP_STATS

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse P-J-J correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  double const TWO_MPI = 2. * M_PI;

  char const reim_str[2][3] = {"re" , "im"};

  char const twopt_correlator_prefix[1][20] = { "local-local" };

  char const twop_flavor_tag[4][20]        = { "u-gf-u-gi" , "d-gf-u-gi" , "u-gf-d-gi" , "d-gf-d-gi" };

  char const threep_flavor_tag[2][20]        = { "u-gc-sud-gi", "u-gd-sud-gi" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "cA211a.30.32";
  int fold_correlator= 0;
  struct timeval ta, tb;
  int twop_flavor_type = -1;
  int threep_flavor_type = -1;

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  char key[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:F:E:t:T:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [cpff_threep_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [cpff_threep_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'F':
      fold_correlator = atoi ( optarg );
      fprintf ( stdout, "# [cpff_threep_analyse] fold_correlator set to %d\n", fold_correlator );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [cpff_threep_analyse] ensemble_name set to %s\n", ensemble_name );
      break;
    case 't':
      twop_flavor_type = atoi ( optarg );
      fprintf ( stdout, "# [cpff_threep_analyse] twop_flavor_type set to %d\n", twop_flavor_type );
      break;
    case 'T':
      threep_flavor_type = atoi ( optarg );
      fprintf ( stdout, "# [cpff_threep_analyse] threep_flavor_type set to %d\n", threep_flavor_type );
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
  if(filename_set==0) strcpy(filename, "cpff.input");
  /* fprintf(stdout, "# [cpff_threep_analyse] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [cpff_threep_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [cpff_threep_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [cpff_threep_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[cpff_threep_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[cpff_threep_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[cpff_threep_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [cpff_threep_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[cpff_threep_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 5 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[cpff_threep_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [cpff_threep_analyse] comment %s\n", line );
      continue;
    }

    sscanf( line, "%d %d %d %d %d", 
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf],
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+1,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+2,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+3,
        conf_src_list[count/num_src_per_conf][count%num_src_per_conf]+4 );

    count++;
  }

  fclose ( ofs );


  if ( g_verbose > 5 ) {
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        fprintf ( stdout, "conf_src_list %6d %3d %3d %3d %3d\n", 
            conf_src_list[iconf][isrc][0],
            conf_src_list[iconf][isrc][1],
            conf_src_list[iconf][isrc][2],
            conf_src_list[iconf][isrc][3],
            conf_src_list[iconf][isrc][4] );

      }
    }
  }

  /***********************************************************
   ***********************************************************
   **
   ** TWOPT
   **
   ***********************************************************
   ***********************************************************/

  double ****** corr = NULL;
  double ***** corr_orbit = NULL;

#ifdef TWOP_STATS

  corr = init_6level_dtable ( num_conf, num_src_per_conf, g_sink_momentum_number, g_sink_gamma_id_number, g_source_gamma_id_number, 2 * T_global );
  if ( corr == NULL ) {
    fprintf(stderr, "[cpff_threep_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  /***********************************************************
   * loop on configs and source locations per config
   ***********************************************************/
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      Nconf = conf_src_list[iconf][isrc][0];

      /***********************************************************
       * copy source coordinates
       ***********************************************************/
      int const gsx[4] = {
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][2],
          conf_src_list[iconf][isrc][3],
          conf_src_list[iconf][isrc][4] };

#ifdef HAVE_LHPC_AFF
      /***********************************************
       * reader for aff input file
       ***********************************************/

      gettimeofday ( &ta, (struct timezone *)NULL );
      struct AffNode_s *affn = NULL, *affdir = NULL;

      sprintf ( filename, "%s.%.4d.t%d.aff", filename_prefix, Nconf, gsx[0] );

      fprintf(stdout, "# [cpff_threep_analyse] reading data from file %s\n", filename);
      affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[cpff_threep_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }

      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[cpff_threep_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        return(103);
      }

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "cpff_threep_analyse", "open-init-aff-reader", g_cart_id == 0 );
#endif

      /**********************************************************
       * loop on gamma structure at sink
       **********************************************************/
      for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ ) {

        int const sink_gamma_id = g_sink_gamma_id_list[ isink_gamma ];

      /**********************************************************
       * loop on gamma structure at source
       **********************************************************/
      for ( int isource_gamma = 0; isource_gamma < g_source_gamma_id_number; isource_gamma++ ) {

        int const source_gamma_id = g_source_gamma_id_list[ isource_gamma ];

        /**********************************************************
         * loop on momenta
         **********************************************************/
        for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

          int const sink_momentum[3] = {
            g_sink_momentum_list[isink_momentum][0],
            g_sink_momentum_list[isink_momentum][1],
            g_sink_momentum_list[isink_momentum][2] };


          int const source_momentum[3] = {
            -sink_momentum[0],
            -sink_momentum[1],
            -sink_momentum[2] };

          double * buffer = init_1level_dtable( 2 * T_global );
          if( buffer == NULL ) {
            fprintf(stderr, "[cpff_threep_analyse] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(15);
          }

          for ( int isample = 0; isample < g_nsample_oet; isample++ ) {

            gettimeofday ( &ta, (struct timezone *)NULL );

            sprintf ( key , "/%s/t%d/s%d/gf%d/gi%d/pix%dpiy%dpiz%d/px%dpy%dpz%d", twop_flavor_tag[ twop_flavor_type ],
                gsx[0], isample, sink_gamma_id, source_gamma_id, 
                source_momentum[0], source_momentum[1], source_momentum[2], sink_momentum[0], sink_momentum[1], sink_momentum[2] );

            if ( g_verbose > 2 ) fprintf ( stdout, "# [cpff_threep_analyse] key = %s\n", key );

            affdir = aff_reader_chpath (affr, affn, key );
            if ( affdir == NULL ) {
              fprintf(stderr, "[cpff_threep_analyse] Error from aff_reader_chpath for key %s %s %d\n", key,  __FILE__, __LINE__);
              EXIT(116);
            }
            uint32_t uitems = (uint32_t)T_global;
            exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer), uitems );
            if( exitstatus != 0 ) {
              fprintf(stderr, "[cpff_threep_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }

            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "cpff_threep_analyse", "read-aff-key", g_cart_id == 0 );
          
#if 0
          /**********************************************************
           * source phase
           **********************************************************/
          double const p[3] = {
              TWO_MPI * (double)sink_momentum[0] / (double)LX_global,
              TWO_MPI * (double)sink_momentum[1] / (double)LY_global,
              TWO_MPI * (double)sink_momentum[2] / (double)LZ_global };

          double const phase = -( p[0] * gsx[1] + p[1] * gsx[2] + p[2] * gsx[3] ) ;

          double _Complex const ephase = cexp ( phase * I );
#endif  /* of if 0 */

            /**********************************************************
             * sort data from buffer into hvp,
             * add source phase
             **********************************************************/
#pragma omp parallel for
            for ( int it = 0; it < T_global; it++ ) {
              int const tt = ( it - gsx[0] + T_global ) % T_global; 

              /* double _Complex ztmp = ( buffer[2*it] +  buffer[2*it+1] * I ) * ephase; */

              corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt  ] += buffer[2*it  ];
              corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt+1] += buffer[2*it+1];
            }

          }  /* end of loop on samples */

          for ( int it = 0; it < 2*T_global; it++ ) {
            corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][it  ] /= (double)g_nsample_oet;
          }

          fini_1level_dtable( &buffer );

        }  /* end of loop on sink momenta */

      }  /* end of loop on source gamma id */

      }  /* end of loop on sink gamma id */

#ifdef HAVE_LHPC_AFF
      aff_reader_close ( affr );
#endif  /* of ifdef HAVE_LHPC_AFF */

    }  /* end of loop on source locations */

  }   /* end of loop on configurations */

  /****************************************
   * show all data
   ****************************************/
  if ( g_verbose > 5 ) {
    gettimeofday ( &ta, (struct timezone *)NULL );
    FILE *ofs = fopen ( "cpff_threep_analyse.data", "w" );


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
    show_time ( &ta, &tb, "cpff_threep_analyse", "show-all-data", g_cart_id == 0 );
  }

  /**********************************************************
    * simple orbit average
    **********************************************************/
  corr_orbit = init_5level_dtable ( num_conf, num_src_per_conf, g_sink_gamma_id_number, g_source_gamma_id_number, 2 * T_global );
  if ( corr_orbit == NULL ) {
    fprintf(stderr, "[cpff_threep_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
#pragma omp parallel for
  for ( int iconf = 0; iconf < num_conf; iconf++ )
  {
    for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
    {
      for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ )
      {
        for ( int isource_gamma = 0; isource_gamma < g_source_gamma_id_number; isource_gamma++ )
        {
          for ( int it = 0; it < T; it++ )
          {
            for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
            {
              /* real part */
              corr_orbit[iconf][isrc][isink_gamma][isource_gamma][2*it  ] += corr[iconf][isrc][imom][isink_gamma][isource_gamma][2*it  ];
 
              /* imag part */              
              corr_orbit[iconf][isrc][isink_gamma][isource_gamma][2*it+1] += corr[iconf][isrc][imom][isink_gamma][isource_gamma][2*it+1];

            }
            corr_orbit[iconf][isrc][isink_gamma][isource_gamma][2*it  ] /= (double)g_sink_momentum_number;
            corr_orbit[iconf][isrc][isink_gamma][isource_gamma][2*it+1] /= (double)g_sink_momentum_number;
          }
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

      for ( int ireim = 0; ireim < 2; ireim++ ) {

        double ** data = init_2level_dtable ( num_conf, T_global );
        if ( data == NULL ) {
          fprintf( stderr, "[cpff_threep_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = 0.;
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              data[iconf][it] += corr_orbit[iconf][isrc][isink_gamma][isource_gamma][2*it+ireim];
            }
            data[iconf][it] /= (double)num_src_per_conf;
          }
        }

        if ( fold_correlator ) {
#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it <= T_global/2; it++ ) {
              data[iconf][it] += data[iconf][T_global - it];
              data[iconf][it] *= 0.5;
              data[iconf][T_global - it] = data[iconf][it];
              }
          }
        }  /* end of if fold correlator */

       /****************************************
        * STATISTICAL ANALYSIS of real and
        * imaginary part
        ****************************************/

        char obs_name[100];
        sprintf ( obs_name, "%s.gf%d.gi%d.pfx%d_pfy%d_pfz%d.%s", twop_flavor_tag[ twop_flavor_type],
            sink_gamma_id, source_gamma_id,
            g_sink_momentum_list[0][0],
            g_sink_momentum_list[0][1],
            g_sink_momentum_list[0][2],
            reim_str[ireim] );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[cpff_threep_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        /****************************************
         * STATISTICAL ANALYSIS of effective
         * mass from time-split acosh ratio
         ****************************************/

        for ( int itau = 1; itau < T_global/2; itau++ )
        {

          char obs_name2[200];
          sprintf ( obs_name2, "%s.acoshratio.tau%d", obs_name, itau );

          int arg_first[3]  = { 0, 2*itau, itau };
          int arg_stride[3] = {1, 1, 1};

          exitstatus = apply_uwerr_func ( data[0], num_conf, T_global, T_global/2-itau, 3, arg_first, arg_stride, obs_name2, acosh_ratio, dacosh_ratio );

          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[cpff_threep_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }

        }

        fini_2level_dtable ( &data );

      }  /* end of loop on re / im */
    }  /* end of loop on source gamma id */
  }  /* end of loop on sink gamma id */

#endif  /* of TWOP_STATS */

  /**********************************************************/
  /**********************************************************/
  
  /***********************************************************
   ***********************************************************
   **
   ** THREEP
   **
   ***********************************************************
   ***********************************************************/

#ifdef THREEP_STATS

  /*****************************************************************
   * loop on sequential source gamma ids
   *****************************************************************/
  for ( int iseq_gamma = 0; iseq_gamma < g_sequential_source_gamma_id_number; iseq_gamma++ ) {

    int seq_source_gamma = g_sequential_source_gamma_id_list[iseq_gamma];

    /*****************************************************************
     * loop on sequential source timeslices
     *****************************************************************/
    for ( int iseq_timeslice = 0; iseq_timeslice < g_sequential_source_timeslice_number; iseq_timeslice++ ) {

      double ****** corr_threep = init_6level_dtable ( num_conf, num_src_per_conf, g_sink_momentum_number, g_sink_gamma_id_number, g_source_gamma_id_number, 2 * T_global );
      if ( corr_threep == NULL ) {
        fprintf(stderr, "[cpff_threep_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

      /***********************************************************
       * loop on configs and source locations per config
       ***********************************************************/
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

          Nconf = conf_src_list[iconf][isrc][0];

          /***********************************************************
           * copy source coordinates
           ***********************************************************/
          int const gsx[4] = {
              conf_src_list[iconf][isrc][1],
              conf_src_list[iconf][isrc][2],
              conf_src_list[iconf][isrc][3],
              conf_src_list[iconf][isrc][4] };

#ifdef HAVE_LHPC_AFF
          /***********************************************
           * reader for aff input file
           ***********************************************/

          gettimeofday ( &ta, (struct timezone *)NULL );
          struct AffNode_s *affn = NULL, *affdir = NULL;

          sprintf ( filename, "%s.%.4d.t%d.aff", filename_prefix, Nconf, gsx[0] );

          fprintf(stdout, "# [cpff_threep_analyse] reading data from file %s\n", filename);
          affr = aff_reader ( filename );
          const char * aff_status_str = aff_reader_errstr ( affr );
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[cpff_threep_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
            EXIT(15);
          }

          if( (affn = aff_reader_root( affr )) == NULL ) {
            fprintf(stderr, "[cpff_threep_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
            return(103);
          }

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "cpff_threep_analyse", "open-init-aff-reader", g_cart_id == 0 );
#endif

          /**********************************************************
           * loop on gamma structure at sink
           **********************************************************/
          for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ ) {

            int const sink_gamma_id = g_sink_gamma_id_list[ isink_gamma ];

            /**********************************************************
             * loop on gamma structure at source
             **********************************************************/
            for ( int isource_gamma = 0; isource_gamma < g_source_gamma_id_number; isource_gamma++ ) {

              int const source_gamma_id = g_source_gamma_id_list[ isource_gamma ];

              /**********************************************************
               * loop on momenta
               **********************************************************/
              for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

                int const sink_momentum[3] = {
                  g_sink_momentum_list[isink_momentum][0],
                  g_sink_momentum_list[isink_momentum][1],
                  g_sink_momentum_list[isink_momentum][2] };

                int const current_momentum[3] = {
                  g_seq_source_momentum_list[isink_momentum][0],
                  g_seq_source_momentum_list[isink_momentum][1],
                  g_seq_source_momentum_list[isink_momentum][2] }; 

                int const source_momentum[3] = {
                  -( current_momentum[0] + sink_momentum[0] ),
                  -( current_momentum[1] + sink_momentum[1] ),
                  -( current_momentum[2] + sink_momentum[2] ) };

                double * buffer = init_1level_dtable( 2 * T_global );
                if( buffer == NULL ) {
                  fprintf(stderr, "[cpff_threep_analyse] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
                  EXIT(15);
                }

                for ( int isample = 0; isample < g_nsample_oet; isample++ ) {

                  gettimeofday ( &ta, (struct timezone *)NULL );
 
                  /* /u-gc-sud-gi/t22/s0/dt32/gf5/gc9/gi0/pfx1pfy1pfz-1/px0py-1pz0 */

                  sprintf ( key , "/%s/t%d/s%d/dt%d/gf%d/gc%d/gi%d/pfx%dpfy%dpfz%d/px%dpy%dpz%d", threep_flavor_tag[ threep_flavor_type ],
                      gsx[0], isample, g_sequential_source_timeslice_list[iseq_timeslice], sink_gamma_id, seq_source_gamma, source_gamma_id, 
                      sink_momentum[0], sink_momentum[1], sink_momentum[2],
                      current_momentum[0], current_momentum[1], current_momentum[2] );

                  if ( g_verbose > 2 ) fprintf ( stdout, "# [cpff_threep_analyse] key = %s\n", key );

                  affdir = aff_reader_chpath (affr, affn, key );
                  if ( affdir == NULL ) {
                    fprintf(stderr, "[cpff_threep_analyse] Error from aff_reader_chpath for key %s %s %d\n", key,  __FILE__, __LINE__);
                    EXIT(116);
                  }
                  uint32_t uitems = (uint32_t)T_global;
                  exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer), uitems );
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[cpff_threep_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  gettimeofday ( &tb, (struct timezone *)NULL );
                  show_time ( &ta, &tb, "cpff_threep_analyse", "read-aff-key", g_cart_id == 0 );
          
                  /**********************************************************
                   * sort data from buffer into hvp,
                   * add source phase
                   **********************************************************/
#pragma omp parallel for
                  for ( int it = 0; it < T_global; it++ ) {
                    int const tt = ( it - gsx[0] + T_global ) % T_global; 
                    corr_threep[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt  ] += buffer[2*it  ];
                    corr_threep[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt+1] += buffer[2*it+1];
                  }

                }  /* end of loop on samples */

                for ( int it = 0; it < 2*T_global; it++ ) {
                  corr_threep[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][it  ] /= (double)g_nsample_oet;
                }

                fini_1level_dtable( &buffer );

              }  /* end of loop on sink momenta */

            }  /* end of loop on source gamma id */

          }  /* end of loop on sink gamma id */

#ifdef HAVE_LHPC_AFF
          aff_reader_close ( affr );
#endif  /* of ifdef HAVE_LHPC_AFF */

        }  /* end of loop on source locations */
      }   /* end of loop on configurations */

      /**********************************************************
       * show all data
       **********************************************************/
      if ( g_verbose > 5 ) {
        gettimeofday ( &ta, (struct timezone *)NULL );
        FILE *ofs = fopen ( "cpff_threep_analyse.data", "w" );
    
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
                    fprintf ( ofs, "c %6d s %3d dt %2d pf %3d %3d %3d pc %3d %3d %3d  gf %d gc %d gi %d  corr_threep %3d  %25.16e %25.16e\n", iconf, isrc, 
                        g_sequential_source_timeslice_list[iseq_timeslice],
                        g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], 
                        g_seq_source_momentum_list[imom][0], g_seq_source_momentum_list[imom][1], g_seq_source_momentum_list[imom][2], 
                        g_sink_gamma_id_list[ isink_gamma ], seq_source_gamma, g_source_gamma_id_list[ isource_gamma ], it, 
                        corr_threep[iconf][isrc][imom][isink_gamma][isource_gamma][2*it], corr_threep[iconf][isrc][imom][isink_gamma][isource_gamma][2*it+1] );
                  }
                }
              }
            }
          }
        }
        fclose ( ofs );
        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "cpff_threep_analyse", "show-all-data", g_cart_id == 0 );
      }
 
      /**********************************************************
       * simple orbit average
       **********************************************************/

      double ***** corr_threep_orbit = init_5level_dtable ( num_conf, num_src_per_conf, g_sink_gamma_id_number, g_source_gamma_id_number, 2 * T_global );
      if ( corr_threep_orbit == NULL ) {
        fprintf(stderr, "[cpff_threep_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ )
      {
        for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
        {
          for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ )
          {
            for ( int isource_gamma = 0; isource_gamma < g_source_gamma_id_number; isource_gamma++ )
            {
              for ( int it = 0; it < T; it++ )
              {
                for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
                {
                  /* real part */
                  corr_threep_orbit[iconf][isrc][isink_gamma][isource_gamma][2*it  ] += corr_threep[iconf][isrc][imom][isink_gamma][isource_gamma][2*it  ];

                  /* imag part */
                  corr_threep_orbit[iconf][isrc][isink_gamma][isource_gamma][2*it+1] += corr_threep[iconf][isrc][imom][isink_gamma][isource_gamma][2*it+1];

                }
                corr_threep_orbit[iconf][isrc][isink_gamma][isource_gamma][2*it  ] /= (double)g_sink_momentum_number;
                corr_threep_orbit[iconf][isrc][isink_gamma][isource_gamma][2*it+1] /= (double)g_sink_momentum_number;
              }
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
    
        for ( int ireim = 0; ireim < 2; ireim++ ) {
    
          double ** data = init_2level_dtable ( num_conf, T_global );
          if ( data == NULL ) {
            fprintf( stderr, "[cpff_threep_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }
    
          /* fill data array */
#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              data[iconf][it] = 0.;
              for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                data[iconf][it] += corr_threep_orbit[iconf][isrc][isink_gamma][isource_gamma][2*it+ireim];
              }
              data[iconf][it] /= (double)num_src_per_conf;
            }
          }
 
#if 0   
          if ( fold_correlator ) {
#pragma omp parallel for
            for ( int iconf = 0; iconf < num_conf; iconf++ ) {
              for ( int it = 0; it <= T_global/2; it++ ) {
                data[iconf][it] += data[iconf][T_global - it];
                data[iconf][it] *= 0.5;
                data[iconf][T_global - it] = data[iconf][it];
                }
            }
          }  /* end of if fold correlator */
#endif
         /****************************************
          * UWerr analysis of real and
          * imaginary part
          ****************************************/
    
          char obs_name[100];
          sprintf ( obs_name, "%s.dt%d.gf%d.gc%d.gi%d.pfx%d_pfy%d_pfz%d.pcx%d_pcy%d_pcz%d.%s", threep_flavor_tag[ threep_flavor_type],
              g_sequential_source_timeslice_list[iseq_timeslice],
              sink_gamma_id, seq_source_gamma, source_gamma_id, 
              g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2],
              g_seq_source_momentum_list[0][0], g_seq_source_momentum_list[0][1], g_seq_source_momentum_list[0][2],
              reim_str[ireim] );
    
          /* apply UWerr analysis */
          exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[cpff_threep_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
   
          /****************************************
           * STATISTICAL ANALYSIS of effective
           * mass from time-split acosh ratio
           ****************************************/
#if 0
          for ( int itau = 1; itau < T_global/2; itau++ )
          {
    
            char obsname2[200];
            sprintf ( obs_name2, "%s.acoshratio.tau%d", twop_flavor_tag[ twop_flavor_type],
                itau, sink_gamma_id, source_gamma_id, momentum[0], momentum[1], momentum[2], reim_str[ireim] );
    
            int arg_first[3]  = { 0, 2*itau, itau };
            int arg_stride[3] = {1, 1, 1};
  
            exitstatus = apply_uwerr_func ( data[0], num_conf*num_src_per_conf, T_global, T_global/2-itau, 3, arg_first, arg_stride, obs_name, acosh_ratio, dacosh_ratio );
  
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[cpff_threep_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(1);
            }
   
          }
#endif
          fini_2level_dtable ( &data );
    
        }  /* end of loop on re / im */
      }}  /* end of loop on source and sink gamma id */

      fini_5level_dtable ( &corr_threep_orbit );
      fini_6level_dtable ( &corr_threep );

    }  /* end of loop on seq. source timeslice */

  }  /* end of loop on seq source gamma */

#endif  /* of THREEP_STATS */

  /**********************************************************
   * free
   **********************************************************/
  fini_6level_dtable ( &corr );
  fini_5level_dtable ( &corr_orbit );

  /**********************************************************
   * free the allocated memory, finalize
   **********************************************************/

  fini_3level_itable ( &conf_src_list );

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [cpff_threep_analyse] %s# [cpff_threep_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [cpff_threep_analyse] %s# [cpff_threep_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
