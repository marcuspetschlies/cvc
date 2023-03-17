/****************************************************
 * sff_threep_analyse
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

#define TWOP_STATS    1
#define THREEP_STATS  1

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

  char const twop_flavor_tag[4][20]        = { "u+-g-u-g" };

  char const threep_flavor_tag[1][20]        = { "sud+-g-u-g" };

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
  int twop_flavor_type = 0;
  int threep_flavor_type = 0;

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
      fprintf ( stdout, "# [sff_threep_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [sff_threep_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'F':
      fold_correlator = atoi ( optarg );
      fprintf ( stdout, "# [sff_threep_analyse] fold_correlator set to %d\n", fold_correlator );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [sff_threep_analyse] ensemble_name set to %s\n", ensemble_name );
      break;
    case 't':
      twop_flavor_type = atoi ( optarg );
      fprintf ( stdout, "# [sff_threep_analyse] twop_flavor_type set to %d\n", twop_flavor_type );
      break;
    case 'T':
      threep_flavor_type = atoi ( optarg );
      fprintf ( stdout, "# [sff_threep_analyse] threep_flavor_type set to %d\n", threep_flavor_type );
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
  /* fprintf(stdout, "# [sff_threep_analyse] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [sff_threep_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [sff_threep_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [sff_threep_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[sff_threep_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[sff_threep_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[sff_threep_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [sff_threep_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[sff_threep_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[sff_threep_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [sff_threep_analyse] comment %s\n", line );
      continue;
    }

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


  if ( g_verbose > 5 ) {
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        fprintf ( stdout, "conf_src_list %c %6d %3d %3d %3d %3d\n", 
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
   ***********************************************************
   **
   ** TWOPT
   **
   ***********************************************************
   ***********************************************************/

  double **** corr = NULL;
      
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

#if TWOP_STATS

      corr = init_4level_dtable ( 2, num_conf, num_src_per_conf, 2 * T_global );
      if ( corr == NULL ) {
        fprintf(stderr, "[sff_threep_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

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
  
          sprintf ( filename, "stream_%c/%s.%.4d.t%d.h5", conf_src_list[iconf][isrc][0], filename_prefix, Nconf, gsx[0] );
   
          fprintf(stdout, "# [sff_threep_analyse] reading data from file %s\n", filename);
   
          double * buffer = init_1level_dtable( 2 * T_global );
          if( buffer == NULL ) {
            fprintf(stderr, "[sff_threep_analyse] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(15);
          }

          /**********************************************************
           * loop on momenta
           **********************************************************/
          for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

            int const pf[3] = {
              g_sink_momentum_list[isink_momentum][0],
              g_sink_momentum_list[isink_momentum][1],
              g_sink_momentum_list[isink_momentum][2] };

            int const pi[3] = {
              -pf[0],
              -pf[1],
              -pf[2] };

            gettimeofday ( &ta, (struct timezone *)NULL );

            /* corr.1386.t41.h5/u+-g-u-g/t41/gf5/pfx-1pfy-1pfz0/gi5/pix0piy0piz0 */
            sprintf ( key , "/%s/t%d/gf%d/pfx%dpfy%dpfz%d/gi%d/pix%dpiy%dpiz%d", twop_flavor_tag[ twop_flavor_type ],
                gsx[0], 
                sink_gamma_id, pf[0], pf[1], pf[2],
                source_gamma_id, pi[0], pi[1], pi[2] );

            if ( g_verbose > 2 ) fprintf ( stdout, "# [sff_threep_analyse] twop key = %s %s %d\n", key , __FILE__, __LINE__ );

            exitstatus = read_from_h5_file ( buffer, filename, key, "double", io_proc );
            if( exitstatus != 0  ) {
              fprintf(stderr, "[sff_threep_analyse] Error from read_from_h5_file for %s %s  %s %d\n", filename, key,  __FILE__, __LINE__);
              EXIT(15);
            }


            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "sff_threep_analyse", "read-aff-key", g_cart_id == 0 );
          
            /**********************************************************
             * sort data from buffer into hvp,
             * add source phase
             **********************************************************/
#pragma omp parallel for
            for ( int it = 0; it < T_global; it++ ) {
              int const tt = ( it - gsx[0] + T_global ) % T_global; 

              /* double _Complex ztmp = ( buffer[2*it] +  buffer[2*it+1] * I ) * ephase; */

              corr[0][iconf][isrc][2*tt  ] += buffer[2*it  ];
              corr[0][iconf][isrc][2*tt+1] += buffer[2*it+1];
            }

          }  /* end of loop on sink momenta */

          for ( int it = 0; it < 2*T_global; it++ ) {
            corr[0][iconf][isrc][it  ] /= (double)g_sink_momentum_number;
          }


          if ( ( g_sink_momentum_list[0][0] * g_sink_momentum_list[0][0] +
                 g_sink_momentum_list[0][1] * g_sink_momentum_list[0][1] +
                 g_sink_momentum_list[0][2] * g_sink_momentum_list[0][2]  
               ) != (
                 g_source_momentum_list[0][0] * g_source_momentum_list[0][0] +
                 g_source_momentum_list[0][1] * g_source_momentum_list[0][1] +
                 g_source_momentum_list[0][2] * g_source_momentum_list[0][2] 
               ) )
          {

            for ( int isource_momentum = 0; isource_momentum < g_source_momentum_number; isource_momentum++ ) {
  
              int const pf[3] = {
                g_source_momentum_list[isource_momentum][0],
                g_source_momentum_list[isource_momentum][1],
                g_source_momentum_list[isource_momentum][2] };
  
              int const pi[3] = {
                -pf[0],
                -pf[1],
                -pf[2] };
  
              gettimeofday ( &ta, (struct timezone *)NULL );
  
              /* corr.1386.t41.h5/u+-g-u-g/t41/gf5/pfx-1pfy-1pfz0/gi5/pix0piy0piz0 */
              sprintf ( key , "/%s/t%d/gf%d/pfx%dpfy%dpfz%d/gi%d/pix%dpiy%dpiz%d", twop_flavor_tag[ twop_flavor_type ],
                  gsx[0], 
                  source_gamma_id, pf[0], pf[1], pf[2],
                  sink_gamma_id, pi[0], pi[1], pi[2] );
  
              if ( g_verbose > 2 ) fprintf ( stdout, "# [sff_threep_analyse] twop key = %s %s %d\n", key , __FILE__, __LINE__ );
  
              exitstatus = read_from_h5_file ( buffer, filename, key, "double", io_proc );
              if( exitstatus != 0  ) {
                fprintf(stderr, "[sff_threep_analyse] Error from read_from_h5_file for %s %s  %s %d\n", filename, key,  __FILE__, __LINE__);
                EXIT(15);
              }
  
  
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "sff_threep_analyse", "read-aff-key", g_cart_id == 0 );
            
              /**********************************************************
               * sort data from buffer into hvp,
               * add source phase
               **********************************************************/
  #pragma omp parallel for
              for ( int it = 0; it < T_global; it++ ) {
                int const tt = ( it - gsx[0] + T_global ) % T_global; 
  
                /* double _Complex ztmp = ( buffer[2*it] +  buffer[2*it+1] * I ) * ephase; */
  
                corr[1][iconf][isrc][2*tt  ] += buffer[2*it  ];
                corr[1][iconf][isrc][2*tt+1] += buffer[2*it+1];
              }
  
            }  /* end of loop on sink momenta */
  
            for ( int it = 0; it < 2*T_global; it++ ) {
              corr[1][iconf][isrc][it  ] /= (double)g_source_momentum_number;
            }

          } else {
            memcpy (  corr[1][0][0], corr[0][0][0], num_conf * num_src_per_conf * 2 * T_global * sizeof ( double ) );
          }  /* end of if orbits differ */

          fini_1level_dtable( &buffer );

        }  /* end of loop on source locations */

      }   /* end of loop on configurations */

      /****************************************
       * show all data
       ****************************************/
      if ( g_verbose > 5 ) {
        gettimeofday ( &ta, (struct timezone *)NULL );
        FILE *ofs = fopen ( "sff_threep_analyse.data", "w" );

        for ( int iconf = 0; iconf < num_conf; iconf++ )
        {
          for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
          {
            for ( int it = 0; it < T; it++ )
            {
                fprintf ( ofs, "c %c %6d %4d pf %3d %3d %3d gf %2d gi %2d  corr %3d  %25.16e %25.16e\n", 
                    conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1], conf_src_list[iconf][isrc][2], 
                    g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], 
                    g_sink_gamma_id_list[ isink_gamma ], g_source_gamma_id_list[ isource_gamma ], 
                    it, corr[0][iconf][isrc][2*it], corr[0][iconf][isrc][2*it+1] );
            }
          }
        }

        for ( int iconf = 0; iconf < num_conf; iconf++ )
        {
          for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
          {
            for ( int it = 0; it < T; it++ )
            {
                fprintf ( ofs, "c %c %6d %4d pi %3d %3d %3d gf %2d gi %2d  corr %3d  %25.16e %25.16e\n", 
                    conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1], conf_src_list[iconf][isrc][2], 
                    g_source_momentum_list[0][0], g_source_momentum_list[0][1], g_source_momentum_list[0][2], 
                    g_source_gamma_id_list[ isource_gamma ], g_sink_gamma_id_list[ isink_gamma ], 
                    it, corr[1][iconf][isrc][2*it], corr[1][iconf][isrc][2*it+1] );
            }
          }
        }

        fclose ( ofs );
        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "sff_threep_analyse", "show-all-data", g_cart_id == 0 );
      }

      /****************************************
       * STATISTICAL ANALYSIS
       ****************************************/
      for ( int k = 0; k <= 1; k++ ) {

        for ( int ireim = 0; ireim < 2; ireim++ ) {

          double ** data = init_2level_dtable ( num_conf, T_global );
          if ( data == NULL ) {
            fprintf( stderr, "[sff_threep_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }

          /* fill data array */
#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              data[iconf][it] = 0.;
              for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                data[iconf][it] += corr[k][iconf][isrc][2*it+ireim];
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
          if ( k == 0 ) {
            sprintf ( obs_name, "%s.gf%d.gi%d.px%d_py%d_pz%d.%s", twop_flavor_tag[ twop_flavor_type],
                sink_gamma_id, source_gamma_id,
                g_sink_momentum_list[0][0],
                g_sink_momentum_list[0][1],
                g_sink_momentum_list[0][2],
                reim_str[ireim] );
          } else {
            sprintf ( obs_name, "%s.gf%d.gi%d.px%d_py%d_pz%d.%s", twop_flavor_tag[ twop_flavor_type],
                source_gamma_id, sink_gamma_id,
                g_source_momentum_list[0][0],
                g_source_momentum_list[0][1],
                g_source_momentum_list[0][2],
                reim_str[ireim] );
          }

          if ( num_conf >= 6 ) {
            /* apply UWerr analysis */
            exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[sff_threep_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
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
                fprintf ( stderr, "[sff_threep_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(1);
              }

            }
          }

          sprintf ( filename, "%s.corr" , obs_name );
          FILE * dfs = fopen ( filename, "w" );
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              fprintf ( dfs, "%3d %26.16e   %c %6d\n", it, data[iconf][it], conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
            }
          }
          fclose ( dfs );

          fini_2level_dtable ( &data );

        }  /* end of loop on re / im */
      }  /* end of loop on pf, pi */

#endif  /* of if TWOP_STATS */

      /**********************************************************/
      /**********************************************************/
  
      /***********************************************************
       ***********************************************************
       **
       ** THREEP
       **
       ***********************************************************
       ***********************************************************/

#if THREEP_STATS

      /*****************************************************************
       * loop on sequential source gamma ids
       *****************************************************************/
      for ( int iseq_gamma = 0; iseq_gamma < g_sequential_source_gamma_id_number; iseq_gamma++ ) {

        int seq_source_gamma = g_sequential_source_gamma_id_list[iseq_gamma];

        /*****************************************************************
         * loop on sequential source timeslices
         *****************************************************************/
        for ( int iseq_timeslice = 0; iseq_timeslice < g_sequential_source_timeslice_number; iseq_timeslice++ ) {

          double **** corr_threep = init_4level_dtable ( 2, num_conf, num_src_per_conf, 2 * T_global );
          if ( corr_threep == NULL ) {
            fprintf(stderr, "[sff_threep_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(16);
          }

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

              gettimeofday ( &ta, (struct timezone *)NULL );

              sprintf ( filename, "stream_%c/%s.%.4d.t%d.h5",  conf_src_list[iconf][isrc][0], filename_prefix, Nconf, gsx[0] );

              fprintf(stdout, "# [sff_threep_analyse] reading data from file %s\n", filename);

              /**********************************************************
               * loop on momenta
               **********************************************************/
              int orbit_size = 0;
              for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

                int const pf[3] = {
                  g_sink_momentum_list[isink_momentum][0],
                  g_sink_momentum_list[isink_momentum][1],
                  g_sink_momentum_list[isink_momentum][2] };

              for ( int isource_momentum = 0; isource_momentum < g_source_momentum_number; isource_momentum++ ) {

                int const pi[3] = {
                  g_source_momentum_list[isource_momentum][0],
                  g_source_momentum_list[isource_momentum][1],
                  g_source_momentum_list[isource_momentum][2] };

                int const pc[3] = { 
                  -( pf[0] + pi[0] ),
                  -( pf[1] + pi[1] ),
                  -( pf[2] + pi[2] ) };

                int const pc2 = 
                  pc[0] * pc[0] +
                  pc[1] * pc[1] +
                  pc[2] * pc[2];

                if ( pc2 != ( g_insertion_momentum_list[0][0] * g_insertion_momentum_list[0][0] +
                              g_insertion_momentum_list[0][1] * g_insertion_momentum_list[0][1] +
                              g_insertion_momentum_list[0][2] * g_insertion_momentum_list[0][2] ) ) {
                  if ( g_verbose > 4 ) fprintf ( stdout, "# [sff_threep_analyse] Warning, skip momentum configuration pf = (%3d, %3d, %3d) pc = (%3d, %3d, %3d) pi = (%3d, %3d, %3d) %s %d\n", 
                      pf[0], pf[1], pf[2],
                      pc[0], pc[1], pc[2],
                      pi[0], pi[1], pi[2] , __FILE__, __LINE__ );
                  continue;
                } else {
                  if ( g_verbose > 4 ) fprintf ( stdout, "# [sff_threep_analyse] adding momentum configuration pf = (%3d, %3d, %3d) pc = (%3d, %3d, %3d) pi = (%3d, %3d, %3d) %s %d\n", 
                      pf[0], pf[1], pf[2],
                      pc[0], pc[1], pc[2],
                      pi[0], pi[1], pi[2] , __FILE__, __LINE__ );

                  orbit_size++;
                }


                double * buffer = init_1level_dtable( 2 * T_global );
                if( buffer == NULL ) {
                  fprintf(stderr, "[sff_threep_analyse] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
                  EXIT(15);
                }

                gettimeofday ( &ta, (struct timezone *)NULL );
 
                /* /sud+-g-u-g/t2/dt16//gf5/pfx0pfy-1pfz0/gc5/gi5/pix1piy0piz0 */

                sprintf ( key , "/%s/t%d/dt%d/gf%d/pfx%dpfy%dpfz%d/gc%d/gi%d/pix%dpiy%dpiz%d", threep_flavor_tag[ threep_flavor_type ],
                      gsx[0], g_sequential_source_timeslice_list[iseq_timeslice], 
                      sink_gamma_id, pf[0], pf[1], pf[2],
                      seq_source_gamma, 
                      source_gamma_id, pi[0], pi[1], pi[2] );

                if ( g_verbose > 2 ) fprintf ( stdout, "# [sff_threep_analyse] key = %s\n", key );

                exitstatus = read_from_h5_file ( buffer, filename, key, "double", io_proc );
                if( exitstatus != 0  ) {
                  fprintf(stderr, "[sff_threep_analyse] Error from read_from_h5_file %s %s %s %d\n", filename, key, __FILE__, __LINE__);
                  EXIT(15);
                }

                gettimeofday ( &tb, (struct timezone *)NULL );
                show_time ( &ta, &tb, "sff_threep_analyse", "read-h5-key", g_cart_id == 0 );
          
                /**********************************************************
                 * sort data from buffer into hvp,
                 * add source phase
                 **********************************************************/
#pragma omp parallel for
                for ( int it = 0; it < T_global; it++ ) {
                  int const tt = ( it - gsx[0] + T_global ) % T_global;
                  corr_threep[0][iconf][isrc][2*tt  ] += buffer[2*it  ];
                  corr_threep[0][iconf][isrc][2*tt+1] += buffer[2*it+1];
                }

                fini_1level_dtable( &buffer );

              }  /* end of loop on source momenta */
              }  /* end of loop on sink momenta */

              if ( g_verbose > 4 ) fprintf ( stdout, "# [sff_threep_analyse] orbit_size = %d %s %d\n", orbit_size, __FILE__, __LINE__ );
              for ( int it = 0; it < 2*T_global; it++ ) {
                corr_threep[0][iconf][isrc][it  ] /= (double)orbit_size;
              }

              /**********************************************************
               * if source and sink orbits differ, 
               * read inverted momentum setup,
               * otherwise, copy
               **********************************************************/

              if ( ( g_sink_momentum_list[0][0] * g_sink_momentum_list[0][0] +
                     g_sink_momentum_list[0][1] * g_sink_momentum_list[0][1] +
                     g_sink_momentum_list[0][2] * g_sink_momentum_list[0][2]
                   ) != (
                     g_source_momentum_list[0][0] * g_source_momentum_list[0][0] +
                     g_source_momentum_list[0][1] * g_source_momentum_list[0][1] +
                     g_source_momentum_list[0][2] * g_source_momentum_list[0][2]
                   ) )
              {

                int orbit_size = 0;
                for ( int isource_momentum = 0; isource_momentum < g_source_momentum_number; isource_momentum++ ) {

                  int const pf[3] = {
                    g_source_momentum_list[isource_momentum][0],
                    g_source_momentum_list[isource_momentum][1],
                    g_source_momentum_list[isource_momentum][2] };
  
                for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

                  int const pi[3] = {
                    g_sink_momentum_list[isink_momentum][0],
                    g_sink_momentum_list[isink_momentum][1],
                    g_sink_momentum_list[isink_momentum][2] };
  
                  int const pc[3] = { 
                    -( pf[0] + pi[0] ),
                    -( pf[1] + pi[1] ),
                    -( pf[2] + pi[2] ) };
  
                  int const pc2 = 
                    pc[0] * pc[0] +
                    pc[1] * pc[1] +
                    pc[2] * pc[2];
  
                  if ( pc2 != ( g_insertion_momentum_list[0][0] * g_insertion_momentum_list[0][0] +
                                g_insertion_momentum_list[0][1] * g_insertion_momentum_list[0][1] +
                                g_insertion_momentum_list[0][2] * g_insertion_momentum_list[0][2] ) ) {
                    if ( g_verbose > 4 ) fprintf ( stdout, "# [sff_threep_analyse] Warning, skip momentum configuration pf = (%3d, %3d, %3d) pc = (%3d, %3d, %3d) pi = (%3d, %3d, %3d) %s %d\n", 
                        pf[0], pf[1], pf[2],
                        pc[0], pc[1], pc[2],
                        pi[0], pi[1], pi[2] , __FILE__, __LINE__ );
                    continue;
                  } else {
                    if ( g_verbose > 4 ) fprintf ( stdout, "# [sff_threep_analyse] adding momentum configuration pf = (%3d, %3d, %3d) pc = (%3d, %3d, %3d) pi = (%3d, %3d, %3d) %s %d\n", 
                        pf[0], pf[1], pf[2],
                        pc[0], pc[1], pc[2],
                        pi[0], pi[1], pi[2] , __FILE__, __LINE__ );
  
                    orbit_size++;
                  }
  
  
                  double * buffer = init_1level_dtable( 2 * T_global );
                  if( buffer == NULL ) {
                    fprintf(stderr, "[sff_threep_analyse] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
                    EXIT(15);
                  }
  
                  gettimeofday ( &ta, (struct timezone *)NULL );
   
                  /* /sud+-g-u-g/t2/dt16//gf5/pfx0pfy-1pfz0/gc5/gi5/pix1piy0piz0 */
  
                  sprintf ( key , "/%s/t%d/dt%d/gf%d/pfx%dpfy%dpfz%d/gc%d/gi%d/pix%dpiy%dpiz%d", threep_flavor_tag[ threep_flavor_type ],
                        gsx[0], g_sequential_source_timeslice_list[iseq_timeslice], 
                        source_gamma_id, pf[0], pf[1], pf[2],
                        seq_source_gamma, 
                        sink_gamma_id, pi[0], pi[1], pi[2] );
  
                  if ( g_verbose > 2 ) fprintf ( stdout, "# [sff_threep_analyse] key = %s\n", key );
  
                  exitstatus = read_from_h5_file ( buffer, filename, key, "double", io_proc );
                  if( exitstatus != 0  ) {
                    fprintf(stderr, "[sff_threep_analyse] Error from read_from_h5_file %s %s %s %d\n", filename, key, __FILE__, __LINE__);
                    EXIT(15);
                  }
  
                  gettimeofday ( &tb, (struct timezone *)NULL );
                  show_time ( &ta, &tb, "sff_threep_analyse", "read-h5-key", g_cart_id == 0 );
            
                  /**********************************************************
                   * sort data from buffer into hvp,
                   * add source phase
                   **********************************************************/
  #pragma omp parallel for
                  for ( int it = 0; it < T_global; it++ ) {
                    int const tt = ( it - gsx[0] + T_global ) % T_global;
                    corr_threep[1][iconf][isrc][2*tt  ] += buffer[2*it  ];
                    corr_threep[1][iconf][isrc][2*tt+1] += buffer[2*it+1];
                  }
  
                  fini_1level_dtable( &buffer );
  
                }  /* end of loop on source momenta */
                }  /* end of loop on sink momenta */
  
                if ( g_verbose > 4 ) fprintf ( stdout, "# [sff_threep_analyse] orbit_size = %d %s %d\n", orbit_size, __FILE__, __LINE__ );
                for ( int it = 0; it < 2*T_global; it++ ) {
                  corr_threep[1][iconf][isrc][it  ] /= (double)orbit_size;
                }

              } else {
                memcpy( corr_threep[1][0][0], corr_threep[0][0][0] , num_conf * num_src_per_conf * 2 * T_global * sizeof ( double ) );
              }

            }  /* end of loop on source locations */
          }   /* end of loop on configurations */

          /**********************************************************
           * show all data
           **********************************************************/
          if ( g_verbose > 5 ) {
            gettimeofday ( &ta, (struct timezone *)NULL );
            FILE *ofs = fopen ( "sff_threep_analyse.data", "w" );
    
            for ( int iconf = 0; iconf < num_conf; iconf++ )
            {
              for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
              {
                for ( int it = 0; it < T; it++ )
                {
                    fprintf ( ofs, "c %6d s %3d dt %2d pf %3d %3d %3d pc %3d %3d %3d  gf %d gc %d gi %d  corr_threep %3d  %25.16e %25.16e\n", iconf, isrc, 
                        g_sequential_source_timeslice_list[iseq_timeslice],
                        g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], 
                        g_insertion_momentum_list[0][0], g_insertion_momentum_list[0][1], g_insertion_momentum_list[0][2], 
                        g_sink_gamma_id_list[ isink_gamma ], seq_source_gamma, g_source_gamma_id_list[ isource_gamma ], it, 
                        corr_threep[0][iconf][isrc][2*it], corr_threep[0][iconf][isrc][2*it+1] );
                }
              }
            }

            fclose ( ofs );
            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "sff_threep_analyse", "show-all-data", g_cart_id == 0 );
          }
 
          /****************************************
           * STATISTICAL ANALYSIS
           ****************************************/
    
          for ( int k = 0; k <= 1; k++ ) {

            for ( int ireim = 0; ireim < 2; ireim++ ) {
      
              double ** data = init_2level_dtable ( num_conf, T_global );
              if ( data == NULL ) {
                fprintf( stderr, "[sff_threep_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
                EXIT(12);
              }
      
              /* fill data array */
  #pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for ( int it = 0; it < T_global; it++ ) {
                  data[iconf][it] = 0.;
                  for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                    data[iconf][it] += corr_threep[k][iconf][isrc][2*it+ireim];
                  }
                  data[iconf][it] /= (double)num_src_per_conf;
                }
              }
   
              /****************************************
               * UWerr analysis of real and
               * imaginary part
               ****************************************/
      
              char obs_name[100];
              if ( k == 0 ) {
                sprintf ( obs_name, "%s.dt%d.gf%d.gc%d.gi%d.pfx%d_pfy%d_pfz%d.pcx%d_pcy%d_pcz%d.pix%d_piy%d_piz%d.%s", threep_flavor_tag[ threep_flavor_type],
                    g_sequential_source_timeslice_list[iseq_timeslice],
                    sink_gamma_id, seq_source_gamma, source_gamma_id, 
                    g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2],
                    g_insertion_momentum_list[0][0], g_insertion_momentum_list[0][1], g_insertion_momentum_list[0][2],
                    g_source_momentum_list[0][0], g_source_momentum_list[0][1], g_source_momentum_list[0][2],
                    /* -( g_sink_momentum_list[0][0]  + g_insertion_momentum_list[0][0] ),
                    -( g_sink_momentum_list[0][1]  + g_insertion_momentum_list[0][1] ),
                    -( g_sink_momentum_list[0][2]  + g_insertion_momentum_list[0][2] ), */
                      reim_str[ireim] );
              } else {
                sprintf ( obs_name, "%s.dt%d.gf%d.gc%d.gi%d.pfx%d_pfy%d_pfz%d.pcx%d_pcy%d_pcz%d.pix%d_piy%d_piz%d.%s", threep_flavor_tag[ threep_flavor_type],
                    g_sequential_source_timeslice_list[iseq_timeslice],
                    source_gamma_id, seq_source_gamma, sink_gamma_id, 
                    g_source_momentum_list[0][0], g_source_momentum_list[0][1], g_source_momentum_list[0][2],
                    g_insertion_momentum_list[0][0], g_insertion_momentum_list[0][1], g_insertion_momentum_list[0][2],
                    g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2],
                    reim_str[ireim] );
              }
      
              /* apply UWerr analysis */
              if ( num_conf >= 6 ) {
                exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
                if ( exitstatus != 0 ) {
                  fprintf ( stderr, "[sff_threep_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(1);
                }
              }
     
              char data_filename[200];
  
              sprintf ( data_filename, "%s.corr", obs_name );
  
              FILE * dfs = fopen ( data_filename, "w" );
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for ( int it = 0; it < T_global; it++ ) {
                  fprintf( dfs, "%4d %25.16e   %c %6d\n", it, data[iconf][it], conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
                }
              }
  
              fclose ( dfs );
  
              fini_2level_dtable ( &data );
      
            }  /* end of loop on re / im */
          }  /* end of loop on pf, pi */

          /**********************************************************
           * analyse ratio R1
           **********************************************************/

          double ** data = init_2level_dtable ( num_conf, 2*T_global + 2 );
          if ( data == NULL ) {
            fprintf( stderr, "[sff_threep_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }

          /* fill data array */
  #pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              data[iconf][it] = 0.;
              for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                data[iconf][           it] += corr_threep[0][iconf][isrc][2*it+1];
                data[iconf][T_global + it] += corr_threep[1][iconf][isrc][2*it+1];
              }
              data[iconf][           it] /= (double)num_src_per_conf;
              data[iconf][T_global + it] /= (double)num_src_per_conf;
            }
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              data[iconf][2*T_global  ] += corr[0][iconf][isrc][ 2 * g_sequential_source_timeslice_list[iseq_timeslice] ];
              data[iconf][2*T_global+1] += corr[1][iconf][isrc][ 2 * g_sequential_source_timeslice_list[iseq_timeslice] ];
            }
            data[iconf][2*T_global  ] /= (double)num_src_per_conf;
            data[iconf][2*T_global+1] /= (double)num_src_per_conf;

          }

          int narg = 4;
          int arg_first[4] = { 0, T_global,  2 * T_global, 2 * T_global + 1 };
          int arg_stride[4] = {1,1,0,0};
          int nT = T_global;

          char obs_name2[500];
          sprintf ( obs_name2, "R1.dt%d.gf%d.gc%d.gi%d.pfx%d_pfy%d_pfz%d.pcx%d_pcy%d_pcz%d.pix%d_piy%d_piz%d",
                    g_sequential_source_timeslice_list[iseq_timeslice],
                    sink_gamma_id, seq_source_gamma, source_gamma_id,
                    g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2],
                    g_insertion_momentum_list[0][0], g_insertion_momentum_list[0][1], g_insertion_momentum_list[0][2],
                    g_source_momentum_list[0][0], g_source_momentum_list[0][1], g_source_momentum_list[0][2]);

          exitstatus = apply_uwerr_func ( data[0], num_conf, 2*T_global+2, nT, narg, arg_first, arg_stride, obs_name2, sqrt_ab_over_cd, dsqrt_ab_over_cd );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(115);
          }

          fini_2level_dtable ( &data ) ;

          fini_4level_dtable ( &corr_threep );

        }  /* end of loop on seq. source timeslice */
      }  /* end of loop on seq source gamma */

#endif  /* of THREEP_STATS */

    }  /* end of loop on source gamma id */

  }  /* end of loop on sink gamma id */

  /**********************************************************
   * free
   **********************************************************/
  fini_4level_dtable ( &corr );

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
    fprintf(stdout, "# [sff_threep_analyse] %s# [sff_threep_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [sff_threep_analyse] %s# [sff_threep_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
