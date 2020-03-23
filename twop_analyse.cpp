/****************************************************
 * twop_analyse
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

void usage() {
  fprintf(stdout, "Code to analyse P-J-J correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  EXIT(0);
}

#define _TWOP_AFF_SINGLE
#undef _TWOP_AFF_MULT

int main(int argc, char **argv) {
  
  double const TWO_MPI = 2. * M_PI;

  char const reim_str[2][3] = {"re" , "im"};

  char const twop_correlator_prefix[1][20] = { "N-N" };

  /* char const twop_flavor_tag[4][20]        = { }; */

  int c;
  int filename_set = 0;
  int check_momentum_space_WI = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "NA";
  int fold_correlator= 0;
  struct timeval ta, tb;
  int correlator_type = -1;
  int flavor_type = -1;

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  char key[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:F:c:s:E:")) != -1) {
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
  double ******* corr = init_7level_dtable ( num_conf, num_src_per_conf, g_sink_momentum_number, g_sink_gamma_id_number, g_source_gamma_id_number, 2,  2 * T_global );
  if ( corr == NULL ) {
    fprintf(stderr, "[twop_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

#ifdef _TWOP_AFF_MULT
  /***********************************************************
   * loop on configs and source locations per config
   ***********************************************************/
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      /***********************************************************
       * copy source coordinates
       ***********************************************************/
      int const gsx[4] = {
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][2],
          conf_src_list[iconf][isrc][3],
          conf_src_list[iconf][isrc][4] };

      /***********************************************
       * reader for aff input file
       ***********************************************/

      gettimeofday ( &ta, (struct timezone *)NULL );
      struct AffNode_s *affn = NULL, *affdir = NULL;

      sprintf ( filename, "stream_%c/%d/%s.%.4d.t%dx%dy%dz%d.aff", conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1], 
          g_outfile_prefix, conf_src_list[iconf][isrc][1], 
          conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );

      fprintf(stdout, "# [twop_analyse] reading data from file %s\n", filename);
      affr = aff_reader ( filename );
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

        sprintf ( key , "/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d/gi%.2d/px%dpy%dpz%d", twop_correlator_prefix[ correlator_type ], twop_flavor_tag[ flavor_type ],
            gsx[0], gsx[1], gsx[2], gsx[3], sink_gamma_id, source_gamma_id, sink_momentum[0], sink_momentum[1], sink_momentum[2] );

        /**********************************************************
         * loop on momenta
         **********************************************************/
        for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

          int const sink_momentum[3] = {
            g_sink_momentum_list[isink_momentum][0],
            g_sink_momentum_list[isink_momentum][1],
            g_sink_momentum_list[isink_momentum][2] };

          double _Complex *** buffer = init_1level_dtable( 2 * T );
          if( buffer == NULL ) {
            fprintf(stderr, "[twop_analyse] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(15);
          }

          gettimeofday ( &ta, (struct timezone *)NULL );

          /* sprintf ( key , "/%s/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d/gi%.2d/px%dpy%dpz%d", twop_correlator_prefix[ correlator_type ], twop_flavor_tag[ flavor_type ],
              gsx[0], gsx[1], gsx[2], gsx[3], sink_gamma_id, source_gamma_id, sink_momentum[0], sink_momentum[1], sink_momentum[2] ); */

          sprintf ( key , "/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d/gi%.2d/px%dpy%dpz%d", twop_correlator_prefix[ correlator_type ], twop_flavor_tag[ flavor_type ],
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

            corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt  ] = creal( ztmp );
            corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt+1] = cimag( ztmp );
          }

          fini_1level_dtable( &buffer );

        }  /* end of loop on sink momenta */

      }  /* end of loop on source gamma id */

      }  /* end of loop on sink gamma id */

      aff_reader_close ( affr );

    }  /* end of loop on source locations */

  }   /* end of loop on configurations */

#endif

#ifdef _TWOP_AFF_SINGLE
  for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ ) {

    int const sink_gamma_id = g_sink_gamma_id_list[ isink_gamma ];

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

        gettimeofday ( &ta, (struct timezone *)NULL );

        sprintf ( filename, "%s/%s.%s.gf%.2d.gi%.2d.px%dpy%dpz%d.aff", filename_prefix, twop_correlator_prefix[correlator_type],
            twop_flavor_tag[flavor_type], sink_gamma_id, source_gamma_id, sink_momentum[0], sink_momentum[1], sink_momentum[2] );

        /***********************************************
         * reader for aff input file
         ***********************************************/

        struct AffNode_s *affn = NULL, *affdir = NULL;
  
        fprintf(stdout, "# [twop_analyse] reading data from file %s\n", filename);
        affr = aff_reader ( filename );
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
  
              corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt  ] = creal( ztmp );
              corr[iconf][isrc][isink_momentum][isink_gamma][isource_gamma][2*tt+1] = cimag( ztmp );
            }
        
          }  /* end of loop on source locations */
        }   /* end of loop on configurations */

        aff_reader_close ( affr );
        fini_1level_dtable( &buffer );

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "twop_analyse", "open-init-aff-reader", g_cart_id == 0 );

      }  /* end of loop on sink momenta */

    }  /* end of loop on source gamma id */

  }  /* end of loop on sink gamma id */

#endif

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
   * STATISTICAL ANALYSIS
   ****************************************/

  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

    int const momentum[3] = {
        g_sink_momentum_list[imom][0],
        g_sink_momentum_list[imom][1],
        g_sink_momentum_list[imom][2] };

    for ( int isink_gamma   = 0; isink_gamma   < g_sink_gamma_id_number;   isink_gamma++ ) {

      int const sink_gamma_id = g_sink_gamma_id_list[ isink_gamma ];

    for ( int isource_gamma = 0; isource_gamma < g_source_gamma_id_number; isource_gamma++ ) {

      int const source_gamma_id = g_source_gamma_id_list[ isource_gamma ];

      for ( int ireim = 0; ireim < 2; ireim++ ) {

        double *** data = init_3level_dtable ( num_conf, num_src_per_conf, T_global );

        /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              data[iconf][isrc][it] = corr[iconf][isrc][imom][isink_gamma][isource_gamma][2*it+ireim];
            }
          }
        }

        if ( fold_correlator ) {
#pragma omp parallel for
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              for ( int it = 1; it < T_global/2; it++ ) {
                data[iconf][isrc][it] += data[iconf][isrc][T_global - it];
                data[iconf][isrc][it] *= 0.5;
                data[iconf][isrc][T_global - it] = data[iconf][isrc][it];
              }
            }
          }
        }

       /****************************************
        * STATISTICAL ANALYSIS of real and
        * imaginary part
        ****************************************/

        char obs_name[100];
        sprintf ( obs_name, "%s.%s.gf%d.gi%d.PX%d_PY%d_PZ%d.%s", twop_correlator_prefix[correlator_type], twop_flavor_tag[ flavor_type],
            sink_gamma_id, source_gamma_id, momentum[0], momentum[1], momentum[2], reim_str[ireim] );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0][0], num_conf*num_src_per_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[twop_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }


        /****************************************
         * STATISTICAL ANALYSIS of effective
         * mass from time-split acosh ratio
         ****************************************/
#if 0
        for ( int itau = 1; itau < T_global/2; itau++ )
        {

          char obs_name2[200];
          sprintf( obs_name2, "%s.acoshratio.tau%d", obs_name, itau );

          int arg_first[3]  = { 0, 2*itau, itau };
          int arg_stride[3] = {1, 1, 1};

          exitstatus = apply_uwerr_func ( data[0][0], num_conf*num_src_per_conf, T_global, T_global/2-itau, 3, arg_first, arg_stride, obs_name, acosh_ratio, dacosh_ratio );

          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[twop_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
        }
#endif

        fini_3level_dtable ( &data );

      }  /* end of loop on re / im */
    }}  /* end of loop on source and sink gamma id */
  }  /* end of loop on momenta */

  /**********************************************************
   * free hvp field
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
