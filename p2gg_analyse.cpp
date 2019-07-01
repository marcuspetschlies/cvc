/****************************************************
 * p2gg_analyse.c
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

  char const hvp_operator_type_tag[3][12]  = { "cvc-cvc"    , "lvc-lvc"    , "cvc-lvc" };

  char const hvp_correlator_prefix[3][20] = { "hvp"        , "local-local", "hvp"     };

  char const hvp_flavor_tag[3][20]        = { "u-cvc-u-cvc", "u-gf-u-gi"  , "u-cvc-u-lvc" };

  /* char const pgg_operator_type_tag[3][12]  = { "p-cvc-cvc"    , "p-lvc-lvc"    , "p-cvc-lvc" }; */
  char const pgg_operator_type_tag[3][12]  = { "p-cvc-cvc"    , "p-loc-loc"    , "p-cvc-lvc" };


  /***********************************************************
   * sign for g5 Gamma^\dagger g5
   *                                                0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   *
   ***********************************************************/
  int const sequential_source_gamma_id_sign[16] ={ -1, -1, -1, -1, +1, +1,  +1,  +1,  +1,  +1,  -1,  -1,  -1,  -1,  -1,  -1 };

  int c;
  int filename_set = 0;
  int check_momentum_space_WI = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "cA211a.30.32";
  int fold_correlator= 0;
  struct timeval ta, tb;
  int operator_type = -1;
  char correlator_prefix[100], flavor_tag[100];

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  char key[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "Wh?f:N:S:F:O:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'W':
      check_momentum_space_WI = 1;
      fprintf ( stdout, "# [p2gg_analyse] check_momentum_space_WI set to %d\n", check_momentum_space_WI );
      break;
    case 'F':
      fold_correlator = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse] fold_correlator set to %d\n", fold_correlator );
      break;
    case 'O':
      operator_type = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse] operator_type set to %d\n", operator_type );
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
  /* fprintf(stdout, "# [p2gg_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [p2gg_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[p2gg_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [p2gg_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf);
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[p2gg_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 5 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[p2gg_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [p2gg_analyse] comment %s\n", line );
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

#if 0
  /***********************************************************
   ***********************************************************
   **
   ** HVP
   **
   ***********************************************************
   ***********************************************************/
  double ****** hvp = init_6level_dtable ( num_conf, num_src_per_conf, g_sink_momentum_number, 4, 4, 2 * T );
  if ( hvp == NULL ) {
    fprintf(stderr, "[p2gg_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__);
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

      sprintf ( filename, "%d/%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", Nconf, g_outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
      /* sprintf ( filename, "%d/%s.%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", Nconf, hvp_correlator_prefix[operator_type], hvp_flavor_tag[operator_type], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] ); */
      fprintf(stdout, "# [p2gg_analyse] reading data from file %s\n", filename);
      affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }

      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[p2gg_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        return(103);
      }

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "p2gg_analyse", "open-init-aff-reader", g_cart_id == 0 );
#endif


      /**********************************************************
       * loop on momenta
       **********************************************************/
      for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

        int const sink_momentum[3] = {
          g_sink_momentum_list[isink_momentum][0],
          g_sink_momentum_list[isink_momentum][1],
          g_sink_momentum_list[isink_momentum][2] };

        double *** buffer = init_3level_dtable( 4, 4, 2 * T );
        if( buffer == NULL ) {
          fprintf(stderr, "[p2gg_analyse] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(15);
        }

        if ( operator_type == 0 || operator_type == 2 ) {

          gettimeofday ( &ta, (struct timezone *)NULL );

          sprintf ( key , "/%s/%s/t%.2dx%.2dy%.2dz%.2d/px%.2dpy%.2dpz%.2d", hvp_correlator_prefix[operator_type], hvp_flavor_tag[operator_type],

              gsx[0], gsx[1], gsx[2], gsx[3],
              sink_momentum[0], sink_momentum[1], sink_momentum[2] );

          if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse] key = %s\n", key );

          affdir = aff_reader_chpath (affr, affn, key );
          uint32_t uitems = 16 * T;
          exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer[0][0]), uitems );
          if( exitstatus != 0 ) {
            fprintf(stderr, "[p2gg_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(105);
          }

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "p2gg_analyse", "read-aff-key", g_cart_id == 0 );

        } else if ( operator_type == 1 ) {

          gettimeofday ( &ta, (struct timezone *)NULL );

          for( int mu = 0; mu < 4; mu++) {
          for( int nu = 0; nu < 4; nu++) {
            sprintf ( key , "/%s/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d/gi%.2d/px%dpy%dpz%d", hvp_correlator_prefix[operator_type], hvp_flavor_tag[operator_type],
                gsx[0], gsx[1], gsx[2], gsx[3], mu, nu, sink_momentum[0], sink_momentum[1], sink_momentum[2] );

            if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse] key = %s\n", key );
            affdir = aff_reader_chpath (affr, affn, key );
            uint32_t uitems = T;
            exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer[mu][nu]), uitems );
            if( exitstatus != 0 ) {
              fprintf(stderr, "[p2gg_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }
          }}

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "p2gg_analyse", "read-ll-tensor-aff", g_cart_id == 0 );
        }  /* end of if operator_type */

        /**********************************************************
         * loop on shifts in directions mu, nu
         **********************************************************/
        for( int mu = 0; mu < 4; mu++) {
        for( int nu = 0; nu < 4; nu++) {

          double const p[4] = {
              0., 
              TWO_MPI * (double)sink_momentum[0] / (double)LX_global,
              TWO_MPI * (double)sink_momentum[1] / (double)LY_global,
              TWO_MPI * (double)sink_momentum[2] / (double)LZ_global };

          double phase = 0.;
          if ( operator_type == 0 ) {
            phase = - ( p[0] * gsx[0] + p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] ) + 0.5 * ( p[mu] - p[nu] );
          } else if ( operator_type == 1 ) {
            phase = - ( p[0] * gsx[0] + p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] );
          } else if ( operator_type == 2 ) {
            phase = - ( p[0] * gsx[0] + p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] ) + 0.5 * ( p[mu] );
          }

          double _Complex ephase = cexp ( phase * I );

          /**********************************************************
           * sort data from buffer into hvp,
           * add source phase
           **********************************************************/
#pragma omp parallel for
          for ( int it = 0; it < T; it++ ) {
            int const tt = ( it - gsx[0] + T_global ) % T_global; 

            double _Complex ztmp = ( buffer[mu][nu][2*it] +  buffer[mu][nu][2*it+1] * I ) * ephase;

            hvp[iconf][isrc][isink_momentum][mu][nu][2*tt  ] = creal( ztmp );
            hvp[iconf][isrc][isink_momentum][mu][nu][2*tt+1] = cimag( ztmp );
          }

        }  /* end of loop on direction nu */
        }  /* end of loop on direction mu */

        fini_3level_dtable( &buffer );

      }  /* end of loop on sink momenta */

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

    for ( int iconf = 0; iconf < num_conf; iconf++ )
    {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
      {
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
          for ( int mu = 0; mu < 4; mu++ ) {
          for ( int nu = 0; nu < 4; nu++ ) {
            for ( int it = 0; it < T; it++ ) {
              fprintf ( stdout, "c %6d s %3d p %3d %3d %3d m %d %d hvp %3d  %25.16e %25.16e\n", iconf, isrc, 
                  g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], mu, nu, it, 
                  hvp[iconf][isrc][imom][mu][nu][2*it], hvp[iconf][isrc][imom][mu][nu][2*it+1] );
            }
          }}
        }
      }
    }
    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "p2gg_analyse", "show-all-data", g_cart_id == 0 );
  }

  /****************************************
   * check WI in momentum space
   ****************************************/
  if ( check_momentum_space_WI ) {
    gettimeofday ( &ta, (struct timezone *)NULL );

    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
  
          exitstatus = check_momentum_space_wi_tpvec ( hvp[iconf][isrc][imom], g_sink_momentum_list[imom] );
          if ( exitstatus != 0  ) {
            fprintf ( stderr, "[p2gg_analyse] Error from check_momentum_space_wi_tpvec, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(2);
          }

        }  /* end of loop on momenta */
  
      }  /* end of loop on sources per config */
    }  /* end of loop on configs */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "p2gg_analyse", "check-wi-in-momentum-space", g_cart_id == 0 );
  }  /* end of if check_momentum_space_WI */ 

  /****************************************
   * now some statistical analysis
   ****************************************/

  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

    int const momentum[3] = {
        g_sink_momentum_list[imom][0],
        g_sink_momentum_list[imom][1],
        g_sink_momentum_list[imom][2] };

    for( int mu = 0; mu < 4; mu++) {
    for( int nu = 0; nu < 4; nu++) {
      for ( int ireim = 0; ireim < 2; ireim++ ) {

        double *** data = init_3level_dtable ( num_conf, num_src_per_conf, T_global );

        /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              data[iconf][isrc][it] = hvp[iconf][isrc][imom][mu][nu][2*it+ireim];
            }
          }
        }

        char obs_name[100];
        sprintf ( obs_name, "%s.%s.jmu%d_jnu%d.PX%d_PY%d_PZ%d.%s", hvp_correlator_prefix[operator_type], hvp_flavor_tag[operator_type],
            mu, nu, momentum[0], momentum[1], momentum[2], reim_str[ireim] );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0][0], num_conf*num_src_per_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[p2gg_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        fini_3level_dtable ( &data );

      }  /* end of loop on re / im */
    }}  /* end of loop on nu, mu */
  }  /* end of loop on momenta */

  /**********************************************************
   * free hvp field
   **********************************************************/
  fini_6level_dtable ( &hvp );

#endif  /* of if 0  */


  /**********************************************************
   **********************************************************
   **
   ** P -> gg 3-point function
   **
   **********************************************************
   **********************************************************/
  double ****** pgg = init_6level_dtable ( num_conf, num_src_per_conf, g_sink_momentum_number, 4, 4, 2 * T );
  if ( pgg == NULL ) {
    fprintf(stderr, "[p2gg_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  for ( int iseq_source_momentum = 0; iseq_source_momentum < g_seq_source_momentum_number; iseq_source_momentum++) 
  {

    int const seq_source_momentum[3] = {
      g_seq_source_momentum_list[iseq_source_momentum][0],
      g_seq_source_momentum_list[iseq_source_momentum][1],
      g_seq_source_momentum_list[iseq_source_momentum][2] };

    for( int isequential_source_gamma_id = 0; isequential_source_gamma_id < g_sequential_source_gamma_id_number; isequential_source_gamma_id++)
    {

      int const sequential_source_gamma_id = g_sequential_source_gamma_id_list[ isequential_source_gamma_id ];

      for ( int isequential_source_timeslice = 0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++)
      {

        int const sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];

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
      
            sprintf ( filename, "%d/%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", Nconf, g_outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
            /* sprintf ( filename, "%d/%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", Nconf, pgg_operator_type_tag[operator_type], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] ); */
            fprintf(stdout, "# [p2gg_analyse] reading data from file %s\n", filename);
            affr = aff_reader ( filename );
            const char * aff_status_str = aff_reader_errstr ( affr );
            if( aff_status_str != NULL ) {
              fprintf(stderr, "[p2gg_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
              EXIT(15);
            }
      
            if( (affn = aff_reader_root( affr )) == NULL ) {
              fprintf(stderr, "[p2gg_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
              return(103);
            }
          
            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "p2gg_analyse", "open-init-aff-reader", g_cart_id == 0 );
#endif
            /**********************************************************
             * read the contact term
             **********************************************************/
            double contact_term[2][4][2] = {
             { {0,0}, {0,0}, {0,0}, {0,0} },
             { {0,0}, {0,0}, {0,0}, {0,0} } };

            if ( operator_type == 0 ) {
              gettimeofday ( &ta, (struct timezone *)NULL );

              for ( int iflavor = 0; iflavor <= 1; iflavor++ )
              {

                int const flavor_id = 1 - 2 * iflavor;

                for ( int mu = 0; mu < 4; mu++ ) {
                  sprintf ( key , "/%s/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/fl%d/contact_term/mu%d", pgg_operator_type_tag[operator_type],
                      gsx[0], gsx[1], gsx[2], gsx[3],
                      flavor_id * seq_source_momentum[0],
                      flavor_id * seq_source_momentum[1],
                      flavor_id * seq_source_momentum[2],
                      sequential_source_gamma_id, sequential_source_timeslice,
                      iflavor, mu );

                  if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse] contact term key = %s\n", key );

                  affdir = aff_reader_chpath (affr, affn, key );
                  exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(contact_term[iflavor][mu]), 1 );
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[p2gg_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse] contact term fl %d mu %d   %25.16e %25.16e\n", iflavor, mu,
                      contact_term[iflavor][mu][0], contact_term[iflavor][mu][1] );

                }  /* end of loop on mu */
              }  /* end of loop on flavor id */
              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "p2gg_analyse", "read-contact-term-aff", g_cart_id == 0 );
            }  /* end of if operator_type == 0 */

            if ( g_verbose > 4 ) {
              fprintf ( stdout, "# [p2gg_analyse] conf %6d src %3d %3d %3d %3d contact term\n", Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
              for ( int mu = 0; mu < 4; mu++ ) {
                fprintf ( stdout, "    %d   %25.16e %25.16e   %25.16e %25.16e\n", 0,
                    contact_term[0][mu][0], contact_term[0][mu][1], contact_term[1][mu][0], contact_term[1][mu][1] );
              }
            }

            /**********************************************************
             * loop on momenta
             **********************************************************/
            for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

              int const sink_momentum[3] = {
                  g_sink_momentum_list[isink_momentum][0],
                  g_sink_momentum_list[isink_momentum][1],
                  g_sink_momentum_list[isink_momentum][2] };
      
              double **** buffer = init_4level_dtable( 2, 4, 4, 2 * T );
              if( buffer == NULL ) {
                fprintf(stderr, "[p2gg_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__);
                EXIT(15);
              }
      
              for ( int iflavor = 0; iflavor <= 1; iflavor++ )
              {

                int const flavor_id = 1 - 2 * iflavor;

                if ( operator_type == 0 || operator_type == 2 ) {

                  gettimeofday ( &ta, (struct timezone *)NULL );

                  sprintf ( key , "/%s/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/fl%d/px%.2dpy%.2dpz%.2d", pgg_operator_type_tag[operator_type],
                      gsx[0], gsx[1], gsx[2], gsx[3],
                      flavor_id * seq_source_momentum[0],
                      flavor_id * seq_source_momentum[1],
                      flavor_id * seq_source_momentum[2],
                      sequential_source_gamma_id, sequential_source_timeslice,
                      iflavor,
                      flavor_id * sink_momentum[0],
                      flavor_id * sink_momentum[1],
                      flavor_id * sink_momentum[2] );
      
                  if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse] pgg key = %s\n", key );
      
                  affdir = aff_reader_chpath (affr, affn, key );
                  uint32_t uitems = 16 * T;
                  exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer[iflavor][0][0]), uitems );
                  if( exitstatus != 0 ) {
                    fprintf(stderr, "[p2gg_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(105);
                  }

                  gettimeofday ( &tb, (struct timezone *)NULL );
                  show_time ( &ta, &tb, "p2gg_analyse", "read-pgg-tensor-aff", g_cart_id == 0 );

                } else if ( operator_type == 1 ) {

                  gettimeofday ( &tb, (struct timezone *)NULL );
                  
                  show_time ( &ta, &tb, "p2gg_analyse", "read-pgg-tensor-aff", g_cart_id == 0 );
                  for ( int mu = 0; mu < 4; mu++ ) {
                  for ( int nu = 0; nu < 4; nu++ ) {
                    sprintf ( key , "/%s/t%.2dx%.2dy%.2dz%.2d/qx%.2dqy%.2dqz%.2d/gseq%.2d/tseq%.2d/fl%d/gf%.2d/gi%.2d/px%dpy%dpz%d", pgg_operator_type_tag[operator_type],
                        gsx[0], gsx[1], gsx[2], gsx[3],
                        flavor_id * seq_source_momentum[0],
                        flavor_id * seq_source_momentum[1],
                        flavor_id * seq_source_momentum[2],
                        sequential_source_gamma_id, sequential_source_timeslice,
                        iflavor,
                        mu, nu,
                        flavor_id * sink_momentum[0],
                        flavor_id * sink_momentum[1],
                        flavor_id * sink_momentum[2] );
      
                    if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse] key = %s\n", key );
      
                    affdir = aff_reader_chpath (affr, affn, key );
                    uint32_t uitems = T;
                    exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer[iflavor][mu][nu]), uitems );
                    if( exitstatus != 0 ) {
                      fprintf(stderr, "[p2gg_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(105);
                    }
                  }}

                  gettimeofday ( &tb, (struct timezone *)NULL );
                  show_time ( &ta, &tb, "p2gg_analyse", "read-pgg-tensor-components-aff", g_cart_id == 0 );
                }  /* end of if operator_type */

              }  /* end of loop on flavors */

              /**********************************************************
               * loop on shifts in directions mu, nu
               **********************************************************/
              for( int mu = 0; mu < 4; mu++) {
               for( int nu = 0; nu < 4; nu++) {
      
                double const p[4] = { 0., 
                    TWO_MPI * (double)sink_momentum[0] / (double)LX_global,
                    TWO_MPI * (double)sink_momentum[1] / (double)LY_global,
                    TWO_MPI * (double)sink_momentum[2] / (double)LZ_global };

                if (g_verbose > 4 ) fprintf ( stdout, "# [p2gg_analyse] p = %25.16e %25.16e %25.16e %25.16e\n", p[0], p[1], p[2], p[3] ); 

                double const q[4] = { 0., 
                    TWO_MPI * (double)seq_source_momentum[0] / (double)LX_global,
                    TWO_MPI * (double)seq_source_momentum[1] / (double)LY_global,
                    TWO_MPI * (double)seq_source_momentum[2] / (double)LZ_global };
                
                if ( g_verbose > 4 ) fprintf ( stdout, "# [p2gg_analyse] q = %25.16e %25.16e %25.16e %25.16e\n", q[0], q[1], q[2], q[3] ); 
      
                double p_phase = 0., q_phase = 0.;

                if ( operator_type == 0 ) {
                  /*                     p_src^nu x_src^nu                                                   + 1/2 p_snk^mu - 1/2 p_snk^nu */
                  p_phase = -( p[0] * gsx[0] + p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] ) + 0.5 * ( p[mu] - p[nu] );
                  q_phase = -( q[0] * gsx[0] + q[1] * gsx[1] + q[2] * gsx[2] + q[3] * gsx[3] ) + 0.5 * (       - q[nu] );
                } else if ( operator_type == 1 ) {
                  p_phase = -( p[0] * gsx[0] + p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] );
                  q_phase = -( q[0] * gsx[0] + q[1] * gsx[1] + q[2] * gsx[2] + q[3] * gsx[3] );

                } else if ( operator_type == 2 ) {
                  p_phase = -( p[0] * gsx[0] + p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] ) + 0.5 * ( p[mu] );
                  q_phase = -( q[0] * gsx[0] + q[1] * gsx[1] + q[2] * gsx[2] + q[3] * gsx[3] );
                }

                double _Complex const p_ephase = cexp ( p_phase * I );
                double _Complex const q_ephase = cexp ( q_phase * I );
      
                if ( g_verbose > 4 ) {
                  fprintf ( stdout, "# [p2gg_analyse] p %3d %3d %3d x %3d %3d %3d %3d p_phase %25.16e p_ephase %25.16e %25.16e\n",
                      sink_momentum[0], sink_momentum[1], sink_momentum[2], 
                      gsx[0], gsx[1], gsx[2], gsx[3], p_phase, creal( p_ephase ), cimag( p_ephase ) );
                  
                  fprintf ( stdout, "# [p2gg_analyse] q %3d %3d %3d x %3d %3d %3d %3d q_phase %25.16e q_ephase %25.16e %25.16e\n",
                      sink_momentum[0], sink_momentum[1], sink_momentum[2], 
                      gsx[0], gsx[1], gsx[2], gsx[3], q_phase, creal( q_ephase ), cimag( q_ephase ) );
                      
                }
      
                /**********************************************************
                 * sort data from buffer into pgg,
                 * multiply source phase
                 **********************************************************/
#pragma omp parallel for
                for ( int it = 0; it < T; it++ ) {

                  /**********************************************************
                   * order from source time
                   **********************************************************/
                  int const tt = ( it - gsx[0] + T_global ) % T_global; 
      
                  /**********************************************************
                   * add the two flavor components
                   **********************************************************/
                  double _Complex ztmp = ( 
                        ( buffer[0][mu][nu][2*it] +  buffer[0][mu][nu][2*it+1] * I )
                      + ( buffer[1][mu][nu][2*it] -  buffer[1][mu][nu][2*it+1] * I ) * (double)sequential_source_gamma_id_sign[ sequential_source_gamma_id ]
                      ) * p_ephase;

                  /**********************************************************
                   * for mu == nu and at source time subtract the contact term
                   **********************************************************/
                  if ( mu == nu ) {
                    if ( tt == 0 ) {
                       ztmp -= ( contact_term[0][mu][0] + I * contact_term[0][mu][1] );
                     }
                  }

                  /**********************************************************
                   * multiply source phase from pion momentum
                   **********************************************************/
                  ztmp *= q_ephase;

                  /**********************************************************
                   * write into pgg
                   **********************************************************/
                  pgg[iconf][isrc][isink_momentum][mu][nu][2*tt  ] = creal( ztmp );
                  pgg[iconf][isrc][isink_momentum][mu][nu][2*tt+1] = cimag( ztmp );
                }  /* end of loop on timeslices */
      
              }  /* end of loop on direction nu */
              }  /* end of loop on direction mu */
      
              fini_4level_dtable( &buffer );
      
            }  /* end of loop on sink momenta */

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

          for ( int iconf = 0; iconf < num_conf; iconf++ )
          {
            for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
            {
              for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
                for ( int mu = 0; mu < 4; mu++ ) {
                for ( int nu = 0; nu < 4; nu++ ) {
                  for ( int it = 0; it < T; it++ ) {
                    fprintf ( stdout, "c %6d s %3d p %3d %3d %3d m %d %d pgg %3d  %25.16e %25.16e\n", iconf, isrc, 
                        g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], mu, nu, it, 
                        pgg[iconf][isrc][imom][mu][nu][2*it], pgg[iconf][isrc][imom][mu][nu][2*it+1] );
                  }
                }}
              }
            }
          }
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "p2gg_analyse", "show-all-data", g_cart_id == 0 );
        }

        /****************************************
         * check WI in momentum space
         ****************************************/
        if ( check_momentum_space_WI ) {
          gettimeofday ( &ta, (struct timezone *)NULL );

         for ( int iconf = 0; iconf < num_conf; iconf++ ) {
           for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
             for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

                exitstatus = check_momentum_space_wi_tpvec ( pgg[iconf][isrc][imom], g_sink_momentum_list[imom] );
                if ( exitstatus != 0  ) {
                  fprintf ( stderr, "[p2gg_analyse] Error from check_momentum_space_wi_tpvec, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(2);
                }

              }  /* end of loop on momenta */

            }  /* end of loop on sources per config */
          }  /* end of loop on configs */

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "p2gg_analyse", "check-wi-in-momentum-space", g_cart_id == 0 );
        }  /* end of if check_momentum_space_WI */

#if 0
#endif  /* if if 0 */

        /****************************************
         * statistical analysis for orbit average
         *
         * ASSUMES MOMENTUM LIST IS AN ORBIT AND
         * SEQUENTIAL MOMENTUM IS ZERO
         ****************************************/
        for ( int ireim = 0; ireim < 2; ireim++ ) {

          double *** data = init_3level_dtable ( num_conf, num_src_per_conf, T_global );
 
          int const dim[3] = { num_conf, num_src_per_conf, T_global };
          antisymmetric_orbit_average_spatial ( data, pgg, dim, g_sink_momentum_number, g_sink_momentum_list, ireim );

          char obs_name[100];
          sprintf ( obs_name, "pgg_conn.%s.j_j_orbit.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.%s", pgg_operator_type_tag[operator_type],
              seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id, sequential_source_timeslice,
                g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

          /* apply UWerr analysis */
          exitstatus = apply_uwerr_real ( data[0][0], num_conf*num_src_per_conf, T_global, 0, 1, obs_name );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[p2gg_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }

          fini_3level_dtable ( &data );

        }  /* end of loop on real / imag */

        /****************************************
         * STATISTICAL ANALYSIS for real and
         * imaginary part of individual
         * individual components
         ****************************************/
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
      
          int const momentum[3] = {
              g_sink_momentum_list[imom][0],
              g_sink_momentum_list[imom][1],
              g_sink_momentum_list[imom][2] };
      
          for( int mu = 0; mu < 4; mu++) {
          for( int nu = 0; nu < 4; nu++) {
            for ( int ireim = 0; ireim < 2; ireim++ ) {
      
              double *** data = init_3level_dtable ( num_conf, num_src_per_conf, T_global );

#pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                  for ( int it = 0; it < T_global; it++ ) {
                    data[iconf][isrc][it] = pgg[iconf][isrc][imom][mu][nu][2*it+ireim];
                  }
                }
              }

              char obs_name[100];
              sprintf ( obs_name, "pgg_conn.%s.jmu%d_jnu%d.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.%s", pgg_operator_type_tag[operator_type],
                  mu, nu, seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2],
                  sequential_source_gamma_id, sequential_source_timeslice,
                  momentum[0], momentum[1], momentum[2], reim_str[ireim] );

              /* apply UWerr analysis */
              exitstatus = apply_uwerr_real ( data[0][0], num_conf*num_src_per_conf, T_global, 0, 1, obs_name );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[p2gg_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(1);
              }

              fini_3level_dtable ( &data );

            }  /* end of loop on real / imag */
          }}  /* end of loop on nu, mu */
        }  /* end of loop on momenta */

        /**********************************************************
         * free p2gg table
         **********************************************************/
        fini_6level_dtable ( &pgg );

      }  /* end of loop on sequential source timeslices */

    }  /* end of loop on sequential source gamma id */
#if 0
#endif  /* of if 0  */
  }  /* end of loop on seq source momentum */


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
    fprintf(stdout, "# [p2gg_analyse] %s# [p2gg_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_analyse] %s# [p2gg_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
