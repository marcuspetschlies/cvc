/****************************************************
 * p2gg_analyse_wdisc.c
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
#include "gamma.h"
#include "contract_loop_inline.h"


using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse P-J-J correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  EXIT(0);
}

/****************************************************/
/****************************************************/

/****************************************************
 * MAIN PROGRAM
 ****************************************************/

int main(int argc, char **argv) {
  
  double const TWO_MPI = 2. * M_PI;

  char const reim_str[2][3] = {"re" , "im"};

  /* char const operator_type_tag[4][8]  = { "cvc-cvc"    , "lvc-lvc"    , "cvc-lvc"    , "lvc-cvc"    }; */

  char const correlator_prefix[4][20] = { "hvp"        , "local-local", "hvp"        , "local-cvc"  };

  char const flavor_tag[4][20]        = { "u-cvc-u-cvc", "u-gf-u-gi"  , "u-cvc-u-lvc", "u-gf-u-cvc" };

  char const loop_type_tag[3][8] = { "NA", "dOp", "Scalar" };

  int const gamma_tmlqcd_to_binary[16] = { 8, 1, 2, 4, 0,  15, 7, 14, 13, 11, 9, 10, 12, 3, 5, 6 };

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
  int loop_type = 0;
  int loop_stats = 1;
  int write_data = 0;

  double ****** pgg_disc = NULL;
  double ****** hvp = NULL;

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  char key[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "Wh?f:N:S:F:O:D:w:E:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_wdisc] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_wdisc] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'W':
      check_momentum_space_WI = 1;
      fprintf ( stdout, "# [p2gg_analyse_wdisc] check_momentum_space_WI set to %d\n", check_momentum_space_WI );
      break;
    case 'F':
      fold_correlator = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_wdisc] fold_correlator set to %d\n", fold_correlator );
      break;
    case 'O':
      operator_type = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_wdisc] operator_type set to %d\n", operator_type );
      break;
    case 'D':
      loop_type = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_wdisc] loop_type set to %d\n", loop_type );
      break;
    case 'w':
      write_data = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_wdisc] write_data set to %d\n", write_data );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [p2gg_analyse_wdisc] ensemble_name set to %d\n", ensemble_name );
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
  /* fprintf(stdout, "# [p2gg_analyse_wdisc] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [p2gg_analyse_wdisc] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_analyse_wdisc] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_analyse_wdisc] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_analyse_wdisc] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_analyse_wdisc] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[p2gg_analyse_wdisc] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [p2gg_analyse_wdisc] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf);
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[p2gg_analyse_wdisc] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 5 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[p2gg_analyse_wdisc] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [p2gg_analyse_wdisc] comment %s\n", line );
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
   * how to normalize loops
   ***********************************************************/
  double loop_norm = 0.;
  if (   strcmp ( loop_type_tag[loop_type], "dOp"     ) == 0  
      || strcmp ( loop_type_tag[loop_type], "LpsDw"   ) == 0
      || strcmp ( loop_type_tag[loop_type], "LpsDwCv" ) == 0 ) {
    loop_norm = -4. * g_kappa;
  } else if ( strcmp ( loop_type_tag[loop_type], "Scalar"  ) == 0 
           || strcmp ( loop_type_tag[loop_type], "Loops"   ) == 0
           || strcmp ( loop_type_tag[loop_type], "LoopsCv" ) == 0 ) {
    loop_norm = -8. * g_mu * g_kappa * g_kappa;
  }
  if ( g_verbose > 0 ) fprintf ( stdout, "# [p2gg_analyse_wdisc] oet_type %s loop_norm = %25.16e\n", loop_type_tag[loop_type], loop_norm );

  /***********************************************************
   ***********************************************************
   **
   ** HVP
   **
   ***********************************************************
   ***********************************************************/
  hvp = init_6level_dtable ( num_conf, num_src_per_conf, g_sink_momentum_number, 4, 4, 2 * T );
  if ( hvp == NULL ) {
    fprintf(stderr, "[p2gg_analyse_wdisc] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__);
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

      /* sprintf ( filename, "%d/%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", Nconf, g_outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] ); */
      sprintf ( filename, "%d/%s.%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", Nconf, correlator_prefix[operator_type], flavor_tag[operator_type], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );

      if ( g_verbose > 0 ) fprintf(stdout, "# [p2gg_analyse_wdisc] reading data from file %s %s %d\n", filename, __FILE__, __LINE__);
      affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg_analyse_wdisc] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }

      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[p2gg_analyse_wdisc] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        return(103);
      }

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "p2gg_analyse_wdisc", "open-init-aff-reader", g_cart_id == 0 );
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
          fprintf(stderr, "[p2gg_analyse_wdisc] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(15);
        }

        if ( operator_type == 0 || operator_type == 2 ) {

          gettimeofday ( &ta, (struct timezone *)NULL );

          sprintf ( key , "/%s/%s/t%.2dx%.2dy%.2dz%.2d/px%.2dpy%.2dpz%.2d", correlator_prefix[operator_type], flavor_tag[operator_type],
              gsx[0], gsx[1], gsx[2], gsx[3],
              sink_momentum[0], sink_momentum[1], sink_momentum[2] );

          if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse_wdisc] key = %s\n", key );

          affdir = aff_reader_chpath (affr, affn, key );
          uint32_t uitems = 16 * T;
          exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer[0][0]), uitems );
          if( exitstatus != 0 ) {
            fprintf(stderr, "[p2gg_analyse_wdisc] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(105);
          }

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "p2gg_analyse_wdisc", "read-aff-key", g_cart_id == 0 );

        } else if ( operator_type == 1 ) {

          gettimeofday ( &ta, (struct timezone *)NULL );

          for( int mu = 0; mu < 4; mu++) {
          for( int nu = 0; nu < 4; nu++) {
            sprintf ( key , "/%s/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d/gi%.2d/px%dpy%dpz%d", correlator_prefix[operator_type], flavor_tag[operator_type],
                gsx[0], gsx[1], gsx[2], gsx[3], mu, nu, sink_momentum[0], sink_momentum[1], sink_momentum[2] );

            if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse_wdisc] key = %s\n", key );
            affdir = aff_reader_chpath (affr, affn, key );
            uint32_t uitems = T;
            exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer[mu][nu]), uitems );
            if( exitstatus != 0 ) {
              fprintf(stderr, "[p2gg_analyse_wdisc] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }
          }}

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "p2gg_analyse_wdisc", "read-ll-tensor-aff", g_cart_id == 0 );
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
    show_time ( &ta, &tb, "p2gg_analyse_wdisc", "show-all-data", g_cart_id == 0 );
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
            fprintf ( stderr, "[p2gg_analyse_wdisc] Error from check_momentum_space_wi_tpvec, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(2);
          }

        }  /* end of loop on momenta */
  
      }  /* end of loop on sources per config */
    }  /* end of loop on configs */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "p2gg_analyse_wdisc", "check-wi-in-momentum-space", g_cart_id == 0 );
  }  /* end of if check_momentum_space_WI */ 


  /****************************************
   * combine source locations 
   ****************************************/

  double ***** hvp_src_avg = init_5level_dtable ( num_conf, g_sink_momentum_number, 4, 4, 2 * T );
  if ( hvp_src_avg == NULL ) {
    fprintf(stderr, "[p2gg_analyse_wdisc] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

#pragma omp parallel for
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    double const norm = 1. / (double)num_src_per_conf;
    for ( int i = 0; i < g_sink_momentum_number * 32 * T; i++ ) {
      hvp_src_avg[iconf][0][0][0][i] = 0.;
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        hvp_src_avg[iconf][0][0][0][i] += hvp[iconf][isrc][0][0][0][i];
      }
      hvp_src_avg[iconf][0][0][0][i] *= norm;
    }
  }

#if 0
  /****************************************
   * STATISTICAL ANALYSIS of real and
   * imaginary part of HVP tensor
   * components
   ****************************************/
  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

    int const momentum[3] = { g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] };

    /* spatial components */
    for( int mu = 1; mu < 4; mu++) {
    for( int nu = 1; nu < 4; nu++) {
      for ( int ireim = 0; ireim < 2; ireim++ ) {

        double ** data = init_2level_dtable ( num_conf, T_global );
        if ( data == NULL ) {
          fprintf(stderr, "[p2gg_analyse_wdisc] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(16);
        }

        /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = hvp_src_avg[iconf][imom][mu][nu][2*it+ireim];
          }
        }

        char obs_name[100];
        sprintf ( obs_name, "%s.%s.jmu%d_jnu%d.PX%d_PY%d_PZ%d.%s", correlator_prefix[operator_type], flavor_tag[operator_type],
            mu, nu, momentum[0], momentum[1], momentum[2], reim_str[ireim] );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[p2gg_analyse_wdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        fini_2level_dtable ( &data );

      }  /* end of loop on re / im */
    }}  /* end of loop on nu, mu */
  }  /* end of loop on momenta */
#endif  /* of if 0 */

  /****************************************
   * STATISTICAL ANALYSIS of real and
   * imaginary part of anti-symmetricly
   * orbit-averaged HVP spatial tensor
   * components
   ****************************************/
  for ( int ireim = 0; ireim < 2; ireim++ ) {
    double ** data = init_2level_dtable ( num_conf, T_global );
    if ( data == NULL ) {
      fprintf(stderr, "[p2gg_analyse_wdisc] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(16);
    }

    int dim[2] = { num_conf, T_global };
    antisymmetric_orbit_average_spatial ( data, hvp_src_avg, dim, g_sink_momentum_number, g_sink_momentum_list, ireim );

    char obs_name[100];
    sprintf ( obs_name, "%s.%s.eps.orbit.PX%d_PY%d_PZ%d.%s", correlator_prefix[operator_type], flavor_tag[operator_type],
        g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

    /* apply UWerr analysis */
    exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[p2gg_analyse_wdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    }
    fini_2level_dtable ( &data );

  }  /* end of loop on re / im */

  /****************************************
   * STATISTICAL ANALYSIS of real and
   * imaginary part of symmetric
   * orbit average per * irrep 
   * of HVP spatial tensor
   ****************************************/
  for ( int ireim = 0; ireim < 2; ireim++ ) {
    char const irrep_name[5][4] = { "A1", "A1p", "T1", "T2", "E" };

    double *** data = init_3level_dtable ( 5, num_conf, T_global );
    if ( data == NULL ) {
      fprintf(stderr, "[p2gg_analyse_wdisc] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(16);
    }

    int dim[2] = { num_conf, T_global };
    hvp_irrep_separation_orbit_average ( data, hvp_src_avg, dim, g_sink_momentum_number, g_sink_momentum_list, ireim );

    for ( int i = 0; i < 5; i++ ) {
      char obs_name[100];
      sprintf ( obs_name, "%s.%s.%s.orbit.PX%d_PY%d_PZ%d.%s", correlator_prefix[operator_type], flavor_tag[operator_type],
          irrep_name[i], g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_real ( data[i][0], num_conf, T_global, 0, 1, obs_name );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[p2gg_analyse_wdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }
    }

    if ( write_data == 1) {
      char obs_name[100];
      for ( int i = 0; i < 5; i++ ) {
        sprintf ( obs_name, "%s.%s.%s.orbit.PX%d_PY%d_PZ%d.%s.dat", correlator_prefix[operator_type], flavor_tag[operator_type],
            irrep_name[i], g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );
        FILE * ofs = fopen ( obs_name, "w" );
        if ( ofs == NULL ) {
          fprintf ( stdout, "[p2gg_analyse_wdisc] Error from fopen for file %s %s %d\n", obs_name, __FILE__, __LINE__ );
          EXIT(12);
        }
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            fprintf ( ofs, "%5d%25.16e%8d\n", it, data[i][iconf][it], conf_src_list[iconf][0][0] );
          }
        }
      }  /* end of loop on irreps */
    }  /* end of if write data  */

    fini_3level_dtable ( &data );

  }  /* end of loop on re / im */

  fini_5level_dtable ( &hvp_src_avg );



  /**********************************************************
   **********************************************************
   **
   ** P -> gg disconnected P-loop 3-point function
   **
   **********************************************************
   **********************************************************/

  for ( int iseq_source_momentum = 0; iseq_source_momentum < g_seq_source_momentum_number; iseq_source_momentum++) 
  {

    int const seq_source_momentum[3] = {
      g_seq_source_momentum_list[iseq_source_momentum][0],
      g_seq_source_momentum_list[iseq_source_momentum][1],
      g_seq_source_momentum_list[iseq_source_momentum][2] };


    /**********************************************************
     * read loop matrix data
     **********************************************************/
    double ***** loops_matrix = init_5level_dtable ( num_conf, g_nsample, T, 4, 8 );
    if ( loops_matrix == NULL ) {
      fprintf ( stderr, "[p2gg_analyse_wdisc] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(112);
    }


    /**********************************************************
     * oet type string
     **********************************************************/
    switch (loop_type) {
      case  1:
      case -1:
      case  2:
      case -2:
        fprintf ( stdout, "# [p2gg_analyse_wdisc] Using loop type %s\n", loop_type_tag[loop_type]);
        break;
      default:
        fprintf ( stderr, "[p2gg_analyse_wdisc] Error, loop type not specified\n");
        EXIT(12);
    }

    /**********************************************************
     * use real or imaginary part of loo
     *
     * loop_type > 0 ---> real part
     * loop_type < 0 ---> imag part
     **********************************************************/
    int const loop_type_reim = loop_type > 0 ? 0 : 1;

    /**********************************************************
     * read the loop data from ASCII file
     **********************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );

    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      Nconf = conf_src_list[iconf][0][0];
      sprintf ( filename, "loops/Nconf_%d.%s.px%dpy%dpz%d", Nconf, loop_type_tag[loop_type], seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2] );
      /* sprintf ( filename, "loops/Nconf_%.4d.%s.loop_gamma4", Nconf, loop_type_tag[loop_type] ); */
      if ( g_verbose > 0 ) fprintf ( stdout, "# [p2gg_analyse_wdisc] reading loop data from file %s %s %d\n", filename, __FILE__, __LINE__ );
      FILE * ofs = fopen ( filename, "r" );
      if ( ofs == NULL ) {
        fprintf ( stderr, "[p2gg_analyse_wdisc] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
        EXIT(113);
      }

      for ( int isample = 0; isample < g_nsample; isample++ ) {
        for ( int t = 0; t < T; t++ ) {
          for ( int mu = 0; mu < 4; mu++ ) {
          for ( int nu = 0; nu < 4; nu++ ) {
            if ( fscanf ( ofs, "%lf %lf\n", loops_matrix[iconf][isample][t][mu]+2*nu, loops_matrix[iconf][isample][t][mu]+2*nu+1 ) != 2 ) {
              fprintf ( stderr, "[p2gg_analyse_wdisc] Error from fscanf %s %d\n", __FILE__, __LINE__ );
              EXIT(126);
            }
            /* show all data */
            if ( g_verbose > 4 ) {
              fprintf ( stdout, "loop_mat c %6d s %3d t %3d m %d nu %d l %25.16e %25.16e\n", Nconf, isample, t, mu, nu,
                  loops_matrix[iconf][isample][t][mu][2*nu], loops_matrix[iconf][isample][t][mu][2*nu+1] );
            }
          }}
        }
      }
      fclose ( ofs );
    }  /* end of loop on configurations */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "p2gg_analyse_wdisc", "read-loop-data", g_cart_id == 0 );

    /**********************************************************
     * loop on sequential source gamma matrices
     **********************************************************/
    for( int isequential_source_gamma_id = 0; isequential_source_gamma_id < g_sequential_source_gamma_id_number; isequential_source_gamma_id++)
    {

      int const sequential_source_gamma_id = g_sequential_source_gamma_id_list[ isequential_source_gamma_id ];

      gamma_matrix_type gf;
      gamma_matrix_ukqcd_binary ( &gf, gamma_tmlqcd_to_binary[sequential_source_gamma_id] );
      if ( g_verbose > 2 ) gamma_matrix_printf ( &gf, "gseq_ukqcd", stdout );

      double *** loops_proj = init_3level_dtable ( num_conf, g_nsample, 2*T );
      if ( loops_proj == NULL ) {
        fprintf ( stderr, "[p2gg_analyse_wdisc] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(112);
      }


      project_loop ( loops_proj[0][0], gf.m, loops_matrix[0][0][0][0], num_conf * g_nsample * T );

      double ** loop_avg = init_2level_dtable ( num_conf, 2*T );
      if ( loop_avg == NULL ) {
        fprintf ( stderr, "[p2gg_analyse_wdisc] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(115);
      }

      /**********************************************************
       * average loop over samples for each config and timeslice
       **********************************************************/
#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        double const norm = loop_norm / (double)g_nsample;
        for ( int t = 0; t < T; t++ ) {
          loop_avg[iconf][2*t  ] = 0.; 
          loop_avg[iconf][2*t+1] = 0.; 

          for ( int isample = 0; isample<g_nsample; isample++ ) {
            loop_avg[iconf][2*t  ] += loops_proj[iconf][isample][2*t  ];
            loop_avg[iconf][2*t+1] += loops_proj[iconf][isample][2*t+1];
          }

          /* normalize */
          loop_avg[iconf][2*t  ] *= norm;
          loop_avg[iconf][2*t+1] *= norm;

        }
      }

      /**********************************************************
       * show averaged loops
       **********************************************************/
      if( g_verbose > 3 ) {
        sprintf ( filename, "loop.g%d.px%dpy%dpz%d.avg", sequential_source_gamma_id,
            seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2] );

        FILE *lfs = fopen ( filename, "w" );
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int t = 0; t < T_global; t++ ) {
            fprintf ( lfs, "%25.16e %25.16e\n", loop_avg[iconf][2*t], loop_avg[iconf][2*t+1] ); 
          }
        }
      }

      /**********************************************************
       *
       * statistical analysis for loops
       *
       **********************************************************/
      if ( loop_stats ) {

        dquant cumulants[4] = { cumulant_1, cumulant_2, cumulant_3, cumulant_4 };
        dquant dcumulants[4] = { dcumulant_1, dcumulant_2, dcumulant_3, dcumulant_4 };

        double ** data = init_2level_dtable ( num_conf, 4 * T_global );

      /**********************************************************
       * STATISTICAL ANALYSIS for loops 
       * simple loops
       **********************************************************/
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
           for ( int it = 0; it < T_global; it++ ) {
            double const ltmp = loop_avg[iconf][2*it+loop_type_reim];
            double dtmp = 1.;
            for ( int icum = 0; icum < 4; icum++ ) {
              dtmp *= ltmp;
              data[iconf][ icum * T_global + it ] = dtmp;
            }
          }
        }

        /* loop on cumulants */
        for ( int icum = 0; icum < 4; icum++ ) {

          char obs_name[100];
          sprintf ( obs_name, "loop.%s.QX%d_QY%d_QZ%d.g%d.k%d", loop_type_tag[loop_type],
              seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id, icum+1 );

          /* apply UWerr analysis */
          int * arg_first  = init_1level_itable ( icum + 1 );
          int * arg_stride = init_1level_itable ( icum + 1 );
          for ( int k = 0; k <= icum ; k++ ) {
            arg_first[k]  = k * T_global;
            arg_stride[k] = 1;
          }

          exitstatus = apply_uwerr_func ( data[0], num_conf, 4*T_global, T_global, icum+1, arg_first, arg_stride, obs_name, cumulants[icum], dcumulants[icum] );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[p2gg_analyse_wdisc] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(115);
          }

          fini_1level_itable ( &arg_first );
          fini_1level_itable ( &arg_stride );

        }  /* end of loop on cumulants */

        fini_2level_dtable ( &data );
#if 0
      /**********************************************************
       * STATISTICAL ANALYSIS for loops 
       * with fwd difference
       **********************************************************/
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = loop_avg[iconf][2*((it+1)%T_global)+loop_type_reim] - loop_avg[iconf][2*it+loop_type_reim];
          }
        }

        sprintf ( obs_name, "loop_avg.dfwd.%s.QX%d_QY%d_QZ%d.g%d", loop_type_tag[loop_type],
            seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[p2gg_analyse_wdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

      /**********************************************************
       * STATISTICAL ANALYSIS for loops 
       * with symmetric 3-point 2nd order difference
       **********************************************************/
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = 
                    loop_avg[iconf][2 * ( (it+1+T_global)%T_global) + loop_type_reim] 
              +     loop_avg[iconf][2 * ( (it-1+T_global)%T_global) + loop_type_reim] 
              - 2 * loop_avg[iconf][2 *    it                       + loop_type_reim];
          }
        }

        sprintf ( obs_name, "loop_avg.ddsym.%s.QX%d_QY%d_QZ%d.g%d", loop_type_tag[loop_type],
            seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[p2gg_analyse_wdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

      /**********************************************************
       * STATISTICAL ANALYSIS for loops
       * 3-point average
       **********************************************************/
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = (
                loop_avg[iconf][2 * ( (it+1+T_global)%T_global) + loop_type_reim] 
              + loop_avg[iconf][2 * ( (it-1+T_global)%T_global) + loop_type_reim] 
              + loop_avg[iconf][2 *    it                       + loop_type_reim] ) / 3.;
          }
        }

        sprintf ( obs_name, "loop_avg.s3sym.%s.QX%d_QY%d_QZ%d.g%d", loop_type_tag[loop_type],
            seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[p2gg_analyse_wdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }
        fini_2level_dtable ( &data );
#endif  /* of if 0 */

      }  /* end of if loop_stats */


      /**********************************************************
       *
       * loop data for pgg
       *
       **********************************************************/
      double ** loop_pgg = init_2level_dtable ( num_conf, 2*T );
      if ( loop_pgg == NULL ) {
        fprintf ( stderr, "[p2gg_analyse_wdisc] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(115);
      }

#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int it = 0; it < T_global; it++ ) {
          for ( int ireim = 0; ireim < 2; ireim++ ) {
            /**********************************************************
             * loop from symmetric 3-point 2nd order difference
             **********************************************************/
            /*
            loop_pgg[iconf][2*it + ireim ] =
                      loop_avg[iconf][2 * ( (it+1+T_global)%T_global) + ireim ]
                +     loop_avg[iconf][2 * ( (it-1+T_global)%T_global) + ireim ]
                - 2 * loop_avg[iconf][2 *    it                       + ireim ];
             */

            /**********************************************************
             * loop from 3-point average
             **********************************************************/
            /*
            loop_pgg[iconf][2*it + ireim ] =
              (
                  loop_avg[iconf][2 * ( (it+1+T_global)%T_global) + ireim ]
                + loop_avg[iconf][2 * ( (it-1+T_global)%T_global) + ireim ]
                + loop_avg[iconf][2 *    it                       + ireim ] ) / 3.;
             */

            /**********************************************************
             * loop
             **********************************************************/
            loop_pgg[iconf][2*it + ireim ] = loop_avg[iconf][2 * it + ireim ];

          }
        }
      }

      fini_2level_dtable ( &loop_avg );

      /**********************************************************
       * loop on sequential source timeslices
       **********************************************************/
      for ( int isequential_source_timeslice = 0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++)
      {

        int const sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];

        /***********************************************************
         * prepare the disconnected contribution loop x hvp
         ***********************************************************/
 
        pgg_disc = init_6level_dtable ( num_conf, num_src_per_conf, g_sink_momentum_number, 4, 4, 2 * T );
        if ( pgg_disc == NULL ) {
          fprintf(stderr, "[p2gg_analyse_wdisc] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(16);
        }

        for ( int iconf = 0; iconf < num_conf; iconf++ ) {

          for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
 
            /* calculate the sequential source timeslice; tseq = source timeslice + source-sink-timeseparation */
            int const tseq = ( conf_src_list[iconf][isrc][1] - sequential_source_timeslice + T_global ) % T_global;
            if ( g_verbose > 3 ) fprintf ( stdout, "# [p2gg_analyse_wdisc] tsnk (v) %3d dt %3d tsrc (loop) %3d\n",
                conf_src_list[iconf][isrc][1] , sequential_source_timeslice, tseq );

            for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
              for ( int s1 = 0; s1 < 4; s1++ ) {
              for ( int s2 = 0; s2 < 4; s2++ ) {
                for ( int it = 0; it < T; it++ ) {

                  pgg_disc[iconf][isrc][imom][s1][s2][2*it  ] = hvp[iconf][isrc][imom][s1][s2][2*it  ] * loop_pgg[iconf][2*tseq+loop_type_reim];
                  pgg_disc[iconf][isrc][imom][s1][s2][2*it+1] = hvp[iconf][isrc][imom][s1][s2][2*it+1] * loop_pgg[iconf][2*tseq+loop_type_reim];

                }
              }}
            }
          }  /* end of loop on sources per conf */

        }  /* end of loop on configurations */
        

        /****************************************
         * check WI in momentum space
         ****************************************/
        if ( check_momentum_space_WI ) {
          gettimeofday ( &ta, (struct timezone *)NULL );

         for ( int iconf = 0; iconf < num_conf; iconf++ ) {
           for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
             for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

                exitstatus = check_momentum_space_wi_tpvec ( pgg_disc[iconf][isrc][imom], g_sink_momentum_list[imom] );
                if ( exitstatus != 0  ) {
                  fprintf ( stderr, "[p2gg_analyse_wdisc] Error from check_momentum_space_wi_tpvec, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(2);
                }

              }  /* end of loop on momenta */

            }  /* end of loop on sources per config */
          }  /* end of loop on configs */

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "p2gg_analyse_wdisc", "check-wi-in-momentum-space", g_cart_id == 0 );
        }  /* end of if check_momentum_space_WI */

        /****************************************
         * pgg_disc source average
         ****************************************/
        double ***** pgg = init_5level_dtable ( num_conf, g_sink_momentum_number, 4, 4, 2 * T_global );
        if ( pgg == NULL ) {
          fprintf(stderr, "[p2gg_analyse_wdisc] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(16);
        }
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int i = 0; i < g_sink_momentum_number * 32 * T_global; i++ ) {
            pgg[iconf][0][0][0][i] = 0.;

            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              pgg[iconf][0][0][0][i] += pgg_disc[iconf][isrc][0][0][0][i];
            }
            pgg[iconf][0][0][0][i] /= (double)num_src_per_conf;
          }
        }

#if 0
        /****************************************
         * STATISTICAL ANALYSIS for real and
         * imaginary part
         ****************************************/
      
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
      
          int const momentum[3] = {
              g_sink_momentum_list[imom][0],
              g_sink_momentum_list[imom][1],
              g_sink_momentum_list[imom][2] };
      
          for( int mu = 1; mu < 4; mu++)
          {
      
          for( int nu = 1; nu < 4; nu++)
          {
            for ( int ireim = 0; ireim < 2; ireim++ ) {
      
              double ** data = init_2level_dtable ( num_conf, T_global );
              if ( data == NULL ) {
                fprintf ( stderr, "[p2gg_analyse_wdisc] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
                EXIT(78);
              }

#pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for ( int it = 0; it < T_global; it++ ) {
                  data[iconf][it] += pgg[iconf][imom][mu][nu][2*it+ireim];
                }
              }
       
              char obs_name[100];
              sprintf ( obs_name, "pgg_disc.%s.%s.%s.jmu%d_jnu%d.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.%s", correlator_prefix[operator_type], flavor_tag[operator_type],
                  loop_type_tag[loop_type],
                  mu, nu, seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2],
                  sequential_source_gamma_id, sequential_source_timeslice,
                  momentum[0], momentum[1], momentum[2], reim_str[ireim] );

              /* apply UWerr analysis */
              exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[p2gg_analyse_wdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(1);
              }

              fini_2level_dtable ( &data );

            }  /* end of loop on real / imag */
          }}  /* end of loop on nu, mu */
        }  /* end of loop on momenta */
#endif  /* of if 0 */

#if 0
#ifdef _USE_SUBTRACTED
        /****************************************
         * STATISTICAL ANALYSIS for subtracted
         * correlation function
         *
         * <C3pt> - <C2pt> x <Loop>
         ****************************************/
      
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
      
          int const momentum[3] = {
              g_sink_momentum_list[imom][0],
              g_sink_momentum_list[imom][1],
              g_sink_momentum_list[imom][2] };
      
          for( int mu = 1; mu < 4; mu++)
          {
      
          for( int nu = 1; nu < 4; nu++)
          {
            for ( int ireim = 0; ireim < 2; ireim++ ) {
      
              double ** data = init_2level_dtable ( num_conf, 2 * T_global + 1);
      
              /****************************************
               * fill the data array
               ****************************************/
#pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for ( int it = 0; it < T_global; it++ ) {
                  /* real part of pgg_disc */
                  data[iconf][           it] += pgg_disc[iconf][imom][mu][nu][2*it+ireim];
                  /* real part of hvp */
                  data[iconf][T_global + it] += hvp[iconf][imom][mu][nu][2*it+ireim];
                }

                /* time-averaged loop */
                for ( int it = 0; it < T_global; it++ ) {
                  data[iconf][2*T_global] += loop_pgg[iconf][2*it+loop_type_reim];
                }
                data[iconf][2*T_global] *= 1. / (double)T_global;

              }  /* end of loop on configurations */
      
              char obs_name[100];
              sprintf ( obs_name, "pgg_disc.%s.%s.%s.sub.jmu%d_jnu%d.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.%s", correlator_prefix[operator_type], flavor_tag[operator_type],
                  loop_type_tag[loop_type],
                  mu, nu, seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2],
                  sequential_source_gamma_id, sequential_source_timeslice, momentum[0], momentum[1], momentum[2], reim_str[ireim] );

              int arg_first[3] = { 0, T_global, 2*T_global };
              int arg_stride[3] = { 1, 1, 0};

              exitstatus = apply_uwerr_func ( data[0], num_conf, 2*T_global+1, T_global, 3, arg_first, arg_stride, obs_name, a_mi_b_ti_c, da_mi_b_ti_c);
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[p2gg_analyse_wdisc] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(115);
              }
      
              fini_2level_dtable ( &data );
      
            }  /* end of loop on real / imag */
          }}  /* end of loop on nu, mu */
        }  /* end of loop on momenta */

#endif  /* of ifdef _USE_SUBTRACTED */
#endif  /* of if 0 */

        /****************************************
         * statistical analysis for orbit average
         *
         * ASSUMES MOMENTUM LIST IS AN ORBIT AND
         * SEQUENTIAL MOMENTUM IS ZERO
         ****************************************/
        for ( int ireim = 0; ireim < 2; ireim++ ) {
          double ** data = init_2level_dtable ( num_conf, T_global );
          if ( data == NULL ) {
            fprintf ( stderr, "[p2gg_analyse_wdisc] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(79);
          }

          int const dim[2] = { num_conf, T_global };
          antisymmetric_orbit_average_spatial ( data, pgg, dim, g_sink_momentum_number, g_sink_momentum_list, ireim );
      

          char obs_name[100];
          sprintf ( obs_name, "pgg_disc.%s.%s.%s.orbit.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.%s", correlator_prefix[operator_type], flavor_tag[operator_type],
              loop_type_tag[loop_type],
              seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id, sequential_source_timeslice,
                g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

          /* apply UWerr analysis */
          exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[p2gg_analyse_wdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }

          if ( write_data == 1 ) {
            sprintf ( obs_name, "pgg_disc.%s.%s.%s.orbit.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.%s.dat", correlator_prefix[operator_type], flavor_tag[operator_type],
                loop_type_tag[loop_type],
                seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id, sequential_source_timeslice,
                  g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

            FILE * ofs = fopen ( obs_name, "w" );
            if ( ofs == NULL ) {
              fprintf ( stdout, "[p2gg_analyse_wdisc] Error from fopen for file %s %s %d\n", obs_name, __FILE__, __LINE__ );
              EXIT(12);
            }

            for ( int iconf = 0; iconf < num_conf; iconf++ ) {
              for ( int tau = -T_global/2+1; tau <= T_global/2; tau++ ) {
                int const it = ( tau < 0 ) ? tau + T_global : tau;

                fprintf ( ofs, "%5d%25.16e%8d\n", tau, data[iconf][it], conf_src_list[iconf][0][0] );
              }
            }
            fclose ( ofs );

          }  /* end of if write data */

          fini_2level_dtable ( &data );
        }  /* end of loop on real / imag */

#ifdef _USE_SUBTRACTED
        /****************************************
         * statistical analysis for orbit average
         *
         * WITH SUBTRACTION
         *
         * ASSUMES MOMENTUM LIST IS AN ORBIT AND
         * SEQUENTIAL MOMENTUM IS ZERO
         ****************************************/
        for ( int ireim = 0; ireim < 2; ireim++ ) {
          double *** data_aux = init_3level_dtable ( 2, num_conf, T_global );
          double ** data = init_2level_dtable ( num_conf, 2 * T_global + 1 );
  
          int const dim[2] = { num_conf, T_global };
          antisymmetric_orbit_average_spatial ( data_aux[0], pgg, dim, g_sink_momentum_number, g_sink_momentum_list, ireim );
          antisymmetric_orbit_average_spatial ( data_aux[1], hvp, dim, g_sink_momentum_number, g_sink_momentum_list, ireim );
     
#pragma omp parallel for         
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < T_global; it++ ) {

              data[iconf][         it] += data_aux[0][iconf][it];
              data[iconf][T_global+it] += data_aux[1][iconf][it];

              /* time-averaged loop */
              data[iconf][2*T_global] += loop_pgg[iconf][2*it+loop_type_reim];
  
            }  /* end of loop on timeslices */

            /* normalize the loop */
            data[iconf][2*T_global] *= 1. / (double)T_global;

          }  /* end of loop on configurations */

          char obs_name[100];
          sprintf ( obs_name, "pgg_disc.%s.%s.%s.sub.orbit.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.%s", correlator_prefix[operator_type], flavor_tag[operator_type],
              loop_type_tag[loop_type],
              seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id, sequential_source_timeslice,
              g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim]);
      
          int arg_first[3] = { 0, T_global, 2*T_global };
          int arg_stride[3] = { 1, 1, 0};

          exitstatus = apply_uwerr_func ( data[0], num_conf, 2*T_global+1, T_global, 3, arg_first, arg_stride, obs_name, a_mi_b_ti_c, da_mi_b_ti_c);
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[p2gg_analyse_wdisc] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(115);
          }

          if ( write_data == 1 ) {
            sprintf ( obs_name, "pgg_disc.%s.%s.%s.sub.orbit.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.%s", correlator_prefix[operator_type], flavor_tag[operator_type],
                loop_type_tag[loop_type],
                seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id, sequential_source_timeslice,
                  g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

            FILE * ofs = fopen ( obs_name, "w" );
            if ( ofs == NULL ) {
              fprintf ( stdout, "[p2gg_analyse_wdisc] Error from fopen for file %s %s %d\n", obs_name, __FILE__, __LINE__ );
              EXIT(12);
            }

            for ( int iconf = 0; iconf < num_conf; iconf++ ) {
              for ( int tau = -T_global/2+1; tau <= T_global/2; tau++ ) {
                int const it = ( tau < 0 ) ? tau + T_global : tau;
                fprintf ( ofs, "%5d%25.16e%25.16e%25.16e%8d\n", tau, data[iconf][it], data[iconf][T_global + it], data[iconf][2*T_global], conf_src_list[iconf][0][0] );
              }
            }
            fclose ( ofs );

          }  /* end of if write data */
          fini_3level_dtable ( &data_aux );
          fini_2level_dtable ( &data );

        }  /* end of loop on real / imag */

#endif  /* of ifdef _USE_SUBTRACTED */

        /**********************************************************
         * free p2gg table
         **********************************************************/
        fini_6level_dtable ( &pgg_disc );
        fini_5level_dtable ( &pgg );

      }  /* end of loop on sequential source timeslices */

      fini_2level_dtable ( &loop_pgg );
#if 0
#endif /* of if 0 */


      fini_3level_dtable ( &loops_proj ); 

    }  /* end of loop on sequential source gamma id */

    fini_5level_dtable ( &loops_matrix );

  }  /* end of loop on seq source momentum */
#if 0
#endif  /* of if 0 */

  /**********************************************************
   * free the allocated memory, finalize
   **********************************************************/

  fini_6level_dtable ( &hvp );

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
    fprintf(stdout, "# [p2gg_analyse_wdisc] %s# [p2gg_analyse_wdisc] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_analyse_wdisc] %s# [p2gg_analyse_wdisc] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
