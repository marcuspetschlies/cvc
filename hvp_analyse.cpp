/****************************************************
 * hvp_analyse
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

#define _ANTISYMMETRIC_ORBIT_AVERAGE_SPATIAL 0

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

  char const correlator_prefix[5][20] = { "hvp"        , "local-local", "hvp"        , "local-cvc" , "local-local" };

  char const flavor_tag[5][20]        = { "u-cvc-u-cvc", "u-gf-u-gi"  , "u-cvc-u-lvc", "u-gf-u-cvc", "d-gf-u-gi"   };

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
  int write_data = 0;

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
      fprintf ( stdout, "# [hvp_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [hvp_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'W':
      check_momentum_space_WI = 1;
      fprintf ( stdout, "# [hvp_analyse] check_momentum_space_WI set to %d\n", check_momentum_space_WI );
      break;
    case 'F':
      fold_correlator = atoi ( optarg );
      fprintf ( stdout, "# [hvp_analyse] fold_correlator set to %d\n", fold_correlator );
      break;
    case 'O':
      operator_type = atoi ( optarg );
      fprintf ( stdout, "# [hvp_analyse] operator_type set to %d\n", operator_type );
      break;
    case 'w':
      write_data = atoi ( optarg );
      fprintf ( stdout, "# [hvp_analyse] write_data set to %d\n", write_data );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [hvp_analyse] ensemble_name set to %s\n", ensemble_name );
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
  /* fprintf(stdout, "# [hvp_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [hvp_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [hvp_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [hvp_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[hvp_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[hvp_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[hvp_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [hvp_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[hvp_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[hvp_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [hvp_analyse] comment %s\n", line );
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
   ** HVP
   **
   ***********************************************************
   ***********************************************************/
  double ****** hvp = init_6level_dtable ( num_conf, num_src_per_conf, g_sink_momentum_number+1, 4, 4, 2 * T );
  if ( hvp == NULL ) {
    fprintf(stderr, "[hvp_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
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

#ifdef HAVE_LHPC_AFF
      /***********************************************
       * reader for aff input file
       ***********************************************/

      gettimeofday ( &ta, (struct timezone *)NULL );
      struct AffNode_s *affn = NULL, *affdir = NULL;

      sprintf ( filename, "stream_%c/%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", conf_src_list[iconf][isrc][0],
          g_outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
      /* sprintf ( filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", g_outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] ); */
      /* sprintf ( filename, "%s.%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff",
          correlator_prefix[operator_type], flavor_tag[operator_type], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] ); */

      if ( g_verbose > 0 ) fprintf(stdout, "# [hvp_analyse] reading data from file %s %s %d\n", filename, __FILE__, __LINE__);
      affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[hvp_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }

      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[hvp_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        return(103);
      }

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "hvp_analyse", "open-init-aff-reader", g_cart_id == 0 );
#endif


      /**********************************************************
       * loop on momenta
       **********************************************************/
      for ( int isink_momentum = 0; isink_momentum <= g_sink_momentum_number; isink_momentum++ ) {

        int sink_momentum[3] = {0,0,0}; 

        if ( isink_momentum < g_sink_momentum_number ) { 
          memcpy ( sink_momentum, g_sink_momentum_list[isink_momentum], 3*sizeof(int));
        }
        if ( g_verbose > 4 ) fprintf( stdout, "# [hvp_analyse] sink_momentum [%3d] = %3d, %3d ,%3d\n",
            isink_momentum, sink_momentum[0], sink_momentum[1], sink_momentum[2]  );
          
        double *** buffer = init_3level_dtable( 4, 4, 2 * T );
        if( buffer == NULL ) {
          fprintf(stderr, "[hvp_analyse] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(15);
        }

        if ( operator_type == 0 || operator_type == 2 ) {

          gettimeofday ( &ta, (struct timezone *)NULL );

          sprintf ( key , "/%s/%s/t%.2dx%.2dy%.2dz%.2d/px%.2dpy%.2dpz%.2d", correlator_prefix[operator_type], flavor_tag[operator_type],
              gsx[0], gsx[1], gsx[2], gsx[3],
              sink_momentum[0], sink_momentum[1], sink_momentum[2] );

          if ( g_verbose > 2 ) fprintf ( stdout, "# [hvp_analyse] key = %s\n", key );

          affdir = aff_reader_chpath (affr, affn, key );
          uint32_t uitems = 16 * T;
          exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer[0][0]), uitems );
          if( exitstatus != 0 ) {
            fprintf(stderr, "[hvp_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(105);
          }

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "hvp_analyse", "read-aff-key", g_cart_id == 0 );

        } else if ( operator_type == 1 || operator_type == 4) {

          gettimeofday ( &ta, (struct timezone *)NULL );

          for( int mu = 0; mu < 4; mu++) {
          for( int nu = 0; nu < 4; nu++) {
            sprintf ( key , "/%s/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d/gi%.2d/px%dpy%dpz%d", correlator_prefix[operator_type], flavor_tag[operator_type],
                gsx[0], gsx[1], gsx[2], gsx[3], g_sink_gamma_id_list[mu], g_source_gamma_id_list[nu], sink_momentum[0], sink_momentum[1], sink_momentum[2] );

            if ( g_verbose > 2 ) fprintf ( stdout, "# [hvp_analyse] key = %s\n", key );
            affdir = aff_reader_chpath (affr, affn, key );
            uint32_t uitems = T;
            exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer[mu][nu]), uitems );
            if( exitstatus != 0 ) {
              fprintf(stderr, "[hvp_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(105);
            }
          }}

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "hvp_analyse", "read-ll-tensor-aff", g_cart_id == 0 );
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

          double phase = - ( p[0] * gsx[0] + p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] );
          if ( operator_type == 0 ) {
            phase = - ( p[0] * gsx[0] + p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] ) + 0.5 * ( p[mu] - p[nu] );
          } else if ( operator_type == 2 ) {
            phase = - ( p[0] * gsx[0] + p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] ) + 0.5 * ( p[mu] );
          }

          double _Complex ephase = cexp ( phase * I );

          /**********************************************************
           * sort data from buffer into hvp,
           * add source phase
           **********************************************************/
#pragma omp parallel for
          for ( int it = 0; it < T_global; it++ ) {
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
        for ( int imom = 0; imom < g_sink_momentum_number+1; imom++ ) {
          int sink_momentum[3] = {0,0,0};
          if ( imom < g_sink_momentum_number ) memcpy ( sink_momentum, g_sink_momentum_list[imom], 3*sizeof(int));
          for ( int mu = 0; mu < 4; mu++ ) {
          for ( int nu = 0; nu < 4; nu++ ) {
            for ( int it = 0; it < T; it++ ) {
              fprintf ( stdout, "c %c %6d s %3d %3d %3d %3d p %3d %3d %3d m %d %d hvp %3d  %25.16e %25.16e\n", 
                  conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1],
                  conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5],
                  sink_momentum[0], sink_momentum[1], sink_momentum[2], mu, nu, it, 
                  hvp[iconf][isrc][imom][mu][nu][2*it], hvp[iconf][isrc][imom][mu][nu][2*it+1] );
            }
          }}
        }
      }
    }
    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "hvp_analyse", "show-all-data", g_cart_id == 0 );
  }

  /****************************************
   * check WI in momentum space
   ****************************************/
  if ( check_momentum_space_WI ) {
    gettimeofday ( &ta, (struct timezone *)NULL );

    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        for ( int imom = 0; imom < g_sink_momentum_number+1; imom++ ) {

          int sink_momentum[3] = {0, 0, 0};
          if ( imom < g_sink_momentum_number ) memcpy ( sink_momentum, g_sink_momentum_list[imom], 3*sizeof(int) );
  
          exitstatus = check_momentum_space_wi_tpvec ( hvp[iconf][isrc][imom], sink_momentum );
          if ( exitstatus != 0  ) {
            fprintf ( stderr, "[hvp_analyse] Error from check_momentum_space_wi_tpvec, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(2);
          }

        }  /* end of loop on momenta */
  
      }  /* end of loop on sources per config */
    }  /* end of loop on configs */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "hvp_analyse", "check-wi-in-momentum-space", g_cart_id == 0 );
  }  /* end of if check_momentum_space_WI */ 


  /****************************************
   * combine source locations 
   ****************************************/

  double ***** hvp_src_avg = init_5level_dtable ( num_conf, g_sink_momentum_number+1, 4, 4, 2 * T );
  if ( hvp_src_avg == NULL ) {
    fprintf(stderr, "[hvp_analyse] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

#pragma omp parallel for
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    double const norm = 1. / (double)num_src_per_conf;
    for ( int i = 0; i <= g_sink_momentum_number * 32 * T; i++ ) {
      hvp_src_avg[iconf][0][0][0][i] = 0.;
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        hvp_src_avg[iconf][0][0][0][i] += hvp[iconf][isrc][0][0][0][i];
      }
      hvp_src_avg[iconf][0][0][0][i] *= norm;
    }
  }

  /****************************************
   * STATISTICAL ANALYSIS of real and
   * imaginary part of HVP tensor
   * components
   ****************************************/
  for ( int imom = 0; imom < g_sink_momentum_number+1; imom++ ) {

    int momentum[3] = {0,0,0};
    
    if ( imom < g_sink_momentum_number ) memcpy( momentum, g_sink_momentum_list[imom], 3*sizeof(int));

    /* spatial components */
    for( int mu = 1; mu < 4; mu++) {
    for( int nu = 1; nu < 4; nu++) {
      for ( int ireim = 0; ireim < 2; ireim++ ) {

        double ** data = init_2level_dtable ( num_conf, T_global );
        if ( data == NULL ) {
          fprintf(stderr, "[hvp_analyse] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
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
          fprintf ( stderr, "[hvp_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        fini_2level_dtable ( &data );

      }  /* end of loop on re / im */
    }}  /* end of loop on nu, mu */
  }  /* end of loop on momenta */
#if 0
#endif  /* of if 0 */

  /****************************************
   * STATISTICAL ANALYSIS of real and
   * imaginary part of anti-symmetricly
   * orbit-averaged HVP spatial tensor
   * components
   ****************************************/
#if _ANTISYMMETRIC_ORBIT_AVERAGE_SPATIAL
  for ( int ireim = 0; ireim < 2; ireim++ ) {
    double ** data = init_2level_dtable ( num_conf, T_global );
    if ( data == NULL ) {
      fprintf(stderr, "[hvp_analyse] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
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
      fprintf ( stderr, "[hvp_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    }
    fini_2level_dtable ( &data );

  }  /* end of loop on re / im */
#endif  /* of _ANTISYMMETRIC_ORBIT_AVERAGE_SPATIAL  */

  /****************************************
   * STATISTICAL ANALYSIS of real and
   * imaginary part of symmetric
   * orbit average per * irrep 
   * of HVP spatial tensor
   ****************************************/

  double *** hvp_irrep = init_3level_dtable ( 5, num_conf, 2*T_global );

  char const irrep_name[5][4] = { "A1", "A1p", "T1", "T2", "E" };
    
  int dim[2] = { num_conf, T_global };
  hvp_irrep_decomposition_orbit_average ( hvp_irrep, hvp_src_avg, dim, g_sink_momentum_number, g_sink_momentum_list );

  for ( int i = 0; i < 5; i++ ) {
 
    for ( int ireim = 0; ireim < 2; ireim++ ) {
    
      double ** data = init_2level_dtable ( num_conf, T_global );
      if ( data == NULL ) {
        fprintf(stderr, "[hvp_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int it = 0; it < T_global; it++ ) {
          data[iconf][it] = hvp_irrep[i][iconf][2*it+ireim];
        }
      }

      char obs_name[100];
      sprintf ( obs_name, "%s.%s.%s.orbit.PX%d_PY%d_PZ%d.%s", correlator_prefix[operator_type], flavor_tag[operator_type],
          irrep_name[i], g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[hvp_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      if ( write_data == 1) {
        sprintf ( filename, "%s.corr", obs_name );
        FILE * ofs = fopen ( filename, "w" );
        if ( ofs == NULL ) {
          fprintf ( stdout, "[hvp_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT(12);
        }
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            fprintf ( ofs, "%5d %25.16e %8d %c\n", it, data[iconf][it], conf_src_list[iconf][0][1], conf_src_list[iconf][0][0] );
          }
        }
        fclose ( ofs );
      }  /* end of if write data  */
    
    }  /* end of loop on re / im */

  }   /* end of loop on irreps */

  /**********************************************************
   * dep. on 4-momentum
   **********************************************************/
  double **** hvp_scalar = init_4level_dtable ( num_conf, num_src_per_conf, g_sink_momentum_number, 2*T_global );
<<<<<<< HEAD
  double ***** hvp_irrep           = init_5level_dtable ( 5, num_conf, num_src_per_conf, g_sink_momentum_number, 2*T_global );
  double ***** hvp_irrep_projected = init_5level_dtable ( 5, num_conf, num_src_per_conf, g_sink_momentum_number, 2*T_global );
=======

  double ***** hvp_pirrep           = init_5level_dtable ( 5, num_conf, num_src_per_conf, g_sink_momentum_number, 2*T_global );
  double ***** hvp_pirrep_projected = init_5level_dtable ( 5, num_conf, num_src_per_conf, g_sink_momentum_number, 2*T_global );
>>>>>>> c5c9b150a6e94d9fce7907641ab7c314b166b9bc

#pragma omp parallel for
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        
      /**********************************************************
       * FT zero 3-momentum case, to subtract tensor_mu,nu at 
       * zero 4-momentum
       **********************************************************/
      double *** const hvp_zero = hvp[iconf][isrc][g_sink_momentum_number];

      hvp_ft ( hvp_zero, operator_type );

      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

        hvp_ft ( hvp[iconf][isrc][imom], operator_type );

        for ( int it = 0; it < T_global; it++ ) {
          int const itt = it < T_global/2 ? it : it - T_global;
          double const sinp[4] = {
              // 2 * sin( M_PI * it/(double)T_global ),
              2 * sin( M_PI * itt/(double)T_global ),
              2 * sin( M_PI * g_sink_momentum_list[imom][0]/(double)LX_global ),
              2 * sin( M_PI * g_sink_momentum_list[imom][1]/(double)LY_global ),
              2 * sin( M_PI * g_sink_momentum_list[imom][2]/(double)LZ_global ) };

          double const sinpp = sinp[0] * sinp[0] + sinp[1] * sinp[1] + sinp[2] * sinp[2] + sinp[3] * sinp[3];
          
          double const sinp2 = sinp[1] * sinp[1] + sinp[2] * sinp[2] + sinp[3] * sinp[3];

          double const sinp2_ti_one_over_three = sinp2 / 3.;
 
          if ( check_momentum_space_WI ) {

            for ( int nu = 0; nu<4; nu++ ) {
              double const dre = 
                   sinp[0] * hvp[iconf][isrc][imom][0][nu][2*it]
                +  sinp[1] * hvp[iconf][isrc][imom][1][nu][2*it]
                +  sinp[2] * hvp[iconf][isrc][imom][2][nu][2*it]
                +  sinp[3] * hvp[iconf][isrc][imom][3][nu][2*it];
              
              double const  dim = 
                   sinp[0] * hvp[iconf][isrc][imom][0][nu][2*it+1]
                +  sinp[1] * hvp[iconf][isrc][imom][1][nu][2*it+1]
                +  sinp[2] * hvp[iconf][isrc][imom][2][nu][2*it+1]
                +  sinp[3] * hvp[iconf][isrc][imom][3][nu][2*it+1];

              fprintf( stdout, "hvp wi %c %6d %3d    %3d %3d %3d    %3d   %d   %16.7e  %16.7e\n", 
                  conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1], conf_src_list[iconf][isrc][2],
                  g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
                  it, nu, dre, dim );
            }
          }

          /**********************************************************
           * subtract tensor
           **********************************************************/
          for(int mu = 0; mu<4; mu++ ) {
            for(int nu = 0; nu<4; nu++ ) {
              hvp[iconf][isrc][imom][mu][nu][2*it  ] -= hvp_zero[mu][nu][0];
              hvp[iconf][isrc][imom][mu][nu][2*it+1] -= hvp_zero[mu][nu][1];
            }
          }

          /**********************************************************
           * scalar hvp by trace of transverse projected and subtracted
           * tensor
           **********************************************************/
          for ( int ireim = 0; ireim < 2; ireim++ ) {
            hvp_scalar[iconf][isrc][imom][2*it+ireim] = 0;     

            for(int mu = 0; mu<4; mu++ ) {
              for(int nu = 0; nu<4; nu++ ) {
                hvp_scalar[iconf][isrc][imom][2*it+ireim] += sinp[mu] * sinp[nu] * hvp[iconf][isrc][imom][mu][nu][2*it+ireim];
              }
              hvp_scalar[iconf][isrc][imom][2*it+ireim] -= sinpp * hvp[iconf][isrc][imom][mu][mu][2*it+ireim];
            }
            hvp_scalar[iconf][isrc][imom][2*it+ireim] /= 3. * sinpp * sinpp;
          }

          /**********************************************************
           * unprojected irreps
           **********************************************************/
          for ( int ireim = 0; ireim < 2; ireim++ ) {

             /* A1 */
             hvp_pirrep[0][iconf][isrc][imom][2*it+ireim] = hvp[iconf][isrc][imom][1][1][2*it+ireim] + hvp[iconf][isrc][imom][2][2][2*it+ireim] + hvp[iconf][isrc][imom][3][3][2*it+ireim];

             /* A1p */
             hvp_pirrep[1][iconf][isrc][imom][2*it+ireim] = hvp[iconf][isrc][imom][0][0][2*it+ireim];

             /* T1 */
             hvp_pirrep[2][iconf][isrc][imom][2*it+ireim] = 
                  + sinp[1] * ( hvp[iconf][isrc][imom][0][1][2*it+ireim] + hvp[iconf][isrc][imom][1][0][2*it+ireim] )
                  + sinp[2] * ( hvp[iconf][isrc][imom][0][2][2*it+ireim] + hvp[iconf][isrc][imom][2][0][2*it+ireim] )
                  + sinp[3] * ( hvp[iconf][isrc][imom][0][3][2*it+ireim] + hvp[iconf][isrc][imom][3][0][2*it+ireim] );

             /* T2 */
             hvp_pirrep[3][iconf][isrc][imom][2*it+ireim] = 
                  + sinp[1] * sinp[2] * ( hvp[iconf][isrc][imom][1][2][2*it+ireim] + hvp[iconf][isrc][imom][2][1][2*it+ireim] )
                  + sinp[1] * sinp[3] * ( hvp[iconf][isrc][imom][1][3][2*it+ireim] + hvp[iconf][isrc][imom][3][1][2*it+ireim] )
                  + sinp[2] * sinp[3] * ( hvp[iconf][isrc][imom][2][3][2*it+ireim] + hvp[iconf][isrc][imom][3][2][2*it+ireim] );

             /* E */
             hvp_pirrep[4][iconf][isrc][imom][2*it+ireim] = 
                  ( sinp2_ti_one_over_three - sinp[1] * sinp[1] ) * hvp[iconf][isrc][imom][1][1][2*it+ireim]
                + ( sinp2_ti_one_over_three - sinp[2] * sinp[2] ) * hvp[iconf][isrc][imom][2][2][2*it+ireim]
                + ( sinp2_ti_one_over_three - sinp[3] * sinp[3] ) * hvp[iconf][isrc][imom][3][3][2*it+ireim];


             hvp_pirrep[0][iconf][isrc][imom][2*it+ireim] /= ( sinp2 - 3 * sinpp );

             hvp_pirrep[1][iconf][isrc][imom][2*it+ireim] /= -sinp2;

             hvp_pirrep[2][iconf][isrc][imom][2*it+ireim] /= 2. * sinp2 * sinp[0];

             hvp_pirrep[3][iconf][isrc][imom][2*it+ireim] /= 2. * ( sinp[1] * sinp[1] * sinp[2] * sinp[2] + sinp[1] * sinp[1] * sinp[3] * sinp[3] + sinp[2] * sinp[2] * sinp[3] * sinp[3] );

             hvp_pirrep[4][iconf][isrc][imom][2*it+ireim] /= sinp2 * sinp2 / 3. - 
                 (   sinp[1] * sinp[1] * sinp[1] * sinp[1] 
                   + sinp[2] * sinp[2] * sinp[2] * sinp[2] 
                   + sinp[3] * sinp[3] * sinp[3] * sinp[3]  );
        }

        /**********************************************************
         * project tensor
         **********************************************************/
        for ( int ireim = 0; ireim < 2; ireim++ ) {
        
          double haux[4][4];
          for ( int imu = 0; imu < 4; imu++ ) {
          for ( int inu = 0; inu < 4; inu++ ) {
            haux[imu][inu] = hvp[iconf][isrc][imom][imu][inu][2*it+ireim];
          }}
 
          for ( int imu = 0; imu < 4; imu++ ) {
          for ( int inu = 0; inu < 4; inu++ ) {

            hvp[iconf][isrc][imom][imu][inu][2*it+ireim] = haux[imu][inu];

            for ( int i = 0; i < 4; i++ ) {
             
              hvp[iconf][isrc][imom][imu][inu][2*it+ireim] -=  sinp[i]  * ( sinp[imu]  * haux[i][inu] + sinp[inu] * haux[imu][i] ) / sinpp;

              for ( int k = 0; k < 4; k++ ) {
                hvp[iconf][isrc][imom][imu][inu][2*it+ireim] += sinp[imu] * sinp[inu] * sinp[i] * sinp[k] * haux[i][k] / ( sinpp * sinpp );
              }}
            }}
           }

          /**********************************************************
           * projected irreps
           **********************************************************/
          for ( int ireim = 0; ireim < 2; ireim++ ) {

             /* A1 */
             hvp_pirrep_projected[0][iconf][isrc][imom][2*it+ireim] = hvp[iconf][isrc][imom][1][1][2*it+ireim] + hvp[iconf][isrc][imom][2][2][2*it+ireim] + hvp[iconf][isrc][imom][3][3][2*it+ireim];

             /* A1p */
             hvp_pirrep_projected[1][iconf][isrc][imom][2*it+ireim] = hvp[iconf][isrc][imom][0][0][2*it+ireim];

             /* T1 */
             hvp_pirrep_projected[2][iconf][isrc][imom][2*it+ireim] = 
                  + sinp[1] * ( hvp[iconf][isrc][imom][0][1][2*it+ireim] + hvp[iconf][isrc][imom][1][0][2*it+ireim] )
                  + sinp[2] * ( hvp[iconf][isrc][imom][0][2][2*it+ireim] + hvp[iconf][isrc][imom][2][0][2*it+ireim] )
                  + sinp[3] * ( hvp[iconf][isrc][imom][0][3][2*it+ireim] + hvp[iconf][isrc][imom][3][0][2*it+ireim] );

             /* T2 */
             hvp_pirrep_projected[3][iconf][isrc][imom][2*it+ireim] = 
                  + sinp[1] * sinp[2] * ( hvp[iconf][isrc][imom][1][2][2*it+ireim] + hvp[iconf][isrc][imom][2][1][2*it+ireim] )
                  + sinp[1] * sinp[3] * ( hvp[iconf][isrc][imom][1][3][2*it+ireim] + hvp[iconf][isrc][imom][3][1][2*it+ireim] )
                  + sinp[2] * sinp[3] * ( hvp[iconf][isrc][imom][2][3][2*it+ireim] + hvp[iconf][isrc][imom][3][2][2*it+ireim] );

             /* E */
             hvp_pirrep_projected[4][iconf][isrc][imom][2*it+ireim] = 
                  ( sinp2_ti_one_over_three - sinp[1] * sinp[1] ) * hvp[iconf][isrc][imom][1][1][2*it+ireim]
                + ( sinp2_ti_one_over_three - sinp[2] * sinp[2] ) * hvp[iconf][isrc][imom][2][2][2*it+ireim]
                + ( sinp2_ti_one_over_three - sinp[3] * sinp[3] ) * hvp[iconf][isrc][imom][3][3][2*it+ireim];


             hvp_pirrep_projected[0][iconf][isrc][imom][2*it+ireim] /= ( sinp2 - 3 * sinpp );

             hvp_pirrep_projected[1][iconf][isrc][imom][2*it+ireim] /= -sinp2;

             hvp_pirrep_projected[2][iconf][isrc][imom][2*it+ireim] /= 2. * sinp2 * sinp[0];

             hvp_pirrep_projected[3][iconf][isrc][imom][2*it+ireim] /= 2. * ( sinp[1] * sinp[1] * sinp[2] * sinp[2] + sinp[1] * sinp[1] * sinp[3] * sinp[3] + sinp[2] * sinp[2] * sinp[3] * sinp[3] );

             hvp_pirrep_projected[4][iconf][isrc][imom][2*it+ireim] /= sinp2 * sinp2 / 3. - 
                 (   sinp[1] * sinp[1] * sinp[1] * sinp[1] 
                   + sinp[2] * sinp[2] * sinp[2] * sinp[2] 
                   + sinp[3] * sinp[3] * sinp[3] * sinp[3]  );
        }

        }  /* emd loop on timeslices */
      }  /* end of loop on momenta */   
    }  /* end of loop on sources */
  }  /* end of loop on configs */

  if ( g_verbose > 5 ) {
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
          for ( int itt = -T_global/2; itt < T_global/2; itt++ ) {
            int const it = itt > 0 ? itt : ( itt + T_global ) % T_global;
            fprintf( stdout, "hvp %c %6d     %3d %3d %3d %3d     %3d %3d %3d    %3d %25.16e %25.16e\n",
                conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1],
                conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5],
                g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
                itt, hvp_scalar[iconf][isrc][imom][2*it], hvp_scalar[iconf][isrc][imom][2*it+1] );
          }
        }
      }
    }
  }

  /**********************************************************
   *
   * STATISTICAL ANALYSIS
   * for hvp_scalar
   *
   **********************************************************/
  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

    for ( int ireim = 0; ireim < 2; ireim++ ) {

      double ** data = init_2level_dtable ( num_conf, T_global );
      if ( data == NULL ) {
        fprintf(stderr, "[hvp_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

      int dim[2] = { num_conf, T_global };
#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int it = 0; it < T_global; it++ ) {
          data[iconf][it] = 0.;
          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            data[iconf][it] += hvp_scalar[iconf][isrc][imom][2*it+ireim];
          }
          data[iconf][it] /= (double)num_src_per_conf;
      }} 

      char obs_name[100];
      sprintf ( obs_name, "%s.%s.scalar.PX%d_PY%d_PZ%d.%s", correlator_prefix[operator_type], flavor_tag[operator_type],
            g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], reim_str[ireim] );

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[hvp_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      if ( write_data == 1) {
        sprintf ( filename, "%s.corr", obs_name );
        FILE * ofs = fopen ( filename, "w" );
        if ( ofs == NULL ) {
          fprintf ( stdout, "[hvp_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT(12);
        }
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            fprintf ( ofs, "%5d%25.16e%8d %c\n", it, data[iconf][it], conf_src_list[iconf][0][1], conf_src_list[iconf][0][0] );
          }
        }
        fclose ( ofs );
      }  /* end of if write data  */

      fini_2level_dtable ( &data );

    }  /* end of loop on re / im */

  }  /* end of loop on momenta */

  fini_4level_dtable ( &hvp_scalar );
  fini_5level_dtable ( &hvp_src_avg );

  /**********************************************************
   *
   * STATISTICAL ANALYSIS
   * for unprojected irreps
   *
   **********************************************************/
  for ( int irrep = 0; irrep < 5; irrep++ ) {

    for ( int ireim = 0; ireim < 2; ireim++ ) {

      double ** data = init_2level_dtable ( num_conf, T_global );
      if ( data == NULL ) {
        fprintf(stderr, "[hvp_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

      int dim[2] = { num_conf, T_global };
#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int it = 0; it < T_global; it++ ) {
          data[iconf][it] = 0.;
          for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              data[iconf][it] += hvp_pirrep[irrep][iconf][isrc][imom][2*it+ireim];
            }
          }
          data[iconf][it] /= (double)num_src_per_conf * (double)g_sink_momentum_number;
      }} 

      char obs_name[100];
      sprintf ( obs_name, "%s.%s.%s.PX%d_PY%d_PZ%d.%s", correlator_prefix[operator_type], flavor_tag[operator_type], irrep_name[irrep],
            g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[hvp_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      if ( write_data == 1) {
        sprintf ( filename, "%s.corr", obs_name );
        FILE * ofs = fopen ( filename, "w" );
        if ( ofs == NULL ) {
          fprintf ( stdout, "[hvp_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT(12);
        }
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            fprintf ( ofs, "%5d%25.16e%8d %c\n", it, data[iconf][it], conf_src_list[iconf][0][1], conf_src_list[iconf][0][0] );
          }
        }
        fclose ( ofs );
      }  /* end of if write data  */

      fini_2level_dtable ( &data );

    }  /* end of loop on re / im */

  }  /* end of loop on irreps */

  fini_5level_dtable ( &hvp_pirrep );

  /**********************************************************
   *
   * STATISTICAL ANALYSIS
   * for projected irreps
   *
   **********************************************************/
  for ( int irrep = 0; irrep < 5; irrep++ ) {

    for ( int ireim = 0; ireim < 2; ireim++ ) {

      double ** data = init_2level_dtable ( num_conf, T_global );
      if ( data == NULL ) {
        fprintf(stderr, "[hvp_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

      int dim[2] = { num_conf, T_global };
#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int it = 0; it < T_global; it++ ) {
          data[iconf][it] = 0.;
          for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              data[iconf][it] += hvp_pirrep_projected[irrep][iconf][isrc][imom][2*it+ireim];
            }
          }
          data[iconf][it] /= (double)num_src_per_conf * (double)g_sink_momentum_number;
      }} 

      char obs_name[100];
      sprintf ( obs_name, "%s.%s.%s.projected.PX%d_PY%d_PZ%d.%s", correlator_prefix[operator_type], flavor_tag[operator_type], irrep_name[irrep],
            g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[hvp_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      if ( write_data == 1) {
        sprintf ( filename, "%s.corr", obs_name );
        FILE * ofs = fopen ( filename, "w" );
        if ( ofs == NULL ) {
          fprintf ( stdout, "[hvp_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT(12);
        }
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            fprintf ( ofs, "%5d%25.16e%8d %c\n", it, data[iconf][it], conf_src_list[iconf][0][1], conf_src_list[iconf][0][0] );
          }
        }
        fclose ( ofs );
      }  /* end of if write data  */

      fini_2level_dtable ( &data );

    }  /* end of loop on re / im */

  }  /* end of loop on irreps */

  fini_5level_dtable ( &hvp_pirrep_projected );

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
    fprintf(stdout, "# [hvp_analyse] %s# [hvp_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [hvp_analyse] %s# [hvp_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);
}
