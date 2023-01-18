/****************************************************
 * avxg_analyse_simple
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
#include "gamma.h"
#include "uwerr.h"
#include "derived_quantities.h"

#ifndef _SQR
#define _SQR(_a) ((_a)*(_a))
#endif

#define _TWOP_SCATT  0
#define _TWOP_CYD_H5 0
#define _TWOP_CYD    0
#define _TWOP_AFF    0
#define _TWOP_H5     0
#define _TWOP_AVGX_H5 1

#define _LOOP_ANALYSIS 1

#define _LOOP_CY       0
#define _LOOP_CVC      1

#define _RAT_METHOD       1
#define _FHT_METHOD_ALLT  0
#define _FHT_METHOD_ACCUM 0


using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse cpff fht correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default cpff.input]\n");
  EXIT(0);
}

/**********************************************************
 *
 **********************************************************/
int main(int argc, char **argv) {
  
  char const correlator_prefix[2][20] = { "local-local" , "charged"};

#if _TWOP_CYD_H5
  const char flavor_tag[2][3] = { "uu", "dd" };
#elif _TWOP_AVGX_H5
  /* char const flavor_tag[2][20]        = { "s-gf-l-gi" , "l-gf-s-gi" }; */
  char const flavor_tag[2][20]        = { "l-gf-l-gi" , "l-gf-l-gi" };
#else
  char const flavor_tag[2][20]        = { "d-gf-u-gi" , "u-gf-d-gi" };
#endif

  char const threep_tag[5][12] = { "g4_D4", "gi_Dk", "g4_Dk" , "44_nosub", "jj_nosub"};

  double const TWO_MPI = 2. * M_PI;

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[600];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "NA";
  /* int use_disc = 0;
  int use_conn = 1; */
  int twop_fold_propagator = 0;
  int write_data = 0;
  double loop_norm = 1.;
  int operator_type = 0;
  char loop_type[10] = "Clv";
  char fbwd_type[6]="NA";

  int stout_level_iter = 0;
  double stout_level_rho = 0.0;

  struct timeval ta, tb, start_time, end_time;

  double twop_weight[2]   = {0., 0.};
  double fbwd_weight[2]   = {0., 0.};
  double mirror_weight[2] = {0., 0.};

  /**********************************************************
   * cC211.06.80
   **********************************************************/
  double g_mus = 0.0006;
  /* double g_mus = 0.01615; */

  /**********************************************************
   * cD211.054.96
   **********************************************************/
  /* double g_mus = 0.00054; */
  /* double g_mus = 0.0136; */


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:F:E:w:m:l:O:T:B:M:s:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [avxg_analyse_simple] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [avxg_analyse_simple] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'F':
      twop_fold_propagator = atoi ( optarg );
      fprintf ( stdout, "# [avxg_analyse_simple] twop fold_propagator set to %d\n", twop_fold_propagator );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [avxg_analyse_simple] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'O':
      operator_type = atoi ( optarg );
      fprintf ( stdout, "# [avxg_analyse_simple] operator_type set to %d\n", operator_type );
      break;
    case 'w':
      write_data = atoi ( optarg );
      fprintf ( stdout, "# [avxg_analyse_simple] write_date set to %d\n", write_data );
      break;
    case 'm':
      loop_norm = atof ( optarg );
      fprintf ( stdout, "# [avxg_analyse_simple] loop_norm set to %e\n", loop_norm );
      break;
    case 'l':
      strcpy ( loop_type, optarg );
      fprintf ( stdout, "# [avxg_analyse_simple] loop_type set to %s\n", loop_type );
      break;
    case 'T':
      sscanf( optarg, "%lf,%lf", twop_weight, twop_weight+1 );
      fprintf ( stdout, "# [avxg_analyse_simple] twop_weight set to %25.16e / %25.16e\n", twop_weight[0], twop_weight[1] );
      break;
    case 'B':
      strcpy ( fbwd_type, optarg );
      /* sscanf( optarg, "%lf,%lf", fbwd_weight, fbwd_weight+1 ); */
      fprintf ( stdout, "# [avxg_analyse_simple] fbwd_type set to %s\n", fbwd_type );
      break;
    case 'M':
      sscanf( optarg, "%lf,%lf", mirror_weight, mirror_weight+1 );
      fprintf ( stdout, "# [avxg_analyse_simple] mirror_weight set to %25.16e / %25.16e\n", mirror_weight[0], mirror_weight[1] );
      break;
    case 's':
      sscanf ( optarg, "%d,%lf", &stout_level_iter, &stout_level_rho);
      fprintf ( stdout, "# [xg_analyse] stout_level iter %2d  rho %6.4f \n", stout_level_iter, stout_level_rho );
      break;

    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  gettimeofday ( &start_time, (struct timezone *)NULL );

  /* set the default values */
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [avxg_analyse_simple] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [avxg_analyse_simple] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [avxg_analyse_simple] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [avxg_analyse_simple] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[avxg_analyse_simple] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

#if 0
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[avxg_analyse_simple] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();
#endif

  T = T_global,
  LX_global = LX;
  LY_global = LY;
  LZ_global = LZ;

  fprintf ( stdout, "# [] T_globaal = %2d, T = %2d\n", T_global, T );
  fprintf ( stdout, "# [] LX_globaal = %2d, LX = %2d\n", LX_global, LX );
  fprintf ( stdout, "# [] LY_globaal = %2d, LY = %2d\n", LY_global, LY );
  fprintf ( stdout, "# [] LZ_globaal = %2d, LZ = %2d\n", LZ_global, LZ );


  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[avxg_analyse_simple] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [avxg_analyse_simple] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  /* sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf); */
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[avxg_analyse_simple] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[avxg_analyse_simple] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [avxg_analyse_simple] comment %s\n", line );
      continue;
    }
    int itmp[5];
    char ctmp;

    sscanf( line, "%c %d %d %d %d %d", &ctmp, itmp, itmp+1, itmp+2, itmp+3, itmp+4 );

    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][0] = (int)ctmp;
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][1] = itmp[0];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][2] = itmp[1];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][3] = itmp[2];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][4] = itmp[3];
    conf_src_list[count/num_src_per_conf][count%num_src_per_conf][5] = itmp[4];

    count++;
  }

  fclose ( ofs );


  if ( g_verbose > 3 ) {
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

  if ( strcmp( fbwd_type, "fwd" ) == 0 )
  {
    fbwd_weight[0] = 1.0;
    fbwd_weight[1] = 0.0;
  } else if ( strcmp( fbwd_type, "bwd" ) == 0 )
  {
    fbwd_weight[0] = 0.0;
    fbwd_weight[1] = 1.0;
  } else if ( strcmp( fbwd_type, "fbwd" ) == 0 )
  {
    fbwd_weight[0] = 1.0;
    fbwd_weight[1] = 1.0;
  }
  fprintf ( stdout, "# [] fbwd_weight set to %e   %e   %s %d\n", fbwd_weight[0], fbwd_weight[1], __FILE__, __LINE__ );


  /**********************************************************
   **********************************************************
   ** 
   ** READ DATA
   ** 
   **********************************************************
   **********************************************************/

  /***********************************************************
   * read twop function data
   ***********************************************************/
  double ****** twop = NULL;

  twop = init_6level_dtable ( g_sink_momentum_number, num_conf, num_src_per_conf, 2, T_global, 2 );
  if( twop == NULL ) {
    fprintf ( stderr, "[avxg_analyse_simple] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT (24);
  }

#if _TWOP_CYD
  for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      Nconf = conf_src_list[iconf][0][1];
     
      for( int imeson = 0; imeson < 2; imeson++ ) {

        sprintf( filename, "stream_%c/%s/twop.%.4d.pseudoscalar.%d.PX%d_PY%d_PZ%d",
            conf_src_list[iconf][0][0], filename_prefix, Nconf, imeson+1,
            ( 1 - 2 * imeson) * g_sink_momentum_list[isink_momentum][0],
            ( 1 - 2 * imeson) * g_sink_momentum_list[isink_momentum][1],
            ( 1 - 2 * imeson) * g_sink_momentum_list[isink_momentum][2] );

        FILE * dfs = fopen ( filename, "r" );
        if( dfs == NULL ) {
          fprintf ( stderr, "[avxg_analyse_simple] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT (24);
        } else {
          if ( g_verbose > 1 ) fprintf ( stdout, "# [avxg_analyse_simple] reading data from file %s filename \n", filename );
        }
        fflush ( stdout );
        fflush ( stderr );

        for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
          char line[400];

          for ( int it = -1; it < T_global; it++ ) {
            if ( fgets ( line, 100, dfs) == NULL ) {
              fprintf ( stderr, "[avxg_analyse_simple] Error from fgets, expecting line input for it %3d conf %3d src %3d filename %s %s %d\n", 
                  it, iconf, isrc, filename, __FILE__, __LINE__ );
              EXIT (26);
            } 
          
            if ( line[0] == '#' &&  it == -1 ) {
              if ( g_verbose > 1 ) fprintf ( stdout, "# [avxg_analyse_simple] reading key %s\n", line );
              continue;
            } /* else {
              fprintf ( stderr, "[avxg_analyse_simple] Error in layout of file %s %s %d\n", filename, __FILE__, __LINE__ );
              EXIT(27);
            }
              */
            sscanf ( line, "%lf %lf\n", twop[isink_momentum][iconf][isrc][imeson][it], twop[isink_momentum][iconf][isrc][imeson][it]+1 );
         
            if ( g_verbose > 5 ) fprintf ( stdout, "%d %25.16e %25.16e\n" , it, twop[isink_momentum][iconf][isrc][imeson][it][0],
                twop[isink_momentum][iconf][isrc][imeson][it][1] );
          }

        }
        fclose ( dfs );

        /**********************************************************
         * write source-averaged correlator to ascii file
         **********************************************************/
        if ( write_data == 1 && g_verbose > 4 ) {

          if ( imeson == 1 ) {
            for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
 
              fprintf ( stdout, "# twop %c %6d  src  %3d %3d %3d %3d  p %3d %3d %3d m %d\n", conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1],
                  conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5],
                  g_sink_momentum_list[isink_momentum][0], g_sink_momentum_list[isink_momentum][1], g_sink_momentum_list[isink_momentum][2],
                  imeson );

              for ( int it = 0; it < T_global; it++ ) {
                fprintf ( stdout, "%3d    %25.16e %25.16e    %25.16e %25.16e    %25.16e %25.16e\n",
                    it, 
                    twop[isink_momentum][iconf][isrc][0][it][0], twop[isink_momentum][iconf][isrc][0][it][1],
                    twop[isink_momentum][iconf][isrc][1][it][0], twop[isink_momentum][iconf][isrc][1][it][1],
                    ( twop[isink_momentum][iconf][isrc][0][it][0] + twop[isink_momentum][iconf][isrc][1][it][0] ) * 0.5,
                    ( twop[isink_momentum][iconf][isrc][0][it][1] + twop[isink_momentum][iconf][isrc][1][it][1] ) * 0.5 );
              }
            }
          }
        }  /* end of if write_data == 1 */

      }  /* end of loop on meson type */
    }  /* end of loop on configurations */
  }  /* end of loop on sink momenta */
#endif  /* end of _TWOP_CYD */

  /***********************************************************/
  /***********************************************************/

#if _TWOP_CYD_H5
  for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

    int ksink_momentum = -1;
    for ( int k = 0; k < g_sink_momentum_number; k++ ) {
      if ( 
             ( g_sink_momentum_list[isink_momentum][0] == -g_sink_momentum_list[k][0] )
          && ( g_sink_momentum_list[isink_momentum][1] == -g_sink_momentum_list[k][1] )
          && ( g_sink_momentum_list[isink_momentum][2] == -g_sink_momentum_list[k][2] )
         ) {
        ksink_momentum = k;
        break;
      }
    }
    if ( ksink_momentum == -1 ) {
      fprintf ( stderr, "[] Error, no negative momentum for %3d   %3d %3d %3d %s %d\n", isink_momentum,
          g_sink_momentum_list[isink_momentum][0], g_sink_momentum_list[isink_momentum][1], g_sink_momentum_list[isink_momentum][2], __FILE__, __LINE__ );
      EXIT(1);
    } else {
      if ( g_verbose > 2 ) {
        fprintf ( stdout, "# [avxg_analyse_simple] matching %3d   %3d %3d %3d and %3d   %3d %3d %3d %s %d\n", 
            isink_momentum, g_sink_momentum_list[isink_momentum][0], g_sink_momentum_list[isink_momentum][1], g_sink_momentum_list[isink_momentum][2],
            ksink_momentum, g_sink_momentum_list[ksink_momentum][0], g_sink_momentum_list[ksink_momentum][1], g_sink_momentum_list[ksink_momentum][2],
            __FILE__, __LINE__ );
      }
    }

    sprintf( filename, "%s/twop.pseudoscalar.PX%d_PY%d_PZ%d.h5", filename_prefix,
        g_sink_momentum_list[isink_momentum][0], g_sink_momentum_list[isink_momentum][1], g_sink_momentum_list[isink_momentum][2] ); 

    for ( int iconf = 0; iconf < num_conf; iconf++ ) {

      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
     
          char h5_tag[200];

          sprintf ( h5_tag, "/s%c/c%d/t%dx%dy%dz%d/%s",
              conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1],
              conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5],
              flavor_tag[ 0 ] );

          if ( g_verbose > 2 ) fprintf ( stdout, "# [avxg_analyse_simple] h5 tag = %s %s %d\n", h5_tag, __FILE__, __LINE__  );

          exitstatus = read_from_h5_file ( (void*)(twop[isink_momentum][iconf][isrc][0][0]), filename, h5_tag, "double", io_proc );
          if ( exitstatus != 0 ) {
            fprintf( stderr, "[avxg_analyse_simple] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(2);
          }

          sprintf ( h5_tag, "/s%c/c%d/t%dx%dy%dz%d/%s",
              conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1],
              conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5],
              flavor_tag[ 1 ] );

          if ( g_verbose > 2 ) fprintf ( stdout, "# [avxg_analyse_simple] h5 tag = %s %s %d\n", h5_tag, __FILE__, __LINE__  );

          exitstatus = read_from_h5_file ( (void*)(twop[ksink_momentum][iconf][isrc][1][0]), filename, h5_tag, "double", io_proc );
          if ( exitstatus != 0 ) {
            fprintf( stderr, "[avxg_analyse_simple] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(2);
          }


      }  /* end of loop on sources  */

    }  /* end of loop on configurations */

  }  /* end of loop on momenta */

#endif  /* end of _TWOP_CYD_H5 */

  /***********************************************************/
  /***********************************************************/

#if _TWOP_SCATT
  for( int iconf = 0; iconf < num_conf; iconf++ ) {

    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      sprintf ( filename, "stream_%c/%s%.4d_sx%.2dsy%.2dsz%.2dst%.3d_P.h5", conf_src_list[iconf][isrc][0], filename_prefix, conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5], conf_src_list[iconf][isrc][2] );

      char h5_tag[200];
      sprintf( h5_tag, "/sx%.2dsy%.2dsz%.2dst%.2d/mvec", conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5], conf_src_list[iconf][isrc][2] );

      int * ibuffer = NULL;
      size_t nmdim=0, * mdim = NULL;

      exitstatus = read_from_h5_file_varsize ( (void**)(&ibuffer), filename, h5_tag,  "int", &nmdim, &mdim,  io_proc );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "[avxg_analyse_simple] Error from read_from_h5_file_varsize for file %s tag %s , status was %d %s %d\n", filename, h5_tag, exitstatus, __FILE__, __LINE__ );
        EXIT(12);
      }

      int const momentum_number = (int)mdim[0];

      int ** momentum_list    = init_2level_itable ( mdim[0], mdim[1] );
      int  * sink_momentum_id = init_1level_itable ( g_sink_momentum_number );
      if ( momentum_list == NULL || sink_momentum_id == NULL ) {
        fprintf( stderr, "[avxg_analyse_simple] Error from init_Xlevel_itable %s %d\n", __FILE__, __LINE__ );
        EXIT(12);
      }

      memcpy ( momentum_list[0], ibuffer, mdim[0] * mdim[1] * sizeof(int) );

      if ( ibuffer != NULL ) free ( ibuffer );

      /* momentum matching with sink momentum list */
      for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

        int ksink_momentum = 0;
        for ( ; ksink_momentum < momentum_number; ksink_momentum++ ) {
          if (
              ( g_sink_momentum_list[isink_momentum][0] == momentum_list[ksink_momentum][0] ) &&
              ( g_sink_momentum_list[isink_momentum][1] == momentum_list[ksink_momentum][1] ) &&
              ( g_sink_momentum_list[isink_momentum][2] == momentum_list[ksink_momentum][2] ) ) break;
        }
        if ( ksink_momentum == momentum_number ) {
          fprintf( stderr, "[avxg_analyse_simple] Error, no matching for sink momentum %d    %3d %3d %3d %s %d\n", isink_momentum,
               g_sink_momentum_list[isink_momentum][0], g_sink_momentum_list[isink_momentum][1], g_sink_momentum_list[isink_momentum][2], __FILE__, __LINE__ );
          EXIT(12);
        }

        sink_momentum_id[isink_momentum] = ksink_momentum;
      }

      for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {
        fprintf( stdout, "# [] sink momentum matching %d (%3d %3d %3d)   %d  (%3d %3d %3d)\n",
            isink_momentum, g_sink_momentum_list[isink_momentum][0], g_sink_momentum_list[isink_momentum][1], g_sink_momentum_list[isink_momentum][2],
            sink_momentum_id[isink_momentum], 
            momentum_list[sink_momentum_id[isink_momentum]][0],
            momentum_list[sink_momentum_id[isink_momentum]][1],
            momentum_list[sink_momentum_id[isink_momentum]][2] );
      }

  
      double **** buffer = init_4level_dtable ( T_global, momentum_number, 1, 2 ); 
      if ( buffer == NULL ) {
        fprintf( stderr, "[avxg_analyse_simple] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(12);
      }

      sprintf( h5_tag, "/sx%.2dsy%.2dsz%.2dst%.2d/P", conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5], conf_src_list[iconf][isrc][2] );

      exitstatus = read_from_h5_file ( buffer[0][0][0], filename, h5_tag,  "double", io_proc );
      if ( exitstatus != 0 ) {
        fprintf( stderr, "[avxg_analyse_simple] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(12);
      }

      for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

#pragma omp parallel for
        for ( int it = 0; it < T_global; it++ ) {

          twop[isink_momentum][iconf][isrc][0][it][0] =  buffer[it][sink_momentum_id[isink_momentum]][0][0];
          twop[isink_momentum][iconf][isrc][0][it][1] =  buffer[it][sink_momentum_id[isink_momentum]][0][1];

          twop[isink_momentum][iconf][isrc][1][it][0] =  buffer[it][sink_momentum_id[isink_momentum]][0][0];
          twop[isink_momentum][iconf][isrc][1][it][1] = -buffer[it][sink_momentum_id[isink_momentum]][0][1];
        }

      }  /* end of loop on momenta */
      fini_4level_dtable ( &buffer );


      fini_2level_itable ( &momentum_list );
      fini_1level_itable ( &sink_momentum_id );
      free ( mdim );

    }  /* end of loop on sources  */

  }  /* end of loop on configurations */

#endif  /* end of _TWOP_SCATT */

  /***********************************************************/
  /***********************************************************/

#if _TWOP_AFF
    gettimeofday ( &ta, (struct timezone *)NULL );

    /***********************************************************
     * loop on flavor ids
     ***********************************************************/
    for ( int iflavor = 0; iflavor <= 1 ; iflavor++ ) {

      /***********************************************************
       * loop on sink momenta
       ***********************************************************/
      for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {
  
        /***********************************************************
         * open AFF reader
         ***********************************************************/
        struct AffReader_s *affr = NULL;
        struct AffNode_s *affn = NULL, *affdir = NULL;
        char key[400];
        char data_filename[500];
    
        /* sprintf( data_filename, "%s/stream_%c/%d/%s.%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", filename_prefix,
            conf_src_list[iconf][isrc][0], 
            conf_src_list[iconf][isrc][1], 
            correlator_prefix[operator_type], flavor_tag[iflavor],
            conf_src_list[iconf][isrc][1], 
            conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] ); */
 
        /* sprintf( data_filename, "%s/stream_%c/light/p2gg_twop_local/%s.%s.%d.gf%.2d.gi%.2d.aff",
            filename_prefix,
            conf_src_list[iconf][0][0], 
            correlator_prefix[operator_type], flavor_tag[iflavor],
            conf_src_list[iconf][0][1], g_sink_gamma_id_list[igf], g_source_gamma_id_list[igi] ); */

        sprintf( data_filename, "%s/%s.%s.gf%.2d.gi%.2d.px%dpy%dpz%d.aff",
            filename_prefix,
            correlator_prefix[operator_type], flavor_tag[iflavor],
            g_sink_gamma_id_list[0], g_source_gamma_id_list[0],
            ( 1 - 2 * iflavor ) * g_sink_momentum_list[ipf][0], 
            ( 1 - 2 * iflavor ) * g_sink_momentum_list[ipf][1],
            ( 1 - 2 * iflavor ) * g_sink_momentum_list[ipf][2] );

        affr = aff_reader ( data_filename );
        const char * aff_status_str = aff_reader_errstr ( affr );
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[avxg_analyse_simple] Error from aff_reader for filename %s, status was %s %s %d\n", data_filename, aff_status_str, __FILE__, __LINE__);
          EXIT(15);
        } else {
          if ( g_verbose > 1 ) fprintf(stdout, "# [avxg_analyse_simple] reading data from file %s\n", data_filename);
        }
  
        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[avxg_analyse_simple] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
          EXIT(103);
        }
  
        double ** buffer = init_2level_dtable ( T_global, 2 );
        if( buffer == NULL ) {
          fprintf(stderr, "[avxg_analyse_simple] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(15);
        }

        /***********************************************************
         * loop on configs
         ***********************************************************/
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {

          /***********************************************************
           * loop on sources
           ***********************************************************/
          for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

            sprintf( key, "/stream_%c/conf_%d/t%.2dx%.2dy%.2dz%.2d",
                conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1],
                conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4], conf_src_list[iconf][isrc][5] );

            if ( g_verbose > 2 ) fprintf ( stdout, "# [avxg_analyse_simple] key = %s\n", key );
  
            affdir = aff_reader_chpath (affr, affn, key );
            if( affdir == NULL ) {
              fprintf(stderr, "[avxg_analyse_simple] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
              EXIT(105);
            }
  
            uint32_t uitems = T_global;
            int texitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)buffer[0], uitems );
            if( texitstatus != 0 ) {
              fprintf(stderr, "[avxg_analyse_simple] Error from aff_node_get_complex, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
              EXIT(105);
            }
  
            /***********************************************************
             * source phase
             ***********************************************************/
            const double phase = -TWO_MPI * ( 1 - 2 * iflavor ) * (
                  conf_src_list[iconf][isrc][3] * g_sink_momentum_list[ipf][0] / (double)LX_global
                + conf_src_list[iconf][isrc][4] * g_sink_momentum_list[ipf][1] / (double)LY_global
                + conf_src_list[iconf][isrc][5] * g_sink_momentum_list[ipf][2] / (double)LZ_global );
  
            const double ephase[2] = { cos( phase ) , sin( phase ) } ;
  
            /***********************************************************
             * order from source time and add source phase
             ***********************************************************/
            for ( int it = 0; it < T_global; it++ ) {
              int const itt = ( conf_src_list[iconf][isrc][2] + it ) % T_global; 
              twop[ipf][iconf][isrc][iflavor][it][0] = buffer[itt][0] * ephase[0] - buffer[itt][1] * ephase[1];
              twop[ipf][iconf][isrc][iflavor][it][1] = buffer[itt][1] * ephase[0] + buffer[itt][0] * ephase[1];
            }
  
          }  /* end of loop on sources */
        }  /* end of loop on configs */
          
        fini_2level_dtable( &buffer );

        /**********************************************************
         * close the reader
         **********************************************************/
        aff_reader_close ( affr );
  
      }  /* end of loop on sink momenta */
    }  /* end of loop on flavor */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "avxg_analyse", "read-twop-aff", g_cart_id == 0 );

#endif  /* of if _TWOP_AFF */

  /***********************************************************/
  /***********************************************************/

#if _TWOP_H5

  /***********************************************************
   * loop on configs
   ***********************************************************/
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {

    double ** buffer = init_2level_dtable ( T_global, 2 );

    /***********************************************************
     * loop on sources
     ***********************************************************/
    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      /***********************************************************
       * open AFF reader
       ***********************************************************/
      char data_filename[500];
      char key[400];
    
      /* sprintf( data_filename, "stream_%c/%s/corr.%.4d.t%d.h5",
          conf_src_list[iconf][isrc][0],
          filename_prefix,
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][2] ); 
      */


      sprintf( data_filename, "%s/r%c/corr.%.4d.t%d.h5",
          filename_prefix, conf_src_list[iconf][isrc][0],
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][2] );

      if ( g_verbose > 1 ) {
        fprintf ( stdout, "# [avxn_conn_analyse] reading from data filename %s %s %d\n", data_filename, __FILE__, __LINE__ );
        fflush(stdout);
      }

      /***********************************************************
       * loop on sink momenta
       ***********************************************************/
      for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

        gettimeofday ( &ta, (struct timezone *)NULL );

        int pf[3] = {
          g_sink_momentum_list[ipf][0],
          g_sink_momentum_list[ipf][1],
          g_sink_momentum_list[ipf][2] 
        };

        int pi[3] = {
          -pf[0],
          -pf[1],
          -pf[2] 
        };

        sprintf( key, "/u+-g-u-g/t%d/gf5/pfx%dpfy%dpfz%d/gi5/pix%dpiy%dpiz%d",
                conf_src_list[iconf][isrc][2], 
                pf[0], pf[1], pf[2],
                pi[0], pi[1], pi[2] );

        
        if ( g_verbose > 2 ) {
          fprintf ( stdout, "# [avxn_conn_analyse] key = %s\n", key );
          fflush(stdout);
        }

        exitstatus = read_from_h5_file ( (void*)buffer[0], data_filename, key, "double", io_proc );
        if ( exitstatus != 0 ) {
          fprintf( stderr, "[avxn_conn_analyse] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        /***********************************************************
         * NOTE: NO SOURCE PHASE NECESSARY
         * ONLY REORDERUNG from source
         ***********************************************************/
#pragma omp parallel for
        for ( int it = 0; it < T_global; it++ ) {
          int const itt = ( it + conf_src_list[iconf][isrc][2] + T_global ) % T_global;
          twop[ipf][iconf][isrc][0][it][0] =  buffer[itt][0];
          twop[ipf][iconf][isrc][0][it][1] =  buffer[itt][1];

          twop[ipf][iconf][isrc][1][it][0] =  buffer[itt][0];
          twop[ipf][iconf][isrc][1][it][1] = -buffer[itt][1];
        }

        /***********************************************************
         * NOTE: opposite parity transformed case is given by 
         *       complex conjugate
         *       
         ***********************************************************/

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "avxg_analyse", "read-twop-h5", g_cart_id == 0 );

      }  /* end of loop on sink momenta */

    }  /* end of loop on sources */

    fini_2level_dtable ( &buffer );
  }  /* end of loop on configs */
          
#endif  /* end of _TWOP_H5 */


  /***********************************************************/
  /***********************************************************/

/**********************************************************/
#if _TWOP_AVGX_H5
/**********************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

  /***********************************************************
   * loop on configs
   ***********************************************************/
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {

    double *** buffer = init_3level_dtable ( 2, T_global, 2 );

    /***********************************************************
     * loop on sources
     ***********************************************************/
    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      /***********************************************************
       * open AFF reader
       ***********************************************************/
      char data_filename[500];
    
      /* sprintf( data_filename, "stream_%c/%s/%d/%s.%.4d.t%d.s%d.h5",
          conf_src_list[iconf][isrc][0],
          filename_prefix,
          conf_src_list[iconf][isrc][1],
          filename_prefix2,
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][2],
          conf_src_list[iconf][isrc][3] ); */

      sprintf( data_filename, "stream_%c/%s.%.4d.t%d.s%d.h5",
          conf_src_list[iconf][isrc][0],
          filename_prefix2,
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][2],
          conf_src_list[iconf][isrc][3] );


      /***********************************************************
       * loop on sink momenta
       ***********************************************************/
      for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

        int pf[3] = {
          g_sink_momentum_list[ipf][0],
          g_sink_momentum_list[ipf][1],
          g_sink_momentum_list[ipf][2] 
        };

        int pi[3] = {
          -pf[0],
          -pf[1],
          -pf[2] 
        };

        char key[400], key2[400];

/*        if ( twop_weight[0] != 0. ) { */
          /* s-gf-l-gi/mu-0.0186/mu0.0007/t116/s0/gf5/gi5/pix-1piy0piz0/px1py0pz0 */
          
          sprintf( key, "/%s/mu%6.4f/mu%6.4f/t%d/s%d/gf5/gi5/pix%dpiy%dpiz%d/px%dpy%dpz%d",
              flavor_tag[0],
                  -g_mus, g_mu,
                  conf_src_list[iconf][isrc][2], 
                  conf_src_list[iconf][isrc][4], 
                  pi[0], pi[1], pi[2],
                  pf[0], pf[1], pf[2]);

          sprintf( key2, "/%s/mu%6.4f/mu%6.4f/t%d/s%d/gf5/gi5/pix%dpiy%dpiz%d/px%dpy%dpz%d",
              flavor_tag[0],
                  -g_mus, g_mu,
                  conf_src_list[iconf][isrc][2], 
                  0, 
                  pi[0], pi[1], pi[2],
                  pf[0], pf[1], pf[2]);



          if ( g_verbose > 3 ) fprintf ( stdout, "# [avxg_analyse_simple] key = %s %s %d\n", key, __FILE__, __LINE__  );

          exitstatus = read_from_h5_file ( (void*)(buffer[0][0]), data_filename, key, "double", io_proc );
          if ( exitstatus != 0 ) {
            fprintf( stderr, "[avxg_analyse_simple] Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", 
                data_filename, key, exitstatus, __FILE__, __LINE__ );
            /* EXIT(1); */
          
            if ( g_verbose > 3 ) fprintf ( stdout, "# [avxg_analyse_simple] key2 = %s %s %d\n", key2, __FILE__, __LINE__  );

            exitstatus = read_from_h5_file ( (void*)(buffer[0][0]), data_filename, key2, "double", io_proc );
            if ( exitstatus != 0 ) {
              fprintf( stderr, "[avxg_analyse_simple] Error from read_from_h5_file for file %s key %s, status was %d %s %d\n", 
                  data_filename, key2, exitstatus, __FILE__, __LINE__ );
              EXIT(1);
            }
          }
/*        } */  /* end of twop_weight 0 */

#if 0
        if ( twop_weight[1] != 0. ) {

          sprintf( key, "/%s/mu%6.4f/mu%6.4f/t%d/s%d/gf5/gi5/pix%dpiy%dpiz%d/px%dpy%dpz%d",
              flavor_type_2pt[flavor_id_2pt],
                  muval_2pt_list[0], -muval_2pt_list[1],
                  conf_src_list[iconf][isrc][2], 
                  conf_src_list[iconf][isrc][3], 
                  -pi[0], -pi[1], -pi[2],
                  -pf[0], -pf[1], -pf[2]);

          sprintf( key2, "/%s/mu%6.4f/mu%6.4f/t%d/s%d/gf5/gi5/pix%dpiy%dpiz%d/px%dpy%dpz%d",
              flavor_type_2pt[flavor_id_2pt],
                  muval_2pt_list[0], -muval_2pt_list[1],
                  conf_src_list[iconf][isrc][2],
                  0,
                  -pi[0], -pi[1], -pi[2],
                  -pf[0], -pf[1], -pf[2]);


          if ( g_verbose > 3 ) fprintf ( stdout, "# [avxg_analyse_simple] key = %s %s %d\n", key, __FILE__, __LINE__ );

          exitstatus = read_from_h5_file ( (void*)(buffer[1][0]), data_filename, key, "double", io_proc );
          if ( exitstatus != 0 ) {
            fprintf( stderr, "[avxg_analyse_simple] Error from read_from_h5_file for file %s key %s, status was %d %s %d\n",
                data_filename, key, exitstatus, __FILE__, __LINE__ );
            /* EXIT(1); */

            if ( g_verbose > 3 ) fprintf ( stdout, "# [avxg_analyse_simple] key2 = %s %s %d\n", key2, __FILE__, __LINE__ );

            exitstatus = read_from_h5_file ( (void*)(buffer[1][0]), data_filename, key2, "double", io_proc );
            if ( exitstatus != 0 ) {
              fprintf( stderr, "[avxg_analyse_simple] Error from read_from_h5_file file %s key %s, status was %d %s %d\n",
                  data_filename, key2, exitstatus, __FILE__, __LINE__ );
              EXIT(1);
            }

          }
        }
#endif  /* fo if 0 */

        /***********************************************************
         * NOTE: NO SOURCE PHASE NECESSARY
         * ONLY REORDERING from source
         ***********************************************************/
#pragma omp parallel for
        for ( int it = 0; it < T_global; it++ ) {
          int const itt = ( it + conf_src_list[iconf][isrc][2] + T_global ) % T_global;

          twop[ipf][iconf][isrc][0][it][0] =  buffer[0][itt][0];
          twop[ipf][iconf][isrc][0][it][1] =  buffer[0][itt][1];

          twop[ipf][iconf][isrc][1][it][0] =  buffer[0][itt][0];
          twop[ipf][iconf][isrc][1][it][1] = -buffer[0][itt][1];
        }

        /***********************************************************
         * NOTE: opposite parity transformed case is given by 
         *       ???
         *       
         ***********************************************************/

      }  /* end of loop on sink momenta */

    }  /* end of loop on sources */

    fini_3level_dtable ( &buffer );
  }  /* end of loop on configs */
          
  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "avxg_analyse", "read-twop-avgx-h5", g_cart_id == 0 );

#endif  /* end of if _TWOP_AVGX_H5 */

  /**********************************************************
   * write source-averaged data
   **********************************************************/
  for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ )
  {
    sprintf ( filename, "twop.gf%d.gi%d.PX%d_PY%d_PZ%d.corr",
        g_sink_gamma_id_list[0],
        g_source_gamma_id_list[0],
        g_sink_momentum_list[ipf][0], g_sink_momentum_list[ipf][1], g_sink_momentum_list[ipf][2] );

    FILE * fs = fopen ( filename, "w" );

    for ( int iconf = 0; iconf < num_conf; iconf++ ) 
    {
      for ( int it = 0; it < T_global; it++ ) {

        double dtmp = 0., dtmp2 = 0.;
        for ( int isrc = 0; isrc < num_src_per_conf; isrc++ )
        {
          dtmp  += twop[ipf][iconf][isrc][0][it][0];
          dtmp2 += twop[ipf][iconf][isrc][0][it][1];
        }
        dtmp /= (double)num_src_per_conf;
        dtmp2 /= (double)num_src_per_conf;
        fprintf( fs, "%4d %25.16e %25.16e  %c %6d\n", it, dtmp, dtmp2, conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
      }
    }
  
    fclose ( fs );
  }


#if _LOOP_ANALYSIS
  /**********************************************************
   *
   * loop fields
   *
   **********************************************************/
  double ****** loop = NULL;
  double ****** loop_sub = NULL;
  double ****** loop_sym = NULL;

  loop = init_6level_dtable ( g_insertion_momentum_number, num_conf, 4, 4, T_global, 2 );
  if ( loop == NULL ) {
    fprintf ( stderr, "[avxg_analyse_simple] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(25);
  }

  /**********************************************************
   *
   * read operaator insertion
   *
   **********************************************************/
#if _LOOP_CY
  for ( int imom = 0; imom < g_insertion_momentum_number; imom++ ) {

      for ( int iconf = 0; iconf < num_conf; iconf++ ) {

        char stream_tag;

        switch ( conf_src_list[iconf][0][0] ) {
          case 'a':
            stream_tag = '0';
            break;
          case 'b':
            stream_tag = '1';
            break;
          default:
            fprintf(stderr, "[xg_analyse] Error, unrecognized stream char %c %d %s %d\n", conf_src_list[iconf][0][0], __FILE__, __LINE__);
            EXIT(105);
            break;
        }


        for ( int imu = 0; imu < 4; imu++ ) {
        for ( int inu = imu; inu < 4; inu++ ) {

          sprintf ( filename, "nStout%d/%.4d_r%c/%s_%s_EMT_%d%d_%.4d_r%c.dat", stout_level_iter, conf_src_list[iconf][0][1],
              stream_tag, filename_prefix2, loop_type, imu, inu, conf_src_list[iconf][0][1], stream_tag );
          
          if ( g_verbose > 1 ) fprintf ( stdout, "# [avxg_analyse_simple] reading data from file %s\n", filename );

          FILE *ifs = fopen ( filename, "r" );
          if( ifs == NULL ) {
            fprintf ( stderr, "[avxg_analyse_simple] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
            EXIT (24);
          }
  
          int itmp[5];
          double dtmp[2];

          for ( int it = 0; it < T_global; it++ ) {
            fscanf ( ifs, "%d %d %d %d %d %lf %lf\n",
                itmp, itmp+1, itmp+2, itmp+3, itmp+4, dtmp, dtmp+1 );

            loop[imom][iconf][imu][inu][it][0] = dtmp[0] * loop_norm; 
            loop[imom][iconf][imu][inu][it][1] = dtmp[1] * loop_norm;
            /* already symmetrized here */
            if ( imu != inu ) {
              loop[imom][iconf][inu][imu][it][0] = loop[imom][iconf][imu][inu][it][0];
              loop[imom][iconf][inu][imu][it][1] = loop[imom][iconf][imu][inu][it][1];
            }
          }

          fclose ( ifs );

        }}

      }  /* end of loop on configs */
  }  /* end of loop on insertion momenta */

#endif  /* of if _LOOP_CY */

#if _LOOP_CVC
/**************************************
  # x x
  b [,1] <-  a[ 1, ] + a[16, ] + a[19, ]

  # x y
  b [,2] <-  a[ 2, ] + a[20, ]

  # x z
  b [,3] <-  a[ 3, ] - a[18, ]

  # x t
  b [,4] <-  a[ 9, ] + a[14, ]

  # y y
  b [,5] <-  a[ 7, ] + a[16, ] + a[21, ]

  # y z
  b [,6] <-  a[ 8, ] + a[17, ]

  # y t
  b [,7] <- -a[ 4, ] + a[15, ]

  # z z
  b [,8] <- a[12, ] +a[19, ] +  a[21, ]

  # z t
  b [,9] <- - a[ 5, ] - a[11, ]

  # t t
  b [,10] <- a[ 1, ] + a[ 7, ] + a[12, ]

  b <- b / 2

 **************************************/

  for ( int iconf = 0; iconf < num_conf; iconf++ ) {

    double ** buffer = init_2level_dtable ( T_global, 21 );
    if ( buffer == NULL ) {
      fprintf( stderr, "[avxg_analyse_simple] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }
    char key[400];

#if 0
    sprintf ( filename, "stream_%c/%s.%d.h5", conf_src_list[iconf][0][0],
        filename_prefix3, conf_src_list[iconf][0][1] );

    if ( g_verbose > 2 ) fprintf( stdout, "# [avxg_analyse_simple] reading data from file %s %s %d\n", filename, __FILE__, __LINE__ );


    sprintf ( key, "/StoutN%d/StoutRho%6.4f/%s/GG/", stout_level_iter, stout_level_rho, loop_type );


    exitstatus = read_from_h5_file ( buffer[0], filename, key, "double", io_proc );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[avxg_analyse_simple] Error from read_from_h5_file for key \"%s\", status was %d %s %d\n", key, exitstatus, __FILE__, __LINE__);
      EXIT( 105 );
    }
#endif

    sprintf ( filename, "stream_%c/%s.%d.aff", conf_src_list[iconf][0][0], filename_prefix3, conf_src_list[iconf][0][1] );

    if ( g_verbose > 2 ) fprintf( stdout, "# [avxg_analyse_simple] reading data from file %s %s %d\n", filename, __FILE__, __LINE__ );

    struct AffReader_s *affr = NULL;
    struct AffNode_s *affn = NULL;
    struct AffNode_s *affdir = NULL;

    affr = aff_reader (filename);
    if( const char * aff_status_str = aff_reader_errstr(affr) ) {
      fprintf(stderr, "[avxg_analyse_simple] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
      EXIT( 4 );
    }

    if( (affn = aff_reader_root( affr )) == NULL ) {
      fprintf(stderr, "[avxg_analyse_simple] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
      EXIT( 2 );
    }

    sprintf ( key, "/StoutN%d/StoutRho%6.4f/%s/GG/", stout_level_iter, stout_level_rho, loop_type );

    affdir = aff_reader_chpath ( affr, affn, key );
    if ( affdir == NULL ) {
      fprintf(stderr, "[avxg_analyse_simple] Error from affdir for dir %s %s %d\n", key, __FILE__, __LINE__);
      EXIT( 2 );
    }

    if ( g_verbose > 2 ) fprintf ( stdout, "# [avxg_analyse_simple] key = %s %s %d\n", key, __FILE__, __LINE__ );

    uint32_t items = 21 * T_global;

    exitstatus = aff_node_get_double ( affr, affdir, buffer[0], items );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[avxg_analyse_simple] Error from aff_node_get_complex for key \"%s\", status was %d errmsg %s %s %d\n", key, exitstatus,
      aff_reader_errstr ( affr ), __FILE__, __LINE__);
      EXIT( 105 );
    }


    aff_reader_close ( affr );

    /* exitstatus = read_aff_contraction ( buffer[0], NULL, filename, key, T_global*21);
    if ( exitstatus != 0 ) {
      fprintf( stderr, "[avxg_analyse_simple] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    } */


    for ( int it = 0; it < T_global; it++ ) {
      int const imom = 0;
      double * const a = buffer[it];
      // x x
      
      loop[imom][iconf][0][0][it][0] = a[0] + a[15] + a[18];

      // x y
      loop[imom][iconf][0][1][it][0] = a[1] + a[19];
      loop[imom][iconf][1][0][it][0] = loop[imom][iconf][0][1][it][0];

      // x z
      loop[imom][iconf][0][2][it][0] = a[2] - a[17];
      loop[imom][iconf][2][0][it][0] = loop[imom][iconf][0][2][it][0];

      // x t
      loop[imom][iconf][0][3][it][0] = a[8] + a[13];
      loop[imom][iconf][3][0][it][0] = loop[imom][iconf][0][3][it][0];

      // y y
      loop[imom][iconf][1][1][it][0] = a[6] + a[15] + a[20];

      // y z
      loop[imom][iconf][1][2][it][0] = a[7] + a[16];
      loop[imom][iconf][2][1][it][0] = loop[imom][iconf][1][2][it][0];

      // y t
      loop[imom][iconf][1][3][it][0] = -a[3] + a[14];
      loop[imom][iconf][3][1][it][0] = loop[imom][iconf][1][3][it][0];

      // z z
      loop[imom][iconf][2][2][it][0] = a[11] +a[18] +  a[20];

      // z t
      loop[imom][iconf][2][3][it][0] = - a[4] - a[10];
      loop[imom][iconf][3][2][it][0] = loop[imom][iconf][2][3][it][0];

      // t t
      loop[imom][iconf][3][3][it][0] = a[0] + a[6] + a[11];

      for ( int imu = 0; imu < 4; imu++ ) {
      for ( int inu = 0; inu < 4; inu++ ) {
        // b <- b / 2
        loop[imom][iconf][imu][inu][it][0] *= 0.5 * loop_norm;
        loop[imom][iconf][imu][inu][it][1]  = 0.0;
      }}

    }  /* end of loop timeslices */

    fini_2level_dtable (  &buffer );

  }  /* end of loop on configs */

#endif  /* of if _LOOP_CVC */

  /**********************************************************
   *
   * build trace-subtracted tensor
   *
   **********************************************************/
  loop_sub = init_6level_dtable ( g_insertion_momentum_number, num_conf, 4, 4, T_global, 2 );
  if ( loop_sub == NULL ) {
    fprintf ( stdout, "[avxg_analyse_simple] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(25);
  }

  loop_sym = init_6level_dtable ( g_insertion_momentum_number, num_conf, 4, 4, T_global, 2 );
  if ( loop_sym == NULL ) {
    fprintf ( stdout, "[avxg_analyse_simple] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(25);
  }

  for ( int imom = 0; imom < g_insertion_momentum_number; imom++ ) {

    for ( int imu = 0; imu < 4; imu++ ) {
      for ( int idir = 0; idir < 4; idir++ ) {

        /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            /* real part */
            loop_sym[imom][iconf][imu][idir][it][0] = 0.5 * ( 
                        loop[imom][iconf][imu][idir][it][0] +       loop[imom][iconf][idir][imu][it][0] 
                );
            /* imaginary part */
            loop_sym[imom][iconf][imu][idir][it][1] = 0.5 * ( 
                        loop[imom][iconf][imu][idir][it][1] +       loop[imom][iconf][idir][imu][it][1] 
                );

            /**********************************************************
             * subtract trace for diagonal
             *
             * loop_sub = loop - 1/4 tr loop
             **********************************************************/
            loop_sub[imom][iconf][imu][idir][it][0] = loop_sym[imom][iconf][imu][idir][it][0];
            loop_sub[imom][iconf][imu][idir][it][1] = loop_sym[imom][iconf][imu][idir][it][1];

            if ( imu == idir )
            {
              /* real part */
              loop_sub[imom][iconf][imu][idir][it][0] -= 0.25 * ( 
                         loop[imom][iconf][0][0][it][0]
                 +       loop[imom][iconf][1][1][it][0]
                 +       loop[imom][iconf][2][2][it][0]
                 +       loop[imom][iconf][3][3][it][0] 
                 );

              /* imarginary */
              loop_sub[imom][iconf][imu][idir][it][1] -= 0.25 * ( 
                         loop[imom][iconf][0][0][it][1]
                 +       loop[imom][iconf][1][1][it][1]
                 +       loop[imom][iconf][2][2][it][1]
                 +       loop[imom][iconf][3][3][it][1] 
                 );
            }
          }
        }
      }
    }
  }  /* end of loop on insertion momentum */

  /**********************************************************
   * tag to characterize the loops w.r.t. low-mode and
   * stochastic part
   **********************************************************/
  char loop_tag[400];
  sprintf ( loop_tag, "nstout%d_%6.4f", stout_level_iter, stout_level_rho );

  /**********************************************************
   * write loop_sub to separate ascii file
   **********************************************************/
  if ( write_data ) {
    for ( int imom = 0; imom < g_insertion_momentum_number; imom++ ) {
      for ( int imu = 0; imu < 4; imu++ ) {
      for ( int idir = 0; idir < 4; idir++ ) {

        sprintf ( filename, "loop_sub.%s.%s.mu%d_nu%d.PX%d_PY%d_PZ%d.corr",
            loop_type, loop_tag, imu, idir,
            g_insertion_momentum_list[imom][0],
            g_insertion_momentum_list[imom][1],
            g_insertion_momentum_list[imom][2] );


        FILE * loop_sub_fs = fopen( filename, "w" );
        if ( loop_sub_fs == NULL ) {
          fprintf ( stderr, "[avxg_analyse_simple] Error from fopen %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        } 

        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          fprintf ( loop_sub_fs , "# %c %6d\n", conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
          for ( int it = 0; it < T_global; it++ ) {
            fprintf ( loop_sub_fs , "%3d %25.16e %25.16e\n", it, loop_sub[imom][iconf][imu][idir][it][0], loop_sub[imom][iconf][imu][idir][it][1] );
          }
        }
        fclose ( loop_sub_fs );

        sprintf ( filename, "loop_sym.%s.%s.mu%d_nu%d.PX%d_PY%d_PZ%d.corr",
            loop_type, loop_tag, imu, idir,
            g_insertion_momentum_list[imom][0],
            g_insertion_momentum_list[imom][1],
            g_insertion_momentum_list[imom][2] );

        FILE * loop_sym_fs = fopen( filename, "w" );
        if ( loop_sym_fs == NULL ) {
          fprintf ( stderr, "[avxg_analyse_simple] Error from fopen %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        } 

        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          fprintf ( loop_sym_fs , "# %c %6d\n", conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
          for ( int it = 0; it < T_global; it++ ) {
            fprintf ( loop_sym_fs , "%3d %25.16e %25.16e\n", it, loop_sym[imom][iconf][imu][idir][it][0], loop_sym[imom][iconf][imu][idir][it][1] );
          }
        }
        fclose ( loop_sym_fs );
      }}  /* end of loop on idir, imu */
    }  /* end of loop on insertion momentum */
  }  /* end of if write data */

  /**********************************************************
   * loop vev for operators
   **********************************************************/
  double *** loop_sub_tavg = init_3level_dtable ( g_insertion_momentum_number, num_conf, 3 );
  if ( loop_sub_tavg == NULL ) {
    fprintf( stderr, "[avxg_analyse_simple] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }

  for ( int imom = 0; imom < g_insertion_momentum_number; imom++ ) {

#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int it = 0; it < T_global; it++ ) {

        /**********************************************************
         * simple average for O44 = op 0
         **********************************************************/
        loop_sub_tavg[imom][iconf][0] += loop_sub[imom][iconf][3][3][it][0];

        /**********************************************************
         * momentum average for Oik = op 1
         **********************************************************/
        for ( int ip = 0; ip < g_sink_momentum_number; ip++ ) {

          double const p[3] = {
              2 * M_PI * g_sink_momentum_list[ip][0] / (double)LX_global,
              2 * M_PI * g_sink_momentum_list[ip][1] / (double)LY_global,
              2 * M_PI * g_sink_momentum_list[ip][2] / (double)LZ_global };

          loop_sub_tavg[imom][iconf][1] += p[0] * p[0] * loop_sub[imom][iconf][0][0][it][0] + p[1] * p[1] * loop_sub[imom][iconf][1][1][it][0] + p[2] * p[2] * loop_sub[imom][iconf][2][2][it][0];
        }

        /**********************************************************
         * zero for O4k = op 2
         **********************************************************/
        /* loop_sub_tavg[imom][iconf][2] += 0.; */

      }

      /**********************************************************
       * normalize
       **********************************************************/
      loop_sub_tavg[imom][iconf][0] /= (double)T_global;

      double const p[3] = {
              2 * M_PI * g_sink_momentum_list[0][0] / (double)LX_global,
              2 * M_PI * g_sink_momentum_list[0][1] / (double)LY_global,
              2 * M_PI * g_sink_momentum_list[0][2] / (double)LZ_global };

      loop_sub_tavg[imom][iconf][1] /= (double)T_global * g_sink_momentum_number * ( p[0] * p[0] + p[1] * p[1] + p[2] * p[2] );
    }
  
  }  /* end of loop on momenta */


  fini_6level_dtable ( &loop );


#if _RAT_METHOD
  /**********************************************************
   *
   **********************************************************/

  /**********************************************************
   * loop on source - sink time separations
   **********************************************************/
  for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {

    int const adt = g_sequential_source_timeslice_list[idt] >= 0 ? g_sequential_source_timeslice_list[idt] : -g_sequential_source_timeslice_list[idt];
    int const sdt  = g_sequential_source_timeslice_list[idt] >= 0 ? 1 : -1;


    double ***** threep = init_5level_dtable ( 3, g_sink_momentum_number, num_conf, T_global, 2 ) ;
    if ( threep == NULL ) {
      fprintf ( stderr, "[avxg_analyse_simple] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) 
      {

#if 0
        /* sink time = source time + dt  */
        int const tsink  = (  g_sequential_source_timeslice_list[idt] + conf_src_list[iconf][isrc][2] + T_global ) % T_global;
        /* sink time with time reversal = source time - dt  */
        int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + conf_src_list[iconf][isrc][2] + T_global ) % T_global;
#endif  /* of if 0  */
        /**********************************************************
         * already ordered from source
         **********************************************************/
        int const tsink  = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
        int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + T_global ) % T_global;

        if ( g_verbose > 4 ) fprintf ( stdout, "# [avxg_analyse_simple] t_src %3d   dt %3d   tsink %3d tsink2 %3d\n", conf_src_list[iconf][isrc][2],
            g_sequential_source_timeslice_list[idt], tsink, tsink2 );

        /**********************************************************
         * !!! LOOK OUT:
         *       This includes the momentum orbit average !!!
         **********************************************************/
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

          double const mom[3] = { 
              2 * M_PI * g_sink_momentum_list[imom][0] / (double)LX_global,
              2 * M_PI * g_sink_momentum_list[imom][1] / (double)LY_global,
              2 * M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global };

              /* int const parity_sign_tensor[4] = { 1, 
                2 * ( g_sink_momentum_list[imom][0] >= 0 ) - 1,
                2 * ( g_sink_momentum_list[imom][1] >= 0 ) - 1,
                2 * ( g_sink_momentum_list[imom][2] >= 0 ) - 1 };
               */

          /* double const source_phase = 0.; */

          /* double const source_phase = -2. * M_PI * (
              g_sink_momentum_list[imom][0] * conf_src_list[iconf][isrc][3] / (double)LX_global +
              g_sink_momentum_list[imom][1] * conf_src_list[iconf][isrc][4] / (double)LY_global +
              g_sink_momentum_list[imom][2] * conf_src_list[iconf][isrc][5] / (double)LZ_global );
              */

          /* double const ephase[2] = { cos ( source_phase ), sin ( source_phase ) }; */

          /* twop values 
           * a = meson 1 at +mom
           * b = meson 2 at -mom
           */
          double const a_fwd[2] = { twop[imom][iconf][isrc][0][tsink ][0], twop[imom][iconf][isrc][0][tsink ][1] };

          double const a_bwd[2] = { twop[imom][iconf][isrc][0][tsink2][0], twop[imom][iconf][isrc][0][tsink2][1] };

          double const b_fwd[2] = { twop[imom][iconf][isrc][1][tsink ][0], twop[imom][iconf][isrc][1][tsink ][1] };
          
          double const b_bwd[2] = { twop[imom][iconf][isrc][1][tsink2][0], twop[imom][iconf][isrc][1][tsink2][1] };

          /* twop x source phase */
          /* double const a_phase[2] = { a[0] * ephase[0] - a[1] * ephase[1],
                                         a[1] * ephase[0] + a[0] * ephase[1] };

          double const b_phase[2] = { b[0] * ephase[0] + b[1] * ephase[1],
                                      b[1] * ephase[0] - b[0] * ephase[1] };
          */

          double const a_fwd_phase[2] = { a_fwd[0], a_fwd[1] };
          double const a_bwd_phase[2] = { a_bwd[0], a_bwd[1] };

          double const b_fwd_phase[2] = { b_fwd[0], b_fwd[1] };
          double const b_bwd_phase[2] = { b_bwd[0], b_bwd[1] };

          double const threep_norm = 1. / ( fabs( twop_weight[0]   ) + fabs( twop_weight[1]   ) )
                                        / ( fabs( fbwd_weight[0]   ) + fabs( fbwd_weight[1]   ) )
                                        / ( fabs( mirror_weight[0] ) + fabs( mirror_weight[1] ) );



          /* loop on insertion times */
          for ( int it = 0; it < T_global; it++ ) {

            /* fwd 1 insertion time = source time      + it */
            int const tins_fwd_1 = (        it + conf_src_list[iconf][isrc][2]                                           + 2*T_global ) % T_global;

            /* fwd 2 insertion time = source time + dt - it */
            int const tins_fwd_2 = ( -sdt * it + conf_src_list[iconf][isrc][2] + g_sequential_source_timeslice_list[idt] + 2*T_global ) % T_global;

            /* bwd 1 insertion time = source time      - it */
            int const tins_bwd_1 = (       -it + conf_src_list[iconf][isrc][2]                                           + 2*T_global ) % T_global;

            /* bwd 2 insertion time = source time - dt + it */
            int const tins_bwd_2 = (  sdt * it + conf_src_list[iconf][isrc][2] - g_sequential_source_timeslice_list[idt] + 2*T_global ) % T_global;

            if ( g_verbose > 4 ) {
              fprintf ( stdout, "# [avxn_average] insertion times stream %c conf %4d tsrc %3d dt %2d it %3d tins %3d %3d %3d %3d\n",
                  conf_src_list[iconf][isrc][0],
                  conf_src_list[iconf][isrc][1],
                  conf_src_list[iconf][isrc][2], g_sequential_source_timeslice_list[idt], it,
                  tins_fwd_1, tins_fwd_2, tins_bwd_1, tins_bwd_2 );
            }

            /**********************************************************
             * O44, real parts only
             **********************************************************/
            threep[0][imom][iconf][it][0] += threep_norm * (
                    /* twop */
                    twop_weight[0] * (
                          fbwd_weight[0] * a_fwd_phase[0] * ( mirror_weight[0] * loop_sub[0][iconf][3][3][tins_fwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][3][3][tins_fwd_2][0] )
                        + fbwd_weight[1] * a_bwd_phase[0] * ( mirror_weight[0] * loop_sub[0][iconf][3][3][tins_bwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][3][3][tins_bwd_2][0] )
                      )
                    /* twop parity partner */
                  + twop_weight[1] * (
                          fbwd_weight[0] * b_fwd_phase[0] * ( mirror_weight[0] * loop_sub[0][iconf][3][3][tins_fwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][3][3][tins_fwd_2][0] )
                        + fbwd_weight[1] * b_bwd_phase[0] * ( mirror_weight[0] * loop_sub[0][iconf][3][3][tins_bwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][3][3][tins_bwd_2][0] )
                      )
                  );



            /**********************************************************
             * Oik, again only real parts
             **********************************************************/
            for ( int i = 0; i < 3; i++ ) {
              for ( int k = 0; k < 3; k++ ) {
                /* threep_ik[iconf][isrc][it][0] += (
                    ( a_fwd_phase[0] + b_fwd_phase[0] ) * ( loop_sub[0][iconf][i][k][tins_fwd_1][0] + loop_sub[0][iconf][i][k][tins_fwd_2][0] )
                  + ( a_bwd_phase[0] + b_bwd_phase[0] ) * ( loop_sub[0][iconf][i][k][tins_bwd_1][0] + loop_sub[0][iconf][i][k][tins_bwd_2][0] )
                ) * 0.125 * mom[i] * mom[k]; */

                threep[1][imom][iconf][it][0] += threep_norm * (
                    /* twop */
                    twop_weight[0] * (
                          fbwd_weight[0] * a_fwd_phase[0] * ( mirror_weight[0] * loop_sub[0][iconf][i][k][tins_fwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][i][k][tins_fwd_2][0] )
                        + fbwd_weight[1] * a_bwd_phase[0] * ( mirror_weight[0] * loop_sub[0][iconf][i][k][tins_bwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][i][k][tins_bwd_2][0] )
                      )
                    /* twop parity partner */
                  + twop_weight[1] * (
                          fbwd_weight[0] * b_fwd_phase[0] * ( mirror_weight[0] * loop_sub[0][iconf][i][k][tins_fwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][i][k][tins_fwd_2][0] )
                        + fbwd_weight[1] * b_bwd_phase[0] * ( mirror_weight[0] * loop_sub[0][iconf][i][k][tins_bwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][i][k][tins_bwd_2][0] )
                      )
                  ) * mom[i] * mom[k];

              }
            }

            /**********************************************************
             * O4k real part of loop, imaginary part of twop
             **********************************************************/
            for ( int k = 0; k < 3; k++ ) {

              /* threep_4k[iconf][isrc][it][0] += (
                   ( a_fwd_phase[1] - b_fwd_phase[1] ) * ( loop_sub[0][iconf][3][k][tins_fwd_1][0] + loop_sub[0][iconf][3][k][tins_fwd_2][0] )
                 + ( a_bwd_phase[1] - b_bwd_phase[1] ) * ( loop_sub[0][iconf][3][k][tins_bwd_1][0] + loop_sub[0][iconf][3][k][tins_bwd_2][0] )
              ) * 0.125 * mom[k]; */

              threep[2][imom][iconf][it][0] += threep_norm * (
                  /* twop */
                  twop_weight[0] * (
                        fbwd_weight[0] * a_fwd_phase[1] * ( mirror_weight[0] * loop_sub[0][iconf][3][k][tins_fwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][3][k][tins_fwd_2][0] )
                      - fbwd_weight[1] * a_bwd_phase[1] * ( mirror_weight[0] * loop_sub[0][iconf][3][k][tins_bwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][3][k][tins_bwd_2][0] )
                    )
                  /* MINUS twop parity partner */
                 - twop_weight[1] * (
                        fbwd_weight[0] * b_fwd_phase[1] * ( mirror_weight[0] * loop_sub[0][iconf][3][k][tins_fwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][3][k][tins_fwd_2][0] )
                      - fbwd_weight[1] * b_bwd_phase[1] * ( mirror_weight[0] * loop_sub[0][iconf][3][k][tins_bwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][3][k][tins_bwd_2][0] )
                    )
                ) * mom[k];

            }  /* end of loop on spatial momentum index */

          }  /* end of loop on it */

        }  /* end of loop on imom */
      }  /* end of loop on isrc */

    }  /* end of loop on iconf */

    /**********************************************************
     * normalize
     **********************************************************/
#pragma omp parallel for
    for ( int k = 0; k < 3; k++ )
    {
      double const norm = 1. / (double)num_src_per_conf;
      for ( int iconf = 0; iconf < num_conf; iconf++ )
      {
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
        {
          for ( int it = 0; it < 2 * T_global; it++ ) {
            threep[k][imom][iconf][0][it] *= norm;
          }
        }
      }
    }

    /**********************************************************
     * write 3pt function to ascii file, per source
     **********************************************************/
    for ( int k = 0; k < 3; k++ ) 
    {
      for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
      {
        sprintf ( filename, "threep.%s.%s.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s.corr",
              loop_type, loop_tag, threep_tag[k],
              g_sequential_source_timeslice_list[idt],
              g_sink_momentum_list[imom][0],
              g_sink_momentum_list[imom][1],
              g_sink_momentum_list[imom][2], fbwd_type );

        FILE * fs = fopen ( filename, "w" );

        for ( int iconf = 0; iconf < num_conf; iconf++ )
        {
          for ( int it = 0; it < T_global; it++ ) 
          {
            fprintf( fs, "%4d %25.16e %25.16e   %c %6d\n", it, threep[k][imom][iconf][0][2*it],  threep[k][imom][iconf][0][2*it+1],
                conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
          }
        }

        fclose ( fs );
      }

    }  /* end of loop on 3pt function types */
    
    fini_5level_dtable ( &threep );

  }  /* end of loop on dt */

#endif  /* end of _RAT_METHOD */

  /**********************************************************/
  /**********************************************************/


  fini_6level_dtable ( &loop_sub );
  fini_6level_dtable ( &loop_sym );
  fini_3level_dtable ( &loop_sub_tavg );

#endif  /* end of ifdef _LOOP_ANALYSIS */

  fini_6level_dtable ( &twop );


  /**********************************************************
   * free and finalize
   **********************************************************/

  fini_3level_itable ( &conf_src_list );

#if 0
  free_geometry();
#endif

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "avxg_analyse", "runtime", g_cart_id == 0 );

  return(0);
}
