/****************************************************
 * avxn_analyse 
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
#include "uwerr.h"
#include "derived_quantities.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse cpff fht correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default cpff.input]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  

  const char flavor_prefix_std[] = "u+-g-u-g";
  const char flavor_prefix_std2[] = "d+-g-d-g";
  const char flavor_prefix_fht[] = "u+-g-suu-g";

  int const gamma_id_to_bin[16] = { 8, 1, 2, 4, 0, 15, 7, 14, 13, 11, 9, 10, 12, 3, 5, 6 };

  char const reim_str[2][3] = { "re", "im" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "cA211a.30.32";
  /* int use_disc = 0;
  int use_conn = 1; */
  int twop_fold_propagator = 0;
  int twop_use_reim = 0;

  char key[400];

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:F:R:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'F':
      twop_fold_propagator = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] fold_propagator set to %d\n", twop_fold_propagator );
      break;
    case 'R':
      twop_use_reim = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] use_reim set to %d\n", twop_use_reim );
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
  /* fprintf(stdout, "# [avxn_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [avxn_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [avxn_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [avxn_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[avxn_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[avxn_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[avxn_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [avxn_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf);
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[avxn_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 5 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[avxn_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [avxn_analyse] comment %s\n", line );
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


  if ( g_verbose > 4 ) {
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
  double ****** twop = init_6level_dtable ( g_sink_momentum_number, num_conf, num_src_per_conf, 2, T_global, 2 );
  if( twop == NULL ) {
    fprintf ( stderr, "[avxn_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT (24);
  }

  for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

    int const sink_momentum[3] = {
        g_sink_momentum_list[isink_momentum][0],
        g_sink_momentum_list[isink_momentum][1],
        g_sink_momentum_list[isink_momentum][2] };
  
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      Nconf = conf_src_list[iconf][0][0];
     
      for( int imeson = 0; imeson < 2; imeson++ ) {

        sprintf( filename, "twop/twop.%.4d.pseudoscalar.%d.PX%d_PY%d_PZ%d",  Nconf, imeson,
            ( 1 - 2 * imeson) * g_sink_momentum_list[isink_momentum][0],
            ( 1 - 2 * imeson) * g_sink_momentum_list[isink_momentum][1],
            ( 1 - 2 * imeson) * g_sink_momentum_list[isink_momentum][2] );
     
        FILE * dfs = fopen ( filename, "r" );
        if( dfs == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT (24);
        } else {
          if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_analyse] reading data from file %s filename \n", filename );
        }

        char key [400];
 
        for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

          /***********************************************************
           * already ordered from source
           ***********************************************************/
          fscanf ( dfs, "%s\n", key );
          if ( g_verbose > 4 ) {
            fprintf ( stdout, "# [avxn_analyse] reading key %s\n", key );
          }
          for ( int it = 0; it < T_global; it++ ) {
            fscanf ( dfs, "%lf %lf\n", twop[isink_momentum][iconf][isrc][imeson][it], twop[isink_momentum][iconf][isrc][imeson][it]+1 );
          }
        }
        fclose ( dfs );
      }
    }
  }  /* end of loop on sink momenta */

  /**********************************************************
   * read loop data
   **********************************************************/
  double ****** loop = NULL;

#if 0
  loop = init_6level_dtable ( g_insertion_momentum_number, num_conf, 4, T_global, 4, 8 );
  if ( loop == NULL ) {
    fprintf ( stdout, "[avxn_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(25);
  }

  for ( int imom = 0; imom < g_insertion_momentum_number; imom++ ) {
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {

      for ( int idir = 0; idir < 4; idir++ ) {

        sprintf ( filename, "loops/loop.%.4d.stoch.%s.nev0.Nstoch1.mu%d.PX%d_PY%d_PZ%d", conf_src_list[iconf][0][0],
            loop_oet_type,
            idir,
            g_insertion_momentum_list[imom][0],
            g_insertion_momentum_list[imom][1],
            g_insertion_momentum_list[imom][2] );

        FILE * dfs = fopen ( filename, "r" );
        if( dfs == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT (24);
        } else {
          if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_analyse] reading data from file %s filename \n", filename );
        }

        for ( int it = 0; it < T_global; it++ ) {
          int itmp[3];
          for ( int ia = 0; ia < 4; ia++ ) {
          for ( int ib = 0; ib < 4; ib++ ) {
            fscanf ( dfs, "%d %d %d %lf %lf\n", 
                itmp, itmp+1, itmp+2, loop[imom][iconf][idir][it][ia][2*ib  ], loop[imom][iconf][idir][it][ia][2*ib+1] );
          }}
        }
        fclose ( dfs );
      }  /* end of loop on directions */
    }  /* end of loop on configs */
  }  /* end of loop on insertion momenta */
#endif  /* of if 0 */

  /**********************************************************
   **********************************************************
   ** 
   ** STATISTICAL ANALYSIS
   ** 
   **********************************************************
   **********************************************************/

  /**********************************************************
   * 2-pt function
   **********************************************************/
  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

    for ( int ireim = 0; ireim < 2; ireim++ ) {
    
      double *** data = init_3level_dtable ( num_conf, num_src_per_conf, T );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      /* fill data array */
#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
          
          for ( int it = 0; it < T_global; it++ ) {
             data[iconf][isrc][it] = 0.5 * ( twop[imom][iconf][isrc][0][it][ireim] + twop[imom][iconf][isrc][1][it][ireim] );
          }

          if ( twop_fold_propagator) {
            for ( int it = 0; it <= T_global/2; it++ ) {
              int const itt = ( T_global - it ) % T_global;
              data[iconf][isrc][it ] = 0.5 * ( data[iconf][isrc][it] + data[iconf][isrc][itt] );
              data[iconf][isrc][itt] = data[iconf][isrc][it];
            } 
          }
        }
      }

      char obs_name[100];
      sprintf( obs_name, "twop.%.4d.pseudoscalar.PX%d_PY%d_PZ%d.%s",  Nconf,
            g_sink_momentum_list[imom][0],
            g_sink_momentum_list[imom][1],
            g_sink_momentum_list[imom][2], reim_str[ireim] );

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_real ( data[0][0], num_conf*num_src_per_conf, T_global, 0, 1, obs_name );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      fini_3level_dtable ( &data );
    }
  }

  fini_6level_dtable ( &loop );
  fini_6level_dtable ( &twop );

        
  /**********************************************************
   * free and finalize
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
    fprintf(stdout, "# [avxn_analyse] %s# [avxn_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [avxn_analyse] %s# [avxn_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);
}
