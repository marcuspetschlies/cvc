/****************************************************
 * disc_analyse 
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
#include "gamma.h"
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
  
  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_conf = 0;
  char ensemble_name[100] = "NA";
  int fold_propagator = 0;
  char oet_type[100] = "NA";

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:F:o:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [disc_analyse] number of configs = %d\n", num_conf );
      break;
    case 'F':
      fold_propagator = atoi ( optarg );
      fprintf ( stdout, "# [disc_analyse] fold_propagator set to %d\n", fold_propagator );
      break;
    case 'o':
      strcpy ( oet_type, optarg );
      fprintf ( stdout, "# [disc_analyse] oet type set to %s\n", oet_type );
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
  if(filename_set==0) strcpy(filename, "cvc.input");
  /* fprintf(stdout, "# [disc_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [disc_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [disc_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [disc_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[disc_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[disc_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[disc_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [disc_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
#if 0
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "conf.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[disc_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int * conf_src_list = init_1level_itable ( num_conf );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[disc_analyse] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [disc_analyse] comment %s\n", line );
      continue;
    }

    sscanf( line, "%d", conf_src_list + count );
    count++;
  }

  fclose ( ofs );

  if ( g_verbose > 4 ) {
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      fprintf ( stdout, "conf_src_list %6d\n", conf_src_list[iconf] )
    }
  }
#endif  /* of if 0 */

  /**********************************************************
   **********************************************************
   **
   ** read and spin-project loops
   **
   **********************************************************
   **********************************************************/

  /***********************************************************
   * allocate loop
   ***********************************************************/
  double ****** cloop = init_6level_dtable ( g_sink_momentum_number, g_nsample, T, 4, 4, 2 );
  if ( cloop == NULL ) {
    fprintf(stderr, "[disc_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  /***********************************************************
   * read cumulative loops
   ***********************************************************/
  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

    int const sink_momentum[3] = {
        g_sink_momentum_list[imom][0],
        g_sink_momentum_list[imom][1],
        g_sink_momentum_list[imom][2] };
 
    sprintf ( filename, "M.%.d.%s.PX%d_PY%d_PZ%d.4x4", Nconf, oet_type, sink_momentum[0], sink_momentum[1], sink_momentum[2] );
    FILE * ifs = fopen ( filename, "r" );
    if ( ifs == NULL ) {
      fprintf ( stderr, "[disc_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
      EXIT(12);
    }

    for ( int isample = 0; isample < g_nsample; isample++ ) {

      for ( int it = 0; it  < T; it++ ) {

        for ( int sigma = 0; sigma < 4; sigma++ ) {

          for ( int tau = 0; tau < 4; tau++ ) {
            int itmp[5];

            fscanf ( ifs , "%d %d %d %d %lf %lf %d", itmp, itmp+1, itmp+2, itmp+3, 
                cloop[imom][isample][it][tau][sigma], cloop[imom][isample][it][tau][sigma]+1, itmp+4 );

            if ( g_verbose > 4 ) {
              fprintf ( stdout, "cloop %3d   %3d %3d %3d   %25.16e %25.16e   %6d\n", itmp[0], itmp[1], itmp[2], itmp[3],
                  cloop[imom][isample][it][tau][sigma][0], cloop[imom][isample][it][tau][sigma][1], itmp[4] );
            }
          }
        }
      }
    }

    fclose ( ifs );
  }  /* end of loop on momenta */


  /**********************************************************
   * build single-source loops
   **********************************************************/
  double ****** sloop = init_6level_dtable ( g_sink_momentum_number, g_nsample, T, 4, 4, 2 );
  if ( sloop == NULL ) {
    fprintf(stderr, "[disc_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

#pragma omp parallel for
  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
    for ( int it = 0; it  < T; it++ ) {
      for ( int sigma = 0; sigma < 4; sigma++ ) {
        for ( int tau = 0; tau < 4; tau++ ) {

          sloop[imom][0][it][sigma][tau][0] = cloop[imom][0][it][sigma][tau][0];
          sloop[imom][0][it][sigma][tau][1] = cloop[imom][0][it][sigma][tau][1];

          for ( int isample = 1; isample < g_nsample; isample++ ) {
            sloop[imom][isample][it][sigma][tau][0] = cloop[imom][isample][it][sigma][tau][0] - cloop[imom][isample-1][it][sigma][tau][0];
            sloop[imom][isample][it][sigma][tau][1] = cloop[imom][isample][it][sigma][tau][1] - cloop[imom][isample-1][it][sigma][tau][1];
          }

        }
      }
    }
  }

  fini_6level_dtable ( &cloop );


  if ( g_verbose > 4 ) {
    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
      for ( int isample = 0; isample < g_nsample; isample++ ) {
        for ( int it = 0; it  < T; it++ ) {
          for ( int sigma = 0; sigma < 4; sigma++ ) {
            for ( int tau = 0; tau < 4; tau++ ) {
              fprintf ( stdout, "sloop %3d %3d %3d %3d %3d   %25.16e %25.16e\n", imom, isample, it, sigma, tau, 
                  sloop[imom][isample][it][sigma][tau][0], sloop[imom][isample][it][sigma][tau][1] );
            }
          }
        }
      }
    }
  }

  /**********************************************************
   * spin project and build correlators
   **********************************************************/

  double ***** ploop = init_5level_dtable ( g_sink_momentum_number, g_source_gamma_id_number, T, g_nsample, 2 );
  if ( ploop == NULL ) {
    fprintf ( stderr, "[disc_analyse] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(2);
  }

#pragma omp parallel for
  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
    for ( int igf = 0; igf < g_source_gamma_id_number; igf++ ) {

      gamma_matrix_type gf;
      gamma_matrix_ukqcd_binary ( &gf,  g_source_gamma_id_list[igf] );
      if ( g_verbose > 2 ) gamma_matrix_printf ( &gf, "gf", stdout );


      for ( int isample = 0; isample < g_nsample; isample++ ) {
        for ( int it = 0; it  < T; it++ ) {

          double _Complex zbuffer = 0.;

          for ( int sigma = 0; sigma < 4; sigma++ ) {
          for ( int tau = 0; tau < 4; tau++ ) {
            double _Complex z = sloop[imom][isample][it][sigma][tau][0] + sloop[imom][isample][it][sigma][tau][1] * I;
            zbuffer += z * gf.m[tau][sigma];
          }}

          ploop[imom][igf][it][isample][0] = creal ( zbuffer );
          ploop[imom][igf][it][isample][1] = cimag ( zbuffer );

        }
      }
    }
  }

  fini_6level_dtable ( &sloop );


  if ( g_verbose > 4 ) {
    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
      for ( int igf = 0; igf < g_source_gamma_id_number; igf++ ) {
        for ( int it = 0; it  < T; it++ ) {
          for ( int isample = 0; isample < g_nsample; isample++ ) {
            fprintf ( stdout, "ploop %3d %3d %3d %3d   %25.16e %25.16e\n", imom, igf, it, isample,
                ploop[imom][igf][it][isample][0], ploop[imom][igf][it][isample][1] );
          }
        }
      }
    }
  }


  /**********************************************************
   **********************************************************
   **
   ** build correlators
   **
   **********************************************************
   **********************************************************/
  
  /**********************************************************
   * loop on sink momenta
   **********************************************************/
  for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

    int const sink_momentum[3] = {
      g_sink_momentum_list[isink_momentum][0],
      g_sink_momentum_list[isink_momentum][1],
      g_sink_momentum_list[isink_momentum][2] };
 
    /**********************************************************
     * loop on source momenta
     **********************************************************/
    for ( int isource_momentum = 0; isource_momentum < g_sink_momentum_number; isource_momentum++ ) {

      int const source_momentum[3] = {
        g_sink_momentum_list[isource_momentum][0],
        g_sink_momentum_list[isource_momentum][1],
        g_sink_momentum_list[isource_momentum][2] };

      /**********************************************************
       * loop on gamma at sink
       **********************************************************/
      for ( int igf = 0; igf < g_source_gamma_id_number; igf++ ) {

        /**********************************************************
         * loop on gamma at source
         **********************************************************/
        for ( int igi = 0; igi < g_source_gamma_id_number; igi++ ) {

          double  *** corr = init_3level_dtable ( T, T, 4 );
          if ( ploop == NULL ) {
            fprintf ( stderr, "[disc_analyse] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(2);
          }

          /**********************************************************
           * loop on source times
           **********************************************************/
          for ( int tsrc = 0; tsrc < T; tsrc++ ) {
            double ** const loop_src = ploop[isource_momentum][igi][tsrc];

            double loop_src_sum[2] = {0., 0.};
            for ( int isample = 0; isample < g_nsample; isample++ ) {
              loop_src_sum[0] += loop_src[isample][0];
              loop_src_sum[1] += loop_src[isample][1];
            }
            if ( g_verbose > 4 ) fprintf ( stdout, "# [disc_analyse] loop_src_sum = %25.16e %25.16e\n", loop_src_sum[0], loop_src_sum[1] );

            /**********************************************************
             * loop on sink times
             **********************************************************/
            for ( int tsnk = 0; tsnk < T; tsnk++ ) {

              double ** const loop_snk = ploop[isink_momentum][igf][tsnk];

              double loop_snk_sum[2] = {0., 0.};

              for ( int isample = 0; isample < g_nsample; isample++ ) {
                loop_snk_sum[0] += loop_snk[isample][0];
                loop_snk_sum[1] += loop_snk[isample][1];
              }
              if ( g_verbose > 4 ) fprintf ( stdout, "# [disc_analyse] loop_snk_sum = %25.16e %25.16e\n", loop_snk_sum[0], loop_snk_sum[1] );

              double corr_diag[4] = {0., 0., 0., 0.};

              for ( int isample = 0; isample < g_nsample; isample++ ) {
                corr_diag[0] += loop_src[isample][0] * loop_snk[isample][0];
                corr_diag[1] += loop_src[isample][0] * loop_snk[isample][1];
                corr_diag[2] -= loop_src[isample][1] * loop_snk[isample][0];
                corr_diag[3] += loop_src[isample][1] * loop_snk[isample][1];
              }

              double const norm = 1. / ( LX_global * LY_global * LZ_global ) / ( LX_global * LY_global * LZ_global ) / g_nsample / ( g_nsample - 1 );

              corr[tsrc][tsnk][0] = (  loop_src_sum[0] * loop_snk_sum[0] - corr_diag[0] ) * norm;
              corr[tsrc][tsnk][1] = (  loop_src_sum[0] * loop_snk_sum[1] - corr_diag[1] ) * norm;
              corr[tsrc][tsnk][2] = ( -loop_src_sum[1] * loop_snk_sum[0] - corr_diag[2] ) * norm;
              corr[tsrc][tsnk][3] = (  loop_src_sum[1] * loop_snk_sum[1] - corr_diag[3] ) * norm;

            }  /* end of loop on tsnk */
          }  /* end of loop on tsrc */


          /**********************************************************
           * write to file
           **********************************************************/

          sprintf ( filename , "disc_%s.%.4d.PFX%d_PFY%d_PFZ%d.GF%d.PIX%d_PIY%d_PIZ%d.GI%d", oet_type, Nconf,
              sink_momentum[0], sink_momentum[1], sink_momentum[2], g_source_gamma_id_list[igf], 
              source_momentum[0], source_momentum[1], source_momentum[2], g_source_gamma_id_list[igf] );
              
          if ( g_verbose > 2 ) fprintf ( stdout, "# [disc_analyse] filename = %s\n", filename );

          FILE * ofs = fopen ( filename, "w" );
          if ( ofs == NULL ) {
            fprintf ( stderr, "[disc_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
          }

          for ( int tsrc = 0; tsrc < T; tsrc++ ) {
            for ( int tsnk = 0; tsnk < T; tsnk++ ) {
              fprintf ( ofs, "%3d %3d %25.16e %25.16e %25.16e %25.16e\n", tsrc, tsnk, 
                  corr[tsrc][tsnk][0], corr[tsrc][tsnk][1], corr[tsrc][tsnk][2], corr[tsrc][tsnk][3] );
            }
          }

          fclose ( ofs );

          fini_3level_dtable ( &corr );

        }  /* end of loop on source gamma id */

      }  /* end of loop on sink gamma id */

    }  /* end of loop on source momentum */

  }  /* end of loop on sink momentum */

  /**********************************************************
   * free the allocated memory, finalize
   **********************************************************/

  fini_5level_dtable ( &ploop );
#if 0 
#endif  /* of if 0 */

#if 0
  fini_1level_itable ( &conf_src_list );
#endif  /* of if 0 */

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
    fprintf(stdout, "# [disc_analyse] %s# [disc_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [disc_analyse] %s# [disc_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);
}
