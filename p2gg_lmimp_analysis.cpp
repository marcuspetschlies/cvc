/****************************************************
 * p2gg_lmimp_analysis
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <sys/time.h>
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
  
  const char infile_prefix[] = "p2gg_exdefl_analyse";
  char const reim_str[2][6] = { "re", "im" };
  char const fbwd_str[2][6] = { "fwd", "bwd" };

  char const correlator_prefix[4][20] = { "hvp"        , "local-local", "hvp"        , "local-cvc"  };

  char const flavor_tag[4][20]        = { "u-cvc-u-cvc", "u-gf-u-gi"  , "u-cvc-u-lvc", "u-gf-u-cvc" };

  char const loop_type_tag[3][8]      = { "NA", "dOp", "Scalar" };

  char const pgg_operator_type_tag[3][12] = { "p-cvc-cvc" , "p-lvc-lvc" , "p-cvc-lvc" };


  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[400];
  int num_conf = 0;
  char ensemble_name[100] = "cA211a.30.32";
  struct timeval ta, tb;
  int evecs_num = -1;
  int evecs_use_step = -1;
  int evecs_use_min = -1;
  int write_data = 0;
  int operator_type_disc = 1;
  int operator_type_conn = 1;
  int loop_type = 2;
  int use_reim_disc = -1;
  int use_reim_conn = -1;
  double lm_am_weight[3] = { 0., 0., 0. };
  int use_conn_disc = 0;
  int use_fbwd = 0;

  while ((c = getopt(argc, argv, "h?f:N:O:E:n:s:m:w:r:b:u:d:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_lmimp_analysis] number of configs = %d\n", num_conf );
      break;
    case 'O':
      sscanf( optarg, "%d,%d", &operator_type_conn, &operator_type_disc );
      fprintf ( stdout, "# [p2gg_lmimp_analysis] operator_type_conn/disc set to %d / %d\n", operator_type_conn, operator_type_disc );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [p2gg_lmimp_analysis] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'n':
      evecs_num = atoi ( optarg );
      break;
    case 's':
      evecs_use_step = atoi ( optarg );
      break;
    case 'm':
      evecs_use_min = atoi ( optarg );
      break;
    case 'w':
      write_data = atoi( optarg );
      fprintf ( stdout, "# [p2gg_lmimp_analysis] write_data set to %d\n", write_data );
      break;
    case 'r':
      sscanf( optarg, "%d,%d", &use_reim_conn, &use_reim_disc );
      fprintf ( stdout, "# [p2gg_lmimp_analysis] use_reim_conn / disc set to %d / %d\n", use_reim_conn, use_reim_disc );
      break;
    case 'b':
      sscanf ( optarg, "%lf,%lf,%lf",  lm_am_weight, lm_am_weight+1, lm_am_weight+2 );
      fprintf ( stdout, "# [p2gg_lmimp_analysis] weights set to pta-am %16.7f   pta-lm %16.7f   ata-lm %16.7f\n", 
          lm_am_weight[0], lm_am_weight[1], lm_am_weight[2] );
      break;
    case 'u':
      use_conn_disc = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_lmimp_analysis] use_conn_disc set to %d\n", use_conn_disc );
      break;
    case 'd':
      use_fbwd = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_lmimp_analysis] use_fbwd set to %d\n", use_fbwd );
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
  if(filename_set==0) strcpy(filename, "analyse.input");
  /* fprintf(stdout, "# [p2gg_lmimp_analysis] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /***********************************************************
   * initialize MPI parameters for cvc
   ***********************************************************/
  mpi_init(argc, argv);

  /***********************************************************
   * report git version
   ***********************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [p2gg_lmimp_analysis] git version = %s\n", g_gitversion);
  }

  /***********************************************************
   * set number of openmp threads
   ***********************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_lmimp_analysis] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_lmimp_analysis] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_lmimp_analysis] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_lmimp_analysis] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[p2gg_lmimp_analysis] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [p2gg_lmimp_analysis] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
 
#if 0  
  /***********************************************************
   * check relative sign
   ***********************************************************/
  if ( lm_am_rel_sign == 0 ) {
    fprintf ( stderr, "[p2gg_lmimp_analysis] Error, lm - am rel sign not set\n" );
    EXIT(34);
  }
#endif  /* of if 0 */

  /**********************************************************
   * check use_reim
   **********************************************************/
  if ( use_conn_disc == 1 || use_conn_disc == 3 ) {
    if ( use_reim_conn == -1 ) {
      fprintf ( stderr, "[p2gg_lmimp_analysis] use_reim_conn must be specified\n" );
      EXIT(1);
    }
  }
  if ( use_conn_disc == 2 || use_conn_disc == 3 ) {
    if ( use_reim_disc == -1 ) {
      fprintf ( stderr, "[p2gg_lmimp_analysis] use_reim_disc must be specified\n" );
      EXIT(1);
    }
  }

  /**********************************************************
   * loop on sequential source momenta
   **********************************************************/
  for ( int iseq_source_momentum = 0; iseq_source_momentum < g_seq_source_momentum_number; iseq_source_momentum++)
  {

    int const seq_source_momentum[3] = {
      g_seq_source_momentum_list[iseq_source_momentum][0],
      g_seq_source_momentum_list[iseq_source_momentum][1],
      g_seq_source_momentum_list[iseq_source_momentum][2] };

    /**********************************************************
     * loop on sequential source gamma matrices
     **********************************************************/
    for( int isequential_source_gamma_id = 0; isequential_source_gamma_id < g_sequential_source_gamma_id_number; isequential_source_gamma_id++)
    {

      int const sequential_source_gamma_id = g_sequential_source_gamma_id_list[ isequential_source_gamma_id ];

      /**********************************************************
       * loop on sequential source timeslice
       **********************************************************/
      for ( int isequential_source_timeslice = 0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++)
      {

        /* int const sequential_source_timeslice = abs( g_sequential_source_timeslice_list[ isequential_source_timeslice ] ); */
        int const sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];

        /**********************************************************
         * loop on sink momentum
         **********************************************************/
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

           int const sink_momentum[3] = {
               g_sink_momentum_list[imom][0], 
               g_sink_momentum_list[imom][1],
               g_sink_momentum_list[imom][2] };

          /***********************************************************
           * read pta am data
           ***********************************************************/
          double *** pta_am_disc = NULL, *** pta_am_conn = NULL;

          if ( use_conn_disc == 2 || use_conn_disc == 3 ) {
            pta_am_disc = init_3level_dtable ( 2, num_conf, T_global );
            if( pta_am_disc == NULL ) {
              fprintf ( stderr, "[p2gg_lmimp_analysis] Error from init_3level_dtable  %s %d\n", __FILE__, __LINE__ );
              EXIT(1);
            }
          }

          if ( use_conn_disc == 1 || use_conn_disc == 3 ) {
            pta_am_conn = init_3level_dtable ( 2, num_conf, T_global );
            if( pta_am_conn == NULL ) {
              fprintf ( stderr, "[p2gg_lmimp_analysis] Error from init_3level_dtable  %s %d\n", __FILE__, __LINE__ );
              EXIT(1);
            }
          }

          for ( int ifbwd = 0; ifbwd <= 1; ifbwd++ ) {

            if ( ifbwd == 0 && use_fbwd != 1 && use_fbwd != 3 ) {
              continue;
            } else if ( ifbwd == 1 && use_fbwd != 2 && use_fbwd != 3 ) {
              continue;
            }
  
            /***********************************************************
             * P-disc
             ***********************************************************/
            if ( use_conn_disc == 2 || use_conn_disc == 3 ) {

              int const tseq = ifbwd == 0 ? sequential_source_timeslice : -sequential_source_timeslice;

              sprintf ( filename, "%s/pgg_disc.%s.%s.%s.orbit.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.%s.dat",
                  filename_prefix,
                  correlator_prefix[operator_type_disc], flavor_tag[operator_type_disc],
                  loop_type_tag[loop_type],
                  seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id, tseq,
                  sink_momentum[0], sink_momentum[1], sink_momentum[2], reim_str[use_reim_disc] );

              FILE * fs_pta_am_disc = fopen ( filename, "r" );
              if( fs_pta_am_disc == NULL ) {
                fprintf ( stderr, "[p2gg_lmimp_analysis] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
                EXIT(2);
              }
              fprintf ( stdout, "# [p2gg_lmimp_analysis] reading pta am from file %s %s %d\n", filename, __FILE__, __LINE__ );

              for ( int iconf = 0; iconf < num_conf; iconf++) {
                int itmp[2];
                for ( int it = 0; it < T_global; it++ ) {
                  fscanf ( fs_pta_am_disc, "%d %lf %d\n", itmp, pta_am_disc[ifbwd][iconf]+it , itmp+1 );
                }
              }
              fclose ( fs_pta_am_disc );
            }  /* end of if use conn disc */

            /***********************************************************
             * fully connected
             ***********************************************************/
            if ( use_conn_disc == 1 || use_conn_disc == 3 ) {

              int const tseq = ifbwd == 0 ? -sequential_source_timeslice : sequential_source_timeslice;

              sprintf ( filename, "%s/pgg_conn.%s.orbit.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.%s.dat",
                  filename_prefix,
                  pgg_operator_type_tag[operator_type_conn],
                  seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id, tseq,
                  sink_momentum[0], sink_momentum[1], sink_momentum[2], reim_str[use_reim_conn] );

              FILE * fs_pta_am_conn = fopen ( filename, "r" );
              if( fs_pta_am_conn == NULL ) {
                fprintf ( stderr, "[p2gg_lmimp_analysis] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
                EXIT(2);
              }
              fprintf ( stdout, "# [p2gg_lmimp_analysis] reading pta am from file %s %s %d\n", filename, __FILE__, __LINE__ );

              for ( int iconf = 0; iconf < num_conf; iconf++) {
                int itmp[2];
                for ( int it = 0; it < T_global; it++ ) {
                  fscanf ( fs_pta_am_conn, "%d %lf %d\n", itmp, pta_am_conn[ifbwd][iconf]+it , itmp+1 );
                }
              }
              fclose ( fs_pta_am_conn );
            }  /* end of if use conn disc */

          }  /* end of loop on ifbwd */

          /***********************************************************
           * loop on deflation space dimension
           ***********************************************************/
          for ( int ievecs = evecs_use_min; ievecs <= evecs_num; ievecs += evecs_use_step ) {

            double *** pta_lm_disc = NULL, *** ata_lm_disc = NULL;
            double *** pta_lm_conn = NULL, *** ata_lm_conn = NULL;

            if ( use_conn_disc == 2 || use_conn_disc == 3 ) {
              pta_lm_disc = init_3level_dtable ( 2, num_conf, T_global );
              if( pta_lm_disc == NULL ) {
                fprintf ( stderr, "[p2gg_lmimp_analysis] Error from init_3level_dtable  %s %d\n", __FILE__, __LINE__ );
                EXIT(3);
              }

              ata_lm_disc = init_3level_dtable ( 2, num_conf, T_global );
              if( ata_lm_disc == NULL ) {
                fprintf ( stderr, "[p2gg_lmimp_analysis] Error from init_3level_dtable  %s %d\n", __FILE__, __LINE__ );
                EXIT(1);
              }
            }
  
            if ( use_conn_disc == 1 || use_conn_disc == 3 ) {
              pta_lm_conn = init_3level_dtable ( 2, num_conf, T_global );
              if( pta_lm_conn == NULL ) {
                fprintf ( stderr, "[p2gg_lmimp_analysis] Error from init_3level_dtable  %s %d\n", __FILE__, __LINE__ );
                EXIT(3);
              }

              ata_lm_conn = init_3level_dtable ( 2, num_conf, T_global );
              if( ata_lm_conn == NULL ) {
                fprintf ( stderr, "[p2gg_lmimp_analysis] Error from init_3level_dtable  %s %d\n", __FILE__, __LINE__ );
                EXIT(1);
              }
            }
  
            if ( ievecs > 0 ) {

              for ( int ifbwd = 0; ifbwd <= 1; ifbwd++ ) {

                if ( ifbwd == 0 && use_fbwd != 1 && use_fbwd != 3 ) {
                  continue;
                } else if ( ifbwd == 1 && use_fbwd != 2 && use_fbwd != 3 ) {
                  continue;
                }
  
                if ( use_conn_disc == 2 || use_conn_disc == 3 ) {

                  /***********************************************************
                   * read pta lm P-disc
                   ***********************************************************/
                  int const tseq = ifbwd == 0 ? sequential_source_timeslice : -sequential_source_timeslice;
  
                  sprintf ( filename, "%s/pta-%s/pgg_disc.%s.%s.%s.lm.pta.orbit.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.nev%d.%s.dat",
                      filename_prefix2, fbwd_str[ifbwd],
                      correlator_prefix[operator_type_disc], flavor_tag[operator_type_disc],
                      loop_type_tag[loop_type],
                      seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id, tseq,
                      sink_momentum[0], sink_momentum[1], sink_momentum[2], ievecs, reim_str[use_reim_disc] );
  
                  FILE * fs_pta_lm_disc = fopen ( filename, "r" );
                  if( fs_pta_lm_disc == NULL ) {
                    fprintf ( stderr, "[p2gg_lmimp_analysis] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
                    EXIT(2);
                  }
                  fprintf ( stdout, "# [p2gg_lmimp_analysis] reading pta lm from file %s %s %d\n", filename, __FILE__, __LINE__ );
  
                  for ( int iconf = 0; iconf < num_conf; iconf++) {
                    int itmp[2];
                    for ( int it = 0; it < T_global; it++ ) {
                      fscanf ( fs_pta_lm_disc, "%d %lf %d\n", itmp, pta_lm_disc[ifbwd][iconf]+it , itmp+1 );
                    }
                  }
                  fclose ( fs_pta_lm_disc );
  
                  /***********************************************************
                   * read ata lm P-disc
                   ***********************************************************/
  
                  sprintf ( filename, "%s/ata-%s/pgg_disc.%s.%s.%s.lm.ata.orbit.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.nev%d.%s.dat",
                      filename_prefix3, fbwd_str[ifbwd],
                      correlator_prefix[operator_type_disc], flavor_tag[operator_type_disc],
                      loop_type_tag[loop_type],
                      seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id, tseq,
                      sink_momentum[0], sink_momentum[1], sink_momentum[2], ievecs, reim_str[use_reim_disc] );
  
                  FILE * fs_ata_lm_disc = fopen ( filename, "r" );
                  if( fs_ata_lm_disc == NULL ) {
                    fprintf ( stderr, "[p2gg_lmimp_analysis] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
                    EXIT(2);
                  }
                  fprintf ( stdout, "# [p2gg_lmimp_analysis] reading ata lm from file %s %s %d\n", filename, __FILE__, __LINE__ );
  
                  for ( int iconf = 0; iconf < num_conf; iconf++) {
                    int itmp[2];
                    for ( int it = 0; it < T_global; it++ ) {
                      fscanf ( fs_ata_lm_disc, "%d %lf %d\n", itmp, ata_lm_disc[ifbwd][iconf]+it , itmp+1 );
                    }
                  }
                  fclose ( fs_ata_lm_disc );
                }

              }
            }  /* end of if ievecs > 0 */

            /***********************************************************
             * low-mode replacement step
             ***********************************************************/
            double ** pgg = init_2level_dtable ( num_conf, T_global );
            if( pgg == NULL ) {
              fprintf ( stderr, "[p2gg_lmimp_analysis] Error from init_2level_dtable  %s %d\n", __FILE__, __LINE__ );
              EXIT(10);
            }


            /***********************************************************
             * add P-disc
             ***********************************************************/
            if ( use_conn_disc == 2 || use_conn_disc == 3 ) {
              double const norm = ( use_fbwd == 3 ) ? 0.5 : 1.; 
#pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++) {
                for ( int it = 0; it < T_global; it++ ) {
                  int const itfwd = it;
                  int const itbwd = T_global - it - 2;
                  pgg[iconf][it] += norm * (
                      lm_am_weight[0] * pta_am_disc[0][iconf][itfwd] + ( -lm_am_weight[1] * pta_lm_disc[0][iconf][itfwd] + lm_am_weight[2] * ata_lm_disc[0][iconf][itfwd] )  /* fwd contribution */
                    - lm_am_weight[0] * pta_am_disc[1][iconf][itbwd] - ( -lm_am_weight[1] * pta_lm_disc[1][iconf][itbwd] + lm_am_weight[2] * ata_lm_disc[1][iconf][itbwd] )  /* bwd contribution */
  
                    );
                }
              }
            }

            /***********************************************************
             * add fully connected
             ***********************************************************/
            if ( use_conn_disc == 1 || use_conn_disc == 3 ) {
              double const norm = ( use_fbwd == 3 ) ? 0.5 : 1.; 
#pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++) {
                for ( int it = 0; it < T_global; it++ ) {
                  int const itfwd = it;
                  int const itbwd = T_global - it - 2;
                  pgg[iconf][it] += norm * (
                      lm_am_weight[0] * pta_am_conn[0][iconf][itfwd] + ( -lm_am_weight[1] * pta_lm_conn[0][iconf][itfwd] + lm_am_weight[2] * ata_lm_conn[0][iconf][itfwd] )  /* fwd contribution */
                    - lm_am_weight[0] * pta_am_conn[1][iconf][itbwd] - ( -lm_am_weight[1] * pta_lm_conn[1][iconf][itbwd] + lm_am_weight[2] * ata_lm_conn[1][iconf][itbwd] )  /* bwd contribution */
                    );
                }
              }
            }

            /***********************************************************
             * UWerr analysis
             ***********************************************************/
            char obs_name[100];
            sprintf ( obs_name, "%s.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.nev%d",
                g_outfile_prefix,
                seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], sequential_source_gamma_id, sequential_source_timeslice,
                sink_momentum[0], sink_momentum[1], sink_momentum[2], ievecs );

            exitstatus = apply_uwerr_real ( pgg[0], num_conf, T_global, 0, 1, obs_name );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[p2gg_lmimp_analysis] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(1);
            }

            if ( write_data == 1 ) {
              char output_filename[200];
              sprintf ( output_filename, "%s.dat", obs_name );

              FILE * ofs = fopen ( output_filename, "w" );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[p2gg_lmimp_analysis] Error from fopen %s %d\n", __FILE__, __LINE__ );
                EXIT(1);
              }
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for ( int it = 0; it < T_global; it++ ) {
                  int const tau = it - T_global/2 + 1;
                  fprintf ( ofs, "%4d%25.16e%8d\n", tau, pgg[iconf][it], iconf );
                }
              }

              fclose ( ofs );
            }  /* end of if write_data */

            fini_3level_dtable ( &pta_lm_disc );
            fini_3level_dtable ( &ata_lm_disc );
            fini_3level_dtable ( &pta_lm_conn );
            fini_3level_dtable ( &ata_lm_conn );

          }  /* end of loop on evecs number */

          fini_3level_dtable ( &pta_am_disc );
          fini_3level_dtable ( &pta_am_conn );

        }  /* end of loop on g_sink_momentum_number */
      }  /* end of loop on g_sequential_source_timeslice_number */
    }  /* end of loop on g_sequential_source_gamma_id_number */
  }  /* end of loop on g_seq_source_momentum_number */

  /**********************************************************
   * free the allocated memory, finalize
   **********************************************************/

  free_geometry();

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_lmimp_analysis] %s# [p2gg_lmimp_analysis] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_lmimp_analysis] %s# [p2gg_lmimp_analysis] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
