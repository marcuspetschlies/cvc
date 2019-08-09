/****************************************************
 * p2gg_exdefl_uwerr_analyse.c
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
  
  const char infile_prefix[] = "p2gg_exdefl_analyse";
  char const reim_str[2][6] = { "re", "im" };
#if 0
  double const TWO_MPI = 2. * M_PI;

  char const reim_str[2][3] = {"re" , "im"};

  char const hvp_operator_type_tag[3][12]  = { "cvc-cvc"    , "lvc-lvc"    , "cvc-lvc" };

  char const hvp_correlator_prefix[3][20] = { "hvp"        , "local-local", "hvp"     };

  char const hvp_flavor_tag[3][20]        = { "u-cvc-u-cvc", "u-gf-u-gi"  , "u-cvc-u-lvc" };

  /* char const pgg_operator_type_tag[3][12]  = { "p-cvc-cvc"    , "p-lvc-lvc"    , "p-cvc-lvc" }; */
  char const pgg_operator_type_tag[3][12]  = { "p-cvc-cvc"    , "p-loc-loc"    , "p-cvc-lvc" };
#endif

  int c;
  int filename_set = 0;
  int check_momentum_space_WI = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_src_per_conf = 1;
  int num_conf = 0;
  char ensemble_name[100] = "cA211a.30.32";
  int fold_correlator= 0;
  struct timeval ta, tb;
  int operator_type = -1;
  /* char correlator_prefix[100], flavor_tag[100]; */
  int evecs_num = -1;
  int evecs_use_step = -1;
  int evecs_use_min = -1;

#ifdef HAVE_LHPC_AFF
  char key[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "Wh?f:N:S:F:O:E:n:s:m:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_exdefl_uwerr_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_exdefl_uwerr_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'W':
      check_momentum_space_WI = 1;
      fprintf ( stdout, "# [p2gg_exdefl_uwerr_analyse] check_momentum_space_WI set to %d\n", check_momentum_space_WI );
      break;
    case 'F':
      fold_correlator = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_exdefl_uwerr_analyse] fold_correlator set to %d\n", fold_correlator );
      break;
    case 'O':
      operator_type = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_exdefl_uwerr_analyse] operator_type set to %d\n", operator_type );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [p2gg_exdefl_uwerr_analyse] ensemble_name set to %s\n", ensemble_name );
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
  /* fprintf(stdout, "# [p2gg_exdefl_uwerr_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [p2gg_exdefl_uwerr_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_exdefl_uwerr_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_exdefl_uwerr_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_exdefl_uwerr_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_exdefl_uwerr_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[p2gg_exdefl_uwerr_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [p2gg_exdefl_uwerr_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf);
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[p2gg_exdefl_uwerr_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 5 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[p2gg_exdefl_uwerr_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [p2gg_exdefl_uwerr_analyse] comment %s\n", line );
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


  if ( g_verbose > 1 ) {
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
   *
   ***********************************************************/
  for ( int igsrc = 0; igsrc < g_source_gamma_id_number; igsrc++ ) {

    for ( int ipsrc = 0; ipsrc < g_source_momentum_number; ipsrc++ ) {

      for ( int ipsnk = 0; ipsnk < g_sink_momentum_number; ipsnk++ ) {

        for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {

          for ( int ievecs = evecs_use_min; ievecs <= evecs_num; ievecs += evecs_use_step ) {

            /***********************************************************
             *
             * ATA PGG
             *   all-to-all 3pt function analysis
             *
             ***********************************************************/
            sprintf( key, "/pgg/disc/orbit/g%d/px%d_py%d_pz%d/qx%d_qy%d_qz%d/nev%d/dt%d", g_source_gamma_id_list[igsrc],
                g_source_momentum_list[ipsrc][0], g_source_momentum_list[ipsrc][1], g_source_momentum_list[ipsrc][2],
                g_sink_momentum_list[ipsnk][0], g_sink_momentum_list[ipsnk][1], g_sink_momentum_list[ipsnk][2],
                ievecs, g_sequential_source_timeslice_list[idt] );
            if ( g_verbose > 0 ) fprintf ( stdout, "# [p2gg_exdefl_uwerr_analyse] reading key %s %s %d\n", key , __FILE__, __LINE__ );

            double _Complex ** pgg_disc = init_2level_ztable ( num_conf, T_global );
            if ( pgg_disc == NULL ) {
              fprintf(stderr, "[p2gg_exdefl_uwerr_analyse] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
              EXIT(15);
            }

            /***********************************************************
             * loop on configurations
             ***********************************************************/
            for ( int iconf = 0; iconf < num_conf; iconf++ ) {

#ifdef HAVE_LHPC_AFF
              /***********************************************************
               * reader for aff input file
               ***********************************************************/
              struct AffReader_s *affr = NULL;
              sprintf ( filename, "%s.pref_%d_%d_%d.%.4d.nev%d.aff", infile_prefix,
                  g_sink_momentum_list[ipsnk][0], g_sink_momentum_list[ipsnk][1], g_sink_momentum_list[ipsnk][2],
                  conf_src_list[iconf][0][0], evecs_num );

              if ( g_verbose > 2 ) fprintf(stdout, "# [p2gg_exdefl_uwerr_analyse] reading data from file %s\n", filename);
              affr = aff_reader ( filename );
              const char * aff_status_str = aff_reader_errstr ( affr );
              if( aff_status_str != NULL ) {
                fprintf(stderr, "[p2gg_exdefl_uwerr_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
                EXIT(15);
              }

              /* set root node */
              struct AffNode_s * affrn = aff_reader_root( affr );
              if( affrn == NULL ) {
                fprintf(stderr, "[p2gg_exdefl_uwerr_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
                EXIT(17);
              }

              struct AffNode_s * affdir = aff_reader_chpath ( affr, affrn, key );
              if ( affdir == NULL ) {
                fprintf ( stderr, "[p2gg_exdefl_uwerr_analyse] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
                EXIT(15);
              }

              uint32_t uitems = T_global;
              exitstatus = aff_node_get_complex ( affr, affdir, pgg_disc[iconf], uitems );
              if(exitstatus != 0) {
                fprintf ( stderr, "[p2gg_exdefl_uwerr_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(16);
              }

              aff_reader_close ( affr );
#else
#error "[p2gg_exdefl_uwerr_analyse] need lhp-aff lib; currently no other input method implemented"
#endif
            }  /* end of loop on configurations */

            /***********************************************************
             * UWerr analysis
             ***********************************************************/
            for ( int ireim = 0; ireim < 2; ireim++ ) {

              double ** data = init_2level_dtable ( num_conf, T_global );
              if( data == NULL ) {
                fprintf ( stderr, "[p2gg_exdefl_uwerr_analyse] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
                EXIT(16);
              }


#pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for ( int it = 0; it < T_global; it++ ) {
                  data[iconf][it] = ( ireim == 0 ) ?  creal ( pgg_disc[iconf][it] ) : cimag ( pgg_disc[iconf][it] );
                }
              }

              char obs_name[100];
              sprintf ( obs_name, "pgg_disc.lm.orbit.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.nev%d.%s", 
                  g_source_momentum_list[ipsrc][0], g_source_momentum_list[ipsrc][1], g_source_momentum_list[ipsrc][2],
                  g_source_gamma_id_list[igsrc], g_sequential_source_timeslice_list[idt],
                  g_sink_momentum_list[ipsnk][0], g_sink_momentum_list[ipsnk][1], g_sink_momentum_list[ipsnk][2],
                  ievecs, reim_str[ireim] );
              if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_exdefl_uwerr_analyse] obs_name = %s %s %d\n", obs_name, __FILE__, __LINE__ );

              /* apply UWerr analysis */
              exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[p2gg_exdefl_uwerr_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(1);
              }

              fini_2level_dtable ( &data );

            }  /* end of loop on real / imag */

            fini_2level_ztable ( &pgg_disc );

            /***********************************************************/
            /***********************************************************/

            /***********************************************************
             *
             * PTA PGG
             *   all-to-all 3pt function analysis
             *
             ***********************************************************/

            double _Complex *** pgg_disc_pta = init_3level_ztable ( num_conf, num_src_per_conf, T_global );
            if ( pgg_disc_pta == NULL ) {
              fprintf(stderr, "[p2gg_exdefl_uwerr_analyse] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__);
              EXIT(15);
            }

            /***********************************************************
             * loop on configurations
             ***********************************************************/
            for ( int iconf = 0; iconf < num_conf; iconf++ ) {

#ifdef HAVE_LHPC_AFF
              /***********************************************************
               * reader for aff input file
               ***********************************************************/
              struct AffReader_s *affr = NULL;
              sprintf ( filename, "%s.pref_%d_%d_%d.%.4d.nev%d.aff", infile_prefix,
                  g_sink_momentum_list[ipsnk][0], g_sink_momentum_list[ipsnk][1], g_sink_momentum_list[ipsnk][2],
                  conf_src_list[iconf][0][0], evecs_num );

              if ( g_verbose > 2 ) fprintf(stdout, "# [p2gg_exdefl_uwerr_analyse] reading data from file %s\n", filename);
              affr = aff_reader ( filename );
              const char * aff_status_str = aff_reader_errstr ( affr );
              if( aff_status_str != NULL ) {
                fprintf(stderr, "[p2gg_exdefl_uwerr_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
                EXIT(15);
              }

              /* set root node */
              struct AffNode_s * affrn = aff_reader_root( affr );
              if( affrn == NULL ) {
                fprintf(stderr, "[p2gg_exdefl_uwerr_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
                EXIT(17);
              }

              for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

                sprintf( key, "/pgg/disc/orbit/g%d/px%d_py%d_pz%d/qx%d_qy%d_qz%d/nev%d/dt%d/t%d_x%d_y%d_z%d", g_source_gamma_id_list[igsrc],
                    g_source_momentum_list[ipsrc][0], g_source_momentum_list[ipsrc][1], g_source_momentum_list[ipsrc][2],
                    g_sink_momentum_list[ipsnk][0], g_sink_momentum_list[ipsnk][1], g_sink_momentum_list[ipsnk][2],
                    ievecs, g_sequential_source_timeslice_list[idt],
                    conf_src_list[iconf][isrc][1], conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][3], conf_src_list[iconf][isrc][4] );

                if ( g_verbose > 0 ) fprintf ( stdout, "# [p2gg_exdefl_uwerr_analyse] reading key %s %s %d\n", key , __FILE__, __LINE__ );

                struct AffNode_s * affdir = aff_reader_chpath ( affr, affrn, key );
                if ( affdir == NULL ) {
                  fprintf ( stderr, "[p2gg_exdefl_uwerr_analyse] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
                  EXIT(15);
                }

                uint32_t uitems = T_global;
                exitstatus = aff_node_get_complex ( affr, affdir, pgg_disc_pta[iconf][isrc], uitems );
                if(exitstatus != 0) {
                  fprintf ( stderr, "[p2gg_exdefl_uwerr_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(16);
                }

              }  /* end of loop on source locations */

              aff_reader_close ( affr );
#else
#error "[p2gg_exdefl_uwerr_analyse] need lhp-aff lib; currently no other input method implemented"
#endif
            }  /* end of loop on configurations */

            /***********************************************************
             * UWerr analysis
             ***********************************************************/
            for ( int ireim = 0; ireim < 2; ireim++ ) {

              double *** data = init_3level_dtable ( num_conf, num_src_per_conf, T_global );
              if( data == NULL ) {
                fprintf ( stderr, "[p2gg_exdefl_uwerr_analyse] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
                EXIT(16);
              }


#pragma omp parallel for
              for ( int iconf = 0; iconf < num_conf; iconf++ ) {
                for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                  for ( int it = 0; it < T_global; it++ ) {
                    data[iconf][isrc][it] = ( ireim == 0 ) ?  creal ( pgg_disc_pta[iconf][isrc][it] ) : cimag ( pgg_disc_pta[iconf][isrc][it] );
                  }
                }
              }

              char obs_name[100];
              sprintf ( obs_name, "pgg_disc.lm.pta.orbit.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.nev%d.%s", 
                  g_source_momentum_list[ipsrc][0], g_source_momentum_list[ipsrc][1], g_source_momentum_list[ipsrc][2],
                  g_source_gamma_id_list[igsrc], g_sequential_source_timeslice_list[idt],
                  g_sink_momentum_list[ipsnk][0], g_sink_momentum_list[ipsnk][1], g_sink_momentum_list[ipsnk][2],
                  ievecs, reim_str[ireim] );

              if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_exdefl_uwerr_analyse] obs_name = %s %s %d\n", obs_name, __FILE__, __LINE__ );

              /* apply UWerr analysis */
              exitstatus = apply_uwerr_real ( data[0][0], num_conf * num_src_per_conf, T_global, 0, 1, obs_name );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[p2gg_exdefl_uwerr_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(1);
              }

              fini_3level_dtable ( &data );

            }  /* end of loop on real / imag */

            fini_3level_ztable ( &pgg_disc_pta );

          }  /* end of loop on evecs number */
        }  /* end of loop on source - sink time separations */
      }  /* end of loop on sink / reference momenta */
    }  /* end of loop on source / reference momena */
  }  /* end of loop on source gamma ids */

#if 0
#endif  /* of if 0  */

  /**********************************************************
   * free the allocated memory, finalize
   **********************************************************/

  fini_3level_itable ( &conf_src_list );

  free_geometry();

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif


#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_exdefl_uwerr_analyse] %s# [p2gg_exdefl_uwerr_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_exdefl_uwerr_analyse] %s# [p2gg_exdefl_uwerr_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
