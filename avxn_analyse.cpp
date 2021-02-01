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
#include "gamma.h"
#include "uwerr.h"
#include "derived_quantities.h"

#ifndef _SQR
#define _SQR(_a) ((_a)*(_a))
#endif

#define _TWOP_CYD 0
#define _TWOP_AFF 0
#define _TWOP_H5  1
#define _TWOP_STATS 1

#define _LOOP_ANALYSIS 1

#define _RAT_METHOD       1
#define _RAT_SUB_METHOD   1
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
inline void write_data_real ( double ** data, char * filename, int *** lst, unsigned int const n0, unsigned int const n1 ) {

  FILE * ofs = fopen ( filename, "w" );
  if ( ofs == NULL ) {
    fprintf ( stderr, "[write_data_real] Error from fopen %s %d\n",  __FILE__, __LINE__ );
    EXIT(1);
  }

  for ( unsigned int i0 = 0; i0 < n0; i0++ ) {
    fprintf ( ofs, "# %c %6d\n", lst[i0][0][0], lst[i0][0][1] );
    for ( unsigned int i1 = 0; i1 < n1; i1++ ) {
      fprintf ( ofs, "%25.16e\n", data[i0][i1] );
    }
  }

  fclose ( ofs );
}  /* end of write_data_real */


/**********************************************************
 *
 **********************************************************/
inline void write_data_real2_reim ( double **** data, char * filename, int *** lst, unsigned int const n0, unsigned int const n1, unsigned int const n2, int const ri ) {

  FILE * ofs = fopen ( filename, "w" );
  if ( ofs == NULL ) {
    fprintf ( stderr, "[write_data_real2_reim] Error from fopen %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  for ( unsigned int i0 = 0; i0 < n0; i0++ ) {
  for ( unsigned int i1 = 0; i1 < n1; i1++ ) {
    fprintf ( ofs , "# %c %6d %3d %3d %3d %3d\n", lst[i0][i1][0], lst[i0][i1][1], lst[i0][i1][2], lst[i0][i1][3], lst[i0][i1][4], lst[i0][i1][5] );

    for ( unsigned int i2 = 0; i2 < n2; i2++ ) {
      fprintf ( ofs, "%25.16e\n", data[i0][i1][i2][ri] );
    }
  }}
  fclose ( ofs );
}  /* end of write_data_real2_reim */

/**********************************************************
 *
 **********************************************************/
inline void src_avg_real2_reim ( double ** data, double ****corr, unsigned int const n0, unsigned int const n1, unsigned int const n2, int const ri ) {

#pragma omp parallel for
  for ( unsigned int iconf = 0; iconf < n0; iconf++ ) {
    for ( unsigned int it = 0; it < n2; it++ ) {
      double dtmp = 0.;

      for ( unsigned int isrc = 0; isrc < n1; isrc++ ) {
        dtmp += corr[iconf][isrc][it][ri];
      }
      data[iconf][it] = dtmp / (double)n1;
    }
  }
}  /* end of src_avg_real2_reim */

/**********************************************************
 *
 **********************************************************/
int main(int argc, char **argv) {
  
  /* int const gamma_id_to_bin[16] = { 8, 1, 2, 4, 0, 15, 7, 14, 13, 11, 9, 10, 12, 3, 5, 6 }; */

  char const reim_str[2][3] = { "re", "im" };

  char const correlator_prefix[2][20] = { "local-local" , "charged"};

  char const flavor_tag[2][20]        = { "d-gf-u-gi" , "u-gf-d-gi" };

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
  int loop_num_evecs = -1;
  int loop_nstoch = 0;
  int loop_block_size = 1;
  int loop_use_es = 0;
  int write_data = 0;
  double loop_norm = 1.;
  int operator_type = 0;

  char loop_type[10] = "LpsDw";
  int loop_transpose = 1;

  struct timeval ta, tb, start_time, end_time;

  double twop_weight[2]   = {0., 0.};
  double fbwd_weight[2]   = {0., 0.};
  double mirror_weight[2] = {0., 0.};


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:N:S:F:E:v:n:u:w:t:b:m:l:O:T:B:M:")) != -1) {
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
      fprintf ( stdout, "# [avxn_analyse] twop fold_propagator set to %d\n", twop_fold_propagator );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [avxn_analyse] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'O':
      operator_type = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] operator_type set to %d\n", operator_type );
      break;
    case 'v':
      loop_num_evecs = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] loop_num_evecs set to %d\n", loop_num_evecs );
      break;
    case 'n':
      loop_nstoch = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] loop_nstoch set to %d\n", loop_nstoch );
      break;
    case 'u':
      loop_use_es = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] loop_use_es set to %d\n", loop_use_es );
      break;
    case 't':
      loop_transpose = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] loop_transpose %d\n", loop_transpose );
      break;
    case 'w':
      write_data = atoi ( optarg );
      fprintf ( stdout, "# [avxn_analyse] write_date set to %d\n", write_data );
      break;
    case 'm':
      loop_norm = atof ( optarg );
      fprintf ( stdout, "# [avxn_analyse] loop_norm set to %e\n", loop_norm );
      break;
    case 'b':
      loop_block_size = atof ( optarg );
      fprintf ( stdout, "# [avxn_analyse] loop_block_size set to %d\n", loop_block_size );
      break;
    case 'l':
      strcpy ( loop_type, optarg );
      fprintf ( stdout, "# [avxn_analyse] loop_type set to %s\n", loop_type );
      break;
    case 'T':
      sscanf( optarg, "%lf,%lf", twop_weight, twop_weight+1 );
      fprintf ( stdout, "# [avxn_analyse] twop_weight set to %25.16e / %25.16e\n", twop_weight[0], twop_weight[1] );
      break;
    case 'B':
      sscanf( optarg, "%lf,%lf", fbwd_weight, fbwd_weight+1 );
      fprintf ( stdout, "# [avxn_analyse] fbwd_weight set to %25.16e / %25.16e\n", fbwd_weight[0], fbwd_weight[1] );
      break;
    case 'M':
      sscanf( optarg, "%lf,%lf", mirror_weight, mirror_weight+1 );
      fprintf ( stdout, "# [avxn_analyse] mirror_weight set to %25.16e / %25.16e\n", mirror_weight[0], mirror_weight[1] );
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
  /* sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf); */
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[avxn_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
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

  /**********************************************************
   * gamma matrices
   **********************************************************/
  init_gamma_matrix ();
 
  gamma_matrix_type gamma_mu[4];

  gamma_matrix_ukqcd_binary ( &(gamma_mu[0]), 1 ); /* gamma_x */
  gamma_matrix_ukqcd_binary ( &(gamma_mu[1]), 2 ); /* gamma_y */
  gamma_matrix_ukqcd_binary ( &(gamma_mu[2]), 4 ); /* gamma_z */
  gamma_matrix_ukqcd_binary ( &(gamma_mu[3]), 8 ); /* gamma_t */

  if ( g_verbose > 1 ) {
    gamma_matrix_printf ( &(gamma_mu[0]), "gamma_x", stdout );
    gamma_matrix_printf ( &(gamma_mu[1]), "gamma_y", stdout );
    gamma_matrix_printf ( &(gamma_mu[2]), "gamma_z", stdout );
    gamma_matrix_printf ( &(gamma_mu[3]), "gamma_t", stdout );
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
  double ****** twop = NULL;

  twop = init_6level_dtable ( g_sink_momentum_number, num_conf, num_src_per_conf, 2, T_global, 2 );
  if( twop == NULL ) {
    fprintf ( stderr, "[avxn_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
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
          fprintf ( stderr, "[avxn_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT (24);
        } else {
          if ( g_verbose > 1 ) fprintf ( stdout, "# [avxn_analyse] reading data from file %s filename \n", filename );
        }
        fflush ( stdout );
        fflush ( stderr );

        for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
          char line[400];

          for ( int it = -1; it < T_global; it++ ) {
            if ( fgets ( line, 100, dfs) == NULL ) {
              fprintf ( stderr, "[avxn_analyse] Error from fgets, expecting line input for it %3d conf %3d src %3d filename %s %s %d\n", 
                  it, iconf, isrc, filename, __FILE__, __LINE__ );
              EXIT (26);
            } 
          
            if ( line[0] == '#' &&  it == -1 ) {
              if ( g_verbose > 1 ) fprintf ( stdout, "# [avxn_analyse] reading key %s\n", line );
              continue;
            } /* else {
              fprintf ( stderr, "[avxn_analyse] Error in layout of file %s %s %d\n", filename, __FILE__, __LINE__ );
              EXIT(27);
            }
              */
            sscanf ( line, "%lf %lf\n", twop[isink_momentum][iconf][isrc][imeson][it], twop[isink_momentum][iconf][isrc][imeson][it]+1 );
         
            if ( g_verbose > 4 ) fprintf ( stdout, "%d %25.16e %25.16e\n" , it, twop[isink_momentum][iconf][isrc][imeson][it][0],
                twop[isink_momentum][iconf][isrc][imeson][it][1] );
          }

        }
        fclose ( dfs );

        /**********************************************************
         * write source-averaged correlator to ascii file
         **********************************************************/
        if ( write_data == 2 ) {
          for ( int ireim = 0; ireim < 2; ireim++ ) {
            double ** data = init_2level_dtable ( num_conf, T_global );

#pragma omp parallel for
           for ( int iconf = 0; iconf < num_conf; iconf++ ) {
             for ( int it = 0; it < T_global; it++ ) {
               double dtmp = 0.;
               for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                  dtmp += twop[isink_momentum][iconf][isrc][imeson][it][ireim];
                }
                data[iconf][it] = dtmp / (double)num_src_per_conf;
              }
            }

            sprintf( filename, "twop.pseudoscalar.%d.PX%d_PY%d_PZ%d.%s.corr", imeson+1,
              ( 1 - 2 * imeson) * g_sink_momentum_list[isink_momentum][0],
              ( 1 - 2 * imeson) * g_sink_momentum_list[isink_momentum][1],
              ( 1 - 2 * imeson) * g_sink_momentum_list[isink_momentum][2], reim_str[ireim] );

            write_data_real ( data, filename, conf_src_list, num_conf, T_global );
            fini_2level_dtable ( &data );
          }  /* end of loop on reim */
        }  /* end of if write_data == 2 */


      }  /* end of loop on meson type */
    }
  }  /* end of loop on sink momenta */
#endif  /* end of _TWOP_CYD */

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
          fprintf(stderr, "[avxn_analyse] Error from aff_reader for filename %s, status was %s %s %d\n", data_filename, aff_status_str, __FILE__, __LINE__);
          EXIT(15);
        } else {
          if ( g_verbose > 1 ) fprintf(stdout, "# [avxn_analyse] reading data from file %s\n", data_filename);
        }
  
        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[avxn_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
          EXIT(103);
        }
  
        double ** buffer = init_2level_dtable ( T_global, 2 );
        if( buffer == NULL ) {
          fprintf(stderr, "[avxn_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
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

            if ( g_verbose > 2 ) fprintf ( stdout, "# [avxn_analyse] key = %s\n", key );
  
            affdir = aff_reader_chpath (affr, affn, key );
            if( affdir == NULL ) {
              fprintf(stderr, "[avxn_analyse] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
              EXIT(105);
            }
  
            uint32_t uitems = T_global;
            int texitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)buffer[0], uitems );
            if( texitstatus != 0 ) {
              fprintf(stderr, "[avxn_analyse] Error from aff_node_get_complex, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
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
    show_time ( &ta, &tb, "avxn_analyse", "read-twop-aff", g_cart_id == 0 );

#endif  /* of if _TWOP_AFF */

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


      sprintf( data_filename, "%s/corr.%.4d.t%d.h5",
          filename_prefix,
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
        show_time ( &ta, &tb, "avxn_analyse", "read-twop-h5", g_cart_id == 0 );

      }  /* end of loop on sink momenta */

    }  /* end of loop on sources */

    fini_2level_dtable ( &buffer );
  }  /* end of loop on configs */
          
#endif  /* end of _TWOP_H5 */


  /**********************************************************
   * average 2-pt over momentum orbit
   **********************************************************/

  double **** twop_orbit = init_4level_dtable ( num_conf, num_src_per_conf, T_global, 2 );
  if( twop_orbit == NULL ) {
    fprintf ( stderr, "[avxn_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT (24);
  }

#pragma omp parallel for
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      /* averaging starts here */
      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

        /* double const source_phase = -2. * M_PI * ( 
            g_sink_momentum_list[imom][0] * conf_src_list[iconf][isrc][3] / (double)LX_global + 
            g_sink_momentum_list[imom][1] * conf_src_list[iconf][isrc][4] / (double)LY_global + 
            g_sink_momentum_list[imom][2] * conf_src_list[iconf][isrc][5] / (double)LZ_global ); */

        /* double const ephase[2] = { cos ( source_phase ), sin ( source_phase ) }; */

        for ( int it = 0; it < T_global; it++ ) {
          double const a[2] = { twop[imom][iconf][isrc][0][it][0] , twop[imom][iconf][isrc][0][it][1] };

          double const b[2] = { twop[imom][iconf][isrc][1][it][0] , twop[imom][iconf][isrc][1][it][1] };
             
          /* double const cre = ( a[0] + b[0] ) * ephase[0] - ( a[1] - b[1] ) * ephase[1];
          double const cim = ( a[1] + b[1] ) * ephase[0] + ( a[0] - b[0] ) * ephase[1];
          */
          double const cre = ( a[0] + b[0] ) * 0.5; 
          double const cim = ( a[1] + b[1] ) * 0.5;

          twop_orbit[iconf][isrc][it][0] += cre;
          twop_orbit[iconf][isrc][it][1] += cim;

        }  /* end of loop on it */
      }  /* end of loop on imom */

      /* multiply norm from averages over momentum orbit and source locations */
      double const norm = 1. / (double)g_sink_momentum_number;
      for ( int it = 0; it < 2*T_global; it++ ) {
        twop_orbit[iconf][isrc][0][it] *= norm;
      }
    }  /* end of loop on isrc */
  }  /* end of loop on iconf */

#if _TWOP_CYD || _TWOP_AFF || _TWOP_H5 
  /**********************************************************
   * write orbit-averaged data to ascii file, per source
   **********************************************************/
  if ( write_data == 1 ) {
    for ( int ireim = 0; ireim <=0; ireim++ ) {
      sprintf ( filename, "twop.orbit.gf%d.gi%d.PX%d_PY%d_PZ%d.%s.corr", 
          g_sink_gamma_id_list[0],
          g_source_gamma_id_list[0],
          g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2],
          reim_str[ireim]);

      write_data_real2_reim ( twop_orbit, filename, conf_src_list, num_conf, num_src_per_conf, T_global, ireim );
    }
  }  /* end of if write data */
#endif  /* of if any twop read-in */

#if _TWOP_STATS
  /**********************************************************
   * 
   * STATISTICAL ANALYSIS of orbit-averaged twop
   * 
   **********************************************************/
  for ( int ireim = 0; ireim < 1; ireim++ )  /* real part only */
  {

    if ( num_conf < 6 ) {
      fprintf ( stderr, "[avxn_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

    double ** data = init_2level_dtable ( num_conf, T_global );
    if ( data == NULL ) {
      fprintf ( stderr, "[avxn_analyse] Error from init_Xlevel_dtable %s %d\n",  __FILE__, __LINE__ );
      EXIT(1);
    }

    /* fill data array */
    if ( twop_fold_propagator != 0 ) {
#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {

        for ( int it = 0; it <= T_global/2; it++ ) {
            int const itt = ( T_global - it ) % T_global;
            data[iconf][it ] = 0.;
            data[iconf][itt] = 0.;
          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              data[iconf][it ] += 0.5 * ( twop_orbit[iconf][isrc][it][ireim] + twop_fold_propagator * twop_orbit[iconf][isrc][itt][ireim] );
          } 
          data[iconf][it ] /= (double)num_src_per_conf;
          data[iconf][itt] = data[iconf][it];
         }
      }
    } else {
#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {

        for ( int it = 0; it < T_global; it++ ) {
          data[iconf][it] = 0.;
          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            data[iconf][it] += twop_orbit[iconf][isrc][it][ireim];
          }
          data[iconf][it] /= (double)num_src_per_conf;
        }
      }
    }

    char obs_name[400];
    sprintf( obs_name, "twop.orbit.gf%d.gi%d.PX%d_PY%d_PZ%d.%s",
          g_sink_gamma_id_list[0],
          g_source_gamma_id_list[0],
          g_sink_momentum_list[0][0],
          g_sink_momentum_list[0][1],
          g_sink_momentum_list[0][2], reim_str[ireim] );

    /* apply UWerr analysis */
    exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    }

    /**********************************************************
     * write data to ascii file
     **********************************************************/
    if ( write_data == 2 ) {
      sprintf ( filename, "%s.corr", obs_name );

      write_data_real ( data, filename, conf_src_list, num_conf, T_global );
      
    }  /* end of if write data */


    /**********************************************************
     * acosh ratio for m_eff
     **********************************************************/
    int const Thp1 = T_global / 2 + 1;
    for ( int itau = 1; itau < Thp1/2; itau++ ) {
      int narg = 3;
      int arg_first[3] = { 0, 2 * itau, itau };
      int arg_stride[3] = {1,1,1};
      int nT = Thp1 - 2 * itau;

      char obs_name2[500];
      sprintf ( obs_name2, "%s.acosh_ratio.tau%d", obs_name, itau );

      exitstatus = apply_uwerr_func ( data[0], num_conf, T_global, nT, narg, arg_first, arg_stride, obs_name2, acosh_ratio, dacosh_ratio );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(115);
      }
    }

    /* fini_3level_dtable ( &data ); */
    fini_2level_dtable ( &data );
  }  /* end of loop on reim */


  /**********************************************************
   * 
   * STATISTICAL ANALYSIS or ogiginal twop
   * 
   **********************************************************/
  for ( int ireim = 0; ireim <= 1; ireim++ )  /* real part only */
  {

    if ( num_conf < 6 ) {
      fprintf ( stderr, "[avxn_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
      for ( int iflavor = 0; iflavor < 2; iflavor++ ) {

        double ** data = init_2level_dtable ( num_conf, T_global );
        if ( data == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from init_Xlevel_dtable %s %d\n",  __FILE__, __LINE__ );
          EXIT(1);
        }

    /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = 0.;
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              data[iconf][it] += twop[imom][iconf][isrc][iflavor][it][ireim];
            }
            data[iconf][it] /= (double)num_src_per_conf;
          }
        }

        char obs_name[400];
        sprintf( obs_name, "twop.%s.gf%d.gi%d.PX%d_PY%d_PZ%d.%s", flavor_tag[iflavor],
              g_sink_gamma_id_list[0],
              g_source_gamma_id_list[0],
              g_sink_momentum_list[imom][0],
              g_sink_momentum_list[imom][1],
              g_sink_momentum_list[imom][2], reim_str[ireim] );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        fini_2level_dtable ( &data );

      }  /* end of loop on flavor */
    }  /* end of loop on momenta */
  }  /* end of loop on reim */

#endif  /* of ifdef _TWOP_STATS */

#if _LOOP_ANALYSIS
  /**********************************************************
   *
   * loop fields
   *
   **********************************************************/
  double ****** loop = NULL;
  double ****** loop_exact = NULL;
  double ****** loop_sub = NULL;
  double ****** loop_sym = NULL;

  loop = init_6level_dtable ( g_insertion_momentum_number, num_conf, 4, 4, T_global, 2 );
  if ( loop == NULL ) {
    fprintf ( stderr, "[avxn_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(25);
  }

  loop_exact = init_6level_dtable ( g_insertion_momentum_number, num_conf, 4, 4, T_global, 2 );
  if ( loop_exact == NULL ) {
    fprintf ( stdout, "[avxn_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(25);
  }

  /**********************************************************
   *
   * read stochastic hp loop data
   *
   **********************************************************/
  if ( loop_nstoch > 0 && ( loop_use_es == 1 || loop_use_es == 3 ) ) {

    double const loop_norm_stoch = loop_norm / loop_nstoch;
    fprintf ( stdout, "# [avx_analyse] loop_norm_stoch = %e  %s %d\n", loop_norm_stoch, __FILE__ , __LINE__ );

    for ( int imom = 0; imom < g_insertion_momentum_number; imom++ ) {
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int idir = 0; idir < 4; idir++ ) {
  
          double _Complex *** zloop_buffer = init_3level_ztable ( T_global, 4, 4 );
  
          /* sprintf ( filename, "stream_%c/%s/loop.%.4d.stoch.%s.nev%d.Nstoch%d.mu%d.PX%d_PY%d_PZ%d", conf_src_list[iconf][0][0], filename_prefix2, conf_src_list[iconf][0][1],
              loop_type,
              loop_num_evecs,
              loop_nstoch,
              idir,
              g_insertion_momentum_list[imom][0],
              g_insertion_momentum_list[imom][1],
              g_insertion_momentum_list[imom][2] );
	      */

          sprintf ( filename, "%s/loop.%.4d.stoch.%s.nev%d.Nstoch%d.mu%d.PX%d_PY%d_PZ%d", filename_prefix2, conf_src_list[iconf][0][1],
              loop_type,
              loop_num_evecs,
              loop_nstoch,
              idir,
              g_insertion_momentum_list[imom][0],
              g_insertion_momentum_list[imom][1],
              g_insertion_momentum_list[imom][2] );
  
          if ( g_verbose > 1 ) fprintf ( stdout, "# [avxn_analyse] reading data from file %s\n", filename );

          FILE * dfs = fopen ( filename, "r" );
          if( dfs == NULL ) {
            fprintf ( stderr, "[avxn_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
            EXIT (24);
          }
  
          for ( int it = 0; it < T_global; it++ ) {
            int itmp[3];
            double dtmp[2];
            for ( int ia = 0; ia < 4; ia++ ) {
            for ( int ib = 0; ib < 4; ib++ ) {
              if ( fscanf ( dfs, "%d %d %d %lf %lf\n", itmp, itmp+1, itmp+2, dtmp, dtmp+1 ) != 5 ) {
                fprintf ( stderr, "[avxn_analyse] Error from fscanf for filename %s %s %d\n", filename, __FILE__, __LINE__ );
                EXIT (24);
              }
              zloop_buffer[it][ia][ib] = dtmp[0] + dtmp[1] * I;
           
              if ( g_verbose > 4 ) fprintf (stdout,"loop %3d %3d %3d  %25.16e %25.16e\n", 
                  itmp[0], itmp[1], itmp[2], creal( zloop_buffer[it][ia][ib]), cimag( zloop_buffer[it][ia][ib]) );
            }}
          }
          fclose ( dfs );
  
          for ( int istoch = 0; istoch < loop_nstoch / loop_block_size; istoch++ ) {
            if ( g_verbose > 1 ) fprintf ( stdout, "# [avxn_analyse] reading data for istoch %d  %s %d\n", istoch, __FILE__, __LINE__ );

#pragma omp parallel for
            for ( int imu = 0; imu < 4; imu++ ) {
              for ( int it = 0; it < T_global; it++ ) {
  
                double _Complex ztmp = 0.;
                for ( int ia = 0; ia < 4; ia++ ) {
                for ( int ib = 0; ib < 4; ib++ ) {
                  ztmp += loop_transpose ?  zloop_buffer[it][ib][ia] * gamma_mu[imu].m[ib][ia] : zloop_buffer[it][ia][ib] * gamma_mu[imu].m[ib][ia];
                }}

                /**********************************************************
                 * factor 0.5 from using doublet vs wanted single flavor
                 *
                 * WHERE DID THAT COME FROM ???
                 **********************************************************/
                loop[imom][iconf][imu][idir][it][0] += creal ( ztmp * loop_norm_stoch );
                loop[imom][iconf][imu][idir][it][1] += cimag ( ztmp * loop_norm_stoch );
              }
            }  /* end of loop on mu */

	  }  /* end of loop on samples */
  
          fini_3level_ztable ( &zloop_buffer );
        }  /* end of loop on directions */
      }  /* end of loop on configs */
    }  /* end of loop on insertion momenta */

  }  /* end of if loop_nstoch > 0 */


  /**********************************************************
   *
   * read exact lm loop data
   *
   **********************************************************/
  if ( loop_num_evecs > 0 && ( loop_use_es == 2 || loop_use_es == 3 ) ) {

    double const loop_norm_exact = loop_norm;

    for ( int imom = 0; imom < g_insertion_momentum_number; imom++ ) {
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int idir = 0; idir < 4; idir++ ) {
  
          double _Complex *** zloop_buffer = init_3level_ztable ( T_global, 4, 4 );
          
          /* sprintf ( filename, "stream_%c/%s/loop.%.4d.exact.%s.nev%d.mu%d.PX%d_PY%d_PZ%d", conf_src_list[iconf][0][0], filename_prefix2, conf_src_list[iconf][0][1],
              loop_type,
              loop_num_evecs,
              idir,
              g_insertion_momentum_list[imom][0],
              g_insertion_momentum_list[imom][1],
              g_insertion_momentum_list[imom][2] );
	      */
          sprintf ( filename, "%s/loop.%.4d.exact.%s.nev%d.mu%d.PX%d_PY%d_PZ%d", filename_prefix2, conf_src_list[iconf][0][1],
              loop_type,
              loop_num_evecs,
              idir,
              g_insertion_momentum_list[imom][0],
              g_insertion_momentum_list[imom][1],
              g_insertion_momentum_list[imom][2] );
  
          if ( g_verbose > 1 ) fprintf ( stdout, "# [avxn_analyse] reading data from file %s %s %d\n", filename, __FILE__, __LINE__ );
          FILE * dfs = fopen ( filename, "r" );
          if( dfs == NULL ) {
            fprintf ( stderr, "[avxn_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
            EXIT (24);
          }
   
          for ( int it = 0; it < T_global; it++ ) {
            int itmp[3];
            double dtmp[2];
            for ( int ia = 0; ia < 4; ia++ ) {
            for ( int ib = 0; ib < 4; ib++ ) {
              if ( fscanf ( dfs, "%d %d %d %lf %lf\n", itmp, itmp+1, itmp+2, dtmp, dtmp+1 ) != 5 ) {
                fprintf ( stderr, "[avxn_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
                EXIT (24);
              }
              zloop_buffer[it][ia][ib] = dtmp[0] + dtmp[1] * I;
           
              if ( g_verbose > 4 ) fprintf (stdout,"loop %3d %3d %3d  %25.16e %25.16e\n", 
                  itmp[0], itmp[1], itmp[2], creal( zloop_buffer[it][ia][ib]), cimag( zloop_buffer[it][ia][ib]) );
            }}
          }
  
#pragma omp parallel for
          for ( int imu = 0; imu < 4; imu++ ) {
            for ( int it = 0; it < T_global; it++ ) {
  
              double _Complex ztmp = 0.;
              for ( int ia = 0; ia < 4; ia++ ) {
              for ( int ib = 0; ib < 4; ib++ ) {
                ztmp += loop_transpose ?  zloop_buffer[it][ib][ia] * gamma_mu[imu].m[ib][ia] : zloop_buffer[it][ia][ib] * gamma_mu[imu].m[ib][ia];
              }}
              /**********************************************************
               * factor 0.5 from using doublet vs wanted single flavor
               *
               * AGAIN, WHERE DID THAT COME FROM ???
               **********************************************************/
              loop_exact[imom][iconf][imu][idir][it][0] += creal ( ztmp * loop_norm_exact );
              loop_exact[imom][iconf][imu][idir][it][1] += cimag ( ztmp * loop_norm_exact  );
            }

          }  /* end of loop on mu */

	  fclose ( dfs );
  
          fini_3level_ztable ( &zloop_buffer );
        }  /* end of loop on directions */
      }  /* end of loop on configs */
    }  /* end of loop on insertion momenta */
  }  /* end of if loop_nstoch > 0 */

  /**********************************************************
   *
   * read stochastic volsrc loop data
   *
   **********************************************************/
  if ( loop_nstoch > 0 && loop_use_es == 4 ) {
    for ( int imom = 0; imom < g_insertion_momentum_number; imom++ ) {
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int idir = 0; idir < 4; idir++ ) {
  
          double _Complex *** zloop_buffer = init_3level_ztable ( T_global, 4, 4 );
  
          if (loop_num_evecs >= 0 ) {
            sprintf ( filename, "loops_%c/%s/loop.%.4d.stoch.%s.nev%d.mu%d.PX%d_PY%d_PZ%d", conf_src_list[iconf][0][0], filename_prefix2, conf_src_list[iconf][0][1],
                loop_type, loop_num_evecs, idir,
                g_insertion_momentum_list[imom][0],
                g_insertion_momentum_list[imom][1],
                g_insertion_momentum_list[imom][2] );
          } else {
            sprintf ( filename, "loops_%c/%s/loop.%d.stoch.%s.mu%d.PX%d_PY%d_PZ%d", conf_src_list[iconf][0][0], filename_prefix2, conf_src_list[iconf][0][1],
                loop_type, idir,
                g_insertion_momentum_list[imom][0],
                g_insertion_momentum_list[imom][1],
                g_insertion_momentum_list[imom][2] );
          }
  
          if ( g_verbose > 1 ) fprintf ( stdout, "# [avxn_analyse] reading data from file %s %s %d\n", filename, __FILE__, __LINE__ );
          FILE * dfs = fopen ( filename, "r" );
          if( dfs == NULL ) {
            fprintf ( stderr, "[avxn_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
            EXIT (24);
          }
  
          for ( int istoch = 0; istoch < loop_nstoch / loop_block_size; istoch++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              int itmp[4];
              double dtmp[2];
              for ( int ia = 0; ia < 4; ia++ ) {
              for ( int ib = 0; ib < 4; ib++ ) {
                if ( fscanf ( dfs, "%d %d %d %d %lf %lf\n", itmp, itmp+1, itmp+2, itmp+3, dtmp, dtmp+1 ) != 6 ) {
                  fprintf ( stderr, "[avxn_analyse] Error from fscanf for filename %s %s %d\n", filename, __FILE__, __LINE__ );
                  EXIT (24);
                }
                zloop_buffer[it][ia][ib] += dtmp[0] + dtmp[1] * I;
              }}
            }
          }
          for ( int it = 0; it < T_global; it++ ) {
            for ( int ia = 0; ia < 4; ia++ ) {
            for ( int ib = 0; ib < 4; ib++ ) {
              zloop_buffer[it][ia][ib] /= (double)loop_nstoch;
            }}
          }

          fclose ( dfs );
  
#pragma omp parallel for
          for ( int imu = 0; imu < 4; imu++ ) {
            for ( int it = 0; it < T_global; it++ ) {
  
              double _Complex ztmp = 0.;
              for ( int ia = 0; ia < 4; ia++ ) {
              for ( int ib = 0; ib < 4; ib++ ) {
                ztmp += loop_transpose ?  zloop_buffer[it][ib][ia] * gamma_mu[imu].m[ib][ia] : zloop_buffer[it][ia][ib] * gamma_mu[imu].m[ib][ia];
              }}

              /**********************************************************
               * normalize with loop norm
               **********************************************************/
              loop[imom][iconf][imu][idir][it][0] = creal ( ztmp * loop_norm );
              loop[imom][iconf][imu][idir][it][1] = cimag ( ztmp * loop_norm );

              if ( g_verbose > 5 ) fprintf ( stdout, "loop orig p %3d %3d %3d conf %c %6d dir %d mu %d t %3d     %25.16e %25.16e\n",
                  g_insertion_momentum_list[imom][0], g_insertion_momentum_list[imom][1], g_insertion_momentum_list[imom][2],
                  conf_src_list[iconf][0][0], conf_src_list[iconf][0][1], 
                  idir, imu, it, loop[imom][iconf][imu][idir][it][0], loop[imom][iconf][imu][idir][it][1] );
            }
          }  /* end of loop on mu */
  
          fini_3level_ztable ( &zloop_buffer );
        }  /* end of loop on directions */
      }  /* end of loop on configs */
    }  /* end of loop on insertion momenta */
  }  /* end of if loop_nstoch > 0 */


  /**********************************************************
   *
   * build trace-subtracted tensor
   *
   **********************************************************/
  loop_sub = init_6level_dtable ( g_insertion_momentum_number, num_conf, 4, 4, T_global, 2 );
  if ( loop_sub == NULL ) {
    fprintf ( stdout, "[avxn_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(25);
  }

  loop_sym = init_6level_dtable ( g_insertion_momentum_number, num_conf, 4, 4, T_global, 2 );
  if ( loop_sym == NULL ) {
    fprintf ( stdout, "[avxn_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__ );
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
                + loop_exact[imom][iconf][imu][idir][it][0] + loop_exact[imom][iconf][idir][imu][it][0] 
                );
            /* imaginary part */
            loop_sym[imom][iconf][imu][idir][it][1] = 0.5 * ( 
                        loop[imom][iconf][imu][idir][it][1] +       loop[imom][iconf][idir][imu][it][1] 
                + loop_exact[imom][iconf][imu][idir][it][1] + loop_exact[imom][iconf][idir][imu][it][1] 
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
                 + loop_exact[imom][iconf][0][0][it][0]
                 + loop_exact[imom][iconf][1][1][it][0]
                 + loop_exact[imom][iconf][2][2][it][0]
                 + loop_exact[imom][iconf][3][3][it][0] 
                 );

              /* imarginary */
              loop_sub[imom][iconf][imu][idir][it][1] -= 0.25 * ( 
                         loop[imom][iconf][0][0][it][1]
                 +       loop[imom][iconf][1][1][it][1]
                 +       loop[imom][iconf][2][2][it][1]
                 +       loop[imom][iconf][3][3][it][1] 
                 + loop_exact[imom][iconf][0][0][it][1]
                 + loop_exact[imom][iconf][1][1][it][1]
                 + loop_exact[imom][iconf][2][2][it][1]
                 + loop_exact[imom][iconf][3][3][it][1] 
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
  sprintf ( loop_tag, "es%d.nev%d.Nstoch%d", loop_use_es, loop_num_evecs, loop_nstoch );

  /**********************************************************
   * write loop_sub to separate ascii file
   **********************************************************/
  if ( write_data ) {
    for ( int imom = 0; imom < g_insertion_momentum_number; imom++ ) {
      for ( int imu = 0; imu < 4; imu++ ) {
      for ( int idir = 0; idir < 4; idir++ ) {

        sprintf ( filename, "loop_sub.stoch.%s.%s.g%d_D%d.PX%d_PY%d_PZ%d.corr",
            loop_type, loop_tag, imu, idir,
            g_insertion_momentum_list[imom][0],
            g_insertion_momentum_list[imom][1],
            g_insertion_momentum_list[imom][2] );


        FILE * loop_sub_fs = fopen( filename, "w" );
        if ( loop_sub_fs == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from fopen %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        } 

        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          fprintf ( loop_sub_fs , "# %c %6d\n", conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
          for ( int it = 0; it < T_global; it++ ) {
              fprintf ( loop_sub_fs , "%25.16e %25.16e\n", loop_sub[imom][iconf][imu][idir][it][0], loop_sub[imom][iconf][imu][idir][it][1] );
            }
          }
        fclose ( loop_sub_fs );
      }}  /* end of loop on idir, imu */
    }  /* end of loop on insertion momentum */
  }  /* end of if write data */

  /**********************************************************
   *
   * STATISTICAL ANALYSIS OF LOOP VEC
   *
   **********************************************************/

  for ( int imom = 0; imom < g_insertion_momentum_number; imom++ ) {

    if ( num_conf < 6 ) {
      fprintf ( stderr, "[avxn_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
      /* EXIT(1); */
      continue;
    }

    /**********************************************************
     * analyse ORIGINAL LOOP
     **********************************************************/
    for ( int ireim = 0; ireim < 2; ireim++ ) {
      
      for ( int imu = 0; imu < 4; imu++ ) {
      for ( int idir = 0; idir < 4; idir++ ) {

        double ** data = init_2level_dtable ( num_conf, T_global );
        if ( data == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = loop[imom][iconf][imu][idir][it][ireim] + loop_exact[imom][iconf][imu][idir][it][ireim];
          }
        }

        char obs_name[400];
        sprintf ( obs_name, "loop.stoch.%s.%s.g%d_D%d.PX%d_PY%d_PZ%d.%s",
            loop_type, 
            loop_tag,
            imu, idir,
            g_insertion_momentum_list[imom][0],
            g_insertion_momentum_list[imom][1],
            g_insertion_momentum_list[imom][2], reim_str[ireim] );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        fini_2level_dtable ( &data );
      }}
    }  /* end of loop on re / im */

    /**********************************************************
     * analyse SYMMETRIZED LOOP
     **********************************************************/
    for ( int ireim = 0; ireim < 2; ireim++ ) {
      
      for ( int imu = 0; imu < 4; imu++ ) {
      for ( int idir = 0; idir < 4; idir++ ) {

        double ** data = init_2level_dtable ( num_conf, T_global );
        if ( data == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = loop_sym[imom][iconf][imu][idir][it][ireim];
          }
        }

        char obs_name[400];
        sprintf ( obs_name, "loop_sym.stoch.%s.%s.g%d_D%d.PX%d_PY%d_PZ%d.%s",
            loop_type, 
            loop_tag,
            imu, idir,
            g_insertion_momentum_list[imom][0],
            g_insertion_momentum_list[imom][1],
            g_insertion_momentum_list[imom][2], reim_str[ireim] );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        fini_2level_dtable ( &data );
      }}
    }  /* end of loop on re / im */

    /**********************************************************
     * analyse SYMMETRIZED - SUBTRACTED / TRACELESS LOOP
     **********************************************************/
    for ( int ireim = 0; ireim < 2; ireim++ ) {
      
      for ( int imu = 0; imu < 4; imu++ ) {
      for ( int idir = 0; idir < 4; idir++ ) {

        double ** data = init_2level_dtable ( num_conf, T_global );
        if ( data == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        /* fill data array */
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            data[iconf][it] = loop_sub[imom][iconf][imu][idir][it][ireim];
          }
        }

        char obs_name[400];
        sprintf ( obs_name, "loop_sub.stoch.%s.%s.g%d_D%d.PX%d_PY%d_PZ%d.%s",
            loop_type, 
            loop_tag,
            imu, idir,
            g_insertion_momentum_list[imom][0],
            g_insertion_momentum_list[imom][1],
            g_insertion_momentum_list[imom][2], reim_str[ireim] );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        fini_2level_dtable ( &data );
      }}
    }  /* end of loop on re / im */
  }  /* end of loop momenta */

  fini_6level_dtable ( &loop );
  fini_6level_dtable ( &loop_exact );


#if _RAT_METHOD
  /**********************************************************
   *
   * STATISTICAL ANALYSIS for products and ratios
   *
   * fixed source - sink separation
   *
   **********************************************************/

  /**********************************************************
   * loop on source - sink time separations
   **********************************************************/
  for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {

    double ***** threep = init_5level_dtable ( 5, num_conf, num_src_per_conf, T_global, 2 ) ;
    if ( threep == NULL ) {
      fprintf ( stderr, "[avxn_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

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

        if ( g_verbose > 4 ) fprintf ( stdout, "# [avxn_analyse] t_src %3d   dt %3d   tsink %3d tsink2 %3d\n", conf_src_list[iconf][isrc][2],
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
            int const tins_fwd_1 = (  it + conf_src_list[iconf][isrc][2]                                           + T_global ) % T_global;

            /* fwd 2 insertion time = source time + dt - it */
            int const tins_fwd_2 = ( -it + conf_src_list[iconf][isrc][2] + g_sequential_source_timeslice_list[idt] + T_global ) % T_global;

            /* bwd 1 insertion time = source time      - it */
            int const tins_bwd_1 = ( -it + conf_src_list[iconf][isrc][2]                                           + T_global ) % T_global;

            /* bwd 2 insertion time = source time - dt + it */
            int const tins_bwd_2 = (  it + conf_src_list[iconf][isrc][2] - g_sequential_source_timeslice_list[idt] + T_global ) % T_global;

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
            /* double const c44[2] = { loop_sub[0][iconf][3][3][tins][0], loop_sub[0][iconf][3][3][tins][1] };
            threep_44[iconf][isrc][it][0] += ( a_phase[0] + b_phase[0] ) * c44[0] * 0.5;
            threep_44[iconf][isrc][it][1] += 0.;
            */

            /* threep_44[iconf][isrc][it][0] += ( 
                    ( a_fwd_phase[0] + b_fwd_phase[0] ) * ( loop_sub[0][iconf][3][3][tins_fwd_1][0] + loop_sub[0][iconf][3][3][tins_fwd_2][0] ) 
                  + ( a_bwd_phase[0] + b_bwd_phase[0] ) * ( loop_sub[0][iconf][3][3][tins_bwd_1][0] + loop_sub[0][iconf][3][3][tins_bwd_2][0] ) 
                ) * 0.125;
             */


            threep[0][iconf][isrc][it][0] += threep_norm * (
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

                threep[1][iconf][isrc][it][0] += threep_norm * (
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

              threep[2][iconf][isrc][it][0] += threep_norm * (
                  /* twop */
                  twop_weight[0] * (
                        fbwd_weight[0] * a_fwd_phase[1] * ( mirror_weight[0] * loop_sub[0][iconf][3][k][tins_fwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][3][k][tins_fwd_2][0] )
                      + fbwd_weight[1] * a_bwd_phase[1] * ( mirror_weight[0] * loop_sub[0][iconf][3][k][tins_bwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][3][k][tins_bwd_2][0] )
                    )
                  /* MINUS twop parity partner */
                - twop_weight[1] * (
                        fbwd_weight[0] * b_fwd_phase[1] * ( mirror_weight[0] * loop_sub[0][iconf][3][k][tins_fwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][3][k][tins_fwd_2][0] )
                      + fbwd_weight[1] * b_bwd_phase[1] * ( mirror_weight[0] * loop_sub[0][iconf][3][k][tins_bwd_1][0] + mirror_weight[1] * loop_sub[0][iconf][3][k][tins_bwd_2][0] )
                    )
                ) * mom[k];

            }  /* end of loop on spatial momentum index */


            /**********************************************************
             * O44 with symmetrized, NOT SUBTRACTED loop
             **********************************************************/
            threep[3][iconf][isrc][it][0] += threep_norm * (
                    /* twop */
                    twop_weight[0] * (
                          fbwd_weight[0] * a_fwd_phase[0] * ( mirror_weight[0] * loop_sym[0][iconf][3][3][tins_fwd_1][0] + mirror_weight[1] * loop_sym[0][iconf][3][3][tins_fwd_2][0] )
                        + fbwd_weight[1] * a_bwd_phase[0] * ( mirror_weight[0] * loop_sym[0][iconf][3][3][tins_bwd_1][0] + mirror_weight[1] * loop_sym[0][iconf][3][3][tins_bwd_2][0] )
                      )
                    /* twop parity partner */
                  + twop_weight[1] * (
                          fbwd_weight[0] * b_fwd_phase[0] * ( mirror_weight[0] * loop_sym[0][iconf][3][3][tins_fwd_1][0] + mirror_weight[1] * loop_sym[0][iconf][3][3][tins_fwd_2][0] )
                        + fbwd_weight[1] * b_bwd_phase[0] * ( mirror_weight[0] * loop_sym[0][iconf][3][3][tins_bwd_1][0] + mirror_weight[1] * loop_sym[0][iconf][3][3][tins_bwd_2][0] )
                      )
                  );

            /**********************************************************
             * Ojj/3 with symmetrized, NOT SUBTRACTED loop
             **********************************************************/
            for ( int k = 0; k < 3; k++ ) {
              threep[4][iconf][isrc][it][0] += threep_norm * (
                    /* twop */
                    twop_weight[0] * (
                          fbwd_weight[0] * a_fwd_phase[0] * ( mirror_weight[0] * loop_sym[0][iconf][k][k][tins_fwd_1][0] + mirror_weight[1] * loop_sym[0][iconf][k][k][tins_fwd_2][0] )
                        + fbwd_weight[1] * a_bwd_phase[0] * ( mirror_weight[0] * loop_sym[0][iconf][k][k][tins_bwd_1][0] + mirror_weight[1] * loop_sym[0][iconf][k][k][tins_bwd_2][0] )
                      )
                    /* twop parity partner */
                  + twop_weight[1] * (
                          fbwd_weight[0] * b_fwd_phase[0] * ( mirror_weight[0] * loop_sym[0][iconf][k][k][tins_fwd_1][0] + mirror_weight[1] * loop_sym[0][iconf][k][k][tins_fwd_2][0] )
                        + fbwd_weight[1] * b_bwd_phase[0] * ( mirror_weight[0] * loop_sym[0][iconf][k][k][tins_bwd_1][0] + mirror_weight[1] * loop_sym[0][iconf][k][k][tins_bwd_2][0] )
                      )
                  );
            }


          }  /* end of loop on it */

        }  /* end of loop on imom */

        /**********************************************************
         * normalize
         **********************************************************/
        /* O44 simple orbit average */
        double const norm44 = 1. / g_sink_momentum_number;
        for ( int it = 0; it < 2 * T_global; it++ ) {
          threep[0][iconf][isrc][0][it] *= norm44;
        }

        /* Oik divide by (p^2) */
        double const mom[3] = {
          2 * M_PI * g_sink_momentum_list[0][0] / (double)LX_global,
          2 * M_PI * g_sink_momentum_list[0][1] / (double)LY_global,
          2 * M_PI * g_sink_momentum_list[0][2] / (double)LZ_global };

        double const mom_squared = mom[0] * mom[0] + mom[1] * mom[1] + mom[2] * mom[2];

        int const mom_is_zero = g_sink_momentum_list[0][0] == 0 && g_sink_momentum_list[0][1] == 0 && g_sink_momentum_list[0][2] == 0;

        double const normik = mom_is_zero ? 0. : 1. / mom_squared / (double)g_sink_momentum_number;

        for ( int it = 0; it < 2 * T_global; it++ ) {
          threep[1][iconf][isrc][0][it] *= normik;
        }

        /* O4k divide by (p^2) */
        double const norm4k = mom_is_zero ? 0. : 1. / mom_squared / (double)g_sink_momentum_number;

        for ( int it = 0; it < 2 * T_global; it++ ) {
          threep[2][iconf][isrc][0][it] *= norm4k;
        }

        /* O44 unsubtracted */
        for ( int it = 0; it < 2 * T_global; it++ ) {
          threep[3][iconf][isrc][0][it] *= norm44;
        }
        /* Ojj unsubtracted */
        for ( int it = 0; it < 2 * T_global; it++ ) {
          threep[4][iconf][isrc][0][it] *= norm44 / 3.;
        }

      }  /* end of loop on isrc */
    }  /* end of loop on iconf */

    /**********************************************************
     * write 3pt function to ascii file, per source
     **********************************************************/
    if ( write_data == 1) {
      /**********************************************************
       * write 3pt
       **********************************************************/
      for ( int k = 0; k < 3; k++ ) {
        for ( int ireim = 0; ireim < 1; ireim++ ) {
          sprintf ( filename, "threep.%s.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s.corr",
              loop_tag, threep_tag[k],
              g_sequential_source_timeslice_list[idt],
              g_sink_momentum_list[0][0],
              g_sink_momentum_list[0][1],
              g_sink_momentum_list[0][2], reim_str[ireim] );
 
          write_data_real2_reim ( threep[k], filename, conf_src_list, num_conf, num_src_per_conf, T_global, ireim );

        }  /* end of loop on ireim */
      }  /* end of loop on 3pt function types */
    
    }  /* end of if write_data */

    /**********************************************************
     *
     * STATISTICAL ANALYSIS for threep
     *
     * with fixed source - sink separation
     *
     **********************************************************/
      
    /**********************************************************
     * loop on 3pt function types
     **********************************************************/
    for ( int k = 0; k < 5; k++ ) {
 

      if ( num_conf < 6 ) {
        fprintf ( stderr, "[avxn_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
        /* EXIT(1); */
        continue;
      }

      for ( int ireim = 0; ireim < 1; ireim++ ) {

        double ** data = init_2level_dtable ( num_conf, T_global );
        if ( data == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        /**********************************************************
         * average threep over source positions per config
         **********************************************************/
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {

          for ( int it = 0; it < T_global; it++ ) {
            double dtmp = 0.;
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              dtmp += threep[k][iconf][isrc][it][ireim];
            }
            data[iconf][it] = dtmp / (double)num_src_per_conf;
          }
        }

        char obs_name[400];
        sprintf ( obs_name, "threep.%s.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s",
            loop_tag, threep_tag[k],
            g_sequential_source_timeslice_list[idt],
            g_sink_momentum_list[0][0],
            g_sink_momentum_list[0][1],
            g_sink_momentum_list[0][2], reim_str[ireim] );

        /* apply UWerr analysis */
        exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(1);
        }

        if ( write_data == 2 ) {
          sprintf ( filename, "%s.corr", obs_name );
          write_data_real ( data, filename, conf_src_list, num_conf, T_global );
        }

        fini_2level_dtable ( &data );

      }  /* end of loop on reim */

      /**********************************************************
       *
       * STATISTICAL ANALYSIS for ratio 
       *   with source - sink fixed
       *
       **********************************************************/
      for ( int ireim = 0; ireim < 1; ireim++ ) {

        /* UWerr parameters */
        int nT = g_sequential_source_timeslice_list[idt] + 1;
        int narg          = 2;
        int arg_first[2]  = { 0, nT };
        int arg_stride[2] = { 1,  0 };
        char obs_name[400];

        double ** data = init_2level_dtable ( num_conf, nT + 1 );
        if ( data == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        /**********************************************************
         * O44
         **********************************************************/
        src_avg_real2_reim ( data, threep[k], num_conf, num_src_per_conf, nT, ireim );

#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {

          double dtmp = 0.;

          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
  
            int const tsink  = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
            int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
            dtmp += twop_orbit[iconf][isrc][tsink][ireim] + (double)twop_fold_propagator * twop_orbit[iconf][isrc][tsink2][ireim];
          }
          data[iconf][nT] = dtmp / (double)num_src_per_conf / ( 1. + abs( twop_fold_propagator ) ) ;
        }

        sprintf ( obs_name, "ratio.%s.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s",
          loop_tag, threep_tag[k],
          g_sequential_source_timeslice_list[idt],
          g_sink_momentum_list[0][0],
          g_sink_momentum_list[0][1],
          g_sink_momentum_list[0][2], reim_str[ireim] );

        exitstatus = apply_uwerr_func ( data[0], num_conf, nT+1, nT, narg, arg_first, arg_stride, obs_name, ratio_1_1, dratio_1_1 );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(115);
        }

        fini_2level_dtable ( &data );

      }  /* end of loop on reim */

#if _RAT_SUB_METHOD
      /**********************************************************
       *
       * STATISTICAL ANALYSIS for vev-subtracted ratio
       *
       * < C_2pt x L >_U  / < C_2pt >_U - < L >_U
       *
       **********************************************************/
      for ( int ireim = 0; ireim < 1; ireim++ ) {

        /* UWerr parameters */
        int nT = g_sequential_source_timeslice_list[idt] + 1;
        int narg          = 3;
        int arg_first[3]  = { 0, nT, nT+1 };
        int arg_stride[3] = { 1,  0,    0 };
        char obs_name[400];

        double ** data = init_2level_dtable ( num_conf, nT + 2 );
        if ( data == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

        /**********************************************************
         * threep source average
         **********************************************************/
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < nT; it++ ) {
            data[iconf][it] = 0.;
            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              data[iconf][it] += threep[k][iconf][isrc][it][ireim];
            }
            data[iconf][it] /= (double)num_src_per_conf;
          }
        }

#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          double dtmp = 0.;

          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

            int const tsink  = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
            int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
            dtmp += twop_orbit[iconf][isrc][tsink][ireim] + (double)twop_fold_propagator * twop_orbit[iconf][isrc][tsink2][ireim];
          }
          data[iconf][nT] = dtmp / (double)num_src_per_conf / ( 1. + abs( twop_fold_propagator ) ) ;

          dtmp = 0.;
          for ( int it = 0; it < T_global; it++ ) {
            dtmp += loop_sub[0][iconf][3][3][it][0];
          }
          data[iconf][nT+1] = dtmp / (double)T_global;
        }

        sprintf ( obs_name, "ratio.sub.%s.%s.dtsnk%d.PX%d_PY%d_PZ%d.%s",
            loop_tag, threep_tag[k],
            g_sequential_source_timeslice_list[idt],
            g_sink_momentum_list[0][0],
            g_sink_momentum_list[0][1],
            g_sink_momentum_list[0][2], reim_str[ireim] );

        exitstatus = apply_uwerr_func ( data[0], num_conf, nT+2, nT, narg, arg_first, arg_stride, obs_name, ratio_1_2_mi_3, dratio_1_2_mi_3 );
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(115);
        }

        fini_2level_dtable ( &data );

      }  /* end of loop on reim */

    }  /*  end of loop 3pt function types */

#endif  /* of if _RAT_SUB_METHOD */

    fini_5level_dtable ( &threep );

  }  /* end of loop on dt */

#endif  /* end of _RAT_METHOD */

  /**********************************************************/
  /**********************************************************/

#if _FHT_METHOD_ALLT
  /**********************************************************
   *
   * STATISTICAL ANALYSIS for products and ratios
   *
   * FHT calculation
   *
   **********************************************************/

  {
    /* loop on sink times */
    int const Thp1 = T_global / 2 + 1;

    double *** threep_44 = init_3level_dtable ( num_conf, num_src_per_conf, Thp1 ) ;
    if ( threep_44 == NULL ) {
      fprintf ( stderr, "[avxn_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

    double *** threep_4k = init_3level_dtable ( num_conf, num_src_per_conf, Thp1 ) ;
    if ( threep_4k == NULL ) {
      fprintf ( stderr, "[avxn_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

    double *** threep_ik = init_3level_dtable ( num_conf, num_src_per_conf, Thp1 ) ;
    if ( threep_ik == NULL ) {
      fprintf ( stderr, "[avxn_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

    double ***** loop_sum = init_5level_dtable ( g_insertion_momentum_number, num_conf, 4, 4, 2 );
    if ( loop_sum == NULL ) {
      fprintf ( stderr, "[avxn_analysis] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(122);
    }

    /**********************************************************
     * sum loop over all timeslices
     **********************************************************/
    for ( int imom = 0; imom < g_insertion_momentum_number; imom++ ) {
#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int ia =0; ia < 4; ia++ ) {
        for ( int ib =0; ib < 4; ib++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            loop_sum[imom][iconf][ia][ib][0] += loop_sub[imom][iconf][ia][ib][it][0];
            loop_sum[imom][iconf][ia][ib][1] += loop_sub[imom][iconf][ia][ib][it][1];
          }
          /* normalize */
          /* loop_sum[imom][iconf][ia][ib][0] *= 1. / T_global;
          loop_sum[imom][iconf][ia][ib][1] *= 1. / T_global;
          */
        }}
      }
    }

#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

          double const mom[3] = { 
              2 * M_PI * g_sink_momentum_list[imom][0] / (double)LX_global,
              2 * M_PI * g_sink_momentum_list[imom][1] / (double)LY_global,
              2 * M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global };

          for ( int it = 0; it < Thp1; it++ ) {
            int const itt = ( T_global - it ) % T_global;

            /**********************************************************
             * O44, real parts only
             * fold propagator
             **********************************************************/
            threep_44[iconf][isrc][it] += 0.125 * ( 
                  twop[imom][iconf][isrc][0][it][0] + twop[imom][iconf][isrc][0][itt][0]
                + twop[imom][iconf][isrc][1][it][0] + twop[imom][iconf][isrc][1][itt][0] ) * loop_sum[0][iconf][3][3][0];

            /**********************************************************
             * Oik, again only real parts
             * fold propagator
             **********************************************************/
            for ( int i = 0; i < 3; i++ ) {
              for ( int k = 0; k < 3; k++ ) {
                threep_ik[iconf][isrc][it] += 0.125 * (
                    twop[imom][iconf][isrc][0][it][0] + twop[imom][iconf][isrc][0][itt][0]
                  + twop[imom][iconf][isrc][1][it][0] + twop[imom][iconf][isrc][1][itt][0] ) * loop_sum[0][iconf][i][k][0] * mom[i] * mom[k];
              }
            }

            /**********************************************************
             * O4k real part of loop, imaginary part of twop
             **********************************************************/
            for ( int k = 0; k < 3; k++ ) {
              threep_4k[iconf][isrc][it] += 0.125 * (
                  twop[imom][iconf][isrc][0][it][1] + twop[imom][iconf][isrc][0][itt][1]
                - twop[imom][iconf][isrc][1][it][1] - twop[imom][iconf][isrc][1][itt][1] ) * loop_sum[0][iconf][3][k][0] *  mom[k];
            }
          }  /* end of loop on it */
        }  /* end of loop on imom */

        /**********************************************************
         * normalize
         **********************************************************/
        /* O44 simple orbit average */
        double const norm44 = 1. / g_sink_momentum_number;
        for ( int it = 0; it < Thp1; it++ ) {
          threep_44[iconf][isrc][it] *= norm44;
        }

        /* Oik divide by (p^2)^2 */
        double const mom[3] = {
          2 * M_PI * g_sink_momentum_list[0][0] / (double)LX_global,
          2 * M_PI * g_sink_momentum_list[0][1] / (double)LY_global,
          2 * M_PI * g_sink_momentum_list[0][2] / (double)LZ_global };
        double const normik = 
          ( g_sink_momentum_list[0][0] == 0 && g_sink_momentum_list[0][1] == 0 && g_sink_momentum_list[0][2] == 0 ) ? 0. :
              1. / _SQR( ( mom[0] * mom[0] + mom[1] * mom[1] + mom[2] * mom[2] ) ) / (double)g_sink_momentum_number;
        for ( int it = 0; it < Thp1; it++ ) {
          threep_ik[iconf][isrc][it] *= normik;
        }

        /* O4k divide by (p^2) */
        double const norm4k = 
            ( g_sink_momentum_list[0][0] == 0 && g_sink_momentum_list[0][1] == 0 && g_sink_momentum_list[0][2] == 0 ) ? 0. :
            1. / ( mom[0] * mom[0] + mom[1] * mom[1] + mom[2] * mom[2] ) / (double)g_sink_momentum_number;
        for ( int it = 0; it < Thp1; it++ ) {
          threep_4k[iconf][isrc][it] *= norm4k;
        }

      }  /* end of loop on isrc */
    }  /* end of loop on iconf */
   
    fini_5level_dtable ( &loop_sum );

    /**********************************************************
     *
     * STATISTICAL ANALYSIS for threep
     *
     * with fixed source - sink separation
     *
     **********************************************************/
    double *** data = init_3level_dtable ( num_conf, num_src_per_conf, 2*Thp1 );
    if ( data == NULL ) {
      fprintf ( stderr, "[avxn_analysis] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(123);
    }
    
    dquant fptr = acosh_ratio_deriv, dfptr = dacosh_ratio_deriv;
    int narg = 6;
    int arg_stride[6] = {1,1,1,1,1,1};
    char obs_name[400];
 
    /**********************************************************
     * threep_44
     **********************************************************/
#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
    for ( int it = 0; it < Thp1; it++ ) {
      data[iconf][isrc][     it] = threep_44[iconf][isrc][it];
      data[iconf][isrc][Thp1+it] = twop_orbit[iconf][isrc][it][0];
    }}}

    for ( int itau = 1; itau < Thp1/2; itau++ ) {

      int arg_first[6] = { 2 * itau, itau , 0, Thp1 + 2 * itau , Thp1, Thp1 + itau };
      int nT = Thp1 - 2 * itau;

      sprintf ( obs_name, "threep.fht.%s.g4_D4.tau%d.PX%d_PY%d_PZ%d.%s",
          loop_tag,
          itau,
          g_sink_momentum_list[0][0],
          g_sink_momentum_list[0][1],
          g_sink_momentum_list[0][2], "re" );

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_func ( data[0][0], num_conf*num_src_per_conf, 2*Thp1, nT, narg, arg_first, arg_stride, obs_name, fptr, dfptr );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(115);
      }
    }

    /**********************************************************
     * threep_4k
     **********************************************************/
#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
    for ( int it = 0; it < Thp1; it++ ) {
      data[iconf][isrc][     it] = threep_4k[iconf][isrc][it];
      data[iconf][isrc][Thp1+it] = twop_orbit[iconf][isrc][it][0];
    }}}

    for ( int itau = 1; itau < Thp1/2; itau++ ) {

      int arg_first[6] = { 2 * itau, itau , 0, Thp1 + 2 * itau , Thp1, Thp1 + itau };
      int nT = Thp1 - 2 * itau;

      char obs_name[400];
      sprintf ( obs_name, "threep.fht.%s.g4_Dk.tau%d.PX%d_PY%d_PZ%d.%s",
          loop_tag,
          itau,
          g_sink_momentum_list[0][0],
          g_sink_momentum_list[0][1],
          g_sink_momentum_list[0][2], "re");

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_func ( data[0][0], num_conf*num_src_per_conf, 2*Thp1, nT, narg, arg_first, arg_stride, obs_name, fptr, dfptr );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(115);
      }
    }

    /**********************************************************
     * threep_ik
     **********************************************************/
#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
    for ( int it = 0; it < Thp1; it++ ) {
      data[iconf][isrc][     it] = threep_ik[iconf][isrc][it];
      data[iconf][isrc][Thp1+it] = twop_orbit[iconf][isrc][it][0];
    }}}
 
    for ( int itau = 1; itau < Thp1/2; itau++ ) {

      int arg_first[6] = { 2 * itau, itau , 0, Thp1 + 2 * itau , Thp1, Thp1 + itau };
      int nT = Thp1 - 2 * itau;

      char obs_name[400];
      sprintf ( obs_name, "threep.fht.%s.gi_Dk.tau%d.PX%d_PY%d_PZ%d.%s",
          loop_tag,
          itau,
          g_sink_momentum_list[0][0],
          g_sink_momentum_list[0][1],
          g_sink_momentum_list[0][2], "re");

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_func ( data[0][0], num_conf*num_src_per_conf, 2*Thp1, nT, narg, arg_first, arg_stride, obs_name, fptr, dfptr );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(115);
      }


    }  /* end of loop on itau */

    fini_3level_dtable ( &data );

    fini_3level_dtable ( &threep_44 );
    fini_3level_dtable ( &threep_4k );
    fini_3level_dtable ( &threep_ik );

  }  /* end FHT calculation */
#endif  /* end of if _FHT_METHOD_ALLT */

  /**********************************************************/
  /**********************************************************/

#if _FHT_METHOD_ACCUM
  /**********************************************************
   *
   * STATISTICAL ANALYSIS for products and ratios
   *
   * FHT calculation with accumulating loops
   *
   **********************************************************/

  {
    int const Thp1 = T_global / 2 + 1;
    /* loop on sink times */
    double *** threep_44 = init_3level_dtable ( num_conf, num_src_per_conf, Thp1 ) ;
    if ( threep_44 == NULL ) {
      fprintf ( stderr, "[avxn_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

    double *** threep_4k = init_3level_dtable ( num_conf, num_src_per_conf, Thp1 ) ;
    if ( threep_4k == NULL ) {
      fprintf ( stderr, "[avxn_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

    double *** threep_ik = init_3level_dtable ( num_conf, num_src_per_conf, Thp1 ) ;
    if ( threep_ik == NULL ) {
      fprintf ( stderr, "[avxn_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

        /**********************************************************
         * sum loop over all timeslices
         **********************************************************/
        double **** loop_accum_fwd = init_4level_dtable ( 4, 4, Thp1, 2 );
        double **** loop_accum_bwd = init_4level_dtable ( 4, 4, Thp1, 2 );
        if ( loop_accum_fwd == NULL || loop_accum_bwd == NULL ) {
          fprintf ( stderr, "[avxn_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }
        for ( int ia = 0; ia < 4; ia++ ) {
        for ( int ib = 0; ib < 4; ib++ ) {
          loop_accum_fwd[ia][ib][0][0] = loop_sub[0][iconf][ia][ib][conf_src_list[iconf][isrc][2]][0];
          loop_accum_fwd[ia][ib][0][1] = loop_sub[0][iconf][ia][ib][conf_src_list[iconf][isrc][2]][1];

          for ( int it = 1; it < Thp1; it++ ) {
            /* fwd case */
            int const ifwd = ( conf_src_list[iconf][isrc][2] + it + T_global ) % T_global;
            loop_accum_fwd[ia][ib][it][0] = loop_accum_fwd[ia][ib][it-1][0] + loop_sub[0][iconf][ia][ib][ifwd][0];
            loop_accum_fwd[ia][ib][it][1] = loop_accum_fwd[ia][ib][it-1][1] + loop_sub[0][iconf][ia][ib][ifwd][1];
            /* bwd case */
            int const ibwd = ( conf_src_list[iconf][isrc][2] - it + T_global ) % T_global;
            loop_accum_bwd[ia][ib][it][0] = loop_accum_bwd[ia][ib][it-1][0] + loop_sub[0][iconf][ia][ib][ibwd][0];
            loop_accum_bwd[ia][ib][it][1] = loop_accum_bwd[ia][ib][it-1][1] + loop_sub[0][iconf][ia][ib][ibwd][1];
          }
        }}

        /**********************************************************
         * loop on twopt momenta
         **********************************************************/
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

          double const mom[3] = { 
              2 * M_PI * g_sink_momentum_list[imom][0] / (double)LX_global,
              2 * M_PI * g_sink_momentum_list[imom][1] / (double)LY_global,
              2 * M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global };

          /**********************************************************
           * loop on source - sink times
           **********************************************************/
          for ( int it = 0; it < Thp1; it++ ) {
            int const itt = ( T_global - it ) % T_global;

            /**********************************************************
             * O44, real parts only
             * fold propagator
             **********************************************************/
            threep_44[iconf][isrc][it] += 0.25 * ( 
                    ( twop[imom][iconf][isrc][0][ it][0] + twop[imom][iconf][isrc][1][ it][0] ) * loop_accum_fwd[3][3][it][0]
                  + ( twop[imom][iconf][isrc][0][itt][0] + twop[imom][iconf][isrc][1][itt][0] ) * loop_accum_bwd[3][3][it][0]  );

            /**********************************************************
             * Oik, again only real parts
             * fold propagator
             **********************************************************/
            for ( int i = 0; i < 3; i++ ) {
              for ( int k = 0; k < 3; k++ ) {
                threep_ik[iconf][isrc][it] += 0.25 * (
                      ( twop[imom][iconf][isrc][0][ it][0] + twop[imom][iconf][isrc][1][ it][0] ) * loop_accum_fwd[i][k][it][0]
                    + ( twop[imom][iconf][isrc][0][itt][0] + twop[imom][iconf][isrc][1][itt][0] ) * loop_accum_bwd[i][k][it][0] ) * mom[i] * mom[k];
              }
            }

            /**********************************************************
             * O4k real part of loop, imaginary part of twop
             **********************************************************/
            for ( int k = 0; k < 3; k++ ) {
              threep_4k[iconf][isrc][it] += 0.25 * (
                    ( twop[imom][iconf][isrc][0][ it][1] - twop[imom][iconf][isrc][1][ it][1] ) * loop_accum_fwd[3][k][it][0]
                  + ( twop[imom][iconf][isrc][0][itt][1] - twop[imom][iconf][isrc][1][itt][1] ) * loop_accum_bwd[3][k][it][0] ) * mom[k];
            }
          }  /* end of loop on it */
        }  /* end of loop on imom */

        /**********************************************************
         * normalize
         **********************************************************/
        /* O44 simple orbit average */
        double const norm44 = 1. / g_sink_momentum_number;
        for ( int it = 0; it < Thp1; it++ ) {
          threep_44[iconf][isrc][it] *= norm44;
        }

        /* Oik divide by (p^2)^2 */
        double const mom[3] = {
          2 * M_PI * g_sink_momentum_list[0][0] / (double)LX_global,
          2 * M_PI * g_sink_momentum_list[0][1] / (double)LY_global,
          2 * M_PI * g_sink_momentum_list[0][2] / (double)LZ_global };
        double const normik = 
          ( g_sink_momentum_list[0][0] == 0 && g_sink_momentum_list[0][1] == 0 && g_sink_momentum_list[0][2] == 0 ) ? 0. :
              1. / _SQR( ( mom[0] * mom[0] + mom[1] * mom[1] + mom[2] * mom[2] ) ) / (double)g_sink_momentum_number;
        for ( int it = 0; it < Thp1; it++ ) {
          threep_ik[iconf][isrc][it] *= normik;
        }

        /* O4k divide by (p^2) */
        double const norm4k = 
            ( g_sink_momentum_list[0][0] == 0 && g_sink_momentum_list[0][1] == 0 && g_sink_momentum_list[0][2] == 0 ) ? 0. :
            1. / ( mom[0] * mom[0] + mom[1] * mom[1] + mom[2] * mom[2] ) / (double)g_sink_momentum_number;
        for ( int it = 0; it < Thp1; it++ ) {
          threep_4k[iconf][isrc][it] *= norm4k;
        }

        fini_4level_dtable ( &loop_accum_fwd );
        fini_4level_dtable ( &loop_accum_bwd );
      }  /* end of loop on isrc */
    }  /* end of loop on iconf */

    /**********************************************************
     *
     * STATISTICAL ANALYSIS for threep
     *
     * with fixed source - sink separation
     *
     **********************************************************/
    /* double *** data = init_3level_dtable ( num_conf, num_src_per_conf, 2*Thp1 ); */
    double ** data = init_2level_dtable ( num_conf, 2*Thp1 );
    if ( data == NULL ) {
      fprintf ( stderr, "[avxn_analysis] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(123);
    }
    
    dquant fptr = acosh_ratio_deriv, dfptr = dacosh_ratio_deriv;
    int narg = 6;
    int arg_stride[6] = {1,1,1,1,1,1};
    char obs_name[400];
 
    /**********************************************************
     * threep_44
     **********************************************************/
#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
#if 0
    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
    for ( int it = 0; it < Thp1; it++ ) {
      data[iconf][isrc][     it] = threep_44[iconf][isrc][it];
      data[iconf][isrc][Thp1+it] = twop_orbit[iconf][isrc][it][0];
    }}
#endif  /* of if 0 */
      for ( int it = 0; it < Thp1; it++ ) {
        data[iconf][     it] = 0.;
        data[iconf][Thp1+it] = 0.;
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        data[iconf][     it] += threep_44[iconf][isrc][it];
        data[iconf][Thp1+it] += twop_orbit[iconf][isrc][it][0];
      }
        data[iconf][     it] /= (double)num_src_per_conf;
        data[iconf][Thp1+it] /= (double)num_src_per_conf;
      }
    }

    for ( int itau = 1; itau < Thp1/2; itau++ ) {

      int arg_first[6] = { 2 * itau, itau , 0, Thp1 + 2 * itau , Thp1, Thp1 + itau };
      int nT = Thp1 - 2 * itau;

      sprintf ( obs_name, "threep.fht.accum.%s.g4_D4.tau%d.PX%d_PY%d_PZ%d.%s",
          loop_tag,
          itau,
          g_sink_momentum_list[0][0],
          g_sink_momentum_list[0][1],
          g_sink_momentum_list[0][2], "re" );

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_func ( data[0], num_conf, 2*Thp1, nT, narg, arg_first, arg_stride, obs_name, fptr, dfptr );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(115);
      }
    }

    /**********************************************************
     * threep_4k
     **********************************************************/
#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
#if 0
    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
    for ( int it = 0; it < Thp1; it++ ) {
      data[iconf][isrc][     it] = threep_4k[iconf][isrc][it];
      data[iconf][isrc][Thp1+it] = twop_orbit[iconf][isrc][it][0];
    }}
#endif
      for ( int it = 0; it < Thp1; it++ ) {
        data[iconf][     it] = 0.;
        data[iconf][Thp1+it] = 0.;
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        data[iconf][     it] += threep_4k[iconf][isrc][it];
        data[iconf][Thp1+it] += twop_orbit[iconf][isrc][it][0];
      }
        data[iconf][     it] /= (double)num_src_per_conf;
        data[iconf][Thp1+it] /= (double)num_src_per_conf;
      }
    }

    for ( int itau = 1; itau < Thp1/2; itau++ ) {

      int arg_first[6] = { 2 * itau, itau , 0, Thp1 + 2 * itau , Thp1, Thp1 + itau };
      int nT = Thp1 - 2 * itau;

      char obs_name[400];
      sprintf ( obs_name, "threep.fht.accum.%s.g4_Dk.tau%d.PX%d_PY%d_PZ%d.%s",
          loop_tag,
          itau,
          g_sink_momentum_list[0][0],
          g_sink_momentum_list[0][1],
          g_sink_momentum_list[0][2], "re");

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_func ( data[0], num_conf, 2*Thp1, nT, narg, arg_first, arg_stride, obs_name, fptr, dfptr );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(115);
      }
    }

    /**********************************************************
     * threep_ik
     **********************************************************/
#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
#if 0
    for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
    for ( int it = 0; it < Thp1; it++ ) {
      data[iconf][isrc][     it] = threep_ik[iconf][isrc][it];
      data[iconf][isrc][Thp1+it] = twop_orbit[iconf][isrc][it][0];
    }}
#endif
      for ( int it = 0; it < Thp1; it++ ) {
        data[iconf][     it] = 0.;
        data[iconf][Thp1+it] = 0.;
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        data[iconf][     it] += threep_ik[iconf][isrc][it];
        data[iconf][Thp1+it] += twop_orbit[iconf][isrc][it][0];
      }
        data[iconf][     it] /= (double)num_src_per_conf;
        data[iconf][Thp1+it] /= (double)num_src_per_conf;
      }
    }
 
    for ( int itau = 1; itau < Thp1/2; itau++ ) {

      int arg_first[6] = { 2 * itau, itau , 0, Thp1 + 2 * itau , Thp1, Thp1 + itau };
      int nT = Thp1 - 2 * itau;

      char obs_name[400];
      sprintf ( obs_name, "threep.fht.accum.%s.gi_Dk.tau%d.PX%d_PY%d_PZ%d.%s",
          loop_tag,
          itau,
          g_sink_momentum_list[0][0],
          g_sink_momentum_list[0][1],
          g_sink_momentum_list[0][2], "re");

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_func ( data[0], num_conf, 2*Thp1, nT, narg, arg_first, arg_stride, obs_name, fptr, dfptr );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[avxn_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(115);
      }
    }  /* end of loop on itau */

    /* fini_3level_dtable ( &data ); */
    fini_2level_dtable ( &data );

    fini_3level_dtable ( &threep_44 );
    fini_3level_dtable ( &threep_4k );
    fini_3level_dtable ( &threep_ik );

  }  /* end FHT calculation */
#endif  /* end of if _FHT_METHOD_ACCUM */


  fini_6level_dtable ( &loop_sub );
  fini_6level_dtable ( &loop_sym );

#endif  /* end of ifdef _LOOP_ANALYSIS */

  fini_6level_dtable ( &twop );
  fini_4level_dtable ( &twop_orbit );

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

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "avxn_analyse", "runtime", g_cart_id == 0 );

  return(0);
}
