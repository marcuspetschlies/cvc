/****************************************************
 * xg_analyse 
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

#define _LOOP_ANALYSIS 1
#define _LOOP_STATS    1

#define  _XG_PION     1
#define  _XG_NUCLEON  0
#define  _XG_CHARGED  0

#define  _TWOP_AFF        0
#define  _TWOP_ASCII      0
#define  _TWOP_CYD_MULT   1
#define  _TWOP_CYD_SINGLE 0

#define _TWOP_STATS  1

#define _RAT_METHOD       1
#define _FHT_METHOD_ALLT  0
#define _FHT_METHOD_ACCUM 0

#define _RAT_SUB_METHOD   1

#define _SUBTRACT_TIME_AVERAGED  1
#define _SUBTRACT_PER_TIMESLICE  0

#define MAX_SMEARING_LEVELS 40

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse < Y-O_g Ybar > correlator contractions\n");
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
  
  const double TWO_MPI = 2. * M_PI;

   char const gamma_id_to_ascii[16][10] = {
    "gt",
    "gx",
    "gy",
    "gz",
    "id",
    "g5",
    "gtg5",
    "gxg5",
    "gyg5",
    "gzg5",
    "gtgx",
    "gtgy",
    "gtgz",
    "gxgy",
    "gxgz",
    "gygz"
  };

#if _XG_NUCLEON
  char const gamma_id_to_Cg_ascii[16][10] = {
    "Cgy",
    "Cgzg5",
    "Cgt",
    "Cgxg5",
    "Cgygt",
    "Cgyg5gt",
    "Cgyg5",
    "Cgz",
    "Cg5gt",
    "Cgx",
    "Cgzg5gt",
    "C",
    "Cgxg5gt",
    "Cgxgt",
    "Cg5",
    "Cgzgt"
  };
#endif



  char const reim_str[2][3] = { "re", "im" };

  char const correlator_prefix[2][20] = { "local-local" , "charged"};

  char const flavor_tag[2][20]        = { "d-gf-u-gi" , "u-gf-d-gi" };

  const char insertion_operator_name[5][20] = { "plaquette" , "clover" , "rectangle" , "clover-tl" , "rectangle-tl" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[600];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "NA";
  int twop_fold_propagator = 0;
  int write_data = 0;
  int operator_type = -1;
  struct timeval ta, tb;
  unsigned int stout_level_iter[MAX_SMEARING_LEVELS];
  double stout_level_rho[MAX_SMEARING_LEVELS];
  unsigned int stout_level_num = 0;
  int insertion_operator_type = 0;
  double temp_spat_weight[2] = { 1., -1. };
  double twop_weight[2]   = {0., 0.};
  double fbwd_weight[2]   = {0., 0.};
  double mirror_weight[2] = {0., 0.};
  int block_sources = 0;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "bh?f:N:S:F:R:E:w:O:s:W:T:B:I:M:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [xg_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [xg_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'F':
      twop_fold_propagator = atoi ( optarg );
      fprintf ( stdout, "# [xg_analyse] twop fold_propagator set to %d\n", twop_fold_propagator );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [xg_analyse] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'w':
      write_data = atoi ( optarg );
      fprintf ( stdout, "# [xg_analyse] write_date set to %d\n", write_data );
      break;
    case 'O':
      operator_type = atoi ( optarg );
      fprintf ( stdout, "# [xg_analyse] operator_type set to %d\n", operator_type );
      break;
    case 's':
      sscanf ( optarg, "%d,%lf", stout_level_iter+stout_level_num, stout_level_rho+stout_level_num );
      fprintf ( stdout, "# [xg_analyse] stout_level %d  iter %2d  rho %6.4f \n", stout_level_num, stout_level_iter[stout_level_num], stout_level_rho[stout_level_num] );
      stout_level_num++;
      break;
    case 'W':
      sscanf( optarg, "%lf,%lf", temp_spat_weight, temp_spat_weight+1 );
      fprintf ( stdout, "# [xg_analyse] temp_spat_weight set to %25.16e / %25.16e\n", temp_spat_weight[0], temp_spat_weight[1] );
      break;
    case 'T':
      sscanf( optarg, "%lf,%lf", twop_weight, twop_weight+1 );
      fprintf ( stdout, "# [xg_analyse] twop_weight set to %25.16e / %25.16e\n", twop_weight[0], twop_weight[1] );
      break;
    case 'B':
      sscanf( optarg, "%lf,%lf", fbwd_weight, fbwd_weight+1 );
      fprintf ( stdout, "# [xg_analyse] fbwd_weight set to %25.16e / %25.16e\n", fbwd_weight[0], fbwd_weight[1] );
      break;
    case 'M':
      sscanf( optarg, "%lf,%lf", mirror_weight, mirror_weight+1 );
      fprintf ( stdout, "# [xg_analyse] mirror_weight set to %25.16e / %25.16e\n", mirror_weight[0], mirror_weight[1] );
      break;
    case 'I':
      insertion_operator_type = atoi ( optarg );
      fprintf ( stdout, "# [xg_analyse] insertion_operator_type set to %d\n", insertion_operator_type );
      break;
    case 'b':
      block_sources = 1;
      fprintf ( stdout, "# [xg_analyse] will block sources\n" );
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
  /* fprintf(stdout, "# [xg_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [xg_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [xg_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [xg_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[xg_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[xg_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[xg_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [xg_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  /* sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf); */
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name );
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[xg_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[xg_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [xg_analyse] comment %s\n", line );
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

  /***********************************************************
   * read twop function data
   ***********************************************************/
  double ******** twop = init_8level_dtable ( g_sink_gamma_id_number, g_source_gamma_id_number, g_sink_momentum_number, num_conf, num_src_per_conf, 2, T_global, 2 );
  if( twop == NULL ) {
    fprintf ( stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT (24);
  }

  double ****** twop_orbit = init_6level_dtable ( g_sink_gamma_id_number, g_source_gamma_id_number, num_conf, num_src_per_conf, T_global, 2 );
  if( twop_orbit == NULL ) {
    fprintf ( stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT (24);
  }

  /***********************************************************
   * loop on gamma at sink
   ***********************************************************/
  for ( int igf = 0; igf < g_sink_gamma_id_number; igf++ ) {

  /***********************************************************
   * loop on gamma at source
   ***********************************************************/
  for ( int igi = 0; igi < g_source_gamma_id_number; igi++ ) {

    /**********************************************************
     **********************************************************
     ** 
     ** READ DATA
     ** 
     **********************************************************
     **********************************************************/

#if _XG_PION

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
            g_sink_gamma_id_list[igf], g_source_gamma_id_list[igi],
            ( 1 - 2 * iflavor ) * g_sink_momentum_list[ipf][0], 
            ( 1 - 2 * iflavor ) * g_sink_momentum_list[ipf][1],
            ( 1 - 2 * iflavor ) * g_sink_momentum_list[ipf][2] );

        affr = aff_reader ( data_filename );
        const char * aff_status_str = aff_reader_errstr ( affr );
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[xg_analyse] Error from aff_reader for filename %s, status was %s %s %d\n", data_filename, aff_status_str, __FILE__, __LINE__);
          EXIT(15);
        } else {
          if ( g_verbose > 1 ) fprintf(stdout, "# [xg_analyse] reading data from file %s\n", data_filename);
        }
  
        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[xg_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
          EXIT(103);
        }
  
        double ** buffer = init_2level_dtable ( T_global, 2 );
        if( buffer == NULL ) {
          fprintf(stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
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

            if ( g_verbose > 2 ) fprintf ( stdout, "# [xg_analyse] key = %s\n", key );
  
            affdir = aff_reader_chpath (affr, affn, key );
            if( affdir == NULL ) {
              fprintf(stderr, "[xg_analyse] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
              EXIT(105);
            }
  
            uint32_t uitems = T_global;
            int texitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)buffer[0], uitems );
            if( texitstatus != 0 ) {
              fprintf(stderr, "[xg_analyse] Error from aff_node_get_complex, status was %d %s %d\n", texitstatus, __FILE__, __LINE__);
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
              twop[igf][igi][ipf][iconf][isrc][iflavor][it][0] = buffer[itt][0] * ephase[0] - buffer[itt][1] * ephase[1];
              twop[igf][igi][ipf][iconf][isrc][iflavor][it][1] = buffer[itt][1] * ephase[0] + buffer[itt][0] * ephase[1];
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
    show_time ( &ta, &tb, "xg_analyse", "read-twop-tensor-aff", g_cart_id == 0 );
#endif  /* of if _TWOP_AFF */

  /***********************************************************/
  /***********************************************************/

#if _TWOP_ASCII
    gettimeofday ( &ta, (struct timezone *)NULL );

    for ( int iflavor = 0; iflavor <= 1 ; iflavor++ ) {
      for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

        char data_filename[500];

        sprintf( data_filename, "%s/%s.%s.gf%.2d.gi%.2d.px%dpy%dpz%d",
            filename_prefix,
            correlator_prefix[operator_type], flavor_tag[iflavor], g_sink_gamma_id_list[igf], g_source_gamma_id_list[igi],
            ( 1 - 2 * iflavor ) * g_sink_momentum_list[ipf][0],
            ( 1 - 2 * iflavor ) * g_sink_momentum_list[ipf][1],
            ( 1 - 2 * iflavor ) * g_sink_momentum_list[ipf][2] );

        fprintf(stdout, "# [xg_analyse] reading data from file %s\n", data_filename);

        FILE * fs = fopen ( data_filename , "r" );
        double ** buffer = init_2level_dtable ( T_global, 2 );

        /***********************************************************
         * loop on configs
         ***********************************************************/
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
  
          /***********************************************************
           * loop on sources
           ***********************************************************/
          for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
  
          /***********************************************************
           * loop on timeslices
           ***********************************************************/
            for( int it = 0; it < T_global; it++ ) {

              /***********************************************************
               * 
               ***********************************************************/
              if ( fscanf ( fs, "%lf %lf\n", buffer[it], buffer[it]+1 ) != 2 ) {
                fprintf ( stderr, "[] Error from fscanf for file %s %s %d\n", data_filename, __FILE__, __LINE__ );
                EXIT(12);
              }
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
              twop[igf][igi][ipf][iconf][isrc][iflavor][it][0] = buffer[itt][0] * ephase[0] - buffer[itt][1] * ephase[1];
              twop[igf][igi][ipf][iconf][isrc][iflavor][it][1] = buffer[itt][1] * ephase[0] + buffer[itt][0] * ephase[1];
            }
  
          }  /* end of loop on sources */
  
        }  /* end of loop on configs */

        fini_2level_dtable( &buffer );

        fclose ( fs );
  
      }  /* end of loop on sink momenta */

    }  /* end of loop on flavor */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "xg_analyse", "read-twop-tensor-aff", g_cart_id == 0 );

#endif  /* of if _TWOP_ASCII */

  /***********************************************************/
  /***********************************************************/

#if _TWOP_CYD_MULT
    gettimeofday ( &ta, (struct timezone *)NULL );

    for ( int iflavor = 0; iflavor <= 1 ; iflavor++ ) {
      for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
 
          char data_filename[500];

          sprintf( data_filename, "stream_%c/%s/twop.%.4d.pseudoscalar.%d.PX%d_PY%d_PZ%d",
              conf_src_list[iconf][0][0], filename_prefix, conf_src_list[iconf][0][1], iflavor+1,
              ( 1 - 2 * iflavor) * g_sink_momentum_list[ipf][0],
              ( 1 - 2 * iflavor) * g_sink_momentum_list[ipf][1],
              ( 1 - 2 * iflavor) * g_sink_momentum_list[ipf][2] );

          FILE * dfs = fopen ( data_filename, "r" );
          if( dfs == NULL ) {
            fprintf ( stderr, "[xg_analyse] Error from fopen for data filename %s %s %d\n", data_filename, __FILE__, __LINE__ );
            EXIT (24);
          } else {
            if ( g_verbose > 1 ) fprintf ( stdout, "# [xg_analyse] reading data from file %s data filename \n", data_filename );
          }
          fflush ( stdout );
          fflush ( stderr );

          for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            char line[400];

            for ( int it = -1; it < T_global; it++ ) {
              if ( fgets ( line, 100, dfs) == NULL ) {
                fprintf ( stderr, "[avxn_analyse] Error from fgets, expecting line input for it %3d conf %3d src %3d data filename %s %s %d\n",
                    it, iconf, isrc, data_filename, __FILE__, __LINE__ );
                EXIT (26);
              }

              if ( line[0] == '#' &&  it == -1 ) {
                if ( g_verbose > 2 ) fprintf ( stdout, "# [xg_analyse] reading key %s\n", line );
                continue;
              } /* else {
                fprintf ( stderr, "[avxn_analyse] Error in layout of file %s %s %d\n", data_filename, __FILE__, __LINE__ );
                EXIT(27);
              }
              */
              sscanf ( line, "%lf %lf\n", twop[igf][igi][ipf][iconf][isrc][iflavor][it], twop[igf][igi][ipf][iconf][isrc][iflavor][it]+1 );

            }  /* end of loop on timeslices */
          }  /* end of loop on source positions */
          fclose ( dfs );
        }  /* end of loop on configurations */

        /***********************************************************
         *
         * NO ORDERING FROM SOURCE
         *
         * NO MULTIPLICATION OF SOURCE PHASE
         *
         ***********************************************************/
      }  /* end of loop on sink momenta */

    }  /* end of loop on flavor */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "xg_analyse", "read-twop-tensor-cyi", g_cart_id == 0 );

#endif  /* of if _TWOP_CYD_MULT */

  /***********************************************************/
  /***********************************************************/

#if _TWOP_CYD_SINGLE
    gettimeofday ( &ta, (struct timezone *)NULL );

    for ( int iflavor = 0; iflavor <= 1 ; iflavor++ ) {
      for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {
          
        char data_filename[500];

        sprintf( data_filename, "%s/twop.pseudoscalar.%d.PX%d_PY%d_PZ%d", filename_prefix, iflavor+1,
            ( 1 - 2 * iflavor) * g_sink_momentum_list[ipf][0],
            ( 1 - 2 * iflavor) * g_sink_momentum_list[ipf][1],
            ( 1 - 2 * iflavor) * g_sink_momentum_list[ipf][2] );

        FILE * dfs = fopen ( data_filename, "r" );
        if( dfs == NULL ) {
          fprintf ( stderr, "[xg_analyse] Error from fopen for data filename %s %s %d\n", data_filename, __FILE__, __LINE__ );
          EXIT (24);
        } else {
          if ( g_verbose > 1 ) fprintf ( stdout, "# [xg_analyse] reading data from file %s\n", data_filename );
        }

        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
 
          for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

            char line[400];

            fscanf( dfs, "# %s\n", line );
            if ( g_verbose > 2 ) fprintf ( stdout, "# [xg_analyse] reading key %s\n", line );

            /***********************************************************
             *
             * NO ORDERING FROM SOURCE
             *
             * NO MULTIPLICATION OF SOURCE PHASE
             *
             ***********************************************************/
            for ( int it = 0; it < T_global; it++ ) {
              fscanf ( dfs, "%lf %lf\n", twop[igf][igi][ipf][iconf][isrc][iflavor][it], twop[igf][igi][ipf][iconf][isrc][iflavor][it]+1 );
            }  /* end of loop on timeslices */

          }  /* end of loop on source positions */
        }  /* end of loop on configurations */

        fclose ( dfs );

      }  /* end of loop on sink momenta */

    }  /* end of loop on flavor */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "xg_analyse", "read-twop-tensor-cyi", g_cart_id == 0 );

#endif  /* of if _TWOP_CYD_SINGLE */

    /**********************************************************
     * write correlator to ascii file
     **********************************************************/
    if ( write_data > 0 ) {

      for ( int ipf = 0; ipf < g_sink_momentum_number; ipf++ ) {

        sprintf( filename, "%s.%s.gf_%s.gi_%s.px%d_py%d_pz%d.corr", correlator_prefix[operator_type], flavor_tag[0],
            gamma_id_to_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_ascii[ g_source_gamma_id_list[igi] ],
            g_sink_momentum_list[ipf][0], g_sink_momentum_list[ipf][1], g_sink_momentum_list[ipf][2] );

        FILE * ofs = fopen( filename, "w" );
        if ( ofs == NULL ) {
          fprintf( stderr, "[xg_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT(23);
        } else {
          fprintf( stdout, "# [xg_analyse] writing data to file %s %s %d\n", filename, __FILE__, __LINE__ );
        }

        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              fprintf( ofs, " %25.16e %25.16e   %25.16e %25.16e\n",
                  twop[igf][igi][ipf][iconf][isrc][0][it][0], twop[igf][igi][ipf][iconf][isrc][0][it][1],
                  twop[igf][igi][ipf][iconf][isrc][1][it][0], twop[igf][igi][ipf][iconf][isrc][1][it][1] );
            }
          }
        }
     
        fclose ( ofs );
      }
    }  /* end of if write data > 0 */
#endif  /* end of _XG_PION */

#if _XG_NUCLEON
#if 0
    sprintf ( filename, "stream_%c/%s", conf_src_list[0][0][0], filename_prefix3 );
    FILE *fs = fopen( filename , "r" );
    if ( fs == NULL ) {
      fprintf ( stderr, "[xg_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
      EXIT(2);
    }
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        char line[400];
        int itmp;
        double dtmp[4];
        fgets ( line, 100, fs);
        fprintf( stdout, "# [xg_analyse] reading line %s\n", line);
        for( int it =0; it < T_global; it++ ) {
          fscanf ( fs, "%d %lf %lf %lf %lf\n", &itmp, dtmp+0, dtmp+1, dtmp+2, dtmp+3 );
              
          twop[igf][igi][0][iconf][isrc][0][it][0] = dtmp[0];
          twop[igf][igi][0][iconf][isrc][0][it][1] = dtmp[1];
          twop[igf][igi][0][iconf][isrc][0][(T_global - it)%T_global][0] = -dtmp[2];
          twop[igf][igi][0][iconf][isrc][0][(T_global - it)%T_global][1] = -dtmp[3];
        }
      }
    }
    fclose ( fs );
#endif  /* of if 0  */

    sprintf ( filename, "%s/twop.nucleon_zeromom.SS.dat", filename_prefix3 );
    FILE *fs = fopen( filename , "r" );
    if ( fs == NULL ) {
      fprintf ( stderr, "[xg_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
      EXIT(2);
    }

    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
        for( int it =0; it < T_global; it++ ) {
          if ( fscanf ( fs, "%lf %lf %lf %lf\n", 
              twop[igf][igi][0][iconf][isrc][0][it], twop[igf][igi][0][iconf][isrc][0][it]+1,
              twop[igf][igi][0][iconf][isrc][1][it], twop[igf][igi][0][iconf][isrc][1][it]+1 ) != 4 ) {

            fprintf ( stderr, "[xg_analyse] Error from fscanf for file %s %s %d\n", filename, __FILE__, __LINE__ );
            EXIT(2);
          }
        }
      }
    }

    fclose ( fs );
#endif  /* end of _XG_NUCLEON */

#if _XG_CHARGED
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

        const int ipf = 0;
        double dtmp;


        sprintf ( filename, "stream_%c/%s.%.2d.%.4d.gf_%s.gi_%s",
            conf_src_list[iconf][isrc][0], correlator_prefix[operator_type], conf_src_list[iconf][isrc][2], conf_src_list[iconf][isrc][1],
            gamma_id_to_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_ascii[ g_source_gamma_id_list[igi] ] );

        FILE *fs = fopen( filename , "r" );
        if ( fs == NULL ) {
          fprintf ( stderr, "[xg_analyse] Error from fopen for file %s %s %d\n", filename, __FILE__, __LINE__ );
          EXIT(2);
        } else {
          if ( g_verbose > 2 ) fprintf( stdout, "# [xg_analyse] reading data from file %s %s %d\n", filename, __FILE__, __LINE__ );
        }
        fscanf( fs, "%lf%lf\n", twop[igf][igi][ipf][iconf][isrc][0][0], &dtmp );
        for ( int it = 1; it < T_global/2; it++ ) {
          fscanf( fs, "%lf%lf\n", twop[igf][igi][ipf][iconf][isrc][0][it] , twop[igf][igi][ipf][iconf][isrc][0][T_global - it] );
        }
        fscanf( fs, "%lf%lf\n", twop[igf][igi][ipf][iconf][isrc][0][T_global/2], &dtmp );
        fclose ( ofs );
        memcpy( twop[igf][igi][ipf][iconf][isrc][1][0], twop[igf][igi][ipf][iconf][isrc][0][0], T_global*2*sizeof(double) );
      }
    }
#endif  /* of _XG_CHARGED */

    /**********************************************************
     *
     * average 2-pt over momentum orbit
     *
     **********************************************************/
#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

        /* averaging starts here */
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            twop_orbit[igf][igi][iconf][isrc][it][0] += ( 
                  twop_weight[0] * twop[igf][igi][imom][iconf][isrc][0][it][0] 
                + twop_weight[1] * twop[igf][igi][imom][iconf][isrc][1][it][0] 
                ) / ( twop_weight[0] + twop_weight[1] );

            twop_orbit[igf][igi][iconf][isrc][it][1] += ( 
                  twop_weight[0] * twop[igf][igi][imom][iconf][isrc][0][it][1] 
                + twop_weight[1] * twop[igf][igi][imom][iconf][isrc][1][it][1] 
                ) / ( twop_weight[0] + twop_weight[1] );
          }  /* end of loop on it */
        }  /* end of loop on imom */

        /* multiply norm from averages over momentum orbit and source locations */
        double const norm = 1. / (double)g_sink_momentum_number;
        for ( int it = 0; it < 2*T_global; it++ ) {
          twop_orbit[igf][igi][iconf][isrc][0][it] *= norm;
        }
      }  /* end of loop on isrc */
    }  /* end of loop on iconf */

    /**********************************************************
     * write orbit-averaged data to ascii file, per source
     **********************************************************/

    char obs_name_prefix[200];
#if _XG_NUCLEON
    sprintf ( obs_name_prefix, "NN.orbit.gf_%s.gi_%s.px%d_py%d_pz%d",
            gamma_id_to_Cg_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_Cg_ascii[ g_source_gamma_id_list[igi] ],
            g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2] );
#else
    sprintf ( obs_name_prefix, "%s.%s.orbit.gf_%s.gi_%s.px%d_py%d_pz%d", correlator_prefix[operator_type], flavor_tag[0],
            gamma_id_to_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_ascii[ g_source_gamma_id_list[igi] ],
            g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2] );
#endif

    if ( write_data > 0 ) {
      for ( int ireim = 0; ireim <= 1; ireim++ ) {
        sprintf ( filename, "%s.%s.corr", obs_name_prefix, reim_str[ireim]);

        write_data_real2_reim ( twop_orbit[igf][igi], filename, conf_src_list, num_conf, num_src_per_conf, T_global, ireim );
      }
    }  /* end of if write data */

#if _TWOP_STATS
    /**********************************************************
     * 
     * STATISTICAL ANALYSIS
     * 
     **********************************************************/
    for ( int ireim = 0; ireim < 1; ireim++ ) {

      /* if ( num_conf < 6 ) {
        fprintf ( stderr, "[xg_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      } */

      int block_size = block_sources ? num_src_per_conf : 1;
      int num_data = ( num_conf * num_src_per_conf ) / block_size;

      double ** data = init_2level_dtable ( num_data, T_global );
      if ( data == NULL ) {
        fprintf ( stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n",  __FILE__, __LINE__ );
        EXIT(1);
      }

      /* fill data array */
      if ( twop_fold_propagator != 0 ) {
#pragma omp parallel for
        for ( int i = 0; i < num_data; i++ ) {
          for ( int it = 0; it <= T_global/2; it++ ) {
            int const itt  = ( T_global - it ) % T_global;

            for ( int k = 0; k < block_size; k++ ) {
              int const idx  = ( ( i * block_size + k ) * T_global + it  ) * 2 + ireim;
              int const idx2 = ( ( i * block_size + k ) * T_global + itt ) * 2 + ireim;

              data[i][it] += 0.5 * ( twop_orbit[igf][igi][0][0][0][idx] + twop_fold_propagator * twop_orbit[igf][igi][0][0][0][idx2] );
            } 

            data[i][it ] /= (double)block_size;
            data[i][itt] = data[i][it];
          }
        }
      } else {
#pragma omp parallel for
        for ( int i = 0; i < num_data; i++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            for ( int k = 0; k < block_size; k++ ) {
              int const idx  = ( ( i * block_size + k ) * T_global + it  ) * 2 + ireim;
              data[i][it] += twop_orbit[igf][igi][0][0][0][idx];
            }
            data[i][it] /= (double)block_size;
          }
        }
      }

      char obs_name[200];
      sprintf( obs_name, "%s.%s", obs_name_prefix, reim_str[ireim] );

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_real ( data[0], num_data, T_global, 0, 1, obs_name );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      /**********************************************************
       * write data to ascii file
       **********************************************************/
      if ( write_data > 1 ) {
        sprintf ( filename, "%s.corr", obs_name );
        FILE * dfs = fopen ( filename, "w" );
        for ( int i = 0; i < num_data; i++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            fprintf ( dfs, "%25.16e\n", data[i][it] );
          }
        }
        fclose ( dfs );
      }  /* end of if write data */

      /**********************************************************
       * acosh ratio for m_eff
       **********************************************************/
      int const Thp1 = T_global / 2 + 1;
      for ( int itau = 1; itau < Thp1/2; itau++ ) {

#if _XG_NUCLEON
        int narg = 2;
        int arg_first[2] = { 0, itau };
        int arg_stride[2] = {1, 1};
        int nT = Thp1 - itau;

        sprintf ( obs_name, "%s.log_ratio.tau%d.%s", obs_name_prefix, itau, reim_str[ireim] );
        exitstatus = apply_uwerr_func ( data[0], num_data, T_global, nT, narg, arg_first, arg_stride, obs_name, log_ratio_1_1, dlog_ratio_1_1 );
#else
        int narg = 3;
        int arg_first[3] = { 0, 2 * itau, itau };
        int arg_stride[3] = {1,1,1};
        int nT = Thp1 - 2 * itau;
        sprintf ( obs_name, "%s.acosh_ratio.tau%d.%s", obs_name_prefix, itau, reim_str[ireim] );
        exitstatus = apply_uwerr_func ( data[0], num_data, T_global, nT, narg, arg_first, arg_stride, obs_name, acosh_ratio, dacosh_ratio );
#endif
        if ( exitstatus != 0 ) {
          fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(115);
        }
      }  /* end of loop on tau */

      fini_2level_dtable ( &data );
    }  /* end of loop on reim */

#endif  /* of if _TWOP_STATS */

  }}  /* end of source and sink gamma id */

  /**********************************************************
   * loop on stout smearing levels
   **********************************************************/
  
  double **** loop = NULL;
#if _LOOP_ANALYSIS
  /**********************************************************
   *
   * loop fields
   *
   **********************************************************/
  loop = init_4level_dtable ( stout_level_num, num_conf, T_global, 2 );
  if ( loop == NULL ) {
    fprintf ( stdout, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(25);
  }

  /**********************************************************
   *
   * read loop data
   *
   **********************************************************/
  int conf_id_prev = -1;
  for ( int iconf = 0; iconf < num_conf; iconf++ ) {

    int conf_id_new = conf_src_list[iconf][0][1];

    if ( conf_id_new == conf_id_prev ) {
      fprintf ( stdout, "# [xg_analyse] conf = %4d / %6d copy from previous\n", iconf, conf_src_list[iconf][0][1] );
      for ( unsigned int istout = 0; istout < stout_level_num; istout++ ) {
        memcpy ( loop[istout][iconf][0], loop[istout][iconf-1][0], 2*T_global*sizeof(double ) );
      }
      continue;
    }

    /***********************************************************
     * open AFF reader
     ***********************************************************/
    struct AffReader_s *affr = NULL;
    struct AffNode_s *affn = NULL, *affdir = NULL;
    char key[400];

  
    /* sprintf ( filename, "%s/stream_%c/%d/cpff.xg.%d.aff", filename_prefix2, conf_src_list[iconf][0][0], conf_src_list[iconf][0][1], conf_src_list[iconf][0][1] ); */
    /* sprintf ( filename, "stream_%c/%s/%s.%d.aff", conf_src_list[iconf][0][0], filename_prefix2, filename_prefix3, conf_src_list[iconf][0][1] ); */
    sprintf ( filename, "cpff.xg.%d.aff", conf_src_list[iconf][0][1] );
  
    fprintf(stdout, "# [xg_analyse] reading data from file %s\n", filename);
    affr = aff_reader ( filename );
    const char * aff_status_str = aff_reader_errstr ( affr );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[xg_analyse] Error from aff_reader for filename %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
  
    if( (affn = aff_reader_root( affr )) == NULL ) {
      fprintf(stderr, "[xg_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
      return(103);
    }
  
    /**********************************************************
     * loop on smearing levels
     **********************************************************/
    for ( unsigned int istout = 0; istout < stout_level_num; istout++ ) {

      /* sprintf( key, "/StoutN%d/StoutRho%6.4f/%s", stout_level_iter[istout], stout_level_rho[istout], insertion_operator_name[insertion_operator_type] ); */
      sprintf( key, "/StoutN%d/StoutRho%6.4f", stout_level_iter[istout], stout_level_rho[istout] );
          
      if ( g_verbose > 2 ) fprintf ( stdout, "# [xg_analyse] key = %s\n", key );
  
      affdir = aff_reader_chpath (affr, affn, key );
      if( affdir == NULL ) {
        fprintf(stderr, "[xg_analyse] Error from aff_reader_chpath for key %s %s %d\n", key, __FILE__, __LINE__);
        EXIT(105);
      }
  
      uint32_t uitems = 2 * T_global;
      exitstatus = aff_node_get_double ( affr, affdir, loop[istout][iconf][0], uitems );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[xg_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(105);
      }
  
    }  /* end of loop on stout smearing levels */

    aff_reader_close ( affr );
    
    conf_id_prev = conf_id_new;

  }  /* end of loop on configs */
  
  for ( unsigned int istout = 0; istout < stout_level_num; istout++ ) {

    char smearing_tag[50];
    sprintf ( smearing_tag, "stout_%d_%6.4f", stout_level_iter[istout], stout_level_rho[istout] );

#if _LOOP_STATS
    /**********************************************************
     * STATISTICAL ANALYSE plaquettes
     **********************************************************/
    {

      int num_data = num_conf;
      double ** data = init_2level_dtable ( num_data, 4 );
      if ( data == NULL ) {
        fprintf ( stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      }


#pragma omp parallel for
      for ( int i = 0; i< num_data; i++ ) {
        data[i][0] = 0.;
        data[i][1] = 0.;
        data[i][2] = 0.;
        data[i][3] = 0.;
        for( int it = 0; it < T_global; it++ ) {
          data[i][0] += loop[istout][i][it][0];
          data[i][1] += loop[istout][i][it][1];
          data[i][2] += loop[istout][i][it][0] + loop[istout][i][it][1];
          data[i][3] += loop[istout][i][it][0] - loop[istout][i][it][1];
        }
        data[i][0] /= (18. * VOLUME);
        data[i][1] /= (18. * VOLUME);
        data[i][2] /= (18. * VOLUME);
        data[i][3] /= (18. * VOLUME);
      }

      char obs_name[100];
      sprintf ( obs_name, "%s.%s" , insertion_operator_name[insertion_operator_type], smearing_tag );

      /* apply UWerr analysis */
      exitstatus = apply_uwerr_real ( data[0], num_data, 4, 0, 1, obs_name );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      fini_2level_dtable ( &data );

    }  /* end of block */

#endif  /* of _LOOP_STATS */

    /**********************************************************
     *
     * build trace-subtracted tensor
     *
     **********************************************************/
    double ** loop_sub = init_2level_dtable ( num_conf, T_global );
    if ( loop_sub == NULL ) {
      fprintf ( stdout, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(25);
    }
  
#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
      for ( int it = 0; it < T_global; it++ ) {
        loop_sub[iconf][it] = temp_spat_weight[0] * loop[istout][iconf][it][0] + temp_spat_weight[1] * loop[istout][iconf][it][1];
      }
    }  /* end of loop on configs */
  
    /**********************************************************
     * tag to characterize the loops w.r.t. low-mode and
     * stochastic part
     **********************************************************/

    /**********************************************************
     * write loop_sub to separate ascii file
     **********************************************************/
    if ( write_data > 0 ) {
      sprintf ( filename, "%s.timeslice.%s.corr", insertion_operator_name[insertion_operator_type], smearing_tag );
  
      FILE * loop_sub_fs = fopen( filename, "w" );
      if ( loop_sub_fs == NULL ) {
        fprintf ( stderr, "[xg_analyse] Error from fopen %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      } 
  
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        fprintf ( loop_sub_fs , "# %c %6d\n", conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
        for ( int it = 0; it < T_global; it++ ) {
          fprintf ( loop_sub_fs , "%25.16e\n", loop_sub[iconf][it] );
        }
      }
      fclose ( loop_sub_fs );
    }  /* end of if write data */
  
#if _LOOP_STATS
    /**********************************************************
     *
     * STATISTICAL ANALYSIS OF LOOP VEC
     *
     **********************************************************/
    /* if ( num_conf < 6 ) {
      fprintf ( stderr, "[xg_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    } */
  
    char obs_name[100];
    sprintf ( obs_name, "%s.timeslice.%s" , insertion_operator_name[insertion_operator_type], smearing_tag );
  
    /* apply UWerr analysis */
    exitstatus = apply_uwerr_real ( loop_sub[0], num_conf, T_global, 0, 1, obs_name );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    }
  
#endif  /* of _LOOP_STATS */

#if _RAT_METHOD
    /**********************************************************
     *
     * STATISTICAL ANALYSIS for products and ratios
     *
     * fixed source - sink separation
     *
     **********************************************************/
  
    for ( int igf = 0; igf < g_sink_gamma_id_number; igf++ ) {

    for ( int igi = 0; igi < g_source_gamma_id_number; igi++ ) {

      /**********************************************************
       * loop on source - sink time separations
       **********************************************************/
      for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {
    
        double **** threep_44 = init_4level_dtable ( num_conf, num_src_per_conf, T_global, 2 ) ;
        if ( threep_44 == NULL ) {
          fprintf ( stderr, "[xg_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }
    
#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
    
            /* sink time = source time + dt  */
            /* int const tsink  = (  g_sequential_source_timeslice_list[idt] + conf_src_list[iconf][isrc][2] + T_global ) % T_global; */
            int const tsink  = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
            /* sink time with time reversal = source time - dt  */
            /* int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + conf_src_list[iconf][isrc][2] + T_global ) % T_global; */
            int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
    
            /* if ( g_verbose > 4 ) fprintf ( stdout, "# [xg_analyse] t_src %3d   dt %3d   tsink %3d tsink2 %3d\n", conf_src_list[iconf][isrc][2],
                g_sequential_source_timeslice_list[idt], tsink, tsink2 ); */
    
            /**********************************************************
             * !!! LOOK OUT:
             *       This includes the momentum orbit average !!!
             **********************************************************/
            for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
    
#if 0
              /**********************************************************
               * Do we need to add a source phase here ?
               **********************************************************/
              double const mom[3] = { 
                  2 * M_PI * g_sink_momentum_list[imom][0] / (double)LX_global,
                  2 * M_PI * g_sink_momentum_list[imom][1] / (double)LY_global,
                  2 * M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global };
#endif
              /**********************************************************
               * twop values 
               * a = field 0 at +mom
               * b = field 1 at -mom AS READ ABOVE = parity partner
               **********************************************************/
              double const a_fwd[2] = { twop[igf][igi][imom][iconf][isrc][0][tsink ][0], twop[igf][igi][imom][iconf][isrc][0][tsink ][1] };
              double const a_bwd[2] = { twop[igf][igi][imom][iconf][isrc][0][tsink2][0], twop[igf][igi][imom][iconf][isrc][0][tsink2][1] };

              double const b_fwd[2] = { twop[igf][igi][imom][iconf][isrc][1][tsink ][0], twop[igf][igi][imom][iconf][isrc][1][tsink ][1] };
              double const b_bwd[2] = { twop[igf][igi][imom][iconf][isrc][1][tsink2][0], twop[igf][igi][imom][iconf][isrc][1][tsink2][1] };
    
              double const threep_norm = 1. / ( fabs( twop_weight[0]   ) + fabs( twop_weight[1]   ) )
                                            / ( fabs( fbwd_weight[0]   ) + fabs( fbwd_weight[1]   ) )
                                            / ( fabs( mirror_weight[0] ) + fabs( mirror_weight[1] ) );


              /**********************************************************
               * loop on insertion times
               **********************************************************/
              for ( int it = 0; it < T_global; it++ ) {
    
                /**********************************************************
                 * fwd 1 insertion time = source time      + it
                 **********************************************************/
                int const tins_fwd_1 = (  it + conf_src_list[iconf][isrc][2]                                           + T_global ) % T_global;
    
                /**********************************************************
                 * fwd 2 insertion time = source time + dt - it
                 **********************************************************/
                int const tins_fwd_2 = ( -it + conf_src_list[iconf][isrc][2] + g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
    
                /**********************************************************
                 * bwd 1 insertion time = source time      - it
                 **********************************************************/
                int const tins_bwd_1 = ( -it + conf_src_list[iconf][isrc][2]                                           + T_global ) % T_global;
    
                /**********************************************************
                 * bwd 2 insertion time = source time - dt + it
                 **********************************************************/
                int const tins_bwd_2 = (  it + conf_src_list[iconf][isrc][2] - g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
    
                if ( g_verbose > 2 ) fprintf ( stdout, "# [xg_average] insertion times tsrc %3d    dt %3d    tc %3d    tins %3d %3d %3d %3d\n",
                    conf_src_list[iconf][isrc][2], g_sequential_source_timeslice_list[idt], it,
                    tins_fwd_1, tins_fwd_2, tins_bwd_1, tins_bwd_2 );
    
                /**********************************************************
                 * O44, real parts only
                 **********************************************************/
#if _XG_NUCLEON
                /**********************************************************
                 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                 * !!! CAREFUL with combination of nn type 0 and 1 !!!
                 * !!! these are fwd / bwd running ???             !!!
                 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                 **********************************************************/
                threep_44[iconf][isrc][it][0] += (
                    /* twop */
                    twop_weight[0] * ( 
                          fbwd_weight[0] * a_fwd[0] * loop_sub[iconf][tins_fwd_1] 
                        + fbwd_weight[1] * a_bwd[0] * loop_sub[iconf][tins_bwd_1] 
                      ) / ( fabs( fbwd_weight[0] ) + fabs( fbwd_weight[1] ) )
                    /* twop parity partner */
                  + twop_weight[1] * ( 
                         fbwd_weight[0] * b_fwd[0] * loop_sub[iconf][tins_bwd_1]
                       + fbwd_weight[1] * b_bwd[0] * loop_sub[iconf][tins_fwd_1]
                      ) / ( fabs( fbwd_weight[0] ) + fabs( fbwd_weight[1] ) )
                  ) / ( fabs( twop_weight[0] ) + fabs( twop_weight[1] ) );

#else  /* of if _XG_NUCLEON */

#if 0
                threep_44[iconf][isrc][it][0] += ( 
                        ( a_fwd[0] + b_fwd[0] ) * ( loop_sub[iconf][tins_fwd_1] + loop_sub[iconf][tins_fwd_2] ) 
                      + ( a_bwd[0] + b_bwd[0] ) * ( loop_sub[iconf][tins_bwd_1] + loop_sub[iconf][tins_bwd_2] )
                    ) * 0.125;
#endif


                // threep_44[iconf][isrc][it][0] += a_fwd[0] * loop_sub[iconf][tins_fwd_1];
                threep_44[iconf][isrc][it][0] += threep_norm * (
                    /* twop */
                    twop_weight[0] * (
                          fbwd_weight[0] * a_fwd[0] * ( mirror_weight[0] * loop_sub[iconf][tins_fwd_1] + mirror_weight[1] * loop_sub[iconf][tins_fwd_2] )
                        + fbwd_weight[1] * a_bwd[0] * ( mirror_weight[0] * loop_sub[iconf][tins_bwd_1] + mirror_weight[1] * loop_sub[iconf][tins_bwd_2] )
                      )
                    /* twop parity partner */
                  + twop_weight[1] * (
                          fbwd_weight[0] * b_fwd[0] * ( mirror_weight[0] * loop_sub[iconf][tins_fwd_1] + mirror_weight[1] * loop_sub[iconf][tins_fwd_2] )
                        + fbwd_weight[1] * b_bwd[0] * ( mirror_weight[0] * loop_sub[iconf][tins_bwd_1] + mirror_weight[1] * loop_sub[iconf][tins_bwd_2] )
                      )
                  );

#endif  /* end of else of if _XG_NUCLEON */

              }  /* end of loop on it */
    
            }  /* end of loop on imom */
    
            /**********************************************************
             * normalize
             **********************************************************/
            /* O44 simple orbit average */
            double const norm44 = 1. / g_sink_momentum_number;
            for ( int it = 0; it < 2 * T_global; it++ ) {
              threep_44[iconf][isrc][0][it] *= norm44;
            }
          }  /* end of loop on isrc */
        }  /* end of loop on iconf */
    
        /**********************************************************
         * name tag for observable
         **********************************************************/
        char obsname_tag[400];
#if _XG_NUCLEON
        sprintf ( obsname_tag, "gf_%s.gi_%s.%s.%s.dtsnk%d.PX%d_PY%d_PZ%d",
            gamma_id_to_Cg_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_Cg_ascii[ g_source_gamma_id_list[igi] ],
            insertion_operator_name[insertion_operator_type], smearing_tag,
            g_sequential_source_timeslice_list[idt],
            g_sink_momentum_list[0][0],
            g_sink_momentum_list[0][1],
            g_sink_momentum_list[0][2] );
#else
        sprintf ( obsname_tag, "gf_%s.gi_%s.%s.%s.dtsnk%d.PX%d_PY%d_PZ%d",
            gamma_id_to_ascii[ g_sink_gamma_id_list[igf] ], gamma_id_to_ascii[ g_source_gamma_id_list[igi] ],
            insertion_operator_name[insertion_operator_type], smearing_tag,
            g_sequential_source_timeslice_list[idt],
            g_sink_momentum_list[0][0],
            g_sink_momentum_list[0][1],
            g_sink_momentum_list[0][2] );
#endif

        /**********************************************************
         * write 3pt function to ascii file, per source
         **********************************************************/
        if ( write_data > 0 ) {
          /**********************************************************
           * write 44 3pt
           **********************************************************/
          for ( int ireim = 0; ireim < 1; ireim++ ) {
    
            sprintf ( filename, "threep.%s.%s.corr", obsname_tag, reim_str[ireim] );

            write_data_real2_reim ( threep_44, filename, conf_src_list, num_conf, num_src_per_conf, T_global, ireim );
    
          }  /* end of loop on ireim */
    
        }  /* end of if write_data */
    
        /**********************************************************
         *
         * STATISTICAL ANALYSIS for threep
         *
         * with fixed source - sink separation
         *
         **********************************************************/
        for ( int ireim = 0; ireim < 1; ireim++ ) {
    
          /* if ( num_conf < 6 ) {
            fprintf ( stderr, "[xg_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          } */

          /* int nT = g_sequential_source_timeslice_list[idt] + 1; */
          int nT = T_global;
          int block_size = block_sources ? num_src_per_conf : 1;
          if ( g_verbose > 1 ) printf ( "# [xg_analyse] block_size = %d %s %d\n", block_size, __FILE__, __LINE__ );
          int num_data = ( num_conf * num_src_per_conf ) / block_size;

          double ** data = init_2level_dtable ( num_data, nT );
          if ( data == NULL ) {
            fprintf ( stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }
    
          /**********************************************************
           * threep_44
           **********************************************************/

#pragma omp parallel for
#if 0
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < nT; it++ ) {
              double dtmp = 0.;
              for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                dtmp += threep_44[iconf][isrc][it][ireim];
              }
              data[iconf][it] = dtmp / (double)num_src_per_conf;
            }
          }
#endif
          for ( int i = 0; i < num_data; i++ ) {
            for ( int it = 0; it < nT; it++ ) {
              for ( int k = 0; k < block_size; k++ ){
                int const idx = ( ( i * block_size + k ) * T_global + it ) * 2 + ireim;
                data[i][it] += threep_44[0][0][0][idx];
              }
              data[i][it] /= (double)block_size;
            }
          }

          char obs_name[500];
          sprintf ( obs_name, "threep.%s.%s", obsname_tag, reim_str[ireim] );

          /* apply UWerr analysis */
          exitstatus = apply_uwerr_real ( data[0], num_data, nT, 0, 1, obs_name );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
    
          if ( write_data > 1 ) {
            sprintf ( filename, "%s.corr", obs_name );
            /* write_data_real ( data, filename, conf_src_list, num_conf, T_global ); */
            FILE * dfs = fopen ( filename, "w" );
            for ( int i = 0; i < num_data; i++ ) {
              for ( int it = 0; it < nT; it++ ) {
                fprintf( dfs, "%25.16e\n", data[i][it] );
              }
            }
            fclose ( dfs );
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
          char obs_name[100];
          int block_size = block_sources ? num_src_per_conf : 1;
          int num_data = ( num_conf * num_src_per_conf ) / block_size;

          double ** data = init_2level_dtable ( num_data, nT + 1 );
          if ( data == NULL ) {
            fprintf ( stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }
    
          /**********************************************************
           * O44
           **********************************************************/
          /* src_avg_real2_reim ( data, threep_44, num_conf, num_src_per_conf, nT, ireim ); */
    
#pragma omp parallel for
          for ( int i = 0; i < num_data; i++ ) {
            for ( int it = 0; it < nT; it++ ) {
              for ( int k = 0; k < block_size; k++ ){
                int const idx = ( ( i * block_size + k ) * T_global + it ) * 2 + ireim;
                data[i][it] += threep_44[0][0][0][idx];
              }
              data[i][it] /= (double)block_size;
            }
            for ( int k = 0; k < block_size; k++ ){
              /* tsink counter from source time  */
              /* int const tsink  = (  g_sequential_source_timeslice_list[idt] + conf_src_list[iconf][isrc][2] + T_global ) % T_global; */

              /* tsink counted from 0, relative to source time */
              int const tsink  = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
              int const idx = ( ( i * block_size + k ) * T_global + tsink ) * 2 + ireim;
#if _XG_NUCLEON
              data[i][nT] += twop_orbit[igf][igi][0][0][0][idx];
#else
              /* tsink2 counted from absolute source time */
              /* int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + conf_src_list[iconf][isrc][2] + T_global ) % T_global; */

              /* tsink2 counted from 0, relative to source time */
              int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
              int const idx2 = ( ( i * block_size + k ) * T_global + tsink2 ) * 2 + ireim;
              /**********************************************************
               * FOLD OR NO FOLD ???
               **********************************************************/
              data[i][nT] += ( twop_orbit[igf][igi][0][0][0][idx] + twop_fold_propagator * twop_orbit[igf][igi][0][0][0][idx2] ) / ( 1 + twop_fold_propagator );
              /* data[i][nT] += twop_orbit[igf][igi][0][0][0][idx]; */
#endif
            }
            data[i][nT] /= (double)block_size;
          }

          sprintf ( obs_name, "ratio.%s.%s", obsname_tag, reim_str[ireim] );

          exitstatus = apply_uwerr_func ( data[0], num_data, nT+1, nT, narg, arg_first, arg_stride, obs_name, ratio_1_1, dratio_1_1 );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(115);
          }
    
          fini_2level_dtable ( &data );

        }  /* end of loop on reim */
    
#ifdef _RAT_SUB_METHOD

        /**********************************************************
         *
         * STATISTICAL ANALYSIS for ratio sub
         *
         * with fixed source - sink separation  and
         *
         * with vev subtraction
         *
         **********************************************************/
        for ( int ireim = 0; ireim < 1; ireim++ ) {
    
          /* if ( num_conf < 6 ) {
            fprintf ( stderr, "[xg_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          } */
    
          /* UWerr parameters */
          int nT = g_sequential_source_timeslice_list[idt] + 1;
          int narg          = 3;
          int arg_first[3]  = { 0, nT, nT + 1 };
          int arg_stride[3] = { 1,  0 , 0};
          int block_size = block_sources ? num_src_per_conf : 1;
          int num_data = ( num_conf * num_src_per_conf ) / block_size;


          double ** data = init_2level_dtable ( num_data, nT + 2 );
          if ( data == NULL ) {
            fprintf ( stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }
    
          /**********************************************************
           * threep_44 sub
           **********************************************************/
#pragma omp parallel for
#if 0
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {

            for ( int it = 0; it < nT; it++ ) {
              double dtmp = 0.;
              for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                dtmp += threep_44[iconf][isrc][it][ireim];
              }
              data[iconf][it] = dtmp / (double)num_src_per_conf;
            }


            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              /* COUNT FROM SOURCE 0 */
              int const tsink  = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
#if _XG_NUCLEON
              data[iconf][nT] += twop_orbit[igf][igi][iconf][isrc][tsink][ireim];
#else
              int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
              data[iconf][nT] += 0.5 * ( twop_orbit[igf][igi][iconf][isrc][tsink][ireim] + twop_orbit[igf][igi][iconf][isrc][tsink2][ireim] );
#endif
            }
            data[iconf][nT] /= (double)num_src_per_conf;

            data[iconf][nT + 1] = 0.;
            for ( int it = 0; it < T_global; it++ ) {
              data[iconf][nT + 1] += loop_sub[iconf][it];
            }
            data[iconf][nT + 1] /= (double)T_global;
          }
#endif
          for ( int i = 0; i < num_data; i++ ) {
            for ( int it = 0; it < nT; it++ ) {
              for ( int k = 0; k < block_size; k++ ){
                int const idx = ( ( i * block_size + k ) * T_global + it ) * 2 + ireim;
                data[i][it] += threep_44[0][0][0][idx];
              }
              data[i][it] /= (double)block_size;
            }
            for ( int k = 0; k < block_size; k++ ){
              /* tsink counter from source time  */
              /* int const tsink  = (  g_sequential_source_timeslice_list[idt] + conf_src_list[iconf][isrc][2] + T_global ) % T_global; */

              /* tsink counted from 0, relative to source time */
              int const tsink = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
              int const idx   = ( ( i * block_size + k ) * T_global + tsink ) * 2 + ireim;
#if _XG_NUCLEON
              data[i][nT] += twop_orbit[igf][igi][0][0][0][idx];
#else
              /* tsink2 counted from absolute source time */
              /* int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + conf_src_list[iconf][isrc][2] + T_global ) % T_global; */

              /* tsink2 counted from 0, relative to source time */
              int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
              int const idx2   = ( ( i * block_size + k ) * T_global + tsink2 ) * 2 + ireim;
              /**********************************************************
               * FOLD OR NO FOLD ???
               **********************************************************/

              data[i][nT] += ( twop_orbit[igf][igi][0][0][0][idx] + twop_fold_propagator * twop_orbit[igf][igi][0][0][0][idx2] ) / ( 1 + twop_fold_propagator );
              /* data[i][nT] += twop_orbit[igf][igi][0][0][0][idx]; */
#endif
            }
            data[i][nT] /= (double)block_size;

#if _SUBTRACT_TIME_AVERAGED
            for ( int it = 0; it < T_global; it++ ) {
              int const idx = ( ( i * block_size ) / num_src_per_conf ) * T_global + it;
              data[i][nT + 1] += loop_sub[0][idx];
            }
            data[i][nT + 1] /= (double)T_global;
#elif _SUBTRACT_PER_TIMESLICE
            int const jc = ( i * block_size ) / num_src_per_conf;  /* config number */
            for ( int k = 0; k < block_size; k++ ){
              int const js = ( i * block_size + k ) % num_src_per_conf;  /* source coords number */
              for ( int it = 0; it < nT; it++ ) {

                int const tins_fwd_1 = (  it + conf_src_list[jc][js][2]                                           + T_global ) % T_global;
                int const tins_fwd_2 = ( -it + conf_src_list[jc][js][2] + g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
                int const tins_bwd_1 = ( -it + conf_src_list[jc][js][2]                                           + T_global ) % T_global;
                int const tins_bwd_2 = (  it + conf_src_list[jc][js][2] - g_sequential_source_timeslice_list[idt] + T_global ) % T_global;

                data[i][nT + 1] += 
                    fbwd_weight[0] * ( mirror_weight[0] * loop_sub[jc][tins_fwd_1] + mirror_weight[1] * loop_sub[jc][tins_fwd_2] )
                  + fbwd_weight[1] * ( mirror_weight[0] * loop_sub[jc][tins_bwd_1] + mirror_weight[1] * loop_sub[jc][tins_bwd_2] );
              }  /* end of loop on timeslices */
              data[i][nT + 1] /= ( fabs( fbwd_weight[0] ) + fabs( fbwd_weight[1] ) ) * ( fabs( mirror_weight[0] ) + fabs( mirror_weight[1] ) ) * nT;
            }  /* end of loop inside block */
#endif

          }  /* end of loop on num_data */

          char obs_name[500];
          sprintf ( obs_name, "ratio.sub.%s.%s", obsname_tag, reim_str[ireim] );

          /* apply UWerr analysis */
          exitstatus = apply_uwerr_func ( data[0], num_data, nT+2, nT, narg, arg_first, arg_stride, obs_name, ratio_1_2_mi_3, dratio_1_2_mi_3 );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
    
          fini_2level_dtable ( &data );
        }  /* end of loop on reim */

        /**********************************************************
         *
         * STATISTICAL ANALYSIS for summed ratio vev-subtracted
         *
         **********************************************************/
        for ( int ireim = 0; ireim < 1; ireim++ ) {
    
          /* if ( num_conf < 6 ) {
            fprintf ( stderr, "[xg_analyse] Error, too few observations for stats %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }*/
    
          /* UWerr parameters */
          int nT = ( g_sequential_source_timeslice_list[idt] / 2 ) + 1;
          int narg          = 3;
          int arg_first[3]  = { 0, nT, nT + 1 };
          int arg_stride[3] = { 1,  0 , 0};
          int block_size = block_sources ? num_src_per_conf : 1;
          int num_data = ( num_conf * num_src_per_conf ) / block_size;

          double ** data = init_2level_dtable ( num_data, nT + 2 );
          if ( data == NULL ) {
            fprintf ( stderr, "[xg_analyse] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }
    
          /**********************************************************
           * partially summed threep_44
           **********************************************************/
#pragma omp parallel for
#if 0
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {

            for ( int it = 0; it < nT; it++ ) {
              const int tau1 =   g_sequential_source_timeslice_list[idt]       / 2 - it;
              const int tau2 = ( g_sequential_source_timeslice_list[idt] + 1 ) / 2 + it;
              double dtmp = 0.;
              for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
                dtmp += threep_44[iconf][isrc][tau1][ireim] + threep_44[iconf][isrc][tau2][ireim];
              }
              data[iconf][it] = ( it == 0 ) ? dtmp : data[iconf][it-1] + dtmp;
            }
            for ( int it = 0; it < nT; it++ ) {
              data[iconf][it] /= (double)num_src_per_conf * 2 * ( it + 1 );
            }

            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
              /* COUNT FROM SOURCE 0 */
              int const tsink  = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
#if _XG_NUCLEON
              data[iconf][nT] += twop_orbit[igf][igi][iconf][isrc][tsink][ireim];
#else
              int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
              data[iconf][nT] += 0.5 * ( twop_orbit[igf][igi][iconf][isrc][tsink][ireim] + twop_orbit[igf][igi][iconf][isrc][tsink2][ireim] );
#endif
            }
            data[iconf][nT] /= (double)num_src_per_conf;

            data[iconf][nT + 1] = 0.;
            for ( int it = 0; it < T_global; it++ ) {
              data[iconf][nT + 1] += loop_sub[iconf][it];
            }
            data[iconf][nT + 1] /= (double)T_global;
          }
#endif
          for ( int i = 0; i < num_data; i++ ) {

            for ( int it = 0; it < nT; it++ ) {
              const int tau1 =   g_sequential_source_timeslice_list[idt]       / 2 - it;
              const int tau2 = ( g_sequential_source_timeslice_list[idt] + 1 ) / 2 + it;
              double dtmp = 0.;

              for ( int k = 0; k < block_size; k++ ) {
                int const idx  = ( ( i * block_size + k ) * T_global + tau1 ) * 2 + ireim;
                int const idx2 = ( ( i * block_size + k ) * T_global + tau2 ) * 2 + ireim;

                dtmp += threep_44[0][0][0][idx] + threep_44[0][0][0][idx2];
              }
              data[i][it] = ( it == 0 ) ? dtmp : data[i][it-1] + dtmp;
            }
            for ( int it = 0; it < nT; it++ ) {
              data[i][it] /= (double)block_size * 2 * ( it + 1 );
            }

            for ( int k = 0; k < block_size; k++ ) {
              /* COUNT FROM SOURCE 0 */
              int const tsink  = (  g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
              int const idx    = ( ( i * block_size + k ) * T_global + tsink ) * 2 + ireim;
#if _XG_NUCLEON
              data[i][nT] += twop_orbit[igf][igi][0][0][0][idx];
#else
              int const tsink2 = ( -g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
              int const idx2   = ( ( i * block_size + k ) * T_global + tsink2 ) * 2 + ireim;
              data[i][nT] += ( twop_orbit[igf][igi][0][0][0][idx] + twop_fold_propagator * twop_orbit[igf][igi][0][0][0][idx2] ) / ( 1 + twop_fold_propagator );
#endif
            }
            data[i][nT] /= (double)block_size;

            data[i][nT + 1] = 0.;
            for ( int it = 0; it < T_global; it++ ) {
              int const idx = ( ( i * block_size ) / num_src_per_conf ) * T_global + it;
              data[i][nT + 1] += loop_sub[0][idx];
            }
            data[i][nT + 1] /= (double)T_global;
          }

          char obs_name[500];
          sprintf ( obs_name, "ratio.sum.sub.%s.%s", obsname_tag, reim_str[ireim] );

          /* apply UWerr analysis */
          exitstatus = apply_uwerr_func ( data[0], num_data, nT+2, nT, narg, arg_first, arg_stride, obs_name, ratio_1_2_mi_3, dratio_1_2_mi_3 );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[xg_analyse] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
    
          fini_2level_dtable ( &data );
        }  /* end of loop on reim */

    
#endif  /* end of ifdef _RAT_SUB_METHOD */

        fini_4level_dtable ( &threep_44 );
    
      }  /* end of loop on dt */


#endif  /* end of if _RAT_METHOD */
  
    }}  /* end of loop on gi, gf */

#endif  /* end of if _LOOP_ANALYSIS */

    fini_2level_dtable ( &loop_sub );

  }  /* end of loop on smearing levels */

  fini_8level_dtable ( &twop );
  fini_6level_dtable ( &twop_orbit );
  fini_4level_dtable ( &loop );


  /**********************************************************
   * free and finalize
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
    fprintf(stdout, "# [xg_analyse] %s# [xg_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [xg_analyse] %s# [xg_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);
}
