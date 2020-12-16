/****************************************************
 * p2gg_analyse_vdisc
 *
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

#define _TWOP_PERCONF 0
#define _TWOP_COMPACT 1

/* #define _USE_SUBTRACTED 1 */

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

  char const loop_type_tag[3][8] = { "NA", "dOp", "Scalar" };

  int const gamma_tmlqcd_to_binary[16] = { 8, 1, 2, 4, 0,  15, 7, 14, 13, 11, 9, 10, 12, 3, 5, 6 };

  /*                            gt  gx  gy  gz */
  int const gamma_v_list[4] = {  0,  1,  2,  3 };
 
  /*                             gtg5 gxg5 gyg5  gzg5 */
  int const gamma_a_list[4] = {  6,   7,   8,    9 };

  /*                            g5 */
  int const gamma_p_list[1] = { 5 };


  /***********************************************************
   * sign for g5 Gamma^\dagger g5
   *                          0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   *
   ***********************************************************/
  int const sigma_g5d[16] ={ -1, -1, -1, -1, +1, +1,  +1,  +1,  +1,  +1,  -1,  -1,  -1,  -1,  -1,  -1 };

  /***********************************************************
   * sign for gt Gamma gt
   *                          0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   *
   ***********************************************************/
  int const sigma_t[16] ={  1, -1, -1, -1, +1, -1,  -1,  +1,  +1,  +1,  -1,  -1,  -1,  +1,  +1,  +1 };

  /***********************************************************
   * sign for C Gamma C = sigma_C Gamma^T
   *                          0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   *
   ***********************************************************/
  int const sigma_C[16] ={  -1, -1, -1, -1, +1, +1,  +1,  +1,  +1,  +1,  -1,  -1,  -1,  -1,  -1,  -1 };

  /***********************************************************
   * sign for g0g5 Gamma g5g0 = sigma_50 Gamma
   *                          0,  1,  2,  3, id,  5, 0_5, 1_5, 2_5, 3_5, 0_1, 0_2, 0_3, 1_2, 1_3, 2_3
   *
   ***********************************************************/
  int const sigma_50[16] ={  -1, +1, +1, +1, +1, -1,  +1,  -1,  -1,  -1,  -1,  -1,  -1,  +1,  +1,  +1 };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "cA211a.30.32";
  struct timeval ta, tb;
  int operator_type = -1;
  int loop_type = -1;
  int loop_stats = 1;
  int twop_stats = 1;
  int write_data = 0;
  double loop_norm = 1.;
  int loop_transpose = 0;
  int loop_step = 1;
  int loop_nev = -1;
  int loop_use_es = 0;
  int loop_nstoch = 0;
  int charged_ps = 1;

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  char key[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "Cth?f:N:S:O:D:w:E:n:s:v:u:m:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_vdisc] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_vdisc] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'O':
      operator_type = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_vdisc] operator_type set to %d\n", operator_type );
      break;
    case 'D':
      loop_type = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_vdisc] loop_type set to %d\n", loop_type );
      break;
    case 'w':
      write_data = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_vdisc] write_data set to %d\n", write_data );
      break;
    case 'E':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [p2gg_analyse_vdisc] ensemble_name set to %s\n", ensemble_name );
      break;
    case 'n':
      loop_norm = atof ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_vdisc] loop_norm set to %f\n", loop_norm );
      break;
    case 't':
      loop_transpose = 1;
      fprintf ( stdout, "# [p2gg_analyse_vdisc] loop_transpose set to %d\n", loop_transpose );
      break;
    case 's':
      loop_step = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_vdisc] loop_step set to %d\n", loop_step );
      break;
    case 'v':
      loop_nev = atoi( optarg );
      fprintf ( stdout, "# [p2gg_analyse_vdisc] loop_nev set to %d\n", loop_nev );
      break;
    case 'u':
      loop_use_es = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_vdisc] loop_use_es set to %d\n", loop_use_es );
      break;
    case 'm':
      loop_nstoch = atoi ( optarg );
      fprintf ( stdout, "# [p2gg_analyse_vdisc] loop_nstoch set to %d\n", loop_nstoch );
      break;
    case 'C':
      charged_ps = 1;
      fprintf ( stdout, "# [p2gg_analyse_vdisc] charged_ps set to %d\n", charged_ps );
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
  /* fprintf(stdout, "# [p2gg_analyse_vdisc] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [p2gg_analyse_vdisc] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_analyse_vdisc] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_analyse_vdisc] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_analyse_vdisc] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_analyse_vdisc] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[p2gg_analyse_vdisc] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [p2gg_analyse_vdisc] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
 

  /***********************************************************
   * set flavor tag pair
   ***********************************************************/
  char flavor_tag[2][20] = { "NA", "NA" };
  if ( charged_ps == 1 ) {
    sprintf( flavor_tag[0], "u-gf-d-gi" );
    sprintf( flavor_tag[1], "d-gf-u-gi" );
  }

  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.lst" , ensemble_name);
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[p2gg_analyse_vdisc] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 6 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[p2gg_analyse_vdisc] Error from init_Xlevel_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [p2gg_analyse_vdisc] comment %s\n", line );
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
   * how to normalize loops
   ***********************************************************/
  if ( strcmp ( loop_type_tag[loop_type], "Scalar"  ) == 0 
    || strcmp ( loop_type_tag[loop_type], "Loops"   ) == 0
    || strcmp ( loop_type_tag[loop_type], "LoopsCv" ) == 0 ) {
    loop_norm *= -1;  /* -1 from single g5^ukqcd entering in the std-oet  */
  }
  if ( g_verbose > 0 ) fprintf ( stdout, "# [p2gg_analyse_vdisc] oet_type %s loop_norm = %25.16e\n", loop_type_tag[loop_type], loop_norm );

  /***********************************************************
   ***********************************************************
   **
   ** TWOP
   **
   ***********************************************************
   ***********************************************************/

  double ******* twop = init_7level_dtable ( num_conf, num_src_per_conf, g_sink_momentum_number, 2, 4, 4, 2 * T );
  if ( twop == NULL ) {
    fprintf(stderr, "[p2gg_analyse_vdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

#if _TWOP_PERCONF

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

      sprintf ( filename, "stream_%c/%s/%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", 
          conf_src_list[iconf][isrc][0], filename_prefix2, filename_prefix3, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );

      if ( g_verbose > 0 ) fprintf(stdout, "# [p2gg_analyse_vdisc] reading data from file %s %s %d\n", filename, __FILE__, __LINE__);
      affr = aff_reader ( filename );
      const char * aff_status_str = aff_reader_errstr ( affr );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[p2gg_analyse_vdisc] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }

      if( (affn = aff_reader_root( affr )) == NULL ) {
        fprintf(stderr, "[p2gg_analyse_vdisc] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
        return(103);
      }

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "p2gg_analyse_vdisc", "open-init-aff-reader", g_cart_id == 0 );
#endif

      /**********************************************************
       * loop on momenta
       **********************************************************/
      for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

        int const sink_momentum[3] = {
            g_sink_momentum_list[isink_momentum][0],
            g_sink_momentum_list[isink_momentum][1],
            g_sink_momentum_list[isink_momentum][2] };

        for ( int iflavor = 0; iflavor < 2; iflavor++ ) {
       
            /* using charged local */
            if ( charged_ps == 1 && operator_type == 1 ) {

              gettimeofday ( &ta, (struct timezone *)NULL );

              for( int mu = 0; mu < 4; mu++) {

                sprintf ( key , "/%s/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d/gi%.2d/px%dpy%dpz%d", correlator_prefix[operator_type], flavor_tag[iflavor],
                    gsx[0], gsx[1], gsx[2], gsx[3], gamma_a_list[mu], gamma_p_list[0], sink_momentum[0], sink_momentum[1], sink_momentum[2] );
  
                if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse_vdisc] key = %s\n", key );
                affdir = aff_reader_chpath (affr, affn, key );
                uint32_t uitems = T;
                exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(twop[iconf][isrc][isink_momentum][0][iflavor][mu]), uitems );
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[p2gg_analyse_vdisc] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }

              }  /* end of loop on mu */

              for( int mu = 0; mu < 4; mu++) {

                sprintf ( key , "/%s/%s/t%.2dx%.2dy%.2dz%.2d/gf%.2d/gi%.2d/px%dpy%dpz%d", correlator_prefix[operator_type], flavor_tag[iflavor],
                    gsx[0], gsx[1], gsx[2], gsx[3], gamma_p_list[0], gamma_a_list[mu], sink_momentum[0], sink_momentum[1], sink_momentum[2] );
  
                if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse_vdisc] key = %s\n", key );
                affdir = aff_reader_chpath (affr, affn, key );
                uint32_t uitems = T;
                exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(twop[iconf][isrc][isink_momentum][1][iflavor][mu]), uitems );
                if( exitstatus != 0 ) {
                  fprintf(stderr, "[p2gg_analyse_vdisc] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(105);
                }

              }  /* end of loop on mu */

              gettimeofday ( &tb, (struct timezone *)NULL );
              show_time ( &ta, &tb, "p2gg_analyse_vdisc", "read-ll-tensor-aff", g_cart_id == 0 );
            }  /* end of if operator_type */

        }  /* end of loop on flavor */
       
      }  /* end of loop on sink momenta */

#ifdef HAVE_LHPC_AFF
      aff_reader_close ( affr );
#endif  /* of ifdef HAVE_LHPC_AFF */

    }  /* end of loop on source locations */

  }   /* end of loop on configurations */

#endif  /* end of _TWOP_PERCONF */

#if _TWOP_COMPACT
  /**********************************************************
   **********************************************************
   **
   ** one flavor & momentum setup per file,
   ** all configs & source coords per file
   **
   **********************************************************
   **********************************************************/

  for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

      int const sink_momentum[3] = {
          g_sink_momentum_list[isink_momentum][0],
          g_sink_momentum_list[isink_momentum][1],
          g_sink_momentum_list[isink_momentum][2] };

      double const p[4] = {
          0., 
          TWO_MPI * (double)sink_momentum[0] / (double)LX_global,
          TWO_MPI * (double)sink_momentum[1] / (double)LY_global,
          TWO_MPI * (double)sink_momentum[2] / (double)LZ_global };

      for ( int iflavor = 0; iflavor < 2; iflavor++ ) {

        /**********************************************************
         * loop on vector index mu at sink
         **********************************************************/
        for( int mu = 0; mu < 4; mu++) {

#ifdef HAVE_LHPC_AFF
          /***********************************************
           * reader for aff input file
           ***********************************************/

          gettimeofday ( &ta, (struct timezone *)NULL );
          struct AffNode_s *affn = NULL, *affdir = NULL;

          if ( charged_ps == 1 ) {
            sprintf ( filename, "%s/%s.%s.gf%.2d.gi%.2d.px%dpy%dpz%d.aff", 
                filename_prefix2, correlator_prefix[operator_type], flavor_tag[iflavor], gamma_a_list[mu], gamma_p_list[0],
                sink_momentum[0], sink_momentum[1], sink_momentum[2] );
          }

          affr = aff_reader ( filename );
          const char * aff_status_str = aff_reader_errstr ( affr );
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[p2gg_analyse_vdisc] Error from aff_reader for filename %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
            EXIT(15);
          } else {
            if ( g_verbose > 0 ) fprintf(stdout, "# [p2gg_analyse_vdisc] reading data from file %s %s %d\n", filename, __FILE__, __LINE__);
          }

          if( (affn = aff_reader_root( affr )) == NULL ) {
            fprintf(stderr, "[p2gg_analyse_vdisc] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
            return(103);
          }

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "p2gg_analyse_vdisc", "open-init-aff-reader", g_cart_id == 0 );
#endif

          gettimeofday ( &ta, (struct timezone *)NULL );
          /***********************************************************
           * loop on configs and source locations per config
           ***********************************************************/
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
           
            for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

              /***********************************************************
               * copy source coordinates
               ***********************************************************/
              int const gsx[4] = {
                conf_src_list[iconf][isrc][2],
                conf_src_list[iconf][isrc][3],
                conf_src_list[iconf][isrc][4],
                conf_src_list[iconf][isrc][5] };

              sprintf ( key , "/stream_%c/conf_%d/t%.2dx%.2dy%.2dz%.2d", conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1], gsx[0], gsx[1], gsx[2], gsx[3] );
  
              if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse_vdisc] key = %s\n", key );
              affdir = aff_reader_chpath (affr, affn, key );
              uint32_t uitems = T_global;
              exitstatus = aff_node_get_complex ( affr, affdir,  (double _Complex*)(twop[iconf][isrc][isink_momentum][0][iflavor][mu]), uitems );
              if( exitstatus != 0 ) {
                fprintf(stderr, "[p2gg_analyse_vdisc] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }

            }  /* end of loop on source positions */
          }  /* end of loop on configs */
        
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "p2gg_analyse_vdisc", "read-ll-tensor-aff", g_cart_id == 0 );

#ifdef HAVE_LHPC_AFF
          aff_reader_close ( affr );
#endif  /* of ifdef HAVE_LHPC_AFF */

        }  /* end of loop on mu */

        /**********************************************************
         * loop on vector index mu at source
         **********************************************************/
        for( int mu = 0; mu < 4; mu++) {

#ifdef HAVE_LHPC_AFF
          /***********************************************
           * reader for aff input file
           ***********************************************/

          gettimeofday ( &ta, (struct timezone *)NULL );
          struct AffNode_s *affn = NULL, *affdir = NULL;

          if ( charged_ps == 1 ) {
            sprintf ( filename, "%s/%s.%s.gf%.2d.gi%.2d.px%dpy%dpz%d.aff", 
                filename_prefix2, correlator_prefix[operator_type], flavor_tag[iflavor], gamma_p_list[0], gamma_a_list[mu],
                sink_momentum[0], sink_momentum[1], sink_momentum[2] );
          }

          affr = aff_reader ( filename );
          const char * aff_status_str = aff_reader_errstr ( affr );
          if( aff_status_str != NULL ) {
            fprintf(stderr, "[p2gg_analyse_vdisc] Error from aff_reader for filename %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
            EXIT(15);
          } else {
            if ( g_verbose > 0 ) fprintf(stdout, "# [p2gg_analyse_vdisc] reading data from file %s %s %d\n", filename, __FILE__, __LINE__);
          }

          if( (affn = aff_reader_root( affr )) == NULL ) {
            fprintf(stderr, "[p2gg_analyse_vdisc] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
            return(103);
          }

          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "p2gg_analyse_vdisc", "open-init-aff-reader", g_cart_id == 0 );
#endif

          gettimeofday ( &ta, (struct timezone *)NULL );
          /***********************************************************
           * loop on configs and source locations per config
           ***********************************************************/
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
           
            for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

              /***********************************************************
               * copy source coordinates
               ***********************************************************/
              int const gsx[4] = {
                conf_src_list[iconf][isrc][2],
                conf_src_list[iconf][isrc][3],
                conf_src_list[iconf][isrc][4],
                conf_src_list[iconf][isrc][5] };

              sprintf ( key , "/stream_%c/conf_%d/t%.2dx%.2dy%.2dz%.2d", conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1], gsx[0], gsx[1], gsx[2], gsx[3] );
  
              if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_analyse_vdisc] key = %s\n", key );
              affdir = aff_reader_chpath (affr, affn, key );
              uint32_t uitems = T_global;
              exitstatus = aff_node_get_complex ( affr, affdir,  (double _Complex*)(twop[iconf][isrc][isink_momentum][1][iflavor][mu]), uitems );
              if( exitstatus != 0 ) {
                fprintf(stderr, "[p2gg_analyse_vdisc] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(105);
              }

            }  /* end of loop on source positions */
          }  /* end of loop on configs */
        
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "p2gg_analyse_vdisc", "read-ll-tensor-aff", g_cart_id == 0 );

#ifdef HAVE_LHPC_AFF
          aff_reader_close ( affr );
#endif  /* of ifdef HAVE_LHPC_AFF */

        }  /* end of loop on mu */

      }  /* end of loop on flavor */

  }  /* end of loop on sink momenta */

#endif  /* of _TWOP_COMPACT */

  /****************************************
   * show all data
   ****************************************/
  if ( g_verbose > 5 ) {
    gettimeofday ( &ta, (struct timezone *)NULL );

    for ( int iconf = 0; iconf < num_conf; iconf++ )
    {
      for( int isrc = 0; isrc < num_src_per_conf; isrc++ )
      {
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
        {
          for ( int mu = 0; mu < 4; mu++ )
          {
            for ( int it = 0; it < T; it++ )
            {
              fprintf ( stdout, "%c  %6d s %3d p %3d %3d %3d m %d twop %3d"\
                  "   %25.16e %25.16e    %25.16e %25.16e    %25.16e %25.16e    %25.16e %25.16e\n",
                  conf_src_list[iconf][isrc][0], conf_src_list[iconf][isrc][1], conf_src_list[iconf][isrc][2],
                  g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], mu, it, 
                  twop[iconf][isrc][imom][0][0][mu][2*it], twop[iconf][isrc][imom][0][0][mu][2*it+1],
                  twop[iconf][isrc][imom][0][1][mu][2*it], twop[iconf][isrc][imom][0][1][mu][2*it+1],
                  twop[iconf][isrc][imom][1][0][mu][2*it], twop[iconf][isrc][imom][1][0][mu][2*it+1],
                  twop[iconf][isrc][imom][1][1][mu][2*it], twop[iconf][isrc][imom][1][1][mu][2*it+1] );
            }
          }
        }
      }
    }
    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "p2gg_analyse_vdisc", "show-all-data", g_cart_id == 0 );
  }

  if ( twop_stats ) {
    /****************************************
     * combine source locations 
     ****************************************/
  
    double **** twop_src_avg = init_4level_dtable ( num_conf, g_sink_momentum_number, 4, 2 * T_global );
    if ( twop_src_avg == NULL ) {
      fprintf(stderr, "[p2gg_analyse_vdisc] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(16);
    }
  
#pragma omp parallel for
    for ( int iconf = 0; iconf < num_conf; iconf++ ) {
  
      double const norm = 1. / (double)num_src_per_conf;
  
      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
  
        for ( int mu = 0; mu < 4; mu++ ) {
          
          /* int const s5d_sign = ( charged_ps == 1 ) ? sigma_g5d[ gamma_a_list[mu] ] * sigma_g5d[ gamma_p_list[0] ] : 0.; */
          /* int const st_sign  = ( charged_ps == 1 ) ? sigma_t  [ gamma_a_list[mu] ] * sigma_t  [ gamma_p_list[0] ] : 0.; */
          int const sC_sign  = ( charged_ps == 1 ) ? sigma_C  [ gamma_a_list[mu] ] * sigma_C  [ gamma_p_list[0] ] : 0.;
          int const s50_sign = ( charged_ps == 1 ) ? sigma_50 [ gamma_a_list[mu] ] * sigma_50 [ gamma_p_list[0] ] : 0.;

          if ( g_verbose > 4 ) fprintf( stdout, "# [p2gg_analyse_vdisc] mu %d sC_sign %d s50_sign %d %s %d\n", mu, sC_sign, s50_sign, __FILE__, __LINE__ );
  
          for ( int tau = 0; tau < T_global; tau++ ) {

            double _Complex ztmp = 0.;

            for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            
              int const it = ( conf_src_list[iconf][isrc][2] + tau + T_global ) % T_global;

              double _Complex const ephase = cexp( -TWO_MPI * I * (
                  g_sink_momentum_list[imom][0] * conf_src_list[iconf][isrc][3] / (double)LX_global
                + g_sink_momentum_list[imom][1] * conf_src_list[iconf][isrc][4] / (double)LY_global
                + g_sink_momentum_list[imom][2] * conf_src_list[iconf][isrc][5] / (double)LZ_global ) );

              ztmp += 0.25 * (
                  +                     ( twop[iconf][isrc][imom][0][0][mu][2*it] + I * twop[iconf][isrc][imom][0][0][mu][2*it+1] ) * ephase
                  +           sC_sign * ( twop[iconf][isrc][imom][0][1][mu][2*it] + I * twop[iconf][isrc][imom][0][1][mu][2*it+1] ) * ephase
                + s50_sign * (
                  +                     ( twop[iconf][isrc][imom][1][0][mu][2*it] + I * twop[iconf][isrc][imom][1][0][mu][2*it+1] ) * ephase
                  +           sC_sign * ( twop[iconf][isrc][imom][1][1][mu][2*it] + I * twop[iconf][isrc][imom][1][1][mu][2*it+1] ) * ephase
                  )
                );
            }
              
            twop_src_avg[iconf][imom][mu][2*tau  ] = creal( ztmp ) * norm;
            twop_src_avg[iconf][imom][mu][2*tau+1] = cimag( ztmp ) * norm;
          }
        }
      }
    }  /* end of loop on conf */
  
    /****************************************
     * STATISTICAL ANALYSIS of real and
     * imaginary part of anti-symmetricly
     * orbit-averaged HVP spatial tensor
     * components
     ****************************************/
    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

      for ( int mu = 0; mu < 4; mu++ ) {

        for ( int ireim = 0; ireim < 2; ireim++ ) {

          double ** data = init_2level_dtable ( num_conf, T_global );
          if ( data == NULL ) {
            fprintf(stderr, "[p2gg_analyse_vdisc] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(16);
          }
  
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int it = 0; it < T_global; it++ ) {
              data[iconf][it] = twop_src_avg[iconf][imom][mu][2*it+ireim];
            }
          }
         
          char obs_name[100];
          sprintf ( obs_name, "%s.%s.gf%d_gi%d.latsymavg.px%dpx%dpz%d.%s", correlator_prefix[operator_type], flavor_tag[0],
              gamma_a_list[mu], gamma_p_list[0],
              g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], reim_str[ireim] );
  
          /* apply UWerr analysis */
          exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[p2gg_analyse_vdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }
          fini_2level_dtable ( &data );
  
        }  /* end of loop on re / im */
      }  /* end of loop on mu */
    }  /* end of loop on sink momenta */
  
    fini_4level_dtable ( &twop_src_avg );
  
  }  /* end of if twop_stats  */

  /**********************************************************
   **********************************************************
   **
   ** loop data for pgg
   **
   **********************************************************
   **********************************************************/

  for ( int iseq_source_mom = 0; iseq_source_mom < g_seq_source_momentum_number; iseq_source_mom++ ) {

    int const seq_source_momentum[3] = {
      g_seq_source_momentum_list[iseq_source_mom][0],
      g_seq_source_momentum_list[iseq_source_mom][1],
      g_seq_source_momentum_list[iseq_source_mom][2] };

    int const loop_momentum_number = g_sink_momentum_number;

    int ** loop_momentum_list = init_2level_itable ( loop_momentum_number, 3 );

    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
      loop_momentum_list[imom][0] = -seq_source_momentum[0] - g_sink_momentum_list[imom][0];
      loop_momentum_list[imom][1] = -seq_source_momentum[1] - g_sink_momentum_list[imom][1];
      loop_momentum_list[imom][2] = -seq_source_momentum[2] - g_sink_momentum_list[imom][2];
    }

    double **** loop_pgg = init_4level_dtable ( num_conf, loop_momentum_number, 4, 2*T_global );
    if ( loop_pgg == NULL ) {
      fprintf ( stderr, "[p2gg_analyse_vdisc] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(115);
    }

    for ( int imom= 0; imom < loop_momentum_number; imom++)
    {

      int const loop_momentum[3] = {
        loop_momentum_list[imom][0],
        loop_momentum_list[imom][1],
        loop_momentum_list[imom][2] };

      /**********************************************************
       * read loop matrix data
       **********************************************************/
      double ***** loops_matrix = init_5level_dtable ( num_conf, loop_nstoch, T, 4, 8 );
      if ( loops_matrix == NULL ) {
        fprintf ( stderr, "[p2gg_analyse_vdisc] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(112);
      }

      /**********************************************************
       * read the loop data from ASCII file
       **********************************************************/
      gettimeofday ( &ta, (struct timezone *)NULL );

      for ( int iconf = 0; iconf < num_conf; iconf++ ) {

        /**********************************************************
         *
         * read stochastic hp loop data
         *
         **********************************************************/
        if ( loop_nstoch > 0 && ( loop_use_es == 1 || loop_use_es == 3 ) ) {

          sprintf ( filename, "stream_%c/%s/loop.%.4d.stoch.%s.nev%d.Nstoch%d.PX%d_PY%d_PZ%d", conf_src_list[iconf][0][0], filename_prefix,
              conf_src_list[iconf][0][1], loop_type_tag[loop_type], loop_nev, loop_nstoch, loop_momentum[0], loop_momentum[1], loop_momentum[2] );
  
          if ( g_verbose > 0 ) fprintf ( stdout, "# [p2gg_analyse_vdisc] reading loop data from file %s %s %d\n", filename, __FILE__, __LINE__ );
          FILE * ofs = fopen ( filename, "r" );
          if ( ofs == NULL ) {
            fprintf ( stderr, "[p2gg_analyse_vdisc] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
            EXIT(113);
          }
  
          for ( int isample = 0; isample < loop_nstoch; isample++ ) {
            int itmp[3];
            double dtmp[2];
            for ( int t = 0; t < T_global; t++ ) {
              for ( int mu = 0; mu < 4; mu++ ) {
              for ( int nu = 0; nu < 4; nu++ ) {
                if ( fscanf ( ofs, "%d %d %d %lf %lf\n", itmp, itmp+1, itmp+2, dtmp, dtmp+1 ) != 5 ) {
                  fprintf ( stderr, "[p2gg_analyse_vdisc] Error from fscanf %s %d\n", __FILE__, __LINE__ );
                  EXIT(126);
                }
                loops_matrix[iconf][isample][t][mu][2*nu  ] += dtmp[0];
                loops_matrix[iconf][isample][t][mu][2*nu+1] += dtmp[1];
                /* show all data */
                if ( g_verbose > 5 ) {
                  fprintf ( stdout, "loop_mat %c c %6d s %3d t %3d m %d nu %d l %25.16e %25.16e\n", conf_src_list[iconf][0][0], conf_src_list[iconf][0][1], isample, t, mu, nu,
                      loops_matrix[iconf][isample][t][mu][2*nu], loops_matrix[iconf][isample][t][mu][2*nu+1] );
                }
                }}
            }
          }
          fclose ( ofs );
        }

        /**********************************************************
         *
         * read exact lm loop data
         *
         **********************************************************/
        if ( loop_nev > 0 && ( loop_use_es == 2 || loop_use_es == 3 ) ) {

          sprintf ( filename, "stream_%c/%s/loop.%.4d.exact.%s.nev%d.PX%d_PY%d_PZ%d", conf_src_list[iconf][0][0], filename_prefix,
              conf_src_list[iconf][0][1], loop_type_tag[loop_type], loop_nev, loop_momentum[0], loop_momentum[1], loop_momentum[2] );
  
          if ( g_verbose > 0 ) fprintf ( stdout, "# [p2gg_analyse_vdisc] reading loop data from file %s %s %d\n", filename, __FILE__, __LINE__ );
          FILE * ofs = fopen ( filename, "r" );
          if ( ofs == NULL ) {
            fprintf ( stderr, "[p2gg_analyse_vdisc] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
            EXIT(113);
          }
  
          for ( int isample = 0; isample < loop_nstoch; isample++ ) {
            int itmp[3];
            double dtmp[2];
            for ( int t = 0; t < T_global; t++ ) {
              for ( int mu = 0; mu < 4; mu++ ) {
              for ( int nu = 0; nu < 4; nu++ ) {
                if ( fscanf ( ofs, "%d %d %d %lf %lf\n", itmp, itmp+1, itmp+2, dtmp, dtmp+1 ) != 5 ) {
                  fprintf ( stderr, "[p2gg_analyse_vdisc] Error from fscanf %s %d\n", __FILE__, __LINE__ );
                  EXIT(126);
                }
                loops_matrix[iconf][isample][t][mu][2*nu  ] += dtmp[0];
                loops_matrix[iconf][isample][t][mu][2*nu+1] += dtmp[1];
                /* show all data */
                if ( g_verbose > 5 ) {
                  fprintf ( stdout, "loop_mat %c c %6d s %3d t %3d m %d nu %d l %25.16e %25.16e\n", conf_src_list[iconf][0][0], conf_src_list[iconf][0][1], isample, t, mu, nu,
                      loops_matrix[iconf][isample][t][mu][2*nu], loops_matrix[iconf][isample][t][mu][2*nu+1] );
                }
                }}
            }
          }
          fclose ( ofs );
 
        }  /* end of if loop_nev > 0 && ( loop_use_es == 2 || loop_use_es == 3 */

        /**********************************************************
         *
         * read stochastic volsrc loop data
         *
         **********************************************************/
        if ( loop_nstoch > 0 && loop_use_es == 4 ) {

          if ( loop_nev < 0 ) {
            sprintf ( filename, "stream_%c/%s/%d/loop.%d.stoch.%s.PX%d_PY%d_PZ%d", conf_src_list[iconf][0][0], filename_prefix,
              conf_src_list[iconf][0][1],
              conf_src_list[iconf][0][1], loop_type_tag[loop_type], loop_momentum[0], loop_momentum[1], loop_momentum[2] );
          } else {
            sprintf ( filename, "stream_%c/%s/loop.%.4d.stoch.%s.nev%d.PX%d_PY%d_PZ%d", conf_src_list[iconf][0][0], filename_prefix,
                conf_src_list[iconf][0][1], loop_type_tag[loop_type], loop_nev, loop_momentum[0], loop_momentum[1], loop_momentum[2] );
          }
  
          if ( g_verbose > 0 ) fprintf ( stdout, "# [p2gg_analyse_vdisc] reading loop data from file %s %s %d\n", filename, __FILE__, __LINE__ );
          FILE * ofs = fopen ( filename, "r" );
          if ( ofs == NULL ) {
            fprintf ( stderr, "[p2gg_analyse_vdisc] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
            EXIT(113);
          }
  
          for ( int isample = 0; isample < loop_nstoch; isample++ ) {
            int itmp[4];
            for ( int t = 0; t < T_global; t++ ) {
              for ( int mu = 0; mu < 4; mu++ ) {
              for ( int nu = 0; nu < 4; nu++ ) {
                if ( fscanf ( ofs, "%d %d %d %d %lf %lf\n", itmp, itmp+1, itmp+2, itmp+3, 
                     loops_matrix[iconf][isample][t][mu]+2*nu, loops_matrix[iconf][isample][t][mu]+2*nu+1 ) != 6 ) {
                  fprintf ( stderr, "[p2gg_analyse_vdisc] Error from fscanf %s %d\n", __FILE__, __LINE__ );
                  EXIT(126);
                }
                /* show all data */
                if ( g_verbose > 5 ) {
                  fprintf ( stdout, "loop_mat %c c %6d s %3d t %3d m %d nu %d l %25.16e %25.16e\n", conf_src_list[iconf][0][0], conf_src_list[iconf][0][1], isample, t, mu, nu,
                      loops_matrix[iconf][isample][t][mu][2*nu], loops_matrix[iconf][isample][t][mu][2*nu+1] );
                }
                }}
            }
          }
          fclose ( ofs );

        }  /* loop_nstoch > 0 && loop_use_es == 4 */

      }  /* end of loop on configurations */

      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "p2gg_analyse_vdisc", "read-loop-data", g_cart_id == 0 );

      /**********************************************************
       * loop on vector indices
       **********************************************************/
      for( int mu = 0; mu < 4; mu++ )
      {

        int const loop_gamma_id = gamma_v_list[mu];

        /**********************************************************
         * project loop matrices to spin structure
         **********************************************************/
        gamma_matrix_type gf;
        gamma_matrix_ukqcd_binary ( &gf, gamma_tmlqcd_to_binary[loop_gamma_id] );
        if ( g_verbose > 2 ) gamma_matrix_printf ( &gf, "gseq_ukqcd", stdout );

        double *** loops_proj = init_3level_dtable ( num_conf, loop_nstoch, 2*T_global );
        if ( loops_proj == NULL ) {
          fprintf ( stderr, "[p2gg_analyse_vdisc] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(112);
        }

        if ( g_verbose > 0 ) fprintf ( stdout, "# [p2gg_analyse_vdisc] WARNING: using loop_transpose = %d %s %d\n", loop_transpose, __FILE__, __LINE__ );
        project_loop ( loops_proj[0][0], gf.m, loops_matrix[0][0][0][0], num_conf * loop_nstoch * T_global, loop_transpose );

        /**********************************************************
         * average loop over samples for each config and timeslice
         **********************************************************/
    
        double *** loop_avg = init_3level_dtable ( num_conf, 4, 2*T_global );
        if ( loop_avg == NULL ) {
          fprintf ( stderr, "[p2gg_analyse_vdisc] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
          EXIT(115);
        }

#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          double const norm = loop_norm / (double)( loop_nstoch * loop_step );
          for ( int t = 0; t < T_global; t++ ) {
            loop_avg[iconf][mu][2*t  ] = 0.; 
            loop_avg[iconf][mu][2*t+1] = 0.; 

            for ( int isample = 0; isample < loop_nstoch; isample++ ) {
              loop_avg[iconf][mu][2*t  ] += loops_proj[iconf][isample][2*t  ];
              loop_avg[iconf][mu][2*t+1] += loops_proj[iconf][isample][2*t+1];
            }

            /* normalize */
            loop_avg[iconf][mu][2*t  ] *= norm;
            loop_avg[iconf][mu][2*t+1] *= norm;
          }
        }
      
        fini_3level_dtable ( &loops_proj ); 

        /**********************************************************
         * show averaged loops
         **********************************************************/
        if( g_verbose > 3 ) {
          sprintf ( filename, "loop.%s.px%dpy%dpz%d.g%d.nev%d.nstoch%d.es%d.corr", loop_type_tag[loop_type],
                  loop_momentum[0], loop_momentum[1], loop_momentum[2], loop_gamma_id,
                  loop_nev, loop_nstoch, loop_use_es );
 
          FILE *lfs = fopen ( filename, "w" );
          for ( int iconf = 0; iconf < num_conf; iconf++ ) {
            for ( int t = 0; t < T_global; t++ ) {
              fprintf ( lfs, "%25.16e %25.16e\n", loop_avg[iconf][mu][2*t], loop_avg[iconf][mu][2*t+1] ); 
            }
          }
          fclose ( lfs );
        }

        /**********************************************************
         *
         * statistical analysis for loops
         *
         **********************************************************/
        if ( loop_stats ) {
  
          dquant cumulants[4] = { cumulant_1, cumulant_2, cumulant_3, cumulant_4 };
          dquant dcumulants[4] = { dcumulant_1, dcumulant_2, dcumulant_3, dcumulant_4 };
  
          for ( int ireim = 0; ireim < 2; ireim++ ) {

            double ** data = init_2level_dtable ( num_conf, 4 * T_global );
  
            /**********************************************************
             * STATISTICAL ANALYSIS for loops 
             * simple loops
             **********************************************************/
#pragma omp parallel for
            for ( int iconf = 0; iconf < num_conf; iconf++ ) {
              for ( int it = 0; it < T_global; it++ ) {
                double const ltmp = loop_avg[iconf][mu][2*it+ireim];
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
              sprintf ( obs_name, "loop.%s.px%dpy%dpz%d.g%d.nev%d.nstoch%d.es%d.k%d.%s", loop_type_tag[loop_type],
                  loop_momentum[0], loop_momentum[1], loop_momentum[2], loop_gamma_id,
                  loop_nev, loop_nstoch, loop_use_es,
                  icum+1, reim_str[ireim] );
  
              /* apply UWerr analysis */
              int * arg_first  = init_1level_itable ( icum + 1 );
              int * arg_stride = init_1level_itable ( icum + 1 );
              for ( int k = 0; k <= icum ; k++ ) {
                arg_first[k]  = k * T_global;
                arg_stride[k] = 1;
              }
  
              exitstatus = apply_uwerr_func ( data[0], num_conf, 4*T_global, T_global, icum+1, arg_first, arg_stride, obs_name, cumulants[icum], dcumulants[icum] );
              if ( exitstatus != 0 ) {
                fprintf ( stderr, "[p2gg_analyse_vdisc] Error from apply_uwerr_func, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                EXIT(115);
              }
  
              fini_1level_itable ( &arg_first );
              fini_1level_itable ( &arg_stride );
  
            }  /* end of loop on cumulants */
  
            fini_2level_dtable ( &data );
  
          }  /* end of loop on reim */

        }  /* end of if loop_stats */

#pragma omp parallel for
        for ( int iconf = 0; iconf < num_conf; iconf++ ) {
          for ( int it = 0; it < T_global; it++ ) {
            for ( int ireim = 0; ireim < 2; ireim++ ) {
              loop_pgg[iconf][imom][mu][2*it + ireim ] = loop_avg[iconf][mu][2 * it + ireim ];
            }
          }
        }

        fini_3level_dtable ( &loop_avg );

      }  /* end of loop on vector components mu */

      fini_5level_dtable ( &loops_matrix );

    }  /* end of loop on loop momenta */

    fini_2level_itable ( &loop_momentum_list );


    /**********************************************************
     **********************************************************
     **
     ** P -> gg disconnected V-loop 3-point function
     **
     **********************************************************
     **********************************************************/

    /**********************************************************
     * loop on sequential source timeslices
     **********************************************************/
    for ( int isequential_source_timeslice = 0; isequential_source_timeslice < g_sequential_source_timeslice_number; isequential_source_timeslice++)
    {

      int const sequential_source_timeslice = g_sequential_source_timeslice_list[ isequential_source_timeslice ];

      /***********************************************************
       * prepare the disconnected contribution loop x hvp
       ***********************************************************/
 
      double ******* pgg_disc = init_7level_dtable ( 2, num_conf, num_src_per_conf, g_sink_momentum_number, 4, 4, 2 * T_global );
      if ( pgg_disc == NULL ) {
        fprintf(stderr, "[p2gg_analyse_vdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

      double ***** pgg_twop_1 = init_5level_dtable ( num_conf, g_sink_momentum_number, 4, 4, 2 );
      if ( pgg_twop_1 == NULL ) {
        fprintf(stderr, "[p2gg_analyse_vdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

      double ***** pgg_twop_2 = init_5level_dtable ( num_conf, g_sink_momentum_number, 4, 4, 2 * T_global );
      if ( pgg_twop_2 == NULL ) {
        fprintf(stderr, "[p2gg_analyse_vdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

      double **** pgg_loop_1 = init_4level_dtable ( num_conf, g_sink_momentum_number, 4, 2 * T_global );
      if ( pgg_loop_1 == NULL ) {
        fprintf(stderr, "[p2gg_analyse_vdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

      double **** pgg_loop_2 = init_4level_dtable ( num_conf, g_sink_momentum_number, 4, 2 );
      if ( pgg_loop_2 == NULL ) {
        fprintf(stderr, "[p2gg_analyse_vdisc] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {

        for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

          int const tsrc = conf_src_list[iconf][isrc][2];
          int const xsrc[3] = {
              conf_src_list[iconf][isrc][3],
              conf_src_list[iconf][isrc][4],
              conf_src_list[iconf][isrc][5] };
 
          /* ABSOLUTE sink time */
          int const tsnk = ( tsrc + sequential_source_timeslice + T_global ) % T_global;

          /* ABSOLUTE insertion time */
          int const tins = ( tsrc + sequential_source_timeslice + T_global ) % T_global;

          if ( g_verbose > 3 ) fprintf ( stdout, "# [p2gg_analyse_vdisc] tsrc %3d dt %3d tsnk %3d tins %3d\n", tsrc, sequential_source_timeslice, tsnk, tins );

          for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

            /* int const pf[3] = {
              g_sink_momentum_list[imom][0],
              g_sink_momentum_list[imom][1],
              g_sink_momentum_list[imom][2] }; */

            int const pi[3] = {
                g_seq_source_momentum_list[iseq_source_mom][0],
                g_seq_source_momentum_list[iseq_source_mom][1],
                g_seq_source_momentum_list[iseq_source_mom][2] };

            /* int const pc[3] = {
              -(pf[0] - pi[0]),
              -(pf[1] - pi[1]),
              -(pf[2] - pi[2]) 
            }; */

            double _Complex ephase = cexp(  TWO_MPI * I * (
                    xsrc[0] * pi[0] / (double)LX_global 
                  + xsrc[1] * pi[1] / (double)LY_global 
                  + xsrc[2] * pi[2] / (double)LZ_global ) );

            for ( int s1 = 0; s1 < 4; s1++ ) {
            for ( int s2 = 0; s2 < 4; s2++ ) {
                
              int st_sign = 0., sC_sign = 0., s50_sign = 0.;

              if ( charged_ps == 1 ) {
                sC_sign  = sigma_C [ gamma_v_list[s1] ] * sigma_C [ gamma_a_list[s2] ] * sigma_C [ gamma_p_list[0] ];

                st_sign  = sigma_t [ gamma_v_list[s1] ] * sigma_t [ gamma_a_list[s2] ] * sigma_t [ gamma_p_list[0] ];
                  
                s50_sign = sigma_50[ gamma_v_list[s1] ] * sigma_50[ gamma_a_list[s2] ] * sigma_50[ gamma_p_list[0] ];

              }
              if ( g_verbose > 4 ) fprintf ( stdout, "# [p2gg_analyse_vdisc] P A%d V%d sC_sign %d st_sign %d s50_sign %d\n", s2, s1, sC_sign, st_sign, s50_sign );

              /***********************************************************
               * (1) A_f P_i x V_c
               ***********************************************************/
              for ( int tau = 0; tau < T_global; tau++ ) {

                /* insertion time for V_c */
                int const it = ( tsnk + tau + T_global ) % T_global;

                double _Complex const z_twop_1 = twop[iconf][isrc][imom][0][0][s2][2*tsnk] + I * twop[iconf][isrc][imom][0][0][s2][2*tsnk+1];
                double _Complex const z_twop_2 = twop[iconf][isrc][imom][0][1][s2][2*tsnk] + I * twop[iconf][isrc][imom][0][1][s2][2*tsnk+1];

                double _Complex const z_loop   = loop_pgg[iconf][imom][s1][2*it] + I * loop_pgg[iconf][imom][s1][2*it+1];

                double _Complex const z_threep = ( z_twop_1 + sC_sign * z_twop_2 ) * z_loop * ephase;

                pgg_disc[0][iconf][isrc][imom][s1][s2][2*tau  ] = creal( z_threep );
                pgg_disc[0][iconf][isrc][imom][s1][s2][2*tau+1] = cimag( z_threep );

                if( tau == 0 ) {
                  double _Complex ztmp = ( z_twop_1 + sC_sign * z_twop_2 ) * ephase;
                  pgg_twop_1[iconf][imom][s1][s2][0] += creal( ztmp );
                  pgg_twop_1[iconf][imom][s1][s2][1] += cimag( ztmp );
                }

                if ( s2 == 0 ) {
                 pgg_loop_1[iconf][imom][s1][2*tau + 0] += loop_pgg[iconf][imom][s1][2*it+0];
                 pgg_loop_1[iconf][imom][s1][2*tau + 1] += loop_pgg[iconf][imom][s1][2*it+1];
                }


              }  /* end of loop on tau */

              /***********************************************************
               * (2) A_c P_i x V_f
               ***********************************************************/
              for ( int tau = 0; tau < T_global; tau++ ) {

                /* insertion time for A_c P_i */
                int const it = ( tsnk + tau + T_global ) % T_global;

                double _Complex const z_twop_1 = twop[iconf][isrc][imom][0][0][s2][2*it] + I * twop[iconf][isrc][imom][0][0][s2][2*it+1];
                double _Complex const z_twop_2 = twop[iconf][isrc][imom][0][1][s2][2*it] + I * twop[iconf][isrc][imom][0][1][s2][2*it+1];

                double _Complex const z_loop = loop_pgg[iconf][imom][s1][2*tsnk] + I * loop_pgg[iconf][imom][s1][2*tsnk+1];

                double _Complex const z_threep = ( z_twop_1 + sC_sign * z_twop_2 ) * z_loop * ephase;

                pgg_disc[1][iconf][isrc][imom][s1][s2][2*it  ] = creal( z_threep );
                pgg_disc[1][iconf][isrc][imom][s1][s2][2*it+1] = cimag( z_threep );

                double _Complex ztmp = ( z_twop_1 + sC_sign * z_twop_2 ) * ephase;
                pgg_twop_2[iconf][imom][s1][s2][2*tau+0] += creal( ztmp );
                pgg_twop_2[iconf][imom][s1][s2][2*tau+1] += cimag( ztmp );

                if ( s2 == 0 && tau == 0 ) {
                 pgg_loop_2[iconf][imom][s1][0] += loop_pgg[iconf][imom][s1][2*tsnk+0];
                 pgg_loop_2[iconf][imom][s1][1] += loop_pgg[iconf][imom][s1][2*tsnk+1];
                }
              }  /* end of loop on tau */

            }}  /* end of loop on vector indices s1, s2 */
          }  /* end of loop on momenta */

        }  /* end of loop on sources per conf */

      }  /* end of loop on configurations */
        
      /****************************************
       * pgg_disc source average
       ****************************************/
      double ****** pgg = init_6level_dtable ( 2, num_conf, g_sink_momentum_number, 4, 4, 2 * T_global );
      if ( pgg == NULL ) {
        fprintf(stderr, "[p2gg_analyse_vdisc] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(16);
      }

#pragma omp parallel for
      for ( int iconf = 0; iconf < num_conf; iconf++ ) {
        for ( int i = 0; i < g_sink_momentum_number * 32 * T_global; i++ ) {
          pgg[0][iconf][0][0][0][i] = 0.;
          pgg[1][iconf][0][0][0][i] = 0.;

          for ( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {
            pgg[0][iconf][0][0][0][i] += pgg_disc[0][iconf][isrc][0][0][0][i];
            pgg[1][iconf][0][0][0][i] += pgg_disc[1][iconf][isrc][0][0][0][i];
          }
          pgg[0][iconf][0][0][0][i] /= (double)num_src_per_conf;
          pgg[1][iconf][0][0][0][i] /= (double)num_src_per_conf;
        }
      }

      char const obs_name_tag[2][12] = { "v1v0p", "v0v1p" };

      for ( int k = 0; k <=1 ; k++ ) {

        /****************************************
         * statistical analysis for orbit average
         *
         * ASSUMES MOMENTUM LIST IS AN ORBIT AND
         * SEQUENTIAL MOMENTUM IS ZERO
         ****************************************/
        for ( int ireim = 0; ireim < 2; ireim++ ) {
  
          double ** data = init_2level_dtable ( num_conf, T_global );
          if ( data == NULL ) {
            fprintf ( stderr, "[p2gg_analyse_vdisc] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(79);
          }

          int const dim[2] = { num_conf, T_global };
          int **  momentum_list = init_2level_itable ( g_sink_momentum_number, 3 );
          memcpy ( momentum_list[0], g_sink_momentum_list[0], g_sink_momentum_number * 3 * sizeof(int) );
          antisymmetric_orbit_average_spatial ( data, pgg[k], dim, g_sink_momentum_number, momentum_list, ireim );
          fini_2level_itable ( &momentum_list );

          char obs_name[100];
          sprintf ( obs_name, "pgg_vdisc.%s.%s.%s.%s.nev%d.nstoch%d.es%d.orbit.QX%d_QY%d_QZ%d.g%d.t%d.PX%d_PY%d_PZ%d.%s", 
              obs_name_tag[k],
              correlator_prefix[operator_type], flavor_tag[0],
              loop_type_tag[loop_type], loop_nev, loop_nstoch, loop_use_es,
              seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], gamma_p_list[0], sequential_source_timeslice,
              g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2], reim_str[ireim] );

          /* apply UWerr analysis */
          exitstatus = apply_uwerr_real ( data[0], num_conf, T_global, 0, 1, obs_name );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[p2gg_analyse_vdisc] Error from apply_uwerr_real, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(1);
          }

          if ( write_data == 1 ) {
            sprintf ( filename, "%s.corr", obs_name );
            FILE * ofs = fopen ( filename, "w" );
            if ( ofs == NULL ) {
              fprintf ( stdout, "[p2gg_analyse_vdisc] Error from fopen for file %s %s %d\n", obs_name, __FILE__, __LINE__ );
              EXIT(12);
            }

            for ( int iconf = 0; iconf < num_conf; iconf++ ) {
              for ( int tau = -T_global/2+1; tau <= T_global/2; tau++ ) {
                int const it = ( tau < 0 ) ? tau + T_global : tau;

                fprintf ( ofs, "%5d%25.16e   %c %8d\n", tau, data[iconf][it], conf_src_list[iconf][0][0], conf_src_list[iconf][0][1] );
              }
            }
            fclose ( ofs );

          }  /* end of if write data */

          fini_2level_dtable ( &data );
        }  /* end of loop on real / imag */
#if _USE_SUBTRACTED

        /****************************************
         * statistical analysis for orbit average
         * including subtraction
         *
         * ASSUMES MOMENTUM LIST IS AN ORBIT AND
         * SEQUENTIAL MOMENTUM IS ZERO
         ****************************************/

        /****************************************
         * (1) < A_f P_i >_f x <V_c>_f
         ****************************************/
        if ( k == 0 ) {

          for ( int ireim = 0; ireim <= 1; ireim++ ) {
 
            double ** data = init_2level_dtable ( num_conf, g_sink_momentum_number * ( 16*T_global + 16 + 4*T_global ) );
            if ( data == NULL ) {
              fprintf ( stderr, "[p2gg_analyse_vdisc] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
              EXIT(79);
            }
STOPPED HERE
#pragma omp parallel for
            for ( int iconf = 0; iconf < num_conf; iconf++ ) {
              for ( int tau = 0; tau < T_global; tau++ ) {
                for ( int s1 = 0; s1 < 4; s1++ ) {
                  
                  data[iconf][16*tau+4*s1+s2] = pgg[k][iconf][imom][s1][s2][2*tau+ireim];


              }
            }

          } 

        } else {  /* k == 1 */
        }


#endif   /* of if _USE_SUBTRACTED  */

      }  /* end of loop on ordering of operators */
       
      /**********************************************************
       * free p2gg table
       **********************************************************/
      fini_7level_dtable ( &pgg_disc );
      fini_6level_dtable ( &pgg );
 
      fini_5level_dtable ( &pgg_twop_1 );
      fini_5level_dtable ( &pgg_twop_2 );
      fini_4level_dtable ( &pgg_loop_1 );
      fini_4level_dtable ( &pgg_loop_2 );

    }  /* end of loop on sequential source timeslices */

    fini_4level_dtable ( &loop_pgg );

  }  /* end of loop on seq source momentum */

  /**********************************************************
   * free the allocated memory, finalize
   **********************************************************/

  fini_7level_dtable ( &twop );

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
    fprintf(stdout, "# [p2gg_analyse_vdisc] %s# [p2gg_analyse_vdisc] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_analyse_vdisc] %s# [p2gg_analyse_vdisc] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
