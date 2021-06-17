/****************************************************
 * MM_analyse
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

#ifdef __cplusplus
extern "C"
{
#endif

#  ifdef HAVE_TMLQCD_LIBWRAPPER
#    include "tmLQCD.h"
#  endif

#ifdef __cplusplus
}
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
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "table_init_c.h"
#include "contract_diagrams.h"
#include "twopoint_function_utils.h"
#include "gamma.h"
#include "uwerr.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code form 2-pt function\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  EXIT(0);
}

typedef struct {
  int nc;
  int ns;
  int * conf;
  int ***src;
  char *stream;
} conf_src_list_type;

int main(int argc, char **argv) {
  
  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char ensemble_name[100] = "NA";
  char filename[100];
  int num_conf = 0, num_src_per_conf = 0;
  char streamc;
  int sink_momentum_number = -1, sink_momentum_id = -1;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:S:N:P:p:e:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'N':
      num_conf = atoi ( optarg );
      fprintf ( stdout, "# [MM_analyse] number of configs = %d\n", num_conf );
      break;
    case 'S':
      num_src_per_conf = atoi ( optarg );
      fprintf ( stdout, "# [MM_analyse] number of sources per config = %d\n", num_src_per_conf );
      break;
    case 'P':
      sink_momentum_number = atoi ( optarg );
      fprintf ( stdout, "# [MM_analyse] number of sink momenta set to = %d\n", sink_momentum_number );
      break;
    case 'p':
      sink_momentum_id = atoi ( optarg );
      fprintf ( stdout, "# [MM_analyse] sink momentum id set to = %d\n", sink_momentum_id );
      break;
    case 'e':
      strcpy ( ensemble_name, optarg );
      fprintf ( stdout, "# [MM_analyse] ensemble name set to = %s\n", ensemble_name );
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
  if(filename_set==0) strcpy(filename, "twopt.input");
  /* fprintf(stdout, "# [MM_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [MM_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [MM_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [MM_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[MM_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[MM_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  /***********************************************************
   * parity projection matrices
   ***********************************************************/
  gamma_matrix_type Pp_ukqcd, Pm_ukqcd;
  gamma_matrix_init ( &Pp_ukqcd );
  gamma_matrix_init ( &Pm_ukqcd );

  gamma_matrix_zero ( &Pp_ukqcd );
  /* ( 1 + gt ) / 2. */
  Pp_ukqcd.m[0][0] =  1.;
  Pp_ukqcd.m[1][1] =  1.;

  gamma_matrix_zero ( &Pm_ukqcd );
  /* ( 1 - gt ) / 2 */
  Pm_ukqcd.m[2][2] =  1.;
  Pm_ukqcd.m[3][3] =  1.;



  /***********************************************************
   * set io process
   ***********************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[MM_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [MM_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.nsrc%d.lst" , ensemble_name, num_src_per_conf);
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[MM_analyse] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__);
    EXIT(15);
  }

  conf_src_list_type conf_src_list;


  conf_src_list.conf = init_1level_itable ( num_conf );
  if ( conf_src_list.conf == NULL ) {
    fprintf(stderr, "[MM_analyse] Error from init_1level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  conf_src_list.src = init_3level_itable ( num_conf, num_src_per_conf, 4 );
  if ( conf_src_list.src == NULL ) {
    fprintf(stderr, "[MM_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  conf_src_list.stream = init_1level_ctable ( num_conf );
  if ( conf_src_list.stream == NULL ) {
    fprintf(stderr, "[MM_analyse] Error from init_1level_ctable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  conf_src_list.nc = num_conf;
  conf_src_list.ns = num_src_per_conf;

  char line[100];
  int countc = -1, counts=0;
  int conf_prev = -1;

  while ( fgets ( line, 100, ofs) != NULL && countc < num_conf && counts <= num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, "# [MM_analyse] comment %s\n", line );
      continue;
    }

    if( g_verbose > 4 ) fprintf ( stdout, "# [MM_analyse] line = \"%s\"\n", line );

    int conf_tmp, src_tmp[4];
    char stream_tmp;

    /***********************************************************
     * stream conf t x y z
     ***********************************************************/
    sscanf( line, "%c %d %d %d %d %d", &stream_tmp, &conf_tmp, src_tmp, src_tmp+1, src_tmp+2, src_tmp+3 );

    /* fprintf ( stdout, "# [MM_analyse] before: conf_tmp = %4d   conf_prev = %4d   countc = %d   counts = %d\n", conf_tmp, conf_prev, countc, counts ); */

    if ( conf_tmp != conf_prev ) {
      /* new config */
      countc++;
      counts=0;
      conf_prev = conf_tmp;

      conf_src_list.stream[countc] = stream_tmp;
      conf_src_list.conf[countc]   = conf_tmp;
    }

    /* fprintf ( stdout, "# [MM_analyse] after : conf_tmp = %4d   conf_prev = %4d   countc = %d   counts = %d\n", conf_tmp, conf_prev, countc, counts ); */

    memcpy ( conf_src_list.src[countc][counts] , src_tmp, 4*sizeof(int) );

    counts++;

#if 0
#endif
  }

  fclose ( ofs );

  /***********************************************************
   * show all configs and source locations
   ***********************************************************/
  if ( g_verbose > 4 ) {
    fprintf ( stdout, "# [MM_analyse] conf_src_list conf t x y z\n" );
    for ( int iconf = 0; iconf < conf_src_list.nc; iconf++ ) {
      for( int isrc = 0; isrc < conf_src_list.ns; isrc++ ) {
        fprintf ( stdout, "  %2c %6d %3d %3d %3d %3d\n", 
            conf_src_list.stream[iconf],
            conf_src_list.conf[iconf],
            conf_src_list.src[iconf][isrc][0],
            conf_src_list.src[iconf][isrc][1],
            conf_src_list.src[iconf][isrc][2],
            conf_src_list.src[iconf][isrc][3] );
      }
    }
  }

  fflush ( stdout );

  /***************************************************************************
   * loop on twopoint functions
   ***************************************************************************/
  for ( int i_2pt = 0; i_2pt < g_twopoint_function_number; i_2pt++ ) {

    twopoint_function_type * tp = &(g_twopoint_function_list[i_2pt]);

    twopoint_function_allocate ( tp );

    for ( int i_diag = 0; i_diag < tp->n; i_diag++ ) {

      char diagram_name[500];

      twopoint_function_get_diagram_name ( diagram_name,  tp, i_diag );

      char output_filename[400] = "NA";

      if ( strcmp ( tp->type , "b-b" ) == 0 ) {
        sprintf ( output_filename, "%s.%s", tp->name, diagram_name );
      }

      if ( g_verbose > 2 ) {
        fprintf ( stdout, "# [MM_analyse] output_filename = %s\n", output_filename );
      }

      FILE * ofs = fopen ( output_filename, "w" );

      /***********************************************************
       * field to store all data
       ***********************************************************/
      /* double *** corr = init_3level_dtable ( num_conf, num_src_per_conf * g_coherent_source_number, 2 * tp->T );
      if ( corr == NULL ) {
        fprintf ( stderr, "[MM_analyse] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      } */

      /***********************************************************
       * loop on configs 
       ***********************************************************/
      for( int iconf = 0; iconf < num_conf; iconf++ ) {
          
        int const Nconf = conf_src_list.conf[iconf];

        char const streamc = conf_src_list.stream[iconf];

        /***********************************************************
         * loop on sources per config
         ***********************************************************/
        for( int isrc = 0; isrc < num_src_per_conf; isrc++) {

          /***********************************************************
           * store the source coordinates
           ***********************************************************/
          int const gsx[4] = {
            conf_src_list.src[iconf][isrc][0],
            conf_src_list.src[iconf][isrc][1],
            conf_src_list.src[iconf][isrc][2],
            conf_src_list.src[iconf][isrc][3] };

          char key[500], data_filename[500];
          if ( strcmp ( tp->type , "m-m" ) == 0 ) {
            sprintf ( key, "/conf_%.4d/sx%.2dsy%.2dsz%.2dst%.2d/%s/%s", Nconf, gsx[1], gsx[2], gsx[3], gsx[0], tp->name, diagram_name );

            sprintf ( data_filename, "%s/%.4d_r%c/twop.%.4d_r%c_%s.%.2d.%.2d.%.2d.%.2d.h5", filename_prefix, Nconf, streamc, Nconf, streamc, filename_prefix2, 
                gsx[1], gsx[2], gsx[3], gsx[0] );
          }
          if ( g_verbose > 2 ) {
            fprintf ( stdout, "# [MM_analyse] key             = %s\n", key );
            fprintf ( stdout, "# [MM_analyse] data_filename   = %s\n", data_filename );
          }



#ifdef HAVE_HDF5
          /***********************************************************
           * read data block from h5 file
           ***********************************************************/
          double **** buffer = init_4level_dtable ( tp->T, sink_momentum_number, tp->d * tp->d, 2 );
          if ( buffer == NULL ) {
            fprintf(stderr, "[MM_analyse] Error from ,init_4level_dtable %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }

          exitstatus = read_from_h5_file ( (void*)(buffer[0][0][0]), data_filename, key, io_proc );
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[MM_analyse] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(12);
          }
#endif

          /***********************************************************
           * write into data field
           ***********************************************************/
          for ( int it = 0; it < tp->T; it++ ) {
            memcpy ( tp->c[i_diag][it][0], buffer[it][sink_momentum_id][0], tp->d * tp->d * 2*sizeof(double) );
          }

          fini_4level_dtable ( &buffer );

          /***********************************************************
           * finalize correlator
           ***********************************************************/

          /***********************************************************
           * write to ofs
           ***********************************************************/
          fprintf ( ofs, "# %s/%s\n", data_filename, key );
          for ( int it = 0; it < tp->T; it++ ) {
            fprintf ( ofs, "%3d %25.16e %25.16e\n" , it, 
                creal( tp->c[i_diag][it][0][0] ), cimag( tp->c[i_diag][it][0][0] ) );
          } 

          fflush ( ofs );

        }  /* end of loop on source locations */

      }  /* end of loop on configs */
      
      /***************************************************************************/
      /***************************************************************************/

      /***************************************************************************
       * free the correlator field
       ***************************************************************************/
      /* fini_3level_dtable ( &corr ); */

      /***************************************************************************
       * close output file
       ***************************************************************************/
      fclose ( ofs );

    }  /* end of loop on diagrams */

    twopoint_function_fini ( tp );

  }  /* end of loop on 2-point functions */



  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/
  fini_1level_itable ( &(conf_src_list.conf) );
  fini_3level_itable ( &(conf_src_list.src) );
  fini_1level_ctable ( &(conf_src_list.stream) );

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [MM_analyse] %s# [MM_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [MM_analyse] %s# [MM_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
