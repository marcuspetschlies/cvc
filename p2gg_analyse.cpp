/****************************************************
 * p2gg_analyse.c
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

#include "clover.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse P-J-J correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "p2gg";

  int c;
  int filename_set = 0;
  int check_momentum_space_WI = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int num_src_per_conf = 0;
  int num_conf = 0;
  char ensemble_name[100] = "cA211a.30.32";

#ifdef HAVE_LHPC_AFF
  struct AffReader_s *affr = NULL;
  char key[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
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
  /* fprintf(stdout, "# [p2gg_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [p2gg_analyse] git version = %s\n", g_gitversion);
  }

  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[p2gg_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  if ( g_verbose > 2 ) fprintf(stdout, "# [p2gg_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
   
  /***********************************************************
   * read list of configs and source locations
   ***********************************************************/
  sprintf ( filename, "source_coords.%s.nsrc%d.tab" , ensemble_name, num_src_per_conf);
  FILE *ofs = fopen ( filename, "r" );
  if ( ofs == NULL ) {
    fprintf(stderr, "[p2gg_analyse] Error from fopen %s %d\n", __FILE__, __LINE__);
    EXIT(15);
  }

  int *** conf_src_list = init_3level_itable ( num_conf, num_src_per_conf, 5 );
  if ( conf_src_list == NULL ) {
    fprintf(stderr, "[p2gg_analyse] Error from init_3level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }
  char line[100];
  int count = 0;
  while ( fgets ( line, 100, ofs) != NULL && count < num_conf * num_src_per_conf ) {
    if ( line[0] == '#' ) {
      fprintf( stdout, " # [p2gg_analyse] comment %s\n", line );
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

  /***********************************************************
   *
   ***********************************************************/
  double ****** hvp = init_6level_dtable ( g_sink_momentum_number, 4, 4, num_conf, num_src_per_conf, 2 * T );
  if ( hvp == NULL ) {
    fprintf(stderr, "[p2gg_analyse] Error from init_6level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(16);
  }

  /***********************************************************
   ***********************************************************
   **
   ** loop on configs and source locations
   **
   ***********************************************************
   ***********************************************************/

  for ( int iconf = 0; iconf < num_conf; iconf++ ) {

    for( int isrc = 0; isrc < num_src_per_conf; isrc++ ) {

      Nconf = conf_src_list[iconf][isrc][0];

      /***********************************************************
       * copy source coordinates
       ***********************************************************/
      int const gsx[4] = {
          conf_src_list[iconf][isrc][1],
          conf_src_list[iconf][isrc][2],
          conf_src_list[iconf][isrc][3],
          conf_src_list[iconf][isrc][4] };

      int source_proc_id = -1, sx[4];
      exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
      if( exitstatus != 0 ) {
        fprintf(stderr, "[p2gg_analyse] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(123);
      }

#ifdef HAVE_LHPC_AFF
      /***********************************************
       * writer for aff output file
       ***********************************************/
      struct AffNode_s *affn = NULL, *affdir = NULL;

      if(io_proc == 2) {
        sprintf ( filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
        fprintf(stdout, "# [p2gg_analyse] reading data to file %s\n", filename);
        affr = aff_reader ( filename );
        const char * aff_status_str = aff_reader_errstr ( affr );
        if( aff_status_str != NULL ) {
          fprintf(stderr, "[p2gg_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
          EXIT(15);
        }

        if( (affn = aff_reader_root( affr )) == NULL ) {
          fprintf(stderr, "[p2gg_analyse] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
          return(103);
        }

      }  /* end of if io_proc == 2 */
#endif

      /**********************************************************
       * loop on momenta
       **********************************************************/
      for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

        int const sink_momentum[3] = {
          g_sink_momentum_list[isink_momentum][0],
          g_sink_momentum_list[isink_momentum][1],
          g_sink_momentum_list[isink_momentum][2] };

        sprintf ( key , "/hvp/full/t%.2dx%.2dy%.2dz%.2d/px%.2d1py%.2dpz%.2d",
            gsx[0], gsx[1], gsx[2], gsx[3],
            sink_momentum[0], sink_momentum[1], sink_momentum[2] );

        double *** buffer = init_3level_dtable( 4, 4, 2 * T );
        if( buffer == NULL ) {
          fprintf(stderr, "[p2gg_analyse] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
          EXIT(15);
        }


        affdir = aff_reader_chpath (affr, affn, key );
        uint32_t uitems = 16 * T;
        exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*)(buffer[0][0]), uitems );
        if( exitstatus != 0 ) {
          fprintf(stderr, "[p2gg_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(105);
        }

        /**********************************************************
         * loop on shifts in directions mu, nu
         **********************************************************/
        for( int mu = 0; mu < 4; mu++) {
        for( int nu = 0; nu < 4; nu++) {

          double const p[4] = {
              0., 
              TWO_MPI * sink_momentum[0] / LX_global,
              TWO_MPI * sink_momentum[1] / LY_global,
              TWO_MPI * sink_momentum[2] / LZ_global };

          double const phase = - ( p[0] * gsx[0] + p[1] * gsx[1] + p[2] * gsx[2] + p[3] * gsx[3] ) + 0.5 * ( p[mu] - p[nu] );

          double _Complex ephase = cexp ( phase * I );

          /**********************************************************
           * sort data from buffer into hvp,
           * add source phase
           **********************************************************/
          for ( int it = 0; it < T; it++ ) {

            double _Complex ztmp = ( buffer[mu][nu][2*it] +  buffer[mu][nu][2*it+1] * I ) * ephase;

            hvp[isink_momentum][mu][nu][iconf][isrc][2*it  ] = creal( ztmp );
            hvp[isink_momentum][mu][nu][iconf][isrc][2*it+1] = cimag( ztmp );
          }

        }  /* end of loop on direction nu */
        }  /* end of loop on direction mu */

        fini_3level_dtable( &buffer );

      }  /* end of loop on sink momenta */

#ifdef HAVE_LHPC_AFF
      if(io_proc == 2) {
        aff_reader_close ( affr );
      }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

    }  /* end of loop on source locations */

  }   /* end of loop on configurations */

  /****************************************
   * now some statistical analysis
   ****************************************/

  for ( int isink_momentum = 0; isink_momentum < g_sink_momentum_number; isink_momentum++ ) {

    int const sink_momentum[3] = {
        g_sink_momentum_list[isink_momentum][0],
        g_sink_momentum_list[isink_momentum][1],
        g_sink_momentum_list[isink_momentum][1] };

    for( int mu = 0; mu < 4; mu++) {
    for( int nu = 0; nu < 4; nu++) {





    }}

  }


  /****************************************
   * free the allocated memory, finalize
   ****************************************/

  fini_6level_dtable ( &hvp );

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
    fprintf(stdout, "# [p2gg_analyse] %s# [p2gg_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [p2gg_analyse] %s# [p2gg_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
