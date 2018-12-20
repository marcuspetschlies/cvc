/****************************************************
 * loop_analyse
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

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "contractions_io.h"
#include "Q_clover_phi.h"
#include "contract_cvc_tensor.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "Q_phi.h"
#include "clover.h"
#include "contract_loop.h"
#include "ranlxd.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1
#define _OP_ID_ST 2

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to analyse loop data\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "loop";

  /* const char fbwd_str[2][4] =  { "fwd", "bwd" }; */

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int Qsq = -1;

  struct timeval ta, tb;
  long unsigned int seconds, useconds;

  char output_filename[400];

  char data_tag[400];

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:Q:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'Q':
      Qsq = atoi ( optarg );
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
  if(filename_set==0) sprintf ( filename, "%s.input", outfile_prefix );
  /* fprintf(stdout, "# [loop_analyse] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [loop_analyse] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[loop_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***************************************************************************
   * some additional xchange operations
   ***************************************************************************/
  mpi_init_xchange_contraction(2);

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[loop_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [loop_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * count momenta and build momentum list
   ***************************************************************************/
  g_sink_momentum_number = 0;
  for( int x1 = 0; x1 < LX; x1++ ) {
  for( int x2 = 0; x2 < LY; x2++ ) {
  for( int x3 = 0; x3 < LZ; x3++ ) {
    int const qq = x1*x1 + x2*x2 + x3*x3;
    if ( qq <= Qsq ) {
      g_sink_momentum_list[g_sink_momentum_number][0] = x1;
      g_sink_momentum_list[g_sink_momentum_number][1] = x2;
      g_sink_momentum_list[g_sink_momentum_number][2] = x3;
      g_sink_momentum_number++;
    }
  }}}

  exitstatus = loop_get_momentum_list_from_h5_file ( g_sink_momentum_list, filename, g_sink_momentum_number, io_proc );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[] Error from loop_get_momentum_list_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  if ( g_verbose > 2 && io_proc == 2 ) {
    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
      fprintf ( stdout, " %3d  ( %3d, %3d, %3d)\n", imom, g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] );
    }
  }

  /***************************************************************************
   * allocate memory for contractions
   ***************************************************************************/
  double **** loop = init_4level_dtable ( g_nsample, T, g_sink_momentum_number, 32 );
  if ( loop == NULL ) {
    fprintf(stderr, "[loop_analyse] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  /***************************************************************************
   * loop data filename
   ***************************************************************************/
  sprintf ( filename, "%s.%.4d_%s_Ns%.4d_step%.4d_Qsq%d.h5", filename_prefix, Nconf, filename_prefix2, g_nsample, Nsave, Qsq );
  if ( io_proc == 2 && g_verbose > 2 ) fprintf ( stdout, "# [loop_analyse] loop filename = %s\n", filename );

  /***************************************************************************
   * loop on stochastic oet samples
   ***************************************************************************/
  for ( int isample = 0; isample < g_nsample; isample++ ) {

    int const Nstoch = isample * Nsave + 1;
    char loop_type[100];
    char loop_name[100];

    sprintf ( loop_type, "%s", "Scalar" );
    sprintf ( loop_name, "%s", "loop" );

    sprintf ( data_tag, "/conf_%.4d/Nstoch_%.4d/%s/%s", Nconf, Nstoch, loop_type, loop_name );
    if ( io_proc == 2 && g_verbose > 2 ) fprintf( stdout, "# [loop_analyse] data_tag = %s\n", data_tag);

    exitstatus = loop_read_from_h5_file ( loop[isample], filename, data_tag, g_sink_momentum_number, 16, io_proc );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[loop_analyse] Error from loop_read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    }

    if ( io_proc > 0 && g_verbose > 4 ) {
      /*****************************************************************
       * write in ASCII format
       *****************************************************************/
      for ( int iproc = 0; iproc < g_nproc_t; iproc++ ) {
        if ( g_tr_id == iproc ) {
          char output_filename[400];
          sprintf ( output_filename, "conf_%.4d.Nstoch_%.4d.%s.%s", Nconf, Nstoch, loop_type, loop_name );
          FILE * ofs = fopen ( output_filename, "w" );
          if ( ofs == NULL ) {
            fprintf ( stderr, "[loop_analyse] Error from fopen %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }

          for ( int x0 = 0; x0 < T; x0++ ) {
            int const y0 = x0 + g_proc_coords[0] * T;

            for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

              for( int ic = 0; ic < 16; ic++ ) {

                fprintf ( ofs, "%4d   %3d% 3d% 3d   %d %d  %25.16e %25.16e\n", y0, 
                    g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
                    ic/4, ic%4, loop[isample][x0][imom][2*ic], loop[isample][x0][imom][2*ic+1] );

              }  /* end of loop on components */
            }  /* end of loop on momenta */
          }  /* end of loop on time slices */

          fclose ( ofs );
        }  /* end of if g_tr_id == iproc */
#ifdef HAVE_MPI
        MPI_Barrier ( g_tr_comm );
#endif
      }  /* end of loop on procs in time direction */
    }  /* end of if io_proc > 0 and verbosity high level enough */

    /*****************************************************************/
    /*****************************************************************/

    /*****************************************************************
     * set loop type
     *****************************************************************/
    /*
     * sprintf ( loop_type, "%s", "dOp" );
     * same for gen-oet
     */

  }  /* end of loop on oet samples */

  /***************************************************************************
   * decallocate fields
   ***************************************************************************/
  fini_4level_dtable ( &loop );

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor ();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [loop_analyse] %s# [loop_analyse] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [loop_analyse] %s# [loop_analyse] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
