/****************************************************
 * loop_extract
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
#include "table_init_z.h"
#include "table_init_d.h"
#include "table_init_i.h"
#include "contract_loop.h"
#include "ranlxd.h"

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to extract loop data\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "loop";

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int Qsq = -1;
  int stream = -1;
  int confid = -1;
  int exdef_nev = -1;
  int hier_prob_D = 0;
  char oet_type[20] = "NA";
  int nsample = 0;
  int nstep = 0;
  int sink_momentum_number = 0;

  struct timeval ta, tb;
  long unsigned int seconds, useconds;

  char output_filename[400];

  char data_tag[400];

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:Q:C:S:V:H:O:R:T:P:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'Q':
      Qsq = atoi ( optarg );
      break;
    case 'C':
      confid = atoi ( optarg );
      break;
    case 'S':
      stream = atoi ( optarg );
      break;
    case 'V':
      exdef_nev = atoi ( optarg );
      break;
    case 'H':
      hier_prob_D = atoi ( optarg );
      break;
    case 'O':
      strcpy ( oet_type, optarg );
      break;
    case 'R':
      nsample = atoi ( optarg );
      break;
    case 'T':
      nstep = atoi ( optarg );
      break;
    case 'P':
      sink_momentum_number = atoi ( optarg );
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
  /* fprintf(stdout, "# [loop_extract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [loop_extract] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[loop_extract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[loop_extract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [loop_extract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * loop data filename
   ***************************************************************************/
  sprintf ( filename, "loop_probD%d.%.4d_r%d_exact_NeV%d_Qsq%d.h5", hier_prob_D,  confid, stream, exdef_nev, Qsq );
 
  /* sprintf ( filename, "loop_probD%d.%.4d_r%d_stoch__NeV%d_Ns%.4d_step%.4d_Qsq%d.h5", hier_prob_D, confid, stream, exdef_nev, nsample, nstep, Qsq ); */

  if ( io_proc == 2 && g_verbose > 2 ) fprintf ( stdout, "# [loop_extract] loop filename = %s\n", filename );

  /***************************************************************************
   * loop normalization
   ***************************************************************************/
  double _Complex loop_norm = 0.;
  if ( strcmp ( oet_type, "dOp" ) == 0 ) {
    loop_norm = -4 * g_kappa;
  } else if ( strcmp ( oet_type, "Scalar" ) == 0 ) {
    loop_norm = -I * 8 * g_mu * g_kappa * g_kappa;
  }
  if ( g_verbose > 0 ) fprintf ( stdout, "# [loop_extract] loop_norm = %25.6e  %26.16e\n", creal ( loop_norm ), cimag ( loop_norm ) );

  /***************************************************************************
   * count momenta and build momentum list
   ***************************************************************************/
#if 0
  unsigned int sink_momentum_number = 0;
  for( int x1 = -LX_global/2+1; x1 < LX_global/2; x1++ ) {
  for( int x2 = -LY_global/2+1; x2 < LY_global/2; x2++ ) {
  for( int x3 = -LZ_global/2+1; x3 < LZ_global/2; x3++ ) {
    int const qq = x1*x1 + x2*x2 + x3*x3;
    if ( qq <= Qsq ) {
      sink_momentum_number++;
    }
  }}}
  if ( sink_momentum_number == 0 ) {
    fprintf ( stderr, "[loop_extract] Error, momentum list is empty %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  } else {
    if (io_proc == 2 && g_verbose > 1 ) fprintf ( stdout, "# [loop_extract] number of momenta <= %3d is %3d\n", Qsq, sink_momentum_number );
  }
#endif  /* of if 0 */

  if (io_proc == 2 && g_verbose > 1 ) fprintf ( stdout, "# [loop_extract] number of momenta <= %3d is %3d\n", Qsq, sink_momentum_number );

  int ** sink_momentum_list = init_2level_itable ( sink_momentum_number, 3 );
  if ( sink_momentum_list == NULL ) {
    fprintf(stderr, "[loop_extract] Error from init_2level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }

  exitstatus = loop_get_momentum_list_from_h5_file ( sink_momentum_list, filename, sink_momentum_number, io_proc );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[loop_extract] Error from loop_get_momentum_list_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  if ( g_verbose > 2 && io_proc == 2 ) {
    for ( int imom = 0; imom < sink_momentum_number; imom++ ) {
      fprintf ( stdout, " %3d  ( %3d, %3d, %3d)\n", imom, sink_momentum_list[imom][0], sink_momentum_list[imom][1], sink_momentum_list[imom][2] );
    }
  }

  /***************************************************************************
   * allocate memory for contractions
   ***************************************************************************/
  int * sink_momentum_matchid = init_1level_itable ( g_sink_momentum_number );
  for ( int i = 0; i < g_sink_momentum_number; i++ ) {
    sink_momentum_matchid[i] = -1;
    for ( int k = 0; k < sink_momentum_number; k++ ) {
      if ( ( g_sink_momentum_list[i][0] == sink_momentum_list[k][0] ) &&
           ( g_sink_momentum_list[i][1] == sink_momentum_list[k][1] ) &&
           ( g_sink_momentum_list[i][2] == sink_momentum_list[k][2] ) ) {
        sink_momentum_matchid[i] = k;
        break;
      }
    }
    if ( sink_momentum_matchid[i] == -1 ) {
      fprintf ( stderr, "[loop_extract] Error, no match found for g_sink_momentum %d %d %s %d\n", i, __FILE__, __LINE__ );
      EXIT(1);
    }
  }

  /***************************************************************************
   * exact part
   ***************************************************************************/

   /* allocate memory for contractions */
  double *** loop_exact = init_3level_dtable ( T, sink_momentum_number, 32 );
  if ( loop_exact == NULL ) {
    fprintf(stderr, "[loop_extract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  sprintf ( filename, "loop_probD%d.%.4d_r%d_exact_NeV%d_Qsq%d.h5", hier_prob_D,  confid, stream, exdef_nev, Qsq );
  if ( io_proc == 2 && g_verbose > 2 ) fprintf ( stdout, "# [loop_extract] loop filename = %s\n", filename );

  sprintf ( data_tag, "/conf_%.4d/%s/loop" , confid, oet_type );
  if ( g_verbose > 2 ) fprintf( stdout, "# [loop_extract] data_tag = %s\n", data_tag);

  exitstatus = loop_read_from_h5_file ( loop_exact, filename, data_tag, sink_momentum_number, 16, io_proc );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[loop_extract] Error from loop_read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  double _Complex **** zloop_exact = init_4level_ztable ( g_sink_momentum_number, T, 4, 4 );
  if ( zloop_exact == NULL ) {
    fprintf(stderr, "[loop_extract] Error from init_4level_ztable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
#pragma omp parallel for
    for ( int it = 0; it < T; it++ ) {
      /* transpose and normalize */
      for ( int ia = 0; ia < 4; ia++ ) {
      for ( int ib = 0; ib < 4; ib++ ) {
        zloop_exact[imom][it][ia][ib] = ( loop_exact[it][sink_momentum_matchid[imom] ][2*(4*ib+ia)] + loop_exact[it][sink_momentum_matchid[imom]][2*(4*ib+ia)+1] * I ) * loop_norm;
      }}
    }

    sprintf ( filename, "loop.%.4d.exact.%s.nev%d.PX%d_PY%d_PZ%d", confid, oet_type, exdef_nev, 
        g_sink_momentum_list[imom][0],
        g_sink_momentum_list[imom][1],
        g_sink_momentum_list[imom][2] );

    if ( g_verbose > 2 ) fprintf ( stdout, "# [loop_extract] loop filename = %s\n", filename );

    /* write to text file */
    FILE * ofs = fopen ( filename, "w" );

    for ( int it = 0; it < T; it++ ) {
      for ( int ia = 0; ia < 4; ia++ ) {
      for ( int ib = 0; ib < 4; ib++ ) {
        fprintf ( ofs, "%3d %d %d %25.16e %25.16e\n", it, ia, ib, creal( zloop_exact[imom][it][ia][ib] ), cimag ( zloop_exact[imom][it][ia][ib] ) );
      }}
    }

    fclose ( ofs );
  }

  fini_3level_dtable ( &loop_exact );
  fini_4level_ztable ( &zloop_exact );


  /***************************************************************************
   * stochastic part
   ***************************************************************************/

  double *** loop_stoch = init_3level_dtable ( T, sink_momentum_number, 32 );
  if ( loop_stoch == NULL ) {
    fprintf(stderr, "[loop_extract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  sprintf ( filename, "loop_probD%d.%.4d_r%d_stoch_NeV%d_Ns%.4d_step%.4d_Qsq%d.h5", hier_prob_D,  confid, stream, exdef_nev, nsample, nstep, Qsq );
  if ( g_verbose > 2 ) fprintf ( stdout, "# [loop_extract] loop filename = %s\n", filename );

  double _Complex **** zloop_stoch = init_4level_ztable ( g_sink_momentum_number, T, 4, 4 );
  if ( zloop_stoch == NULL ) {
    fprintf(stderr, "[loop_extract] Error from init_4level_ztable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  /***************************************************************************
   * loop on stochastic oet samples
   ***************************************************************************/
  for ( int isample = 0; isample < nsample; isample++ )
  {
    
    unsigned int const Nstoch = isample * nstep + 1;

    double _Complex const norm = loop_norm / Nstoch / ( hier_prob_D * hier_prob_D * hier_prob_D  );
    if ( g_verbose > 0 ) fprintf ( stdout, "# [loop_extract] norm stoch %25.16e %26.16e\n", creal( norm ), cimag ( norm ) );

    sprintf ( data_tag, "/conf_%.4d/Nstoch_%.4d/%s/loop" , confid, Nstoch, oet_type );
    if ( g_verbose > 2 ) fprintf( stdout, "# [loop_extract] data_tag = %s\n", data_tag);

    exitstatus = loop_read_from_h5_file ( loop_stoch, filename, data_tag, sink_momentum_number, 16, io_proc );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[loop_extract] Error from loop_read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    }

    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

      for ( int it = 0; it < T; it++ ) {
        /* transpose and normalize */
        for ( int ia = 0; ia < 4; ia++ ) {
        for ( int ib = 0; ib < 4; ib++ ) {
          zloop_stoch[imom][it][ia][ib] = ( loop_stoch[it][sink_momentum_matchid[imom]][2*(4*ib + ia)] + loop_stoch[it][sink_momentum_matchid[imom]][2*(4*ib + ia)+1] * I ) * norm;
        }}
      }

      sprintf ( filename, "loop.%.4d.stoch.%s.nev%d.Nstoch%d.PX%d_PY%d_PZ%d", confid, oet_type, exdef_nev, Nstoch,
          g_sink_momentum_list[imom][0],
          g_sink_momentum_list[imom][1],
          g_sink_momentum_list[imom][2] );

      if ( g_verbose > 2 ) fprintf ( stdout, "# [loop_extract] loop filename = %s\n", filename );

      FILE * ofs = fopen ( filename, "w" );

      for ( int it = 0; it < T; it++ ) {
        /* transpose and normalize */
        for ( int ia = 0; ia < 4; ia++ ) {
        for ( int ib = 0; ib < 4; ib++ ) {
          fprintf ( ofs, "%3d %d %d %25.16e %25.16e\n", it, ia, ib, creal( zloop_stoch[imom][it][ia][ib] ), cimag ( zloop_stoch[imom][it][ia][ib] ) );
        }}
      }

      fclose ( ofs );

    }

#if 0
#endif  /* of if 0 */

    /*****************************************************************/
    /*****************************************************************/

  }  /* end of loop on oet samples */

  fini_4level_ztable ( &zloop_stoch );
  fini_3level_dtable ( &loop_stoch );

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

  free_geometry();
  fini_2level_itable ( &sink_momentum_list );

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor ();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [loop_extract] %s# [loop_extract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [loop_extract] %s# [loop_extract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
