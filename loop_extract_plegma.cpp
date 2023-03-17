/****************************************************
 * loop_extract_plegma
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
// #include "contract_loop.h"
#include "ranlxd.h"
#include "gamma.h"

using namespace cvc;

// #define _EXDEFL_LOOP
#define _STOCHASTIC_HP

void usage() {
  fprintf(stdout, "Code to extract loop data\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "loop";

  const char gamma_binary_to_string[16][12] = {
    "1",
    "gx",
    "gy",
    "gxgy",
    "gz",
    "gxgz",
    "gygz",
    "gxgygz",
    "gt",
    "gxgt",
    "gygt",
    "gxgygt",
    "gzgt",
    "gxgzgt",
    "gygzgt",
    "gxgygzgt" };

  const char gamma_cvc_to_string[16][12] = {
    "gt",
    "gx",
    "gy",
    "gz",
    "1",
    "g5",
    "gtg5",
    "gxg5",
    "gyg5",
    "gzg5",
    "gtgx",
    "gtgty",
    "gtgz",
    "gxgy",
    "gxgz",
    "gygz" };


  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[100];
  int Qsq = -1;
  int stream = -1;
  int confid = -1;
  int exdef_nev = 0;
  int hier_prob_D = 0;
  char oet_type[20] = "NA";
  char loop_type[20] = "NA";
  unsigned int nsample = 0;
  int nstep = 0;
  int sink_momentum_number = 0;

  struct timeval ta, tb;

  char data_tag_prefix[300];
  char data_tag[400];

  char flavor[20] = "NA";

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:L:Q:C:S:V:H:O:R:T:P:F:")) != -1) {
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
    case 'L':
      strcpy ( loop_type, optarg );
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
    case 'F':
      strcpy ( flavor, optarg );
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
  /* fprintf(stdout, "# [loop_extract_plegma] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [loop_extract_plegma] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[loop_extract_plegma] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    fprintf(stderr, "[loop_extract_plegma] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [loop_extract_plegma] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * loop data filename
   ***************************************************************************/
  sprintf ( filename, "%s_S1/%.4d_r%d/stoch_part_%s.h5", flavor, confid, stream, oet_type );
  // sprintf ( filename, "%.4d_r%d/exact_part_%s.h5", confid, stream, oet_type );

  if ( io_proc == 2 && g_verbose > 2 ) fprintf ( stdout, "# [loop_extract_plegma] loop filename = %s\n", filename );

  /***************************************************************************
   * loop normalization
   ***************************************************************************/
  double _Complex loop_norm = 0.;
  if ( strcmp ( oet_type, "gen" ) == 0  ) {
    loop_norm = -4 * g_kappa;
  } else if ( strcmp ( oet_type, "std" ) == 0 ) {
    loop_norm = -I * 8 * g_mu * g_kappa * g_kappa;
  }
  if ( g_verbose > 0 )
	  fprintf ( stdout, "# [loop_extract_plegma] oet_type %s loop_norm = %25.6e  %26.16e\n", oet_type, creal ( loop_norm ), cimag ( loop_norm ) );

  /***************************************************************************
   * potential directions
   ***************************************************************************/
  int const have_deriv = (
    strcmp( loop_type, "oneD"  ) == 0 ||
    strcmp( loop_type, "oneDC" ) == 0 );

  int const num_dir = have_deriv ? 4 : 1;
  if ( g_verbose > 4 ) fprintf ( stdout, "# [loop_extract_plegma] have_deriv %d num_dir %d\n", have_deriv, num_dir );

  /***************************************************************************
   *
   ***************************************************************************/
  int ** sink_momentum_list = init_2level_itable ( sink_momentum_number, 3 );
  if ( sink_momentum_list == NULL ) {
    fprintf(stderr, "[loop_extract_plegma] Error from init_2level_itable %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }

  char key[500];
  if ( have_deriv ) {
    sprintf ( key, "Conf%.4d_r%d/Ns%d/%s/dir0/mvec", confid, stream, 0, loop_type );
  } else {
    sprintf ( key, "Conf%.4d_r%d/Ns%d/%s/mvec", confid, stream, 0, loop_type );
    // sprintf ( key, "Conf%.4d_r%d/%s/mvec", confid, stream, loop_type );
  }
  exitstatus = read_from_h5_file ( (void*)(sink_momentum_list[0]), filename, key,  "int", io_proc );
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[loop_extract_plegma] Error from read_from_h5_file for %s %s, status was %d %s %d\n", filename, key,
        exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  if ( g_verbose > 2 && io_proc == 2 ) {
    for ( int imom = 0; imom < sink_momentum_number; imom++ ) {
      fprintf ( stdout, " %3d  ( %3d, %3d, %3d)\n", imom, sink_momentum_list[imom][0], sink_momentum_list[imom][1], sink_momentum_list[imom][2] );
    }
  }

  /***************************************************************************
   * allocate memory for contractions
   *
   * match g_sink_momentum == -sink_momentum
   ***************************************************************************/
  int * sink_momentum_matchid = init_1level_itable ( g_sink_momentum_number );
  for ( int i = 0; i < g_sink_momentum_number; i++ ) {
    sink_momentum_matchid[i] = -1;
    for ( int k = 0; k < sink_momentum_number; k++ ) {
      if ( ( g_sink_momentum_list[i][0] == -sink_momentum_list[k][0] ) &&
           ( g_sink_momentum_list[i][1] == -sink_momentum_list[k][1] ) &&
           ( g_sink_momentum_list[i][2] == -sink_momentum_list[k][2] ) ) {
        sink_momentum_matchid[i] = k;
        break;
      }
    }
    if ( sink_momentum_matchid[i] == -1 ) {
      fprintf ( stderr, "[loop_extract_plegma] Error, no match found for g_sink_momentum %d %s %d\n", i, __FILE__, __LINE__ );
      EXIT(1);
    } else {
      if ( g_verbose > 4 ) fprintf ( stdout, "# [loop_extract_plegma] momentum %3d p %3d, %3d %3d  machid %3d\n", i,
          g_sink_momentum_list[i][0], g_sink_momentum_list[i][1], g_sink_momentum_list[i][2], sink_momentum_matchid[i] ); 
    }
  }

  /***************************************************************************
   * exact part
   ***************************************************************************/
#ifdef _EXDEFL_LOOP
  if ( exdef_nev > 0 ) {
     /* allocate memory for contractions */
    double ***** loop_exact = init_5level_dtable ( T, 4, 4, sink_momentum_number, 2 );
    if ( loop_exact == NULL ) {
      fprintf(stderr, "[loop_extract_plegma] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__ );;
      EXIT(48);
    }

    char data_filename[400];
    // sprintf ( data_filename, "s1/%.4d_r%d/exact_part_%s.h5", confid, stream, oet_type );
    sprintf ( data_filename, "%.4d_r%d/exact_part_%s.h5", confid, stream, oet_type );
    if ( io_proc == 2 && g_verbose > 2 ) fprintf ( stdout, "# [loop_extract_plegma] loop filename = %s\n", data_filename );

    sprintf ( data_tag_prefix, "/Conf%.4d_r%d/%s" , confid, stream, loop_type );

    for ( int idir = 0; idir < num_dir; idir++ ) {

      if ( have_deriv ) {
        sprintf ( data_tag, "%s/dir%d/loop" , data_tag_prefix, idir );
      } else {
        sprintf ( data_tag, "%s/loop" , data_tag_prefix );
      }

      if ( g_verbose > 2 ) fprintf( stdout, "# [loop_extract_plegma] data_tag = %s\n", data_tag);

      exitstatus = read_from_h5_file ( (void*)(loop_exact[0][0][0][0]), data_filename, data_tag, "double", io_proc );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[loop_extract_plegma] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      double _Complex **** zloop_exact = init_4level_ztable ( g_sink_momentum_number, T, 4, 4 );
      if ( zloop_exact == NULL ) {
        fprintf(stderr, "[loop_extract_plegma] Error from init_4level_ztable %s %d\n", __FILE__, __LINE__ );;
        EXIT(48);
      }

      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
#pragma omp parallel for
        for ( int it = 0; it < T; it++ ) {
          /* transpose and normalize */
          for ( int ia = 0; ia < 4; ia++ ) {
          for ( int ib = 0; ib < 4; ib++ ) {
            zloop_exact[imom][it][ia][ib] = ( 
			    loop_exact[it][ib][ia][sink_momentum_matchid[imom] ][ 0 ] 
			  + loop_exact[it][ib][ia][sink_momentum_matchid[imom] ][ 1 ] * I ) * loop_norm;
          }}
        }
 
        if ( have_deriv ) {
          sprintf ( filename, "loop.%.4d_r%d.exact.%s.%s.nev%d.mu%d.PX%d_PY%d_PZ%d", confid, stream, oet_type, loop_type, exdef_nev, idir, 
              g_sink_momentum_list[imom][0],
              g_sink_momentum_list[imom][1],
              g_sink_momentum_list[imom][2] );
        } else  {
          sprintf ( filename, "loop.%.4d_r%d.exact.%s.%s.nev%d.PX%d_PY%d_PZ%d", confid, stream, oet_type, loop_type, exdef_nev,
              g_sink_momentum_list[imom][0],
              g_sink_momentum_list[imom][1],
              g_sink_momentum_list[imom][2] );
        }

        if ( g_verbose > 2 ) fprintf ( stdout, "# [loop_extract_plegma] loop filename = %s\n", filename );

        /* write to text file */
        FILE * ofs = fopen ( filename, "w" );

        for ( int it = 0; it < T; it++ ) {
          for ( int ia = 0; ia < 4; ia++ ) {
          for ( int ib = 0; ib < 4; ib++ ) {
            fprintf ( ofs, "%3d %d %d %25.16e %25.16e\n", it, ia, ib, creal( zloop_exact[imom][it][ia][ib] ), cimag ( zloop_exact[imom][it][ia][ib] ) );
          }}
        }

        fclose ( ofs );

      }  /* end of loop on momenta */

      fini_4level_ztable ( &zloop_exact );

    }  /* end of loop on direction */

    fini_5level_dtable ( &loop_exact );

  }  /* of if exdef_nev > 0 */

#endif  /* _EXDEFL_LOOP */

  /***************************************************************************
   * stochastic part
   ***************************************************************************/
#ifdef _STOCHASTIC_HP
  double ***** loop_stoch = init_5level_dtable ( T, 4, 4, sink_momentum_number, 2 );
  if ( loop_stoch == NULL ) {
    fprintf(stderr, "[loop_extract_plegma] Error from init_5level_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  double _Complex ****** zloop_stoch = init_6level_ztable ( (nsample/nstep), num_dir, g_sink_momentum_number, T, 4, 4 );
  if ( zloop_stoch == NULL ) {
    fprintf(stderr, "[loop_extract_plegma] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

  double _Complex const norm = loop_norm / ( hier_prob_D * hier_prob_D * hier_prob_D  );
  if ( g_verbose > 0 ) fprintf ( stdout, "# [loop_extract_plegma] norm Nstoch %25.16e %26.16e\n", creal( norm ), cimag ( norm ) );

  /***************************************************************************
   * loop on stochastic samples
   ***************************************************************************/
  for ( unsigned int isample = 0; isample < nsample/nstep; isample ++ ) {

    unsigned int const Nstoch = ( isample + 1 ) * nstep;
  
    char data_filename[400];
    sprintf ( data_filename, "%s_S%d/%.4d_r%d/stoch_part_%s.h5", flavor, Nstoch, confid, stream, oet_type );
    if ( g_verbose > 2 ) fprintf ( stdout, "# [loop_extract_plegma] loop filename = %s\n", data_filename );

    sprintf ( data_tag_prefix, "/Conf%.4d_r%d/Ns%d/%s" , confid, stream, 0, loop_type );

    for ( int idir = 0; idir < num_dir; idir++ ) {

      if ( have_deriv ) {
        sprintf ( data_tag, "%s/dir%d/loop" , data_tag_prefix, idir );
      } else {
        sprintf ( data_tag, "%s/loop" , data_tag_prefix );
      }

      if ( g_verbose > 2 ) fprintf( stdout, "# [loop_extract_plegma] data_tag = %s\n", data_tag);

      exitstatus = read_from_h5_file ( (void*)(loop_stoch[0][0][0][0]), data_filename, data_tag, "double", io_proc );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[loop_extract_plegma] Error from read_from_h5_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
#pragma omp parallel for
        for ( int it = 0; it < T; it++ ) {
          /* transpose and normalize */
          for ( int ia = 0; ia < 4; ia++ ) {
          for ( int ib = 0; ib < 4; ib++ ) {
            zloop_stoch[isample][idir][imom][it][ia][ib] = ( 
			    loop_stoch[it][ib][ia][sink_momentum_matchid[imom]][ 0 ]
			  + loop_stoch[it][ib][ia][sink_momentum_matchid[imom]][ 1 ] * I
			 ) * norm;
          }}
        }

      }  /* end of loop on momenta */
    }  /* end of loop on directions */
  }  /* end of loop on samples */

  /***************************************************************************
   * accumulate
   ***************************************************************************/
  double _Complex ***** zloop_stoch_accum = init_5level_ztable ( num_dir, g_sink_momentum_number, T, 4, 4 );
  if ( zloop_stoch_accum == NULL ) {
    fprintf(stderr, "[loop_extract_plegma] Error from init_Xlevel_Ytable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

#pragma omp parallel for
  for ( unsigned int ix = 0; ix < num_dir * g_sink_momentum_number * T * 16; ix++ )
  {
    zloop_stoch_accum[0][0][0][0][ix] = 0.;
    for ( unsigned int isample = 0; isample < nsample/nstep; isample ++ )
    {
      zloop_stoch_accum[0][0][0][0][ix] += zloop_stoch[isample][0][0][0][0][ix];
    }
    zloop_stoch_accum[0][0][0][0][ix] /= (double)(nsample);
  }


  /***************************************************************************
   * write to file
   ***************************************************************************/
  for ( int idir = 0; idir < num_dir; idir++ ) {
    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

      if ( have_deriv ) {
        sprintf ( filename, "loop.%.4d_r%d.stoch.%s.%s.nev%d.Nstoch%d.mu%d.PX%d_PY%d_PZ%d", confid, stream, oet_type, loop_type, exdef_nev, nsample, idir,
            g_sink_momentum_list[imom][0],
            g_sink_momentum_list[imom][1],
            g_sink_momentum_list[imom][2] );
      } else {
        sprintf ( filename, "loop.%.4d_r%d.stoch.%s.%s.nev%d.Nstoch%d.PX%d_PY%d_PZ%d", confid, stream, oet_type, loop_type, exdef_nev, nsample,
            g_sink_momentum_list[imom][0],
            g_sink_momentum_list[imom][1],
            g_sink_momentum_list[imom][2] );
      } 


      FILE * ofs = NULL;

      if ( g_verbose > 2 ) fprintf ( stdout, "# [loop_extract_plegma] loop filename = %s\n", filename );

      ofs = fopen ( filename, "w" );
      if ( ofs == NULL ) {
        fprintf ( stderr, "[loop_extract_plegma] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
        EXIT(23);
      }

      for ( unsigned int isample = 0; isample < nsample/nstep; isample ++ ) {

        for ( int it = 0; it < T; it++ ) {
          /* transpose and normalize */
          for ( int ia = 0; ia < 4; ia++ ) {
          for ( int ib = 0; ib < 4; ib++ ) {
            fprintf ( ofs, "%3d %d %d %25.16e %25.16e\n", it, ia, ib,
			    creal ( zloop_stoch[isample][idir][imom][it][ia][ib] ),
			    cimag ( zloop_stoch[isample][idir][imom][it][ia][ib] ) );
          }}
        }
  
      }  /* end of loop on samples */

      fclose ( ofs );
#if 0
      strcat ( filename, ".avg" );
      ofs = fopen ( filename, "w" );
      if ( ofs == NULL ) {
        fprintf ( stderr, "[loop_extract_plegma] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
        EXIT(23);
      }

      if ( g_verbose > 2 ) fprintf ( stdout, "# [loop_extract_plegma] loop filename = %s\n", filename );

      for ( int it = 0; it < T; it++ ) {
        for ( int ia = 0; ia < 4; ia++ ) {
        for ( int ib = 0; ib < 4; ib++ ) {
          fprintf ( ofs, "%3d %d %d %25.16e %25.16e\n", it, ia, ib,
                          creal ( zloop_stoch_accum[idir][imom][it][ia][ib] ),
                          cimag ( zloop_stoch_accum[idir][imom][it][ia][ib] ) );
        }}
      }

      fclose ( ofs );
#endif
    }  /* end of loop on momenta */
  }  /* end of loop on directions */

  fini_6level_ztable ( &zloop_stoch );
  fini_5level_dtable ( &loop_stoch );

#endif  /* of _STOCHASTIC_HP */

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

  free_geometry();
  fini_2level_itable ( &sink_momentum_list );
  fini_1level_itable ( &sink_momentum_matchid );

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor ();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [loop_extract_plegma] %s# [loop_extract_plegma] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [loop_extract_plegma] %s# [loop_extract_plegma] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
