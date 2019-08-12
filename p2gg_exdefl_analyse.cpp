/****************************************************
 * p2gg_exdefl_analyse
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
#include "read_input_parser.h"
#include "contractions_io.h"
#include "scalar_products.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "clover.h"

#define _USE_ATA_TENSOR
#define _USE_PTA_TENSOR

/****************************************************
 * return momentum id from list according
 * to momentum conservation
 ****************************************************/
int get_momentum_id ( int const p[3], int const q[3], int const n, int (* const r)[3] ) {

  int const s[3] = {
    -( p[0] + q[0] ),
    -( p[1] + q[1] ),
    -( p[2] + q[2] ) };

  for ( int i = 0; i < n; i++ ) {
    if ( r[i][0] == s[0] && r[i][1] == s[1] && r[i][2] == s[2] ) return ( i );
  }
  return( -1 );
}  /* end of get_momentum_id */

using namespace cvc;

/****************************************************
 * main program
 ****************************************************/
int main(int argc, char **argv) {

  char const infile_prefix[] = "p2gg";
  /* char const outfile_prefix[] = "p2gg_exdefl_analyse"; */
  double const TWO_MPI = 2. * M_PI;

  int c;
  int filename_set = 0;
  /* int check_position_space_WI=0; */
  int exitstatus;
  char filename[100];
  int evecs_num = 0;
  struct timeval ta, tb, start_time, end_time;
  char key[400];
  int evecs_use_step = -1;
  int evecs_use_min  = -1;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:n:s:m:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
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
      exit(1);
      break;
    }
  }

  gettimeofday ( &start_time, (struct timezone *)NULL );


  /* set the default values */
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [p2gg_exdefl_analyse] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [p2gg_exdefl_analyse] git version = %s\n", g_gitversion);
  }


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_exdefl_analyse] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_exdefl_analyse] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_exdefl_analyse] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_exdefl_analyse] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************************
   * set io process
   ***********************************************************/
  int io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[p2gg_exdefl_analyse] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [p2gg_exdefl_analyse] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

#ifdef HAVE_LHPC_AFF
  /***********************************************************
   * reader for aff input file
   ***********************************************************/
  struct AffReader_s *affr = NULL;
  sprintf ( filename, "%s.%.4d.nev%d.aff", infile_prefix, Nconf, evecs_num );
  fprintf(stdout, "# [p2gg_exdefl_analyse] reading data from file %s\n", filename);
  affr = aff_reader ( filename );
  char * aff_status_str = (char*)aff_reader_errstr ( affr );
  if( aff_status_str != NULL ) {
    fprintf(stderr, "[p2gg_exdefl_analyse] Error from aff_reader, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
    EXIT(15);
  }

  /* set root node */
  struct AffNode_s * affrn = aff_reader_root( affr );
  if( affrn == NULL ) {
    fprintf(stderr, "[p2gg_exdefl_analyse] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
    EXIT(17);
  }

  /***********************************************************
   * writer for aff output file
   ***********************************************************/
  struct AffWriter_s *affw = NULL;
  sprintf ( filename, "%s.%.4d.nev%d.aff", g_outfile_prefix, Nconf, evecs_num );
  fprintf(stdout, "# [p2gg_exdefl_analyse] writing data to file %s\n", filename);
  affw = aff_writer ( filename );
  aff_status_str = (char*)aff_writer_errstr ( affw );
  if( aff_status_str != NULL ) {
    fprintf(stderr, "[p2gg_exdefl_analyse] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
    EXIT(16);
  }

  struct AffNode_s * affwn = aff_writer_root( affw );
  if( affwn == NULL ) {
    fprintf(stderr, "[p2gg_exdefl_analyse] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
    EXIT(17);
  }

#else
#error "need lhp-aff lib; currently no other output method implemented"
#endif

  /***********************************************************
   *
   * read the eigenvalues
   *
   ***********************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

  /***********************************************************
   * allocate and read
   ***********************************************************/
  double * evecs_eval = init_1level_dtable ( evecs_num );
  if ( evecs_eval == NULL ) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(11);
  }

  double * evecs_eval_inv = init_1level_dtable ( evecs_num );
  if ( evecs_eval_inv == NULL ) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(11);
  }

  /* AFF read */
  uint32_t uitems = ( uint32_t )evecs_num;

  sprintf ( key, "/eval/C%d/N%d", Nconf, evecs_num );
  if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_exdefl_analyse] reading key %s %s %d\n", key, __FILE__, __LINE__ );

  struct AffNode_s * affdir = aff_reader_chpath ( affr, affrn, key );
  if ( affdir == NULL ) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_writer_mkpath %s %d\n", __FILE__, __LINE__);
    EXIT(15);
  }

  exitstatus = aff_node_get_double ( affr, affdir, evecs_eval, uitems );
  if(exitstatus != 0) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_node_get_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(16);
  }

  for ( int iw = 0; iw < evecs_num; iw++ ) {
    evecs_eval_inv[iw] = 1. / evecs_eval[iw];
  }

  if ( g_verbose > 2 ) {
    for ( int iw = 0; iw < evecs_num; iw++ ) {
      fprintf( stdout, "# [p2gg_exdefl_analyse] eval %6d   %25.16e   %25.16e\n", iw, evecs_eval[iw] , evecs_eval_inv[iw] );
    }
  }

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "p2gg_exdefl_analyse", "aff-read-evecs", g_cart_id == 0 );

  /***********************************************************
   *
   * read p mat
   *
   ***********************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

  /***********************************************************
   * allocate
   ***********************************************************/
  double _Complex *** vw_mat_p = init_3level_ztable ( T_global, evecs_num, evecs_num );
  if ( vw_mat_p == NULL ) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(11);
  }
  
  int source_momentum[3] = {0,0,0};
  int source_gamma_id = 4;

  /* AFF read */
  uitems = ( uint32_t )T_global * evecs_num * evecs_num;

  sprintf ( key, "/vdag-gp-w/C%d/N%d/PX%d_PY%d_PZ%d/G%d", Nconf, evecs_num,
      source_momentum[0], source_momentum[1], source_momentum[2], source_gamma_id);
  if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_exdefl_analyse] reading key %s %s %d\n", key, __FILE__, __LINE__ );

  affdir = aff_reader_chpath ( affr, affrn, key );
  if ( affdir == NULL ) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
    EXIT(15);
  }

  exitstatus = aff_node_get_complex ( affr, affdir, vw_mat_p[0][0], uitems );
  if(exitstatus != 0) {
    fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(16);
  }

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "p2gg_exdefl_analyse", "aff-read-mat-p", g_cart_id == 0 );

  /***********************************************************
   *
   * read v mat
   *
   ***********************************************************/

  /***********************************************************
   * allocate
   ***********************************************************/
  double _Complex ***** vw_mat_v = NULL;
  if ( g_sink_momentum_number > 0 && g_sink_gamma_id_number > 0 ) {
    gettimeofday ( &ta, (struct timezone *)NULL );

    vw_mat_v = init_5level_ztable ( g_sink_momentum_number, g_sink_gamma_id_number, T_global, evecs_num, evecs_num );
    if ( vw_mat_v == NULL ) {
      fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(11);
    }
    
    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
      int sink_momentum[3] = {
        g_sink_momentum_list[imom][0],
        g_sink_momentum_list[imom][1],
        g_sink_momentum_list[imom][2] };
  
      for ( int igam = 0; igam < g_sink_gamma_id_number; igam++ ) {
        int sink_gamma_id = g_sink_gamma_id_list[igam];
  
        /* AFF read */
        uitems = ( uint32_t )T_global * evecs_num * evecs_num;
  
        sprintf ( key, "/vdag-gp-w/C%d/N%d/PX%d_PY%d_PZ%d/G%d", Nconf, evecs_num,
            sink_momentum[0], sink_momentum[1], sink_momentum[2], sink_gamma_id);
        if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_exdefl_analyse] reading key %s %s %d\n", key, __FILE__, __LINE__ );
  
        affdir = aff_reader_chpath ( affr, affrn, key );
        if ( affdir == NULL ) {
          fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
          EXIT(15);
        }
  
        exitstatus = aff_node_get_complex ( affr, affdir, vw_mat_v[imom][igam][0][0], uitems );
        if(exitstatus != 0) {
          fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(16);
        }
      }
    }
  
    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "p2gg_exdefl_analyse", "aff-read-mat-v", g_cart_id == 0 );

  }

  /***********************************************************
   *
   * read v mat pt
   *
   ***********************************************************/

  /***********************************************************
   * allocate
   ***********************************************************/
  double _Complex **** vw_mat_vpt = NULL;
  if ( g_source_location_number > 0 && g_sink_gamma_id_number > 0 ) {
    gettimeofday ( &ta, (struct timezone *)NULL );
    vw_mat_vpt = init_4level_ztable ( g_source_location_number, g_sink_gamma_id_number, evecs_num, evecs_num );
    if ( vw_mat_vpt == NULL ) {
      fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_4level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(11);
    }
  
    /***********************************************************
     * loop on source points
     ***********************************************************/
    for ( int ix = 0; ix < g_source_location_number; ix++ ) {
  
      for ( int igam = 0; igam < g_sink_gamma_id_number; igam++ ) {
        int sink_gamma_id = g_sink_gamma_id_list[igam];
  
        /* AFF read */
        uitems = ( uint32_t )evecs_num * evecs_num;
  
        sprintf ( key, "/vdag-gp-w/C%d/N%d/T%d_X%d_Y%d_Z%d/G%d", Nconf, evecs_num,
            g_source_coords_list[ix][0], g_source_coords_list[ix][1], g_source_coords_list[ix][2], g_source_coords_list[ix][3], sink_gamma_id);
  
        if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_exdefl_analyse] reading key %s %s %d\n", key, __FILE__, __LINE__ );
  
        affdir = aff_reader_chpath ( affr, affrn, key );
        if ( affdir == NULL ) {
          fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_reader_chpath %s %d\n", __FILE__, __LINE__);
          EXIT(15);
        }
  
        exitstatus = aff_node_get_complex ( affr, affdir, vw_mat_vpt[ix][igam][0], uitems );
        if(exitstatus != 0) {
          fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_node_get_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(16);
        }
      }
    }
    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "p2gg_exdefl_analyse", "aff-read-mat-vpt", g_cart_id == 0 );

  }

  /***********************************************************
   * set default values for step size and minimal number
   * of evecs to be used in partial trace
   ***********************************************************/
  if ( evecs_use_step == -1 ) evecs_use_step = 1;
  if ( evecs_use_min  == -1 ) evecs_use_min  = evecs_num;

  /***********************************************************
   * loop on upper limit of eigenvectors
   ***********************************************************/
  for ( int evecs_use = evecs_use_min; evecs_use <= evecs_num; evecs_use += evecs_use_step ) {

    double _Complex * loop_p = init_1level_ztable ( T_global );
    if ( loop_p == NULL ) {
      fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_1level_ztable %s %d\n", __FILE__, __LINE__);
      EXIT(15);
    }

    /***********************************************************
     * partial trace
     ***********************************************************/
    gettimeofday ( &ta, (struct timezone *)NULL );

#pragma omp parallel for
    for ( int x0 = 0; x0 < T_global; x0++ ) {
      for ( int iw = 0; iw < evecs_use; iw++ ) {
        loop_p[x0] += vw_mat_p[x0][iw][iw] * evecs_eval_inv[iw];
      }
    }

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "p2gg_exdefl_analyse", "loop-partial-trace", g_cart_id == 0 );

    /***********************************************************
     * write loop to file
     ***********************************************************/
#ifdef HAVE_LHPC_AFF
    sprintf ( key, "/loop/g%d/px%d_py%d_pz%d/nev%d", source_gamma_id, source_momentum[0], source_momentum[1], source_momentum[2], evecs_use ); 
 
    affdir = aff_writer_mkpath ( affw, affwn, key );
    if ( affdir == NULL ) {
      fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_writer_mkpath %s %d\n", __FILE__, __LINE__);
      EXIT(17);
    }

    exitstatus = aff_node_put_complex ( affw, affdir, loop_p, (uint32_t)T_global );
    if(exitstatus != 0) {
      fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_node_put_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(18);
    }
#else
    sprintf ( filename, "%s.loop.g%d.px%d_py%d_pz%d.nev%d.%.4d", g_outfile_prefix, source_gamma_id,
        source_momentum[0], source_momentum[1], source_momentum[2], evecs_use, Nconf ); 
    FILE * ofs = fopen ( filename, "w" );
    if ( ofs == NULL ) {
      fprintf ( stderr, "[p2gg_exdefl_analyse] Error from fopen %s %d\n", __FILE__, __LINE__);
      EXIT(21);
    }
    for ( int x0 = 0; x0 < T_global; x0++ ) {
      fprintf ( ofs, "%25.16e  %25.16e\n", creal(loop_p[x0]), cimag(loop_p[x0]) );
    }
    fclose ( ofs );
#endif

#ifdef _USE_ATA_TENSOR
    if ( vw_mat_v != NULL ) {
      /***********************************************************
       *
       * ATA JJ TENSOR
       *
       * construct all-to-all jj tensor and the 3-point
       * function
       *
       ***********************************************************/
  
      /***********************************************************
       * allocate corr_v = jj tensor
       ***********************************************************/
      double _Complex ***** corr_v = init_5level_ztable ( g_sink_momentum_number, g_sink_gamma_id_number, g_sink_gamma_id_number, T_global, T_global );
      if ( corr_v == NULL ) {
        fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(15);
      }
  
      /***********************************************************
       * allocate 3-point function
       ***********************************************************/
      double _Complex ** corr_3pt = init_2level_ztable ( g_sequential_source_timeslice_number, T_global );
      if ( corr_3pt == NULL ) {
        fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(15);
      }
  
      /***********************************************************
       * loop on sink momenta
       *
       *  ( source momentum is fixed )
       ***********************************************************/
      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
  
        /***********************************************************
         * id of momentum vector in g_sink_momentum_list, which
         * fulfills momentum conservation
         * psnk[imom] + source_momentum + psnk[kmom] = 0
         ***********************************************************/
        int kmom = get_momentum_id ( g_sink_momentum_list[imom], source_momentum, g_sink_momentum_number, g_sink_momentum_list );
  
        if ( kmom == -1 ) {
          fprintf ( stderr, "[p2gg_exdefl_analyse] Warning momentum not found %s %d\n", __FILE__, __LINE__ );
          continue;  /* no such momentum was found */
        }
  
        if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_exdefl_analyse] psrc = %3d %3d %3d   psnk = %3d %3d %3d   pcur = %3d %3d %3d\n",
            source_momentum[0], source_momentum[1], source_momentum[2],
            g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
            g_sink_momentum_list[kmom][0], g_sink_momentum_list[kmom][1], g_sink_momentum_list[kmom][2] );
  
        /***********************************************************
         * loop on tensor components at sink and current side
         ***********************************************************/
        for ( int ig1 = 0; ig1 < g_sink_gamma_id_number; ig1++ ) {
        for ( int ig2 = 0; ig2 < g_sink_gamma_id_number; ig2++ ) {
  
          gettimeofday ( &ta, (struct timezone *)NULL );
  
          /***********************************************************
           * loop on time slice combinations at sink and current side
           ***********************************************************/
#pragma omp parallel for
          for ( int x0 = 0; x0 < T_global; x0++ ) {
          for ( int y0 = 0; y0 < T_global; y0++ ) {
  
            /***********************************************************
             * loop on evec ids, partial trace
             ***********************************************************/
            for ( int iw = 0; iw < evecs_use; iw++ ) {
            for ( int iv = 0; iv < evecs_use; iv++ ) {
              corr_v[imom][ig1][ig2][x0][y0] += vw_mat_v[imom][ig1][x0][iw][iv] * vw_mat_v[kmom][ig2][y0][iv][iw] * evecs_eval_inv[iw] * evecs_eval_inv[iv];
            }}
  
          }}
  
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "p2gg_exdefl_analyse", "jj-tensor-TxT-partial-trace", g_cart_id == 0 );
  
          /***********************************************************
           * write tensor to file
           ***********************************************************/
          gettimeofday ( &ta, (struct timezone *)NULL );
#ifdef HAVE_LHPC_AFF
          sprintf ( key, "/jj/g%d_g%d/px%d_py%d_pz%d/qx%d_qy%d_qz%d/nev%d",
              g_sink_gamma_id_list[ig1], g_sink_gamma_id_list[ig2],
              g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
              g_sink_momentum_list[kmom][0], g_sink_momentum_list[kmom][1], g_sink_momentum_list[kmom][2],
              evecs_use );
  
          affdir = aff_writer_mkpath ( affw, affwn, key );
          if ( affdir == NULL ) {
            fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_writer_mkpath %s %d\n", __FILE__, __LINE__);
            EXIT(17);
          }
  
          exitstatus = aff_node_put_complex ( affw, affdir, corr_v[imom][ig1][ig2][0], (uint32_t)T_global*T_global );
          if(exitstatus != 0) { 
            fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_node_put_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(18);
          }
#else
          sprintf ( filename, "%s.jj.g%d_g%d.px%d_py%d_pz%d.qx%d_qy%d_qz%d.nev%d.%.4d", g_outfile_prefix,
              g_sink_gamma_id_list[ig1], g_sink_gamma_id_list[ig2],
              g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
              g_sink_momentum_list[kmom][0], g_sink_momentum_list[kmom][1], g_sink_momentum_list[kmom][2],
              evecs_use, Nconf );
  
          FILE * ofs = fopen ( filename, "w" );
          if ( ofs == NULL ) {
            fprintf ( stderr, "[p2gg_exdefl_analyse] Error from fopen %s %d\n", __FILE__, __LINE__);
            EXIT(21);
          }
  
          for ( int x0 = 0; x0 < T_global; x0++ ) {
          for ( int y0 = 0; y0 < T_global; y0++ ) {
            fprintf ( ofs, "%3d %3d %25.16e  %25.16e\n", x0, y0, creal( corr_v[imom][ig1][ig2][x0][y0] ), cimag( corr_v[imom][ig1][ig2][x0][y0] ) );
          }}
          fclose ( ofs );
#endif
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "p2gg_exdefl_analyse", "jj-tensor-a2a-write", g_cart_id == 0 );
        }}  /* end of loop on tensor components at current and sink */
  
      }  /* end of loop on sink momenta */
  

      /***********************************************************
       * (half of the) epsion tensor
       *   all even permutations
       ***********************************************************/
      int const epsilon_tensor[3][3] = { {0,1,2}, {1,2,0}, {2,0,1} };
  
      /***********************************************************
       * set norm
       * norm = 1 / pvec^2 / #{psnk}
       *
       * NOTE: g_sink_momentum_list should be one momentum orbit
       *       AT MOST
       *       as done here, this ONLY WORKS for total momentum
       *       zero
       *
       * THIS SHOULD MATCH the steps in
       * contract_cvc_tensor.cp/antisymmetric_orbit_average_spatial
       ***********************************************************/
      double pvec[3] = {
          2 * sin ( M_PI * g_sink_momentum_list[0][0] / (double)LX_global ),
          2 * sin ( M_PI * g_sink_momentum_list[0][1] / (double)LY_global ),
          2 * sin ( M_PI * g_sink_momentum_list[0][2] / (double)LZ_global ) };
  
      double const norm = 1. / ( pvec[0] * pvec[0] + pvec[1] * pvec[1] + pvec[2] * pvec[2] ) / (double)g_sink_momentum_number / 2.;
  
      gettimeofday ( &ta, (struct timezone *)NULL );
  
      /***********************************************************
       * loop over sink momenta = average over (sub-)orbit
       ***********************************************************/
      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
  
        pvec[0] = 2 * sin ( M_PI * g_sink_momentum_list[imom][0] / (double)LX_global );
        pvec[1] = 2 * sin ( M_PI * g_sink_momentum_list[imom][1] / (double)LY_global );
        pvec[2] = 2 * sin ( M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global );
  
        /***********************************************************
         * loop over permuations for 3-dim. epsilon tensor
         ***********************************************************/
        for ( int iperm = 0; iperm < 3; iperm++ ) {
          int const ia = epsilon_tensor[iperm][0];
          int const ib = epsilon_tensor[iperm][1];
          int const ic = epsilon_tensor[iperm][2];
  
          /***********************************************************
           * loop on source - sink time separations
           *   tsnk - tsrc = g_sequential_source_timeslice_list[idt]
           ***********************************************************/
          for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {
  
            /***********************************************************
             * loop on source timeslices
             *   all T_global lattice timeslices enter
             ***********************************************************/
            for ( int tsrc = 0; tsrc < T_global; tsrc++ ) {
  
              /***********************************************************
               * tsnk = tsrc + ( source - sink time sep. )
               ***********************************************************/
              int const tsnk = ( tsrc + g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
  
              /***********************************************************
               * loop on current time
               *   all T_global timeslices enter
               ***********************************************************/
#pragma omp parallel for
              for ( int itsc = 0; itsc < T_global; itsc++ ) {
  
                /***********************************************************
                 * time difference current - sink
                 ***********************************************************/
                int tcur = ( tsnk + itsc + T_global ) % T_global;
  
                /***********************************************************
                 * contribution to the 3-pt function
                 *
                 *   loop_p at source time x 2-pt jj tensor at current , sink time
                 *
                 *   real part = Re ( Loop ) x Re ( jj tensor ) 
                 *             = pion channel ( iso-triplet pseudoscalar )
                 *
                 *   imag part = Im ( Loop ) x Im ( jj tensor ) 
                 *             = eta  channel ( iso-singlet pseudoscalar )
                 ***********************************************************/
                corr_3pt[idt][itsc] += 
                    ( creal( loop_p[tsrc] ) * ( creal( corr_v[imom][ia][ib][tcur][tsnk] ) - creal( corr_v[imom][ib][ia][tcur][tsnk] ) ) * pvec[ic] * norm )
                  + ( cimag( loop_p[tsrc] ) * ( cimag( corr_v[imom][ia][ib][tcur][tsnk] ) - cimag( corr_v[imom][ib][ia][tcur][tsnk] ) ) * pvec[ic] * norm ) * I;
              }
            }
          }  /* end of loop on source - sink time separations */
        }  /* end of loop on permutations */
      }  /* end of loop on sink momenta */
  
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "p2gg_exdefl_analyse", "3pt-a2a-dt-T-orbit-average", g_cart_id == 0 );
  
      /***********************************************************
       * write 3-point to file
       ***********************************************************/
      gettimeofday ( &ta, (struct timezone *)NULL );
#ifdef HAVE_LHPC_AFF
      for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {
  
        sprintf ( key, "/pgg/disc/orbit/g%d/px%d_py%d_pz%d/qx%d_qy%d_qz%d/nev%d/dt%d", source_gamma_id,
            source_momentum[0], source_momentum[1], source_momentum[2], 
            g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2],
            evecs_use , g_sequential_source_timeslice_list[idt] ); 
  
        affdir = aff_writer_mkpath ( affw, affwn, key );
        if ( affdir == NULL ) {
          fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_writer_mkpath %s %d\n", __FILE__, __LINE__);
          EXIT(17);
        }
  
        exitstatus = aff_node_put_complex ( affw, affdir, corr_3pt[idt], (uint32_t)T_global );
        if(exitstatus != 0) {
          fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_node_put_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(18);
        }
      }
#else
      sprintf ( filename, "%s.3pt.disc.g%d.px%d_py%d_pz%d.qx%d_qy%d_qz%d.nev%d.%.4d", g_outfile_prefix, source_gamma_id,
          source_momentum[0], source_momentum[1], source_momentum[2], 
          g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2],
          evecs_use, Nconf ); 
      ofs = fopen ( filename, "w" );
      if ( ofs == NULL ) {
        fprintf ( stderr, "[p2gg_exdefl_analyse] Error from fopen %s %d\n", __FILE__, __LINE__);
        EXIT(21);
      }
  
      for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {
        for ( int itsc = 0; itsc < T_global; itsc++ ) {
          fprintf ( ofs, "%3d %3d %25.16e  %25.16e\n", 
              g_sequential_source_timeslice_list[idt], itsc, creal( corr_3pt[idt][itsc]), cimag( corr_3pt[idt][itsc]) );
        }
      }
      fclose ( ofs );
#endif
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "p2gg_exdefl_analyse", "3pt-ata-write", g_cart_id == 0 );
  
      fini_5level_ztable ( &corr_v );
      fini_2level_ztable ( &corr_3pt );

    }  /* end of if vw_mat_v != NULL */

#endif  /* of ifdef _USE_ATA_TENSOR */

    /***********************************************************/
    /***********************************************************/

#ifdef _USE_PTA_TENSOR

    if ( vw_mat_v != NULL && vw_mat_vpt != NULL ) {
      /***********************************************************
       *
       * PTA JJ TENSOR
       *
       * construct point-to-all jj tensor and the 3-point
       * function
       *
       ***********************************************************/
  
      /***********************************************************
       * allocate corr_v = jj tensor
       ***********************************************************/
      double _Complex ***** corr_v = init_5level_ztable ( g_sink_momentum_number, g_sink_gamma_id_number, g_sink_gamma_id_number, g_source_location_number, T_global);
      if ( corr_v == NULL ) {
        fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__);
        EXIT(15);
      }
  
      /***********************************************************
       * loop on sink momenta
       *
       *  ( source momentum is fixed )
       ***********************************************************/
      for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
  
        /***********************************************************
         * id of momentum vector in g_sink_momentum_list, which
         * fulfills momentum conservation
         * psnk[imom] + source_momentum + psnk[kmom] = 0
         ***********************************************************/
        int kmom = get_momentum_id ( g_sink_momentum_list[imom], source_momentum, g_sink_momentum_number, g_sink_momentum_list );
  
        if ( kmom == -1 ) continue;  /* no such momentum was found */

        if ( g_verbose > 2 ) fprintf ( stdout, "# [p2gg_exdefl_analyse] psrc = %3d %3d %3d   psnk = %3d %3d %3d   pcur = %3d %3d %3d\n",
            source_momentum[0], source_momentum[1], source_momentum[2],
            g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
            g_sink_momentum_list[kmom][0], g_sink_momentum_list[kmom][1], g_sink_momentum_list[kmom][2] );
  
        /***********************************************************
         * loop on source locations
         ***********************************************************/
        for ( int isx = 0; isx < g_source_location_number; isx++ ) {
          int const tsnk = g_source_coords_list[isx][0];
          int const xsnk[3] = {
              g_source_coords_list[isx][1],
              g_source_coords_list[isx][2],
              g_source_coords_list[isx][3] };

          double _Complex const ephase = cexp ( TWO_MPI * (
              ( g_sink_momentum_list[kmom][0] / (double)LX_global ) * xsnk[0] 
            + ( g_sink_momentum_list[kmom][1] / (double)LY_global ) * xsnk[1] 
            + ( g_sink_momentum_list[kmom][2] / (double)LZ_global ) * xsnk[2] ) * 1.i );

          /***********************************************************
           * loop on tensor components at sink and current side
           ***********************************************************/
          for ( int ig1 = 0; ig1 < g_sink_gamma_id_number; ig1++ ) {
          for ( int ig2 = 0; ig2 < g_sink_gamma_id_number; ig2++ ) {
  
            gettimeofday ( &ta, (struct timezone *)NULL );
  
            /***********************************************************
             * loop on time slice combinations at sink and current side
             ***********************************************************/
#pragma omp parallel for
            for ( int x0 = 0; x0 < T_global; x0++ ) {
  
              /***********************************************************
               * loop on evec ids, partial trace
               * add the Fourier phase from source location
               *
               * v_all [x0] x v_pt [y0]
               ***********************************************************/
              for ( int iw = 0; iw < evecs_use; iw++ ) {
              for ( int iv = 0; iv < evecs_use; iv++ ) {
                corr_v[imom][ig1][ig2][isx][x0] += vw_mat_v[imom][ig1][x0][iw][iv] * vw_mat_vpt[isx][ig2][iv][iw] * evecs_eval_inv[iw] * evecs_eval_inv[iv] * ephase;
              }}
  
            }
  
            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "p2gg_exdefl_analyse", "jj-tensor-pta-partial-trace", g_cart_id == 0 );
  
            /***********************************************************
             * write tensor to file
             ***********************************************************/
            gettimeofday ( &ta, (struct timezone *)NULL );
#ifdef HAVE_LHPC_AFF
            sprintf ( key, "/jj/g%d_g%d/px%d_py%d_pz%d/qx%d_qy%d_qz%d/nev%d/t%d_x%d_y%d_z%d",
                g_sink_gamma_id_list[ig1], g_sink_gamma_id_list[ig2],
                g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
                g_sink_momentum_list[kmom][0], g_sink_momentum_list[kmom][1], g_sink_momentum_list[kmom][2],
                evecs_use, tsnk, xsnk[0], xsnk[1], xsnk[2]);
  
            affdir = aff_writer_mkpath ( affw, affwn, key );
            if ( affdir == NULL ) {
              fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_writer_mkpath %s %d\n", __FILE__, __LINE__);
              EXIT(17);
            }
  
            exitstatus = aff_node_put_complex ( affw, affdir, corr_v[imom][ig1][ig2][isx], (uint32_t)T_global);
            if(exitstatus != 0) { 
              fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_node_put_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(18);
            }
#else
            sprintf ( filename, "%s.jj.g%d_g%d.px%d_py%d_pz%d.qx%d_qy%d_qz%d.nev%d.%.4d.t%d_x%d_y%d_z%d", g_outfile_prefix,
                g_sink_gamma_id_list[ig1], g_sink_gamma_id_list[ig2],
                g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
                g_sink_momentum_list[kmom][0], g_sink_momentum_list[kmom][1], g_sink_momentum_list[kmom][2],
                evecs_use, Nconf, tsnk, xnk[0], xsnk[1], xsnk[2] );
  
            FILE * ofs = fopen ( filename, "w" );
            if ( ofs == NULL ) {
              fprintf ( stderr, "[p2gg_exdefl_analyse] Error from fopen %s %d\n", __FILE__, __LINE__);
              EXIT(21);
            }
  
            for ( int x0 = 0; x0 < T_global; x0++ ) {
              fprintf ( ofs, "%3d %3d %25.16e  %25.16e\n", x0, y0, creal( corr_v[imom][ig1][ig2][isx][x0] ), cimag( corr_v[imom][ig1][ig2][isx][x0] ) );
            }
            fclose ( ofs );
#endif
            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "p2gg_exdefl_analyse", "jj-tensor-pta-write", g_cart_id == 0 );

          }}  /* end of loop on tensor components at current and sink */
        }  /* end of loop on source locations */
      }  /* end of loop on sink momenta */
  
      /***********************************************************
       * (half of the) epsion tensor
       *   all even permutations
       ***********************************************************/
      int const epsilon_tensor[3][3] = { {0,1,2}, {1,2,0}, {2,0,1} };
  
      /***********************************************************
       * set norm
       * norm = 1 / pvec^2 / #{psnk}
       *
       * NOTE: g_sink_momentum_list should be one momentum orbit
       *       AT MOST
       *       as done here, this ONLY WORKS for total momentum
       *       zero
       ***********************************************************/
      double pvec[3] = {
          2 * sin ( M_PI * g_sink_momentum_list[0][0] / (double)LX_global ),
          2 * sin ( M_PI * g_sink_momentum_list[0][1] / (double)LY_global ),
          2 * sin ( M_PI * g_sink_momentum_list[0][2] / (double)LZ_global ) };
  
      double const norm = 1. / ( pvec[0] * pvec[0] + pvec[1] * pvec[1] + pvec[2] * pvec[2] ) / (double)g_sink_momentum_number;
  
      gettimeofday ( &ta, (struct timezone *)NULL );
  
      for ( int isx = 0; isx < g_source_location_number; isx++ ) {

        int const tsnk = g_source_coords_list[isx][0];
        int const xsnk[3] = {
            g_source_coords_list[isx][1],
            g_source_coords_list[isx][2],
            g_source_coords_list[isx][3] };

        /***********************************************************
         * allocate 3-point function
         ***********************************************************/
        double _Complex ** corr_3pt = init_2level_ztable ( g_sequential_source_timeslice_number, T_global );
        if ( corr_3pt == NULL ) {
          fprintf ( stderr, "[p2gg_exdefl_analyse] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__);
          EXIT(15);
        }
    
        /***********************************************************
         * loop over sink momenta = average over (sub-)orbit
         ***********************************************************/
        for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
    
          pvec[0] = 2 * sin ( M_PI * g_sink_momentum_list[imom][0] / (double)LX_global );
          pvec[1] = 2 * sin ( M_PI * g_sink_momentum_list[imom][1] / (double)LY_global );
          pvec[2] = 2 * sin ( M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global );
    
          /***********************************************************
           * loop over permuations for 3-dim. epsilon tensor
           ***********************************************************/
          for ( int iperm = 0; iperm < 3; iperm++ ) {
            int const ia = epsilon_tensor[iperm][0];
            int const ib = epsilon_tensor[iperm][1];
            int const ic = epsilon_tensor[iperm][2];
    
            /***********************************************************
             * loop on source - sink time separations
             *   tsnk - tsrc = g_sequential_source_timeslice_list[idt]
             ***********************************************************/
            for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {

              /***********************************************************
               * tsrc = tsnk - ( source - sink time sep. )
               ***********************************************************/
              int const tsrc = ( tsnk - g_sequential_source_timeslice_list[idt] + T_global ) % T_global;
    
              /***********************************************************
               * loop on current time
               *   all T_global timeslices enter
               ***********************************************************/
#pragma omp parallel for
              for ( int itsc = 0; itsc < T_global; itsc++ ) {
    
                /***********************************************************
                 * time difference current - sink
                 ***********************************************************/
                int tcur = ( tsnk + itsc + T_global ) % T_global;
    
                /***********************************************************
                 * contribution to the 3-pt function
                 *
                 *   loop_p at source time x 2-pt jj tensor at current , sink time
                 *
                 *   real part = Re ( Loop ) x Re ( jj tensor ) 
                 *             = pion channel ( iso-triplet pseudoscalar )
                 *
                 *   imag part = Im ( Loop ) x Im ( jj tensor ) 
                 *             = eta  channel ( iso-singlet pseudoscalar )
                 ***********************************************************/
                corr_3pt[idt][itsc] += 
                    ( creal( loop_p[tsrc] ) * ( creal( corr_v[imom][ia][ib][isx][tcur] ) - creal( corr_v[imom][ib][ia][isx][tcur] ) ) * pvec[ic] * norm )
                  + ( cimag( loop_p[tsrc] ) * ( cimag( corr_v[imom][ia][ib][isx][tcur] ) - cimag( corr_v[imom][ib][ia][isx][tcur] ) ) * pvec[ic] * norm ) * I;

              }  /* end of loop on current-sink time separations */
            }  /* end of loop on source - sink time separations */
          }  /* end of loop on permutations */
        }  /* end of loop on sink momenta */
    
        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "p2gg_exdefl_analyse", "3pt-p2a-dt-T-orbit-average", g_cart_id == 0 );
    
        /***********************************************************
         * write 3-point to file
         ***********************************************************/
        gettimeofday ( &ta, (struct timezone *)NULL );
#ifdef HAVE_LHPC_AFF
        for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {
    
          sprintf ( key, "/pgg/disc/orbit/g%d/px%d_py%d_pz%d/qx%d_qy%d_qz%d/nev%d/dt%d/t%d_x%d_y%d_z%d", source_gamma_id,
              source_momentum[0], source_momentum[1], source_momentum[2], 
              g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2],
              evecs_use , g_sequential_source_timeslice_list[idt],
              tsnk, xsnk[0], xsnk[1], xsnk[2] ); 
    
          affdir = aff_writer_mkpath ( affw, affwn, key );
          if ( affdir == NULL ) {
            fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_writer_mkpath %s %d\n", __FILE__, __LINE__);
            EXIT(17);
          }
    
          exitstatus = aff_node_put_complex ( affw, affdir, corr_3pt[idt], (uint32_t)T_global );
          if(exitstatus != 0) {
            fprintf ( stderr, "[p2gg_exdefl_analyse] Error from aff_node_put_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(18);
          }
        }  /* end of loop on source-sink time separations */
#else
        sprintf ( filename, "%s.3pt.disc.g%d.px%d_py%d_pz%d.qx%d_qy%d_qz%d.nev%d.%.4d.t%d_x%d_y%d_z%d", g_outfile_prefix, source_gamma_id,
            source_momentum[0], source_momentum[1], source_momentum[2], 
            g_sink_momentum_list[0][0], g_sink_momentum_list[0][1], g_sink_momentum_list[0][2],
            evecs_use, Nconf,
            tsnk, xsnk[0], xsnk[1], xsnk[2] ); 
        ofs = fopen ( filename, "w" );
        if ( ofs == NULL ) {
          fprintf ( stderr, "[p2gg_exdefl_analyse] Error from fopen %s %d\n", __FILE__, __LINE__);
          EXIT(21);
        }
    
        for ( int idt = 0; idt < g_sequential_source_timeslice_number; idt++ ) {
          for ( int itsc = 0; itsc < T_global; itsc++ ) {
            fprintf ( ofs, "%3d %3d %25.16e  %25.16e\n", 
                g_sequential_source_timeslice_list[idt], itsc, creal( corr_3pt[idt][itsc]), cimag( corr_3pt[idt][itsc]) );
          }
        }
        fclose ( ofs );
#endif
        fini_2level_ztable ( &corr_3pt );

        gettimeofday ( &tb, (struct timezone *)NULL );
        show_time ( &ta, &tb, "p2gg_exdefl_analyse", "3pt-p2a-writ3e", g_cart_id == 0 );

      }  /* end of loop on source locations */
  
      fini_5level_ztable ( &corr_v );

    }  /* end of if vw_mat_p != NULL */
#endif  /* of _USE_PTA_TENSOR */

    /***********************************************************/
    /***********************************************************/

    fini_1level_ztable ( &loop_p );

  }  /* end of loop on upper limits */

  fini_4level_ztable ( &vw_mat_vpt );
  fini_5level_ztable ( &vw_mat_v   );
  fini_3level_ztable ( &vw_mat_p   );

  /***********************************************************
   * close writer
   ***********************************************************/
#ifdef HAVE_LHPC_AFF
  aff_reader_close ( affr );

  aff_status_str = (char*)aff_writer_close ( affw );
  if( aff_status_str != NULL ) {
    fprintf(stderr, "[p2gg_exdefl_analyse] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
    EXIT(32);
  }
#endif
  
  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/
  
  fini_1level_dtable ( &evecs_eval );
  fini_1level_dtable ( &evecs_eval_inv );

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "p2gg_exdefl_analyse", "total time", g_cart_id == 0 );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_exdefl_analyse] end of run\n");
    fprintf(stderr, "# [p2gg_exdefl_analyse] end of run\n");
  }

  return(0);
}
