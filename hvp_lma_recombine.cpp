/****************************************************
 * hvp_lma_recombine.c
 *
 * Do 29. Mär 16:01:03 CEST 2018
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
#include "table_init_d.h"
#include "table_init_z.h"
#include "clover.h"
#include "gsp_utils.h"
#include "gsp_recombine.h"


using namespace cvc;

void usage() {
  fprintf(stdout, "Code to perform cvc correlator conn. contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  fprintf(stdout, "          -w                  : check position space WI [default false]\n");
  EXIT(0);
}

/****************************************************/
/****************************************************/

/****************************************************
 ****************************************************/
int get_momentum_id ( int const p[3], int (* const p_list)[3], int const n ) {

  for ( int i = 0; i < n; i++ ) {
    if ( ( p[0] == p_list[i][0] ) &&
         ( p[1] == p_list[i][1] ) &&
         ( p[2] == p_list[i][2] ) ) {
      return(i);
    }
  }

  return(-1);
}  // end of get_momentum_id

/****************************************************/
/****************************************************/

int main(int argc, char **argv) {
 

  double const epsilon = 6.e-8;

  int c;
  int filename_set = 0;
  int exitstatus;
  char filename[100];
  // double ratime, retime;
  unsigned int evecs_block_length = 0;
  unsigned int evecs_num = 0;
  unsigned int nproc_t = 1;


#ifdef HAVE_LHPC_AFF
  char tag[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "sh?f:b:n:t:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'b':
      evecs_block_length = atoi ( optarg );
      break;
    case 'n':
      evecs_num = atoi( optarg );
      break;
    case 't':
      nproc_t = atoi( optarg );
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  g_the_time = time(NULL);

  /***********************************************************
   * set the default values
   ***********************************************************/
  if(filename_set==0) strcpy(filename, "p2gg.input");
  // fprintf(stdout, "# [hvp_lma_recombine] Reading input from file %s\n", filename);
  read_input_parser(filename);


  /***********************************************************
   * initialize MPI parameters for cvc
   ***********************************************************/
  mpi_init(argc, argv);

  /***********************************************************
   * set number of openmp threads
   ***********************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [hvp_lma_recombine] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [hvp_lma_recombine] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[hvp_lma_recombine] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * initialize geometry fields
   ***********************************************************/
  exitstatus = init_geometry();
  if( exitstatus != 0) {
    fprintf(stderr, "[hvp_lma_recombine] Error from init_geometry, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_contraction(2);
  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * read eigenvalues
   ***********************************************************/
  double * evecs_eval = NULL;
  sprintf( tag, "/hvp/eval/N%d", evecs_num );
  sprintf( filename, "%s.%.4d.eval", filename_prefix, Nconf );
  exitstatus = gsp_read_eval( &evecs_eval, evecs_num, filename, tag);
  if( exitstatus != 0 ) {
    fprintf(stderr, "[hvp_lma_recombine] Error from gsp_read_eval, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(39);
  }


  /***********************************************************
   * set auxilliary eigenvalue fields
   ***********************************************************/
  double * const evecs_lambdainv           = (double*)malloc(evecs_num*sizeof(double));
  double * const evecs_4kappasqr_lambdainv = (double*)malloc(evecs_num*sizeof(double));
  if( evecs_lambdainv == NULL || evecs_4kappasqr_lambdainv == NULL ) {
    fprintf(stderr, "[hvp_lma_recombine] Error from malloc %s %d\n", __FILE__, __LINE__);
    EXIT(39);
  }
  for( unsigned int i = 0; i < evecs_num; i++) {
    evecs_lambdainv[i]           = 2.* g_kappa / evecs_eval[i];
    evecs_4kappasqr_lambdainv[i] = 4.* g_kappa * g_kappa / evecs_eval[i];
  }

  /***********************************************************
   * check evecs_block_length
   ***********************************************************/
  if ( evecs_block_length == 0 ) {
    evecs_block_length = evecs_num;
    if ( g_cart_id == 0 ) fprintf ( stdout, "# [hvp_lma_recombine] WARNING, reset evecs_block_length to %u\n", evecs_num );
  }

  /***********************************************************
   * set io process
   ***********************************************************/
  int const io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[hvp_lma_recombine] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [hvp_lma_recombine] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***********************************************************/
  /***********************************************************/

#if 0
  /***********************************************************
   * read Phi
   ***********************************************************/
  sprintf( tag, "/hvp/lma/N%d/B%d", evecs_num, evecs_block_length );
  sprintf( filename, "%s.%.4d", filename_prefix, Nconf );

  double _Complex ***** phi = init_5level_ztable ( 4, g_sink_momentum_number, T, evecs_num, evecs_num );
  if ( phi == NULL ) {
    fprintf (stderr, "[hvp_lma_recombine] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(3);
  }

  for ( int imu = 0; imu < 4; imu++ ) {

    for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

      // loop on timeslices
      for ( int it = 0; it < T; it++ ) {
      
        exitstatus = gsp_read_cvc_node ( phi[imu][imom][it], evecs_num, evecs_block_length, g_sink_momentum_list[imom], "mu", imu, filename, tag, it, nproc_t );

        if( exitstatus != 0 ) {
          fprintf(stderr, "[hvp_lma_recombine] Error from gsp_read_cvc_node, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(14);
        }

        // TEST
        // write the complete Nev x Nev complex field to stdout
        if ( g_verbose > 4 )  {
          // show the data read by gsp_read_cvc_node
          fprintf ( stdout, "# [hvp_lma_recombine] /hvp/lma/N%d/t%.2d/mu%d/px%.2dpy%.2dpz%.2d\n", evecs_num, it, imu,
              g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] );
          for ( unsigned int i1 = 0; i1 < evecs_num; i1++ ) {
          for ( unsigned int i2 = 0; i2 < evecs_num; i2++ ) {
            fprintf ( stdout, "  %4d  %4d    %25.16e   %25.16e\n", i1, i2, 
                creal( phi[imu][imom][it][i1][i2] ), cimag( phi[imu][imom][it][i1][i2] ) );
          }}
        }  // end of if verbose > 4

      }  // end of loop on timeslices

      /***********************************************************/
      /***********************************************************/

#if 0
    /***********************************************************
     * test loops
     ***********************************************************/
      double _Complex *phi_tr = init_1level_ztable ( T );

      gsp_tr_mat_weight ( phi_tr , phi[imu][imom] , evecs_4kappasqr_lambdainv , evecs_num, T );

      if ( g_verbose > 4 )  {
        // show the trace
        fprintf ( stdout, "# [hvp_lma_recombine] /loop/cvc/nev%.4d/px%.2dpy%.2dpz%.2d/mu%d\n", evecs_num, g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], imu );
        for ( int it = 0; it < T; it++ ) {
          fprintf ( stdout, "%26.16e  %25.16e\n", creal( phi_tr[it] ), cimag( phi_tr[it] ) );
        }
      }  // end of if verbose > 4

      fini_1level_ztable ( &phi_tr );
#endif  // of if 0

      /***********************************************************/
      /***********************************************************/

    }  // end of loop on momenta


  }  // end of loop on mu
#endif
  /***********************************************************/
  /***********************************************************/

#if 0
  /***********************************************************
   * read vv and ww, which are needed for the Ward identity
   ***********************************************************/

  double _Complex **** ww = init_4level_ztable ( g_sink_momentum_number, T, evecs_num, evecs_num );
  double _Complex **** vv = init_4level_ztable ( g_sink_momentum_number, T, evecs_num, evecs_num );
  if ( ww == NULL || vv == NULL ) {
    fprintf (stderr, "[hvp_lma_recombine] Error from init_4level_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(3);
  }

  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

    // loop on timeslices
    for ( int it = 0; it < T; it++ ) {
      
      exitstatus = gsp_read_cvc_node ( ww[imom][it], evecs_num, evecs_block_length, g_sink_momentum_list[imom], "ww", -1, filename, tag, it, nproc_t );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[hvp_lma_recombine] Error from gsp_read_cvc_node, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(14);
      }

      exitstatus = gsp_read_cvc_node ( vv[imom][it], evecs_num, evecs_block_length, g_sink_momentum_list[imom], "vv", -1, filename, tag, it, nproc_t );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[hvp_lma_recombine] Error from gsp_read_cvc_node, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(14);
      }

      // TEST
      // write the complete Nev x Nev complex field to stdout
      if ( g_verbose > 4 )  {
        // show the ww data read by gsp_read_cvc_node
        fprintf ( stdout, "# [hvp_lma_recombine] /hvp/lma/N%d/ww/t%.2d/px%.2dpy%.2dpz%.2d\n", evecs_num, it,
            g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] );
        for ( unsigned int i1 = 0; i1 < evecs_num; i1++ ) {
        for ( unsigned int i2 = 0; i2 < evecs_num; i2++ ) {
          fprintf ( stdout, "  %4d  %4d    %25.16e   %25.16e\n", i1, i2, creal( ww[imom][it][i1][i2] ), cimag( ww[imom][it][i1][i2] ) );
        }}

        // show the vv data read by gsp_read_cvc_node
        fprintf ( stdout, "# [hvp_lma_recombine] /hvp/lma/N%d/vv/t%.2d/px%.2dpy%.2dpz%.2d\n", evecs_num, it,
            g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] );
        for ( unsigned int i1 = 0; i1 < evecs_num; i1++ ) {
        for ( unsigned int i2 = 0; i2 < evecs_num; i2++ ) {
          fprintf ( stdout, "  %4d  %4d    %25.16e   %25.16e\n", i1, i2, creal( vv[imom][it][i1][i2] ), cimag( vv[imom][it][i1][i2] ) );
        }}

      }  // end of if verbose > 4

    }  // end of loop on timeslices

  }  // end of loop on momenta
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * read mee part of hvp
   ***********************************************************/
  sprintf( tag, "/hvp/lma/N%d/B%d/mee", evecs_num, evecs_block_length );
  sprintf( filename, "%s.%.4d", filename_prefix, Nconf );

  double _Complex ****** hvp_mee_ts = init_6level_ztable ( g_sink_momentum_number, T, 4, 4, 3, evecs_num );
  if ( hvp_mee_ts == NULL ) {
    fprintf (stderr, "[hvp_lma_recombine] Error from init_6level_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(3);
  }

  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

    for ( int it = 0; it < T; it++ ) {

      exitstatus = gsp_read_cvc_mee_node ( hvp_mee_ts[imom][it], evecs_num, g_sink_momentum_list[imom], filename, tag, it);
      if ( exitstatus != 0 ) {
        fprintf(stderr, "[hvp_lma_recombine] Error from gsp_read_cvc_mee_node, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(14);
      }

      for ( int imu = 0; imu < 4; imu++ ) {
      for ( int inu = 0; inu < 4; inu++ ) {
        for ( int idt = 0; idt < 3; idt++ ) {
          fprintf ( stdout, "# [hvp_lma_recombine] /hvp/lma/N%d/mee/mu%d/nu%d/px%.2dpy%.2dpz%.2d/t%.2d/dt%d\n", evecs_num, imu, inu,
              g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2], it, idt-1 );
          for ( int i = 0; i < evecs_num; i++ ) {
            fprintf ( stdout, "  %25.16e  %25.16e\n", creal( hvp_mee_ts[imom][it][imu][inu][idt][i] ), cimag( hvp_mee_ts[imom][it][imu][inu][idt][i] ) );
          }
        }
      }}
    }  // end of loop on timeslices
  }  // end of loop on momenta

  /***********************************************************/
  /***********************************************************/

#if 0
  /***********************************************************
   * check WI for mee part
   ***********************************************************/
  double _Complex **** hvp_mee = init_4level_ztable ( g_sink_momentum_number, 4, 4, T );
  if ( hvp_mee == NULL ) {
    fprintf (stderr, "[hvp_lma_recombine] Error from init_4level_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(3);
  }

  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
    for ( int imu = 0; imu < 4; imu++ ) {
    for ( int inu = 0; inu < 4; inu++ ) {

      for ( int ts = 0; ts < T; ts++ ) {

        for ( int idt = 0; idt < 3; idt++ ) {
          int const it = ( idt - 1 + T ) % T;

          double _Complex ztmp = 0; 
          for ( int i = 0; i < evecs_num; i++ ) {
            ztmp += hvp_mee_ts[imom][ts][imu][inu][idt][i]  * evecs_4kappasqr_lambdainv[i];
          }
        

          hvp_mee[imom][imu][inu][it] += ztmp;

        }  // end of loop on dt

      }  // end of loop on timeslices

      // FT
      gsp_ft_p0_shift ( hvp_mee[imom][imu][inu], hvp_mee[imom][imu][inu], g_sink_momentum_list[imom], imu, inu, +1 );

    }}  // end of loop on nu, mu



    for ( int ip0 = 0; ip0 < T; ip0 ++ ) {

      double const sinp[4] = {
        2. * sin( M_PI * ip0 / (double)T_global ),
        2. * sin( M_PI * g_sink_momentum_list[imom][0] / (double)LX_global ),
        2. * sin( M_PI * g_sink_momentum_list[imom][1] / (double)LY_global ),
        2. * sin( M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global ) 
      };

      for ( int inu = 0; inu < 4; inu++ ) {

        double _Complex const dhvp_mee = 
          sinp[0] * hvp_mee[imom][0][inu][ip0] +
          sinp[1] * hvp_mee[imom][1][inu][ip0] +
          sinp[2] * hvp_mee[imom][2][inu][ip0] +
          sinp[3] * hvp_mee[imom][3][inu][ip0];

        double _Complex const hvp_meed = 
          sinp[0] * hvp_mee[imom][inu][0][ip0] +
          sinp[1] * hvp_mee[imom][inu][1][ip0] +
          sinp[2] * hvp_mee[imom][inu][2][ip0] +
          sinp[3] * hvp_mee[imom][inu][3][ip0];

        fprintf ( stdout, " WI p %3d %3d %3d %3d nu %d dhvp_mee %25.16e %25.16e hvp_meed %25.16e %25.16e\n", 
            ip0, g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
            inu, creal ( dhvp_mee ), cimag ( dhvp_mee ), creal ( hvp_meed ), cimag ( hvp_meed ) );
      }  // end of loop on nu

    }  // end of loop on p0



  }  // end of loop on momenta
  
  

  fini_4level_ztable ( &hvp_mee );

#endif

  /***********************************************************/
  /***********************************************************/
#if 0

  /***********************************************************
   * check the Ward identity for Phi per
   *   3-momentum vector
   *   eigenvector pair k1, k2 = 0, ..., Nev - 1
   *   p0
   *
   ***********************************************************/

  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {
    for ( int k1 = 0; k1 < evecs_num; k1++ ) {
    for ( int k2 = 0; k2 < evecs_num; k2++ ) {

      // t-dep fields
      double _Complex * vvt = init_1level_ztable ( T );
      double _Complex * wwt = init_1level_ztable ( T );
      double _Complex ** phit = init_2level_ztable ( 4, T );

      // p0-dep fields
      double _Complex * vvp = init_1level_ztable ( T );
      double _Complex * wwp = init_1level_ztable ( T );
      double _Complex ** phip = init_2level_ztable ( 4, T );

      // copy the timeslice
      for ( int it = 0; it < T; it++ ) {
        vvt[it] = vv[imom][it][k1][k2];
        wwt[it] = ww[imom][it][k1][k2];
      }
      for ( int imu = 0; imu < 4; imu++ ) {
        for ( int it = 0; it < T; it++ ) {
          phit[imu][it] = phi[imu][imom][it][k1][k2];
        }
      }

      // FT in t
      for ( int ip0 = 0; ip0 < T; ip0 ++ ) {

        double const p[4] = {
          M_PI * ip0 / (double)T_global,
          M_PI * g_sink_momentum_list[imom][0] / (double)LX_global,
          M_PI * g_sink_momentum_list[imom][1] / (double)LY_global,
          M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global };

        double const sinp[4] = { 2*sin( p[0] ), 2*sin( p[1] ), 2*sin( p[2] ), 2*sin( p[3] ) };


        for ( int it = 0; it < T; it++ ) {
          double const phase = p[0] * 2 * ( it + g_proc_coords[0] * T);

          double _Complex const ephase = cexp ( phase * I );

          wwp[ip0] += wwt[it] * ephase;
          vvp[ip0] += vvt[it] * ephase;

          for ( int imu = 0; imu < 4; imu++ ) {
            phip[imu][ip0] += phit[imu][it] * ephase;
          }

        }

        for ( int imu = 0; imu < 4; imu++ ) {
          phip[imu][ip0] *= cexp( p[imu] * I);
        }
 
        // check WI
        double _Complex const dphip = I * (
          sinp[0] * phip[0][ip0] + 
          sinp[1] * phip[1][ip0] + 
          sinp[2] * phip[2][ip0] + 
          sinp[3] * phip[3][ip0] );

        double _Complex const ct = wwp[ip0] - vvp[ip0] * evecs_eval[k2]/(4.*g_kappa*g_kappa);

        double const adiff = cabs ( ct - dphip );

        double rdiff = 0.;

        if ( cabs( ct ) != 0. && cabs ( dphip ) != 0. ) {
          rdiff = adiff * 2. / cabs ( ct + dphip );
        }


        fprintf ( stdout, "WI p %3d %3d %3d   k %2d %2d  p0 %3d dphip %25.16e %25.16e ct %25.16e %25.16e  diff %25.16e %25.16e\n",
            g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
            k1, k2, ip0, creal( dphip ), cimag( dphip ), creal( ct ), cimag( ct ) , adiff  , rdiff );

        if ( rdiff >= epsilon ) {
          fprintf ( stderr, "WI epsilon too large for p %3d %3d %3d   k %2d %2d  p0 %3d dphip %25.16e %25.16e ct %25.16e %25.16e  diff %25.16e %25.16e\n",
              g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
              k1, k2, ip0, creal( dphip ), cimag( dphip ), creal( ct ), cimag( ct ) , adiff  , rdiff );

          EXIT(128);
        }

      }  // end of loop on ip0

      // free the timeslice fields
      fini_1level_ztable ( &vvt );
      fini_1level_ztable ( &wwt );
      fini_2level_ztable ( &phit );
      fini_1level_ztable ( &vvp );
      fini_1level_ztable ( &wwp );
      fini_2level_ztable ( &phip );

    }}  // end of loops on k2, k1

  }  // end of loop on 3-momenta
#endif

  /***********************************************************/
  /***********************************************************/

#if 0
  /***********************************************************
   * recombine to hvp tensor
   ***********************************************************/

  for ( int imom = 0; imom < g_sink_momentum_number; imom++ ) {

      int psource[3] = {
        -g_sink_momentum_list[imom][0],
        -g_sink_momentum_list[imom][1],
        -g_sink_momentum_list[imom][2] };
      int imom2 = get_momentum_id ( psource, g_sink_momentum_list, g_sink_momentum_number );
      if ( imom2 == -1 ) {
        fprintf( stderr, "[hvp_lma_recombine] Error, could not find matching id for psource = %3d %3d %3d\n", 
            psource[0], psource[1], psource[2] );
        
        EXIT(4);
      }
      if ( g_verbose > 4 ) {
        fprintf ( stdout, "# [hvp_lma_recombine] sink momentum %3d %3d %3d  source momentum %3d %3d %3d\n",
            g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2] ,
            g_sink_momentum_list[imom2][0], g_sink_momentum_list[imom2][1], g_sink_momentum_list[imom2][2] );
      }

      double _Complex *** hvp = init_3level_ztable ( 4, 4, T );
      double _Complex ** vv_phi = init_2level_ztable ( 4, T );
      double _Complex ** ww_phi = init_2level_ztable ( 4, T );
      double _Complex ** phi_vv = init_2level_ztable ( 4, T );
      double _Complex ** phi_ww = init_2level_ztable ( 4, T );

      double * unit_weight = init_1level_dtable ( evecs_num );
      for ( int i = 0; i < evecs_num; i++ ) unit_weight[i] = 1.;

      for ( int imu = 0; imu < 4; imu++ ) {

        gsp_tr_mat_weight_mat_weight ( vv_phi[imu], vv[imom],       unit_weight,               phi[imu][imom2], evecs_4kappasqr_lambdainv, evecs_num, T );
        gsp_tr_mat_weight_mat_weight ( phi_vv[imu], phi[imu][imom], evecs_4kappasqr_lambdainv, vv[imom2],       unit_weight,               evecs_num, T );

        gsp_tr_mat_weight_mat_weight ( ww_phi[imu], ww[imom],       evecs_4kappasqr_lambdainv, phi[imu][imom2], evecs_4kappasqr_lambdainv, evecs_num, T );
        gsp_tr_mat_weight_mat_weight ( phi_ww[imu], phi[imu][imom], evecs_4kappasqr_lambdainv, ww[imom2],       evecs_4kappasqr_lambdainv, evecs_num, T );

      for ( int inu = 0; inu < 4; inu++ ) {

        gsp_tr_mat_weight_mat_weight ( hvp[imu][inu], phi[imu][imom], evecs_4kappasqr_lambdainv, phi[inu][imom2], evecs_4kappasqr_lambdainv, evecs_num, T );

      }}  // end of loop on nu, mu

      /***********************************************************/
      /***********************************************************/

      // FT

      for ( int imu = 0; imu < 4; imu++ ) {
        gsp_ft_p0_shift ( phi_vv[imu], phi_vv[imu], g_sink_momentum_list[imom], imu, -1, +1 );
        gsp_ft_p0_shift ( phi_ww[imu], phi_ww[imu], g_sink_momentum_list[imom], imu, -1, +1 );

        gsp_ft_p0_shift ( vv_phi[imu], vv_phi[imu], g_sink_momentum_list[imom], -1, imu, +1 );
        gsp_ft_p0_shift ( ww_phi[imu], ww_phi[imu], g_sink_momentum_list[imom], -1, imu, +1 );

      for ( int inu = 0; inu < 4; inu++ ) {
        gsp_ft_p0_shift ( hvp[imu][inu], hvp[imu][inu], g_sink_momentum_list[imom], imu, inu, +1 );
      }}

      /***********************************************************/
      /***********************************************************/

      /***********************************************************
       * check the WI
       ***********************************************************/
      for ( int ip0 = 0; ip0 < T; ip0 ++ ) {

        double const sinp[4] = {
          2. * sin( M_PI * ip0 / (double)T_global ),
          2. * sin( M_PI * g_sink_momentum_list[imom][0] / (double)LX_global ),
          2. * sin( M_PI * g_sink_momentum_list[imom][1] / (double)LY_global ),
          2. * sin( M_PI * g_sink_momentum_list[imom][2] / (double)LZ_global ) 
        };


        for ( int inu = 0; inu < 4; inu++ ) {

          double _Complex const dhvp = sinp[0] * hvp[0][inu][ip0]
                                     + sinp[1] * hvp[1][inu][ip0]
                                     + sinp[2] * hvp[2][inu][ip0]
                                     + sinp[3] * hvp[3][inu][ip0];

          double _Complex const dct = ww_phi[inu][ip0] - vv_phi[inu][ip0];

          double const ddiff = cabs ( I * dhvp - dct );

          double _Complex const hvpd = sinp[0] * hvp[inu][0][ip0]
                                     + sinp[1] * hvp[inu][1][ip0]
                                     + sinp[2] * hvp[inu][2][ip0]
                                     + sinp[3] * hvp[inu][3][ip0];

          double _Complex const ctd = phi_ww[inu][ip0] - phi_vv[inu][ip0];

          double const diffd = cabs( -I * hvpd - ctd );

          fprintf ( stdout, " WI p %3d %3d %3d %3d nu %d dhvp %25.16e %25.16e dct %25.16e %25.16e d %9.2e hvpd %25.16e %25.16e ctd %25.16e %25.16e d %9.2e\n", 
              ip0, g_sink_momentum_list[imom][0], g_sink_momentum_list[imom][1], g_sink_momentum_list[imom][2],
              inu, creal ( dhvp ), cimag ( dhvp ), creal ( dct ), cimag ( dct ), ddiff, creal ( hvpd ), cimag ( hvpd ), creal ( ctd ), cimag ( ctd ), diffd );
              

        }  // end of loop on nu
      }  // end of loop on p0

      /***********************************************************/
      /***********************************************************/

      fini_1level_dtable ( &unit_weight );
      fini_3level_ztable ( &hvp );
      fini_2level_ztable ( &vv_phi );
      fini_2level_ztable ( &ww_phi );
      fini_2level_ztable ( &phi_vv );
      fini_2level_ztable ( &phi_ww );

  }  // end of loop on momenta

#endif  // of if 0


  /***********************************************************/
  /***********************************************************/
#if 0
  fini_5level_ztable ( &phi );
  fini_4level_ztable ( &vv );
  fini_4level_ztable ( &ww );
#endif
  fini_6level_ztable ( &hvp_mee_ts );

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/

  if ( evecs_lambdainv           != NULL ) free ( evecs_lambdainv );
  if ( evecs_4kappasqr_lambdainv != NULL ) free ( evecs_4kappasqr_lambdainv ); 
  if ( evecs_eval                != NULL ) free ( evecs_eval );

  /***********************************************************
   * free clover matrix terms
   ***********************************************************/
  fini_clover ();

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_xchange_eo_propagator();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif


  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [hvp_lma_recombine] %s# [hvp_lma_recombine] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [hvp_lma_recombine] %s# [hvp_lma_recombine] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
