/****************************************************
 * hlbl_test_kqed
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
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

#ifdef __cplusplus
extern "C"
{
#endif

#  ifdef HAVE_TMLQCD_LIBWRAPPER
#    include "tmLQCD.h"
#  endif

#  ifdef HAVE_KQED
#    include "KQED.h"
#  endif

#ifdef __cplusplus
}
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
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "clover.h"
#include "scalar_products.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1

#define _WITH_TIMER 1


using namespace cvc;

/***********************************************************
 * KQED kernel function pointer
 ***********************************************************/
/* void QED_kernel_L0( const double xv[4] , const double yv[4] , const struct QED_kernel_temps t , double kerv[6][4][4][4] ) ; */
typedef void (*QED_kernel_LX_ptr)( const double xv[4], const double yv[4], const struct QED_kernel_temps t, double kerv[6][4][4][4] );

/***********************************************************
 * x must be in { 0, ..., L-1 }
 * mapping as in 2006.16224, eq. 8
 ***********************************************************/
inline void site_map (int xv[4], int const x[4] )
{
  xv[0] = ( x[0] >= T_global   / 2 ) ? (x[0] - T_global )  : x[0];
  xv[1] = ( x[1] >= LX_global  / 2 ) ? (x[1] - LX_global)  : x[1];
  xv[2] = ( x[2] >= LY_global  / 2 ) ? (x[2] - LY_global)  : x[2];
  xv[3] = ( x[3] >= LZ_global  / 2 ) ? (x[3] - LZ_global)  : x[3];

  return;
}

/***********************************************************
 * as above, but set L/2 to 0 and -L/2 to 0
 ***********************************************************/
inline void site_map_zerohalf (int xv[4], int const x[4] )
{
  xv[0] = ( x[0] > T_global   / 2 ) ? x[0] - T_global   : (  ( x[0] < T_global   / 2 ) ? x[0] : 0 );
  xv[1] = ( x[1] > LX_global  / 2 ) ? x[1] - LX_global  : (  ( x[1] < LX_global  / 2 ) ? x[1] : 0 );
  xv[2] = ( x[2] > LY_global  / 2 ) ? x[2] - LY_global  : (  ( x[2] < LY_global  / 2 ) ? x[2] : 0 );
  xv[3] = ( x[3] > LZ_global  / 2 ) ? x[3] - LZ_global  : (  ( x[3] < LZ_global  / 2 ) ? x[3] : 0 );

  return;
}

/***********************************************************/
/***********************************************************/

void usage() {
  fprintf(stdout, "Code to perform contractions for hlbl tensor\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {

  double const mmuon = 105.6583745 /* MeV */  / 197.3269804 /* MeV fm */;
  double const alat[2] = { 0.07957, 0.00013 };  /* fm */


  int const ysign_num = 4;
  int const ysign_comb[16][4] = {
    { 1, 1, 1, 1},
    { 1, 1,-1,-1},
    { 1,-1, 1,-1},
    { 1,-1,-1, 1},
    {-1, 1,-1, 1},
    {-1,-1, 1, 1},
    {-1, 1, 1,-1},
    { 1, 1, 1,-1},
    { 1, 1,-1, 1},
    { 1,-1, 1, 1},
    { 1,-1,-1,-1},
    {-1, 1, 1, 1},
    {-1, 1,-1,-1},
    {-1,-1, 1,-1},
    {-1,-1,-1, 1},
    {-1,-1,-1,-1}
  };

  int idx_comb[6][2] = {
        {0,1},
        {0,2},
        {0,3},
        {1,2},
        {1,3},
        {2,3} };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[400];
  struct timeval start_time, end_time;
  int ymax = 0;

  struct timeval ta, tb;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:y:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'y':
      ymax = atoi ( optarg );
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
  /* fprintf(stdout, "# [hlbl_test_kqed] Reading input from file %s\n", filename); */
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
    fprintf(stdout, "# [hlbl_test_kqed] git version = %s\n", g_gitversion);
  }


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [hlbl_test_kqed] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [hlbl_test_kqed] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[hlbl_test_kqed] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[hlbl_test_kqed] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  /***********************************************
   * set io process
   ***********************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[hlbl_test_kqed] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [hlbl_test_kqed] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  /***********************************************************
   * unit for x, y
   ***********************************************************/
  double const xunit[2] = { mmuon * alat[0], mmuon * alat[1] };

  /***********************************************************
   * output filename
   ***********************************************************/
  char output_filename[400];
  sprintf ( output_filename, "%s.%d.h5", g_outfile_prefix, Nconf );

  /***********************************************************
   * set up QED Kernel package
   ***********************************************************/
  struct QED_kernel_temps kqed_t ;

  if( initialise( &kqed_t ) )
  {
    fprintf(stderr, "[hlbl_test_kqed] Error from kqed initialise, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(19);
  }


            /***********************************************************/
            /***********************************************************/

            /***********************************************************
             * summation with QED kernel
             ***********************************************************/
            double const xm[4] = {
              xv[0] * xunit[0],
              xv[1] * xunit[0],
              xv[2] * xunit[0],
              xv[3] * xunit[0] };

            double const ym[4] = {
              yv[0] * xunit[0],
              yv[1] * xunit[0],
              yv[2] * xunit[0],
              yv[3] * xunit[0] };


            double * const _kerv1   = (double * const )kerv1;
            double * const _kerv2   = (double * const )kerv2;
            double * const _kerv3   = (double * const )kerv3;

            double * const _corr_I  = corr_I[0][0][0];
            double * const _corr_II = corr_II[0][0][0];

            double const xm_mi_ym[4] = {
              xm[0] - ym[0],
              xm[1] - ym[1],
              xm[2] - ym[2],
              xm[3] - ym[3] };

            /***********************************************************
             * loop on kernsl
             ***********************************************************/
            for ( int ikernel = 0; ikernel < 4; ikernel++ )
            {

              /* QED_kernel_L0( xm, ym, kqed_t, kerv );*/
              KQED_LX[ikernel]( xm, ym,       kqed_t, kerv1 );
              KQED_LX[ikernel]( ym, xm,       kqed_t, kerv2 );
              KQED_LX[ikernel]( xm, xm_mi_ym, kqed_t, kerv3 );
              double dtmp = 0.;
              int i = 0;
              for( int k = 0; k < 6; k++ )
              {
                for ( int mu = 0; mu < 4; mu++ )
                {
                  for ( int nu = 0; nu < 4; nu++ )
                  {
                    for ( int lambda = 0; lambda < 4; lambda++ )
                    {
                      dtmp += ( kerv1[k][mu][nu][lambda] + kerv2[k][nu][mu][lambda] - kerv3[k][lambda][nu][mu] ) * _corr_I[2*i] 
                              + kerv3[k][lambda][nu][mu] * _corr_II[2*i];

                      i++;
                    }
                  }
                }
              }
              kernel_sum_thread[ikernel] += dtmp;


            }  /* end of loop on kernels */


          }  /* end of loop on ix */

          /***********************************************************
           * summation with QED kernel
           ***********************************************************/

  }  /* end of loop on source locations */

  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/
          
  free_QED_temps( &kqed_t  );

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
  show_time ( &start_time, &end_time, "hlbl_test_kqed", "runtime", g_cart_id == 0 );

  return(0);
}
