/****************************************************
 * test-x-kernel
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <sys/time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif


#ifdef __cplusplus
}
#endif

#define MAIN_PROGRAM

#include "iblas.h"
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
#include "table_init_z.h"
#include "table_init_d.h"
#include "table_init_i.h"
#include "table_init_c.h"
#include "clover.h"
#include "prepare_source.h"

#ifdef HAVE_CUDA
#include "hlbl_lm_cuda.cuh"
#endif

#include "ranlxd.h"

#define _WITH_TIMER 1

#define _EVEC_TEST 0

using namespace cvc;

typedef struct {
  double re, im;
} cplx_t;


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

/***********************************************************
 * improve that as need be
 *
 * NOTE: the permutation of the mu, nu, lambda indices
 * mu runs fastest for summation in X
 ***********************************************************/
inline void set_kernel_pointx ( double * const kx, double kerv[6][4][4][4] )
{
  for ( int irhosigma = 0; irhosigma < 6; irhosigma++ )
  {
    for ( int imu = 0; imu < 4; imu++ )
    {
      for ( int inu = 0; inu < 4; inu++ )
      {
        for ( int ilambda = 0; ilambda < 4; ilambda++ )
        {
          int const idx = ( ( 4 * irhosigma + inu ) * 4 + ilambda ) * 4 + imu;
          kx[idx] = kerv[irhosigma][imu][inu][ilambda];
        }
      }
    }
  }
  return;
} 


/***********************************************************/
/***********************************************************/

/***********************************************************
 * compute p = V^H s
 *
 * V is nv x nx (C) = nx x nv (F)
 * s is ns x nx (C) = nx x ns (F)
 *
 * p is [nx x nv]^H x [nx x ns] = nv x ns (F) = ns x nv (C)
 ***********************************************************/
inline void project (double _Complex * const p, double _Complex * const V, double _Complex * const s, int const nv, int const ns, int const nx )
{
  double _Complex * BLAS_A = V;
  double _Complex * BLAS_B = s;
  double _Complex * BLAS_C = p;

  double _Complex BLAS_ALPHA  = 1.;
  double _Complex BLAS_BETA   = 0.;
  char BLAS_TRANSA = 'C';
  char BLAS_TRANSB = 'N';
  int BLAS_M      = nv;
  int BLAS_K      = nx;
  int BLAS_N      = ns;
  int BLAS_LDA    = BLAS_K;
  int BLAS_LDB    = BLAS_K;
  int BLAS_LDC    = BLAS_M;

  _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

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
  /*                                -0_5, +0_1, +0_2, +0_3     */
  int    const gamma_map_id[4]   = {   6,   10,   11,  12 };
  double const gamma_map_sign[4] = { -1.,  +1.,  +1.,  +1. };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  char filename[400];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  struct timeval start_time, end_time;
  int ymax = 0;

  struct timeval ta, te;

#ifdef HAVE_CUDA
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
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

  gettimeofday ( &start_time, (struct timezone *)NULL );


  /* set the default values */
  if(filename_set==0) strcpy(filename, "p2gg.input");
  /* fprintf(stdout, "# [test-x-kernel] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /******************************************************
   * initialize MPI parameters for cvc
   ******************************************************/
  mpi_init(argc, argv);


  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [test-x-kernel] git version = %s\n", g_gitversion);
  }



  /******************************************************
   * set number of openmp threads
   ******************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [test-x-kernel] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test-x-kernel] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test-x-kernel] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

#ifdef HAVE_CUDA

  /******************************************************
   * initialize cuda device
   ******************************************************/
  if ( !g_cart_id ) 
  {
    fprintf(stdout, "# [test-x-kernel] calling cuda initialization\n");
  }

  {
    int driver_version;
    CUDA_CHECK( cudaDriverGetVersion(&driver_version) );
    if ( g_cart_id == 0 ) printf ( "# [test-x-kernel] CUDA Driver version = %d\n", driver_version);

    int runtime_version;
    CUDA_CHECK( cudaRuntimeGetVersion(&runtime_version) );
    if ( g_cart_id == 0 ) printf ("# [test-x-kernel] CUDA Runtime version = %d\n", runtime_version);

    int device_count = 0;
    CUDA_CHECK ( cudaGetDeviceCount(&device_count) );
    if ( g_cart_id == 0 ) printf ("# [test-x-kernel] CUDA device count = %d\n", device_count );
    if (device_count == 0) fprintf(stderr, "[test-x-kernel] No CUDA devices found");


    cudaDeviceProp deviceProp;
    for (int i = 0; i < device_count; i++) 
    {
      CUDA_CHECK( cudaGetDeviceProperties(&deviceProp, i) );
      if ( !g_cart_id ) printf ("# [test-x-kernel] Found device %d: %s\n", i, deviceProp.name);
    }


    // assign a device id to call cudaSetDevice; seems to require some work in case of mpi with multi gpu;
    // following the method in quda here

    g_device_id = 0;
#ifdef HAVE_MPI
    int hostname_len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    char ** hostname_list = init_2level_ctable (g_nproc, MPI_MAX_PROCESSOR_NAME );
    if ( hostname_list == NULL )
    {
      fprintf (stderr, "[test-x-kernel] Error from init_level_table   %s %d\n", __FILE__, __LINE__ );
      EXIT (1);
    }
    
    MPI_Get_processor_name ( hostname, &hostname_len );

    if ( g_verbose > 2 ) fprintf (stdout, "# [test-x-kernel] proc %4d hostname %s   %s %d\n", g_cart_id, hostname, __FILE__, __LINE__ );

    MPI_Allgather ( hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hostname_list[0], MPI_MAX_PROCESSOR_NAME, MPI_CHAR, g_cart_grid );

    for ( int i = 0; i < g_cart_id; i++) {
      if (!strncmp( hostname, hostname_list[i], MPI_MAX_PROCESSOR_NAME) ) { g_device_id++; }
    }

    // exit with error is not necessary here; but needs check on how to over-use gpus by more than one mpi process
    if ( g_device_id >= device_count )
    {
      fprintf (stderr, "[test-x-kernel] proc %4d device id %d >= number of devices found %d   %s %d\n", g_cart_id, g_device_id, device_count,
         __FILE__, __LINE__  );
      EXIT (1);
    }
    fini_2level_ctable ( &hostname_list );
#endif  // of HAVE_MPI

    if ( g_verbose > 2 ) fprintf (stdout, "# [test-x-kernel] proc %4d device id %d   %s %d\n", g_cart_id, g_device_id, __FILE__, __LINE__ );

    CUDA_CHECK ( cudaSetDevice ( g_device_id ) );

    /* create cublas handle, bind a stream */
    CUBLAS_CHECK ( cublasCreate(&cublasH) );

    CUDA_CHECK ( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );
    CUBLAS_CHECK ( cublasSetStream(cublasH, stream) );

  }

#endif  // of HAVE_CUDA

  /******************************************************
   * set up lattice geometry fields
   ******************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[test-x-kernel] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  size_t sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);

  /******************************************************
   * set up mpi exchanges for even-odd fields
   ******************************************************/
  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();


  /******************************************************
   * init gauge field
   ******************************************************/
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);

  if( strcmp(gaugefilename_prefix,"identity")==0 ) 
  {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [test-x-kernel] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  } else if( strcmp(gaugefilename_prefix,"random")==0 )
  {
    if(g_cart_id==0) fprintf(stdout, "\n# [test-x-kernel] initializing random matrices\n");
    random_gauge_field ( g_gauge_field, 1.0 );
  } else if( strcmp(gaugefilename_prefix,"none")==0 )
  {
    if(g_cart_id==0) fprintf(stdout, "\n# [test-x-kernel] initializing zero gauge field\n");
  } else
  {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [test-x-kernel] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  }

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
#if _WITH_TIMER
  gettimeofday ( &ta, (struct timezone *)NULL );
#endif
  /* exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up ); */
  exitstatus = gauge_field_eq_gauge_field_ti_bcfactor ( &gauge_field_with_phase, g_gauge_field, -1. );

  if(exitstatus != 0) {
    fprintf(stderr, "[test-x-kernel] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }
#if _WITH_TIMER
  gettimeofday ( &te, (struct timezone *)NULL );
  show_time ( &ta, &te, __FILE__, "gauge_field_eq_gauge_field_ti_phase", g_cart_id == 0 );
#endif


#if _WITH_TIMER
  gettimeofday ( &ta, (struct timezone *)NULL );
#endif
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[test-x-kernel] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }
#if _WITH_TIMER
  gettimeofday ( &te, (struct timezone *)NULL );
  show_time ( &ta, &te, __FILE__ , "plaquetteria", g_cart_id == 0 );
#endif

  /***********************************************
   * set io process
   ***********************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[test-x-kernel] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [test-x-kernel] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  /***********************************************
   * initialize clover, mzz and mzz_inv
   *
   * FOR ZERO TWISTED MASS g_mu = 0
   ***********************************************/
#if _WITH_TIMER
  gettimeofday ( &ta, (struct timezone *)NULL );
#endif
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, 0, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test-x-kernel] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }
#if _WITH_TIMER
  gettimeofday ( &te, (struct timezone *)NULL );
  show_time ( &ta, &te, __FILE__ , "init_clover", g_cart_id == 0 );
#endif

  /***********************************************************
   * filename for output data
   ***********************************************************/
  char output_filename[400];
  sprintf ( output_filename, "%s.%d.h5", g_outfile_prefix, Nconf );

  /***********************************************************
   * space for spinor field
   ***********************************************************/
  double ** spinor_field = init_2level_dtable ( 4, _GSI( (size_t)(VOLUME) ));
  if( spinor_field == NULL ) 
  {
    fprintf(stderr, "[test-x-kernel] Error from init_level_table %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }


  /***********************************************************
   * random volume source
   ***********************************************************/
  prepare_volume_source ( spinor_field[0], VOLUME );

  /***********************************************************
   * prepare kervx
   *
   * !!!! mu runs fasters (innermost index) !!!!
   * !!!! mu is summed over in the X-kernel !!!!
   ***********************************************************/
#if _WITH_TIMER
  gettimeofday ( &ta, (struct timezone *)NULL );
#endif
  double * kervx = (double*) malloc ( 384 * VOLUME * sizeof ( double ) );
  if ( kervx == NULL )
  {
    fprintf(stderr, "[test-x-kernel] Error from malloc   %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  // set kervx
  ranlxd ( kervx, 384 * VOLUME );

#if _WITH_TIMER
  gettimeofday ( &te, (struct timezone *)NULL );
  show_time ( &ta, &te, __FILE__ , "X-prepare-kervx", g_cart_id == 0 );
#endif
  
  double _Complex * h_x = init_1level_ztable ( 96 * 12 * VOLUME * sizeof (double _Complex ) );


#ifdef HAVE_CUDA
  /***********************************************************
   * evec field on device
   ***********************************************************/

  /* allocate eigenvector field on device */
  double * d_v = nullptr;
  CUDA_CHECK_MALLOC ( cudaMalloc(reinterpret_cast<void **>(&d_v), sizeof_spinor_field) );
  
  // double * d_w = nullptr;
  // CUDA_CHECK_MALLOC ( cudaMalloc(reinterpret_cast<void **>(&d_w), sizeof_spinor_field ) );
  cuDoubleComplex * d_x = nullptr;
  CUDA_CHECK_MALLOC ( cudaMalloc(reinterpret_cast<void **>(&d_x), 96 * 12 * VOLUME * sizeof(cuDoubleComplex) ) );
   
  /* allocate kernel field on device */
  double * d_kervx =  nullptr;
  CUDA_CHECK_MALLOC ( cudaMalloc(reinterpret_cast<void **>(&d_kervx), sizeof(double) * 384 * VOLUME ) );

#endif

  /***********************************************************/
  /***********************************************************/

#ifdef HAVE_CUDA
  /* copy kerv field to device */
  // CUDA_CHECK(cudaMemcpyAsync( d_kervx, kervx, sizeof(double) * 384 * VOLUME, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpy ( d_kervx, kervx, sizeof(double) * 384 * VOLUME, cudaMemcpyHostToDevice ));
#endif

  /***********************************************************/
  /***********************************************************/

  /* copy data from host to device */
  // CUDA_CHECK ( cudaMemcpyAsync( d_v, spinor_field[0], sizeof_spinor_field, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK ( cudaMemcpy( d_v, spinor_field[0], sizeof_spinor_field, cudaMemcpyHostToDevice ));
  CUDA_CHECK(cudaStreamSynchronize(stream));
 
  /***********************************************************
   * vertex application
   ***********************************************************/
#if _WITH_TIMER
  gettimeofday ( &ta, (struct timezone *)NULL );
#endif
  apply_kernel ( stream, cublasH, d_x, d_v, d_kervx, VOLUME );

#if _WITH_TIMER
  gettimeofday ( &te, (struct timezone *)NULL );
  show_time ( &ta, &te, __FILE__ , "project_v_dag_g_v", g_cart_id == 0 );
#endif

  /***********************************************************
   * copy data from device to host
   ***********************************************************/
  // CUDA_CHECK ( cudaMemcpyAsync( spinor_field[1], d_w, sizeof_spinor_field, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK ( cudaMemcpy( h_x, d_x, 96 * 12 * VOLUME * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost ));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  /***********************************************************
   * apply kernel on cpu
   ***********************************************************/

  // memset ( spinor_field[2], 0, sizeof_spinor_field );
  // memcpy (spinor_field[2], spinor_field[0], sizeof_spinor_field );
  // g5_phi ( spinor_field[2], VOLUME );

  // spinor_field_eq_gamma_ti_spinor_field ( spinor_field[2], 12, spinor_field[0], VOLUME );

  // apply kernel X with given kervx
  double _Complex *** x_field = init_3level_ztable ( VOLUME, 12, 96 );
  if ( x_field == NULL )
  {
    fprintf (stderr, "[] Error from init_level_table %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }
#pragma omp parallel
{
  double sp[4][24], spsum[24];
#pragma omp for
  for ( size_t ix = 0; ix < VOLUME; ix++ )
  {
    double * const _ev = spinor_field[0] + _GSI(ix);

    for ( int imu = 0; imu < 4; imu++)
    {
      _fv_eq_gamma_ti_fv ( sp[imu], gamma_map_id[imu], _ev );
      _fv_ti_eq_re ( sp[imu], gamma_map_sign[imu] );
    }

    for ( int icomb = 0; icomb < 6; icomb++ )
    {
      for ( int inu = 0; inu < 4; inu++ )
      {
        for ( int ilam = 0; ilam < 4; ilam++ )
        {
          int const k = 4 * ( 4 * icomb + inu ) + ilam;

          memset (spsum, 0, 24*sizeof(double) );
          for ( int imu = 0; imu < 4; imu++ )
          {
            int const kk = 4 * k + imu;
            _fv_pl_eq_fv_ti_re ( spsum, sp[imu], kervx[384 * ix + kk] );
          }

          for ( int ia = 0; ia < 12; ia++ )
          {
            x_field[ix][ia][k] = spsum[2*ia+0] + spsum[2*ia+1] * I;
          }

        }  // end of loop on lambda
      }  // end of loop on nu
    }  // end of loop on icomb
  }  // loop on VOLUME

}  // end of parallel region



  /* for ( unsigned ix = 0; ix < 12*VOLUME; ix++ )
  {
    fprintf (stdout, "sf %6d    %25.16e %25.16e     %25.16e %25.16e    %25.16e %25.16e    %25.16e %25.16e\n",
        ix,
        spinor_field[0][2*ix+0],
        spinor_field[0][2*ix+1],
        spinor_field[1][2*ix+0],
        spinor_field[1][2*ix+1],
        spinor_field[2][2*ix+0],
        spinor_field[2][2*ix+1],
        fabs ( spinor_field[1][2*ix+0] - spinor_field[2][2*ix+0] ),
        fabs ( spinor_field[1][2*ix+1] - spinor_field[2][2*ix+1] ) );
  } */

  double norm_diff = 0.;

  for ( unsigned ix = 0; ix < VOLUME; ix++ )
  {
    for ( int i = 0; i < 12; i++ )
    {
      for ( unsigned int k = 0; k < 96; k++)
      {
        double dtmp =  cabs( h_x[ 96*(12*ix+i)+k]  - x_field[ix][i][k] );

        fprintf (stdout, "x %6d %3d %3d   %25.16e %25.16e     %25.16e %25.16e     %25.16e \n",
          ix, i, k,
          creal ( h_x[ 96*(12*ix+i) + k] ), cimag ( h_x[ 96*(12*ix+i) + k] ),
          creal( x_field[ix][i][k]), cimag( x_field[ix][i][k]), dtmp);

        norm_diff += dtmp + dtmp;
      }
    }
  }
  fprintf ( stdout, "# [] total norm diff = %e  %s %d\n", norm_diff, __FILE__, __LINE__ );


  fini_3level_ztable ( &x_field );
  fini_1level_ztable ( &h_x );
  fini_1level_dtable ( &kervx );

#if HAVE_CUDA
  CUDA_CHECK(cudaFree(d_v));
  // CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_kervx));
#endif

#if HAVE_CUDA
  CUBLAS_CHECK(cublasDestroy(cublasH));
  CUDA_CHECK(cudaStreamDestroy(stream));

  // end this CUDA context
  CUDA_CHECK ( cudaDeviceReset() );
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/


  fini_2level_dtable ( &spinor_field);


  /* free clover matrix terms */
  fini_clover ( &mzz, &mzzinv );


  free(g_gauge_field);
  free( gauge_field_with_phase );


  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_datatypes();
  MPI_Finalize();
#endif


  gettimeofday ( &end_time, (struct timezone *)NULL );
  // show_time ( &start_time, &end_time, "test-x-kernel", "runtime", g_cart_id == 0 );
  show_time ( &start_time, &end_time, __FILE__, "runtime", g_cart_id == 0 );

  return(0);
}
