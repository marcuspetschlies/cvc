/****************************************************
 * hlbl_lm_contract_gpu
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

#  ifdef HAVE_KQED
#    include "KQED.h"
#  endif

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

#define _WITH_TIMER 1

#define _EVEC_TEST 0

using namespace cvc;

typedef struct {
  double re, im;
} cplx_t;

typedef double kerv_type[6][4][4][4] KQED_ALIGN;

/***********************************************************
 * wrapper for Lambda = 0.4 kernel
 ***********************************************************/
void QED_kernel_L0P4( const double xv[4], const double yv[4], const struct QED_kernel_temps t, double kerv[6][4][4][4] )
{
  QED_Mkernel_L2(0.4, xv, yv, t, kerv);
}

/***********************************************************
 * KQED kernel function pointer
 ***********************************************************/
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
  int evec_num =  0;

  struct timeval ta, te;

#ifdef HAVE_CUDA
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
#endif

  // QED_kernel_LX_ptr KQED_LX[1] = { QED_kernel_L2 };
  QED_kernel_LX_ptr KQED_LX[1] = { QED_kernel_L0P4 };


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:y:n:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'y':
      ymax = atoi ( optarg );
      break;
    case 'n':
      evec_num = atoi ( optarg );
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
  /* fprintf(stdout, "# [hlbl_lm_contract_gpu] Reading input from file %s\n", filename); */
  read_input_parser(filename);

  /******************************************************
   * initialize MPI parameters for cvc
   ******************************************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [hlbl_lm_contract_gpu] git version = %s\n", g_gitversion);
  }


  /******************************************************
   * set number of openmp threads
   ******************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [hlbl_lm_contract_gpu] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [hlbl_lm_contract_gpu] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[hlbl_lm_contract_gpu] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

#ifdef HAVE_CUDA

  /******************************************************
   * initialize cuda device
   ******************************************************/
  if ( !g_cart_id ) 
  {
    fprintf(stdout, "# [hlbl_lm_contract_gpu] calling cuda initialization\n");
  }

  {
    int driver_version;
    CUDA_CHECK( cudaDriverGetVersion(&driver_version) );
    if ( g_cart_id == 0 ) printf ( "# [hlbl_lm_contract_gpu] CUDA Driver version = %d\n", driver_version);

    int runtime_version;
    CUDA_CHECK( cudaRuntimeGetVersion(&runtime_version) );
    if ( g_cart_id == 0 ) printf ("# [hlbl_lm_contract_gpu] CUDA Runtime version = %d\n", runtime_version);

    int device_count = 0;
    CUDA_CHECK ( cudaGetDeviceCount(&device_count) );
    if ( g_cart_id == 0 ) printf ("# [hlbl_lm_contract_gpu] CUDA device count = %d\n", device_count );
    if (device_count == 0) fprintf(stderr, "[hlbl_lm_contract_gpu] No CUDA devices found");


    cudaDeviceProp deviceProp;
    for (int i = 0; i < device_count; i++) 
    {
      CUDA_CHECK( cudaGetDeviceProperties(&deviceProp, i) );
      if ( !g_cart_id ) printf ("# [hlbl_lm_contract_gpu] Found device %d: %s\n", i, deviceProp.name);
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
      fprintf (stderr, "[hlbl_lm_contract_gpu] Error from init_level_table   %s %d\n", __FILE__, __LINE__ );
      EXIT (1);
    }
    
    MPI_Get_processor_name ( hostname, &hostname_len );

    if ( g_verbose > 2 ) fprintf (stdout, "# [hlbl_lm_contract_gpu] proc %4d hostname %s   %s %d\n", g_cart_id, hostname, __FILE__, __LINE__ );

    MPI_Allgather ( hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hostname_list[0], MPI_MAX_PROCESSOR_NAME, MPI_CHAR, g_cart_grid );

    for ( int i = 0; i < g_cart_id; i++) {
      if (!strncmp( hostname, hostname_list[i], MPI_MAX_PROCESSOR_NAME) ) { g_device_id++; }
    }

    // exit with error is not necessary here; but needs check on how to over-use gpus by more than one mpi process
    if ( g_device_id >= device_count )
    {
      fprintf (stderr, "[hlbl_lm_contract_gpu] proc %4d device id %d >= number of devices found %d   %s %d\n", g_cart_id, g_device_id, device_count,
         __FILE__, __LINE__  );
      EXIT (1);
    }
#endif  // of HAVE_MPI

    if ( g_verbose > 2 ) fprintf (stdout, "# [hlbl_lm_contract_gpu] proc %4d device id %d   %s %d\n", g_cart_id, g_device_id, __FILE__, __LINE__ );

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
    fprintf(stderr, "[hlbl_lm_contract_gpu] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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

  if(!(strcmp(gaugefilename_prefix,"identity")==0)) 
  {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [hlbl_lm_contract_gpu] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else
  {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [hlbl_lm_contract_gpu] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
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
    fprintf(stderr, "[hlbl_lm_contract_gpu] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
    fprintf(stderr, "[hlbl_lm_contract_gpu] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }
#if _WITH_TIMER
  gettimeofday ( &te, (struct timezone *)NULL );
  show_time ( &ta, &te, __FILE__ , "plaquetteria", g_cart_id == 0 );
#endif

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
    fprintf(stderr, "[hlbl_lm_contract_gpu] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }
#if _WITH_TIMER
  gettimeofday ( &te, (struct timezone *)NULL );
  show_time ( &ta, &te, __FILE__ , "init_clover", g_cart_id == 0 );
#endif



  /***********************************************
   * set io process
   ***********************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[hlbl_lm_contract_gpu] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [hlbl_lm_contract_gpu] proc%.4d has io proc id %d\n", g_cart_id, io_proc );


  /***********************************************************
   * unit for x, y
   ***********************************************************/
  double const xunit[2] = { mmuon * alat[0], mmuon * alat[1] };

  /***********************************************************
   * filename for output data
   ***********************************************************/
  char output_filename[400];
  sprintf ( output_filename, "%s.%d.h5", g_outfile_prefix, Nconf );

  /***********************************************************
   * set up QED Kernel package
   ***********************************************************/
  struct QED_kernel_temps kqed_t ;

  if( initialise( &kqed_t ) )
  {
    fprintf(stderr, "[hlbl_lm_contract_gpu] Error from kqed initialise, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(19);
  }

  /***********************************************************
   * space for eigenvectors
   ***********************************************************/
  double ** evec_field = init_2level_dtable ( evec_num, _GSI( (size_t)(VOLUME) ));
  if( evec_field == NULL ) 
  {
    fprintf(stderr, "[hlbl_lm_contract_gpu] Error from init_level_table %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }

  /***********************************************************
   * read eigenvectors from file
   ***********************************************************/
  for ( int ievec = 0; ievec < evec_num; ievec++ )
  {
    double ** spinor_field = init_2level_dtable ( 1, _GSI( (size_t)(VOLUME) ));
    if( spinor_field == NULL ) 
    {
      fprintf(stderr, "[hlbl_lm_contract_gpu] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    // TEST
    // random evec fields
    // prepare_volume_source ( spinor_field[0], VOLUME );

    sprintf (filename, "%s/eigVec_eV%d", filename_prefix, ievec);
#if _WITH_TIMER
    gettimeofday ( &ta, (struct timezone *)NULL );
#endif
    exitstatus = read_lime_spinor ( spinor_field[0], filename, 0);

    // TEST
    // write evec to file
    // exitstatus = write_propagator ( spinor_field[0], filename, 0, 64 );

#if _WITH_TIMER
    gettimeofday ( &te, (struct timezone *)NULL );
    show_time ( &ta, &te, __FILE__ , "read_lime_spinor", g_cart_id == 0 );
#endif
    if ( exitstatus != 0 )
    {
      fprintf( stderr, "[hlbl_lm_contract_gpu] Error from read_lime_spinor, status %d  %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(1);
    }

    memcpy ( evec_field[ievec], spinor_field[0], sizeof_spinor_field );

    fini_2level_dtable ( &spinor_field );
  }

#ifdef HAVE_CUDA
  /***********************************************************
   * evec field on device
   ***********************************************************/

  /* allocate eigenvector field on device */
  double * d_v = nullptr;
  CUDA_CHECK ( cudaMalloc(reinterpret_cast<void **>(&d_v), sizeof(double) * evec_num * 24 * VOLUME ) );
   
  /* copy data from host to device */
  CUDA_CHECK(cudaMemcpyAsync( d_v, evec_field[0], sizeof(double) * evec_num * 24*VOLUME, cudaMemcpyHostToDevice, stream));

  /* allocate kernel field on device */
  double * d_kervx =  nullptr;
  CUDA_CHECK ( cudaMalloc(reinterpret_cast<void **>(&d_kervx), sizeof(double) * 384 * VOLUME ) );

#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * X contractions
   ***********************************************************/
#if _WITH_TIMER
  struct timeval X_timer[2];
  gettimeofday ( X_timer, (struct timezone *)NULL );
#endif

  for ( int isrc = 0; isrc < g_source_location_number; isrc++ )
  {
    int gsx[4], sx[4];
    gsx[0] = ( g_source_coords_list[isrc][0] +  T_global ) %  T_global;
    gsx[1] = ( g_source_coords_list[isrc][1] + LX_global ) % LX_global;
    gsx[2] = ( g_source_coords_list[isrc][2] + LY_global ) % LY_global;
    gsx[3] = ( g_source_coords_list[isrc][3] + LZ_global ) % LZ_global;

    int source_proc_id = -1;
    exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[hlbl_lm_contract_gpu] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    int ** xv = init_2level_itable ( VOLUME, 4 );
    if ( xv == 0 )
    {
      fprintf(stderr, "[hlbl_lm_contract_gpu] Error from init_level_table    %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

#if _WITH_TIMER
    gettimeofday ( &ta, (struct timezone *)NULL );
#endif
#pragma omp parallel for
    for ( size_t ix = 0; ix < VOLUME; ix++ )
    {
      int const x[4] = {
                    ( g_lexic2coords[ix][0] + g_proc_coords[0] * T  - gsx[0] + T_global  ) % T_global,
                    ( g_lexic2coords[ix][1] + g_proc_coords[1] * LX - gsx[1] + LX_global ) % LX_global,
                    ( g_lexic2coords[ix][2] + g_proc_coords[2] * LY - gsx[2] + LY_global ) % LY_global,
                    ( g_lexic2coords[ix][3] + g_proc_coords[3] * LZ - gsx[3] + LZ_global ) % LZ_global };

      site_map_zerohalf ( xv[ix], x );
    }
#if _WITH_TIMER
    gettimeofday ( &te, (struct timezone *)NULL );
    show_time ( &ta, &te, __FILE__ , "X-prepare-xv", g_cart_id == 0 );
#endif


    double ** spinor_field = init_2level_dtable ( 96, _GSI( (size_t)(VOLUME) ));
    if( spinor_field == NULL )
    {
      fprintf(stderr, "[hlbl_lm_contract_gpu] Error from init_level_table %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }


    for ( int iy = 1; iy <= ymax; iy++ ) 
    {
      for ( int isign = 0; isign < ysign_num; isign++ )
      {

        int const yv[4] = {
            iy * ysign_comb[isign][0],
            iy * ysign_comb[isign][1],
            iy * ysign_comb[isign][2],
            iy * ysign_comb[isign][3] };

        double const ym[4] = {
            yv[0] * xunit[0],
            yv[1] * xunit[0],
            yv[2] * xunit[0],
            yv[3] * xunit[0] };

        double _Complex ***** X = init_5level_ztable ( 6, 4, 4, evec_num, evec_num );
        double _Complex * p = init_1level_ztable ( evec_num * 96 );
        if ( X == NULL || p == NULL )
        {
          fprintf(stderr, "[hlbl_lm_contract_gpu] Error from init_level_table  %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }
#ifdef HAVE_CUDA
        double _Complex * h_p = init_1level_ztable ( evec_num * evec_num * 96 );
        if ( h_p == NULL )
        {
          fprintf(stderr, "[hlbl_lm_contract_gpu] Error from init_level_table  %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }
#endif

#if _WITH_TIMER
        gettimeofday ( &ta, (struct timezone *)NULL );
#endif
        double * kervx = (double*) malloc ( VOLUME * sizeof ( double ) );
        if ( kervx == NULL )
        {
          fprintf(stderr, "[hlbl_lm_contract_gpu] Error from malloc   %s %d\n", __FILE__, __LINE__ );
          EXIT(1);
        }

#pragma omp parallel
{
#pragma omp for
        for ( size_t ix = 0; ix < VOLUME; ix++ )
        {
          double kerv[6][4][4][4] KQED_ALIGN ;

          double const xm[4] = {
                xv[ix][0] * xunit[0],
                xv[ix][1] * xunit[0],
                xv[ix][2] * xunit[0],
                xv[ix][3] * xunit[0] };

          KQED_LX[0]( xm, ym, kqed_t, kerv );
          set_kernel_pointx ( kervx+384*ix, kerv );
        }
}  // end parallel region
#if _WITH_TIMER
        gettimeofday ( &te, (struct timezone *)NULL );
        show_time ( &ta, &te, __FILE__ , "X-prepare-kervx", g_cart_id == 0 );
#endif
#ifdef HAVE_CUDA
        /* copy kerv field to device */
        CUDA_CHECK(cudaMemcpyAsync( d_kervx, kervx, sizeof(double) * 384 * VOLUME, cudaMemcpyHostToDevice, stream));
#endif

        /***********************************************************
         * X vertex application and eigenvector subspace projection
         * for all eigenvectors
         ***********************************************************/
#if _WITH_TIMER
        gettimeofday ( &ta, (struct timezone *)NULL );
#endif
        project_v_dag_g_v ( stream, cublasH, h_p, d_v, d_kervx, evec_num, VOLUME );

#if _WITH_TIMER
        gettimeofday ( &te, (struct timezone *)NULL );
        show_time ( &ta, &te, __FILE__ , "X-update-X", g_cart_id == 0 );
#endif

#if _WITH_TIMER
        gettimeofday ( &ta, (struct timezone *)NULL );
#endif

        /***********************************************************
         * add h_p to X field for V^H Gamma S
         *
         * outer evec loop counter iv for s in S field
         * inner evec loop counter iv2 for v in V field
         ***********************************************************/
#pragma omp parallel for
        for ( unsigned int iv = 0; iv < evec_num; iv++ )
        {
          for ( int icomb = 0; icomb < 6; icomb++ )
          {
            for ( int inu = 0; inu < 4; inu++ )
            {
              for ( int ilam = 0; ilam < 4; ilam++ )
              {
                unsigned int const i = 4 * ( 4 * icomb + inu ) + ilam;

                for ( int iv2 = 0; iv2 < evec_num; iv2++ )
                {
                  X[icomb][inu][ilam][iv2][iv] += h_p[ ( evec_num * ( 96 * iv + i ) + iv2 ) ];
                }  // end of loop on evecs

              }  // end of loop on lambda
            }  // end of loop on nu
          }  // end of loop on comb = [rho,sigma]
        }  // end of loop on evecs

#if _WITH_TIMER
        gettimeofday ( &te, (struct timezone *)NULL );
        show_time ( &ta, &te, __FILE__ , "X-update-X", g_cart_id == 0 );
#endif

        free ( kervx ); kervx = NULL;

#ifdef HAVE_MPI
        double _Complex * X_buffer = (double _Complex*)malloc ( 96*evec_num*evec_num * sizeof(double _Complex) );
        memcpy ( X_buffer, X[0][0][0][0], 96*evec_num*evec_num * sizeof(double _Complex) );

        if ( MPI_Allreduce ( X_buffer, X[0][0][0][0], 192*evec_num*evec_num, MPI_DOUBLE, MPI_SUM, g_cart_grid) != 0 )
        {
          fprintf(stderr, "[hlbl_lm_contract_gpu] Error from MPI_Allreduce\n" );
          EXIT(1);
        }

        free ( X_buffer ); X_buffer = NULL;
#endif

        if ( io_proc == 2 )
        {
          int const ndim = 6;
          int const cdim[6] = { 6, 4, 4, evec_num, evec_num, 2};
          char tag[400];
          sprintf( tag, "/X/T%dX%dY%dZ%d/YT%dYX%dYY%dYZ%d", gsx[0], gsx[1], gsx[2], gsx[3], yv[0], yv[1], yv[2], yv[3] );
          exitstatus = write_h5_contraction ( (void *)(X[0][0][0][0]), NULL, output_filename, tag, "double", ndim, cdim );
          if ( exitstatus != 0 )
          {
            fprintf ( stderr, "[hlbl_lm_contract_gpu] Error from write_h5_contraction   %s %d\n", __FILE__, __LINE__ );
            EXIT(1);
          }
        }

        fini_5level_ztable ( &X );
#ifdef HAVE_CUDA
        fini_1level_ztable ( &h_p );
#endif
    
      }  // end of loop on y directions
    
    }  // end of loop on y distances

    fini_2level_dtable ( &spinor_field );
    fini_2level_itable ( &xv );

  }  // of loop on source locations

#if _WITH_TIMER
  gettimeofday ( X_timer+1, (struct timezone *)NULL );
  show_time ( X_timer, X_timer+1, __FILE__ , "X-total", g_cart_id == 0 );
#endif

  /***********************************************************/
  /***********************************************************/

  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/

#if HAVE_CUDA
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_kervx));
  CUBLAS_CHECK(cublasDestroy(cublasH));
  CUDA_CHECK(cudaStreamDestroy(stream));

  // end this CUDA context
  CUDA_CHECK ( cudaDeviceReset() );
#endif

#if 0
#endif  // of if 0

#ifdef HAVE_KQED
  free_QED_temps( &kqed_t  );
#endif

  fini_2level_dtable ( &evec_field );

  free(g_gauge_field);
  free( gauge_field_with_phase );

  /* free clover matrix terms */
  fini_clover ( &mzz, &mzzinv );

  free_geometry();

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif

  gettimeofday ( &end_time, (struct timezone *)NULL );
  // show_time ( &start_time, &end_time, "hlbl_lm_contract_gpu", "runtime", g_cart_id == 0 );
  show_time ( &start_time, &end_time, __FILE__, "runtime", g_cart_id == 0 );

  return(0);
}
