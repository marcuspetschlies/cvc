/****************************************************
 * hlbl_2p2_invert_contract
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
  int check_propagator_residual = 0;
  char filename[400];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  int first_solve_dummy = 1;
  struct timeval start_time, end_time;
  int ymax = 0;

  struct timeval ta, tb;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ch?f:y:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
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
  /* fprintf(stdout, "# [hlbl_2p2_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [hlbl_2p2_invert_contract] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1, 0);
  if(exitstatus != 0) {
    EXIT(1);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    EXIT(2);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(exitstatus != 0) {
    EXIT(3);
  }
#endif

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /******************************************************
   * report git version
   ******************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [hlbl_2p2_invert_contract] git version = %s\n", g_gitversion);
  }


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [hlbl_2p2_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [hlbl_2p2_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[hlbl_2p2_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[hlbl_2p2_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  size_t sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [hlbl_2p2_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [hlbl_2p2_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "# [hlbl_mII_invert_contract] Nconf = %d\n", Nconf);

  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [hlbl_mII_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = tmLQCD_read_gauge(Nconf);
    if(exitstatus != 0) {
      EXIT(5);
    }
  }
  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[hlbl_mII_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
  if (strcmp(gaugefilename_prefix,"identity")==0) {
    if(g_cart_id==0) fprintf(stdout, "\n# [hlbl_mII_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
    if(exitstatus != 0) {
      EXIT(6);
    }
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[hlbl_2p2_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[hlbl_2p2_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[hlbl_2p2_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***********************************************
   * set io process
   ***********************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[hlbl_2p2_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [hlbl_2p2_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***********************************************************
   ***********************************************************
   **
   ** dummy inversion for solver tuning
   **
   ** use volume source
   **
   ***********************************************************
   ***********************************************************/

  if ( first_solve_dummy )
  {
    /***********************************************************
     * initialize rng state
     ***********************************************************/
    exitstatus = init_rng_stat_file ( g_seed, NULL );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[hlbl_2p2_invert_contract] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
      EXIT( 50 );
    }
  
    double ** spinor_field = init_2level_dtable ( 2, _GSI( (size_t)VOLUME ));
    if( spinor_field == NULL ) {
      fprintf(stderr, "[hlbl_2p2_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    double ** spinor_work = init_2level_dtable ( 2, _GSI( (size_t)(VOLUME+RAND) ));
    if( spinor_work == NULL ) {
      fprintf(stderr, "[hlbl_2p2_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    if( ( exitstatus = prepare_volume_source ( spinor_field[0], VOLUME ) ) != 0 ) {
      fprintf(stderr, "[hlbl_2p2_invert_contract] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(64);
    }

    memcpy ( spinor_work[0], spinor_field[0], sizeof_spinor_field );
    memset ( spinor_work[1], 0, sizeof_spinor_field );

    /* full_spinor_work[1] = D^-1 full_spinor_work[0],
     * flavor id 0 
     */
    exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], 0 );
    if(exitstatus < 0) {
      fprintf(stderr, "[hlbl_2p2_invert_contract] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(19);
    }

    /* check residuum */
    if ( check_propagator_residual )
    {
      exitstatus = check_residual_clover (&(spinor_work[1]) , &(spinor_work[0]), gauge_field_with_phase, mzz[0], mzzinv[0], 1);
      if( exitstatus != 0 )
      {
        fprintf(stderr, "[hlbl_2p2_invert_contract] Error from check_residual_clover   %s %d\n", __FILE__, __LINE__);
        EXIT(123);
      }
    }

    fini_2level_dtable ( &spinor_work );
    fini_2level_dtable ( &spinor_field );

  }  /* end of first_solve_dummy */

  double ** spinor_work = init_2level_dtable ( 2, _GSI( (size_t)(VOLUME+RAND) ));
  if( spinor_work == NULL ) {
    fprintf(stderr, "[hlbl_2p2_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }

  double *** fwd_src = init_3level_dtable ( 2, 12, _GSI( (size_t)VOLUME ) );
  
  if( fwd_src == NULL ) 
  {
    fprintf(stderr, "[hlbl_2p2_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }

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
    fprintf(stderr, "[hlbl_2p2_invert_contract] Error from kqed initialise, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(19);
  } 

  /***********************************************************
   * loop on source positions
   ***********************************************************/
  for ( int isrc = 0; isrc < g_source_location_number; isrc++ )
  {
    /***********************************************************
     * determine source coordinates, find out, if source_location is in this process
     ***********************************************************/
    int gsx[4], sx[4];
    gsx[0] = ( g_source_coords_list[isrc][0] +  T_global ) %  T_global;
    gsx[1] = ( g_source_coords_list[isrc][1] + LX_global ) % LX_global;
    gsx[2] = ( g_source_coords_list[isrc][2] + LY_global ) % LY_global;
    gsx[3] = ( g_source_coords_list[isrc][3] + LZ_global ) % LZ_global;

    int source_proc_id = -1;
    exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[p2gg_invert_contract_local] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

   /***********************************************************
    * forward proapgators from source
    ***********************************************************/

    for ( int iflavor = 0; iflavor <= 1; iflavor ++ ) 
    {
      for ( int i = 0; i < 12; i++ ) 
      {
        memset ( spinor_work[0], 0, sizeof_spinor_field );
        memset ( spinor_work[1], 0, sizeof_spinor_field );

        if ( source_proc_id == g_cart_id ) 
        {
          spinor_work[0][_GSI(g_ipt[sx[0]][sx[1]][sx[2]][sx[3]]) + 2*i ] = 1.;
        }

        exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );

        if(exitstatus < 0) {
          fprintf(stderr, "[hlbl_2p2_invert_contract] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(19);
        }
 
        /* check residuum */
        if ( check_propagator_residual )
        {
          exitstatus = check_residual_clover (&(spinor_work[1]) , &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1);
          if(exitstatus != 0) {
            fprintf(stderr, "[hlbl_2p2_invert_contract] Error from check_residual_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(19);
          }
        }

        memcpy ( fwd_src[iflavor][i], spinor_work[1], sizeof_spinor_field );
     
        if ( g_write_propagator ) 
        {
          sprintf ( filename, "fwd_src.f%d.t%dx%dy%dz%d.sc%d.lime", iflavor, gsx[0] , gsx[1] ,gsx[2] , gsx[3], i );

          if ( ( exitstatus = write_propagator( spinor_work[1], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[hlbl_2p2_invert_contract] Error from write_propagator for %s, status was %d   %s %d\n", filename, exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
        }
      
      }  /* end of loop on spin-color components */

    }  /* end of loop on flavor */

    /***********************************************************
     * contractions
     ***********************************************************/

    double *** pimn = init_3level_dtable ( 4, 4, 2*VOLUME );

    for ( int nu = 0; nu < 4; nu++ )
    {
      for ( int mu = 0; mu < 4; mu++ )
      {
        contract_twopoint_xdep ( pimn[mu][nu], nu, mu, fwd_src[0], fwd_src[1], 4, 3, 1, 1., 64 );
      }
    }

    double *** z_pi = init_3level_dtable ( g_source_location_number, 6, 8 );

    for ( int isrc2 = 0; isrc2 < g_source_location_number; isrc2++ )
    {
      int const csx[4] = {
        g_source_coords_list[isrc2][0],
        g_source_coords_list[isrc2][1],
        g_source_coords_list[isrc2][2],
        g_source_coords_list[isrc2][3] };

      for ( int irs = 0; irs < 6; irs++ )
      {
        int const rho   = idx_comb[irs][0];
        // Updated 0 -> 1
        int const sigma = idx_comb[irs][1];

        for ( int nu = 0; nu < 4; nu++ )
        {

#ifdef HAVE_OPENMP
#pragma omp parallel
{
          double dtmp[2] = { 0 };
#endif
#ifdef HAVE_OPENMP
#pragma omp for
#endif
          for ( unsigned int iz = 0; iz < VOLUME; iz++ )
          {

            int const z[4] = {
                    ( g_lexic2coords[iz][0] + g_proc_coords[0] * T  - csx[0] + T_global  ) % T_global,
                    ( g_lexic2coords[iz][1] + g_proc_coords[1] * LX - csx[1] + LX_global ) % LX_global,
                    ( g_lexic2coords[iz][2] + g_proc_coords[2] * LY - csx[2] + LY_global ) % LY_global,
                    ( g_lexic2coords[iz][3] + g_proc_coords[3] * LZ - csx[3] + LZ_global ) % LZ_global };

            int zv[4];
            site_map_zerohalf ( zv, z );
            dtmp[0] += pimn[sigma][nu][2*iz  ] * zv[rho];
            dtmp[1] += pimn[sigma][nu][2*iz+1] * zv[rho];
          }  /* end of loop on ix */
#ifdef HAVE_OPENMP
#pragma omp critical 
{
#endif
          // Updated isrc -> isrc2
          z_pi[isrc2][irs][2 * nu    ] += dtmp[0];
          z_pi[isrc2][irs][2 * nu + 1] += dtmp[1];
#ifdef HAVE_OPENMP
}  /* end of critical region */

}  /* end of parallel region*/
#endif
        }  /* end of loop on nu */
      }
    }  /* end of loop on source locations  */

#ifdef HAVE_MPI
    {
      int const nitem = g_source_location_number * 6 * 8;
      double *** gz_pi = init_3level_dtable ( g_source_location_number, 6, 8 );

      if ( MPI_Reduce ( z_pi[0][0], gz_pi[0][0], nitem, MPI_DOUBLE, MPI_SUM, 0, g_cart_grid ) != MPI_SUCCESS )
      {
        fprintf ( stderr, "[] Error from MPI_Reduce   %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      }
      memcpy ( z_pi[0][0], gz_pi[0][0], nitem * sizeof ( double ) );
      fini_3level_dtable ( &gz_pi );
    }
#endif

    if ( io_proc == 2 )
    {
      int const ncdim = 3;
      // NOTE: Updated 4 -> 8
      int const cdim[3] = { g_source_location_number, 6, 8 };
      char tag[100];
      sprintf ( tag, "/z_pi/T%d_X%d_Y%d_Z%d", gsx[0], gsx[1], gsx[2], gsx[3] );
      exitstatus = write_h5_contraction ( z_pi[0][0], NULL, output_filename, tag, "double", ncdim, cdim );
      if ( exitstatus != 0 )
      {
        fprintf ( stderr, "# [hlbl_2p2_invert_contract] Error from write_h5_contraction, status %d   %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

    }
    
    fini_3level_dtable ( &z_pi );

    /***********************************************************
     * reductions with kernel
     ***********************************************************/

    fprintf(stdout, "[DEBUG] Begin reductions with kernel\n");

    double ***** kernel_sum = init_5level_dtable ( g_source_location_number, 6, 4, 4, 4 );
    if ( kernel_sum == NULL )
    {
      fprintf ( stderr, "[hlbl_2p2_invert_contract] Error from init_5level_dtable   %s %d\n", __FILE__, __LINE__ );
      EXIT(1);
    }

    QED_kernel_LX_ptr KQED_LX[4] = {
            QED_kernel_L0,
            QED_kernel_L1,
            QED_kernel_L2,
            QED_kernel_L3 };

    /***********************************************************
     * loop on source positions
     ***********************************************************/
    // NOTE: Updated from g_source_location -> g_source_location_number
    for ( int isrc2 = 0; isrc2 < g_source_location_number; isrc2++ )
    {
      int const csx[4] = {
          g_source_coords_list[isrc2][0],
          g_source_coords_list[isrc2][1],
          g_source_coords_list[isrc2][2],
          g_source_coords_list[isrc2][3] };
      fprintf(stdout, "[DEBUG] src = %d,%d,%d,%d  src2 = %d,%d,%d,%d\n",
              gsx[0], gsx[1], gsx[2], gsx[3],
              csx[0], csx[1], csx[2], csx[3]);

      // Updated to apply site_map_zerohalf to y = src2 - src
      // and to compute y_minus = src - src2
      int const y[4] = { csx[0] - gsx[0], csx[1] - gsx[1], csx[2] - gsx[2], csx[3] - gsx[3] };
      int yv[4];
      site_map_zerohalf(yv, y);

      double const ym[4] = {
        yv[0] * xunit[0],
        yv[1] * xunit[0],
        yv[2] * xunit[0],
        yv[3] * xunit[0] };
      double const ym_minus[4] = {
        -yv[0] * xunit[0],
        -yv[1] * xunit[0],
        -yv[2] * xunit[0],
        -yv[3] * xunit[0] };

                  
      for ( int irs = 0; irs < 6; irs++ )
      {
        for ( int nu = 0; nu < 4; nu++ )
        {

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
          double kerv1[6][4][4][4] KQED_ALIGN ;
          double kerv2[6][4][4][4] KQED_ALIGN ;

          double kernel_sum_thread[4][4] = {
              { 0., 0., 0., 0. }, { 0., 0., 0., 0. },
              { 0., 0., 0., 0. }, { 0., 0., 0., 0. } };
#ifdef HAVE_OPENMP
#pragma omp for
#endif
          for ( unsigned int ix = 0; ix < VOLUME; ix++ )
          {
            int const x[4] = { 
                         ( g_proc_coords[0]*T  + g_lexic2coords[ix][0] - gsx[0] + T_global  ) % T_global,
                         ( g_proc_coords[1]*LX + g_lexic2coords[ix][1] - gsx[1] + LX_global ) % LX_global,
                         ( g_proc_coords[2]*LY + g_lexic2coords[ix][2] - gsx[2] + LY_global ) % LY_global,
                         ( g_proc_coords[3]*LZ + g_lexic2coords[ix][3] - gsx[3] + LZ_global ) % LZ_global };

            int xv[4];
            site_map_zerohalf ( xv, x );

            double const xm[4] = {
                xv[0] * xunit[0],
                xv[1] * xunit[0],
                xv[2] * xunit[0],
                xv[3] * xunit[0] };

            // Updated to also compute x - y
            int const x_mi_y[4] = {
              x[0] - y[0],
              x[1] - y[1],
              x[2] - y[2],
              x[3] - y[3] };

            int xv_mi_yv[4];
            site_map_zerohalf ( xv_mi_yv, x_mi_y );

            double const xm_mi_ym[4] = {
                xv_mi_yv[0] * xunit[0],
                xv_mi_yv[1] * xunit[0],
                xv_mi_yv[2] * xunit[0],
                xv_mi_yv[3] * xunit[0] };


            /***********************************************************
             * loop on kernel types
             ***********************************************************/
            for ( int ikernel = 0; ikernel < 4; ikernel++ )
            {
              KQED_LX[ikernel]( xm, ym,       kqed_t, kerv1 );
              KQED_LX[ikernel]( ym, xm,       kqed_t, kerv2 );
    
              for ( int mu = 0; mu < 4; mu++ )
              {
                for ( int lambda = 0; lambda < 4; lambda++ )
                {
                  kernel_sum_thread[ikernel][0] +=  ( kerv1[irs][mu][nu][lambda] + kerv2[irs][nu][mu][lambda] ) * pimn[mu][lambda][2*ix  ];
                  kernel_sum_thread[ikernel][1] +=  ( kerv1[irs][mu][nu][lambda] + kerv2[irs][nu][mu][lambda] ) * pimn[mu][lambda][2*ix+1];
                  // TODO: Don't think we need this
                  kerv1[irs][mu][nu][lambda] = 0.0;
                }
              }

              KQED_LX[ikernel]( xm_mi_ym, ym_minus, kqed_t, kerv1 );

              for ( int mu = 0; mu < 4; mu++ )
              {
                for ( int lambda = 0; lambda < 4; lambda++ )
                {
                  kernel_sum_thread[ikernel][2] +=    kerv1[irs][mu][lambda][nu] * pimn[mu][lambda][2*ix  ];
                  kernel_sum_thread[ikernel][3] +=    kerv1[irs][mu][lambda][nu] * pimn[mu][lambda][2*ix+1];
                }
              }
            }

          }  /* end of loop on ix */
#ifdef HAVE_OPENMP
#pragma omp critical
{
#endif
          for ( int ikernel = 0; ikernel < 4; ikernel++ )
          {
            // NOTE: Updated isrc -> isrc2
            kernel_sum[isrc2][irs][nu][ikernel][0] += kernel_sum_thread[ikernel][0];
            kernel_sum[isrc2][irs][nu][ikernel][1] += kernel_sum_thread[ikernel][1];
            kernel_sum[isrc2][irs][nu][ikernel][2] += kernel_sum_thread[ikernel][2];
            kernel_sum[isrc2][irs][nu][ikernel][3] += kernel_sum_thread[ikernel][3];
          }
#ifdef HAVE_OPENMP
}  /* end of critical region */
#endif

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

        }  /* end of nu */
      }  /* end of [rho,sigma] */
    }  /* end of loop on source locations */

#ifdef HAVE_MPI
    {
      int const nitem = g_source_location_number * 6 * 4 * 4 * 4;
      double *buffer = init_1level_dtable ( nitem );

      if ( MPI_Reduce ( kernel_sum[0][0][0][0], buffer, nitem, MPI_DOUBLE, MPI_SUM, 0, g_cart_grid ) != MPI_SUCCESS )
      {
        fprintf ( stderr, "[] Error from MPI_Reduce   %s %d\n", __FILE__, __LINE__ );
        EXIT(1);
      }
      memcpy ( kernel_sum[0][0][0][0], buffer, nitem * sizeof ( double ) );
      fini_1level_dtable ( &buffer);
    }
#endif

    if ( io_proc == 2 )
    {
      int const ncdim = 5;
      int const cdim[5] = { g_source_location_number, 6, 4, 4, 4 };
      char tag[100];
      sprintf ( tag, "/L_pi/T%d_X%d_Y%d_Z%d", gsx[0], gsx[1], gsx[2], gsx[3] );
      exitstatus = write_h5_contraction ( kernel_sum[0][0][0][0], NULL, output_filename, tag, "double", ncdim, cdim );
      if ( exitstatus != 0 )
      {
        fprintf ( stderr, "# [hlbl_2p2_invert_contract] Error from write_h5_contraction, status %d   %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(1);
      }

    }

    fini_5level_dtable ( &kernel_sum );

#if 0
    /***********************************************************
     * output
     ***********************************************************/
    char output_filename[400], type[200];
    sprintf ( output_filename, "%s.%d.T%d_X%d_Y%d_Z%d.lime", g_outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3] );
    sprintf ( type, "polarization-tensor-position-space" );

    exitstatus = write_lime_contraction ( pimn[0][0], output_filename, 64, 16, type, Nconf,  0);
    if ( exitstatus != 0 )
    {
      fprintf ( stderr, "# [hlbl_2p2_invert_contract] Error from write_lime_contraction, status %d   %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(12);
    }
#endif

    fini_3level_dtable ( &pimn );

  }  /* end of loop on source locations */

  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/
          
  fini_3level_dtable ( &fwd_src );
  fini_2level_dtable ( &spinor_work );

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

  /* free clover matrix terms */
  fini_clover ( &mzz, &mzzinv );

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
  show_time ( &start_time, &end_time, "hlbl_2p2_invert_contract", "runtime", g_cart_id == 0 );

  return(0);
}
