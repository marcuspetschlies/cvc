/****************************************************
 * hlbl_mII_invert_contract
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
 ***********************************************************/
inline void site_map (int xv[4], int const x[4] )
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


  int const ysign_num = 2;
  int const ysign_comb[16][4] = {
    { 1, 1, 1, 1},
    { 1, 1, 1,-1},
    { 1, 1,-1, 1},
    { 1, 1,-1,-1},
    { 1,-1, 1, 1},
    { 1,-1, 1,-1},
    { 1,-1,-1, 1},
    { 1,-1,-1,-1},
    {-1, 1, 1, 1},
    {-1, 1, 1,-1},
    {-1, 1,-1, 1},
    {-1, 1,-1,-1},
    {-1,-1, 1, 1},
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
  /* fprintf(stdout, "# [hlbl_mII_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [hlbl_mII_invert_contract] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [hlbl_mII_invert_contract] git version = %s\n", g_gitversion);
  }


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [hlbl_mII_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [hlbl_mII_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[hlbl_mII_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[hlbl_mII_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [hlbl_mII_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [hlbl_mII_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[hlbl_mII_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[hlbl_mII_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[hlbl_mII_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[hlbl_mII_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[hlbl_mII_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***********************************************
   * set io process
   ***********************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[hlbl_mII_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [hlbl_mII_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

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
      fprintf(stderr, "[hlbl_mII_invert_contract] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
      EXIT( 50 );
    }
  
    double ** spinor_field = init_2level_dtable ( 2, _GSI( (size_t)VOLUME ));
    if( spinor_field == NULL ) {
      fprintf(stderr, "[hlbl_mII_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    double ** spinor_work = init_2level_dtable ( 2, _GSI( (size_t)(VOLUME+RAND) ));
    if( spinor_work == NULL ) {
      fprintf(stderr, "[hlbl_mII_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    if( ( exitstatus = prepare_volume_source ( spinor_field[0], VOLUME ) ) != 0 ) {
      fprintf(stderr, "[hlbl_mII_invert_contract] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(64);
    }

    memcpy ( spinor_work[0], spinor_field[0], sizeof_spinor_field );
    memset ( spinor_work[1], 0, sizeof_spinor_field );

    /* full_spinor_work[1] = D^-1 full_spinor_work[0],
     * flavor id 0 
     */
    exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], 0 );
    if(exitstatus < 0) {
      fprintf(stderr, "[hlbl_mII_invert_contract] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(19);
    }

    /* check residuum */
    if ( check_propagator_residual )
    {
      exitstatus = check_residual_clover (&(spinor_work[1]) , &(spinor_work[0]), gauge_field_with_phase, mzz[0], mzzinv[0], 1);
      if( exitstatus != 0 )
      {
        fprintf(stderr, "[hlbl_mII_invert_contract] Error from check_residual_clover   %s %d\n", __FILE__, __LINE__);
        EXIT(123);
      }
    }

    fini_2level_dtable ( &spinor_work );
    fini_2level_dtable ( &spinor_field );

  }  /* end of first_solve_dummy */

  double ** spinor_work = init_2level_dtable ( 2, _GSI( (size_t)(VOLUME+RAND) ));
  if( spinor_work == NULL ) {
    fprintf(stderr, "[hlbl_mII_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }

  double *** fwd_src = init_3level_dtable ( 2, 12, _GSI( (size_t)VOLUME ) );
  
  double *** fwd_y   = init_3level_dtable ( 2, 12, _GSI( (size_t)VOLUME ) );

  double **** g_fwd_src = init_4level_dtable ( 2, 4, 12, _GSI( (size_t)VOLUME ) );

  if( fwd_src == NULL || fwd_y == NULL || g_fwd_src == NULL  )
  {
    fprintf(stderr, "[hlbl_mII_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }


  /***********************************************************
   * unit for x, y
   ***********************************************************/
  double const xunit[2] = { mmuon * alat[0], mmuon * alat[1] };

  /***********************************************************
   * output filename
   ***********************************************************/
  char output_filename[400];
  sprintf ( output_filename, "%s.h5", g_outfile_prefix );

  /***********************************************************
   * set up QED Kernel package
   ***********************************************************/
  struct QED_kernel_temps kqed_t ;

  if( initialise( &kqed_t ) )
  {
    fprintf(stderr, "[hlbl_mII_invert_contract] Error from kqed initialise, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
     * local kernel sum
     ***********************************************************/
    double ** kernel_sum = init_2level_dtable ( ymax + 1, 4 );
    if ( kernel_sum == NULL ) 
    {
      fprintf(stderr, "[hlbl_mII_invert_contract] Error from kqed initialise, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(19);
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
          fprintf(stderr, "[hlbl_mII_invert_contract] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(19);
        }
 
        /* check residuum */
        if ( check_propagator_residual )
        {
          exitstatus = check_residual_clover (&(spinor_work[1]) , &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1);
          if(exitstatus != 0) {
            fprintf(stderr, "[hlbl_mII_invert_contract] Error from check_residual_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(19);
          }
        }

        memcpy ( fwd_src[iflavor][i], spinor_work[1], sizeof_spinor_field );
     
        if ( g_write_propagator ) 
        {
          sprintf ( filename, "fwd_0.f%d.t%dx%dy%dz%d.sc%d.lime", iflavor, gsx[0] , gsx[1] ,gsx[2] , gsx[3], i );

          if ( ( exitstatus = write_propagator( spinor_work[1], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[hlbl_mII_invert_contract] Error from write_propagator for %s, status was %d   %s %d\n", filename, exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
        }
      
      }  /* end of loop on spin-color components */

    }  /* end of loop on flavor */

    /***********************************************************/
    /***********************************************************/


    /***********************************************************
     * g5 gsigma fwd
     ***********************************************************/
    for ( int iflavor = 0; iflavor < 2; iflavor++ )
    {
      for( int mu = 0; mu < 4; mu++ )
      {
        for ( int ib = 0; ib < 12; ib++)
        {
          spinor_field_eq_gamma_ti_spinor_field ( g_fwd_src[iflavor][mu][ib], mu, fwd_src[iflavor][ib], VOLUME );
          g5_phi ( g_fwd_src[iflavor][mu][ib], VOLUME );
        }
      }
    }

    /***********************************************************/
    /***********************************************************/


    /***********************************************************
     * loop on y = iy ( 1,1,1,1)
     ***********************************************************/
    for ( int iy = 1; iy <= ymax; iy++ )
    {

      /***********************************************************
       * loop on directions in 4-space
       ***********************************************************/
      for ( int isign = 0; isign < ysign_num; isign++ )
      {

        sprintf ( filename, "pi-tensor-mII.y%d.st%dsx%dsy%dsz%d", iy, 
            ysign_comb[isign][0], ysign_comb[isign][1], ysign_comb[isign][2], ysign_comb[isign][3] );

        int gsy[4], sy[4];
        gsy[0] = ( iy * ysign_comb[isign][0] + gsx[0] +  T_global ) %  T_global;
        gsy[1] = ( iy * ysign_comb[isign][1] + gsx[1] + LX_global ) % LX_global;
        gsy[2] = ( iy * ysign_comb[isign][2] + gsx[2] + LY_global ) % LY_global;
        gsy[3] = ( iy * ysign_comb[isign][3] + gsx[3] + LZ_global ) % LZ_global;

        int source_proc_id_y = -1;
        exitstatus = get_point_source_info (gsy, sy, &source_proc_id_y);
        if( exitstatus != 0 ) {
          fprintf(stderr, "[p2gg_invert_contract_local] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(123);
        }

        int const yv[4] = {
            iy * ysign_comb[isign][0],
            iy * ysign_comb[isign][1],
            iy * ysign_comb[isign][2],
            iy * ysign_comb[isign][3] };


        /***********************************************************/
        /***********************************************************/

        for ( int iflavor = 0; iflavor <= 1; iflavor++ ) 
        {
 
          /***********************************************************
           * forward proapgators from y
           ***********************************************************/
  
          for ( int i = 0; i < 12; i++ ) 
          {
            memset ( spinor_work[0], 0, sizeof_spinor_field );
            memset ( spinor_work[1], 0, sizeof_spinor_field );
      
            if ( source_proc_id_y == g_cart_id ) 
            {
              spinor_work[0][_GSI(g_ipt[sy[0]][sy[1]][sy[2]][sy[3]]) + 2*i ] = 1.;
            }
      
            exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
    
            if(exitstatus < 0) {
              fprintf(stderr, "[hlbl_mII_invert_contract] Error from _TMLQCD_INVERT, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(19);
            }
       
            /* check residuum */
            if ( check_propagator_residual ) 
            {
              exitstatus = check_residual_clover (&(spinor_work[1]) , &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1);
              if(exitstatus != 0) {
                fprintf(stderr, "[hlbl_mII_invert_contract] Error from check_residual_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(19);
              }
            }
      
            memcpy ( fwd_y[iflavor][i], spinor_work[1], sizeof_spinor_field );
           
            if ( g_write_propagator ) 
            {
              sprintf ( filename, "fwd_y.f%d.t%dx%dy%dz%d.y%d.st%dsx%dsy%dsz%d.sc%d.lime", iflavor, gsy[0] , gsy[1] ,gsy[2] , gsy[3], iy,
                ysign_comb[isign][0], ysign_comb[isign][1], ysign_comb[isign][2], ysign_comb[isign][3], i );
    
              if ( ( exitstatus = write_propagator( spinor_work[1], filename, 0, g_propagator_precision) ) != 0 ) {
                fprintf(stderr, "[hlbl_mII_invert_contract] Error from write_propagator for %s, status was %d   %s %d\n", filename, exitstatus, __FILE__, __LINE__);
                EXIT(2);
              }
            }
            
          }  /* end of loop on spin-color components */
        }  /* end of loop on flavor for fwd_y */

        /***********************************************************/
        /***********************************************************/


        for ( int iflavor = 0; iflavor <= 1; iflavor++ ) 
        {

          /***********************************************************
           * D_y^+ z g5 gsigma U_src
           ***********************************************************/
          double *** dzu = init_3level_dtable ( 6, 12, 24 );
          double *** dzsu = init_3level_dtable ( 4, 12, 24 );
          if ( dzu == NULL || dzsu == NULL )
          {
            fprintf(stderr, "[hlbl_mII_invert_contract] Error from init_Xlevel_dtable  %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }

#if _WITH_TIMER
          gettimeofday ( &ta, (struct timezone *)NULL );
#endif
          for(int ia = 0; ia < 12; ia++ )
          {

            for ( int k = 0; k < 6; k++ )
            {
              int const sigma = idx_comb[k][1];
              int const rho   = idx_comb[k][0];

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
              for ( unsigned int iz = 0; iz < VOLUME; iz++ ) 
              {
                double * const _u = g_fwd_src[iflavor][sigma][ia] + _GSI(iz);
                double * const _s = spinor_work[0] + _GSI(iz);

                int const z[4] = {
                    ( g_lexic2coords[iz][0] + g_proc_coords[0] * T  - gsx[0] + T_global  ) % T_global,
                    ( g_lexic2coords[iz][1] + g_proc_coords[1] * LX - gsx[1] + LX_global ) % LX_global,
                    ( g_lexic2coords[iz][2] + g_proc_coords[2] * LY - gsx[2] + LY_global ) % LY_global,
                    ( g_lexic2coords[iz][3] + g_proc_coords[3] * LZ - gsx[3] + LZ_global ) % LZ_global };

                int zv[4];
                site_map ( zv, z );

                _fv_eq_fv_ti_re ( _s, _u,  zv[rho  ] );
              }
            
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
              for ( unsigned int iz = 0; iz < VOLUME; iz++ ) 
              {
                double * const _u = g_fwd_src[iflavor][rho  ][ia] + _GSI(iz);
                double * const _s = spinor_work[0] + _GSI(iz);

                int const z[4] = {
                    ( g_lexic2coords[iz][0] + g_proc_coords[0] * T  - gsx[0] + T_global  ) % T_global,
                    ( g_lexic2coords[iz][1] + g_proc_coords[1] * LX - gsx[1] + LX_global ) % LX_global,
                    ( g_lexic2coords[iz][2] + g_proc_coords[2] * LY - gsx[2] + LY_global ) % LY_global,
                    ( g_lexic2coords[iz][3] + g_proc_coords[3] * LZ - gsx[3] + LZ_global ) % LZ_global };

                int zv[4];
                site_map ( zv, z );

                _fv_eq_fv_pl_fv_ti_re ( _s, _s , _u, -zv[sigma] );
              }

              for(int ib = 0; ib < 12; ib++ )
              {
                complex w = {0.,0.};
                spinor_scalar_product_co ( &w, fwd_y[1-iflavor][ib], spinor_work[0], VOLUME );

                dzu[k][ia][2*ib  ] = w.re;
                dzu[k][ia][2*ib+1] = w.im;

              }  /* of ib */
            }  /* of index combinations */

            for ( int sigma = 0; sigma < 4; sigma++ )
            {
              complex w = { 0., 0. };

              for(int ib = 0; ib < 12; ib++ )
              {
                spinor_scalar_product_co ( &w, fwd_y[1-iflavor][ib], g_fwd_src[iflavor][sigma][ia], VOLUME );
                dzsu[sigma][ia][2*ib  ] = w.re;
                dzsu[sigma][ia][2*ib+1] = w.im;
              }
            }

          }  /* of ia */

#if _WITH_TIMER
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "hlbl_mII_invert_contract", "dzu-dzsu", io_proc == 2 );
#endif


          double **** g_dzu  = init_4level_dtable ( 6, 4, 12, 24 );
          double **** g_dzsu = init_4level_dtable ( 4, 4, 12, 24 );
          if ( g_dzu == NULL || g_dzsu == NULL )
          {
            fprintf(stderr, "[hlbl_mII_invert_contract] Error from init_Xlevel_dtable  %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }

#if _WITH_TIMER
          gettimeofday ( &ta, (struct timezone *)NULL );
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
          for ( int k = 0; k < 6; k++ )
          {
            double spinor1[24];
            for(int ia = 0; ia < 12; ia++ )
            {
              _fv_eq_gamma_ti_fv ( spinor1, 5, dzu[k][ia] );

              for ( int mu = 0; mu < 4; mu++ )
              {
                _fv_eq_gamma_ti_fv ( g_dzu[k][mu][ia], mu, spinor1 );
              }
            }
          }

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
          for ( int k = 0; k < 4; k++ )
          {
            double spinor1[24];
            for(int ia = 0; ia < 12; ia++ )
            {
              _fv_eq_gamma_ti_fv ( spinor1, 5, dzsu[k][ia] );

              for ( int mu = 0; mu < 4; mu++ )
              {
                _fv_eq_gamma_ti_fv ( g_dzsu[k][mu][ia], mu, spinor1 );
              }
            }
          }

#if _WITH_TIMER
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "hlbl_mII_invert_contract", "g_dzu-g_dzsu", io_proc == 2 );
#endif
          /***********************************************************/
          /***********************************************************/

          /***********************************************************
           * contractions for term I and II
           ***********************************************************/
 
#if _WITH_TIMER
          gettimeofday ( &ta, (struct timezone *)NULL );
#endif

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
          unsigned int rank;          
          double **** corr_I  = init_4level_dtable ( 6, 4, 4, 8 );
          double **** corr_II = init_4level_dtable ( 6, 4, 4, 8 );
          double ***  dxu     = init_3level_dtable ( 4, 12, 24 );
          double **** g_dxu   = init_4level_dtable ( 4, 4, 12, 24 );

          if ( corr_I == NULL || corr_II == NULL || dxu == NULL || g_dxu == NULL )
          {
            fprintf(stderr, "[hlbl_mII_invert_contract] Error from init_Xlevel_dtable  %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }

          double spinor1[24];

          double kerv1[6][4][4][4] KQED_ALIGN ;
          double kerv2[6][4][4][4] KQED_ALIGN ;
          double kerv3[6][4][4][4] KQED_ALIGN ;

          double kernel_sum_thread[4] = { 0., 0., 0., 0. };

          QED_kernel_LX_ptr KQED_LX[4] = {
            QED_kernel_L0,
            QED_kernel_L1,
            QED_kernel_L2,
            QED_kernel_L3 };

          /***********************************************************
           ***********************************************************
           **
           ** loop on volume
           **
           ***********************************************************
           ***********************************************************/
#ifdef HAVE_OPENMP
#pragma omp for
#endif
          for ( unsigned int ix = 0; ix < VOLUME; ix++ )
          {
            int x[4] = { g_proc_coords[0]*T  + g_lexic2coords[ix][0],
                         g_proc_coords[1]*LX + g_lexic2coords[ix][1],
                         g_proc_coords[2]*LY + g_lexic2coords[ix][2],
                         g_proc_coords[3]*LZ + g_lexic2coords[ix][3] };
            
            rank = ( ( x[0] * LX_global + x[1] ) * LY_global + x[2] ) * LZ_global + x[3];

            x[0] = ( x[0] - gsx[0] + T_global  ) % T_global;
            x[1] = ( x[1] - gsx[1] + LX_global ) % LX_global;
            x[2] = ( x[2] - gsx[2] + LY_global ) % LY_global;
            x[3] = ( x[3] - gsx[3] + LZ_global ) % LZ_global;

            int xv[4];
            site_map ( xv, x );

            for ( int ib = 0; ib < 12; ib++) 
            {
              double * const _u = fwd_y[iflavor][ib] + _GSI(ix);

              for ( int mu = 0; mu < 4; mu++ )
              {
            
                for ( int ia = 0; ia < 12; ia++) 
                {
                
                  double * const _d = g_fwd_src[1-iflavor][mu][ia] + _GSI(ix);
                  complex w;

                  _co_eq_fv_dag_ti_fv ( &w ,_d, _u );

                  /* -1 factor due to (g5 gmu)^+ = -g5 gmu */
                  dxu[mu][ib][2*ia  ] = -w.re;
                  dxu[mu][ib][2*ia+1] = -w.im;
                }
              } /* end of loop on gamma_mu */
            }


            for ( int mu = 0; mu < 4; mu++ )
            {
              for ( int ib = 0; ib < 12; ib++)
              {
                _fv_eq_gamma_ti_fv ( spinor1, 5, dxu[mu][ib] );
                for ( int lambda = 0; lambda < 4; lambda++ )
                {
                  _fv_eq_gamma_ti_fv ( g_dxu[lambda][mu][ib], lambda, spinor1 );
                }
              }
            }

            /***********************************************************
             * combine g_dxu and g_dzu
             ***********************************************************/
            for ( int mu = 0; mu < 4; mu++ )
            {
              for ( int nu = 0; nu < 4; nu++ ) 
              {
                for ( int lambda = 0; lambda < 4; lambda++ )
                {
                  for( int k = 0; k < 6; k++ )
                  {
 
                    double dtmp[2] = {0., 0.};
                    for ( int ia = 0; ia < 12; ia++)
                    {
                      for ( int ib = 0; ib < 12; ib++)
                      {

                        double u[2] = { g_dxu[lambda][mu][ia][2*ib], g_dxu[lambda][mu][ia][2*ib+1] };

                        double v[2] = { g_dzu[k][nu][ib][2*ia], g_dzu[k][nu][ib][2*ia+1] };

                        dtmp[0] += u[0] * v[0] - u[1] * v[1];
                        dtmp[1] += u[0] * v[1] + u[1] * v[0];
                      }
                    }
                    corr_I[k][mu][nu][2*lambda  ] = -dtmp[0];
                    corr_I[k][mu][nu][2*lambda+1] = -dtmp[1];
                  }
                }
              }
            }

            /***********************************************************/
            /***********************************************************/

            /***********************************************************
             * combine g_dxu and g_dzsu
             ***********************************************************/
            for ( int mu = 0; mu < 4; mu++ )
            {
              for ( int nu = 0; nu < 4; nu++ ) 
              {
                for ( int lambda = 0; lambda < 4; lambda++ )
                {
                  for( int k = 0; k < 6; k++ )
                  {
                    int const sigma = idx_comb[k][1];
                    int const rho   = idx_comb[k][0];

                    double dtmp[2] = {0., 0.};
                    for ( int ia = 0; ia < 12; ia++)
                    {
                      for ( int ib = 0; ib < 12; ib++)
                      {

                        double u[2] = { g_dxu[lambda][mu][ia][2*ib], g_dxu[lambda][mu][ia][2*ib+1] };

                        double v[2] = { xv[rho] * g_dzsu[sigma][nu][ib][2*ia  ] - xv[sigma] * g_dzsu[rho][nu][ib][2*ia  ],
                                        xv[rho] * g_dzsu[sigma][nu][ib][2*ia+1] - xv[sigma] * g_dzsu[rho][nu][ib][2*ia+1] };

                        dtmp[0] += u[0] * v[0] - u[1] * v[1];
                        dtmp[1] += u[0] * v[1] + u[1] * v[0];
                      }
                    }
                    corr_II[k][mu][nu][2*lambda  ] = -dtmp[0];
                    corr_II[k][mu][nu][2*lambda+1] = -dtmp[1];
                  }
                }
              }
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
#if 0
#pragma omp critical
{
            for ( int mu = 0; mu < 4; mu++ )
            {
              for ( int nu = 0; nu < 4; nu++ )
              {
                for ( int lambda = 0; lambda < 4; lambda++ )
                {
                  for( int k = 0; k < 6; k++ )
                  {
                    fprintf ( stdout, "xv %3d %3d %3d %3d yv %3d %3d %3d %3d  fl %d L %d  idx %d %d %d %d %d K %16.7e %16.7e %16.7e   P1 %25.16e   P2 %25.16e \n", 
                        xv[0], xv[1], xv[2], xv[3],
                        yv[0], yv[1], yv[2], yv[3],
                        iflavor,
                        ikernel,
                        mu, nu, lambda, idx_comb[k][0], idx_comb[k][1],
                        kerv1[k][mu][nu][lambda],
                        kerv2[k][mu][nu][lambda],
                        kerv3[k][mu][nu][lambda],
                        corr_I[k][mu][nu][2*lambda],
                        corr_II[k][mu][nu][2*lambda] );
                  }
                }
              }
            }
}  /* end of critical region */
#endif
            }

#if 0
            QED_kernel_L1( xm, ym, kqed_t, kerv );
            dtmp = 0.;
            for( int i = 0 ; i < 384 ; i++ )
            {
              dtmp +=  _kerv[i] * _corr_I[2*i];
            }
            kernel_sum_thread[1] += dtmp;

            QED_kernel_L2( xm, ym, kqed_t, kerv );
            dtmp = 0.;
            for( int i = 0 ; i < 384 ; i++ )
            {
              dtmp +=  _kerv[i] * _corr_I[2*i];
            }
            kernel_sum_thread[2] += dtmp;

            QED_kernel_L3( xm, ym, kqed_t, kerv );
            dtmp = 0.;
            for( int i = 0 ; i < 384 ; i++ )
            {
              dtmp +=  _kerv[i] * _corr_I[2*i];
            }
            kernel_sum_thread[3] += dtmp;
#endif

#if 0
#pragma omp critical
{
            for ( int mu = 0; mu < 4; mu++ )
            {
              for ( int nu = 0; nu < 4; nu++ )
              {
                for ( int lambda = 0; lambda < 4; lambda++ )
                {
                  for( int k = 0; k < 6; k++ )
                  {
                    fprintf ( stdout, "y %2d s %d %d %d %d fl %d  x %6u  i %3d %3d %3d %3d %3d   %25.16e %25.16e \n", 
                        iy,
                        ysign_comb[isign][0],
                        ysign_comb[isign][1],
                        ysign_comb[isign][2],
                        ysign_comb[isign][3],
                        iflavor,
                        rank,
                        mu, nu, lambda, idx_comb[k][0], idx_comb[k][1],
                        corr_I[k][mu][nu][2*lambda  ], corr_I[k][mu][nu][2*lambda+1] );
                  }
                }
              }
            }
}  /* end of critical region */
#endif
          }  /* end of loop on ix */

          /***********************************************************
           * summation with QED kernel
           ***********************************************************/
#ifdef HAVE_OPENMP
#pragma omp critical
{
#endif
          kernel_sum[iy][0] += kernel_sum_thread[0];
          kernel_sum[iy][1] += kernel_sum_thread[1];
          kernel_sum[iy][2] += kernel_sum_thread[2];
          kernel_sum[iy][3] += kernel_sum_thread[3];

#ifdef HAVE_OPENMP
   /***********************************************************/
}  /* end of critical region */
   /***********************************************************/
#endif


          fini_4level_dtable ( &corr_I  );
          fini_4level_dtable ( &corr_II );
          fini_4level_dtable ( &g_dxu   );
          fini_3level_dtable ( &dxu     );

#ifdef HAVE_OPENMP
   /***********************************************************/
}  /* end of parallel region */
   /***********************************************************/
#endif

#if _WITH_TIMER
          gettimeofday ( &tb, (struct timezone *)NULL );
          show_time ( &ta, &tb, "hlbl_mII_invert_contract", "kernel-sum", io_proc == 2 );
#endif

          /***********************************************************
           * end of contractions for term I and II
           ***********************************************************/

          /***********************************************************/
          /***********************************************************/

          fini_3level_dtable ( &dzu    );
          fini_3level_dtable ( &dzsu   );
          fini_4level_dtable ( &g_dzu  );
          fini_4level_dtable ( &g_dzsu );

        }  /* end of loop on flavor */


      }  /* end of loop on signs */

    }  /* end of loop on |y| */



#ifdef HAVE_MPI
    /***********************************************************
     * sum over MPI processes
     ***********************************************************/
    double * mbuffer = init_1level_dtable ( 4 * ( ymax + 1 ) );

    memcpy ( mbuffer, kernel_sum[0], 4 * ( ymax + 1 ) * sizeof ( double ) );

    if ( MPI_Reduce ( mbuffer, kernel_sum[0], 4 * ( ymax + 1 ), MPI_DOUBLE, MPI_SUM, 0, g_cart_grid ) != MPI_SUCCESS )
    {
      fprintf (stderr, "[hlbl_mII_invert_contract] Error from MP_Reduce  %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }

    fini_1level_dtable ( &mbuffer );

#endif  /* end of ifdef HAVE_MPI */


    if ( io_proc == 2 )
    {
      int ncdim = 2;
      int cdim[2] = { ymax+1, 4 };
      char key[100];
      sprintf (key, "t%dx%dy%dz%d", gsx[0], gsx[1], gsx[2], gsx[3] );

      exitstatus = write_h5_contraction ( kernel_sum[0], NULL, output_filename, key, "double", ncdim, cdim );
      if ( exitstatus != 0 )
      {
        fprintf (stderr, "[hlbl_mII_invert_contract] Error from MP_Reduce  %s %d\n", __FILE__, __LINE__ );
        EXIT(12);
      }
    }

    fini_2level_dtable ( &kernel_sum );

  }  /* end of loop on source locations */

  /***********************************************************
   * free the allocated memory, finalize
   ***********************************************************/
          
  free_QED_temps( &kqed_t  );

  fini_3level_dtable ( &fwd_src );
  fini_3level_dtable ( &fwd_y );
  fini_4level_dtable ( &g_fwd_src );
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
  show_time ( &start_time, &end_time, "hlbl_mII_invert_contract", "runtime", g_cart_id == 0 );

  return(0);
}
