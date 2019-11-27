/****************************************************
 * apply_Dtm
 *
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#  ifdef HAVE_TMLQCD_LIBWRAPPER
#    include "tmLQCD.h"
#  endif

#ifdef __cplusplus
}
#endif


#define MAIN_PROGRAM

#include "types.h"
#include "cvc_complex.h"
#include "ilinalg.h"
#include "table_init_d.h"
#include "table_init_f.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "read_input_parser.h"
#include "scalar_products.h"
#include "clover.h"

using namespace cvc;


static inline void _fv_cvc_eq_convert_fv_ukqcd ( double * const r , double * const s ) {
  double const _sqrt2inv = 0.7071067811865475;
  double _spinor1[24], _spinor2[24];
  _fv_eq_gamma_ti_fv ( _spinor1, 5, s );
  _fv_eq_gamma_ti_fv ( _spinor2, 0, s );
  _fv_eq_fv_pl_fv ( r, _spinor1, _spinor2 );
  _fv_ti_eq_re ( r, _sqrt2inv );
}  /* end of _fv_cvc_eq_convert_fv_ukqcd */


void usage(void) {
  fprintf(stdout, "oldascii2binary -- usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, exitstatus;
  int filename_set = 0;
  char filename[200];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_bc = NULL;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?vf:")) != -1) {
    switch (c) {
    case 'v':
      g_verbose = 1;
      break;
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

  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  if(g_cart_id==0) fprintf(stdout, "# [apply_Dtm] Reading input from file %s\n", filename);
  read_input_parser(filename);


#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [apply_Dtm] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1);
  if(exitstatus != 0) {
    EXIT(14);
  }
  exitstatus = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
  if(exitstatus != 0) {
    EXIT(15);
  }
  exitstatus = tmLQCD_get_lat_params(&g_tmLQCD_lat);
  if(exitstatus != 0) {
    EXIT(16);
  }
#endif

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  mpi_init(argc, argv);

  if(init_geometry() != 0) {
    fprintf(stderr, "Error from init_geometry\n");
    exit(101);
  }

  geometry();

  unsigned int const VOL3     = LX * LY * LZ;
  unsigned int const VOL3half = VOL3 / 2;
  size_t const sizeof_spinor_field = _GSI(VOLUME) * sizeof( double );
  size_t const sizeof_eo_spinor_field = _GSI(VOLUME/2) * sizeof( double );
  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();


#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);

  if( strcmp(gaugefilename_prefix,"identity") == 0 ) {
    exitstatus = unit_gauge_field ( g_gauge_field,  VOLUME );

  } else {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [apply_Dtm] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  }
#else
   Nconf = g_tmLQCD_lat.nstore;
   if(g_cart_id== 0) fprintf(stdout, "[apply_Dtm] Nconf = %d\n", Nconf);

   exitstatus = tmLQCD_read_gauge(Nconf);
   if(exitstatus != 0) {
     EXIT(3);
   }

   exitstatus = tmLQCD_get_gauge_field_pointer(&g_gauge_field);
   if(exitstatus != 0) {
     EXIT(4);
   }
   if(&g_gauge_field == NULL) {
     fprintf(stderr, "[apply_Dtm] Error, &g_gauge_field is NULL\n");
     EXIT(5);
   }
#endif

  if(exitstatus != 0) {
    fprintf(stderr, "[apply_Dtm] Error from setting g_gauge_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }



#ifdef HAVE_MPI
   xchange_gauge();
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  if ( g_propagator_bc_type == 0 ) {
    exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_bc, g_gauge_field, co_phase_up );
  } else if ( g_propagator_bc_type == 1 ) {
    exitstatus = gauge_field_eq_gauge_field_ti_bcfactor ( &gauge_field_with_bc, g_gauge_field, -1. );
  }
  if(exitstatus != 0) {
    fprintf(stderr, "[apply_Dtm] Error from gauge_field with boundary condition, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************************
   * measure the plaquette
   ***********************************************************/
  exitstatus = plaquetteria ( gauge_field_with_bc );
  if(exitstatus != 0) {
    fprintf(stderr, "[apply_Dtm] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_bc );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[apply_Dtm] Error from init_clover, status was %d\n", exitstatus );
    EXIT(1);
  }


  /***********************************************
   * allocate fields
   ***********************************************/
#if 0
  float *** propagator_field = init_3level_ftable ( VOLUME, 144, 2 );
  if ( propagator_field == NULL ) {
    fprintf ( stderr, "# [apply_Dtm] Error from init_Xlevel_ftable %s %d\n", __FILE__, __LINE__ );
    EXIT(9);
  }
#endif

  double ** spinor_field = init_2level_dtable ( 12, _GSI(VOLUME) );
  if ( spinor_field == NULL ) {
    fprintf ( stderr, "# [apply_Dtm] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(9);
  }

  double ** eo_spinor_work = init_2level_dtable ( 6, _GSI(VOLUME+RAND)/2 );
  if ( eo_spinor_work == NULL ) {
    fprintf ( stderr, "# [apply_Dtm] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(9);
  }


#if 0
  sprintf ( filename, "%s.binary", filename_prefix );
  if(g_cart_id==0) fprintf(stdout, "# [apply_Dtm] Reading prop. from file %s\n", filename );


  float *** buffer = init_3level_ftable ( VOLUME, 144, 2 );
  if ( propagator_field == NULL ) {
    fprintf ( stderr, "# [apply_Dtm] Error from init_Xlevel_ftable %s %d\n", __FILE__, __LINE__ );
    EXIT(9);
  }

  FILE * pfs = fopen( filename, "r" );
  fread( buffer[0][0], sizeof( float ), VOLUME*12*12*2, pfs );
  fclose ( pfs );

#if 0
  sprintf ( filename, "%s.ascii", filename_prefix );
  pfs = fopen( filename, "w" );

  for( unsigned int ix = 0; ix < VOLUME; ix++ ) {
    for ( int i = 0; i< 144; i++ ) {
      float ftmp[2];
      
      byte_swap_assign_singleprec ( ftmp, propagator_field[ix][i], 2 );
      propagator_field[ix][i][0] = ftmp[0];
      propagator_field[ix][i][1] = ftmp[1];

      fprintf ( pfs, "%16.7e %16.7e\n", 
          propagator_field[ix][i][0], propagator_field[ix][i][1]
          /* propagator_field[0][0][0][2*(144*ix+12*i+k)], propagator_field[0][0][0][2*(144*ix+12*i+k)+1] */
          );
    }
  }
  fclose ( pfs );
#endif

#pragma omp parallel for
  for( unsigned int it = 0; it < T; it++ ) {
  for( unsigned int ix = 0; ix < LX; ix++ ) {
  for( unsigned int iy = 0; iy < LY; iy++ ) {
  for( unsigned int iz = 0; iz < LZ; iz++ ) {

    unsigned int const iix = g_ipt[it][ix][iy][iz];
    unsigned int const iiy =  LX * ( LY * ( LZ * it + iz ) + iy ) + ix;

    for ( int i = 0; i< 144; i++ ) {
      float ftmp[2];
      
      byte_swap_assign_singleprec ( ftmp, buffer[iiy][i], 2 );
      propagator_field[iix][i][0] = ftmp[0];
      propagator_field[iix][i][1] = ftmp[1];
    }
  }}}}

  fini_3level_ftable ( &buffer );
#endif

  for ( int i = 0; i < 12; i++ )
  {
    /****************************************
     * read read the spinor fields
     ****************************************/

    sprintf ( filename, "%s_s%d_c%d", filename_prefix, i/3, i%3 );
    if(g_cart_id==0) fprintf(stdout, "# [apply_Dtm] Reading prop. from file %s\n", filename_prefix);
    exitstatus = read_lime_spinor ( spinor_field[i], filename, 0 ); 
    if( exitstatus != 0 ) {
      fprintf(stderr, "[apply_Dtm] Error from read_lime_spinor for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
      EXIT(9);
    }
#if 0
#endif

#pragma omp parallel for
    for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
#if 0
      double spinor1[24];
      for ( int isc = 0; isc < 12; isc++ ) {
        /* int const idx = 3 * ( 3 * ( 4*(isc/3) + i/3 ) * isc%3 ) + i%3; */
        int const idx = 3 * ( 3 * ( 4*(i/3) + isc/3 ) * i%3 ) + isc%3;

        spinor1[2*isc  ] = (double)( propagator_field[ix][idx][0] );
        spinor1[2*isc+1] = (double)( propagator_field[ix][idx][1] ); 
      }
      _fv_cvc_eq_convert_fv_ukqcd ( spinor_field[i]+_GSI(ix) , spinor1 );
#endif
      _fv_cvc_eq_convert_fv_ukqcd ( spinor_field[i]+_GSI(ix) , spinor_field[i]+_GSI(ix) );
    }
  }

#pragma omp parallel for
  for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
    double spinor1[4][24];
    double const norm = 1. / sqrt( 2. );
    for ( int ic =0; ic < 3; ic++ ) {
      _fv_eq_fv ( spinor1[0], spinor_field[  ic] + _GSI(ix) );
      _fv_eq_fv ( spinor1[1], spinor_field[3+ic] + _GSI(ix) );
      _fv_eq_fv ( spinor1[2], spinor_field[6+ic] + _GSI(ix) );
      _fv_eq_fv ( spinor1[3], spinor_field[9+ic] + _GSI(ix) );

      /*  */
      _fv_eq_fv_mi_fv (spinor_field[  ic]+_GSI(ix), spinor1[0], spinor1[2] );

      /*  */
      _fv_eq_fv_mi_fv (spinor_field[3+ic]+_GSI(ix), spinor1[1], spinor1[3] );

      /*  */
      _fv_eq_fv    ( spinor_field[6+ic]+_GSI(ix), spinor1[0] );
      _fv_ti_eq_re ( spinor_field[6+ic]+_GSI(ix), -1. );
      _fv_mi_eq_fv ( spinor_field[6+ic]+_GSI(ix), spinor1[2] );

      /*  */
      _fv_eq_fv    ( spinor_field[9+ic]+_GSI(ix), spinor1[1] );
      _fv_ti_eq_re ( spinor_field[9+ic]+_GSI(ix), -1. );
      _fv_mi_eq_fv ( spinor_field[9+ic]+_GSI(ix), spinor1[3] );

      _fv_ti_eq_re ( spinor_field[  ic]+_GSI(ix), norm );
      _fv_ti_eq_re ( spinor_field[3+ic]+_GSI(ix), norm );
      _fv_ti_eq_re ( spinor_field[6+ic]+_GSI(ix), norm );
      _fv_ti_eq_re ( spinor_field[9+ic]+_GSI(ix), norm );
    }
  }

  for ( int i = 0; i < 12; i++ ) {
    if ( g_write_propagator ) {
      /* sprintf ( filename, "%s.%d.lime", filename_prefix , i ); */
      sprintf ( filename, "%s_s%d_c%d.converted", filename_prefix, i/3, i%3 );
      exitstatus = write_propagator ( spinor_field[i],  filename, 0, 64 );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[apply_Dtm] Error from read_lime_spinor for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
        EXIT(9);
      }
    }

    double norm = 0.;
    spinor_scalar_product_re ( &norm,      spinor_field[i], spinor_field[i], VOLUME );
    fprintf(stdout, "# [apply_Dtm] norm %d2  %e\n", i, sqrt(norm));

    spinor_field_lexic2eo ( spinor_field[i], eo_spinor_work[0], eo_spinor_work[1] );

    Q_clover_phi_matrix_eo ( eo_spinor_work[2], eo_spinor_work[3], eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_bc, eo_spinor_work[4], mzz[1] );

    spinor_field_eo2lexic ( spinor_field[i], eo_spinor_work[2], eo_spinor_work[3] );

#if 0
    sprintf ( filename, "spinor.%d.ascii", i );
    FILE * ffs = fopen( filename, "w" );
    exitstatus = printf_spinor_field( spinor_field[i], 0, ffs);
    fclose ( ffs );
#endif

    int sx[4], source_proc_id;
    exitstatus = get_point_source_info ( g_source_coords_list[0], sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[apply_Dtm] Error from get_point_source_info, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(9);
    }

    if ( source_proc_id == g_cart_id ) {
      spinor_field[i][_GSI( g_ipt[sx[0]][sx[1]][sx[2]][sx[3]]) + 2*i] -= 1.;
    }

    double norm_diff = 0.;
    spinor_scalar_product_re ( &norm_diff, spinor_field[i], spinor_field[i], VOLUME);
    fprintf(stdout, "# [apply_Dtm] norm-diff %2d %e\n", i, sqrt(norm_diff));


  }  /* end of loop on spin color i */

  /***********************************************
   * free the allocated memory, finalize 
   ***********************************************/
  free_geometry();

  if ( g_gauge_field       != NULL ) free ( g_gauge_field );
  if ( gauge_field_with_bc != NULL ) free ( gauge_field_with_bc );


  fini_clover();
#if 0
  fini_3level_ftable ( &propagator_field );
#endif
  fini_2level_dtable ( &eo_spinor_work );
  fini_2level_dtable ( &spinor_field );

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [apply_Dtm] %s# [apply_Dtm] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [apply_Dtm] %s# [apply_Dtm] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }
  return(0);
}

