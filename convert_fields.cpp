/****************************************************
 * convert_fields
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


static inline void convert_ukqcd_to_cvc_at_source ( double ** spinor_field ) {

    /***********************************************************
     * ukqcd -> cvc gamma basis rotation at source;
     * assumes specific forms of gamma_t and gamma_5
     * requires all 12 spin-color components
     ***********************************************************/
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
}  /* end of convert_ukqcd_to_cvc_at_source */


void usage(void) {
  fprintf(stdout, "oldascii2binary -- usage:\n");
  exit(0);
}

/***********************************************************
 *
 * MAIN PROGRAM
 *
 ***********************************************************/
int main(int argc, char **argv) {
  
  int c, exitstatus;
  int filename_set = 0;
  char filename[200];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_bc = NULL;
  char field_type[12] = "NA";
  int rotate_source_side = 0;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "sh?vf:t:")) != -1) {
    switch (c) {
    case 'v':
      g_verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 't':
      strcpy ( field_type , optarg );
      fprintf ( stdout, "# [convert_fields] field_type set to %s\n", field_type );
      break;
    case 's':
      rotate_source_side = 1;
      fprintf ( stdout, "# [convert_fields] rotate_source_side set to %d\n", rotate_source_side);
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
  if(g_cart_id==0) fprintf(stdout, "# [convert_fields] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /***********************************************************
   * initialize MPI parameters for cvc
   ***********************************************************/
  mpi_init(argc, argv);

  if(init_geometry() != 0) {
    fprintf(stderr, "[convert_fields] Error from init_geometry\n");
    exit(101);
  }

  geometry();

  unsigned int const VOL3     = LX * LY * LZ;
  unsigned int const VOL3half = VOL3 / 2;
  size_t const sizeof_spinor_field = _GSI(VOLUME) * sizeof( double );
  size_t const sizeof_eo_spinor_field = _GSI(VOLUME/2) * sizeof( double );
  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();


  /***********************************************************
   * read gauge field
   ***********************************************************/

  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);

  if( strcmp(gaugefilename_prefix,"identity") == 0 ) {
    exitstatus = unit_gauge_field ( g_gauge_field,  VOLUME );
  } else {
    /* read the gauge field */
    sprintf(filename, "%s.%.4d", gaugefilename_prefix, Nconf);
    if(g_cart_id==0) fprintf(stdout, "# [convert_fields] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  }

  if(exitstatus != 0) {
    fprintf(stderr, "[convert_fields] Error from setting g_gauge_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
    fprintf(stderr, "[convert_fields] Error from gauge_field with boundary condition, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************************
   * measure the plaquette
   ***********************************************************/
  exitstatus = plaquetteria ( gauge_field_with_bc );
  if(exitstatus != 0) {
    fprintf(stderr, "[convert_fields] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_bc );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[convert_fields] Error from init_clover, status was %d\n", exitstatus );
    EXIT(1);
  }

  /***********************************************
   * flavor tag
   ***********************************************/
  char flavor_tag = ( g_mu > 0 ) ? 'u' : 'd';

  /***********************************************
   * allocate fields
   *****a******************************************/
  double **  spinor_field = init_2level_dtable ( 12, _GSI(VOLUME) );
  if ( spinor_field == NULL ) {
    fprintf ( stderr, "# [convert_fields] Error from init_Xlevel_ftable %s %d\n", __FILE__, __LINE__ );
    EXIT(9);
  }

  double ** eo_spinor_work = init_2level_dtable ( 6, _GSI(VOLUME+RAND)/2 );
  if ( eo_spinor_work == NULL ) {
    fprintf ( stderr, "# [convert_fields] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(9);
  }

  if ( strcmp ( field_type , "fp" ) == 0 ) {
    sprintf ( filename, "%s.binary", filename_prefix );
    if(g_cart_id==0) fprintf(stdout, "# [convert_fields] Reading prop. from file %s\n", filename );


    float ** buffer = init_2level_ftable ( VOLUME, 288 );
    if ( buffer == NULL ) {
      fprintf ( stderr, "# [convert_fields] Error from init_Xlevel_ftable %s %d\n", __FILE__, __LINE__ );
      EXIT(9);
    }

    FILE * pfs = fopen( filename, "r" );
    fread( buffer[0], sizeof( float ), VOLUME*12*12*2, pfs );
    fclose ( pfs );

  /***********************************************************
   * reorder t,z,y,x to t,x,y,z
   ***********************************************************/
#pragma omp parallel for
    for( unsigned int it = 0; it < T; it++ ) {
    for( unsigned int ix = 0; ix < LX; ix++ ) {
    for( unsigned int iy = 0; iy < LY; iy++ ) {
    for( unsigned int iz = 0; iz < LZ; iz++ ) {
      double dtmp[288];

      unsigned int const iix = g_ipt[it][ix][iy][iz];
      unsigned int const iiy =  LX * ( LY * ( LZ * it + iz ) + iy ) + ix;

      byte_swap_assign_single2double ( dtmp , buffer[iiy], 288 );
  
      /***********************************************************
       * reorder spin-color at source and sink
       ***********************************************************/
      for ( int alpha = 0; alpha < 4; alpha++ ) {
      for ( int a = 0; a < 3; a++ ) {
        for ( int beta = 0; beta < 4; beta++ ) {
        for ( int b = 0; b < 3; b++ ) {
          spinor_field[3*beta + b][_GSI(iix) + 2 * ( 3*alpha + a )     ]  = buffer[iiy][ 2 * ( ( 4 *  (  3 * alpha + a ) + beta ) + b )     ];
          spinor_field[3*beta + b][_GSI(iix) + 2 * ( 3*alpha + a ) + 1 ]  = buffer[iiy][ 2 * ( ( 4 *  (  3 * alpha + a ) + beta ) + b ) + 1 ];
        }}
      }}
    }}}}

    fini_2level_ftable ( &buffer );

  }  /* end of field type fp */

  int const nsc = ( strcmp ( field_type, "ds" ) == 0 ) ? 1 : 12;

  /***********************************************************
   * loop on fields
   ***********************************************************/
  for ( int i = 0; i < nsc; i++ )
  {

    /***********************************************************
     * read read the spinor fields
     ***********************************************************/
    if ( strncmp( field_type , "ds" , 2) == 0 ) {

      if ( strcmp( field_type , "ds12" ) == 0 ) {
        sprintf ( filename, "%s_s%d_c%d", filename_prefix, i/3, i%3 ); 
      } else {
        sprintf ( filename, "%s", filename_prefix );
      }

      if(g_cart_id==0) fprintf(stdout, "# [convert_fields] Reading field from file %s %s %d\n", filename, __FILE__, __LINE__ );
      exitstatus = read_lime_spinor ( spinor_field[i], filename, 0 ); 
      if( exitstatus != 0 ) {
        fprintf(stderr, "[convert_fields] Error from read_lime_spinor for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
        EXIT(9);
      }
    }  /* end of if field_type has ds */

    /***********************************************************
     * ukqcd -> cvc gamma basis rotation at sink;
     ***********************************************************/
#pragma omp parallel for
    for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
      _fv_cvc_eq_convert_fv_ukqcd ( spinor_field[i]+_GSI(ix) , spinor_field[i]+_GSI(ix) );
    }
  
  }

  if ( ( strcmp( field_type, "ds12" ) == 0 ) && rotate_source_side ) {
    convert_ukqcd_to_cvc_at_source ( spinor_field );
  }

  for ( int i = 0; i < nsc; i++ ) {
    if ( g_write_propagator ) {
      if ( g_source_type == 0 ) {
        sprintf ( filename, "%s.%c.%.4d.t%dx%dy%dz%d.%.2d.inverted", filename_prefix2, flavor_tag, Nconf,
            g_source_coords_list[0][0], g_source_coords_list[0][1], g_source_coords_list[0][2], g_source_coords_list[0][3], i );
      } else if ( g_source_type == 1 ) {
        sprintf ( filename, "prop.%c.%.4d.%.5d", flavor_tag, Nconf, i );
      }
      exitstatus = write_propagator ( spinor_field[i],  filename, 0, g_propagator_precision );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[convert_fields] Error from write_propagator for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
        EXIT(9);
      }
    }

    /***********************************************************
     * norm of input field
     ***********************************************************/
    double norm = 0.;
    spinor_scalar_product_re ( &norm,      spinor_field[i], spinor_field[i], VOLUME );
    fprintf(stdout, "# [convert_fields] norm propagator %2d  %e\n", i, sqrt(norm));

    /***********************************************************
     * apply D
     ***********************************************************/
    spinor_field_lexic2eo ( spinor_field[i], eo_spinor_work[0], eo_spinor_work[1] );

    Q_clover_phi_matrix_eo ( eo_spinor_work[2], eo_spinor_work[3], eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_bc, eo_spinor_work[4], mzz[1] );

    spinor_field_eo2lexic ( spinor_field[i], eo_spinor_work[2], eo_spinor_work[3] );

    norm = 0.;
    spinor_scalar_product_re ( &norm,      spinor_field[i], spinor_field[i], VOLUME );
    fprintf(stdout, "# [convert_fields] norm source     %2d  %e\n", i, sqrt(norm));

    if ( g_write_source ) {

      if ( g_source_type == 0 ) {
        sprintf ( filename, "%s.%c.%.4d.t%dx%dy%dz%d.%.2d.ascii", filename_prefix2, flavor_tag, Nconf, 
            g_source_coords_list[0][0], g_source_coords_list[0][1], g_source_coords_list[0][2], g_source_coords_list[0][3], i );
      } else if ( g_source_type == 1 ) {
        sprintf ( filename, "%s.%c.%.4d.%.5d.ascii", filename_prefix, flavor_tag, Nconf, i );
      }

      FILE * ffs = fopen( filename, "w" );
      exitstatus = printf_spinor_field( spinor_field[i], 0, ffs);
      if( exitstatus != 0 ) {
        fprintf(stderr, "[convert_fields] Error from printf_spinor_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(9);
      }
      fclose ( ffs );


      if ( g_source_type == 1 ) {
        sprintf ( filename, "%s.%c.%.4d.%.5d", filename_prefix2, flavor_tag, Nconf, i );
        exitstatus = write_propagator ( spinor_field[i],  filename, 0, g_propagator_precision );
        if( exitstatus != 0 ) {
          fprintf(stderr, "[convert_fields] Error from write_propagator for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
          EXIT(9);
        }


#pragma omp parallel for
        for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
          _fv_cvc_eq_convert_fv_ukqcd ( spinor_field[i]+_GSI(ix) , spinor_field[i]+_GSI(ix) );
        }
        sprintf ( filename, "%s.%c.%.4d.%.5d.orig.ascii", filename_prefix2, flavor_tag, Nconf, i );
        ffs = fopen( filename, "w" );
        exitstatus = printf_spinor_field( spinor_field[i], 0, ffs);
        if( exitstatus != 0 ) {
          fprintf(stderr, "[convert_fields] Error from printf_spinor_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(9);
        }
        fclose ( ffs );
        sprintf ( filename, "%s.%c.%.4d.%.5d.orig", filename_prefix2, flavor_tag, Nconf, i );
        exitstatus = write_propagator ( spinor_field[i],  filename, 0, g_propagator_precision );
        if( exitstatus != 0 ) {
          fprintf(stderr, "[convert_fields] Error from write_propagator for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
          EXIT(9);
        }

#if 0
#endif
      }
    }

    if ( g_source_type == 0 ) {
      /***********************************************************
       * check point source
       ***********************************************************/
      int sx[4], source_proc_id;
      exitstatus = get_point_source_info ( g_source_coords_list[0], sx, &source_proc_id);
      if( exitstatus != 0 ) {
        fprintf(stderr, "[convert_fields] Error from get_point_source_info, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(9);
      }

      if ( source_proc_id == g_cart_id ) {
        spinor_field[i][_GSI( g_ipt[sx[0]][sx[1]][sx[2]][sx[3]]) + 2*i] -= 1.;
      }

      double norm_diff = 0.;
      spinor_scalar_product_re ( &norm_diff, spinor_field[i], spinor_field[i], VOLUME);
      fprintf(stdout, "# [convert_fields] norm-diff %2d %e\n", i, sqrt(norm_diff));
    }

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
    fprintf(stdout, "# [convert_fields] %s# [convert_fields] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [convert_fields] %s# [convert_fields] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }
  return(0);
}

