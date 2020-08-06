/****************************************************
 * test_seq_prop
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
      fprintf ( stdout, "# [test_seq_prop] field_type set to %s\n", field_type );
      break;
    case 's':
      rotate_source_side = 1;
      fprintf ( stdout, "# [test_seq_prop] rotate_source_side set to %d\n", rotate_source_side);
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
  if(g_cart_id==0) fprintf(stdout, "# [test_seq_prop] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /***********************************************************
   * initialize MPI parameters for cvc
   ***********************************************************/
  mpi_init(argc, argv);

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_seq_prop] Error from init_geometry\n");
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
    if(g_cart_id==0) fprintf(stdout, "# [test_seq_prop] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  }

  if(exitstatus != 0) {
    fprintf(stderr, "[test_seq_prop] Error from setting g_gauge_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
    fprintf(stderr, "[test_seq_prop] Error from gauge_field with boundary condition, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************************
   * measure the plaquette
   ***********************************************************/
  exitstatus = plaquetteria ( gauge_field_with_bc );
  if(exitstatus != 0) {
    fprintf(stderr, "[test_seq_prop] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_bc );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_seq_prop] Error from init_clover, status was %d\n", exitstatus );
    EXIT(1);
  }

  /***********************************************
   * flavor tag
   ***********************************************/
  char flavor_tag = ( g_mu > 0 ) ? 'u' : 'd';

  /***********************************************
   * allocate fields
   ***********************************************/

  double **  fprop = init_2level_dtable ( 12, _GSI(VOLUME) );
  if ( fprop == NULL ) {
    fprintf ( stderr, "# [test_seq_prop] Error from init_Xlevel_ftable %s %d\n", __FILE__, __LINE__ );
    EXIT(9);
  }
  double **  sprop = init_2level_dtable ( 12, _GSI(VOLUME) );
  if ( sprop == NULL ) {
    fprintf ( stderr, "# [test_seq_prop] Error from init_Xlevel_ftable %s %d\n", __FILE__, __LINE__ );
    EXIT(9);
  }

  double ** eo_spinor_work = init_2level_dtable ( 6, _GSI(VOLUME+RAND)/2 );
  if ( eo_spinor_work == NULL ) {
    fprintf ( stderr, "# [test_seq_prop] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(9);
  }

  double * spinor_work[3] = {
    eo_spinor_work[0],
    eo_spinor_work[2],
    eo_spinor_work[4] };

  /* int const nsc = ( strcmp ( field_type, "ds" ) == 0 ) ? 1 : 12; */
  int const nsc = 12;

  /***********************************************************
   * loop on fields for forward propagator
   ***********************************************************/
  for ( int i = 0; i < nsc; i++ )
  {

    /***********************************************************
     * read read the spinor fields
     ***********************************************************/
    sprintf ( filename, "%s_s%d_c%d", filename_prefix, i/3, i%3 ); 

    if(g_cart_id==0) fprintf(stdout, "# [test_seq_prop] Reading field from file %s %s %d\n", filename, __FILE__, __LINE__ );
    exitstatus = read_lime_spinor ( fprop[i], filename, 0 ); 
    if( exitstatus != 0 ) {
      fprintf(stderr, "[test_seq_prop] Error from read_lime_spinor for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
      EXIT(9);
    }

    /***********************************************************
     * ukqcd -> cvc gamma basis rotation at sink;
     ***********************************************************/
#pragma omp parallel for
    for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
      _fv_cvc_eq_convert_fv_ukqcd ( fprop[i]+_GSI(ix) , fprop[i]+_GSI(ix) );
    }
  
  }  /* end of loop on nsc */

  if ( rotate_source_side ) {
    convert_ukqcd_to_cvc_at_source ( fprop );
  }

  for ( int i = 0; i < nsc; i++ ) {
    if ( g_write_propagator ) {
      if ( g_source_type == 0 ) {
        sprintf ( filename, "%s.%c.%.4d.t%dx%dy%dz%d.%.2d.inverted", filename_prefix3, flavor_tag, Nconf,
            g_source_coords_list[0][0], g_source_coords_list[0][1], g_source_coords_list[0][2], g_source_coords_list[0][3], i );
      }
      exitstatus = write_propagator ( fprop[i],  filename, 0, g_propagator_precision );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[test_seq_prop] Error from write_propagator for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
        EXIT(9);
      }
    }

    /***********************************************************
     * norm of input field
     ***********************************************************/
    double norm = 0.;
    spinor_scalar_product_re ( &norm, fprop[i], fprop[i], VOLUME );
    fprintf(stdout, "# [test_seq_prop] norm propagator %2d  %e\n", i, sqrt(norm));

    /***********************************************************
     * apply D
     ***********************************************************/
    spinor_field_lexic2eo ( fprop[i], eo_spinor_work[0], eo_spinor_work[1] );

    Q_clover_phi_matrix_eo ( eo_spinor_work[2], eo_spinor_work[3], eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_bc, eo_spinor_work[4], mzz[0] );

    spinor_field_eo2lexic ( spinor_work[0], eo_spinor_work[2], eo_spinor_work[3] );

    norm = 0.;
    spinor_scalar_product_re ( &norm, spinor_work[0], spinor_work[0], VOLUME );
    fprintf(stdout, "# [test_seq_prop] norm source     %2d  %e\n", i, sqrt(norm));

#if 0
    if ( g_write_source ) {
      if ( g_source_type == 0 ) {
        sprintf ( filename, "%s.%c.%.4d.t%dx%dy%dz%d.%.2d.ascii", filename_prefix3, flavor_tag, Nconf, 
            g_source_coords_list[0][0], g_source_coords_list[0][1], g_source_coords_list[0][2], g_source_coords_list[0][3], i );
      }

      FILE * ffs = fopen( filename, "w" );
      exitstatus = printf_spinor_field( spinor_field[i], 0, ffs);
      if( exitstatus != 0 ) {
        fprintf(stderr, "[test_seq_prop] Error from printf_spinor_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(9);
      }
      fclose ( ffs );
    }
#endif  /* of if 0 */

    if ( g_source_type == 0 ) {
      /***********************************************************
       * check point source
       ***********************************************************/
      int sx[4], source_proc_id;
      exitstatus = get_point_source_info ( g_source_coords_list[0], sx, &source_proc_id);
      if( exitstatus != 0 ) {
        fprintf(stderr, "[test_seq_prop] Error from get_point_source_info, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(9);
      }

      if ( source_proc_id == g_cart_id ) {
        spinor_work[0][_GSI( g_ipt[sx[0]][sx[1]][sx[2]][sx[3]]) + 2*i] -= 1.;
      }

      double norm_diff = 0.;
      spinor_scalar_product_re ( &norm_diff, spinor_work[0], spinor_work[0], VOLUME);
      fprintf(stdout, "# [test_seq_prop] norm-diff %2d %e\n", i, sqrt(norm_diff));
    }

  }  /* end of loop on spin color i */

  /***********************************************************
   * loop on fields for sequential propagator
   ***********************************************************/
  for ( int i = 0; i < nsc; i++ )
  {

    /***********************************************************
     * read read the spinor fields
     ***********************************************************/
    sprintf ( filename, "%s_s%d_c%d", filename_prefix2, i/3, i%3 ); 

    if(g_cart_id==0) fprintf(stdout, "# [test_seq_prop] Reading field from file %s %s %d\n", filename, __FILE__, __LINE__ );
    exitstatus = read_lime_spinor ( sprop[i], filename, 0 ); 
    if( exitstatus != 0 ) {
      fprintf(stderr, "[test_seq_prop] Error from read_lime_spinor for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
      EXIT(9);
    }

    double norm = 0.;
    spinor_scalar_product_re ( &norm, sprop[i], sprop[i], VOLUME );
    fprintf(stdout, "# [test_seq_prop] norm(1) seq-propagator %2d  %e\n", i, sqrt(norm));

    /***********************************************************
     * ukqcd -> cvc gamma basis rotation at sink;
     ***********************************************************/
#pragma omp parallel for
    for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
      _fv_cvc_eq_convert_fv_ukqcd ( sprop[i]+_GSI(ix) , sprop[i]+_GSI(ix) );
    }
  
  }  /* end of loop on nsc */

  if ( rotate_source_side ) {
    convert_ukqcd_to_cvc_at_source ( sprop );
  }

  for ( int i = 0; i < nsc; i++ ) {
    if ( g_write_propagator ) {
      if ( g_source_type == 0 ) {
        sprintf ( filename, "seq-%s.%c.%.4d.t%dx%dy%dz%d.%.2d.inverted", filename_prefix3, flavor_tag, Nconf,
            g_source_coords_list[0][0], g_source_coords_list[0][1], g_source_coords_list[0][2], g_source_coords_list[0][3], i );
      }
      exitstatus = write_propagator ( sprop[i],  filename, 0, g_propagator_precision );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[test_seq_prop] Error from write_propagator for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
        EXIT(9);
      }
    }

    /***********************************************************
     * norm of input field
     ***********************************************************/
    double norm = 0.;
    spinor_scalar_product_re ( &norm, sprop[i], sprop[i], VOLUME );
    fprintf(stdout, "# [test_seq_prop] norm(2) seq-propagator %2d  %e\n", i, sqrt(norm));

    /***********************************************************
     * apply D
     ***********************************************************/
    spinor_field_lexic2eo ( sprop[i], eo_spinor_work[0], eo_spinor_work[1] );

    Q_clover_phi_matrix_eo ( eo_spinor_work[2], eo_spinor_work[3], eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_bc, eo_spinor_work[4], mzz[1] );

    spinor_field_eo2lexic ( sprop[i], eo_spinor_work[2], eo_spinor_work[3] );

    norm = 0.;
    spinor_scalar_product_re ( &norm, sprop[i], sprop[i], VOLUME );
    fprintf(stdout, "# [test_seq_prop] norm seq-source     %2d  %e\n", i, sqrt(norm));


    if ( g_write_sequential_source ) {
      if ( g_source_type == 0 ) {
        sprintf ( filename, "seq-%s.%c.%.4d.t%dx%dy%dz%d.%.2d", filename_prefix3, flavor_tag, Nconf, 
            g_source_coords_list[0][0], g_source_coords_list[0][1], g_source_coords_list[0][2], g_source_coords_list[0][3], i );
      }

      exitstatus = write_propagator ( sprop[i],  filename, 0, g_propagator_precision );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[test_seq_prop] Error from write_propagator for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
        EXIT(9);
      }
    }

    /***********************************************************
     * check sequential source
     ***********************************************************/
    int sx[4], source_proc_id;
    exitstatus = get_point_source_info ( g_source_coords_list[0], sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[test_seq_prop] Error from get_point_source_info, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(9);
    }

    double const p[3] = {
      2. * M_PI * (double)g_seq_source_momentum_list[0][0] / (double)LX_global,
      2. * M_PI * (double)g_seq_source_momentum_list[0][1] / (double)LY_global,
      2. * M_PI * (double)g_seq_source_momentum_list[0][2] / (double)LZ_global };

    memset( spinor_work[0], 0, sizeof_spinor_field );
    if ( g_source_coords_list[0][0] / T == g_proc_coords[0] ) {
      for ( int x1 = 0; x1 < LX; x1++ ) {
      for ( int x2 = 0; x2 < LY; x2++ ) {
      for ( int x3 = 0; x3 < LZ; x3++ ) {
        double spinor1[24];
        double const phase = x1 * p[0] + x2 * p[1] + x3 * p[2];
        complex ephase = { cos(phase) , sin(phase) };
        unsigned int const ix = g_ipt[sx[0]][x1][x2][x3];
        _fv_eq_gamma_ti_fv( spinor1, g_sequential_source_gamma_id_list[0], fprop[i]+_GSI(ix) );
        _fv_eq_fv_ti_co ( spinor_work[0]+_GSI(ix), spinor1, &ephase );
      }}}
    }

     if ( g_write_sequential_source ) {
      if ( g_source_type == 0 ) {
        sprintf ( filename, "seq2-%s.%c.%.4d.t%dx%dy%dz%d.%.2d", filename_prefix3, flavor_tag, Nconf,
            g_source_coords_list[0][0], g_source_coords_list[0][1], g_source_coords_list[0][2], g_source_coords_list[0][3], i );
      }

      exitstatus = write_propagator ( spinor_work[0],  filename, 0, g_propagator_precision );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[test_seq_prop] Error from write_propagator for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
        EXIT(9);
      }
    }


    double norm_diff = 0.;

    spinor_field_ti_eq_re ( spinor_work[0], -1., VOLUME );

    spinor_field_norm_diff ( &norm_diff, spinor_work[0], sprop[i], VOLUME );

    fprintf(stdout, "# [test_seq_prop] seq norm-diff %2d %e\n", i, sqrt(norm_diff));

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
  fini_2level_dtable ( &fprop );
  fini_2level_dtable ( &sprop );

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_seq_prop] %s# [test_seq_prop] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_seq_prop] %s# [test_seq_prop] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }
  return(0);
}

