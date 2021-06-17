/****************************************************
 * test_oet_prop
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

/***********************************************************
 *
 ***********************************************************/
static inline void convert_ukqcd_to_cvc_at_source ( double ** spinor_field , const int nc ) {

  double const norm = 1. / sqrt( 2. );
  const int nc2 = 2 * nc, nc3 = 3 * nc;

  /***********************************************************
   * ukqcd -> cvc gamma basis rotation at source;
   * assumes specific forms of gamma_t and gamma_5
   * requires all 12 spin-color components
   ***********************************************************/
#pragma omp parallel for
    for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
      double spinor1[4][24];
      for ( int ic = 0; ic < nc; ic++ ) {
        _fv_eq_fv ( spinor1[0], spinor_field[      ic] + _GSI(ix) );
        _fv_eq_fv ( spinor1[1], spinor_field[nc  + ic] + _GSI(ix) );
        _fv_eq_fv ( spinor1[2], spinor_field[nc2 + ic] + _GSI(ix) );
        _fv_eq_fv ( spinor1[3], spinor_field[nc3 + ic] + _GSI(ix) );

        /*  */
        _fv_eq_fv_mi_fv (spinor_field[  ic]+_GSI(ix), spinor1[0], spinor1[2] );

        /*  */
        _fv_eq_fv_mi_fv (spinor_field[nc+ic]+_GSI(ix), spinor1[1], spinor1[3] );

        /*  */
        _fv_eq_fv    ( spinor_field[nc2 + ic]+_GSI(ix), spinor1[0] );
        _fv_ti_eq_re ( spinor_field[nc2 + ic]+_GSI(ix), -1. );
        _fv_mi_eq_fv ( spinor_field[nc2 + ic]+_GSI(ix), spinor1[2] );

        /*  */
        _fv_eq_fv    ( spinor_field[nc3 + ic]+_GSI(ix), spinor1[1] );
        _fv_ti_eq_re ( spinor_field[nc3 + ic]+_GSI(ix), -1. );
        _fv_mi_eq_fv ( spinor_field[nc3 + ic]+_GSI(ix), spinor1[3] );

        _fv_ti_eq_re ( spinor_field[      ic]+_GSI(ix), norm );
        _fv_ti_eq_re ( spinor_field[nc  + ic]+_GSI(ix), norm );
        _fv_ti_eq_re ( spinor_field[nc2 + ic]+_GSI(ix), norm );
        _fv_ti_eq_re ( spinor_field[nc3 + ic]+_GSI(ix), norm );
      }
    }
}  /* end of convert_ukqcd_to_cvc_at_source */


void usage(void) {
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
  int rotate_source_side = 1;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "sh?vf:")) != -1) {
    switch (c) {
    case 'v':
      g_verbose = 1;
      break;
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 's':
      rotate_source_side = 1;
      fprintf ( stdout, "# [test_oet_prop] rotate_source_side set to %d\n", rotate_source_side);
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
  if(g_cart_id==0) fprintf(stdout, "# [test_oet_prop] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /***********************************************************
   * initialize MPI parameters for cvc
   ***********************************************************/
  mpi_init(argc, argv);

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_oet_prop] Error from init_geometry\n");
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
    if(g_cart_id==0) fprintf(stdout, "# [test_oet_prop] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  }

  if(exitstatus != 0) {
    fprintf(stderr, "[test_oet_prop] Error from setting g_gauge_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
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
    fprintf(stderr, "[test_oet_prop] Error from gauge_field with boundary condition, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************************
   * measure the plaquette
   ***********************************************************/
  exitstatus = plaquetteria ( gauge_field_with_bc );
  if(exitstatus != 0) {
    fprintf(stderr, "[test_oet_prop] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &mzz, &mzzinv, gauge_field_with_bc );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[test_oet_prop] Error from init_clover, status was %d\n", exitstatus );
    EXIT(1);
  }

  /***********************************************
   * flavor tag
   ***********************************************/
  char flavor_tag[2];
  if ( g_mu > 0 ) {
    flavor_tag[0] = 'u';
    flavor_tag[1] = 'd';
  } else {
    flavor_tag[0] = 'd';
    flavor_tag[1] = 'u';
  }


  /***********************************************
   * allocate fields
   ***********************************************/

  double **  prop = init_2level_dtable ( 4, _GSI(VOLUME) );
  if ( prop == NULL ) {
    fprintf ( stderr, "# [test_oet_prop] Error from init_Xlevel_ftable %s %d\n", __FILE__, __LINE__ );
    EXIT(9);
  }
  double **  source = init_2level_dtable ( 4, _GSI(VOLUME) );
  if ( source == NULL ) {
    fprintf ( stderr, "# [test_oet_prop] Error from init_Xlevel_ftable %s %d\n", __FILE__, __LINE__ );
    EXIT(9);
  }

  double ** eo_spinor_work = init_2level_dtable ( 6, _GSI(VOLUME+RAND)/2 );
  if ( eo_spinor_work == NULL ) {
    fprintf ( stderr, "# [test_oet_prop] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(9);
  }

  double * spinor_work[3] = {
    eo_spinor_work[0],
    eo_spinor_work[2],
    eo_spinor_work[4] };

  /* int const nsc = ( strcmp ( field_type, "ds" ) == 0 ) ? 1 : 12; */
  int const nsc = 4;

  /***********************************************************
   * loop on fields for forward propagator
   ***********************************************************/
  for ( int i = 0; i < nsc; i++ )
  {

    /***********************************************************
     * read read the spinor fields
     ***********************************************************/
    sprintf ( filename, "%s%d", filename_prefix, i ); 

    if(g_cart_id==0) fprintf(stdout, "# [test_oet_prop] Reading field from file %s %s %d\n", filename, __FILE__, __LINE__ );
    exitstatus = read_lime_spinor ( prop[i], filename, 0 ); 
    if( exitstatus != 0 ) {
      fprintf(stderr, "[test_oet_prop] Error from read_lime_spinor for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
      EXIT(9);
    }

    sprintf ( filename, "%s%d", filename_prefix2, i ); 

    if(g_cart_id==0) fprintf(stdout, "# [test_oet_prop] Reading field from file %s %s %d\n", filename, __FILE__, __LINE__ );
    exitstatus = read_lime_spinor ( source[i], filename, 0 ); 
    if( exitstatus != 0 ) {
      fprintf(stderr, "[test_oet_prop] Error from read_lime_spinor for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
      EXIT(9);
    }

  }

  for ( int i = 1; i < 4; i++ ) {
    double dtmp = 0.;
    for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
      for ( int k = 0; k < 6; k++ ) {
        dtmp += _SQR( (source[0][_GSI(ix)+6*0+k] - source[i][_GSI(ix)+6*i+k]) );
      }
    }
    fprintf ( stdout, "# [test_oet_prop] spin component difference 0 - %d   %e\n", i, dtmp );
  }

  for ( int i = 0; i < nsc; i++ )
  {
    /***********************************************************
     * ukqcd -> cvc gamma basis rotation at sink;
     ***********************************************************/
#pragma omp parallel for
    for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
      _fv_cvc_eq_convert_fv_ukqcd ( prop[i]+_GSI(ix) , prop[i]+_GSI(ix) );
    }

#pragma omp parallel for
    for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
      _fv_cvc_eq_convert_fv_ukqcd ( source[i]+_GSI(ix) , source[i]+_GSI(ix) );
    }
  
  }  /* end of loop on nsc */

  /***********************************************************
   * ukqcd -> cvc gamma basis rotation at source; 1 color
   ***********************************************************/
  if ( rotate_source_side ) {
    convert_ukqcd_to_cvc_at_source ( prop, 1 );
    convert_ukqcd_to_cvc_at_source ( source, 1 );
  }

  for ( int i = 0; i < nsc; i++ ) {

    if ( g_write_propagator ) {
      sprintf ( filename, "%s-oet.%c.%.4d.t%.2d.px%dpy%dpz%d.%.2d.%.5d.inverted", filename_prefix3, flavor_tag[0], Nconf,
          g_source_coords_list[0][0], g_seq_source_momentum_list[0][0], g_seq_source_momentum_list[0][1], g_seq_source_momentum_list[0][2], i , 0);

      exitstatus = write_propagator ( prop[i],  filename, 0, g_propagator_precision );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[test_oet_prop] Error from write_propagator for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
        EXIT(9);
      }
    }

    if ( g_write_source ) {
      sprintf ( filename, "%s-oet.%c.%.4d.t%.2d.px%dpy%dpz%d.%.2d.%.5d", filename_prefix3, flavor_tag[0], Nconf,
          g_source_coords_list[0][0], g_seq_source_momentum_list[0][0], g_seq_source_momentum_list[0][1], g_seq_source_momentum_list[0][2], i , 0);

      exitstatus = write_propagator ( source[i],  filename, 0, g_propagator_precision );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[test_oet_prop] Error from write_propagator for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
        EXIT(9);
      }
    }

    /***********************************************************
     * norm of input field
     ***********************************************************/
    double norm = 0.;
    spinor_scalar_product_re ( &norm, source[i], source[i], VOLUME );
    fprintf(stdout, "# [test_oet_prop] norm source (1) %d  %e\n", i, sqrt(norm));

    /***********************************************************
     * apply D
     ***********************************************************/
    spinor_field_lexic2eo ( prop[i], eo_spinor_work[0], eo_spinor_work[1] );

    Q_clover_phi_matrix_eo ( eo_spinor_work[2], eo_spinor_work[3], eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_bc, eo_spinor_work[4], mzz[0] );

    spinor_field_eo2lexic ( spinor_work[0], eo_spinor_work[2], eo_spinor_work[3] );

    norm = 0.;
    spinor_scalar_product_re ( &norm, spinor_work[0], spinor_work[0], VOLUME );
    fprintf(stdout, "# [test_oet_prop] norm source (2) %d  %e\n", i, sqrt(norm));

    if ( g_write_source ) {
      sprintf ( filename, "%s2-oet.%c.%.4d.t%.2d.px%dpy%dpz%d.%.2d.%.5d", filename_prefix3, flavor_tag[0], Nconf,
          g_source_coords_list[0][0], g_seq_source_momentum_list[0][0], g_seq_source_momentum_list[0][1], g_seq_source_momentum_list[0][2], i , 0);

      exitstatus = write_propagator ( spinor_work[0],  filename, 0, g_propagator_precision );
      if( exitstatus != 0 ) {
        fprintf(stderr, "[test_oet_prop] Error from write_propagator for file %s, status was %d %s %d\n", filename, exitstatus, __FILE__, __LINE__ );
        EXIT(9);
      }
    }

    double norm_diff = 0.;

    spinor_field_norm_diff ( &norm_diff, spinor_work[0], source[i], VOLUME );
    fprintf(stdout, "# [test_oet_prop] norm-diff (1) %2d %e  %e\n", i, norm_diff , norm_diff / sqrt( norm ));

    for ( int it = 0; it < T; it++ ) {
      if ( it + g_proc_coords[0] * T != g_source_coords_list[0][0] ) {
        memset ( spinor_work[0] + _GSI( it * VOL3 ), 0, _GSI(VOL3)*sizeof( double ) );
      }
    }

    norm = 0., norm_diff = 0.;
    spinor_scalar_product_re ( &norm, spinor_work[0], spinor_work[0], VOLUME );
    fprintf(stdout, "# [test_oet_prop] norm source (3) %d  %e\n", i, sqrt(norm));

    spinor_field_norm_diff ( &norm_diff, spinor_work[0], source[i], VOLUME );
    fprintf(stdout, "# [test_oet_prop] norm-diff (2) %2d %e  %e\n", i, norm_diff , norm_diff / sqrt( norm ));

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
  fini_2level_dtable ( &prop );
  fini_2level_dtable ( &source );

#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  MPI_Finalize();
#endif

  if(g_cart_id == 0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [test_oet_prop] %s# [test_oet_prop] end fo run\n", ctime(&g_the_time));
    fflush(stdout);
    fprintf(stderr, "# [test_oet_prop] %s# [test_oet_prop] end fo run\n", ctime(&g_the_time));
    fflush(stderr);
  }
  return(0);
}

