/****************************************************
 * p2gg_invert_contract_exdefl
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
#include "scalar_products.h"
#include "Q_clover_phi.h"
#include "contract_cvc_tensor.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"

#include "clover.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1
#define _OP_ID_ST 1


using namespace cvc;

static inline void _fv_cvc_eq_convert_fv_ukqcd ( double * const r , double * const s ) {
  double const _sqrt2inv = 0.7071067811865475;
  double _spinor1[24], _spinor2[24];
  _fv_eq_gamma_ti_fv ( _spinor1, 5, s );
  _fv_eq_gamma_ti_fv ( _spinor2, 0, s );
  _fv_eq_fv_pl_fv ( r, _spinor1, _spinor2 );
  _fv_ti_eq_re ( r, _sqrt2inv );
}  /* end of _fv_cvc_eq_convert_fv_ukqcd */


void usage() {
  fprintf(stdout, "Code to perform P-J-J correlator contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for cvc  [default p2gg.input]\n");
  fprintf(stdout, "          -w                  : check position space WI [default false]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {

  const char outfile_prefix[] = "p2gg";

  int c;
  int filename_set = 0;
  int gsx[4], sx[4];
  int check_position_space_WI=0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[100];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_bc = NULL;
  int evecs_num = 0;
  int check_eigenpair_equation = 0;
  int rotate_eigenvectors = 1;
  struct timeval ta, tb, start_time, end_time;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "pcwh?f:n:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'w':
      check_position_space_WI = 1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 'n':
      evecs_num = atoi ( optarg );
      break;
    case 'p':
      check_eigenpair_equation = 1;
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
  /* fprintf(stdout, "# [p2gg_invert_contract_exdefl] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [p2gg_invert_contract_exdefl] calling tmLQCD wrapper init functions\n");

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1, 0);
  /* exitstatus = tmLQCD_invert_init(argc, argv, 1 ; */
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
    fprintf(stdout, "# [p2gg_invert_contract_exdefl] git version = %s\n", g_gitversion);
  }


  /*********************************
   * set number of openmp threads
   *********************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [p2gg_invert_contract_exdefl] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [p2gg_invert_contract_exdefl] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[p2gg_invert_contract_exdefl] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[p2gg_invert_contract_exdefl] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  unsigned int const VOL3 = LX * LY * LZ;
  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  unsigned int const Vhalf            = VOLUME / 2;
  size_t const sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);
  size_t const sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [p2gg_invert_contract_exdefl] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [p2gg_invert_contract_exdefl] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[p2gg_invert_contract_exdefl] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[p2gg_invert_contract_exdefl] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /*************************************************
   * allocate memory for eo spinor fields 
   * WITH HALO
   *************************************************/
  int const no_eo_fields = 6;
  double ** eo_spinor_work  = init_2level_dtable ( (size_t)no_eo_fields, _GSI( (size_t)(VOLUME+RAND)/2) );
  if ( eo_spinor_work == NULL ) {
    fprintf(stderr, "[p2gg_invert_contract_exdefl] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_bcfactor ( &gauge_field_with_bc, g_gauge_field, -1. );
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_invert_contract_exdefl] Error from gauge_field_eq_gauge_field_ti_bcfactor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  exitstatus = plaquetteria ( gauge_field_with_bc );
  if(exitstatus != 0) {
    fprintf(stderr, "[p2gg_invert_contract_exdefl] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * initialize clover, mzz and mzz_inv
   ***********************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_bc, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[p2gg_invert_contract_exdefl] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***********************************************
   * set io process
   ***********************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[p2gg_invert_contract_exdefl] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [p2gg_invert_contract_exdefl] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***********************************************************
   * allocate eo_spinor_field
   ***********************************************************/
  double *** eo_evecs_field = init_3level_dtable ( evecs_num, 2, _GSI( (size_t)Vhalf));
  if( eo_evecs_field == NULL ) {
    fprintf(stderr, "[p2gg_invert_contract_exdefl] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__);
    EXIT(123);
  }
  
#if 0
  /***********************************************************
   * read evecs,
   * convert to cvc spinor conventions
   ***********************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

  sprintf ( filename, "eves.%.4d.NeV%d.proc%d", Nconf, evecs_num, g_cart_id );
  FILE * fs = fopen ( filename, "r" );
  if ( fs == NULL ) {
    fprintf ( stderr, "[p2gg_invert_contract_exdefl] Error from fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
    EXIT(12);
  } else {
    if ( g_verbose > 0 ) fprintf ( stdout, "[p2gg_invert_contract_exdefl] reading evecs from part-file %s\n", filename );
  }

  size_t count = _GSI(VOLUME) * evecs_num;

  if ( count != fread ( eo_evecs_field[0][0], sizeof(double), count, fs) ) {
    fprintf ( stderr, "[p2gg_invert_contract_exdefl] Error from fread %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }

  fclose ( fs );

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "p2gg_invert_contract_exdefl", "read-evecs-partfile", g_cart_id == 0 );

  /***********************************************************
   * rotate in position space x <-> z
   * rotate in spinor space  U = ( g5 + gt ) / sqrt(2)
   ***********************************************************/
  if ( rotate_eigenvectors ) {

    gettimeofday ( &ta, (struct timezone *)NULL );
  
    for( int i = 0; i < evecs_num; i++ ) {
  
      double * evecs_buffer = init_1level_dtable ( _GSI(VOLUME) );
      if( evecs_buffer == NULL ) {
        fprintf(stderr, "[p2gg_invert_contract_exdefl] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(124);
      }
  
      memcpy ( evecs_buffer, eo_evecs_field[i][0], sizeof_spinor_field );
  
  #ifdef HAVE_OPENMP
  #pragma omp parallel for
  #endif
      for ( int it = 0; it <  T; it++ ) {
      for ( int ix = 0; ix < LX; ix++ ) {
      for ( int iy = 0; iy < LY; iy++ ) {
      for ( int iz = 0; iz < LZ; iz++ ) {
        /* x_source in lexic t,z,y,x ordering from QUDA */
        unsigned int const x_source = it * VOL3 + ( iz * LY + iy ) * LX + ix;
  
        /* x_target in lexic t,x,y,z order */
        unsigned int const x_target = g_ipt[it][ix][iy][iz];
  
        int const ieo = !g_iseven[x_target]; /* ieo = 0, if x_target is even;
                                                ieo = 1, if x_target is odd */
  
        /* x_target_eo for even-odd decomposed storage */
        unsigned int const x_target_eo = g_lexic2eosub[x_target];
  
        _fv_cvc_eq_convert_fv_ukqcd ( eo_evecs_field[i][ieo] + _GSI(x_target_eo) , evecs_buffer + _GSI(x_source) );
      }}}}
  
      fini_1level_dtable ( &evecs_buffer );
  
    }  /* end of loop on evecs */
  
    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "p2gg_invert_contract_exdefl", "rotate-evecs", g_cart_id == 0 );
  
  }  /* end of if rotate eigenvectors */

  /***********************************************************
   * calcualte eigenvalues
   * check eigenpair equation
   ***********************************************************/
  if ( check_eigenpair_equation ) {

    gettimeofday ( &ta, (struct timezone *)NULL );

    for( int i = 0; i < evecs_num; i++ ) {
  
      /***********************************************************
       * apply Q Qbar ( or equivalently Qbar Q )
       ***********************************************************/
      memcpy ( eo_spinor_work[0], eo_evecs_field[i][0], sizeof_eo_spinor_field );
      memcpy ( eo_spinor_work[1], eo_evecs_field[i][1], sizeof_eo_spinor_field );
  
      /* Qbar 
       * 2,3 <- Qbar 0,1 with aux 4
       */
      Q_clover_phi_matrix_eo ( eo_spinor_work[2], eo_spinor_work[3], eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_bc, eo_spinor_work[4], mzz[1] );
      g5_phi ( eo_spinor_work[2], Vhalf );
      g5_phi ( eo_spinor_work[3], Vhalf );
      /* Q
       * 0,1 <- Q 2,3 with aux 4
       */
      Q_clover_phi_matrix_eo ( eo_spinor_work[0], eo_spinor_work[1], eo_spinor_work[2], eo_spinor_work[3], gauge_field_with_bc, eo_spinor_work[4], mzz[0] );
      g5_phi ( eo_spinor_work[0], Vhalf );
      g5_phi ( eo_spinor_work[1], Vhalf );
      
      /***********************************************************
       * estimate eigenvector
       ***********************************************************/
      double norm_e, norm_o;
      spinor_scalar_product_re ( &norm_e, eo_evecs_field[i][0], eo_evecs_field[i][0], Vhalf);
      spinor_scalar_product_re ( &norm_o, eo_evecs_field[i][1], eo_evecs_field[i][1], Vhalf);
  
      complex w_e, w_o;
      spinor_scalar_product_co ( &w_e, eo_evecs_field[i][0], eo_spinor_work[0], Vhalf );
      spinor_scalar_product_co ( &w_o, eo_evecs_field[i][1], eo_spinor_work[1], Vhalf );
  
      if ( g_cart_id == 0 ) {
        fprintf ( stdout, "# [p2gg_invert_contract_exdefl] evec %2d   norm   e %25.16e   o %25.16e\n", i, norm_e, norm_o );
        fprintf ( stdout, "# [p2gg_invert_contract_exdefl] evec %2d   eval   e %25.16e +I %25.16e   o %25.16e +I %25.16e\n", i, w_e.re, w_e.im, w_o.re, w_o.im );
        fflush ( stdout );
      }
  
      double const eval = ( w_e.re + w_o.re ) / ( norm_e + norm_o );
  
      spinor_field_eq_spinor_field_pl_spinor_field_ti_re ( eo_spinor_work[0], eo_spinor_work[0], eo_evecs_field[i][0], -eval, Vhalf );
      spinor_field_eq_spinor_field_pl_spinor_field_ti_re ( eo_spinor_work[1], eo_spinor_work[1], eo_evecs_field[i][1], -eval, Vhalf );
  
      spinor_scalar_product_re ( &norm_e, eo_spinor_work[0], eo_spinor_work[0], Vhalf);
      spinor_scalar_product_re ( &norm_o, eo_spinor_work[1], eo_spinor_work[1], Vhalf);
  
      if ( g_cart_id == 0 ) {
        fprintf ( stdout, "# [p2gg_invert_contract_exdefl] epair %2d diff-norm   e %25.16e   o %25.16e\n", i, norm_e, norm_o );
        fflush ( stdout );
      }
  
    }  /* end of loop on eigenvectors */

    gettimeofday ( &tb, (struct timezone *)NULL );
    show_time ( &ta, &tb, "p2gg_invert_contract_exdefl", "check-eigenpairs", g_cart_id == 0 );

  }  /* end of if check eigenpair equation */

#endif  /* of if 0 */

  /***********************************************************
   * calculate scalar products
   ***********************************************************/

  gettimeofday ( &ta, (struct timezone *)NULL );

  double * evecs_eval = init_1level_dtable ( evecs_num );
  if ( evecs_eval == NULL ) {
    fprintf ( stderr, "[p2gg_invert_contract_exdefl] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(11);
  }

  double _Complex ***** vw_mat = init_5level_ztable ( g_source_momentum_number, g_source_gamma_id_number, T, evecs_num, evecs_num );
  if ( vw_mat == NULL ) {
    fprintf ( stderr, "[p2gg_invert_contract_exdefl] Error from init_5level_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(11);
  }
  
  double **** vw_p = init_4level_dtable ( g_source_momentum_number, g_source_gamma_id_number, evecs_num , 2 * T );
  if ( vw_p == NULL ) {
    fprintf ( stderr, "[p2gg_invert_contract_exdefl] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(11);
  }

  double **** vw_x = init_4level_dtable ( g_source_gamma_id_number, evecs_num, T, 2 * VOL3 );
  if ( vw_p == NULL ) {
    fprintf ( stderr, "[p2gg_invert_contract_exdefl] Error from init_4level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }
  
  for ( int iw = 0; iw < evecs_num; iw++ ) {
    /* apply g5 Dbar to eigenvector iw */
    memcpy ( eo_spinor_work[0], eo_evecs_field[iw][0], sizeof_eo_spinor_field );
    memcpy ( eo_spinor_work[1], eo_evecs_field[iw][1], sizeof_eo_spinor_field );

    Q_clover_phi_matrix_eo ( eo_spinor_work[2], eo_spinor_work[3], eo_spinor_work[0], eo_spinor_work[1], gauge_field_with_bc, eo_spinor_work[4], mzz[1] );
    g5_phi ( eo_spinor_work[2], Vhalf );
    g5_phi ( eo_spinor_work[3], Vhalf );

    double norm_e = 0., norm_o = 0.;
    spinor_scalar_product_re ( &norm_e, eo_spinor_work[2], eo_spinor_work[2], Vhalf);
    spinor_scalar_product_re ( &norm_o, eo_spinor_work[3], eo_spinor_work[3], Vhalf);
    evecs_eval[iw] = norm_e + norm_o;

    /* loop on source gamma's */
    for ( int igamma = 0; igamma < g_source_gamma_id_number; igamma++ ) {

      /* apply local gamma matrix */
      spinor_field_eq_gamma_ti_spinor_field ( eo_spinor_work[0], g_source_gamma_id_list[igamma], eo_spinor_work[2], Vhalf );
      spinor_field_eq_gamma_ti_spinor_field ( eo_spinor_work[1], g_source_gamma_id_list[igamma], eo_spinor_work[3], Vhalf );

      /* apply g5  */
      g5_phi ( eo_spinor_work[0], Vhalf );
      g5_phi ( eo_spinor_work[1], Vhalf );

      /* loop on evecs */
      for ( int iv = 0; iv < evecs_num; iv++ ) {

#pragma omp parallel for
        /* loop on timeslices */
        for ( int it = 0; it < T; it++ ) {
          complex z_e, z_o;
          /* loop on 3-volume */
          for ( int ix = 0; ix < VOL3; ix++ ) {
            unsigned int const iix = _GSI( it*VOL3+ix);

            double * const __ve = eo_evecs_field[iv][0] + iix;
            double * const __we = eo_spinor_work[0]     + iix;
            _co_eq_fv_dag_ti_fv ( &z_e, __ve, __we );

            double * const __vo = eo_evecs_field[iv][1] + iix;
            double * const __wo = eo_spinor_work[1]     + iix;
            _co_eq_fv_dag_ti_fv ( &z_o, __vo, __wo );

            vw_x[igamma][iv][it][2*ix  ] = ( z_e.re + z_o.re );  /* even + odd real parts */
            vw_x[igamma][iv][it][2*ix+1] = ( z_e.im + z_o.im );  /* even + odd imag parts */
          }
        }
      }  /* end of loop on evecs iv */
    }  /* end of loop on source gamma */

    /* momentum projection */
    exitstatus = momentum_projection ( vw_x[0][0][0], vw_p[0][0][0], T * evecs_num * g_source_gamma_id_number, g_source_momentum_number, g_source_momentum_list );
    if ( exitstatus != 0 ) {
      fprintf ( stderr, "[p2gg_invert_contract_exdefl] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      EXIT(12);
    }

    /* sort into vw_mat */
    for ( int imom = 0; imom < g_source_momentum_number; imom++ ) {
#pragma omp parallel for
      for ( int igamma = 0; igamma < g_source_gamma_id_number; igamma++ ) {
        for ( int it = 0; it < T; it++ ) {
          for ( int iv = 0; iv < evecs_num; iv++ ) {
            vw_mat[imom][igamma][it][iw][iv] = vw_p[imom][igamma][iv][2*it] + vw_p[imom][igamma][iv][2*it+1] * I;
          }
        }
      }
    }  /* end of loop on momenta */

  }  /* end of loop on evecs iw */

  fini_4level_dtable ( &vw_x );
  fini_4level_dtable ( &vw_p );

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "p2gg_invert_contract_exdefl", "scalar-products-calculuation", g_cart_id == 0 );

  /***********************************************************
   * write to file
   ***********************************************************/
  gettimeofday ( &ta, (struct timezone *)NULL );

#ifdef HAVE_LHPC_AFF
  /***********************************************************
   * writer for aff output file
   ***********************************************************/
  struct AffWriter_s *affw = NULL;
  if(io_proc == 2) {
    sprintf(filename, "%s.%.4d.nev%d.aff", outfile_prefix, Nconf, evecs_num );
    fprintf(stdout, "# [p2gg_invert_contract_exdefl] writing data to file %s\n", filename);
    affw = aff_writer(filename);
    const char * aff_status_str = aff_writer_errstr ( affw );
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[p2gg_invert_contract_exdefl] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(15);
    }
  }  /* end of if io_proc == 2 */
#endif

  /* write data sets to file */
  for ( int imom = 0; imom < g_source_momentum_number; imom++ ) {

    double _Complex *** buffer = NULL;

    if ( io_proc == 2 ) {
      buffer = init_3level_ztable ( T_global, evecs_num, evecs_num );
      if ( buffer == NULL ) {
        fprintf ( stderr, "[p2gg_invert_contract_exdefl] Error from init_3level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(11);
      }
    }

    for ( int igamma = 0; igamma < g_source_gamma_id_number; igamma++ ) {
#ifdef HAVE_MPI
      int count = 2 * T * evecs_num * evecs_num;

      if ( io_proc > 0 ) {
        exitstatus = MPI_Gather ( vw_mat[imom][igamma][0][0], count, MPI_DOUBLE, buffer[0][0], count, MPI_DOUBLE, 0, g_tr_comm );
        if ( exitstatus != MPI_SUCCESS ) {
          fprintf ( stderr, "[p2gg_invert_contract_exdefl] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(11);
        }
      }
#else
      memcpy ( buffer[0][0], vw_mat[imom][igamma][0][0], T * evecs_num * evecs_num * sizeof( double _Complex ) );
#endif

      if ( io_proc == 2 ) {
        struct AffNode_s * affn = aff_writer_root( affw );
        if( affn == NULL ) {
          fprintf(stderr, "[p2gg_invert_contract_exdefl] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
          EXIT(17);
        }

        /* AFF write */
        uint32_t items = ( uint32_t )T_global * evecs_num * evecs_num;
        char key[400];

        sprintf ( key, "vdag_gp_w/C%d/PX%d_PY%d_PZ%d/G%d/N%d", 
            Nconf,
            g_source_momentum_list[imom][0],
            g_source_momentum_list[imom][1],
            g_source_momentum_list[imom][2],
            g_source_gamma_id_list[igamma], evecs_num );

        struct AffNode_s * affdir = aff_writer_mkpath ( affw, affn, key );
        if ( affdir == NULL ) {
          fprintf ( stderr, "[p2gg_invert_contract_exdefl] Error from aff_writer_mkpath %d %s %d\n", __FILE__, __LINE__);
          EXIT(15);
        }

        exitstatus = aff_node_put_complex ( affw, affdir, buffer[0][0], items );
        if(exitstatus != 0) {
          fprintf ( stderr, "[p2gg_invert_contract_exdefl] Error from aff_node_put_complex, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(16);
        }

      }  /* end of if io_proc == 2 */

    }  /* end of loop on gamma matrices */

    fini_3level_ztable ( &buffer );

  }

  gettimeofday ( &tb, (struct timezone *)NULL );
  show_time ( &ta, &tb, "p2gg_invert_contract_exdefl", "scalar-products-write", g_cart_id == 0 );

  fini_5level_ztable ( &vw_mat );

  /***********************************************************
   * write eigenvalues to file
   ***********************************************************/
  if ( io_proc == 2 ) {
    struct AffNode_s * affn = aff_writer_root( affw );
    if( affn == NULL ) {
      fprintf(stderr, "[p2gg_invert_contract_exdefl] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      EXIT(17);
    }
  
    /* AFF write */
    uint32_t items = ( uint32_t )evecs_num;
    char key[400];
  
    sprintf ( key, "/eval/C%d/N%d", Nconf, evecs_num );
  
    struct AffNode_s * affdir = aff_writer_mkpath ( affw, affn, key );
    if ( affdir == NULL ) {
      fprintf ( stderr, "[p2gg_invert_contract_exdefl] Error from aff_writer_mkpath %d %s %d\n", __FILE__, __LINE__);
      EXIT(15);
    }
  
    exitstatus = aff_node_put_double ( affw, affdir, evecs_eval, items );
    if(exitstatus != 0) {
      fprintf ( stderr, "[p2gg_invert_contract_exdefl] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(16);
    }
  }  /* end of if io_proc == 2 */
  
#ifdef HAVE_LHPC_AFF
  if(io_proc == 2) {
    const char * aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[p2gg_invert_contract_exdefl] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      EXIT(32);
    }
  }  /* end of if io_proc == 2 */
#endif
  
  /****************************************
   * free the allocated memory, finalize
   ****************************************/
  
#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_bc );
  
  fini_3level_dtable ( &eo_evecs_field );
  fini_2level_dtable ( &eo_spinor_work );
  fini_1level_dtable ( &evecs_eval );


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
  show_time ( &start_time, &end_time, "p2gg_invert_contract_exdefl", "total time", g_cart_id == 0 );

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [p2gg_invert_contract_exdefl] end of run\n");
    fprintf(stderr, "# [p2gg_invert_contract_exdefl] end of run\n");
  }

  return(0);
}
