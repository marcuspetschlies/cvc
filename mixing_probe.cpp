/***************************************************************************
 *
 * mixing_probe
 *
 ***************************************************************************/

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
#include "cvc_timer.h"
#include "mpi_init.h"
#include "set_default.h"
#include "io.h"
#include "propagator_io.h"
#include "read_input_parser.h"
#include "smearing_techniques.h"
#include "contractions_io.h"
#include "Q_clover_phi.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "contractions_io.h"
#include "contract_factorized.h"
#include "contract_diagrams.h"
#include "gamma.h"

#include "clover.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1

#define _PART_I   0  /* loops */
#define _PART_IIb 0  /* N1, N2 */
#define _PART_III 0  /* B/Z and D1c/i sequential diagrams */
#define _PART_IV  1  /* W type sequential diagrams */

#ifndef _USE_TIME_DILUTION
#define _USE_TIME_DILUTION 1
#endif


using namespace cvc;

/* typedef int ( * reduction_operation ) (double**, double*, fermion_propagator_type*, unsigned int ); */

typedef int ( * reduction_operation ) (double**, fermion_propagator_type*, fermion_propagator_type*, fermion_propagator_type*, unsigned int);


/***************************************************************************
 * 
 ***************************************************************************/
static inline int reduce_project_write ( double ** vx, double *** vp, fermion_propagator_type * fa, fermion_propagator_type * fb, fermion_propagator_type * fc, reduction_operation reduce,
    struct AffWriter_s *affw, char * tag, int (*momentum_list)[3], int momentum_number, int const nd, unsigned int const N, int const io_proc ) {

  int exitstatus;

  /* contraction */
  exitstatus = reduce ( vx, fa, fb, fc, N );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from reduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return( 1 );
  }

  /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
  exitstatus = contract_vn_momentum_projection ( vp, vx, nd, momentum_list, momentum_number);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from contract_vn_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    return( 2 );
  }

#if defined HAVE_LHPC_AFF
  /* write to AFF file */
  exitstatus = contract_vn_write_aff ( vp, nd, (struct AffWriter_s *)affw, tag, momentum_list, momentum_number, io_proc );
#endif
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from contract_vn_write for tag %s, status was %d %s %d\n", tag, exitstatus, __FILE__, __LINE__ );
    return( 3 );
  }

  return ( 0 );

}  /* end of reduce_project_write */

/***************************************************************************
 * x <- gamma * y
 ***************************************************************************/
inline void pm_eq_gamma_ti_pm ( double _Complex ** const x , gamma_matrix_type * const g, double _Complex ** const y ) {

  for ( int ll = 0; ll < 12; ll++ ) {
    
    for ( int i = 0; i < 4; i++ ) {
    
      for (int kc = 0; kc < 3; kc++ ) {

        int const ii = 3 * i + kc;
    
        double _Complex z = 0.;

        for ( int k = 0; k < 4; k++ ) {

          int const kk = 3 * k + kc;

          z += g->m[i][k] * y[kk][ll];

        }  /* end of loop on k */

        x[ii][ll] = z;

      }
    }
  }
  return;
}  /* end of pm_eq_gamma_ti_pm */

/***************************************************************************
 * x <- gamma * y^+
 ***************************************************************************/
inline void pm_eq_gamma_ti_pm_dag ( double _Complex ** const x , gamma_matrix_type * const g, double _Complex ** const y ) {

  for ( int ll = 0; ll < 12; ll++ ) {
    
    for ( int i = 0; i < 4; i++ ) {
    
      for (int kc = 0; kc < 3; kc++ ) {

        int const ii = 3 * i + kc;
    
        double _Complex z = 0.;

        for ( int k = 0; k < 4; k++ ) {

          int const kk = 3 * k + kc;

          z += g->m[i][k] * conj( y[ll][kk] );

        }  /* end of loop on k */

        x[ii][ll] = z;

      }
    }
  }
  return;
}  /* end of pm_eq_gamma_ti_pm_dag */

/***************************************************************************
 *
 ***************************************************************************/
inline void pm_eq_pm_ti_pm ( double _Complex ** const x, double _Complex ** const y, double _Complex ** const x ) {

  for ( int ii = 0; ii < 12; ii++ ) {
  for ( int ll = 0; ll < 12; ll++ ) {
    double _Complex a = 0.;

    for ( int kk = 0; kk < 12; kk++ ) {
      a += y[ii][kk] * z[kk][ll];
    }

    x[ii][ll] = a;
  }}
  return;
}  /* pm_eq_pm_ti_pm */

/***************************************************************************
 *
 ***************************************************************************/
inline double _Complex co_eq_tr_pm ( double _Complex ** const y ) {

  double _Complex x = 0.;

  for ( int ii = 0; ii < 12; ii++ ) {
    x += y[ii][ii];
  }

  return ( x );
}  /* end of co_eq_tr_pm */

/***************************************************************************
 *
 ***************************************************************************/
inline void pm_eq_pm_dag_ti_pm ( double _Complex ** const x, double _Complex ** const y, double _Complex ** const x ) {

  for ( int ii = 0; ii < 12; ii++ ) {
  for ( int ll = 0; ll < 12; ll++ ) {
    double _Complex a = 0.;

    for ( int kk = 0; kk < 12; kk++ ) {
      a += conj ( y[kk][ii] ) * z[kk][ll];
    }

    x[ii][ll] = a;
  }}
  return;
}  /* pm_eq_pm_dag_ti_pm */

/***************************************************************************
 *
 ***************************************************************************/
inline void pm_eq_pm_ti_pm_dag ( double _Complex ** const x, double _Complex ** const y, double _Complex ** const x ) {

  for ( int ii = 0; ii < 12; ii++ ) {
  for ( int ll = 0; ll < 12; ll++ ) {
    double _Complex a = 0.;

    for ( int kk = 0; kk < 12; kk++ ) {
      a += y[ii][kk] * conj ( z[ll][kk] );
    }

    x[ii][ll] = a;
  }}
  return;
}  /* pm_eq_pm_ti_pm_dag */

/***************************************************************************
 * x += gamma * y * gamma
 ***************************************************************************/
inline void pm_pl_eq_gamma_ti_pm_ti_gamma ( double _Complex ** const x , gamma_matrix_type * const g, double _Complex ** const y, double _Complex const a ) {

  for ( int j = 0; j < 4; j++ ) {
    for ( int jc = 0; jc < 3; ic++ ) {

      int const jj = 3 * j + jc;

      for ( int i = 0; i < 4; i++ ) {
    
        for (int ic = 0; ic < 3; ic++ ) {
        
          int const ii = 3 * i + ic;

          double _Complex z = 0.;

          for ( int l = 0; l < 3; l++ ) {

            int const ll = 3 * l + jc;

            for ( int k = 0; k < 4; k++ ) {

              int const kk = 3 * k + kc;

              z += g->m[i][k] * y[kk][ll] * g->m[l][j];

            }  /* end of loop on k */

          }  /* end of loop on l */

          x[ii][jj] = a * x[ii][jj] + z;

        }  /* ic */
      }  /* i */
    }  /* jc */
  }  /* j */

  return;

}  /* end of pm_pl_eq_gamma_ti_pm_ti_gamma */


/***************************************************************************
 * helper message
 ***************************************************************************/
void usage() {
  fprintf(stdout, "Code for FHT-type nucleon-nucleon 2-pt function inversion and contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f input <filename> : input filename for [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual [default false]\n");
  fprintf(stdout, "          -h                  : this message\n");
  EXIT(0);
}

/***************************************************************************
 *
 * MAIN PROGRAM
 *
 ***************************************************************************/
int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "mx_prb";

  const char flavor_tag[4] = { 'u', 'd', 's', 'c' };

  char const gamma_id_to_Cg_ascii[16][10] = {
    "Cgy",
    "Cgzg5",
    "Cgt",
    "Cgxg5",
    "Cgygt",
    "Cgyg5gt",
    "Cgyg5",
    "Cgz",
    "Cg5gt",
    "Cgx",
    "Cgzg5gt",
    "C",
    "Cgxg5gt",
    "Cgxgt",
    "Cg5",
    "Cgzgt"
  };


  char const gamma_id_to_ascii[16][10] = {
    "gt",
    "gx",
    "gy",
    "gz",
    "id",
    "g5",
    "gtg5",
    "gxg5",
    "gyg5",
    "gzg5",
    "gtgx",
    "gtgy",
    "gtgz",
    "gxgy",
    "gxgz",
    "gygz" 
  };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[400];
  double **lmzz[2] = { NULL, NULL }, **lmzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  struct timeval ta, tb, start_time, end_time;

  int const gamma_i_number = 10;

  int const gamma_i_list[10] = { 4, 5, 0,  1,  2,  3,  6,  7,  8,  9 };


#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char aff_tag[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "sSrwch?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  gettimeofday ( &start_time, (struct timezone *)NULL );


  /***************************************************************************
   * read input and set the default values
   ***************************************************************************/
  if(filename_set==0) strcpy(filename, "twopt.input");
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [mixing_probe] calling tmLQCD wrapper init functions\n");

  /***************************************************************************
   * initialize tmLQCD solvers
   ***************************************************************************/
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

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  mpi_init(argc, argv);
  mpi_init_xchange_contraction(2);

  /***************************************************************************
   * report git version
   * make sure the version running here has been commited before program call
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [mixing_probe] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [mixing_probe] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [mixing_probe] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[mixing_probe] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[mixing_probe] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  
  /***************************************************************************
   * initialize lattice geometry
   *
   * allocate and fill geometry arrays
   ***************************************************************************/
  geometry();


  /***************************************************************************
   * set up some mpi exchangers for
   * (1) even-odd decomposed spinor field
   * (2) even-odd decomposed propagator field
   ***************************************************************************/
  mpi_init_xchange_eo_spinor();
  mpi_init_xchange_eo_propagator();

  /***************************************************************************
   * set up the gauge field
   *
   *   either read it from file or get it from tmLQCD interface
   *
   *   lime format is used
   ***************************************************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [mixing_probe] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [mixing_probe] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[mixing_probe] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[mixing_probe] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[mixing_probe] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize the clover term, 
   * lmzz and lmzzinv
   *
   *   mzz = space-time diagonal part of the Dirac matrix
   *   l   = light quark mass
   ***************************************************************************/
  exitstatus = init_clover ( &lmzz, &lmzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[mixing_probe] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[mixing_probe] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [mixing_probe] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * set the gamma matrices
   ***************************************************************************/
  gamma_matrix_type gamma_list[16];

  for ( int i = 0; i < 16; i++ ) {
    gamma_matrix_set( &( gamma_list[i]), i, 1. ); 
  }


  /***************************************************************************
   * prepare the Fourier phase field
   ***************************************************************************/
  unsigned int const VOL3 = LX * LY * LZ;
  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );

  /***************************************************************************
   * init rng state
   ***************************************************************************/
  exitstatus = init_rng_state ( g_seed, &g_rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[mixing_probe] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  /***************************************************************************
   ***************************************************************************
   **
   ** PART I

   ** point-to-all propagators
   **
   ***************************************************************************
   ***************************************************************************/

  /***************************************************************************
   * loop on source locations
   *
   *   each source location is given by 4-coordinates in
   *   global variable
   *   g_source_coords_list[count][0-3] for t,x,y,z
   ***************************************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) {

    /***************************************************************************
     * determine source coordinates,
     * find out, if source_location is in this process
     ***************************************************************************/

    int const gsx[4] = {
        ( g_source_coords_list[isource_location][0] +  T_global ) %  T_global,
        ( g_source_coords_list[isource_location][1] + LX_global ) % LX_global,
        ( g_source_coords_list[isource_location][2] + LY_global ) % LY_global,
        ( g_source_coords_list[isource_location][3] + LZ_global ) % LZ_global };

    int sx[4], source_proc_id = -1;
    exitstatus = get_point_source_info (gsx, sx, &source_proc_id);
    if( exitstatus != 0 ) {
      fprintf(stderr, "[mixing_probe] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /***************************************************************************
     * open output file reader
     * we use the AFF format here
     * https://github.com/usqcd-software/aff
     *
     * one data file per source position
     ***************************************************************************/
#if defined HAVE_LHPC_AFF
    /***************************************************************************
     * writer for aff output file
     * only I/O process id 2 opens a writer
     ***************************************************************************/
    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.t%dx%dy%dz%d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [mixing_probe] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[mixing_probe] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
#else
      fprintf(stderr, "[mixing_probe] Error, no outupt variant selected %s %d\n",  __FILE__, __LINE__);
      EXIT(15);
#endif
    }  /* end of if io_proc == 2 */

    /***************************************************************************
     * allocate point-to-all propagators,
     * spin-color dilution (i.e. 12 fields per flavor of size 24xVOLUME real )
     ***************************************************************************/

    /* up and down quark propagator with source smearing */
    double *** propagator = init_3level_dtable ( 2, 12, _GSI( VOLUME ) );
    if( propagator == NULL ) {
      fprintf(stderr, "[mixing_probe] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    for ( int iflavor = 0; iflavor < 2; iflavor++ ) 
    {

      gettimeofday ( &ta, (struct timezone *)NULL );

      /***********************************************************
       * flavor-type point-to-all propagator
       *
       * NO SOURCE OR SINK smearing here
       *
       * NOTE: quark flavor is controlled by value of iflavor
       ***********************************************************/
      /*                                     output field         src coords flavor type  src smear  sink smear gauge field for smearing,  for residual check ...                                   */
      exitstatus = point_source_propagator ( propagator[iflavor], csx,       iflavor,     0,         0,         NULL,       check_propagator_residual, gauge_field_with_phase, lmzz );
      if(exitstatus != 0) {
        fprintf(stderr, "[mixing_probe] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(12);
      }
      
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "mixing_probe", "forward-light-smear-invert-check", g_cart_id == 0 );

    }  /* end of loop on flavor */

    /***************************************************************************
     ***************************************************************************
     **
     ** Part II
     **
     ** point-to-all propagator contractions
     **
     ***************************************************************************
     ***************************************************************************/
#if _PART_II

    /***************************************************************************
     * loop on flavor combinations
     ***************************************************************************/
    for ( int iflavor = 0; iflavor < 2; iflavor++ ) {

      gettimeofday ( &ta, (struct timezone *)NULL );

      double _Complex **** corr_accum[2][T][4][4];

      double _Complex *** gug = init_3level_ztable ( 4, 12, 12 );
      double _Complex * gu = init_1level_ztable ( 10 )  

      if ( source_proc_id == g_cart_id ) {

        double _Complex up[12][12], aux[12][12];
        unsigned int const iix = _GSI( g_ipt[sx[0]][sx[1]][sx[2]][sx[3]] );
        for ( int i=0; i<12;i++) {
        for ( int k=0; k<12;k++) {
          up[i][k] = propagator[  iflavor][i][ iix + 2 * k    ] + I * propagator[  iflavor][i][ iix + 2 * k + 1 ];
        }}

        /* scalar */
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[0], gamma_list[4], up, 0. );

        /* pseudoscalar */
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[1], gamma_list[5], up, 0. );

        /* vector */
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[2], gamma_list[0], up, 0. );
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[2], gamma_list[1], up, 1. );
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[2], gamma_list[2], up, 1. );
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[2], gamma_list[3], up, 1. );

        /* axial-vector */
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[3], gamma_list[6], up, 0. );
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[3], gamma_list[7], up, 1. );
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[3], gamma_list[8], up, 1. );
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[3], gamma_list[9], up, 1. );
     
        for ( int i=0; i<gamma_i_number; i++ ) {

          pm_eq_gamma_ti_pm ( aux , gamma_list[ gamma_i_list[i] ], up );
 
          gu[i] = co_eq_tr_pm ( aux );
        }

      }  /* end of if have source point */
#ifdef HAVE_MPI
      int nitem = 1152;
      exitstatus = MPI_Bcast (gug[0][0], nitem, MPI_DOUBLE, source_proc_id, g_cart_grid );
      if ( exitstatus != MPI_SUCCESS ) {
        fprintf ( stderr, "[mixing_probe] Error from MPI_Bcast, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(2);
      }
      nitem = 20;
      exitstatus = MPI_Bcast ( (double*)gu, nitem, MPI_DOUBLE, source_proc_id, g_cart_grid );
      if ( exitstatus != MPI_SUCCESS ) {
        fprintf ( stderr, "[mixing_probe] Error from MPI_Bcast, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(2);
      }
#endif

      /***************************************************************************
       *
       ***************************************************************************/
#ifdef HAVE_OPENMP
      omp_lock_t writelock;

      omp_init_lock(&writelock);

#pragma omp parallel
{
#endif

      double _Complex **** corr = init_4level_ztable ( T, 2, 4, 4 );

#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {

        int const it = ix / VOL3;

        unsigned int const iix = _GSI( ix );

        double _Complex up[12][12], dn[12][12];

        for ( int i=0; i<12;i++) {
        for ( int k=0; k<12;k++) {
          up[i][k] = propagator[  iflavor][i][ iix + 2 * k    ] + I * propagator[  iflavor][i][ iix + 2 * k +1 ];
          dn[i][k] = propagator[1-iflavor][i][ iix + 2 * k    ] + I * propagator[1-iflavor][i][ iix + 2 * k +1 ];
        }}


        double _Complex gdg[12][12], pm[12][12];

        /* gdg <- g5 dn g5  */
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gdg, gamma_list[5], dn, 0. );


        double _Complex aux[4][12][12];

        for ( int i = 0; i < 4; i++ ) {
          /* pm <- ( gc up gc )  g5 dn^+ g5 */
          pm_eq_pm_ti_pm_dag ( pm, gug[i], gdg );
       
          /* aux <- up pm = up ( gc up gc ) [ g5 dn^+ g5 ] */
          pm_eq_pm_ti_pm ( aux[i], up, pm );
        }

        double _Complex aux2[10][12][12];
        for ( int i = 0; i < 10; i++ ) {

          /* pm <- gc ( g5 dn g5 )^+ */
          pm_eq_gamma_ti_pm_dag ( pm , gamma_list[ gamma_i_list[i] ], gdg );

          /* aux2 <- up pm */
          pm_eq_pm_ti_pm ( aux2[i], up, pm );
        }

        /***************************************************************************
         * Tr { Gamma_f aux } = Tr { Gamma_f up ( gc up gc ) [ g5 dn^+ g5 ] }
         ***************************************************************************/
        for ( int i = 0; i < 4; i++ ) {
          corr[it][0][i][0] += co_eq_tr_pm ( aux[i] );
        }

        for ( int i = 0; i < 4; i++ ) {
          pm_eq_gamma_ti_pm ( pm, gamma_list[5], aux[i] );
          corr[it][0][i][1] += co_eq_tr_pm ( pm );
        }

        /* Gamma_f = 1 */
        /*   Gamma_c = 1 */
        corr[it][1][0][0] += gu[0] * co_eq_tr_pm ( aux2[0] );

        /*   Gamma_c = g5 */
        corr[it[1]][1][0] += gu[1] * co_eq_tr_pm ( aux2[1] );

        /*   Gamma_c = g_mu */
        for ( int i = 2; i<6; i++ ) {
          corr[it][1][2][0] += gu[i] * co_eq_tr_pm ( aux2[i] );
        }

        /*   Gamma_c = g_mu g5 */
        for ( int i = 6; i<10; i++ ) {
          corr[it][1][3][0] += gu[i] * co_eq_tr_pm ( aux2[i] );
        }

        /* Gamma_f = g5 */
        pm_eq_gamma_ti_pm ( pm, gamma_list[5], aux2[0] );
        corr[it][1][0][1] += gu[0] * co_eq_tr_pm ( pm] )

        pm_eq_gamma_ti_pm ( pm, gamma_list[5], aux2[1] );
        corr[it][1][1][1] += gu[1] * co_eq_tr_pm ( pm] )

        for ( int i = 2; i < 6; i++ ) {
          pm_eq_gamma_ti_pm ( pm, gamma_list[5], aux2[i] );
          corr[it][1][2][1] += gu[i] * co_eq_tr_pm ( pm] )
        }

        for ( int i = 6; i < 10; i++ ) {
          pm_eq_gamma_ti_pm ( pm, gamma_list[5], aux2[i] );
          corr[it][1][3][1] += gu[i] * co_eq_tr_pm ( pm] )
        }

      }  /* end of loop on x */

      fini_4level_ztable ( &corr );

      fini_3level_ztable ( &gug );
      fini_1level_ztable ( &gu );

#ifdef HAVE_OPENMP
      omp_set_lock(&writelock);
#endif     
      for ( int i = 0; i < T*2*4*4; i++ ) corr_accum[0][0][0][i] += corr[0][0][0][i];

#ifdef HAVE_OPENMP
      omp_unset_lock(&writelock);

}  /* end of parallel region */

      omp_destroy_lock(&writelock);
#endif

      corr_out = init_4level_dtable ( T_global, 2, 4, 4 );

#ifdef HAVE_MPI
      /***************************************************************************
       * reduce within timeslice
       ***************************************************************************/
      int nitem = 2 * T * 4 * 4 * 2;
      double * buffer = init_1level_dtable ( nitem );

      exitstatus = MPI_Allreduce( corr_accum[0][0][0], buffer, nitem, MPI_DOUBLE, MPI_SUM, g_ts_comm );

      if(exitstatus != MPI_SUCCESS) {
        fprintf(stderr, "[mixing_probe] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(4);
      }

      /***************************************************************************
       * gather within time ray
       ***************************************************************************/
      exitstatus = MPI_Gather ( buffer, nitem, MPI_DOUBLE, corr_out, nitem, MPI_DOUBLE, 0, g_tr_comm);

      if(exitstatus != MPI_SUCCESS) {
        fprintf(stderr, "[mixing_probe] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(4);
      }

      fini_1level_dtable ( &buffer );
#else
      memcpy ( corr_out[0][0][0], corr_accum[0][0][0], nitem * sizeof ( double ) );
#endif

      fini_4level_dtable ( &corr_out );

      /***************************************************************************
       * write to file
       ***************************************************************************/
      if ( io_proc == 2 ) {
        char const diag_list[2][20] = { "b", "d" };
        char const gc_name_list[4][2] = { "s", "p", "v", "a" };
        char const op_name_list[4][12] = { "g5", "m", "D", "g5sigmaG" };

        double _Complex * buffer = init_1level_ztable ( T_global );

        for ( int idiag = 0; idiag < 2; idiag++ ) {
          for ( int ic = 0; ic < 4; ic++ ) {
            for ( int iop = 0; iop < 4; iop++ ) {
              for ( int it = 0; it < T_global; it++ ) {
                buffer[it] = corr_out[idiag][it][ic][iop];
              }
              char tag[400];
              sprintf ( tag, "" );
              exitstatus = write_aff_contraction ( buffer,  affr, NULL, tag, T_global );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[mixing_probe] Error from write_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(4);
              }

            }
          }
        }
        fini_1level_ztable ( &buffer );
      }



    }  /* end of loop on flavor */
    
#endif  /* end of if _PART_II  */


    /***************************************************************************
     * clean up
     ***************************************************************************/

#ifdef HAVE_LHPC_AFF
    /***************************************************************************
     * I/O process id 2 closes its AFF writer
     ***************************************************************************/
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[mixing_probe] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

    /***************************************************************************
     * free propagator fields
     ***************************************************************************/
    fini_3level_dtable ( &propagator );

  }  /* end of loop on source locations */

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
  ***************************************************************************/

  fini_rng_state ( &g_rng_state);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  if ( gauge_field_with_phase != NULL ) free ( gauge_field_with_phase );

  /* free clover matrix terms */
  fini_clover ( );

  /* free lattice geometry arrays */
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
  show_time ( &start_time, &end_time, "mixing_probe", "runtime", g_cart_id == 0 );

  return(0);

}
