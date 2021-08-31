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
#include "contractions_io.h"
#include "Q_phi.h"
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
#include "gluon_operators.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1

#define _PART_I   1
#define _PART_II  1
#define _PART_III 1

#ifndef _USE_TIME_DILUTION
#define _USE_TIME_DILUTION 1
#endif


using namespace cvc;

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
inline void pm_eq_pm_ti_pm ( double _Complex ** const x, double _Complex ** const y, double _Complex ** const z ) {

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
 * x <- a x y + b * z
 ***************************************************************************/
inline void pm_eq_pm_pl_pm ( double _Complex ** const x, double _Complex ** const y, double _Complex ** const z, double _Complex const a, double _Complex const b ) {

  for ( int ii = 0; ii < 12; ii++ ) {
  for ( int ll = 0; ll < 12; ll++ ) {
    x[ii][ll] = a * y[ii][ll] + b * z[ii][ll];
  }}
  return;
}  /* pm_eq_pm_pl_pm */

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
inline void pm_eq_pm_dag_ti_pm ( double _Complex ** const x, double _Complex ** const y, double _Complex ** const z ) {

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
inline void pm_eq_pm_ti_pm_dag ( double _Complex ** const x, double _Complex ** const y, double _Complex ** const z ) {

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
    for ( int jc = 0; jc < 3; jc++ ) {

      int const jj = 3 * j + jc;

      for ( int i = 0; i < 4; i++ ) {
    
        for (int ic = 0; ic < 3; ic++ ) {
        
          int const ii = 3 * i + ic;

          double _Complex z = 0.;

          for ( int l = 0; l < 4; l++ ) {

            int const ll = 3 * l + jc;

            for ( int k = 0; k < 4; k++ ) {

              int const kk = 3 * k + ic;

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
 *
 ***************************************************************************/
inline void pm_eq_pm_ti_co  ( double _Complex ** const x , double _Complex ** const y, double _Complex const a ) {

  for ( int ii = 0; ii < 12; ii++ ) {
  for ( int ll = 0; ll < 12; ll++ ) {
    x[ii][ll] = y[ii][ll] * a;
  }}
  return;
}

/***************************************************************************
 *
 ***************************************************************************/
inline void pm_set_from_sf_point ( double _Complex ** const p, double ** const s, unsigned int ix ) {

  unsigned int const iix = _GSI( ix );

  for ( int k=0; k<12; k++) {
    double * const _prop = s[k] + iix;
    for ( int i=0; i<12;i++) {
      p[i][k] = _prop[2 * i ] + I * _prop[2 * i + 1];
    }
  }

return;
}  /* pm_set_from_sf */


/***************************************************************************
 *
 ***************************************************************************/
inline void pm_print (double _Complex ** const p, char * const name, FILE * ofs ) {
  fprintf ( ofs, "%s <- array( dim=c(12,12)) \n", name );
  for( int i = 0; i < 12; i++ ) {
  for( int k = 0; k < 12; k++ ) {
    fprintf ( ofs, "%s[%d,%d] <- %25.16e + %25.16e*1.i\n", name, i+1, k+1, creal( p[i][k] ), cimag ( p[i][k] ) );
  }}
  return;
}  /* pm_print */

/***************************************************************************
 *
 ***************************************************************************/
inline void gamma_print ( gamma_matrix_type * const g, char * const name, FILE * ofs ) {
  fprintf ( ofs, "%s <- array( dim=c(12,12)) \n", name );

  for( int i = 0; i < 4; i++ ) {
    for( int ic = 0; ic < 3; ic++ ) {

      int const ii = 3 * i + ic;

      for( int k = 0; k < 4; k++ ) {
        for( int kc = 0; kc < 3; kc++ ) {

          int const kk = 3 * k + kc;

          fprintf ( ofs, "%s[%d,%d] <- %25.16e + %25.16e*1.i\n", name, ii+1, kk+1, (kc==ic) * creal( g->m[i][k] ), (kc==ic) * cimag ( g->m[i][k] ) );
        }
      }
    }
  }
  return;
}  /* gamma_print */


/***************************************************************************/
/***************************************************************************/

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

  /* char const gamma_id_to_ascii[16][10] = {
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
  }; */

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[400];
  double **lmzz[2] = { NULL, NULL }, **lmzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  struct timeval ta, tb, start_time, end_time;

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
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

#if _TEST
  /* TEST */
  FILE * ofs = fopen ( "test.out", "w");
  /* END */
#endif

  /***************************************************************************
   * set the gamma matrices
   ***************************************************************************/
  /*gamma_matrix_type gamma_list[16];
  for ( int i = 0; i < 16; i++ ) {
    gamma_matrix_set( &( gamma_list[i]), i, 1. ); 
  } */

  gamma_matrix_type gamma_v[4];
  gamma_matrix_set( &( gamma_v[0]), 0, 1. ); 
  gamma_matrix_set( &( gamma_v[1]), 1, 1. ); 
  gamma_matrix_set( &( gamma_v[2]), 2, 1. ); 
  gamma_matrix_set( &( gamma_v[3]), 3, 1. ); 

  gamma_matrix_type gamma_s[1], gamma_p[1] ;
  gamma_matrix_set( &( gamma_s[0]), 4, 1. ); 
  gamma_matrix_set( &( gamma_p[0]), 5, 1. ); 

  gamma_matrix_type gamma_a[4];
  gamma_matrix_set( &( gamma_a[0]), 6, 1. ); 
  gamma_matrix_set( &( gamma_a[1]), 7, 1. ); 
  gamma_matrix_set( &( gamma_a[2]), 8, 1. ); 
  gamma_matrix_set( &( gamma_a[3]), 9, 1. ); 

  gamma_matrix_type sigma_munu[6];
  gamma_matrix_set( &( sigma_munu[0]), 10, 1. ); 
  gamma_matrix_set( &( sigma_munu[1]), 11, 1. ); 
  gamma_matrix_set( &( sigma_munu[2]), 12, 1. ); 
  gamma_matrix_set( &( sigma_munu[3]), 13, 1. ); 
  gamma_matrix_set( &( sigma_munu[4]), 14, 1. ); 
  gamma_matrix_set( &( sigma_munu[5]), 15, 1. ); 

#if _TEST
  /* TEST */
  gamma_print ( &(gamma_s[0]), "g4", ofs );
  gamma_print ( &(gamma_p[0]), "g5", ofs );
  gamma_print ( &(gamma_v[0]), "g0", ofs );
  gamma_print ( &(gamma_v[1]), "g1", ofs );
  gamma_print ( &(gamma_v[2]), "g2", ofs );
  gamma_print ( &(gamma_v[3]), "g3", ofs );
  gamma_print ( &(gamma_a[0]), "g6", ofs );
  gamma_print ( &(gamma_a[1]), "g7", ofs );
  gamma_print ( &(gamma_a[2]), "g8", ofs );
  gamma_print ( &(gamma_a[3]), "g9", ofs );
  /* END */
#endif

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
   ** Part I
   **
   ** gluon field strength tensor
   **
   ***************************************************************************
   ***************************************************************************/

  double *** Gp = init_3level_dtable ( VOLUME, 6, 9 );
  if ( Gp == NULL ) {
    fprintf ( stderr, "[mixing_probe] Error from  init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(8);
  }

  exitstatus = G_plaq ( Gp, gauge_field_with_phase, 1);
  if ( exitstatus != 0 ) {
    fprintf ( stderr, "[mixing_probe] Error from G_plaq, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }

  double *** Gpm = init_3level_dtable ( VOLUME, 6, 18 );
  if ( Gpm == NULL ) {
    fprintf ( stderr, "[mixing_probe] Error from  init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(8);
  }

#pragma omp parallel for
  for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
    for ( int i = 0; i<6;i++ ) {
      restore_from_generators ( Gpm[ix][i], Gp[ix][i] );
    }
  }

  fini_3level_dtable ( &Gp );


  /***************************************************************************
   ***************************************************************************
   **
   ** PART II
   **
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

    /* up and down quark propagator */
    double *** propagator = init_3level_dtable ( 2, 12, _GSI( VOLUME ) );
    if( propagator == NULL ) {
      fprintf(stderr, "[mixing_probe] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /* loop on propagator flavors */
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
      exitstatus = point_source_propagator ( propagator[iflavor], gsx,       iflavor,     0,         0,         NULL,       check_propagator_residual, gauge_field_with_phase, lmzz );
      if(exitstatus != 0) {
        fprintf(stderr, "[mixing_probe] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(12);
      }
      
      gettimeofday ( &tb, (struct timezone *)NULL );
      show_time ( &ta, &tb, "mixing_probe", "forward-light-invert-check", g_cart_id == 0 );

      if ( g_write_propagator ) {
        for ( int i = 0; i < 12; i++ ) {
          sprintf(filename, "%s.%c.c%d.t%dx%dy%dz%d.sc%d.inverted", filename_prefix, flavor_tag[iflavor], Nconf, gsx[0], gsx[1], gsx[2], gsx[3], i );
          if ( ( exitstatus = write_propagator ( propagator[iflavor][i], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[mixing_probe] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
        }
      }


    }  /* end of loop on flavor */

    /***************************************************************************
     ***************************************************************************
     **
     ** Part III
     **
     ** point-to-all propagator contractions for 4q operator
     **
     ***************************************************************************
     ***************************************************************************/

#if _PART_III

    /***************************************************************************
     * loop on flavor combinations
     ***************************************************************************/
    for ( int iflavor = 0; iflavor < 2; iflavor++ )
    {

      gettimeofday ( &ta, (struct timezone *)NULL );

      int const ncorr = 2 * 4 * 4 + 3;

      double _Complex ** corr_accum = init_2level_ztable ( T, ncorr );
      if ( corr_accum == NULL ) {
        fprintf ( stderr, "[mixing_probe] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(2);
      }

      /* these store sum_c Gamma_c U Gamma_c and Tr[ Gamma_c U ] for each c */
      double _Complex *** gug = init_3level_ztable ( 4, 12, 12 );
      double _Complex * gu = init_1level_ztable ( 10 );

      /* only source process can fill gug and gu */
      if ( source_proc_id == g_cart_id ) {

        double _Complex **  up = init_2level_ztable ( 12, 12 );
        double _Complex ** aux = init_2level_ztable ( 12, 12 );

        pm_set_from_sf_point ( up, propagator[iflavor], g_ipt[sx[0]][sx[1]][sx[2]][sx[3]] );

#if _TEST
        /* TEST */
        pm_print ( up, "up0", ofs );
        /* END */
#endif

        /***************************************************************************
         * ugu; first time init to zero, then add up
         ***************************************************************************/
        /* scalar */
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[0], &(gamma_s[0]), up, 0. );

        /* pseudoscalar */
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[1], &(gamma_p[0]), up, 0. );

        /* vector */
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[2], &(gamma_v[0]), up, 0. );
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[2], &(gamma_v[1]), up, 1. );
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[2], &(gamma_v[2]), up, 1. );
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[2], &(gamma_v[3]), up, 1. );

        /* axial-vector */
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[3], &(gamma_a[0]), up, 0. );
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[3], &(gamma_a[1]), up, 1. );
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[3], &(gamma_a[2]), up, 1. );
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gug[3], &(gamma_a[3]), up, 1. );

        /***************************************************************************
         * ug; single number per entry
         *
         * NOTE:sequence of gamma_i ( = Gamma_c ) must be the same everywhere
         * id g5 v a
         ***************************************************************************/
        /* scalar */
        pm_eq_gamma_ti_pm ( aux , &(gamma_s[0]), up );
        gu[0] = co_eq_tr_pm ( aux );

        /* pseudoscalar */
        pm_eq_gamma_ti_pm ( aux , &(gamma_p[0]), up );
        gu[1] = co_eq_tr_pm ( aux );

        /* vector */
        for ( int i = 0; i < 4; i++ ) {
          pm_eq_gamma_ti_pm ( aux , gamma_v+i, up );
          gu[2+i] = co_eq_tr_pm ( aux );
        }

        /* pseudovector */
        for ( int i = 0; i < 4; i++ ) {
          pm_eq_gamma_ti_pm ( aux , gamma_a+i, up );
          gu[6+i] = co_eq_tr_pm ( aux );
        }

        fini_2level_ztable ( &up );
        fini_2level_ztable ( &aux );

      }  /* end of if have source point */

#ifdef HAVE_MPI
      /***************************************************************************
       * broadcast to all
       ***************************************************************************/
      exitstatus = MPI_Bcast (gug[0][0], 1152, MPI_DOUBLE, source_proc_id, g_cart_grid );
      if ( exitstatus != MPI_SUCCESS ) {
        fprintf ( stderr, "[mixing_probe] Error from MPI_Bcast, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(2);
      }
      exitstatus = MPI_Bcast ( (double*)gu, 20, MPI_DOUBLE, source_proc_id, g_cart_grid );
      if ( exitstatus != MPI_SUCCESS ) {
        fprintf ( stderr, "[mixing_probe] Error from MPI_Bcast, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(2);
      }
#endif

#if _TEST
      /* TEST */
      pm_print ( gug[0], "sus", ofs );
      pm_print ( gug[1], "pup", ofs );
      pm_print ( gug[2], "vuv", ofs );
      pm_print ( gug[3], "aua", ofs );

      fprintf ( ofs, "gu <- numeric()\n" );
      for ( int i = 0; i < 10; i++ ) {
        fprintf ( ofs, " gu[%d] <- %25.16e + %25.16e*1.i\n", i+1, creal(gu[i]), cimag ( gu[i] ) );
      }
      /* END */
#endif



      /***************************************************************************
       * build displaced prop
       ***************************************************************************/
      /* up and down quark propagator with covariant displacement */
      double ** propagator_disp = init_2level_dtable ( 12, _GSI( VOLUME ) );
      if( propagator_disp == NULL ) {
        fprintf(stderr, "[mixing_probe] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(123);
      }

      double ** spinor_field = init_2level_dtable ( 2, _GSI(VOLUME) );
      if( spinor_field == NULL ) {
        fprintf(stderr, "[mixing_probe] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
        EXIT(123);
      }

      /* loop spin-color components */
      for ( int i = 0; i < 12; i++ ) {
        
        /* loop on directions */
        for ( int imu = 0; imu < 4; imu++ ) {

          /* forward right application */

          exitstatus = spinor_field_eq_cov_displ_spinor_field ( spinor_field[0], propagator[iflavor][i], imu, 0, gauge_field_with_phase );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[mixing_probe] Error from spinor_field_eq_cov_displ_spinor_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(2);
          }

          /* backward right application */

          exitstatus = spinor_field_eq_cov_displ_spinor_field ( spinor_field[1], propagator[iflavor][i], imu, 1, gauge_field_with_phase );
          if ( exitstatus != 0 ) {
            fprintf ( stderr, "[mixing_probe] Error from spinor_field_eq_cov_displ_spinor_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(2);
          }
         
          /* 0 <- 0 - 1 = Dfwd p - Dbwd p */ 
          spinor_field_eq_spinor_field_mi_spinor_field ( spinor_field[0], spinor_field[0], spinor_field[1], VOLUME );

          spinor_field_eq_gamma_ti_spinor_field ( spinor_field[1], imu, spinor_field[0], VOLUME );

          spinor_field_pl_eq_spinor_field ( propagator_disp[i], spinor_field[1], VOLUME );

        }  /* end of loop on directions */

        /* multiply by 1/2 */
        spinor_field_ti_eq_re ( propagator_disp[i], 0.5, VOLUME );

      }  /* end of loop on spin color components */

      fini_2level_dtable ( &spinor_field );


      /***************************************************************************
       * loop on space time for contractions
       ***************************************************************************/
#ifdef HAVE_OPENMP
      omp_lock_t writelock;

      omp_init_lock(&writelock);

#pragma omp parallel
{
#endif

      double _Complex ** corr = init_2level_ztable ( T, ncorr );
      if ( corr == NULL ) {
        fprintf ( stderr, "[mixing_probe] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(2);
      }

      double _Complex  ** pm       = init_2level_ztable ( 12, 12 );
      double _Complex  ** pm2      = init_2level_ztable ( 12, 12 );
      double _Complex  ** g5sigmaG = init_2level_ztable ( 12, 12 );
      double _Complex  **  up      = init_2level_ztable ( 12, 12 );
      double _Complex  **  dn      = init_2level_ztable ( 12, 12 );
      double _Complex  ** dup      = init_2level_ztable ( 12, 12 );
      double _Complex  ** gdg      = init_2level_ztable ( 12, 12 );
      double _Complex *** aux      = init_3level_ztable ( 4, 12, 12 );
      double _Complex *** aux2     = init_3level_ztable ( 4, 12, 12 );

#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for ( unsigned int ix = 0; ix < VOLUME; ix++ )
      {
        

        int const it = ix / VOL3;

        unsigned int const iix = _GSI( ix );

        /***************************************************************************
         * build g5 sigma_munu G_munu
         * for the current spacetime point
         ***************************************************************************/
        memset ( g5sigmaG[0], 0, 144 * sizeof( double _Complex ) );

        memset ( pm[0], 0, 144 * sizeof( double _Complex ) );

        for (int i=0; i<6;i++ ) {

          double * const _gpm = Gpm[ix][i];

          for ( int j = 0; j<4; j++ ) {
          for ( int jc = 0; jc<3; jc++ ) {

            int const jj = 3 * j + jc;
          
            for ( int k = 0; k<4; k++ ) {
            for ( int kc = 0; kc<3; kc++ ) {
            
              int const kk = 3 * k + kc;
              pm[jj][kk] += ( _gpm[2*(3*jc+kc) ] + I * _gpm[2*(3*jc+kc)+1] ) * sigma_munu[i].m[j][k];
            }}
          }}
        }
        pm_eq_gamma_ti_pm ( g5sigmaG, &(gamma_p[0]), pm );

#if _TEST
        /* TEST */
        pm_print ( g5sigmaG, "g5sigmaG", ofs );
        /* END */
#endif


        /***************************************************************************
         * make the point-wise up and down propagator matrix
         ***************************************************************************/
        /* fill up from 12 spinor fields */
        pm_set_from_sf_point ( up, propagator[iflavor], ix );

        /* fill dn from 12 spinor fields */
        pm_set_from_sf_point ( dn, propagator[1-iflavor], ix );

        /* fill dup from 12 spinor fields */
        pm_set_from_sf_point ( dup, propagator_disp, ix );

#if _TEST
        /* TEST */
        pm_print ( up, "up", ofs );
        pm_print ( dn, "dn", ofs );
        pm_print ( dup, "dup", ofs );
        /* END */
#endif

        /***************************************************************************
         * auxilliary field
         * gdg <- g5 dn g5
         ***************************************************************************/
        pm_pl_eq_gamma_ti_pm_ti_gamma ( gdg, &(gamma_p[0]), dn, 0. );

#if _TEST
        /* TEST */
        pm_print ( gdg, "gdg", ofs );
        /* END */
#endif

        /***************************************************************************
         * auxialliary fields aux and aux2, which contain
         *
         * aux = up_fc * ( gc up_cc gc ) * ( g5 D^+ g5 )_fc
         ***************************************************************************/
        memset ( aux[0][0], 0, 576 * sizeof( double _Complex ) );

        for ( int i = 0; i < 4; i++ ) {
          /* pm <- ( gc up gc )  g5 dn^+ g5 */
          pm_eq_pm_ti_pm_dag ( pm, gug[i], gdg );
       
          /* aux <- up pm = up ( gc up gc ) [ g5 dn^+ g5 ] */
          pm_eq_pm_ti_pm ( aux[i], up, pm );
        }

        memset ( aux2[0][0], 0, 576 * sizeof( double _Complex ) );
        /* scalar */
        pm_eq_gamma_ti_pm_dag ( pm , &(gamma_s[0]), gdg );
        pm_eq_pm_ti_pm ( aux2[0], up, pm );
        pm_eq_pm_ti_co  ( aux2[0] , aux2[0], gu[0] );

        /* pseudoscalar */
        pm_eq_gamma_ti_pm_dag ( pm , &(gamma_p[0]), gdg );
        pm_eq_pm_ti_pm ( aux2[1], up, pm );
        pm_eq_pm_ti_co  ( aux2[1] , aux2[1], gu[1] );

        /* vector */
        for ( int i = 0; i < 4; i++ ) {
          /* pm <- gc ( g5 dn g5 )^+ */
          pm_eq_gamma_ti_pm_dag ( pm , &(gamma_v[i]), gdg );
          /* pm2 <- up pm */
          pm_eq_pm_ti_pm ( pm2, up, pm );
          /* aux2 <- (i>0) aux2 + gu * pm2 */
          pm_eq_pm_pl_pm ( aux2[2], aux2[2], pm2, (double _Complex)(i>0), gu[2+i] );
        }

        /* axial vector */
        for ( int i = 0; i < 4; i++ ) {
          /* pm <- gc ( g5 dn g5 )^+ */
          pm_eq_gamma_ti_pm_dag ( pm , &(gamma_a[i]), gdg );
          /* pm2 <- up pm */
          pm_eq_pm_ti_pm ( pm2, up, pm );
          /* aux2 <- (i>0) aux2 + gu * pm2 */
          pm_eq_pm_pl_pm ( aux2[3], aux2[3], pm2, (double _Complex)(i>0), gu[6+i] );
        }

#if _TEST
        /* TEST */
        pm_print ( aux[0], "u_sus_pDp", ofs );
        pm_print ( aux[1], "u_pup_pDp", ofs );
        pm_print ( aux[2], "u_vuv_pDp", ofs );
        pm_print ( aux[3], "u_aua_pDp", ofs );

        pm_print ( aux2[0], "tsu_u_s_pDp", ofs );
        pm_print ( aux2[1], "tpu_u_p_pDp", ofs );
        pm_print ( aux2[2], "tvu_u_v_pDp", ofs );
        pm_print ( aux2[3], "tau_u_a_pDp", ofs );
        /* END */
#endif

#if 0
        /***************************************************************************
         * finally, combine aux and aux2 into aux
         *
         * aux <- -2 aux + 2 aux2
         ***************************************************************************/
        for ( int i = 0; i < 4; i++ ) {
          pm_eq_pm_pl_pm ( aux[i], aux[i], aux2[i], -2., 2. );
        }
#endif

        /***************************************************************************
         * probing op at sink, 4q at source
         * Tr { Gamma_f aux } = Tr { Gamma_f up ( gc up gc ) [ g5 dn^+ g5 ] }
         ***************************************************************************/

        /***************************************************************************
         * dim-3 operator psibar g5 psi
         ***************************************************************************/

        /* pseudoscalar , B */
        for ( int i = 0; i < 4; i++ ) {
          pm_eq_gamma_ti_pm ( pm, &(gamma_p[0]), aux[i] );
          corr[it][i] += co_eq_tr_pm ( pm );
        }

        /* pseudoscalar , D */
        for ( int i = 0; i < 4; i++ ) {
          pm_eq_gamma_ti_pm ( pm, &(gamma_p[0]), aux2[i] );
          corr[it][4+i] += co_eq_tr_pm ( pm );
        }

        /***************************************************************************
         * dim-4 operator psibar id psi, to be multiplied with m_q
         ***************************************************************************/

        /* scalar, B */
        for ( int i = 0; i < 4; i++ ) {
          corr[it][8+i] += co_eq_tr_pm ( aux[i] );
        }
        /* scalar, D */
        for ( int i = 0; i < 4; i++ ) {
          corr[it][12+i] += co_eq_tr_pm ( aux2[i] );
        }

        /***************************************************************************
         * dim-4 operator psibar g_mu D_mu psi
         ***************************************************************************/

        /* gD, B */
        for ( int i = 0; i < 4; i++ ) {
          pm_eq_pm_ti_pm_dag ( pm, gug[i], gdg );

          pm_eq_pm_ti_pm ( pm2, dup, pm );

          corr[it][16+i] += co_eq_tr_pm ( pm2 );

        }

        /* gD, D */
        /* scalar */
        pm_eq_gamma_ti_pm_dag ( pm , &(gamma_s[0]), gdg );
        pm_eq_pm_ti_pm ( pm2, dup, pm );
        corr[it][20] += co_eq_tr_pm ( pm2 ) * gu[0];

        /* pseudoscalar */
        pm_eq_gamma_ti_pm_dag ( pm , &(gamma_p[0]), gdg );
        pm_eq_pm_ti_pm ( pm2, dup, pm );
        corr[it][21] += co_eq_tr_pm ( pm2 ) * gu[1];

        /* vector */
        for ( int i = 0; i < 4; i++ ) {
          /* pm <- gc ( g5 dn g5 )^+ */
          pm_eq_gamma_ti_pm_dag ( pm , &(gamma_v[i]), gdg );
          /* pm2 <- up pm */
          pm_eq_pm_ti_pm ( pm2, dup, pm );
          /* corr += Tr [ pm2 ] * gu */
          corr[it][22] += co_eq_tr_pm ( pm2 ) * gu[2+i];
        }

        /* axial vector */
        for ( int i = 0; i < 4; i++ ) {
          /* pm <- gc ( g5 dn g5 )^+ */
          pm_eq_gamma_ti_pm_dag ( pm , &(gamma_a[i]), gdg );
          /* pm2 <- up pm */
          pm_eq_pm_ti_pm ( pm2, dup, pm );
          /*  */
          corr[it][23] += co_eq_tr_pm ( pm2 ) * gu[6+i];
        }

        /***************************************************************************
         * dim-5 operator psibar g5 sigma_munu G_munu psi
         ***************************************************************************/

        /* g5sigmaG , B */
        for ( int i = 0; i < 4; i++ ) {
          pm_eq_pm_ti_pm ( pm, g5sigmaG, aux[i] );
          corr[it][24+i] += co_eq_tr_pm ( pm );
        }

        /* g5sigmaG , D */
        for ( int i = 0; i < 4; i++ ) {
          pm_eq_pm_ti_pm ( pm, g5sigmaG, aux2[i] );
          corr[it][28+i] += co_eq_tr_pm ( pm );
        }

        /***************************************************************************/
        /***************************************************************************/

        /***************************************************************************
         * pobing operator correlation functions at source and sink
         ***************************************************************************/

        /* Tr [ g5 U g5 g5 D^+ g5 ] */
        pm_eq_gamma_ti_pm_dag ( pm, &(gamma_p[0]), gdg );
        pm_eq_pm_ti_pm ( pm2, up, pm );
        pm_eq_gamma_ti_pm ( pm, &(gamma_p[0]), pm2 );
        corr[it][32] += co_eq_tr_pm ( pm );

        /* Tr [ Id U Id g5 D^+ g5 ] */
        pm_eq_pm_ti_pm_dag ( pm, up, gdg );
        corr[it][33] += co_eq_tr_pm ( pm );

        /* Tr [ g5 U g5 g5 U^+ g5 ] */
        pm_eq_pm_ti_pm_dag ( pm, up, up );
        corr[it][34] += co_eq_tr_pm ( pm );

      }  /* end of loop on x */

      fini_2level_ztable ( &pm       );
      fini_2level_ztable ( &pm2      );
      fini_2level_ztable ( &g5sigmaG );
      fini_2level_ztable ( &up       );
      fini_2level_ztable ( &dn       );
      fini_2level_ztable ( &dup      );
      fini_2level_ztable ( &gdg      );
      fini_3level_ztable ( &aux      );
      fini_3level_ztable ( &aux2     );

#ifdef HAVE_OPENMP
      omp_set_lock(&writelock);
#endif     
      for ( int i = 0; i < T * ncorr; i++ ) corr_accum[0][i] += corr[0][i];

#ifdef HAVE_OPENMP
      omp_unset_lock(&writelock);
#endif

      fini_2level_ztable ( &corr );

#ifdef HAVE_OPENMP
}  /* end of parallel region */

      omp_destroy_lock(&writelock);
#endif

      fini_2level_dtable ( &propagator_disp );
      fini_3level_ztable ( &gug );
      fini_1level_ztable ( &gu );

      double _Complex ** corr_out = init_2level_ztable ( T_global, ncorr );
      if ( corr_out == NULL ) {
        fprintf ( stderr, "[mixing_probe] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__ );
        EXIT(2);
      }

#ifdef HAVE_MPI
      /***************************************************************************
       * reduce within timeslice
       ***************************************************************************/
      int nitem = 2 * T * ncorr;
      double * buffer = init_1level_dtable ( nitem );

      exitstatus = MPI_Allreduce( (double*)(corr_accum[0]), buffer, nitem, MPI_DOUBLE, MPI_SUM, g_ts_comm );

      if(exitstatus != MPI_SUCCESS) {
        fprintf(stderr, "[mixing_probe] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(4);
      }

      /***************************************************************************
       * gather within time ray
       ***************************************************************************/
      exitstatus = MPI_Gather ( buffer, nitem, MPI_DOUBLE, (double*)(corr_out[0]), nitem, MPI_DOUBLE, 0, g_tr_comm);

      if(exitstatus != MPI_SUCCESS) {
        fprintf(stderr, "[mixing_probe] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(4);
      }

      fini_1level_dtable ( &buffer );
#else
      memcpy ( corr_out[0], corr_accum[0], T * ncorr * sizeof ( double _Complex ) );
#endif

      /***************************************************************************
       * write to file
       ***************************************************************************/
      if ( io_proc == 2 ) {
        char const diag_list[ncorr][4] = { "b", "d", "m1", "m2" };
        char const op_c_list[ncorr][2] = { "s", "p", "v", "a" };
        char const op_f_list[6][12] = { "g5", "m", "D", "g5sigmaG", "g5", "id" };

        double _Complex * buffer = init_1level_ztable ( T_global );

        for ( int iop = 0; iop < 4; iop++ ) {
          for ( int idiag = 0; idiag < 2; idiag++ ) {
            for ( int ic = 0; ic < 4; ic++ ) {
              int const corr_id =  4 * ( 2 * iop + idiag ) + ic;
              /* copy data */
              for ( int it = 0; it < T_global; it++ ) {
                buffer[it] = corr_out[it][corr_id];
              }
              char tag[400];
              sprintf ( tag, "/fl_%c/op_%s/d_%s/c_%s", flavor_tag[iflavor], op_f_list[iop], diag_list[idiag], op_c_list[ic] );
              exitstatus = write_aff_contraction ( buffer, affw, NULL, tag, T_global, "complex" );

              if ( exitstatus != 0 ) {
                fprintf(stderr, "[mixing_probe] Error from write_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(4);
              }

            }
          }
        }

        {
          for ( int i = 0; i< 2; i++ ) {

            int const corr_id = 32 + i;
            for ( int it = 0; it < T_global; it++ ) {
              buffer[it] = corr_out[it][corr_id];
            }
            char tag[400];
            sprintf ( tag, "/fl_%c/op_%s/d_%s/c_%s", flavor_tag[iflavor], op_f_list[4+i], diag_list[2], op_f_list[4+i] );
            exitstatus = write_aff_contraction ( buffer, affw, NULL, tag, T_global, "complex" );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[mixing_probe] Error from write_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(4);
            }
          }
          for ( int i = 0; i< 1; i++ ) {

            int const corr_id = 34 + i;
            for ( int it = 0; it < T_global; it++ ) {
              buffer[it] = corr_out[it][corr_id];
            }
            char tag[400];
            sprintf ( tag, "/fl_%c/op_%s/d_%s/c_%s", flavor_tag[iflavor], op_f_list[4], diag_list[3], op_f_list[4] );
            exitstatus = write_aff_contraction ( buffer, affw, NULL, tag, T_global, "complex" );
            if ( exitstatus != 0 ) {
              fprintf(stderr, "[mixing_probe] Error from write_aff_contraction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(4);
            }
          }
        }
        fini_1level_ztable ( &buffer );
      }  /* end of if io_proc == 2 */

      fini_2level_ztable ( &corr_out );

      fini_2level_ztable ( &corr_accum );

    }  /* end of loop on flavor */
    
#endif  /* end of if _PART_III  */


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
  
#if _TEST
  /* TEST */
  fclose ( ofs );
  /* END */
#endif

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
  ***************************************************************************/

  fini_3level_dtable ( &Gpm );

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
