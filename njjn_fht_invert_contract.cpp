/***************************************************************************
 *
 * njjn_fht_invert_contract
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

#define _PART_II  1  /* B/Z and D1c/i sequential diagrams */
#define _PART_III 1  /* W type sequential diagrams */

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
    fprintf(stderr, "[reduce_project_write] Error from reduce, status was %d\n", exitstatus);
    return( 1 );
  }

  /* (partial) Fourier transform, projection from position space to a (small) subset of momentum space */
  exitstatus = contract_vn_momentum_projection ( vp, vx, nd, momentum_list, momentum_number);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from contract_vn_momentum_projection, status was %d\n", exitstatus);
    return( 2 );
  }

  /* write to AFF file */
  exitstatus = contract_vn_write_aff ( vp, nd, affw, tag, momentum_list, momentum_number, io_proc );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[reduce_project_write] Error from contract_vn_write_aff, status was %d\n", exitstatus);
    return( 3 );
  }

  return ( 0 );

}  /* end of reduce_project_write */



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
  
  const char outfile_prefix[] = "njjn_fht";

  const char flavor_tag[4] = { 'u', 'd', 's', 'c' };

  const int sequential_gamma_sets = 4;

  int const sequential_gamma_num[4] = {4, 4, 1, 1};

  int const sequential_gamma_id[4][4] = {
    { 0,  1,  2,  3 },
    { 6,  7,  8,  9 },
    { 4, -1, -1, -1 },
    { 5, -1, -1, -1 } };
    
  char const sequential_gamma_tag[4][3] = { "vv", "aa", "ss", "pp" };

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
  char filename[100];
  double **lmzz[2] = { NULL, NULL }, **lmzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  double *gauge_field_smeared = NULL;

  int const    gamma_f1_number                           = 4;
  int const    gamma_f1_list[gamma_f1_number]            = { 14 , 11,  8,  2 };
  double const gamma_f1_sign[gamma_f1_number]            = { +1 , +1, -1, -1 };

  /*
  int const    gamma_f1_number                           = 1;
  int const    gamma_f1_list[gamma_f1_number]            = { 14 };
  double const gamma_f1_sign[gamma_f1_number]            = { +1 };
  */
  int read_loop_field = 0;
  char read_loop_filename[400];

#ifdef HAVE_LHPC_AFF
  struct AffWriter_s *affw = NULL;
  char aff_tag[400];
#endif

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "ch?f:l:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'c':
      check_propagator_residual = 1;
      break;
    case 'l':
      read_loop_field = 1;
      strcpy ( read_loop_filename, optarg );
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  g_the_time = time(NULL);

  /***************************************************************************
   * read input and set the default values
   ***************************************************************************/
  if(filename_set==0) strcpy(filename, "twopt.input");
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [njjn_fht_invert_contract] calling tmLQCD wrapper init functions\n");

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
    fprintf(stdout, "# [njjn_fht_invert_contract] git version = %s\n", g_gitversion);
  }

  /***************************************************************************
   * set number of openmp threads
   *
   *   each process and thread reports
   ***************************************************************************/
#ifdef HAVE_OPENMP
  if(g_cart_id == 0) fprintf(stdout, "# [njjn_fht_invert_contract] setting omp number of threads to %d\n", g_num_threads);
  omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [njjn_fht_invert_contract] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[njjn_fht_invert_contract] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[njjn_fht_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
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
    if(g_cart_id==0) fprintf(stdout, "# [njjn_fht_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [njjn_fht_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[njjn_fht_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[njjn_fht_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif

  /***********************************************************
   * multiply the phase to the gauge field
   ***********************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[njjn_fht_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***********************************************
   * if we want to use Jacobi smearing, we need 
   * smeared gauge field
   ***********************************************/
  if( N_Jacobi > 0 ) {

    alloc_gauge_field ( &gauge_field_smeared, VOLUMEPLUSRAND);

    memcpy ( gauge_field_smeared, g_gauge_field, 72*VOLUME*sizeof(double));

    if ( N_ape > 0 ) {
      exitstatus = APE_Smearing(gauge_field_smeared, alpha_ape, N_ape);
      if(exitstatus != 0) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
    }  /* end of if N_aoe > 0 */
  }  /* end of if N_Jacobi > 0 */

  /***************************************************************************
   * initialize the clover term, 
   * lmzz and lmzzinv
   *
   *   mzz = space-time diagonal part of the Dirac matrix
   *   l   = light quark mass
   ***************************************************************************/
  exitstatus = init_clover ( &lmzz, &lmzzinv, gauge_field_with_phase );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[njjn_fht_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[njjn_fht_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [njjn_fht_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***************************************************************************
   * set the gamma matrices
   ***************************************************************************/
  gamma_matrix_type sequential_gamma_list[4][4];
  /* vector */
  gamma_matrix_set( &( sequential_gamma_list[0][0] ), 0, 1. );  /*  gamma_0 = gamma_t */
  gamma_matrix_set( &( sequential_gamma_list[0][1] ), 1, 1. );  /*  gamma_1 = gamma_x */
  gamma_matrix_set( &( sequential_gamma_list[0][2] ), 2, 1. );  /*  gamma_2 = gamma_y */
  gamma_matrix_set( &( sequential_gamma_list[0][3] ), 3, 1. );  /*  gamma_3 = gamma_z */
  /* pseudovector */
  gamma_matrix_set( &( sequential_gamma_list[1][0] ), 6, 1. );  /*  gamma_6 = gamma_5 gamma_t */
  gamma_matrix_set( &( sequential_gamma_list[1][1] ), 7, 1. );  /*  gamma_7 = gamma_5 gamma_x */
  gamma_matrix_set( &( sequential_gamma_list[1][2] ), 8, 1. );  /*  gamma_8 = gamma_5 gamma_y */
  gamma_matrix_set( &( sequential_gamma_list[1][3] ), 9, 1. );  /*  gamma_9 = gamma_5 gamma_z */
  /* scalar */
  gamma_matrix_set( &( sequential_gamma_list[2][0] ), 4, 1. );  /*  gamma_4 = id */
  /* pseudoscalar */
  gamma_matrix_set( &( sequential_gamma_list[3][0] ), 5, 1. );  /*  gamma_5 */

  gamma_matrix_type gammafive;
  gamma_matrix_set( &gammafive, 5, 1. );  /*  gamma_5 */


  /***************************************************************************
   * prepare the Fourier phase field
   ***************************************************************************/
  unsigned int const VOL3 = LX * LY * LZ;
  size_t const sizeof_spinor_field = _GSI( VOLUME ) * sizeof( double );

  double _Complex ** ephase = NULL;
  if ( g_seq_source_momentum_number > 0 ) {
    ephase = init_2level_ztable ( g_seq_source_momentum_number, VOL3 );
    if ( ephase == NULL ) {
      fprintf ( stderr, "[njjn_fht_invert_contract] Error from init_2level_ztable %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }

    make_phase_field_timeslice ( ephase, g_seq_source_momentum_number, g_seq_source_momentum_list );
  }  /* end of if g_seq_source_momentum_number > 0 */

  /***************************************************************************
   * init rng state
   ***************************************************************************/
  exitstatus = init_rng_state ( g_seed, &g_rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[njjn_fht_invert_contract] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  /***************************************************************************
   ***************************************************************************
   **
   ** Part I
   **
   ** prepare stochastic sources and propagators
   ** to contract the loop for insertion as part of
   ** sequential source
   **
   ***************************************************************************
   ***************************************************************************/

  double _Complex *** loop = init_3level_ztable ( VOLUME, 12, 12 );
  if ( loop  == NULL ) {
    fprintf ( stderr, "[njjn_fht_invert_contract] Error from init_Xlevel_ztable %s %d\n", __FILE__, __LINE__ );
    EXIT(12);
  }


  if ( ! read_loop_field ) {

    if ( g_cart_id == 0 ) {
      fprintf ( stdout, "# [njjn_fht_invert_contract] produce loop field %s %d\n",  __FILE__, __LINE__ );
    }

    /***************************************************************************
     * loop on samples
     * invert and contract loops
     ***************************************************************************/
    for ( int isample = 0; isample < g_nsample; isample++ ) {

      double * stochastic_source = init_1level_dtable ( _GSI( VOLUME ) );
      if ( stochastic_source == NULL ) {
        fprintf ( stderr, "[njjn_fht_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(12);
      }
 
      double ** spinor_work = init_2level_dtable ( 2, _GSI( VOLUME+RAND ) );
      if ( spinor_work == NULL ) {
        fprintf ( stderr, "[njjn_fht_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
        EXIT(12);
      }

      exitstatus = prepare_volume_source ( stochastic_source, VOLUME );
      if ( exitstatus != 0 ) {
        fprintf ( stderr, "[njjn_fht_invert_contract] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
        EXIT(12);
      }

      /***************************************************************************
       * loop on spin and color index 
       *
       * we use spin-color dilution here:
       *   for each inversion we select only a single spin-color component 
       *   from stochastic_source
       ***************************************************************************/
      for ( int ispin = 0; ispin < 4; ispin++ ) {
        for ( int icol = 0; icol < 3; icol++ ) {

          int const isc = 3 * ispin + icol;

	  memset ( spinor_work[0], 0, sizeof_spinor_field );

#pragma omp parallel for
          for ( unsigned int ix = 0; ix < VOLUME; ix++  ) {
            unsigned int const iy = _GSI(ix) + 2 * isc;  /* offset for site ix and spin-color isc */
            spinor_work[0][ iy    ] = stochastic_source[ iy     ];
            spinor_work[0][ iy + 1] = stochastic_source[ iy + 1 ];
          }

          /* tm-rotate stochastic propagator at source, in-place */
          if( g_fermion_type == _TM_FERMION ) {
            spinor_field_tm_rotation(spinor_work[0], spinor_work[0], 1, g_fermion_type, VOLUME);
          }

          /* call to (external/dummy) inverter / solver */
          exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], _OP_ID_UP );
          if(exitstatus != 0) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from tmLQCD_invert, status was %d\n", exitstatus);
            EXIT(12);
          }

          if ( check_propagator_residual ) {
            check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, lmzz[_OP_ID_UP], 1 );
          }

          /* tm-rotate stochastic propagator at sink */
          if( g_fermion_type == _TM_FERMION ) {
            spinor_field_tm_rotation(spinor_work[1], spinor_work[1], 1, g_fermion_type, VOLUME);
          }

          /***************************************************************************
           * fill in loop matrix element ksc = (kspin, kcol ), isc = (ispin, icol )
           * as
           * loop[ksc][isc] = prop[ksc] * source[isc]^+
           ***************************************************************************/
#pragma omp parallel for
          for ( unsigned int ix = 0; ix < VOLUME; ix++  ) {
            unsigned int const iy = _GSI( ix );
            for ( int kspin = 0; kspin < 4; kspin++ ) {
              for ( int kcol = 0; kcol < 3; kcol++ ) {
                int const ksc = 3 * kspin + kcol;
 
                loop[ix][ksc][isc] +=
                    /* complex conjugate of source vector element */
                    ( stochastic_source[ iy + 2*isc  ] - I * stochastic_source[ iy + 2*isc+1] )
                    /* times prop vector element */
                  * ( spinor_work[1][    iy + 2*ksc  ] + I * spinor_work[1][    iy + 2*ksc+1] );
              }
            }
          }
        }  /* end of loop on color dilution component */
      }  /* end of loop on spin dilution component */

      /* free fields */
      fini_1level_dtable ( &stochastic_source );
      fini_2level_dtable ( &spinor_work );
    }  /* end of loop on samples */

    /***************************************************************************
     * normalize
     ***************************************************************************/
#pragma omp parallel for
    for ( unsigned int ix = 0; ix < 144 * VOLUME; ix++  ) {
      loop[0][0][ix] /= (double)g_nsample;
    }

  } else {

    /***************************************************************************
     * read loop field from file
     * for testing
     ***************************************************************************/

    if ( g_cart_id == 0 ) {
      fprintf ( stdout, "# [njjn_fht_invert_contract] reading loop field from file %s %s %d\n", read_loop_filename,  __FILE__, __LINE__ );
    }
#ifndef HAVE_MPI
    FILE * lfs = fopen ( read_loop_filename, "r" );
    if ( lfs == NULL ) {
      fprintf ( stderr, "[njjn_fht_invert_contract] Error from fopen %s %d\n", __FILE__, __LINE__ );
      EXIT(12);
    }

    /***************************************************************************
     * !!! THIS IS TEMPORARY; ADAPT TO WHAT IS NEEDED !!!
     ***************************************************************************/

    /* for ( unsigned int ix = 0; ix < VOLUME; ix++ ) */
    for ( unsigned int ix = 0; ix < 1; ix++ )
    {
      double dtmp;
      for ( int i = 0; i< 12; i++ ) {
        for ( int k = 0; k< 12; k++ ) {
          if ( fscanf ( lfs, "%lf ", &dtmp ) != 1 ) {
            fprintf ( stderr, "[njjn_fht_invert_contract] Error from fscanf %s %d\n", __FILE__, __LINE__ );
            EXIT(12);
          }
          loop[ix][i][k] = dtmp;
        }
        fscanf ( lfs, "\n" );
      }
      /* TEST */
      for ( int i = 0; i< 12; i++ ) {
        for ( int k = 0; k< 12; k++ ) {
          fprintf ( stdout, "# [njjn_fht_invert_contract] loop[0][%2d][%2d] = %25.16e + I %25.16e\n", i, k, creal( loop[ix][i][k] ), cimag( loop[ix][i][k] ) );
        }
      }
      /* END OF TEST */
    }
    fclose ( lfs );

    for ( unsigned int ix = 1; ix < VOLUME; ix++ ) {
      memcpy ( loop[ix][0], loop[0][0], 144 * sizeof ( double _Complex ) );
    }

#endif  /* end of if HAVE_MPI not defined */
  }  /* end of if on read stoch. source  */

  /***************************************************************************
   *
   * point-to-all version
   *
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
     * allocate point-to-all propagators,
     * spin-color dilution (i.e. 12 fields per flavor of size 24xVOLUME real )
     ***************************************************************************/

    /* up and down quark propagator with source smearing */
    double *** propagator = init_3level_dtable ( 2, 12, _GSI( VOLUME ) );
    if( propagator == NULL ) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

    /* up and down quark propagator with source and sink smearing,
     * to use for baryon 2-pt function 
     */
    double *** propagator_snk_smeared = init_3level_dtable ( 2, 12, _GSI( VOLUME ) );
    if ( propagator_snk_smeared == NULL ) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }

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
      fprintf(stderr, "[njjn_fht_invert_contract] Error from get_point_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

    /***************************************************************************
     * open output file reader
     * we use the AFF format here
     * https://github.com/usqcd-software/aff
     *
     * one data file per source position
     ***************************************************************************/
#if ( defined HAVE_LHPC_AFF )
    /***************************************************************************
     * writer for aff output file
     * only I/O process id 2 opens a writer
     ***************************************************************************/
    if(io_proc == 2) {
      sprintf(filename, "%s.%.4d.t%dx%dy%dz%d.aff", outfile_prefix, Nconf, gsx[0], gsx[1], gsx[2], gsx[3]);
      fprintf(stdout, "# [njjn_fht_invert_contract] writing data to file %s\n", filename);
      affw = aff_writer(filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#else
    fprintf(stderr, "[njjn_fht_invert_contract] Error, no AFF lib %s %d\n",  __FILE__, __LINE__);
    EXIT(15);
#endif

    /**********************************************************
     *
     * point-to-all propagators with source at gsx
     *
     **********************************************************/

    for ( int iflavor = 0; iflavor < 2; iflavor++ ) 
    {

      /***********************************************************
       * flavor-type point-to-all propagator
       *
       * ONLY SOURCE smearing here
       *
       * NOTE: quark flavor is controlled by value of iflavor
       ***********************************************************/
      /*                                     output field         src coords flavor type  src smear  sink smear gauge field for smearing,  for residual check ...                                   */
      exitstatus = point_source_propagator ( propagator[iflavor], gsx,       iflavor,     1,         0,         gauge_field_smeared,       check_propagator_residual, gauge_field_with_phase, lmzz );
      if(exitstatus != 0) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from point_source_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(12);
      }

      /***********************************************************
       * sink-smear the flavor-type point-to-all propagator
       * store extra
       ***********************************************************/
      for ( int i = 0; i < 12; i++ ) {
        /* copy propagator */
        memcpy ( propagator_snk_smeared[iflavor][i], propagator[iflavor][i], sizeof_spinor_field );

        /* sink-smear propagator */
        exitstatus = Jacobi_Smearing ( gauge_field_smeared, propagator_snk_smeared[iflavor][i], N_Jacobi, kappa_Jacobi);
        if(exitstatus != 0) {
          fprintf(stderr, "[njjn_fht_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          return(11);
        }
      }

      /***********************************************************
       * optionally write the propagator to disc
       *
       * we use the standard lime format here
       * https://github.com/usqcd-software/c-lime
       ***********************************************************/
      if ( g_write_propagator ) {
        /* each spin-color component into a separate file */
        for ( int i = 0; i < 12; i++ ) {
          sprintf ( filename, "propagator_%c.%.4d.t%dx%dy%dz%d.%d.inverted", flavor_tag[iflavor], Nconf, gsx[0], gsx[1], gsx[2], gsx[3] , i );

          if ( ( exitstatus = write_propagator( propagator[iflavor][i], filename, 0, g_propagator_precision) ) != 0 ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(2);
          }
        }
      }

    }  /* end of loop on flavor */


    /***************************************************************************
     * allocate propagator fields
     *
     * these are ordered as as
     * t,x,y,z,spin,color,spin,color
     * so a 12x12 complex matrix per space-time point
     ***************************************************************************/
    fermion_propagator_type * fp  = create_fp_field ( VOLUME );
    fermion_propagator_type * fp2 = create_fp_field ( VOLUME );
    fermion_propagator_type * fp3 = create_fp_field ( VOLUME );

    /***************************************************************************
     * vx holds the x-dependent nucleon-nucleon spin propagator,
     * i.e. a 4x4 complex matrix per space time point
     ***************************************************************************/
    double ** vx = init_2level_dtable ( VOLUME, 32 );
    if ( vx == NULL ) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from init_2level_dtable, %s %d\n", __FILE__, __LINE__);
      EXIT(47);
    }

    /***************************************************************************
     * vp holds the nucleon-nucleon spin propagator in momentum space,
     * i.e. the momentum projected vx
     ***************************************************************************/
    double *** vp = init_3level_dtable ( T, g_sink_momentum_number, 32 );
    if ( vp == NULL ) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from init_3level_dtable %s %d\n", __FILE__, __LINE__ );
      EXIT(47);
    }

    /***************************************************************************
     ***************************************************************************
     **
     ** Part II
     **
     ** fwd propagator contractions for baryon 2pts
     **
     ***************************************************************************
     ***************************************************************************/

    /***************************************************************************
     * loop on flavor combinations
     ***************************************************************************/
    for ( int iflavor = 0; iflavor < 2; iflavor++ ) {

      /***************************************************************************
       *
       * [ X^T Xb X ] - [ Xb^+ X^* X^+ ]
       *
       ***************************************************************************/

      char  aff_tag_prefix[100];
      sprintf ( aff_tag_prefix, "/N-N/%c%c%c/T%d_X%d_Y%d_Z%d", flavor_tag[iflavor], flavor_tag[1-iflavor], flavor_tag[iflavor], gsx[0], gsx[1], gsx[2], gsx[3] );
         
      /***************************************************************************
       * fill the fermion propagator fp with the 12 spinor fields
       * in propagator of flavor X
       ***************************************************************************/
      assign_fermion_propagator_from_spinor_field ( fp, propagator_snk_smeared[iflavor], VOLUME);

      /***************************************************************************
       * fill fp2 with 12 spinor fields from propagator of flavor Xb
       ***************************************************************************/
      assign_fermion_propagator_from_spinor_field ( fp2, propagator_snk_smeared[1-iflavor], VOLUME);

      /***************************************************************************
       * contractions for n1, n2
       *
       * if1/2 loop over various Dirac Gamma-structures for
       * baryon interpolators at source and sink
       ***************************************************************************/
      for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {
      for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {

        /***************************************************************************
         * here we calculate fp3 = Gamma[if2] x propagator[1-iflavor] / fp2 x Gamma[if1]
         ***************************************************************************/
        fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp2, VOLUME );

        fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_list[if1], fp3, VOLUME );

        fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, -gamma_f1_sign[if1]*gamma_f1_sign[if2], VOLUME );

        /***************************************************************************
         * diagram n1
         ***************************************************************************/
        sprintf(aff_tag, "%s/Gi_%s/Gf_%s/t1", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ]);

        exitstatus = reduce_project_write ( vx, vp, fp, fp3, fp, contract_v5, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[njjn_fht_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(48);
        }

        /***************************************************************************
         * diagram n2
         ***************************************************************************/
        sprintf(aff_tag, "%s/Gi_%s/Gf_%s/t2", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ]);

        exitstatus = reduce_project_write ( vx, vp, fp, fp3, fp, contract_v6, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[njjn_fht_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(48);
        }

      }}  /* end of loop on Dirac Gamma structures */

    }  /* end of loop on flavor */
    
    /***************************************************************************
     *
     * sequential inversion and contraction
     *
     ***************************************************************************/

    /***************************************************************************
     * loop on sequential source momenta
     ***************************************************************************/
    for ( int imom = 0; imom < g_seq_source_momentum_number; imom++ ) {

      int momentum[3] = {
          g_seq_source_momentum_list[imom][0],
          g_seq_source_momentum_list[imom][1],
          g_seq_source_momentum_list[imom][2] };

      if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [njjn_fht_invert_contract] start seq source mom %3d %3d %3d  %s %d\n", 
          momentum[0], momentum[1], momentum[2], __FILE__, __LINE__);

#if _PART_II
      /***************************************************************************
       * loop on flavor
       ***************************************************************************/
      for ( int iflavor = 0; iflavor < 2; iflavor++ )
      {

        if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [njjn_fht_invert_contract] start seq source flavor %3d   %s %d\n", 
            iflavor, __FILE__, __LINE__);

        /***************************************************************************
         ***************************************************************************
         **
         ** Part III
         **
         ** sequential inversion with loop-product sequential sources
         ** and contractions for N - qbar q qbar q - N B,Z,D_1c/i diagrams
         **
         ***************************************************************************
         ***************************************************************************/

        /***************************************************************************
         * loop on 2 types of sequential fht sources
         ***************************************************************************/
        for ( int seq_source_type = 0; seq_source_type <= 1; seq_source_type++ )
        {

          if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [njjn_fht_invert_contract] start seq source type %3d   %s %d\n", 
             seq_source_type, __FILE__, __LINE__);

          /***************************************************************************
           * allocate for sequential propagator and source
           ***************************************************************************/
          double ** sequential_propagator = init_2level_dtable ( 12, _GSI( VOLUME ) );
          if( sequential_propagator == NULL ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(123);
          }

          double ** sequential_source = init_2level_dtable ( 12,  _GSI(VOLUME) );
          if( sequential_source == NULL ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(132);
          }

          char const sequential_propagator_name = ( seq_source_type == 0 ) ? 'b' : 'd';


          /***************************************************************************
           * loop on sequential source gamma matrices
           ***************************************************************************/
          for ( int igamma = 0; igamma < sequential_gamma_sets; igamma++ ) {

            if ( g_cart_id == 0 && g_verbose > 2 ) fprintf(stdout, "# [njjn_fht_invert_contract] start seq source gamma set %s   %s %d\n", 
               sequential_gamma_tag[igamma], __FILE__, __LINE__);

            /***************************************************************************
             * add sequential fht vertex
             ***************************************************************************/
            exitstatus = prepare_sequential_fht_loop_source ( sequential_source, loop, propagator[iflavor], sequential_gamma_list[igamma], sequential_gamma_num[igamma], ephase[imom], seq_source_type, ( iflavor == 0 ? NULL : &gammafive ) );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[njjn_fht_invert_contract] Error from prepare_sequential_fht_loop_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(123);
            }

            if ( g_write_sequential_source ) {
              for ( int i = 0; i < 12; i++ ) {
                sprintf ( filename, "sequential_source_%c.%.4d.t%dx%dy%dz%d.px%dpy%dpz%d.%s.type%d.%d", flavor_tag[iflavor], Nconf, gsx[0], gsx[1], gsx[2], gsx[3],
                    momentum[0], momentum[1], momentum[2], sequential_gamma_tag[igamma], seq_source_type, i );

                if ( ( exitstatus = write_propagator( sequential_source[i], filename, 0, g_propagator_precision) ) != 0 ) {
                  fprintf(stderr, "[njjn_fht_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(2);
                }
              }
            }
  
            /***************************************************************************
             * invert the Dirac operator on the sequential source
             *
             * ONLY SINK smearing here
             ***************************************************************************/
            exitstatus = prepare_propagator_from_source ( sequential_propagator, sequential_source, 12, iflavor, 0, 1, gauge_field_smeared,
                check_propagator_residual, gauge_field_with_phase, lmzz, NULL );
            if ( exitstatus != 0 ) {
              fprintf ( stderr, "[njjn_fht_invert_contract] Error from prepare_propagator_from_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              EXIT(123);
            }


            if ( g_write_sequential_propagator ) {
              for ( int i = 0; i < 12; i++ ) {
                sprintf ( filename, "sequential_source_%c.%.4d.t%dx%dy%dz%d.px%dpy%dpz%d.%s.type%d.%d.inverted", flavor_tag[iflavor], Nconf, gsx[0], gsx[1], gsx[2], gsx[3],
                    momentum[0], momentum[1], momentum[2], sequential_gamma_tag[igamma], seq_source_type, i );

                if ( ( exitstatus = write_propagator( sequential_propagator[i], filename, 0, g_propagator_precision) ) != 0 ) {
                  fprintf(stderr, "[njjn_fht_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                  EXIT(2);
                }
              }
            }
  
            /***************************************************************************
             *
             * contractions
             *
             ***************************************************************************/
  
            char correlator_tag[20] = "N-qbGqqbGq-N";
            
            char aff_tag_prefix[200], aff_tag_prefix2[200];

            sprintf ( aff_tag_prefix, "/%s/%c%c%c%c-f%c-f%c/T%d_X%d_Y%d_Z%d/QX%d_QY%d_QZ%d/nsample%d/Gc_%s",
                    correlator_tag, 
                    sequential_propagator_name, flavor_tag[iflavor], flavor_tag[iflavor], flavor_tag[iflavor],
                    flavor_tag[1-iflavor],
                    flavor_tag[iflavor],
                    gsx[0], gsx[1], gsx[2], gsx[3],
                    momentum[0], momentum[1], momentum[2],
                    g_nsample,
                    sequential_gamma_tag[igamma] );

            sprintf ( aff_tag_prefix2, "/%s/f%c-f%c-%c%c%c%c/T%d_X%d_Y%d_Z%d/QX%d_QY%d_QZ%d/nsample%d/Gc_%s",
                    correlator_tag,
                    flavor_tag[iflavor],
                    flavor_tag[1-iflavor],
                    sequential_propagator_name, flavor_tag[iflavor], flavor_tag[iflavor], flavor_tag[iflavor],
                    gsx[0], gsx[1], gsx[2], gsx[3],
                    momentum[0], momentum[1], momentum[2],
                    g_nsample,
                    sequential_gamma_tag[igamma] );

            /***************************************************************************
             * B/D1c/i for uu uu insertion
             ***************************************************************************/
         
            /***************************************************************************
             * fp = fwd up
             ***************************************************************************/
            assign_fermion_propagator_from_spinor_field ( fp, propagator_snk_smeared[iflavor], VOLUME);

            /***************************************************************************
             * fp2 = b up-after-up-after-up 
             ***************************************************************************/
            assign_fermion_propagator_from_spinor_field ( fp2, sequential_propagator, VOLUME);
  
            /***************************************************************************
             * contractions as for t1,...,t4 of N-N type diagrams
             ***************************************************************************/
            for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {
            for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {
      
              /***************************************************************************
               * fp3 = fwd dn
               ***************************************************************************/
              assign_fermion_propagator_from_spinor_field ( fp3, propagator_snk_smeared[1-iflavor], VOLUME);

              /***************************************************************************
               * fp3 <- Gamma[if2] x fwd dn x Gamma[if1]
               * in-place
               ***************************************************************************/
              fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp3, VOLUME );
       
              fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_list[if1], fp3, VOLUME );
      
              fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, -gamma_f1_sign[if1]*gamma_f1_sign[if2], VOLUME );
      
              /***************************************************************************
               * diagram t1
               ***************************************************************************/
              sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t1", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
 
              exitstatus = reduce_project_write ( vx, vp, fp2, fp3, fp, contract_v5, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[njjn_fht_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(48);
              }
     
              /***************************************************************************
               * diagram t2
               ***************************************************************************/
              sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t2", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
      
              exitstatus = reduce_project_write ( vx, vp, fp2, fp3, fp, contract_v6, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[njjn_fht_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(48);
              }

              /***************************************************************************
               * diagram t1
               ***************************************************************************/
              sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t1", aff_tag_prefix2, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
     
              exitstatus = reduce_project_write ( vx, vp, fp, fp3, fp2, contract_v5, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[njjn_fht_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(48);
              }

              /***************************************************************************
               * diagram t2
               ***************************************************************************/
              sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t2", aff_tag_prefix2, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
  
              exitstatus = reduce_project_write ( vx, vp, fp, fp3, fp2, contract_v6, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[njjn_fht_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(48);
              }

            }} // end of loop on Dirac gamma structures
  
            /***************************************************************************/
            /***************************************************************************/
  
            /***************************************************************************
             * B/D1ci for dd dd insertion
             ***************************************************************************/

            sprintf ( aff_tag_prefix, "/%s/f%c-%c%c%c%c-f%c/T%d_X%d_Y%d_Z%d/QX%d_QY%d_QZ%d/nsample%d/Gc_%s",
                    correlator_tag,
                    flavor_tag[1-iflavor], 
                    sequential_propagator_name, flavor_tag[iflavor], flavor_tag[iflavor], flavor_tag[iflavor],
                    flavor_tag[1-iflavor],
                    gsx[0], gsx[1], gsx[2], gsx[3],
                    momentum[0], momentum[1], momentum[2],
                    g_nsample,
                    sequential_gamma_tag[igamma] );
  
            /***************************************************************************
             * fp = fwd dn
             ***************************************************************************/
            assign_fermion_propagator_from_spinor_field ( fp, propagator_snk_smeared[1-iflavor], VOLUME);
  
            /***************************************************************************
             * fp2 = b up up up
             ***************************************************************************/
            assign_fermion_propagator_from_spinor_field ( fp2, sequential_propagator, VOLUME);
  
            /***************************************************************************
             *
             ***************************************************************************/
            for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {
            for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {
      
              /***************************************************************************
               * calculate fp3 = Gamma[if2] x propagator_dn / fp3 x Gamma[if1]
               ***************************************************************************/
              fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp2, VOLUME );
        
              fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_list[if1], fp3, VOLUME );
        
              fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, -gamma_f1_sign[if1]*gamma_f1_sign[if2], VOLUME );
      
              /***************************************************************************
               * diagram t1
               ***************************************************************************/
              sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t1", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
       
              exitstatus = reduce_project_write ( vx, vp, fp, fp3, fp, contract_v5, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[njjn_fht_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(48);
              }

              /***************************************************************************
               * diagram t2
               ***************************************************************************/
              sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t2", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );

              exitstatus = reduce_project_write ( vx, vp, fp, fp3, fp, contract_v6, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
              if ( exitstatus != 0 ) {
                fprintf(stderr, "[njjn_fht_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(48);
              }

            }} // end of loop on Dirac gamma structures
  
            /***************************************************************************/
            /***************************************************************************/
  
          } // end of loop on sequential source gamma matrices
      
          fini_2level_dtable ( &sequential_source );
          fini_2level_dtable ( &sequential_propagator );

        }  /* end of loop on seq. source type */

      }  /* loop on flavor type */
#endif  /* of if _PART_II */

      /***************************************************************************/
      /***************************************************************************/

#if _PART_III
      /***************************************************************************
       ***************************************************************************
       **
       ** Part IV
       **
       ** sequential inversion with binary noise insertion for twin-peak diagram
       **
       ***************************************************************************
       ***************************************************************************/

      for ( int isample = 0; isample < g_nsample_oet; isample++ ) {

          /***************************************************************************
           * allocate for sequential propagator and source
           ***************************************************************************/
          double *** sequential_propagator = init_3level_dtable ( 2, 12, _GSI( VOLUME ) );
          if( sequential_propagator == NULL ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(123);
          }

          double ** sequential_source = init_2level_dtable ( 12,  _GSI(VOLUME) );
          if( sequential_source == NULL ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(132);
          }

          double * scalar_field = init_1level_dtable ( VOLUME );
          if( scalar_field == NULL ) {
            fprintf(stderr, "[njjn_fht_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(132);
          }

          /***************************************************************************
           * draw a stochastic binary source (real, +/1 one per site )
           ***************************************************************************/
          ranbinaryd ( scalar_field, VOLUME );

          /***************************************************************************
           * loop on sequential source gamma matrices
           ***************************************************************************/
          for ( int igamma = 0; igamma < sequential_gamma_sets; igamma++ ) 
          {
          
            for ( int ig = 0; ig < sequential_gamma_num[igamma]; ig++ ) {

              for ( int iflavor = 0; iflavor < 2; iflavor++ ) 
              {

                /***************************************************************************
                 * sequential source for "twin-peak" diagram, 
                 * needed as both up-after-up and down-after-down type sequential propagator
                 ***************************************************************************/
                exitstatus =  prepare_sequential_fht_twinpeak_source ( sequential_source, propagator[iflavor], scalar_field, sequential_gamma_id[igamma][ig], ephase[imom] ) ;
                if ( exitstatus != 0 ) {
                  fprintf ( stderr, "[njjn_fht_invert_contract] Error from prepare_sequential_fht_twinpeak_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(123);
                }

                if ( g_write_sequential_source ) {
                  for ( int i = 0; i < 12; i++ ) {
                    sprintf ( filename, "sequential_source_%c.%.4d.t%dx%dy%dz%d.px%dpy%dpz%d.%s.type%d.%d", flavor_tag[iflavor], Nconf, gsx[0], gsx[1], gsx[2], gsx[3],
                    momentum[0], momentum[1], momentum[2], gamma_id_to_ascii[sequential_gamma_id[igamma][ig]], 2, i );

                    if ( ( exitstatus = write_propagator( sequential_source[i], filename, 0, g_propagator_precision) ) != 0 ) {
                      fprintf(stderr, "[njjn_fht_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(2);
                    }
                  }
                }

                /***************************************************************************
                 * seq. prop. from seq. source
                 *
                 * INCLUDING SINK-SMEARING
                 ***************************************************************************/
                exitstatus = prepare_propagator_from_source ( sequential_propagator[iflavor], sequential_source, 12, iflavor, 0, 1, gauge_field_smeared,
                    check_propagator_residual, gauge_field_with_phase, lmzz, NULL );
                if ( exitstatus != 0 ) {
                  fprintf ( stderr, "[njjn_fht_invert_contract] Error from prepare_propagator_from_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
                  EXIT(123);
                }

                if ( g_write_sequential_propagator ) {
                  for ( int i = 0; i < 12; i++ ) {
                    sprintf ( filename, "sequential_source_%c.%.4d.t%dx%dy%dz%d.px%dpy%dpz%d.%s.type%d.%d.inverted", flavor_tag[iflavor], Nconf, gsx[0], gsx[1], gsx[2], gsx[3],
                    momentum[0], momentum[1], momentum[2], gamma_id_to_ascii[sequential_gamma_id[igamma][ig]], 2, i );

                    if ( ( exitstatus = write_propagator( sequential_propagator[iflavor][i], filename, 0, g_propagator_precision) ) != 0 ) {
                      fprintf(stderr, "[njjn_fht_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                      EXIT(2);
                    }
                  }
                }

              }  /* end of loop on quark propagator flavor */

              /***************************************************************************
               *
               * contractions for W-type diagrams
               *
               ***************************************************************************/
              char correlator_tag[20] = "N-qbGqqbGq-N";

              /***************************************************************************
               *  W for  uu uu type insertion
               ***************************************************************************/
              for ( int iflavor = 0; iflavor < 2; iflavor++ ) {

                char aff_tag_prefix[200];
                sprintf ( aff_tag_prefix, "/%s/w%c%c-f%c-w%c%c/T%d_X%d_Y%d_Z%d/QX%d_QY%d_QZ%d/sample%d/Gc_%s",
                    correlator_tag, 
                    flavor_tag[iflavor], flavor_tag[iflavor], flavor_tag[1-iflavor], 
                    flavor_tag[iflavor], flavor_tag[iflavor],
                    gsx[0], gsx[1], gsx[2], gsx[3], 
                    momentum[0], momentum[1], momentum[2],
                    isample, 
                    gamma_id_to_ascii[ sequential_gamma_id[igamma][ig] ]);

                /***************************************************************************
                 * fp = fwd dn
                 ***************************************************************************/
                assign_fermion_propagator_from_spinor_field ( fp, propagator_snk_smeared[1-iflavor], VOLUME);

                /***************************************************************************
                 * fp2 = seq up-after-up
                 ***************************************************************************/
                assign_fermion_propagator_from_spinor_field ( fp2, sequential_propagator[iflavor], VOLUME);

                for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {
                for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {

                  /***************************************************************************
                   * calculate fp3 <- Gamma[if2] x fwd dn x Gamma[if1]
                   ***************************************************************************/
                  fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp, VOLUME );
    
                  fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_list[if1], fp3, VOLUME );
    
                  fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, -gamma_f1_sign[if1]*gamma_f1_sign[if2], VOLUME );
    
                  /***************************************************************************
                   * diagram t1
                   ***************************************************************************/
                  sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t1", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
    
                  exitstatus = reduce_project_write ( vx, vp, fp2, fp3, fp2, contract_v5, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[njjn_fht_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(48);
                  }
    
                  /***************************************************************************
                   * diagram t2
                   ***************************************************************************/
                  sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t2", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
    
                  exitstatus = reduce_project_write ( vx, vp, fp2, fp3, fp2, contract_v6, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[njjn_fht_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(48);
                  }
    
                }} // end of loop on Dirac gamma structures

              }  /* end of loop on flavor */

              /***************************************************************************
               *  W for  uu dd type insertion
               *  4 diagrams
               ***************************************************************************/
              for ( int iflavor = 0; iflavor < 2; iflavor++ ) {

                char aff_tag_prefix[200], aff_tag_prefix2[200];

                sprintf ( aff_tag_prefix, "/%s/w%c%c-w%c%c-f%c/T%d_X%d_Y%d_Z%d/QX%d_QY%d_QZ%d/sample%d/Gc_%s",
                    correlator_tag, 
                    flavor_tag[iflavor], flavor_tag[iflavor], 
                    flavor_tag[1-iflavor], flavor_tag[1-iflavor],
                    flavor_tag[iflavor],
                    gsx[0], gsx[1], gsx[2], gsx[3], 
                    momentum[0], momentum[1], momentum[2],
                    isample, 
                    gamma_id_to_ascii[ sequential_gamma_id[igamma][ig] ]);

                sprintf ( aff_tag_prefix2, "/%s/f%c-w%c%c-w%c%c/T%d_X%d_Y%d_Z%d/QX%d_QY%d_QZ%d/sample%d/Gc_%s",
                    correlator_tag, 
                    flavor_tag[iflavor],
                    flavor_tag[1-iflavor], flavor_tag[1-iflavor],
                    flavor_tag[iflavor], flavor_tag[iflavor], 
                    gsx[0], gsx[1], gsx[2], gsx[3], 
                    momentum[0], momentum[1], momentum[2],
                    isample, 
                    gamma_id_to_ascii[ sequential_gamma_id[igamma][ig] ]);

                /***************************************************************************
                 * fp = fwd up
                 ***************************************************************************/
                assign_fermion_propagator_from_spinor_field ( fp, propagator_snk_smeared[iflavor], VOLUME);

                /***************************************************************************
                 * fp2 = seq up-after-up
                 ***************************************************************************/
                assign_fermion_propagator_from_spinor_field ( fp2, sequential_propagator[iflavor], VOLUME);

                for ( int if1 = 0; if1 < gamma_f1_number; if1++ ) {
                for ( int if2 = 0; if2 < gamma_f1_number; if2++ ) {

                  /***************************************************************************
                   * calculate fp3 <- Gamma[if2] x seq dn-after-dn x Gamma[if1]
                   ***************************************************************************/
                  assign_fermion_propagator_from_spinor_field ( fp3, sequential_propagator[1-iflavor], VOLUME);

                  fermion_propagator_field_eq_gamma_ti_fermion_propagator_field ( fp3, gamma_f1_list[if2], fp3, VOLUME );
    
                  fermion_propagator_field_eq_fermion_propagator_field_ti_gamma ( fp3, gamma_f1_list[if1], fp3, VOLUME );
    
                  fermion_propagator_field_eq_fermion_propagator_field_ti_re    ( fp3, fp3, -gamma_f1_sign[if1]*gamma_f1_sign[if2], VOLUME );
    
                  /***************************************************************************
                   * diagram t1
                   ***************************************************************************/
                  sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t1", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
    
                  exitstatus = reduce_project_write ( vx, vp, fp2, fp3, fp, contract_v5, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[njjn_fht_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(48);
                  }
    
                  /***************************************************************************
                   * diagram t2
                   ***************************************************************************/
                  sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t2", aff_tag_prefix, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
    
                  exitstatus = reduce_project_write ( vx, vp, fp2, fp3, fp, contract_v6, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[njjn_fht_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(48);
                  }

                  /***************************************************************************
                   * diagram t1
                   ***************************************************************************/
                  sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t1", aff_tag_prefix2, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
    
                  exitstatus = reduce_project_write ( vx, vp, fp, fp3, fp2, contract_v5, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[njjn_fht_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(48);
                  }
    
                  /***************************************************************************
                   * diagram t2
                   ***************************************************************************/
                  sprintf(aff_tag, "/%s/Gf_%s/Gi_%s/t2", aff_tag_prefix2, gamma_id_to_Cg_ascii[ gamma_f1_list[if2] ], gamma_id_to_Cg_ascii[ gamma_f1_list[if1] ] );
    
                  exitstatus = reduce_project_write ( vx, vp, fp, fp3, fp2, contract_v6, affw, aff_tag, g_sink_momentum_list, g_sink_momentum_number, 16, VOLUME, io_proc );
                  if ( exitstatus != 0 ) {
                    fprintf(stderr, "[njjn_fht_invert_contract] Error from reduce_project_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                    EXIT(48);
                  }

                }} // end of loop on Dirac gamma structures

              }  /* end of loop on flavor */


            }  /* end of loop on gamma id inside gamma set */
          }  /* end of loop on gamma set */

          fini_1level_dtable ( &scalar_field );
          fini_2level_dtable ( &sequential_source );
          fini_3level_dtable ( &sequential_propagator );

      }  /* end of loop oet samples  */

#endif  /* of _PART_III  */
    }  /* loop on sequential source momenta */

    /***************************************************************************/
    /***************************************************************************/

    /***************************************************************************
     * clean up
     ***************************************************************************/
    free_fp_field ( &fp  );
    free_fp_field ( &fp2 );
    free_fp_field ( &fp3 );
    fini_2level_dtable ( &vx );
    fini_3level_dtable ( &vp );

#ifdef HAVE_LHPC_AFF
    /***************************************************************************
     * I/O process id 2 closes its AFF writer
     ***************************************************************************/
    if(io_proc == 2) {
      const char * aff_status_str = (char*)aff_writer_close (affw);
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[njjn_fht_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(32);
      }
    }  /* end of if io_proc == 2 */
#endif  /* of ifdef HAVE_LHPC_AFF */

    /***************************************************************************
     * free propagator fields
     ***************************************************************************/
    fini_3level_dtable ( &propagator );
    fini_3level_dtable ( &propagator_snk_smeared );

  }  /* end of loop on source locations */

  /***************************************************************************/
  /***************************************************************************/

  /***************************************************************************
   * free the allocated memory, finalize
  ***************************************************************************/
  fini_2level_ztable ( &ephase );

  fini_3level_ztable ( &loop );


  fini_rng_state ( &g_rng_state);

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  if ( gauge_field_with_phase != NULL ) free ( gauge_field_with_phase );
  if ( gauge_field_smeared    != NULL ) free ( gauge_field_smeared );

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

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [njjn_fht_invert_contract] %s# [njjn_fht_invert_contract] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [njjn_fht_invert_contract] %s# [njjn_fht_invert_contract] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
