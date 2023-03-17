/****************************************************
 * pppo_invert_contract
 *
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
#include "contract_cvc_tensor.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_i.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "Q_phi.h"
#include "clover.h"
#include "ranlxd.h"
#include "smearing_techniques.h"

#define _OP_ID_UP 0
#define _OP_ID_DN 1
#define _OP_ID_ST 2

#define _DERIV  1
#define _DDERIV 0

using namespace cvc;

void usage() {
  fprintf(stdout, "Code to calculate charged pion FF inversions + contractions\n");
  fprintf(stdout, "Usage:    [options]\n");
  fprintf(stdout, "Options:  -f <input filename> : input filename for cvc      [default cpff.input]\n");
  fprintf(stdout, "          -c                  : check propagator residual   [default false]\n");
  EXIT(0);
}

int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "avgx";

  const char fbwd_str[2][4] =  { "fwd", "bwd" };
  
  const char flavor_tag[4][2] =  { "l", "l", "s", "s" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  int check_propagator_residual = 0;
  char filename[400];
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL, *gauge_field_smeared = NULL;
  char output_filename[400];
  int spin_dilution = 4;
  int color_dilution = 1;
  double g_mus =0.;

  int const gamma_v_number = 8;
  int gamma_v_list[8] = { 0, 1, 2, 3, 6, 7, 8, 9 };
  /* int const gamma_v_number = 4;
  int gamma_v_list[4] = { 0, 1, 2, 3  }; */

  int const gamma_t_number = 6;
  int gamma_t_list[6] = { 10 , 11 , 12 , 13 , 14 , 15 };

  char data_tag[400];
#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
  struct AffWriter_s *affw = NULL;
#endif


  struct timeval ta, tb, start_time, end_time;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "rh?f:")) != -1) {
    switch (c) {
    case 'f':
      strcpy(filename, optarg);
      filename_set=1;
      break;
    case 'r':
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

  /* set the default values */
  if(filename_set==0) sprintf ( filename, "%s.input", outfile_prefix );
  /* fprintf(stdout, "# [pppo_invert_contract] Reading input from file %s\n", filename); */
  read_input_parser(filename);

#ifdef HAVE_TMLQCD_LIBWRAPPER

  fprintf(stdout, "# [pppo_invert_contract] calling tmLQCD wrapper init functions\n");

  /***************************************************************************
   * initialize MPI parameters for cvc
   ***************************************************************************/
  exitstatus = tmLQCD_invert_init(argc, argv, 1, 0);
  /* exitstatus = tmLQCD_invert_init(argc, argv, 1); */
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

  /***************************************************************************
   * report git version
   ***************************************************************************/
  if ( g_cart_id == 0 ) {
    fprintf(stdout, "# [pppo_invert_contract] git version = %s\n", g_gitversion);
  }


  /***************************************************************************
   * set number of openmp threads
   ***************************************************************************/
  set_omp_number_threads ();

  /***************************************************************************
   * initialize geometry fields
   ***************************************************************************/
  if ( init_geometry() != 0 ) {
    fprintf(stderr, "[pppo_invert_contract] Error from init_geometry %s %d\n", __FILE__, __LINE__);
    EXIT(4);
  }

  geometry();

  size_t const sizeof_spinor_field = _GSI(VOLUME) * sizeof(double);

  /***************************************************************************
   * some additional xchange operations
   ***************************************************************************/
  mpi_init_xchange_contraction(2);
  mpi_init_xchange_eo_spinor ();

  /***************************************************************************
   * initialize own gauge field or get from tmLQCD wrapper
   ***************************************************************************/
#ifndef HAVE_TMLQCD_LIBWRAPPER
  alloc_gauge_field(&g_gauge_field, VOLUMEPLUSRAND);
  if(!(strcmp(gaugefilename_prefix,"identity")==0)) {
    /* read the gauge field */
    sprintf ( filename, "%s.%.4d", gaugefilename_prefix, Nconf );
    if(g_cart_id==0) fprintf(stdout, "# [pppo_invert_contract] reading gauge field from file %s\n", filename);
    exitstatus = read_lime_gauge_field_doubleprec(filename);
  } else {
    /* initialize unit matrices */
    if(g_cart_id==0) fprintf(stdout, "\n# [pppo_invert_contract] initializing unit matrices\n");
    exitstatus = unit_gauge_field ( g_gauge_field, VOLUME );
  }
  if(exitstatus != 0) {
    fprintf ( stderr, "[pppo_invert_contract] Error initializing gauge field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(8);
  }
#else
  Nconf = g_tmLQCD_lat.nstore;
  if(g_cart_id== 0) fprintf(stdout, "[pppo_invert_contract] Nconf = %d\n", Nconf);

  exitstatus = tmLQCD_read_gauge(Nconf);
  if(exitstatus != 0) {
    EXIT(5);
  }

  exitstatus = tmLQCD_get_gauge_field_pointer( &g_gauge_field );
  if(exitstatus != 0) {
    EXIT(6);
  }
  if( g_gauge_field == NULL) {
    fprintf(stderr, "[pppo_invert_contract] Error, g_gauge_field is NULL %s %d\n", __FILE__, __LINE__);
    EXIT(7);
  }
#endif


  /***************************************************************************
   * multiply the temporal phase to the gauge field
   ***************************************************************************/
  exitstatus = gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up );
  if(exitstatus != 0) {
    fprintf(stderr, "[pppo_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  exitstatus = plaquetteria ( gauge_field_with_phase );
  if(exitstatus != 0) {
    fprintf(stderr, "[pppo_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(38);
  }

  /***************************************************************************
   * initialize clover, mzz and mzz_inv for light quark
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[pppo_invert_contract] Error from init_clover, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    fprintf(stderr, "[pppo_invert_contract] Error, io proc must be ge 0 %s %d\n", __FILE__, __LINE__);
    EXIT(14);
  }
  fprintf(stdout, "# [pppo_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );

  /***********************************************************
   *
   ***********************************************************/

  /***************************************************************************
   * allocate memory for spinor fields 
   * WITH HALO
   ***************************************************************************/
  size_t nelem = _GSI( VOLUME+RAND );
  double ** spinor_work  = init_2level_dtable ( 2, nelem );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[pppo_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * allocate memory for spinor fields
   * WITHOUT halo
   ***************************************************************************/
  nelem = _GSI( VOLUME );
  double **** stochastic_propagator_mom_list = init_4level_dtable ( 2, 3, g_source_momentum_number, nelem );
  if ( stochastic_propagator_mom_list == NULL ) {
    fprintf(stderr, "[pppo_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }

  /***************************************************************************/

  double **** stochastic_propagator_zero_list = init_4level_dtable ( 2, 3, T_global nelem );
  if ( stochastic_propagator_zero_list == NULL ) {
    fprintf(stderr, "[pppo_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(48);
  }


  /***************************************************************************/

  double ** stochastic_source_list = init_2level_dtable ( 3, nelem );
  if ( stochastic_source_list == NULL ) {
    fprintf(stderr, "[pppo_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__ );;
    EXIT(48);
  }

#ifdef _SMEAR_QUDA
  /***************************************************************************
   * dummy solve, just to have original gauge field up on device,
   * for subsequent APE smearing
   ***************************************************************************/
  memset(spinor_work[1], 0, sizeof_spinor_field);
  memset(spinor_work[0], 0, sizeof_spinor_field);
  if ( g_cart_id == 0 ) spinor_work[0][0] = 1.;
  exitstatus = _TMLQCD_INVERT(spinor_work[1], spinor_work[0], 0 );
#  if ( defined GPU_DIRECT_SOLVER )
  if(exitstatus < 0)
#  else
  if(exitstatus != 0)
#  endif
  {
    fprintf(stderr, "[njjn_fht_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(12);
  }
#endif  /* of if _SMEAR_QUDA */

  /***********************************************
   * if we want to use Jacobi smearing, we need
   * smeared gauge field
   ***********************************************/
  if( N_Jacobi > 0 ) {

#ifndef _SMEAR_QUDA 

    alloc_gauge_field ( &gauge_field_smeared, VOLUMEPLUSRAND);

    memcpy ( gauge_field_smeared, g_gauge_field, 72*VOLUME*sizeof(double));

    if ( N_ape > 0 ) {
#endif
      exitstatus = APE_Smearing(gauge_field_smeared, alpha_ape, N_ape);
      if(exitstatus != 0) {
        fprintf(stderr, "[pppo_invert_contract] Error from APE_Smearing, status was %d\n", exitstatus);
        EXIT(47);
      }
#ifndef _SMEAR_QUDA
    }  /* end of if N_aoe > 0 */

    exitstatus = plaquetteria( gauge_field_smeared );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[njjn_fht_invert_contract] Error from plaquetteria, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(2);
    }

#endif

  }  /* end of if N_Jacobi > 0 */

  /***************************************************************************
   * initialize rng state
   ***************************************************************************/

  exitstatus = init_rng_stat_file ( g_seed, NULL );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[pppo_invert_contract] Error from init_rng_stat_file %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  sprintf ( output_filename, "%s.%d.h5", g_outfile_prefix, Nconf );
  if(io_proc == 2 && g_verbose > 1 ) { 
    fprintf(stdout, "# [pppo_invert_contract] writing data to file %s\n", output_filename);
  }

  /***************************************************************************
   * generate all stochastic sources beforehand
   ***************************************************************************/
  for ( int isample = 0; isample < 3; isample++ )
  {
     exitstatus = prepare_volume_source ( stochastic_source_list[isample], VOLUME );
     if( exitstatus != 0 ) {
       fprintf(stderr, "[pppo_invert_contract] Error from prepare_volume_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
       EXIT(123);
     }

  }  /* end of loop on samples */


  /***************************************************************************
   * loop on timeslices for zero momentum propagators
   ***************************************************************************/
  for ( int gts = 0; gts < T_global; gts++ ) 
  {

    double **** momentum_propagator = init_4level_dtable ( 2, 3, g_source_momentum_number, _GSI(VOLUME) );
    if ( momentum_propagator == NULL ) {
      fprintf(stderr, "[pppo_invert_contract] Error from init_4level_dtable  %s %d\n", __FILE__, __LINE__);
      EXIT(123);
    }


    /***************************************************************************
     * local source timeslice and source process ids
     ***************************************************************************/

    int source_timeslice = -1;
    int source_proc_id   = -1;

    exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[pppo_invert_contract] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }
          
    size_t const source_timeslice_offset = source_timeslice * _GSI ( VOL3 );

    /***************************************************************************
     ***************************************************************************
     **
     ** do inversions
     **
     ***************************************************************************
     ***************************************************************************/

    /***************************************************************************
     * loop on timeslices for zero momentum propagators
     ***************************************************************************/
    for ( int isample = 0; isample < 3; isample++ ) 
    {

      for ( int imom = 0; imom < g_source_momentum_number; imom++ )
      {

        double const p[3] = { 
          TWO_MPI * g_source_momentum_list[imom][0] / LX_global,
          TWO_MPI * g_source_momentum_list[imom][1] / LY_global,
          TWO_MPI * g_source_momentum_list[imom][2] / LZ_global };

        double const phase_offset = 
            p[0] * g_proc_coords[1] * LX
          + p[1] * g_proc_coords[2] * LY
          + p[2] * g_proc_coords[3] * LZ;

        double * momentum_source = init_1level_dtable ( _GSI ( VOLUME ) );
        if ( momentum_sourcde == NULL ) {
          fprintf(stderr, "[pppo_invert_contract] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(123);
        }
 
#pragma omp parallel for
        for ( unsigned int ix = 0; ix < VOLUME; ix++ ) 
        {
          double const phase = phase_offset \
          + g_lexic2coords[ix][1] * p[0] \
          + g_lexic2coords[ix][2] * p[1] \
          + g_lexic2coords[ix][3] * p[2];

          double * const _r = momentum_source   + _GSI ( ix );
          double * const _s = stochastic_source + _GSI ( ix );

          complex const w = { cos (phase), sin(phase) };

          _fv_eq_fv_ti_co( _r, _s, &w );
        }

        /***************************************************************************
         * SOURCE SMEARING
         ***************************************************************************/
        if ( N_Jacobi > 0 ) {
          exitstatus = Jacobi_Smearing ( gauge_field_smeared, momentum_source, N_Jacobi, kappa_Jacobi);
          if(exitstatus != 0) {
            fprintf(stderr, "[pppo_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            return(11);
          }
        }

        /***************************************************************************
         * loop on flavor
         ***************************************************************************/
        for ( int iflavor = 0; iflavor < 2; iflavor++ ) 
        {

          memset ( spinor_work[1], 0, sizeof_spinor_field );
          memset ( spinor_work[0], 0, sizeof_spinor_field );

          if ( source_proc_id == g_cart_id ) 
          {
            memcpy ( spinor_work[0] + source_timeslice_offset, momentum_source + source_timeslice_offset, sizeof_spinor_field_timeslice );
          }

          exitstatus = spinor_field_tm_rotation ( spinor_work[0], spinor_work[0], 1-2*iflavor, g_ermion_type, VOLUME );
          if( exitstatus != 0 ) {
            fprintf(stderr, "[pppo_invert_contract] Error from spinor_field_tm_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(123);
          }
       
          exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], iflavor );
          if(exitstatus < 0) {
            fprintf(stderr, "[pppo_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(44);
          }

          if ( check_propagator_residual ) {
            check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[iflavor], mzzinv[iflavor], 1 );
          }

          exitstatus = spinor_field_tm_rotation ( spinor_work[1], spinor_work[1], 1-2*iflavor, g_ermion_type, VOLUME );
          if( exitstatus != 0 ) {
            fprintf(stderr, "[pppo_invert_contract] Error from spinor_field_tm_rotation, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            EXIT(123);
          }

          if ( N_Jacobi > 0 ) {
            gettimeofday ( &ta, (struct timezone *)NULL );
            /***************************************************************************
             * SINK SMEARING
             ***************************************************************************/
            exitstatus = Jacobi_Smearing ( gauge_field_smeared, spinor_work[1], N_Jacobi, kappa_Jacobi);
            if(exitstatus != 0) {
              fprintf(stderr, "[pppo_invert_contract] Error from Jacobi_Smearing, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
              return(11);
            }
            gettimeofday ( &tb, (struct timezone *)NULL );
            show_time ( &ta, &tb, "pppo_invert_contract", "Jacobi_Smearing-stochastic-propagator", g_cart_id == 0 );
          }

          memcpy( momentum_propagator[iflavor][isample][imom], spinor_work[1], sizeof_spinor_field );

        }  /* end of loop on flavor */

        fini_1level_dtable ( &momentum_source );

      }  /* end of loop on momenta */

    }  /* end of loop on samples */

    /***************************************************************************
     ***************************************************************************
     **
     ** do contractions
     **
     ***************************************************************************
     ***************************************************************************/

    /***************************************************************************
     * simple meson-meson
     ***************************************************************************/
    double ** contr_x = init_2level_dtable ( g_source_momentum_number,  2 * VOLUME );
    if ( contr_x == NULL ) {
      fprintf(stderr, "[pppo_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(3);
    }

    double *** contr_p = init_3level_dtable ( g_sink_momentum_number, g_source_momentum_number, 2 * T );
    if ( contr_p == NULL ) {
      fprintf(stderr, "[avgx_invert_contract] Error from init_Xlevel_dtable %s %d\n", __FILE__, __LINE__);
      EXIT(3);
    }
    
    for ( int ifl1 = 0; ifl1 < 2; ifl1++) 
    {
      for ( int ifl2 = 0; ifl2 < 2; ifl2++) 
      {

        for ( int ismp1 = 0; ismp1 < 3; ismpl1++ ) 
        {
          for ( int ismp2 = 0; ismp2 < 3; ismpl2++ ) 
          {

            memset ( contr_x[0], 0, g_sink_momentum_number * 2 * VOLUME * sizeof( double ) );
  
            memset ( contr_p[0][0], 0, g_sink_momentum_number * 2 * g_sink_momentum_number * sizeof( double ) );

            double * const chi = momentum_propagator[ifl2][ismp2][0];

            for ( int imom = 0; imom < g_source_momentum_number; imom++ )
            {
              double * const phi = momentum_propagator[ifl1][ismp1][imom];

#pragma omp parallel for
              for ( unsigned int ix = 0; ix < VOLUME; ix++ ) 
              {
                size_t const iy = _GSI(ix);
                double * const _chi = chi + iy;
                double * const _phi = phi + iy;

                complex w = {0,0};

                _co_eq_fv_dag_ti_fv ( &w, _chi, _phi );

                contr_x[imom][2*ix  ] = w.re;
                contr_x[imom][2*ix+1] = w.im;
              }

            }  /* end of loop on momenta */

            /***************************************************************************
             * momentum projection at sink
             ***************************************************************************/
            exitstatus = momentum_projection ( contr_x[0], contr_p[0][0], g_source_momentum_number * T, g_sink_momentum_number, g_sink_momentum_list );
            if(exitstatus != 0) {
              fprintf(stderr, "[pppo_invert_contract] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(3);
            }

            /***************************************************************************
             * parallel HDF5 output
             ***************************************************************************/
STOPPED HERE
            /* ... */
            exitstatus = contract_write_to_h5_file ( &contr_p, output_filename, data_tag, &sink_momentum, 1, io_proc );
            if(exitstatus != 0) {
              fprintf(stderr, "[avgx_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(3);
            }

          }  /* end of loop on samples 2 */
        }  /* end of loop on samples 1 */
      }  /* end of loop on flavor 2 (zero mom, dagger) */
    }  /* end of loop on flavor 1 (mom) */

    /* deallocate the contraction fields */
    fini_1level_dtable ( &contr_x );
    fini_1level_dtable ( &contr_p );

    /***************************************************************************
     * store the propagators fior subsequent contractions
     ***************************************************************************/
    for ( int iflavor = 0; iflavor < 2; iflavor++ )
    {
      for ( int isample = 0; isample < 3; isample++ ) 
      {
        memcpy ( stochastic_propagator_zero_list[iflavor][isample][gts], momentum_propagator[iflavor][isample][0], sizeof_spinor_field );

        for ( int imom = 0; imom < g_sink_momentum_number; imom++ )
        {
          memcpy ( stochastic_propagator_mom_list[iflavor][isample][imom] + source_timeslice_offset, 
                   momentum_propagator[iflavor][isample][imom]            + source_timeslice_offset, sizeof_spinor_field_timeslice );
        }
      }
    }

    fini_4level_dtable ( &momentum_propagator );

  }  /* end of loop on timeslices */

  /***************************************************************************
   * decallocate spinor fields
   ***************************************************************************/
  fini_4level_dtable ( &stochastic_propagator_mom_list );
  fini_4level_dtable ( &stochastic_propagator_zero_list );
  fini_2level_dtable ( &stochastic_source_list );
  fini_2level_dtable ( &spinor_work );

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

  if ( gauge_field_with_phase != NULL ) free( gauge_field_with_phase );
  if ( gauge_field_smeared    != NULL ) free( gauge_field_smeared );

  /* free clover matrix terms */
  fini_clover ( &mzz, &mzzinv );


#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif

  free_geometry();


#ifdef HAVE_TMLQCD_LIBWRAPPER
  tmLQCD_finalise();
#endif

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor ();
  mpi_fini_datatypes();
  MPI_Finalize();
#endif


  gettimeofday ( &end_time, (struct timezone *)NULL );
  show_time ( &start_time, &end_time, "pppo_invert_contract", "runtime", g_cart_id == 0 );

  return(0);

}
