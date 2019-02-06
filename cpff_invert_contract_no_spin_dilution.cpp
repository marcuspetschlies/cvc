/****************************************************
 * cpff_invert_contract.c
 *
 * Mi 31. Okt 07:32:01 CET 2018
 *
 * - originally copied from p2gg_contract
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
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
#include "contractions_io.h"
#include "Q_clover_phi.h"
#include "contract_cvc_tensor.h"
#include "prepare_source.h"
#include "prepare_propagator.h"
#include "project.h"
#include "table_init_z.h"
#include "table_init_d.h"
#include "dummy_solver.h"
#include "Q_phi.h"
#include "clover.h"
#include "ranlxd.h"

#include "init_g_gauge_field.hpp"
#include "Stopwatch.hpp"
#include "Core.hpp"
#include "enums.hpp"
#include "ParallelMT19937_64.hpp"
#include "meta_types.hpp"
#include "meta_parsing.hpp"

#include <vector>
#include <map>
#include <string>
#include <utility>


using namespace cvc;


int main(int argc, char **argv) {
  
  const char outfile_prefix[] = "cpff";

  const char fbwd_str[2][4] =  { "fwd", "bwd" };

  int c;
  int filename_set = 0;
  int exitstatus;
  int io_proc = -1;
  size_t sizeof_spinor_field;
  char filename[100];
  // double ratime, retime;
  double **mzz[2] = { NULL, NULL }, **mzzinv[2] = { NULL, NULL };
  double *gauge_field_with_phase = NULL;
  int op_id_up = -1, op_id_dn = -1;
  char output_filename[400];
  int * rng_state = NULL;

  char data_tag[400];

  // initialise CVC core functionality and read input files
  Core core(argc, argv, "cpff_invert_contract_no_spin_dilution");
  if( !core.is_initialised() ){
    debug_printf(0,g_proc_id,"Core initialisation failed!\n");
    return(CVC_EXIT_CORE_INIT_FAILURE);
  }

  std::vector<twopt_oet_meta_t> twopt_correls;
  std::vector<threept_oet_meta_t> local_threept_correls;
  std::vector<threept_oet_meta_t> deriv_threept_correls;
  
  std::map<std::string, stoch_prop_meta_t> props_meta;
  std::map<std::string, std::vector<double>> props;

  std::map<std::string, int> flav_op_ids;
  flav_op_ids["u"] = 0;
  flav_op_ids["d"] = 1;
  flav_op_ids["sp"] = 2;
  flav_op_ids["sm"] = 3;
  flav_op_ids["cp"] = 4;
  flav_op_ids["cm"] = 5;


  // these various options are included here but we will not use them in practice
  // I keep the code below because it is nice documentation of what is going on
  const bool use_g0_g5_sink_pion_2pt = false;
  const bool use_g0_g5_sink_pion_3pt = false;
  const bool do_singlet_2pt = false;
  const bool do_singlet_3pt = false;
  const bool use_g0_g5_sink_singlet_2pt = false;
  const bool use_g0_g5_sink_singlet_3pt = false;

  const double one_ov_vol3 = 1.0/(LX*LY*LZ);

  debug_printf(0,0, "one_ov_vol3 = %f\n", one_ov_vol3);

  // \bar{chi}_d g5 chi_u \bar{chi}_u g5 chi_d
  // calculated via
  // = - <g5 S_u^dag | g5 g5 | S_u g5 [eta eta^dag]>
  //       ^               ^       ^
  //       gb              gf      gi
  // 
  // = - <g5 S_u^dag | g5 g5 | S_u g5 [eta eta^dag] >
  //            this g5 ^ is implicitly included in the contraction later such that source
  //            and sink gammas are explicit
  // The gamma structure at the backwards propagator, however, must be specified,
  // see "gb" below.
  twopt_correls.push_back( twopt_oet_meta_t("u", /* forward (undaggered) prop flavor */ 
                                            "u", /* backward (daggered) prop flavor */
                                            "fwd", /* source momentum carried by forward prop */
                                            5, /* gi */ 
                                            5, /* gf */
                                            5, /* gb: gamma structure to get the correct backward propagator */
                                            {one_ov_vol3, 0.0} /* normalisation */) );

  // Kaon two-pt function
  twopt_correls.push_back( twopt_oet_meta_t("u",
                                            "sp",
                                            "fwd",
                                            5,
                                            5,
                                            5,
                                            {one_ov_vol3, 0.0}) );
  
  // note that in the twisted basis, \bar{d} g0 g5 u -> i\bar{chi}_d g0 chi_u
  //               and at the source \bar{u} g5 g0 d -> i\bar{chi}_u g0 chi_d
  // it can be shown that the resulting correlation functions are real
  if( use_g0_g5_sink_pion_2pt ){
    // +i \bar{chi}_d g0 chi_u \bar{chi}_u g5 chi_d 
    // = -i (g5 S_u^dag | g5 g0 | S_u g5)
    twopt_correls.push_back( twopt_oet_meta_t("u", "u", "fwd", 5, 0, 5, {0.0, -one_ov_vol3}) );
    // we cannot put this at the source using OET unless we do spin dilution
  }

  // while we can't have non-diagonal gamma structures at the source
  // without spin-dilution, a diagaonal gamma, as required for the eta
  // two-pt function, is not a problem
  if( do_singlet_2pt ){
    // \bar{chi}_u chi_u \bar{chi}_u chi_u
    // = g5 S_d^dag | g5 1 | S_u 1
    twopt_correls.push_back( twopt_oet_meta_t("u", "d", "fwd", 4, 4, 5, {-one_ov_vol3, 0.0}) );

    // \bar{chi}_d chi_d \bar{chi}_d chi_d
    // = g5 S_u^dag | g5 1 | S_d 1
    twopt_correls.push_back( twopt_oet_meta_t("d", "u", "fwd", 4, 4, 5, {-one_ov_vol3, 0.0}) );
    
    // for the flavor singlet, we have \bar{u/d} g0 g5 u/d -> \bar{chi}_u/d g0 g5 chi_u/d
    if( use_g0_g5_sink_singlet_2pt ){
      // \bar{chi}_u g0 g5 chi_u  \bar{chi}_u chi_u
      // = g5 S_d^dag | g5 g0 g5 | S_u 1
      twopt_correls.push_back( twopt_oet_meta_t("u", "d", "fwd", 4, 6, 5, {-one_ov_vol3, 0.0}) );
      // \bar{chi}_d g0 g5 chi_d  \bar{chi}_d chi_d
      // = g5 S_u^dag | g5 g0 g5 | S_d 1
      twopt_correls.push_back( twopt_oet_meta_t("d", "u", "fwd", 4, 6, 5, {-one_ov_vol3, 0.0}) );
    }
  }

  // 0 component of vector current between pi^+ states
  // \bar{chi}_d g5 chi_u \bar{chi}_u g0 chi_u \bar{chi}_u g5 \bar{chi}_d
  // = - g5 S_u^dag g5 g5 g5 S_d^dag | g5 g0 | S_u g5
  //     ^             ^                  ^        ^
  //     gb            gf                 gc       gi
  // = - g5 S_u^dag g5 g5 g5 S_d^dag | g5 g0 | S_u g5
  //    this g5                         ^  is implicitly included
  //    these g5     ^     ^  are suppressed and their effect must be included
  //                          in the normalisation of the three-pt function
  //      such that source, sink and current gamma are explicit
  local_threept_correls.push_back( 
      threept_oet_meta_t("u", /* forward prop flavor */
                         "u", /* backward prop flavor */
                         "d", /* sequential prop flavor */
                         "fwd", /* source momentum carried by */ 
                         5, /* gi */ 
                         5, /* gf */ 
                         0, /* gc */ 
                         5, /* gb */ 
                         {-one_ov_vol3, 0.0}) );

  //// scalar density (twisted basis -> gamma_5) between pi^+ states
  //// \bar{chi}_d g5 chi_u \bar{chi}_u g5 chi_u \bar{chi}_u g5 \bar{chi}_d
  //// = - g5 S_u^dag g5 g5 S_d^dag | g5 g5 | S_u g5
  local_threept_correls.push_back( 
      threept_oet_meta_t("u", "u", "d", "fwd", 5, 5, 5, 5, {-one_ov_vol3, 0.0} ) );

  if( use_g0_g5_sink_pion_3pt ){
    // \bar{chi}_d g0 chi_u \bar{chi}_u g0 chi_u \bar{chi}_u g5 \bar{chi}_d
    // = + g5 S_u^dag g5 g0 g5 S_d^dag | g5 g0 | S_u g5
    local_threept_correls.push_back( threept_oet_meta_t("u", "u", "d", "fwd", 5, 5, 0, 5, {0.0, one_ov_vol3}) );
    // \bar{chi}_d g0 chi_u \bar{chi}_u g5 chi_u \bar{chi}_u g5 \bar{chi}_d
    // = + g5 S_u^dag g5 g0 g5 S_d^dag | g5 g5 | S_u g5
    local_threept_correls.push_back( threept_oet_meta_t("u", "u", "d", "fwd", 5, 5, 5, 5, {0.0, one_ov_vol3}) );
  }
 
  if( do_singlet_3pt ){
    // \bar{chi}_u 1 chi_u \bar{chi}_u g0 chi_u \bar{chi}_u 1 \bar{chi}_u
    // g5 S_d^dag g5 1 g5 S_d^dag g5 g0 S_u 1
    local_threept_correls.push_back( 
        threept_oet_meta_t("u", "d", "d", "fwd", 4, 4, 0, 5, {-one_ov_vol3, 0.0} ) );

    // \bar{chi}_u 1 chi_u \bar{chi}_u 1 chi_u \bar{chi}_u 1 \bar{chi}_u
    // g5 S_d^dag g5 1 g5 S_d^dag g5 1 S_u 1
    local_threept_correls.push_back( 
        threept_oet_meta_t("u", "d", "d", "fwd", 4, 4, 4, 5, {-one_ov_vol3, 0.0} ) );
    
    // \bar{chi}_d 1 chi_d \bar{chi}_d g0 chi_d \bar{chi}_d 1 \bar{chi}_d
    // g5 S_u^dag g5 1 g5 S_u^dag g5 g0 S_d 1
    local_threept_correls.push_back( 
        threept_oet_meta_t("d", "u", "u", "fwd", 4, 4, 0, 5, {-one_ov_vol3, 0.0} ) );
    
    // \bar{chi}_d 1 chi_d \bar{chi}_d 1 chi_d \bar{chi}_d 1 \bar{chi}_d
    // g5 S_u^dag g5 1 g5 S_u^dag g5 1 S_d 1
    local_threept_correls.push_back( 
        threept_oet_meta_t("d", "u", "u", "fwd", 4, 4, 4, 5, {-one_ov_vol3, 0.0} ) );

    if( use_g0_g5_sink_singlet_3pt ){
      // TODO add eta 3pt funtions with g0 g5 at the sink
      // this is not implemented since we are not presently interested in the
      // singlet three point function
    }
  }

  // TODO define appropriate correlators for derivatives

  // derive the required fwd / bwd propagators from the correlation function definitions
  get_fwd_bwd_props_meta_from_npt_oet_meta(twopt_correls, props_meta);
  get_fwd_bwd_props_meta_from_npt_oet_meta(local_threept_correls, props_meta);
  get_fwd_bwd_props_meta_from_npt_oet_meta(deriv_threept_correls, props_meta);

  sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);

  debug_printf(0, 0, "\n%lu forward / backward propagators will use %f GB of memory\n\n", 
      props_meta.size(),
      1.0e-9*props_meta.size()*(double)sizeof_spinor_field*g_nproc );

  debug_printf(0, 0, "\n\nList of forward propagators to be generated\n");
  for( auto const & prop : props_meta ) {
    debug_printf(0, 0, "Propagator: %s to be generated\n", prop.first.c_str());
  }
  debug_printf(0, 0, "\n\n");

  /***************************************************************************
   * some additional xchange operations
   ***************************************************************************/
  mpi_init_xchange_contraction(2);
  mpi_init_xchange_eo_spinor ();

  /***************************************************************************
   * initialize own gauge field or get from tmLQCD wrapper
   ***************************************************************************/
  CHECK_EXITSTATUS_NONZERO(
      exitstatus,
      init_g_gauge_field(),
      "[cpff_invert_contract] Error initialising gauge field!",
      true, 
      CVC_EXIT_GAUGE_INIT_FAILURE);

  // set the configuration number
#ifdef HAVE_TMLQCD_LIBWRAPPER
  Nconf = g_tmLQCD_lat.nstore;
#endif

  /***************************************************************************
   * multiply the temporal phase to the gauge field
   ***************************************************************************/
  CHECK_EXITSTATUS_NONZERO(
      exitstatus,
      gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up ),
      "[cpff_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase!",
      true, 
      CVC_EXIT_UTIL_FUNCTION_FAILURE);

  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  CHECK_EXITSTATUS_NONZERO(
      exitstatus,
      plaquetteria( gauge_field_with_phase ),
      "[cpff_invert_contract] Error from plaquetteria!",
      true, 
      CVC_EXIT_UTIL_FUNCTION_FAILURE);

  /***************************************************************************
   * initialize clover, mzz and mzz_inv
   ***************************************************************************/
  exitstatus = init_clover ( &g_clover, &mzz, &mzzinv, gauge_field_with_phase, g_mu, g_csw );
  if ( exitstatus != 0 ) {
    PRINT_STATUS(exitstatus, "[cpff_invert_contract] Error from init_clover!");
    EXIT(1);
  }

  /***************************************************************************
   * set io process
   ***************************************************************************/
  io_proc = get_io_proc ();
  if( io_proc < 0 ) {
    PRINT_STATUS(io_proc, "[cpff_invert_contract] Error, io proc must be ge 0!");
    EXIT(14);
  }
  fprintf(stdout, "# [cpff_invert_contract] proc%.4d has io proc id %d\n", g_cart_id, io_proc );
  
  // memory for a random spinor and the non-spin-diluted stochastic source
  std::vector<double> ranspinor( _GSI(VOLUME) );
  std::vector<double> stochastic_source( _GSI(VOLUME) );

  /***************************************************************************
   * allocate memory for spinor fields 
   * WITH HALO
   ***************************************************************************/
  const size_t spinor_length_with_halo = _GSI( VOLUME+RAND );
  double ** spinor_work  = init_2level_dtable ( 2, spinor_length_with_halo );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[cpff_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
    EXIT(1);
  }

  /***************************************************************************
   * initialize rng state
   ***************************************************************************/
  exitstatus = init_rng_state ( g_seed, &rng_state);
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[cpff_invert_contract] Error from init_rng_state %s %d\n", __FILE__, __LINE__ );;
    EXIT( 50 );
  }

  // we xor the starting seed with the configuration number to obtain a unique seed on
  // each gauge configuration
  ParallelMT19937_64 rng( (unsigned long long)(g_seed^Nconf) );

  // handy stopwatch which we may use throughout for timings
  Stopwatch sw(g_cart_grid);

  /***************************************************************************
   * loop on source timeslices
   ***************************************************************************/
  for( int isource_location = 0; isource_location < g_source_location_number; isource_location++ ) {

    /***************************************************************************
     * local source timeslice and source process ids
     ***************************************************************************/

    int source_timeslice = -1;
    int source_proc_id   = -1;
    int gts              = ( g_source_coords_list[isource_location][0] +  T_global ) %  T_global;

    exitstatus = get_timeslice_source_info ( gts, &source_timeslice, &source_proc_id );
    if( exitstatus != 0 ) {
      fprintf(stderr, "[cpff_invert_contract] Error from get_timeslice_source_info status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      EXIT(123);
    }

#if ( defined HAVE_LHPC_AFF ) && !(defined HAVE_HDF5 )
    /***************************************************************************
     * output filename
     ***************************************************************************/
    sprintf ( output_filename, "%s.%.4d.t%d.aff", outfile_prefix, Nconf, gts );
    /***************************************************************************
     * writer for aff output file
     ***************************************************************************/
    if(io_proc == 2) {
      affw = aff_writer ( output_filename);
      const char * aff_status_str = aff_writer_errstr ( affw );
      if( aff_status_str != NULL ) {
        fprintf(stderr, "[cpff_invert_contract] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
        EXIT(15);
      }
    }  /* end of if io_proc == 2 */
#elif ( defined HAVE_HDF5 )
    sprintf ( output_filename, "%s.%.4d.t%d.h5", outfile_prefix, Nconf, gts );
#endif
    if(io_proc == 2 && g_verbose > 1 ) { 
      fprintf(stdout, "# [cpff_invert_contract] writing data to file %s\n", output_filename);
    }

    /***************************************************************************
     * loop on stochastic oet samples
     ***************************************************************************/
    for ( int isample = 0; isample < g_nsample_oet; isample++ ) {

      /***************************************************************************
       * synchronize rng states to state at zero
       ***************************************************************************/
      exitstatus = sync_rng_state ( rng_state, 0, 0 );
      if(exitstatus != 0) {
        fprintf(stderr, "[cpff_invert_contract] Error from sync_rng_state, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        EXIT(38);
      }
      
      // generate the stochastic source
      // first a z2 (x) z2 volume source
      sw.reset();
      rng.gen_z2(ranspinor.data(), 24);
      sw.elapsed_print("Z2 volume source");
      // we can write the random spinor to file if we want
      if ( g_write_source ) {
        sprintf(filename, "%s.conf%.4d.t%d.sample%.5d", filename_prefix, Nconf, gts, isample);
        if ( ( exitstatus = write_propagator( ranspinor.data(), filename, 0, g_propagator_precision) ) != 0 ) {
          fprintf(stderr, "[cpff_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }
      }
      

      // loop over all required forward / backward propagators
      for( auto const & prop_meta_pair : props_meta ){
        stoch_prop_meta_t prop_meta = prop_meta_pair.second;
        std::string prop_key = prop_meta.key();

        /***************************************************************************
         * prepare stochastic timeslice source at source momentum
         ***************************************************************************/
        CHECK_EXITSTATUS_NONZERO(
          exitstatus,
          prepare_gamma_timeslice_oet(
            stochastic_source.data(),
            ranspinor.data(),
            prop_meta.gamma,
            gts,
            (prop_meta.p[0] == 0 && 
             prop_meta.p[1] == 0 &&
             prop_meta.p[2] == 0 ) ? NULL : prop_meta.p ),
          "[cpff_invert_contract] Error from prepare_gamma_timeslice_oet!",
          true, 
          CVC_EXIT_MALLOC_FAILURE);

        memcpy ( spinor_work[0], stochastic_source.data(), sizeof_spinor_field );
        memset ( spinor_work[1], 0, sizeof_spinor_field );

        debug_printf(0, 0, "Inverting to generate propagator %s\n", prop_key.c_str() );
        
        sw.reset();
        CHECK_EXITSTATUS_NEGATIVE(
          exitstatus,
          _TMLQCD_INVERT( spinor_work[1], spinor_work[0], flav_op_ids[ prop_meta.flav ] ),
          "[cpff_invert_contract] Error from TMLQCD_INVERT",
          true,
          CVC_EXIT_UTIL_FUNCTION_FAILURE);
        sw.elapsed_print("TMLQCD_INVERT");

        if ( g_check_inversion ) {
          sw.reset();
          check_residual_clover (&(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, 
                                 mzz[ flav_op_ids[ prop_meta.flav] ], mzzinv[ flav_op_ids[ prop_meta.flav] ], 1 );
          sw.elapsed_print("check_residual_clover");
        }

        if( props.count( prop_key ) == 0 ){
          sw.reset();
          props.emplace( std::make_pair( prop_key,
                                         std::vector<double>( _GSI(VOLUME) ) ) );
          sw.elapsed_print("Propagator memory allocation");
        }
        memcpy( props[ prop_key ].data(), spinor_work[1], sizeof_spinor_field);

        if ( g_write_propagator ) {
          sprintf(filename, "%s.conf%.4d.t%d.px%dpy%dpz%d.gamma%d.f%s.sample%.5d.inverted", 
              filename_prefix, Nconf, gts,
              prop_meta.p[0], prop_meta.p[1], prop_meta.p[1], 
              prop_meta.gamma, prop_meta.flav.c_str(), isample);
          exitstatus = write_propagator( props[ prop_key ].data(), filename, 0, g_propagator_precision);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[cpff_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
            EXIT(2);
          }
        }
      }  // end of loop over required propagators

      /*****************************************************************
       * contractions for 2-point functons
       *****************************************************************/
      for( int icor = 0; icor < twopt_correls.size(); icor++ ){
        for ( int isrc_mom = 0; isrc_mom < g_source_momentum_number; isrc_mom++ ) {
          int zero_mom[3] = {0,0,0};
          int prop_mom[3] = {
            g_source_momentum_list[isrc_mom][0],
            g_source_momentum_list[isrc_mom][1],
            g_source_momentum_list[isrc_mom][2] };

          // when the correlator has been defined with the source momentum carried by
          // the backward propagator, we need to retrieve the propagator with
          // negative momemntum because the backward propagator is daggered
          stoch_prop_meta_t fwdprop_meta;
          stoch_prop_meta_t bwdprop_meta;
          if( twopt_correls[icor].src_mom_prop == "bwd" ){
            for( auto & mom_component : prop_mom ){
              mom_component = -mom_component;
            }
            bwdprop_meta = stoch_prop_meta_t( prop_mom, twopt_correls[icor].gb, 
                                              twopt_correls[icor].bprop_flav );
            fwdprop_meta = stoch_prop_meta_t( zero_mom, twopt_correls[icor].gi,
                                              twopt_correls[icor].fprop_flav );
          } else {
            bwdprop_meta = stoch_prop_meta_t( zero_mom, twopt_correls[icor].gb, 
                                              twopt_correls[icor].bprop_flav );
            fwdprop_meta = stoch_prop_meta_t( prop_mom, twopt_correls[icor].gi,
                                              twopt_correls[icor].fprop_flav );
          }

          double * contr_x = init_1level_dtable ( 2 * VOLUME );
          if ( contr_x == NULL ) {
            fprintf(stderr, "[cpff_invert_contract] Error from init_1level_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(3);
          }

          double ** contr_p = init_2level_dtable ( g_sink_momentum_number , 2 * T );
          if ( contr_p == NULL ) {
            fprintf(stderr, "[cpff_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
            EXIT(3);
          }

          int source_momentum[3] = {
            g_source_momentum_list[isrc_mom][0],
            g_source_momentum_list[isrc_mom][1],
            g_source_momentum_list[isrc_mom][2] };

            char twopttag[20];
            snprintf(twopttag, 20, "%s+-g-%s-g", twopt_correls[icor].bprop_flav.c_str(),
                                                 twopt_correls[icor].fprop_flav.c_str() );

            debug_printf(0, 3,
                         "Contracting [%s]^dag g5 g%d [%s] in x-space\n",
                         bwdprop_meta.key().c_str(), twopt_correls[icor].gf, fwdprop_meta.key().c_str() );

            sw.reset();
            contract_twopoint_xdep_gamma5_gamma_snk_only( 
              contr_x, 
              twopt_correls[icor].gf, 
              props[ bwdprop_meta.key() ].data(), 
              props[ fwdprop_meta.key() ].data(),
              1 /*stride*/);
            sw.elapsed_print("contract_twopoint_xdep");

            /* momentum projection at sink */
            sw.reset();
            exitstatus = momentum_projection ( contr_x, contr_p[0], T, g_sink_momentum_number, g_sink_momentum_list );
            if(exitstatus != 0) {
              fprintf(stderr, "[cpff_invert_contract] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(3);
            }
            for( int i = 0; i < g_sink_momentum_number; ++i){
              scale_cplx(contr_p[i], T, twopt_correls[icor].normalisation);
            }
            sw.elapsed_print("sink momentum_projection and normalisation");

            snprintf ( 
                data_tag, 400, "/%s/t%d/s%d/gf%d/gi%d/pix%dpiy%dpiz%d", 
                twopttag, gts, isample,
                twopt_correls[icor].gf, twopt_correls[icor].gi,
                source_momentum[0], source_momentum[1], source_momentum[2] );

#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
          exitstatus = contract_write_to_aff_file ( contr_p, affw, data_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
#elif ( defined HAVE_HDF5 )          
          exitstatus = contract_write_to_h5_file ( contr_p, output_filename, data_tag, g_sink_momentum_list, g_sink_momentum_number, io_proc );
#endif
          if(exitstatus != 0) {
            fprintf(stderr, "[cpff_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(3);
          }

        fini_1level_dtable ( &contr_x );
        fini_2level_dtable ( &contr_p );
        } // end of loop over source momenta
      }  /* end of loop on 2pt function contractions */
      
      /* *************************************************************************
       * inversions / contractions for local current insertion threept functions
       *
       * rather than generating all sequential propagators in one go (as we do for
       * the forward and backward propagators), we iterate over the sequential
       * source time slices and momenta and generate only the required subsets
       * for this particular source ts / momentum combination
       *
       * We then invert and do all relevant contractions. This way, we use the
       * minimum amount of memory possible.
       *
       **************************************************************************/
      if( local_threept_correls.size() > 0 ){
        for( int iseq_timeslice = 0; iseq_timeslice < g_sequential_source_timeslice_number; iseq_timeslice++ ){
          // this is the sink time slice (global coords) counted from the current source
          // time slice 'gts'
          const int gtseq = ( gts + g_sequential_source_timeslice_list[iseq_timeslice] + T_global ) % T_global;
          for( int iseq_mom = 0; iseq_mom < g_seq_source_momentum_number; iseq_mom++){
            debug_printf(0, 0,
                         "iseq_timeslice = %d / %d, iseq_mom = %d / %d\n",
                         iseq_timeslice, g_sequential_source_timeslice_number,
                         iseq_mom, g_seq_source_momentum_number);

            // the sequential propagator is daggered in the contraction below, so we have to dagger the momentum projector
            // which will be applied at the sequential (sink) time slice
            int seq_source_momentum[3] = { -g_seq_source_momentum_list[iseq_mom][0],
                                           -g_seq_source_momentum_list[iseq_mom][1],
                                           -g_seq_source_momentum_list[iseq_mom][2] };

            std::map<std::string, seq_stoch_prop_meta_t> seq_props_meta;
            std::map<std::string, std::vector<double>> seq_props;

            // generate meta-data for all sequential propagators required for this sequential source time slice
            // and sequential source momentum
            get_seq_props_meta_from_npt_oet_meta(local_threept_correls, seq_source_momentum, gtseq, seq_props_meta);

            debug_printf(0, 0, 
                         "\n\n%lu sequential propagators will use %f GB of memory\n\n",
                         seq_props_meta.size(), seq_props_meta.size()*sizeof_spinor_field*1.0e-9); 
            
            for( auto const & seq_prop_meta_pair : seq_props_meta ){
              seq_stoch_prop_meta_t seq_prop_meta = seq_prop_meta_pair.second;

              // the key for the current sequential propagator to be generated
              std::string seq_prop_key = seq_prop_meta.key();

              // in order to generate the source, we need to retrieve the correct backward propagator
              // from our 'props' map and pass that to 'init_sequential_source'
              std::string bprop_key = seq_prop_meta.src_prop.key();

              sw.reset();
              CHECK_EXITSTATUS_NONZERO(
                exitstatus,
                init_sequential_source(spinor_work[0], 
                                       props[ bprop_key ].data(),
                                       gtseq, 
                                       seq_source_momentum,
                                       seq_prop_meta.gamma),
                "[cpff_invert_contract] Error from init_sequential_source",
                true,
                CVC_EXIT_UTIL_FUNCTION_FAILURE);
              sw.elapsed_print("init_sequential_source");

              memset(spinor_work[1], 0, sizeof_spinor_field);
             
              debug_printf(0, 0,
                           "Inverting to generate propagator %s\n",
                           seq_prop_key.c_str()); 
              sw.reset();
              CHECK_EXITSTATUS_NEGATIVE(
                exitstatus,
                _TMLQCD_INVERT( spinor_work[1], spinor_work[0], flav_op_ids[ seq_prop_meta.flav ] ),
                "[cpff_invert_contract] Error from TMLQCD_INVERT",
                true,
                CVC_EXIT_UTIL_FUNCTION_FAILURE);
              sw.elapsed_print("TMLQCD_INVERT");

              if( g_check_inversion ){
                check_residual_clover( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase,
                                       mzz[ flav_op_ids[ seq_prop_meta.flav ] ],
                                       mzzinv[ flav_op_ids[ seq_prop_meta.flav ] ], 1 );
              }

              if( seq_props.count( seq_prop_key ) == 0 ){
                sw.reset();
                seq_props.emplace( 
                  std::make_pair( seq_prop_key,
                                  std::vector<double>( _GSI(VOLUME) ) ) );
                sw.elapsed_print("Propagator memory allocation");
              }
              memcpy( seq_props[ seq_prop_key ].data(), spinor_work[1], sizeof_spinor_field);
            } // end of loop over inversions for sequential propagators

            // now the contractions
            for(int icor = 0; icor < local_threept_correls.size(); ++icor){
              const threept_oet_meta_t & correl = local_threept_correls[icor];
              int bwd_prop_mom[3];
              int fwd_prop_mom[3];
             
              double ** contr_p = init_2level_dtable ( g_source_momentum_number, 2*T );
              if ( contr_p == NULL ) {
                fprintf(stderr, "[cpff_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
                EXIT(47);
              }
              
              for(int isrc_mom = 0; isrc_mom < g_source_momentum_number; ++isrc_mom){
                // make sure that we get the propagator with correct source momentum, depending on which
                // one is carrying it
                if( correl.src_mom_prop == "bwd" ){
                  for( int i : {0,1,2} ) {
                    // the backward propagator, part of the sequential propagator, is daggered
                    // as part of the sequential propagator, so the momentum projector is daggered 
                    // to get the correct momentum
                    bwd_prop_mom[i] = -g_source_momentum_list[isrc_mom][i]; 
                    fwd_prop_mom[i] = 0;
                  }
                } else {
                  for( int i : {0,1,2} ) {
                    bwd_prop_mom[i] = 0;
                    fwd_prop_mom[i] = g_source_momentum_list[isrc_mom][i]; 
                  }
                }

                std::string seq_prop_key = seq_stoch_prop_meta_t(
                    seq_source_momentum,
                    correl.gf,
                    gtseq,
                    correl.sprop_flav,
                    bwd_prop_mom,
                    correl.gb,
                    correl.bprop_flav).key();

                std::string fwd_prop_key = stoch_prop_meta_t(
                    fwd_prop_mom,
                    correl.gi,
                    correl.fprop_flav).key();

                // momentum conservation in the CVC phase convention implies 
                //   p_c = -( p_src + p_seq )
                // where p_seq is interchangable with the sink momentum (in this case)
                // However, as noted above, in the contaction below we dagger the sequential
                // propagator such that our seq_source_momentum carries an implicit minus
                // sign which we compensate for here (see setting of seq_source_momentum above)
                int current_momentum[3] = {
                  -( g_source_momentum_list[isrc_mom][0] - seq_source_momentum[0] ),
                  -( g_source_momentum_list[isrc_mom][1] - seq_source_momentum[1] ),
                  -( g_source_momentum_list[isrc_mom][2] - seq_source_momentum[2] ) };

                sw.reset();
                contract_twopoint_gamma5_gamma_snk_only_snk_momentum( 
                  contr_p[isrc_mom], correl.gc, current_momentum,
                  seq_props[ seq_prop_key ].data(), props[ fwd_prop_key ].data());
                scale_cplx( contr_p[isrc_mom], T, correl.normalisation );
                sw.elapsed_print("contract_twopoint_gamma5_gamma_snk_only_snk_momentum local current and normalisation");
              }

              // as above, the sequential propagator (and hence sequential momentum projector) is daggered, 
              // so it has an implicit minus sign
              snprintf ( data_tag, 400, "/s%s%s+-g-%s/t%d/s%d/dt%d/gf%d/gc%d/gi%d/pfx%dpfy%dpfz%d/",
                         correl.bprop_flav.c_str(), correl.sprop_flav.c_str(), correl.fprop_flav.c_str(),
                         gts, isample, g_sequential_source_timeslice_list[iseq_timeslice],
                         correl.gf, correl.gc, correl.gi,
                         -seq_source_momentum[0], -seq_source_momentum[1], -seq_source_momentum[2] );

              #if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
                  exitstatus = contract_write_to_aff_file ( contr_p, affw, data_tag, g_source_momentum_list, g_source_momentum_number, io_proc );
              #elif ( defined HAVE_HDF5 )
                  exitstatus = contract_write_to_h5_file ( contr_p, output_filename, data_tag, g_source_momentum_list, g_source_momentum_number, io_proc );
              #endif
              if(exitstatus != 0) {
                fprintf(stderr, "[cpff_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
                EXIT(3);
              }
              fini_2level_dtable ( &contr_p );
            } // end of loop over local threept correlator contractions

            // now the covariant derivative insertions for which:
            // 1) we compute the first derivative insertion only for zero momentum
            // 2) we compute the second derivative insertion only for the case of zero
            //    momentum exchange and then only if the particle momentum squared is
            //    smaller or equal to 5 and then only if at least two momentum components
            //    are non-zero
            if( (seq_source_momentum[0] == 0 && 
                 seq_source_momentum[1] == 0 && 
                 seq_source_momentum[2] == 0) ||
                ( (seq_source_momentum[0]*seq_source_momentum[0] +
                   seq_source_momentum[1]*seq_source_momentum[1] +
                   seq_source_momentum[2]*seq_source_momentum[2]) <= 5 ) ){
              for( int i_d1_fwdbwd : {0,1} ){
                for( int d1_mu = 0; d1_mu < 4; d1_mu++ ){

                  
                  // restrict the cases for which the second derivative is computed
                  if( ( abs(seq_source_momentum[0]) > 0 && abs(seq_source_momentum[1]) > 0 ) ||
                      ( abs(seq_source_momentum[0]) > 0 && abs(seq_source_momentum[2]) > 0 ) ||
                      ( abs(seq_source_momentum[1]) > 0 && abs(seq_source_momentum[2]) > 0 ) ){
                    for( int i_d2_fwdbwd : {0,1} ){
                      for( int d2_mu = 0; d2_mu < 4; d2_mu++ ){
                      }
                    }
                  }
                }
              }
            }
          } // end of loop over sequential momenta
        } // end of loop over sequential source time slices
      } // if(local_threept_correls.size() > 0)

//            /*****************************************************************
//             * contractions for cov deriv insertion
//             *****************************************************************/
//
//            /*****************************************************************
//             * loop on fbwd for cov deriv
//             *****************************************************************/
//            for ( int ifbwd = 0; ifbwd <= 1; ifbwd++ ) {
//
//              double ** sequential_propagator_deriv_list     = init_2level_dtable ( 4, _GSI(VOLUME) );
//              if ( sequential_propagator_deriv_list == NULL ) {
//                fprintf(stderr, "[cpff_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
//                EXIT(33);
//              }
//
//              double ** sequential_propagator_dderiv_list     = init_2level_dtable ( 4, _GSI(VOLUME) );
//              if ( sequential_propagator_dderiv_list == NULL ) {
//                fprintf(stderr, "[cpff_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__);
//                EXIT(33);
//              }
//
//              /*****************************************************************
//               * loop on directions for cov deriv
//               *****************************************************************/
//              for ( int mu = 0; mu < 4; mu++ ) {
//
//                for ( int i = 0; i < 4; i++ ) {
//                  sw.reset();
//                  spinor_field_eq_cov_deriv_spinor_field ( sequential_propagator_deriv_list[i], sequential_propagator_list[i], mu, ifbwd, gauge_field_with_phase );
//                  sw.elapsed_print("spinor_field_eq_cov_deriv_spinor_field first derivative");
//                }
//
//                for ( int isrc_gamma = 0; isrc_gamma < g_source_gamma_id_number; isrc_gamma++ ) {
//
//                  int gamma_source = g_source_gamma_id_list[isrc_gamma];
//
//                  double ** contr_p = init_2level_dtable ( g_source_momentum_number, 2*T );
//                  if ( contr_p == NULL ) {
//                    fprintf(stderr, "[cpff_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
//                    EXIT(47);
//                  }
//
//                  /*****************************************************************
//                   * loop on source momenta
//                   *****************************************************************/
//                  for ( int isrc_mom = 0; isrc_mom < g_source_momentum_number; isrc_mom++ ) {
//
//                    int source_momentum[3] = {
//                      g_source_momentum_list[isrc_mom][0],
//                      g_source_momentum_list[isrc_mom][1],
//                      g_source_momentum_list[isrc_mom][2] };
//
//                    int current_momentum[3] = {
//                      -( source_momentum[0] + seq_source_momentum[0] ),
//                      -( source_momentum[1] + seq_source_momentum[1] ),
//                      -( source_momentum[2] + seq_source_momentum[2] ) };
//
//                    sw.reset();
//                    contract_twopoint_snk_momentum ( contr_p[isrc_mom], gamma_source,  mu, 
//                        stochastic_propagator_mom_list[isrc_mom], 
//                        sequential_propagator_deriv_list, 4, 1, current_momentum, 1);
//                    sw.elapsed_print("contract_twopoint_snk_momentum covariant derivative");
//
//                  }  /* end of loop on source momenta */
//
//                  sprintf ( data_tag, "/d+-gd-sud/t%d/s%d/dt%d/gf%d/gc%d/d%d/%s/gi%d/pfx%dpfy%dpfz%d/", 
//                      gts, isample, g_sequential_source_timeslice_list[iseq_timeslice],
//                      seq_source_gamma, mu, mu, fbwd_str[ifbwd], gamma_source,
//                      seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2] );
//
//#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
//                  exitstatus = contract_write_to_aff_file ( contr_p, affw, data_tag, g_source_momentum_list, g_source_momentum_number, io_proc );
//#elif ( defined HAVE_HDF5 )
//                  exitstatus = contract_write_to_h5_file ( contr_p, output_filename, data_tag, g_source_momentum_list, g_source_momentum_number, io_proc );
//#endif
//                  if(exitstatus != 0) {
//                    fprintf(stderr, "[cpff_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
//                    EXIT(3);
//                  }
//              
//                  fini_2level_dtable ( &contr_p );
//
//                }  /* end of loop on source gamma id */
//
//                /*****************************************************************/
//                /*****************************************************************/
//
//                for ( int kfbwd = 0; kfbwd <= 1; kfbwd++ ) {
//  
//                  for ( int i = 0; i < 4; i++ ) {
//                    sw.reset();
//                    spinor_field_eq_cov_deriv_spinor_field ( sequential_propagator_dderiv_list[i], sequential_propagator_deriv_list[i], mu, kfbwd, gauge_field_with_phase );
//                    sw.elapsed_print("spinor_field_eq_cov_deriv_spinor_field second derivative");
//                  }
//
//                  for ( int isrc_gamma = 0; isrc_gamma < g_source_gamma_id_number; isrc_gamma++ ) {
//
//                    int gamma_source = g_source_gamma_id_list[isrc_gamma];
//
//                    double ** contr_p = init_2level_dtable ( g_source_momentum_number, 2*T );
//                    if ( contr_p == NULL ) {
//                      fprintf(stderr, "[cpff_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
//                      EXIT(47);
//                    }
//
//                    /*****************************************************************
//                     * loop on source momenta
//                     *****************************************************************/
//                    for ( int isrc_mom = 0; isrc_mom < g_source_momentum_number; isrc_mom++ ) {
//
//                      int source_momentum[3] = {
//                        g_source_momentum_list[isrc_mom][0],
//                        g_source_momentum_list[isrc_mom][1],
//                        g_source_momentum_list[isrc_mom][2] };
//
//                      int current_momentum[3] = {
//                        -( source_momentum[0] + seq_source_momentum[0] ),
//                        -( source_momentum[1] + seq_source_momentum[1] ),
//                        -( source_momentum[2] + seq_source_momentum[2] ) };
//                      
//                      sw.reset();
//                      contract_twopoint_snk_momentum ( contr_p[isrc_mom], gamma_source,  4, 
//                          stochastic_propagator_mom_list[isrc_mom], 
//                          sequential_propagator_deriv_list, 4, 1, current_momentum, 1);
//                      sw.elapsed_print("contract_twopoint_snk_momentum second derivative");
//
//                    }  /* end of loop on source momenta */
//
//                    sprintf ( data_tag, "/d+-dd-sud/t%d/s%d/dt%d/gf%d/d%d/%s/d%d/%s/gi%d/pfx%dpfy%dpfz%d/", 
//                        gts, isample, g_sequential_source_timeslice_list[iseq_timeslice],
//                        seq_source_gamma, mu, fbwd_str[kfbwd], mu, fbwd_str[ifbwd], gamma_source,
//                        seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2] );
//
//#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
//                    exitstatus = contract_write_to_aff_file ( contr_p, affw, data_tag, g_source_momentum_list, g_source_momentum_number, io_proc );
//#elif ( defined HAVE_HDF5 )
//                    exitstatus = contract_write_to_h5_file ( contr_p, output_filename, data_tag, g_source_momentum_list, g_source_momentum_number, io_proc );
//#endif
//                    if(exitstatus != 0) {
//                      fprintf(stderr, "[cpff_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
//                      EXIT(3);
//                    }
//              
//                    fini_2level_dtable ( &contr_p );
//
//                  }  /* end of loop on source gamma id */
//
//                }  /* end of loop on fbwd directions k */
//
//              }  /* end of loop on directions for cov deriv */
//
//              fini_2level_dtable ( &sequential_propagator_deriv_list );
//              fini_2level_dtable ( &sequential_propagator_dderiv_list );
//
//            }  /* end of loop on fbwd */
//
//            /*****************************************************************/
//            /*****************************************************************/
//          }  /* loop on sequential source timeslices */
//
//        }  /* end of loop on sequential source gamma ids */
//
//      }  /* end of loop on sequential source momenta */
//#if 0
//#endif  /* of if 0 */
//
//      exitstatus = init_timeslice_source_oet ( NULL, -1, NULL, -2 );
//
    }  /* end of loop on oet samples */
//
//#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
//    if(io_proc == 2) {
//      const char * aff_status_str = (char*)aff_writer_close (affw);
//      if( aff_status_str != NULL ) {
//        fprintf(stderr, "[cpff_invert_contract] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
//        EXIT(32);
//      }
//    }  /* end of if io_proc == 2 */
//#endif  /* of ifdef HAVE_LHPC_AFF */
//
  }  /* end of loop on source timeslices */

  /***************************************************************************
   * decallocate spinor fields
   ***************************************************************************/
  fini_2level_dtable ( &spinor_work );

  /***************************************************************************
   * fini rng state
   ***************************************************************************/
  fini_rng_state ( &rng_state);

  /***************************************************************************
   * free the allocated memory, finalize
   ***************************************************************************/

#ifndef HAVE_TMLQCD_LIBWRAPPER
  free(g_gauge_field);
#endif
  free( gauge_field_with_phase );

  /* free clover matrix terms */
  fini_clover ( &mzz, &mzzinv );

#ifdef HAVE_MPI
  mpi_fini_xchange_contraction();
  mpi_fini_xchange_eo_spinor ();
#endif

  if(g_cart_id==0) {
    g_the_time = time(NULL);
    fprintf(stdout, "# [cpff_invert_contract_no_spin_dilution] %s# [cpff_invert_contract_no_spin_dilution] end of run\n", ctime(&g_the_time));
    fprintf(stderr, "# [cpff_invert_contract_no_spin_dilution] %s# [cpff_invert_contract_no_spin_dilution] end of run\n", ctime(&g_the_time));
  }

  return(0);

}
