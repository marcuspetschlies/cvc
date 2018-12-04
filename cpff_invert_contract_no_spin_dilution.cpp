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
#include "types.h"

#include "init_g_gauge_field.hpp"
#include "Stopwatch.hpp"
#include "Core.hpp"
#include "enums.hpp"
#include "ParallelMT19937_64.hpp"

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
  Core core(argc, argv);

  std::vector<twopt_oet_meta_t> twopt_correls;
  std::vector<threept_oet_meta_t> threept_correls;
  
  std::map<std::string, stoch_prop_meta_t> props_meta;
  std::map<std::string, std::vector<double>> props;

  std::map<std::string, int> flav_op_ids;
  flav_op_ids["u"] = 0;
  flav_op_ids["d"] = 1;
  flav_op_ids["sp"] = 2;
  flav_op_ids["sm"] = 3;
  flav_op_ids["cp"] = 4;
  flav_op_ids["cm"] = 5;

  const bool do_singlet = true;
  const bool use_g0_g5_pion = true;
  const bool use_g0_g5_singlet = true;

  // \bar{chi}_d g5 chi_u \bar{chi}_u g5 chi_d
  twopt_correls.push_back( twopt_oet_meta_t("u", "u", "fwd", 5, 5, 5) );
  if( use_g0_g5_pion ){
    // \bar{chi}_d g0 chi_u \bar{chi}_u g5 chi_d 
    twopt_correls.push_back( twopt_oet_meta_t("u", "u", "fwd", 5, 0, 5) );
    // \bar{chi}_d g5 chi_u \bar{chi}_u g0 chi_d
    twopt_correls.push_back( twopt_oet_meta_t("u", "u", "fwd", 0, 5, 5) );
    // \bar{chi}_d g0 chi_u \bar{chi}_u g0 chi_d
    twopt_correls.push_back( twopt_oet_meta_t("u", "u", "fwd", 0, 0, 5) ); 
  }

  if( do_singlet ){
    // \bar{chi}_u chi_u \bar{chi}_u chi_u
    twopt_correls.push_back( twopt_oet_meta_t("u", "d", "fwd", 4, 4, 5) );
    twopt_correls.push_back( twopt_oet_meta_t("d", "u", "fwd", 4, 4, 5) );
    if( use_g0_g5_singlet ){
      // \bar{chi}_u g0 g5 chi_u  \bar{chi}_u chi_u
      twopt_correls.push_back( twopt_oet_meta_t("u", "d", "fwd", 4, 6, 5) );
      twopt_correls.push_back( twopt_oet_meta_t("d", "u", "fwd", 4, 6, 5) );
      // \bar{chi}_u chi_u  \bar{chi}_u g5 g0 chi_u =
      // - \bar{chi}_u chi_u  \bar{chi}_u g0 g5 chi_u
      twopt_correls.push_back( twopt_oet_meta_t("u", "d", "fwd", 0, 5, 4) );
      twopt_correls.push_back( twopt_oet_meta_t("d", "u", "fwd", 0, 5, 4) );
      // \bar{chi}_u g0 g5 chi_u  \bar{chi}_u g5 g0 chi_u =
      // - \bar{chi}_u g0 g5 chi_u  \bar{chi}_u g0 g5 chi_u
      twopt_correls.push_back( twopt_oet_meta_t("u", "d", "fwd", 0, 6, 4) );
      twopt_correls.push_back( twopt_oet_meta_t("d", "u", "fwd", 0, 6, 4) );
    }
  }

  // iterate through the correlation function definitions and create
  // a map of all the forward and backward propagators required
  // to compute the requested correlation functions
  // we use maps to avoid duplicates
  for( int icor = 0; icor < twopt_correls.size(); ++icor ){
    debug_printf(0, 0, 
                 "2pt function [ g%02d %s^dag g05 g%02d %s g%02d ] with src_mom at on %s prop\n",
                 twopt_correls[icor].gb, 
                 twopt_correls[icor].bprop_flav.c_str(),
                 twopt_correls[icor].gf,
                 twopt_correls[icor].fprop_flav.c_str(),
                 twopt_correls[icor].gi,
                 twopt_correls[icor].src_mom_prop.c_str());

    for( int isrc_mom = 0; isrc_mom < g_source_momentum_number; ++isrc_mom ){
      int source_momentum[3];
      // if the source momentum is carried by the backward propagator,
      // we need to dagger the momentum projector
      if( twopt_correls[icor].src_mom_prop == "bwd" ){
        source_momentum[0] = -g_source_momentum_list[isrc_mom][0];
        source_momentum[1] = -g_source_momentum_list[isrc_mom][1];
        source_momentum[2] = -g_source_momentum_list[isrc_mom][2];
        stoch_prop_meta_t bwd_prop( source_momentum,
                                    twopt_correls[icor].gb,
                                    twopt_correls[icor].bprop_flav );
        std::string prop_key = bwd_prop.make_key();
        props_meta[ prop_key ] = bwd_prop;
      } else {
        source_momentum[0] = g_source_momentum_list[isrc_mom][0];
        source_momentum[1] = g_source_momentum_list[isrc_mom][1];
        source_momentum[2] = g_source_momentum_list[isrc_mom][2];
        stoch_prop_meta_t fwd_prop( source_momentum,
                                    twopt_correls[icor].gi,
                                    twopt_correls[icor].fprop_flav );
        std::string prop_key = fwd_prop.make_key();
        props_meta[ prop_key ] = fwd_prop;
      }
    } // loop over source momenta
    // if the source momentum is carried by the backward propagator,
    // we need a corresponding zero momentum forward propagator
    if( twopt_correls[icor].src_mom_prop == "bwd" ){
      int source_momentum[3] = {0,0,0};
      stoch_prop_meta_t fwd_prop( source_momentum,
                                  twopt_correls[icor].gi,
                                  twopt_correls[icor].fprop_flav );
      props_meta[ fwd_prop.make_key() ] = fwd_prop;
    // if it is carried by the forward propagator, instead,
    // we need a corresponding 
    } else {
      int bprop_momentum[3] = {0,0,0};
      stoch_prop_meta_t bwd_prop( bprop_momentum,
                                  twopt_correls[icor].gb,
                                  twopt_correls[icor].bprop_flav );
      props_meta[ bwd_prop.make_key() ] = bwd_prop;
    }
  } // end of loop over two pt functions to generate map of fwd/bwd propagator metadata

  sizeof_spinor_field    = _GSI(VOLUME) * sizeof(double);

  debug_printf(0, 0, "%d propagators\n will use %f GB of memory\n\n", 
      props_meta.size(),
      1.0e-9*props_meta.size()*(double)sizeof_spinor_field*g_nproc );

  debug_printf(0, 0, "\n\nList of non-zero momentum propagators to be generated\n");
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
  CHECK_EXITSTATUS_NONZERO(exitstatus,
      init_g_gauge_field(),
      "[cpff_invert_contract] Error initialising gauge field!",
      true, CVC_EXIT_GAUGE_INIT_FAILURE);
  // set the configuration number
#ifdef HAVE_TMLQCD_LIBWRAPPER
  Nconf = g_tmLQCD_lat.nstore;
#endif

  /***************************************************************************
   * multiply the temporal phase to the gauge field
   ***************************************************************************/
  CHECK_EXITSTATUS_NONZERO(exitstatus,
      gauge_field_eq_gauge_field_ti_phase ( &gauge_field_with_phase, g_gauge_field, co_phase_up ),
      "[cpff_invert_contract] Error from gauge_field_eq_gauge_field_ti_phase!",
      true, CVC_EXIT_UTIL_FUNCTION_FAILURE);

  /***************************************************************************
   * check plaquettes
   ***************************************************************************/
  CHECK_EXITSTATUS_NONZERO(exitstatus,
      plaquetteria( gauge_field_with_phase ),
      "[cpff_invert_contract] Error from plaquetteria!",
      true, CVC_EXIT_UTIL_FUNCTION_FAILURE);

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
  
  const size_t nelem = _GSI( VOLUME+RAND );

  // memory for a random spinor and the non-spin-diluted stochastic source
  std::vector<double> ranspinor(nelem);
  std::vector<double> stochastic_source(nelem);

  /***************************************************************************
   * allocate memory for spinor fields 
   * WITH HALO
   ***************************************************************************/
  double ** spinor_work  = init_2level_dtable ( 2, nelem );
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
      rng.gen_z2(ranspinor.data(), 24);
      // we can write the random spinor to file if we want
      if ( g_write_source ) {
        sprintf(filename, "%s.conf%.4d.t%d.sample%.5d", filename_prefix, Nconf, gts, isample);
        if ( ( exitstatus = write_propagator( ranspinor.data(), filename, 0, g_propagator_precision) ) != 0 ) {
          fprintf(stderr, "[cpff_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          EXIT(2);
        }
      }
      

      // loop over all required propagators
      for( auto const & prop : props_meta ){
        stoch_prop_meta_t prop_meta = prop.second;
        std::string prop_key = prop_meta.make_key();

        /***************************************************************************
         * prepare stochastic timeslice source at source momentum
         ***************************************************************************/
        CHECK_EXITSTATUS_NONZERO(exitstatus,
          prepare_gamma_timeslice_oet(stochastic_source.data(),
                                      ranspinor.data(),
                                      prop_meta.gamma,
                                      gts,
                                      (prop_meta.p[0] == 0 && 
                                       prop_meta.p[1] == 0 &&
                                       prop_meta.p[2] == 0 ) ? NULL : prop_meta.p ),
          "[cpff_invert_contract] Error from prepare_gamma_timeslice_oet!",
          true, CVC_EXIT_MALLOC_FAILURE);

        memcpy ( spinor_work[0], stochastic_source.data(), sizeof_spinor_field );
        memset ( spinor_work[1], 0, sizeof_spinor_field );

        debug_printf(0, 0, "Inverting to generate propagator %s\n", prop_meta.make_key().c_str() );

        sw.reset();
        exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], flav_op_ids[ prop_meta.flav ] );
        if(exitstatus < 0) {
          fprintf(stderr, "[cpff_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
          EXIT(44);
        }
        sw.elapsed_print("TMQLCD_INVERT");

        if ( g_check_inversion ) {
          sw.reset();
          check_residual_clover (&(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, 
                                 mzz[ flav_op_ids[ prop_meta.flav] ], mzzinv[ flav_op_ids[ prop_meta.flav] ], 1 );
          sw.elapsed_print("check_residual_clover");
        }

        if( props.count( prop_key ) == 0 ){
          props.emplace( std::make_pair( prop_key,
                                         std::vector<double>(nelem) ) );
        }
        memcpy( props[ prop_key ].data(), spinor_work[1], sizeof_spinor_field);

        if ( g_write_propagator ) {
          sprintf(filename, "%s.conf%.4d.t%d.px%dpy%dpz%d.gamma%02d.f%s.sample%.5d.inverted", 
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
          // negative momemntum
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

            sw.reset();
            contract_twopoint_xdep_snk_gamma_only( contr_x, 
                twopt_correls[icor].gf, 
                props[ bwdprop_meta.make_key() ].data(), 
                props[ fwdprop_meta.make_key() ].data(),
                1 /*stride*/, 1.0 /*factor*/);
            sw.elapsed_print("contract_twopoint_xdep");

            /* momentum projection at sink */
            sw.reset();
            exitstatus = momentum_projection ( contr_x, contr_p[0], T, g_sink_momentum_number, g_sink_momentum_list );
            if(exitstatus != 0) {
              fprintf(stderr, "[cpff_invert_contract] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              EXIT(3);
            }
            sw.elapsed_print("sink momentum_projection");

            sprintf ( data_tag, "/%s/t%d/s%d/gf%d/gi%d/pix%dpiy%dpiz%d", 
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
        
//
//      /*****************************************************************/
//      /*****************************************************************/
//
//      /*****************************************************************
//       * loop on sequential source momenta p_f
//       *****************************************************************/
//      for( int iseq_mom=0; iseq_mom < g_seq_source_momentum_number; iseq_mom++) {
//
//        int seq_source_momentum[3] = { g_seq_source_momentum_list[iseq_mom][0], g_seq_source_momentum_list[iseq_mom][1], g_seq_source_momentum_list[iseq_mom][2] };
//
//        /*****************************************************************
//         * loop on sequential source gamma ids
//         *****************************************************************/
//        for ( int iseq_gamma = 0; iseq_gamma < g_sequential_source_gamma_id_number; iseq_gamma++ ) {
//
//          int seq_source_gamma = g_sequential_source_gamma_id_list[iseq_gamma];
//
//          /*****************************************************************
//           * loop on sequential source timeslices
//           *****************************************************************/
//          for ( int iseq_timeslice = 0; iseq_timeslice < g_sequential_source_timeslice_number; iseq_timeslice++ ) {
//
//            /*****************************************************************
//             * global sequential source timeslice
//             * NOTE: counted from current source timeslice
//             *****************************************************************/
//            int gtseq = ( gts + g_sequential_source_timeslice_list[iseq_timeslice] + T_global ) % T_global;
//
//            /*****************************************************************
//             * invert for sequential timeslice propagator
//             *****************************************************************/
//            for ( int i = 0; i < 4; i++ ) {
//
//              /*****************************************************************
//               * prepare sequential timeslice source 
//               *****************************************************************/
//              exitstatus = init_sequential_source ( spinor_work[0], stochastic_propagator_zero_list[i], gtseq, seq_source_momentum, seq_source_gamma );
//              if( exitstatus != 0 ) {
//                fprintf(stderr, "[cpff_invert_contract] Error from init_sequential_source, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
//                EXIT(64);
//              }
//              if ( g_write_sequential_source ) {
//                sprintf(filename, "%s.%.4d.t%d.qx%dqy%dqz%d.g%d.dt%d.%d.%.5d", filename_prefix, Nconf, gts,
//                    seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], seq_source_gamma,
//                    g_sequential_source_timeslice_list[iseq_timeslice], i, isample);
//                if ( ( exitstatus = write_propagator( spinor_work[0], filename, 0, g_propagator_precision) ) != 0 ) {
//                  fprintf(stderr, "[cpff_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
//                  EXIT(2);
//                }
//              }  /* end of if g_write_sequential_source */
//
//              memset ( spinor_work[1], 0, sizeof_spinor_field );
//
//              exitstatus = _TMLQCD_INVERT ( spinor_work[1], spinor_work[0], op_id_up );
//              if(exitstatus < 0) {
//                fprintf(stderr, "[cpff_invert_contract] Error from invert, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
//                EXIT(44);
//              }
//
//              if ( g_check_inversion ) {
//                check_residual_clover ( &(spinor_work[1]), &(spinor_work[0]), gauge_field_with_phase, mzz[op_id_up], mzzinv[op_id_up], 1 );
//              }
//
//              memcpy( sequential_propagator_list[i], spinor_work[1], sizeof_spinor_field );
//            }  /* end of loop on oet spin components */
//
//            if ( g_write_sequential_propagator ) {
//              for ( int ispin = 0; ispin < 4; ispin++ ) {
//                sprintf ( filename, "%s.%.4d.t%d.qx%dqy%dqz%d.g%d.dt%d.%d.%.5d.inverted", filename_prefix, Nconf, gts,
//                    seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2], seq_source_gamma, 
//                    g_sequential_source_timeslice_list[iseq_timeslice], ispin, isample);
//                if ( ( exitstatus = write_propagator( sequential_propagator_list[ispin], filename, 0, g_propagator_precision) ) != 0 ) {
//                  fprintf(stderr, "[cpff_invert_contract] Error from write_propagator, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
//                  EXIT(2);
//                }
//              }
//            }  /* end of if g_write_sequential_propagator */
//
//            /*****************************************************************/
//            /*****************************************************************/
//
//            /*****************************************************************
//             * contractions for local current insertion
//             *****************************************************************/
//
//            /*****************************************************************
//             * loop on local gamma matrices
//             *****************************************************************/
//            for ( int icur_gamma = 0; icur_gamma < gamma_current_number; icur_gamma++ ) {
//
//              int gamma_current = gamma_current_list[icur_gamma];
//
//              for ( int isrc_gamma = 0; isrc_gamma < g_source_gamma_id_number; isrc_gamma++ ) {
//
//                int gamma_source = g_source_gamma_id_list[isrc_gamma];
//
//                double ** contr_p = init_2level_dtable ( g_source_momentum_number, 2*T );
//                if ( contr_p == NULL ) {
//                  fprintf(stderr, "[cpff_invert_contract] Error from init_2level_dtable %s %d\n", __FILE__, __LINE__ );
//                  EXIT(47);
//                }
//
//                /*****************************************************************
//                 * loop on source momenta
//                 *****************************************************************/
//                for ( int isrc_mom = 0; isrc_mom < g_source_momentum_number; isrc_mom++ ) {
//
//                  int source_momentum[3] = {
//                    g_source_momentum_list[isrc_mom][0],
//                    g_source_momentum_list[isrc_mom][1],
//                    g_source_momentum_list[isrc_mom][2] };
//
//                  int current_momentum[3] = {
//                    -( source_momentum[0] + seq_source_momentum[0] ),
//                    -( source_momentum[1] + seq_source_momentum[1] ),
//                    -( source_momentum[2] + seq_source_momentum[2] ) };
//
//                  sw.reset();
//                  contract_twopoint_snk_momentum ( contr_p[isrc_mom], gamma_source,  gamma_current, 
//                      stochastic_propagator_mom_list[isrc_mom], 
//                      sequential_propagator_list, 4, 1, current_momentum, 1);
//                  sw.elapsed_print("contract_twopoint_snk_momentum local current");
//
//                }  /* end of loop on source momenta */
//
//                sprintf ( data_tag, "/d+-g-sud/t%d/s%d/dt%d/gf%d/gc%d/gi%d/pfx%dpfy%dpfz%d/", 
//                    gts, isample, g_sequential_source_timeslice_list[iseq_timeslice],
//                    seq_source_gamma, gamma_current, gamma_source,
//                    seq_source_momentum[0], seq_source_momentum[1], seq_source_momentum[2] );
//
//#if ( defined HAVE_LHPC_AFF ) && ! ( defined HAVE_HDF5 )
//                exitstatus = contract_write_to_aff_file ( contr_p, affw, data_tag, g_source_momentum_list, g_source_momentum_number, io_proc );
//#elif ( defined HAVE_HDF5 )
//                exitstatus = contract_write_to_h5_file ( contr_p, output_filename, data_tag, g_source_momentum_list, g_source_momentum_number, io_proc );
//#endif
//                if(exitstatus != 0) {
//                  fprintf(stderr, "[cpff_invert_contract] Error from contract_write_to_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
//                  EXIT(3);
//                }
//              
//                fini_2level_dtable ( &contr_p );
//
//              }  /* end of loop on source gamma id */
//
//            }  /* end of loop on current gamma id */
//
//            /*****************************************************************/
//            /*****************************************************************/
//
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
