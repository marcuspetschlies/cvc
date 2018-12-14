#pragma once

#include "debug_printf.hpp"

#include <string>
#include <vector>
#include <map>

template <typename T>                                                                                                                                                                                                                                      
void get_fwd_bwd_props_meta_from_npt_oet_meta(const std::vector<T> & correls,
                                              std::map<std::string, stoch_prop_meta_t> & props_meta)
{
  // iterate through the correlation function definitions and create
  // a map of all the forward and backward propagators required
  // to compute the requested correlation functions
  // we use maps to avoid duplicates
  for( int icor = 0; icor < correls.size(); ++icor ){
    debug_printf(0, 0,
                 "parsing 2pt function [ g%d %s^dag g5 g%d %s g%d ] with src_mom on %s prop\n",
                 correls[icor].gb,
                 correls[icor].bprop_flav.c_str(),
                 correls[icor].gf,
                 correls[icor].fprop_flav.c_str(),
                 correls[icor].gi,
                 correls[icor].src_mom_prop.c_str());

    for( int isrc_mom = 0; isrc_mom < g_source_momentum_number; ++isrc_mom ){
      int source_momentum[3];
      // if the source momentum is carried by the backward propagator,
      // we need to dagger the momentum projector
      if( correls[icor].src_mom_prop == "bwd" ){
        source_momentum[0] = -g_source_momentum_list[isrc_mom][0];
        source_momentum[1] = -g_source_momentum_list[isrc_mom][1];
        source_momentum[2] = -g_source_momentum_list[isrc_mom][2];
        stoch_prop_meta_t bwd_prop( source_momentum,
                                    correls[icor].gb,
                                    correls[icor].bprop_flav );
        std::string prop_key = bwd_prop.make_key();
        props_meta[ prop_key ] = bwd_prop;
      // source momentum can also be carried by the forward propagator,
      // no daggering is necessary
      } else if( correls[icor].src_mom_prop == "fwd" ) {
        source_momentum[0] = g_source_momentum_list[isrc_mom][0];
        source_momentum[1] = g_source_momentum_list[isrc_mom][1];
        source_momentum[2] = g_source_momentum_list[isrc_mom][2];
        stoch_prop_meta_t fwd_prop( source_momentum,
                                    correls[icor].gi,
                                    correls[icor].fprop_flav );
        std::string prop_key = fwd_prop.make_key();
        props_meta[ prop_key ] = fwd_prop;
      } else {
      // finally, a correlation function might have been specified
      // with src_mom_prop == "neither". In this case only the zero
      // momentum forward propagator should be added to the map
      // and the corresponding backward propagator will be added
      // below
        int source_momentum[3] = {0,0,0};
        stoch_prop_meta_t fwd_prop( source_momentum,
                                    correls[icor].gi,
                                    correls[icor].fprop_flav );
        std::string prop_key = fwd_prop.make_key();
        props_meta[ prop_key ] = fwd_prop;
      }
    } // loop over source momenta

    // if the source momentum is carried by the backward propagator,
    // we need a corresponding zero momentum forward propagator
    if( correls[icor].src_mom_prop == "bwd" ){
      int source_momentum[3] = {0,0,0};
      stoch_prop_meta_t fwd_prop( source_momentum,
                                  correls[icor].gi,
                                  correls[icor].fprop_flav );
      props_meta[ fwd_prop.make_key() ] = fwd_prop;
    // if it is carried by the forward propagator, instead,
    // we need a corresponding backward propagator
    } else {
      int bprop_momentum[3] = {0,0,0};
      stoch_prop_meta_t bwd_prop( bprop_momentum,
                                  correls[icor].gb,
                                  correls[icor].bprop_flav );
      props_meta[ bwd_prop.make_key() ] = bwd_prop;
    }
  } // end of loop over two pt functions to generate map of fwd/bwd propagator metadata
}

