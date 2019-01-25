#pragma once

#include "debug_printf.hpp"
#include "global.h"

#include <string>
#include <vector>
#include <map>

template <typename T>
static inline void extract_seq_props(const int seq_mom[3],
                                     const int seq_src_ts,
                                     const int bprop_mom[3],
                                     const T & correl,
                                     std::map<std::string, seq_stoch_prop_meta_t> & seq_props_meta)
{
  seq_stoch_prop_meta_t seq_prop(seq_mom,
                                 correl.gf,
                                 seq_src_ts,
                                 correl.sprop_flav,
                                 bprop_mom,
                                 correl.gb,
                                 correl.bprop_flav);

  seq_props_meta[ seq_prop.key() ] = seq_prop;
}

template <typename T>
void get_seq_props_meta_from_npt_oet_meta(const std::vector<T> & correls,
                                          const int seq_mom[3],
                                          const int seq_src_ts,
                                          std::map<std::string, seq_stoch_prop_meta_t> & seq_props_meta)
{
  // iterate through the correlation function definitions and create
  // a map of all the unique sequential propagators required
  // to compute the requested correlation functions
  // we use a map to avoid duplicates
  for( int icor = 0; icor < correls.size(); ++icor ){
    debug_printf(0, 0,
                 "parsing 3pt function [ g%d %s^dag g5 g%d g5 %s^dag g5 g%d %s g%d ] with src_mom on %s prop\n",
                 correls[icor].gb,
                 correls[icor].bprop_flav.c_str(),
                 correls[icor].gf,
                 correls[icor].sprop_flav.c_str(),
                 correls[icor].gc,
                 correls[icor].fprop_flav.c_str(),
                 correls[icor].gi,
                 correls[icor].src_mom_prop.c_str());
    if( correls[icor].src_mom_prop == "bwd" ){
      // if the source momentum is carried by the backward propagator
      // which is part of the sequential propagator, we need to generate
      // one sequential propagator per source momentum
      // since in the contractions we dagger the sequential propagator,
      // the backward propagator corresponding to the correct momentum
      // is the one with the source momentum projector daggered
      for(int isrc_mom = 0; isrc_mom < g_source_momentum_number; ++isrc_mom){
        const int bprop_mom[3] = { -g_source_momentum_list[isrc_mom][0],
                                   -g_source_momentum_list[isrc_mom][1],
                                   -g_source_momentum_list[isrc_mom][2] };
        extract_seq_props(seq_mom,
                          seq_src_ts,
                          bprop_mom,
                          correls[icor],
                          seq_props_meta);
      }
    } else {
      const int bprop_mom[3] = {0,0,0};
      extract_seq_props(seq_mom,
                        seq_src_ts,
                        bprop_mom,
                        correls[icor],
                        seq_props_meta);
    }
  } // end of loop over correlation functions to generate map of seq propagator metadata
  
}

