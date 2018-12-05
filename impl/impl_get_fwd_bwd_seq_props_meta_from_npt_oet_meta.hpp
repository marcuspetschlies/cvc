#pragma once

#include "debug_printf.hpp"

#include <string>
#include <vector>
#include <map>

template <typename T>
void get_fwd_bwd_seq_props_meta_from_npt_oet_meta(const std::vector<T> & correls,
                                                  std::map<std::string, stoch_prop_meta_t> & props_meta,
                                                  std::map<std::string, stoch_prop_meta_t> & seq_props_meta)
{
  // first let the corresponding function for twopt_oet_meta_t extract all the necessary
  // forward and backward propagators by casting to the base class
  get_fwd_bwd_props_meta_from_npt_oet_meta( correls,
                                            props_meta );

  // iterate through the correlation function definitions and create
  // a map of all the forward, backward and sequential propagators required
  // to compute the requested correlation functions
  // we use maps to avoid duplicates
  for( int icor = 0; icor < correls.size(); ++icor ){
    debug_printf(0, 0,
                 "parsing 3pt function [ g%02d %s^dag g05 g%02d g05 %s^dag g05 g%02d %s g%02d ] with src_mom on %s prop\n",
                 correls[icor].gb,
                 correls[icor].bprop_flav.c_str(),
                 correls[icor].gf,
                 correls[icor].sprop_flav.c_str(),
                 correls[icor].gc,
                 correls[icor].fprop_flav.c_str(),
                 correls[icor].gi,
                 correls[icor].src_mom_prop.c_str());
  } // end of loop over 3pt functions to generate map of fwd/bwd/seq propagator metadata
  
}

