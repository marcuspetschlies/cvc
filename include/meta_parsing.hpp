#pragma once

#include "meta_types.hpp"
#include <string>
#include <vector>
#include <map>

namespace cvc {

  /**
   * @brief extract information about fwd and bwd propagators from npt function meta
   *
   * Extract meta-data about fwd and backward propagators from a vector of correlator
   * meta-data types [two,three,...]pt_oet_meta_t. The templating here is a hack
   * which allows this to be used for a vector of any type, but it is still
   * kind of type-safe because if T does not contain the members required
   * by the function, the compiler will complain. 
   *
   * Since The alternative would
   * have been to pass begin and end iterators (like in the standard libary)
   *
   * @tparam T correlator meta-definition type twopt_oet_meta_t, threept_oet_meta_t ...
   * @param correls const reference to vector of correlator meta-data, in 
   * @param props_meta reference to map of propagator meta-data, out
   */
template <typename T>                                                                                                                                                                                                                                      
void get_fwd_bwd_props_meta_from_npt_oet_meta(const std::vector<T> & correls,
                                              std::map<std::string, stoch_prop_meta_t> & props_meta);
#include "impl/impl_get_fwd_bwd_props_meta_from_npt_oet_meta.hpp"

/**
 * @brief extract information about seq propagators from npt function meta
 *
 * Extract meta-data about sequential propagators from a vector
 * of correlator meta-data types [three,four,...]pt_oet_meta_t. The templating is
 * a hack, as explained for get_fwd_bwd_props_meta_from_npt_oet_meta.
 *
 * @tparam T correlator meta definition type threept_oet_meta_t, fourpt_oet_meta_t, ... 
 * @param correls const reference to vector of correlator meta-data, in
 * @param seq_mom the sequential source momentum (this is generally the sink momentum), in
 * @param seq_src_ts global time slice at which the source of the sequential propagator
 *                   is to sit, in
 * @param seq_props_meta reference to map of seq propagator meta-data, out
 */
template <typename T>
void get_seq_props_meta_from_npt_oet_meta(const std::vector<T> & correls,
                                          const int seq_mom[3],
                                          const int seq_src_ts,
                                          std::map<std::string, seq_stoch_prop_meta_t> & seq_props_meta);
#include "impl/impl_get_seq_props_meta_from_npt_oet_meta.hpp"

} // namespace(cvc)
