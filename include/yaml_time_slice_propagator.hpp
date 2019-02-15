#pragma once

#include <map>
#include <string>
#include <vector>

namespace cvc {
namespace yaml {

/**
* @brief Parse a 'TimeSlicePropagator' object in the object definitions file
*
* @param node Node of the 'TimeSlicePropagator'
* @param verbose Trigger verbose output
* @param mom_lists Reference to map of vector of momentum triplets (input)
* @param props_meta Reference to map of meta info for the propagators (output)
*/
void construct_time_slice_propagator(const YAML::Node &node, 
                                     const bool verbose,
                                     std::map< std::string, std::vector< std::vector<int> > > & mom_lists,
                                     std::map< std::string, cvc::stoch_prop_meta_t > & props_meta);

} //namespace(yaml)
} //namespace(cvc)

