#pragma once

#include <yaml-cpp/yaml.h>
#include "meta_types.hpp"

namespace cvc {
namespace yaml {

  /**
   * @brief Driver for parsing a YAML node hierarchy for CVC object definitions
   *
   * @param node Starting node.
   * @param depth Current depth.
   * @param metas Collection of maps of meta types.
   * @param verbose Toggle verbose output.
   */
  void enter_node(const YAML::Node &node,
                  const unsigned int depth, 
                  MetaCollection & metas,
                  const bool verbose = true);

} //namespace(yaml)
} //namespace(cvc)
