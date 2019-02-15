#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <exception>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "meta_types.hpp"
#include "algorithms.hpp"
#include "yaml_parsers.hpp"

std::map< std::string, std::vector< std::vector<int> > > momentum_lists;
std::map< std::string, cvc::stoch_prop_meta_t > ts_prop_meta; 

int main(int argc, char ** argv){
  cvc::MetaCollection metas;

  YAML::Node input_node = YAML::LoadFile("definitions.yaml");
  cvc::yaml::enter_node(input_node, 0, metas, true); std::cout << std::endl;
  return 0;
}
