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

void enter_node(const YAML::Node &node, const unsigned int depth, const bool verbose = true){
  YAML::NodeType::value type = node.Type();
  std::string indent( 2*(size_t)depth, ' ');
  switch(type){
    case YAML::NodeType::Scalar:
      if(verbose){
        std::cout << node;
      }
      break;
    case YAML::NodeType::Sequence:
      for(unsigned int i = 0; i < node.size(); ++i){
        if(verbose && depth <= 2){
          std::cout << "[ ";
        }
        const YAML::Node & subnode = node[i];
        enter_node(subnode, depth+1);
        if(verbose){
          if( depth <= 2 ){
            std::cout << " ]";
          } else if( i < node.size()-1 ) {
            std::cout << ",";
          }
        }
      }
      break;
    case YAML::NodeType::Map:
      for(YAML::const_iterator it = node.begin(); it != node.end(); ++it){
        if(verbose){
          if(depth != 0){ std::cout << std::endl << indent; }
          std::cout << it->first << ": ";
        }
        if( it->first.as<std::string>() == "MomentumList" ){
          construct_momentum_list(it->second, verbose, momentum_lists);
        } else if ( it->first.as<std::string>() == "TimeSlicePropagator" ){
          construct_time_slice_propagator(it->second, verbose, momentum_lists, ts_prop_meta);
        } else {
          char msg[200];
          snprintf(msg, 200,
                   "%s is not a valid Object name\n",
                   it->first.as<std::string>().c_str());
          throw( std::invalid_argument(msg) );
        }
      }
    case YAML::NodeType::Null:
      if(verbose)
        std::cout << std::endl;
      break;
    default:
      break;
  }
  for(const auto & prop : ts_prop_meta){
    std::cout << prop.first << std::endl;
  }
}


int main(int argc, char ** argv){
  YAML::Node input_node = YAML::LoadFile("definitions.yaml");
  enter_node(input_node, 0); std::cout << std::endl;
  return 0;
}
