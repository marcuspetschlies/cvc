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

#include <meta_types.hpp>

std::map< std::string, std::vector< std::vector<int> > > momentum_lists;
std::map< std::string, cvc::stoch_prop_meta_t > ts_prop_meta; 

// we sort momenta by psq first.
// For equal psq, we sort by maximum component: (2,2,1) < (0,0,3)
// and then by the number of non-zeros (not sure if this last condition would ever occur)
struct momentum_sort {
  bool operator()( const std::vector<int> & a, const std::vector<int> & b ){
    if( a.size() != 3 || b.size() != 3 ){
      throw( std::invalid_argument("in momentum_sort, 'a' and 'b' must of size 3!") );
    }
    int asq = a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
    int anonzero = static_cast<int>( a[0] != 0 ) + static_cast<int>( a[1] != 0 ) + static_cast<int>( a[2] != 0 );
    int amax = 0;
    for( const auto & p : a ){
      amax = abs(p) > amax ? abs(p) : amax;
    }

    int bsq = b[0]*b[0] + b[1]*b[1] + b[2]*b[2];
    int bnonzero = static_cast<int>( b[0] != 0 ) + static_cast<int>( b[1] != 0 ) + static_cast<int>( b[2] != 0 );
    int bmax = 0;
    for( const auto & p : b ){
      bmax = abs(p) > bmax ? abs(p) : bmax;
    }

    if( asq == bsq ){
      if( amax == bmax ){
        return anonzero < bnonzero;
      } else {
        return amax < bmax;
      }
    } else {
      return asq < bsq;
    }
  }
} momentum_sort;

std::vector< std::vector<int> > parse_momentum_list(const YAML::Node & node)
{
  if( node.Type() != YAML::NodeType::Sequence ){
    throw( std::invalid_argument("in parse_momentum_list, 'node' must be of type YAML::NodeType::Sequence\n") );
  }
  std::vector< std::vector<int> > momenta;
  for(size_t i = 0; i < node.size(); ++i){
    std::vector<int> mom{ node[i][0].as<int>(), node[i][1].as<int>(), node[i][2].as<int>() };
    momenta.push_back(mom);
  }
  std::sort( momenta.begin(), momenta.end(), momentum_sort);
  return momenta;
}

std::vector< std::vector<int> > psq_to_momentum_list(const YAML::Node & node)
{
  if( node.Type() != YAML::NodeType::Scalar ){
    throw( std::invalid_argument("in psq_to_momentum_list, 'node' must be of type YAML::NodeType::Scalar\n") );
  }
  const int psqmax = node.as<int>();
  const int pmax = static_cast<int>(sqrt( node.as<double>() )); 
  std::vector< std::vector<int> > momenta;
  for( int px = -pmax; px <= pmax; ++px ){
    for( int py = -pmax; py <= pmax; ++py ){
      for( int pz = -pmax; pz <= pmax; ++pz ){
        if( (px*px + py*py + pz*pz) <= psqmax ){
          std::vector<int> mom{ px, py, pz };
          momenta.push_back(mom);
        }
      }
    }
  }
  std::sort( momenta.begin(), momenta.end(), momentum_sort );
  return momenta; 
}

void construct_momentum_list(const YAML::Node & node, 
                             const bool verbose,
                             std::map< std::string, std::vector< std::vector<int> > > & mom_lists )
{
  if( node.Type() != YAML::NodeType::Map ){
    throw( std::invalid_argument("in construct_momentum_list, 'node' must be of type YAML::NodeType::Map\n") );
  }
  if( !node["id"] || !(node["Psqmax"] || node["Plist"]) ){
    throw( std::invalid_argument("For 'MomentumList', the 'id' property and one of 'Psqmax' or 'Plist' must be defined!\n") );
  }

  std::string id;
  std::vector<std::vector<int>> momentum_list;
  for(YAML::const_iterator it = node.begin(); it != node.end(); ++it){
    if(verbose){
      std::cout << "\n  " << it->first << ": " << it->second;
    }
    if( it->first.as<std::string>() == "id" ){
      id = it->second.as<std::string>();
    } else if( it->first.as<std::string>() == "Plist" ){
      momentum_list = parse_momentum_list( it->second );
    } else if( it->first.as<std::string>() == "Psqmax" ){
      momentum_list = psq_to_momentum_list( it->second );
    } else {
      char msg[200];
      snprintf(msg, 200,
               "%s is not a valid property for a MomentumList!\n",
               it->first.as<std::string>().c_str());
      throw( std::invalid_argument(msg) );
    }
  }
  if( verbose ){
    std::cout << std::endl << "   ";
    for( const auto & mom : momentum_list ){
      std::cout << "[";
      for(size_t i = 0; i < mom.size(); ++i){
        std::cout << mom[i];
        if( i < mom.size()-1 ){
          std::cout << ",";
        } else {
          std::cout << "] ";
        } 
      }
    }
    std::cout << std::endl;
  }
  mom_lists[id] = momentum_list;
}

void construct_time_slice_propagator(const YAML::Node &node, 
                                     const bool verbose,
                                     std::map< std::string, std::vector< std::vector<int> > > & mom_lists,
                                     std::map< std::string, cvc::stoch_prop_meta_t > & props_meta) {
  if( node.Type() != YAML::NodeType::Map ){
    throw( std::invalid_argument("in construct_time_slice_propagator, 'node' must be of type YAML::NodeType::Map\n") );
  }
  if( !(node["id"]) || !(node["solver_id"]) || !(node["solver_driver"]) ||
      !(node["g_src"]) || !(node["P_src"]) ){
    throw( std::invalid_argument("for TimeSlicePropagator, the properties 'id', 'solver_id',"
                                 " 'solver_driver', 'g_src' and 'P_src' must be defined!\n") );
  }
  if( verbose ){
    for(YAML::const_iterator it = node.begin(); it != node.end(); ++it){
      std::cout << "\n  " << it->first << ": " << it->second;
    }
  }
  if( !mom_lists.count( node["P_src"].as<std::string>() ) ){
    char msg[200];
    snprintf(msg, 200,
             "The momentum list '%s' does not seem to exist!\n",
             node["P_src"].as<std::string>().c_str() );
    throw( std::invalid_argument(msg) );
  }
  const std::string momlist_key = node["P_src"].as<std::string>();
  for( auto & mom : mom_lists[ momlist_key ] ){
    if( node["g_src"].Type() == YAML::NodeType::Scalar ){
      if(verbose) std::cout << std::endl;
      
      int g_src;
      if( node["g_src"].as<std::string>().find("spin_dilution") != std::string::npos ){
        g_src = -1;
      } else {
        g_src = node["g_src"].as<int>();
      }

      cvc::stoch_prop_meta_t stoch_prop(mom.data(), g_src, node["id"].as<std::string>());
      props_meta[stoch_prop.key()] = stoch_prop;
      if(verbose){
        std::cout << "Added stoch_prop_meta_t: " << stoch_prop.key();
      }
    } else {
      for(size_t i = 0; i < node["g_src"].size(); ++i){
        cvc::stoch_prop_meta_t stoch_prop(mom.data(), node["g_src"][i].as<int>(), 
                                          node["id"].as<std::string>());
        props_meta[stoch_prop.key()] = stoch_prop;
        if(verbose){
          std::cout << "Added stoch_prop_meta_t: " << stoch_prop.key() << std::endl;
        }
      }
    }
  }
  if(verbose) std::cout << std::endl;
}

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
