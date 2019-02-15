#pragma once

#include <vector>
#include <map>
#include <yaml-cpp/yaml.h>

std::vector< std::vector<int> > parse_momentum_list(const YAML::Node & node);

std::vector< std::vector<int> > psq_to_momentum_list(const YAML::Node & node);

void construct_momentum_list(const YAML::Node & node, const bool verbose, std::map< std::string, std::vector< std::vector<int> > > & mom_lists );
