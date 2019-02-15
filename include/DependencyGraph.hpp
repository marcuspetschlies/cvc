#pragma once

#include "DependencyFulfilling.hpp"

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/make_shared.hpp>

namespace cvc {

struct VertexProperties {
  VertexProperties() : fulfilled(false), independent(false), level(1) {}
  VertexProperties(const std::string& _name) : name(_name), fulfilled(false), independent(false), level(1) {}; 

  std::string name;
  int component;
  bool fulfilled;
  bool independent;
  unsigned int level;
  std::shared_ptr<FulfillDependency> fulfill;
};

}

// we want to create a graph with unique, named vertices, so we specialize the
// internal vertex name
namespace boost { 
  namespace graph {

    template<>
    struct internal_vertex_name<cvc::VertexProperties>
    {
      typedef multi_index::member<cvc::VertexProperties, std::string, &cvc::VertexProperties::name> type;
    };

    template<>
    struct internal_vertex_constructor<cvc::VertexProperties>
    {
      typedef vertex_from_name<cvc::VertexProperties> type;
    };

} }

namespace cvc {

typedef boost::adjacency_list<boost::vecS,                                                                                                        
                              boost::vecS, 
                              boost::undirectedS,
                              VertexProperties
                              > DepGraph;

typedef typename boost::graph_traits<DepGraph>::vertex_descriptor Vertex;
typedef typename boost::graph_traits<DepGraph>::edge_descriptor Edge;
typedef boost::graph_traits<DepGraph> DepGraphTraits;

typedef boost::shared_ptr<std::vector<unsigned long>> vertex_component_map;

/**
 * @brief is edge in component 'which'?                                                                                                           
 */
struct EdgeInComponent {
  vertex_component_map mapping;
  unsigned long which;
  DepGraph const& master;

  EdgeInComponent() = delete;
  EdgeInComponent(vertex_component_map _mapping, const unsigned long _which, DepGraph const& _master) :
    mapping(_mapping), which(_which), master(_master) {}

  template <typename Edge>
  bool operator()(const Edge& e) const {
    return mapping->at(source(e,master)) == which ||
           mapping->at(target(e,master)) == which;
  }
};

struct VertexInComponent {
  vertex_component_map mapping;
  unsigned long which;
  DepGraph const& master;

  VertexInComponent() = delete;
  VertexInComponent(vertex_component_map _mapping, const unsigned long _which, DepGraph const& _master) :
    mapping(_mapping), which(_which), master(_master) {}

  template <typename Vertex>
  bool operator()(const Vertex& v) const {
    return mapping->at(v) == which;
  }
};

typedef boost::filtered_graph<DepGraph, EdgeInComponent, VertexInComponent> ComponentGraph;

static inline std::vector<ComponentGraph> connected_components_subgraphs(DepGraph const &g)
{
  vertex_component_map mapping = boost::make_shared<std::vector<unsigned long>>(num_vertices(g));
  size_t num = connected_components(g, mapping->data());

  std::vector<ComponentGraph> component_graphs;

  for( size_t i = 0; i < num; ++i){
    component_graphs.push_back(ComponentGraph(g, EdgeInComponent(mapping, i, g),
                                              VertexInComponent(mapping, i, g)));
  }

  return component_graphs;
}

template <typename Graph>
static inline
void
add_unique_edge(typename boost::graph_traits<Graph>::vertex_descriptor from,
                typename boost::graph_traits<Graph>::vertex_descriptor to,
                Graph & g)
{
  if( edge(from, to, g).second == false ){
    add_edge(from, to, g);
    if( g[from].level <= g[to].level ){
      g[from].level = g[to].level + 1;
    }
  }
}

}
