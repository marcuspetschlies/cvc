#include <iostream>
#include <utility>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/make_shared.hpp>

struct VertexProperties {
  VertexProperties() {}
  VertexProperties(const std::string& _name) : name(_name) {};

  std::string name;
  int component;
};

// we want to create a graph with unique, named vertices, so we specialize the
// internal vertex name
namespace boost { 
  namespace graph {

    template<>
    struct internal_vertex_name<VertexProperties>
    {
      typedef multi_index::member<VertexProperties, std::string, &VertexProperties::name> type;
    };

    template<>
    struct internal_vertex_constructor<VertexProperties>
    {
      typedef vertex_from_name<VertexProperties> type;
    };

} }

using namespace boost;
  
typedef adjacency_list<vecS, 
                        vecS, 
                        undirectedS,
                        VertexProperties
                       > DepGraph;

typedef typename graph_traits<DepGraph>::vertex_descriptor Vertex;
typedef typename graph_traits<DepGraph>::edge_descriptor Edge;
typedef graph_traits<DepGraph> DepGraphTraits;

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

typedef filtered_graph<DepGraph, EdgeInComponent, VertexInComponent> ComponentGraph;

std::vector<ComponentGraph> connected_components_subgraphs(DepGraph const &g)
{
  vertex_component_map mapping = make_shared<std::vector<unsigned long>>(num_vertices(g));
  size_t num = connected_components(g, mapping->data());

  std::vector<ComponentGraph> component_graphs;

  for( size_t i = 0; i < num; ++i){
    component_graphs.push_back(ComponentGraph(g, EdgeInComponent(mapping, i, g),
                                              VertexInComponent(mapping, i, g)));
  }

  return component_graphs;
}

typedef struct mom_t {
    int x;
    int y;
    int z;
} mom_t;

int main(int, char*[])
{
  DepGraph g;

  std::vector<mom_t> in_momenta;
  std::vector<mom_t> out_momenta;

  for(int mom_x = -3; mom_x <= 3; mom_x++){
    for(int mom_y = -3; mom_y <= 3; mom_y++){
      for(int mom_z = -3; mom_z <= 3; mom_z++){
        if( mom_x*mom_x + mom_y*mom_y + mom_z*mom_z < 2 ){
          in_momenta.push_back( mom_t{ mom_x, mom_y, mom_z } );
          out_momenta.push_back( mom_t{ mom_x, mom_y, mom_z } );
        }
      }
    }
  }

  for( auto const & in_mom : in_momenta ){
    for( auto const & out_mom : out_momenta ){
      char corrname[200];
      snprintf(corrname,
               200,
               "sdu+-g-u/gf%d/pfx%dpfy%dpfz%d/"
               "gc%d/"
               "gi%d/pix%dpiy%dpiz%d",
               5, out_mom.x, out_mom.y, out_mom.z,
               0,
               5, in_mom.x, in_mom.y, in_mom.z);

      char seqpropname[200];
      snprintf(seqpropname,
               200,
               "sdu/gf%d/pfx%dpfy%dpfz%d/gi%d/pix%dpiy%dpiz%d",
               5, out_mom.x, out_mom.y, out_mom.z,
               5, in_mom.x, in_mom.y, in_mom.z);

      Vertex corrvertex = add_vertex(corrname, g);
      Vertex seqpropvertex = add_vertex(seqpropname, g);
      if( edge(corrvertex, seqpropvertex, g).second == false ){
        add_edge(corrvertex, seqpropvertex, g);
      }

      if( (in_mom.x == -out_mom.x && in_mom.y == -out_mom.y && in_mom.z == -out_mom.z) ||
          ( (in_mom.x == 0 && in_mom.x == out_mom.x) && 
            (in_mom.y == 0 && in_mom.y == out_mom.y) && 
            (in_mom.z == 0 && in_mom.z == out_mom.z) ) ){
        for( std::string dim1 : {"t","x","y","z"} ){
          for( std::string dir1 : {"f", "b"} ){
            for( std::string dim2 : {"t", "x", "y", "z"} ){
              for( std::string dir2 : {"f", "b"} ){
                for( int gc : { 0, 1, 2, 3, 4 } ){
                  char Dpropname[200];
                  char DDpropname[200];
                  snprintf(Dpropname,
                           200,
                           "Du/d1_%s%s/pix%dpiy%dpiz%d",
                           dim1.c_str(),
                           dir1.c_str(),
                           in_mom.x, in_mom.y, in_mom.z);
                  snprintf(DDpropname,
                           200,
                           "DDu/d2_%s%s/d1_%s%s/pix%dpiy%dpiz%d",
                           dim2.c_str(),
                           dir2.c_str(),
                           dim1.c_str(),
                           dir1.c_str(),
                           in_mom.x, in_mom.y, in_mom.z);
        
                  Vertex Dpropvertex = add_vertex(Dpropname, g);
                  Vertex DDpropvertex = add_vertex(DDpropname, g);
                  
                  if( edge(DDpropvertex, Dpropvertex, g).second == false ){
                    add_edge(DDpropvertex, Dpropvertex, g);
                  }

                  char Dcorrname[200];
                  snprintf(Dcorrname,
                           200,
                           "sdu+-g-Du/gf%d/pfx%dpfy%dpfz%d/"
                           "gc%d/d1_%s%s/"
                           "gi%d/pix%dpiy%dpiz%d",
                           5, out_mom.x, out_mom.y, out_mom.z,
                           gc, dim1.c_str(), dir1.c_str(),
                           5, in_mom.x, in_mom.y, in_mom.z);

                  Vertex Dcorrvertex = add_vertex(Dcorrname,g);
                  if( edge(Dcorrvertex, Dpropvertex, g).second == false ){
                    add_edge(Dcorrvertex,Dpropvertex,g);
                  }
                  if( edge(Dcorrvertex, seqpropvertex, g).second == false ){
                    add_edge(Dcorrvertex, seqpropvertex, g);
                  }

                  char DDcorrname[200];
                  snprintf(DDcorrname,
                           200,
                           "sdu+-g-DDu/gf%d/pfx%dpfy%dpfz%d/"
                           "gc%d/"
                           "d2_%s%s/d1_%s%s/"
                           "gi%d/pix%dpiy%dpiz%d",
                           5, out_mom.x, out_mom.y, out_mom.z,
                           gc, 
                           dim2.c_str(), dir2.c_str(),
                           dim1.c_str(), dir1.c_str(),
                           5, in_mom.x, in_mom.y, in_mom.z);
                  Vertex DDcorrvertex = add_vertex(DDcorrname, g);
                  if( edge(DDcorrvertex, DDpropvertex, g).second == false ){
                    add_edge(DDcorrvertex, DDpropvertex, g);
                  }
                  //if( edge(DDcorrvertex, Dpropvertex, g).second == false ){
                  //  add_edge(DDcorrvertex, Dpropvertex, g);
                  //}
                  if( edge(DDcorrvertex, seqpropvertex, g).second == false ){
                    add_edge(DDcorrvertex, seqpropvertex, g);
                  }
                } // gc
              } // dir2
            } // dim2
          } // dir1
        } // dim1
      } // if(momenta)
    } // out_mom
  } // in_mom

  property_map<DepGraph, std::string VertexProperties::*>::type name_map = get(&VertexProperties::name, g);

  typedef graph_traits<DepGraph>::vertex_iterator vertex_iter;
  std::pair<vertex_iter, vertex_iter> vp;
  for(vp = vertices(g); vp.first != vp.second; ++vp.first){
    std::cout << name_map[*vp.first] << std::endl;
  }
  std::cout << std::endl;

  graph_traits<DepGraph>::edge_iterator ei, ei_end;
  for(boost::tie(ei, ei_end) = edges(g); ei != ei_end; ++ei){
    std::cout << "( " << name_map[source(*ei, g)] << " -> " <<
      name_map[target(*ei,g)] << " )" << std::endl;
  }
  std::cout << std::endl;

  std::vector<int> component(num_vertices(g));
  int num = connected_components(g, &component[0]);
  std::vector<int>::size_type i;
  std::cout << "Total number of components: " << num << std::endl;
  for(i = 0; i != component.size(); ++i){
    g[i].component = component[i];
    std::cout << "Vertex " << name_map[i] << " is in component " << component[i];
    std::cout << " also stored " << g[i].component << std::endl;
  }
  std::cout << std::endl;

  for( auto const& component : connected_components_subgraphs(g))
  {
    std::cout << "Component" << std::endl;
    for( auto e : make_iterator_range(edges(component))){
      std::cout << name_map[source(e, component)] << " -> " << name_map[target(e, component)] << std::endl;
    }
    std::cout << std::endl;
  }


  return 0;
}
