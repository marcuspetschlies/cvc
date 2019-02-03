#include "CorrelatorDependencies.hpp"
#include "enums.hpp"
#include "types.h"

#include <iostream>
#include <utility>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/make_shared.hpp>

using namespace cvc;
using namespace boost;

struct VertexProperties {
  VertexProperties() : fulfilled(false) {}
  VertexProperties(const std::string& _name) : name(_name), fulfilled(false) {};

  std::string name;
  int component;
  bool fulfilled;
  std::shared_ptr<FulfillDependency> fulfill;
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

  int src_ts = 12;
  int dt = 12;

  for( auto const & in_mom : in_momenta ){
    double dummy_fwd_prop = in_mom.x;
    double dummy_bwd_prop = 0.0;

    char bwdpropname[200];
    snprintf(bwdpropname,
             200,
             "u/t23/g5/px0py0pz0");

    char fwdpropname[200];
    snprintf(fwdpropname,
             200,
             "u/t23/g%d/px%dpy%dpz%d",
             5,
             in_mom.x, in_mom.y, in_mom.z);

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

      char seqsrcname[200];
      snprintf(seqsrcname,
               200,
               "u/t%d/dt%d/gf%d/pfx%dpfy%dpfz%d",
               src_ts, dt,
               5, out_mom.x, out_mom.y, out_mom.z);

      Vertex seqsrcvertex = add_vertex(seqsrcname, g);
      g[seqsrcvertex].fulfill.reset( new SeqSourceFulfill(dt, out_mom, bwdpropname) );

      char seqpropname[200];
      snprintf(seqpropname,
               200,
               "sdu/gf%d/pfx%dpfy%dpfz%d/gi%d/pix%dpiy%dpiz%d",
               5, out_mom.x, out_mom.y, out_mom.z,
               5, in_mom.x, in_mom.y, in_mom.z);

      Vertex corrvertex = add_vertex(corrname, g);
      Vertex seqpropvertex = add_vertex(seqpropname, g);
      g[seqpropvertex].fulfill.reset( new PropFulfill("d", seqsrcname) );
      
      if( edge(seqpropvertex, seqsrcvertex, g).second == false ){
        add_edge(seqpropvertex, seqsrcvertex, g);
      }
      if( edge(corrvertex, seqpropvertex, g).second == false ){
        add_edge(corrvertex, seqpropvertex, g);
      }

      if( (in_mom.x == -out_mom.x && in_mom.y == -out_mom.y && in_mom.z == -out_mom.z) ||
          ( (in_mom.x == 0 && in_mom.x == out_mom.x) && 
            (in_mom.y == 0 && in_mom.y == out_mom.y) && 
            (in_mom.z == 0 && in_mom.z == out_mom.z) ) ){

        // for derivative operators we don't have any momentum transfer
        mom_t seqmom = {0, 0, 0};
        for( int dim1 : {DIM_T, DIM_X, DIM_Y, DIM_Z} ){
          for( int dir1 : {DIR_FWD, DIR_BWD}  ){
            for( int dim2 : {DIM_T, DIM_X, DIM_Y, DIM_Z} ){
              for( int dir2 : {DIR_FWD, DIR_BWD} ){
                for( int gc : { 0, 1, 2, 3, 4 } ){
                  char d1name[10];
                  snprintf(d1name,
                           10,
                           "d1_%c%c",
                           latDim_names[dim1],
                           shift_dir_names[dir1]);
                  char d2name[20];
                  snprintf(d2name,
                           20,
                           "d2_%c%c/%s",
                           latDim_names[dim2],
                           shift_dir_names[dir2],
                           d1name);

                  char Dpropname[200];
                  char DDpropname[200];
                  snprintf(Dpropname,
                           200,
                           "Du/%s/pix%dpiy%dpiz%d",
                           d1name,
                           in_mom.x, in_mom.y, in_mom.z);
                  snprintf(DDpropname,
                           200,
                           "DDu/%s/pix%dpiy%dpiz%d",
                           d2name,
                           in_mom.x, in_mom.y, in_mom.z);
        
                  Vertex Dpropvertex = add_vertex(Dpropname, g);
                  g[Dpropvertex].fulfill.reset( new CovDevFulfill( fwdpropname, dir1, dim1 ) ); 

                  Vertex DDpropvertex = add_vertex(DDpropname, g);
                  if( edge(DDpropvertex, Dpropvertex, g).second == false ){
                    add_edge(DDpropvertex, Dpropvertex, g);
                  }
                  g[DDpropvertex].fulfill.reset( new CovDevFulfill( Dpropname, dir2, dim2 ) );
                  

                  char Dcorrname[200];
                  snprintf(Dcorrname,
                           200,
                           "sdu+-g-Du/gf%d/pfx%dpfy%dpfz%d/"
                           "gc%d/%s/"
                           "gi%d/pix%dpiy%dpiz%d",
                           5, out_mom.x, out_mom.y, out_mom.z,
                           gc, d1name,
                           5, in_mom.x, in_mom.y, in_mom.z);
                  
                  // even if we add this a million times, the vertex will be unique
                  Vertex Dcorrvertex = add_vertex(Dcorrname,g);
                  if( edge(Dcorrvertex, Dpropvertex, g).second == false ){
                    add_edge(Dcorrvertex,Dpropvertex,g);
                  }
                  if( edge(Dcorrvertex, seqpropvertex, g).second == false ){
                    add_edge(Dcorrvertex, seqpropvertex, g);
                  }
                  g[Dcorrvertex].fulfill.reset( new CorrFulfill(Dpropname, seqpropname, seqmom, gc ) ); 

                  char DDcorrname[200];
                  snprintf(DDcorrname,
                           200,
                           "sdu+-g-DDu/gf%d/pfx%dpfy%dpfz%d/"
                           "gc%d/"
                           "%s/"
                           "gi%d/pix%dpiy%dpiz%d",
                           5, out_mom.x, out_mom.y, out_mom.z,
                           gc, 
                           d2name,
                           5, in_mom.x, in_mom.y, in_mom.z);
                  Vertex DDcorrvertex = add_vertex(DDcorrname, g);
                  if( edge(DDcorrvertex, DDpropvertex, g).second == false ){
                    add_edge(DDcorrvertex, DDpropvertex, g);
                  }
                  if( edge(DDcorrvertex, seqpropvertex, g).second == false ){
                    add_edge(DDcorrvertex, seqpropvertex, g);
                  }
                  g[DDcorrvertex].fulfill.reset( new CorrFulfill(DDpropname, seqpropname, seqmom, gc ) );
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
    
    // fulfill all dependencies, this will need to be tweaked, of course
    for( auto e : make_iterator_range(edges(component))){
      if( !g[target(e, component)].fulfilled ){
        (*g[target(e, component)].fulfill)();
        g[target(e, component)].fulfilled = true;
      }
    }
  }
  return 0;
}
