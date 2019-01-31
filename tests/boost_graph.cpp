#include <iostream>
#include <utility>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

struct VertexName {
  VertexName() {}
  VertexName(const std::string& _name) : name(_name) {};

  std::string name;
};

struct EdgeName {
  EdgeName() {}
  EdgeName(const std::string& _name) : name(_name) {};

  std::string name;
};

// we want to create a graph with unique, named vertices, so we specialize the
// internal vertex name
namespace boost { 
  namespace graph {

    template<>
    struct internal_vertex_name<VertexName>
    {
      typedef multi_index::member<VertexName, std::string, &VertexName::name> type;
    };

    template<>
    struct internal_vertex_constructor<VertexName>
    {
      typedef vertex_from_name<VertexName> type;
    };

} }

using namespace boost;

typedef struct mom_t {
    int x;
    int y;
    int z;
} mom_t;

int main(int, char*[])
{
  typedef adjacency_list<vecS, 
                         vecS, 
                         bidirectionalS,
                         VertexName
                        > lab_graph;

  typedef graph_traits<lab_graph>::vertex_descriptor dvertex_t;

  lab_graph g;

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
               "sdu+-g-u/pfx%dpfy%dpfz%d/gf%d/"
               "gc%d/"
               "gi%d/pix%dpiy%dpiz%d",
               out_mom.x, out_mom.y, out_mom.z, 5,
               0,
               5, in_mom.x, in_mom.y, in_mom.z);

      char seqpropname[200];
      snprintf(seqpropname,
               200,
               "sdu/gf%d/pfx%dpfy%dpfz%d/gi%d/pix%dpiy%dpiz%d",
               5, in_mom.x, in_mom.y, in_mom.z,
               5, out_mom.x, out_mom.y, out_mom.z);

      dvertex_t corrvertex = add_vertex(corrname, g);
      dvertex_t seqpropvertex = add_vertex(seqpropname, g);
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
                           "Du/d1_%s%s",
                           dim1.c_str(),
                           dir1.c_str());
                  snprintf(DDpropname,
                           200,
                           "DDu/d2_%s%s/d1_%s%s",
                           dim2.c_str(),
                           dir2.c_str(),
                           dim1.c_str(),
                           dir1.c_str());
        
                  dvertex_t Dpropvertex = add_vertex(Dpropname, g);
                  dvertex_t DDpropvertex = add_vertex(DDpropname, g);
                  
                  if( edge(DDpropvertex, Dpropvertex, g).second == false ){
                    add_edge(DDpropvertex, Dpropvertex, g);
                  }

                  char Dcorrname[200];
                  snprintf(Dcorrname,
                           200,
                           "sdu+-g-Du/pfx%dpfy%dpfz%d/gf%d/"
                           "gc%d/d1_%s%s/"
                           "gi%d/pix%dpiy%dpiz%d",
                           out_mom.x, out_mom.y, out_mom.z, 5,
                           gc, dim1.c_str(), dir1.c_str(),
                           5, in_mom.x, in_mom.y, in_mom.z);

                  dvertex_t Dcorrvertex = add_vertex(Dcorrname,g);
                  if( edge(Dcorrvertex, Dpropvertex, g).second == false ){
                    add_edge(Dcorrvertex,Dpropvertex,g);
                  }
                  if( edge(Dcorrvertex, seqpropvertex, g).second == false ){
                    add_edge(Dcorrvertex, seqpropvertex, g);
                  }

                  char DDcorrname[200];
                  snprintf(DDcorrname,
                           200,
                           "sdu+-g-DDu/pfx%spfy%dpfz%d/gf%d/"
                           "gc%d/"
                           "d2_%s%s/d1_%s%s/"
                           "gi%d/pix%dpiy%dpiz%d",
                           out_mom.x, out_mom.y, out_mom.z, 5,
                           gc, 
                           dim2.c_str(), dir2.c_str(),
                           dim1.c_str(), dir1.c_str(),
                           5, in_mom.x, in_mom.y, in_mom.z);
                  dvertex_t DDcorrvertex = add_vertex(DDcorrname, g);
                  if( edge(DDcorrvertex, DDpropveretx, g).second == false ){
                    add_edge(DDcorrvertex, DDpropvertex, g);
                  }
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

  property_map<lab_graph, std::string VertexName::*>::type name_map = get(&VertexName::name, g);

  typedef graph_traits<lab_graph>::vertex_iterator vertex_iter;
  std::pair<vertex_iter, vertex_iter> vp;
  for(vp = vertices(g); vp.first != vp.second; ++vp.first){
    std::cout << name_map[*vp.first] << std::endl;
  }

  graph_traits<lab_graph>::edge_iterator ei, ei_end;
  for(boost::tie(ei, ei_end) = edges(g); ei != ei_end; ++ei){
    std::cout << "( " << name_map[source(*ei, g)] << " -> " <<
      name_map[target(*ei,g)] << " )" << std::endl;
  }
  

  return 0;
}
