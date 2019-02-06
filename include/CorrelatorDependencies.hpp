#pragma once

#include "enums.hpp"
#include "types.h"

#include <string>
#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#include <cstdio>

namespace cvc {

struct FulfillDependency {
    virtual void operator()() const = 0;
};

struct SeqSourceFulfill : public FulfillDependency {
  int src_ts;
  mom_t pf;
  std::string src_prop_key;

  SeqSourceFulfill(const int _src_ts, const mom_t _pf, const std::string& _src_prop_key) :
    src_ts(_src_ts), pf(_pf), src_prop_key(_src_prop_key) {}

  void operator()() const
  {
    char msg[500];
    snprintf(msg, 500, "SeqSourceFulfill: Creating source on ts %d of %s\n", src_ts, src_prop_key.c_str());
    std::cout << msg;
  }
};

struct PropFulfill : public FulfillDependency {
  std::string src_key;
  std::string flav;

  PropFulfill(const std::string& _src_key, const std::string& _flav) : 
    src_key(_src_key), flav(_flav) {}

  void operator()() const
  {
    char msg[500];
    snprintf(msg, 500, "PropFulfill: Inverting %s on %s\n", flav.c_str(), src_key.c_str());
    std::cout << msg;
  }
};

struct CovDevFulfill : public FulfillDependency {
  std::string spinor_key;
  int dir;
  int dim;

  CovDevFulfill(const std::string& _spinor_key, const int _dir, const int _dim) :
    spinor_key(_spinor_key), dir(_dir), dim(_dim) {}

  void operator()() const
  {
    char msg[500];
    snprintf(msg, 500, "CovDevFulfill: Applying CovDev in dir %c, dim %c on %s\n", 
        shift_dir_names[dir],
        latDim_names[dim], 
        spinor_key.c_str());
    std::cout << msg;
  }

};

struct CorrFulfill : public FulfillDependency {
  std::string propkey;
  std::string dagpropkey;
  mom_t p;
  int gamma;

  CorrFulfill(const std::string& _propkey, const std::string& _dagpropkey, const mom_t& _p, const int _gamma) :
    propkey(_propkey), dagpropkey(_dagpropkey), p(_p), gamma(_gamma) {}

  void operator()() const
  {
    char msg[500];
    snprintf(msg, 500, "CorrFullfill: Contracting %s+-g%d/px%dpy%dpz%d-%s\n",
        dagpropkey.c_str(), gamma, p.x, p.y, p.z, propkey.c_str());
    std::cout << msg;
  }
};

/**
 * @brief recursively fulfill dependencies starting at a particular vertex
 *
 * @tparam Graph
 * @param v
 * @param g
 */
template <typename Graph>
static inline void descend_and_fulfill(typename boost::graph_traits<Graph>::vertex_descriptor v,
                                       Graph & g)
{
  std::cout << "Entered " << g[v].name << std::endl;
  
  // if we hit a vertex which can be fulfilled immediately, let's do so
  // this will break one class of infinite recursions
  if( g[v].independent && !g[v].fulfilled ){
    std::cout << "Calling fulfill of " << g[v].name << std::endl;
    fflush(stdout);
    (*(g[v].fulfill))();
    g[v].fulfilled = true;
  }
 
  // otherwise we descend further, but never up the level hierarchy
  typename boost::graph_traits<Graph>::out_edge_iterator e, e_end;
  for( boost::tie(e, e_end) = boost::out_edges(v, g); e != e_end; ++e)
    if( g[boost::target(*e, g)].fulfilled == false && 
        g[boost::target(*e, g)].level < g[v].level ){
      std::cout << "Descending into " << g[boost::target(*e, g)].name << std::endl;
      fflush(stdout);
      descend_and_fulfill( boost::target(*e, g), g );
    }

  std::cout << "Came up the hierarchy, ready to fulfill!" << std::endl;
  fflush(stdout);

  // in any case, when we come back here, we are ready to fulfill
  //if( g[v].fulfilled == false ){
    std::cout << "Calling fulfill of " << g[v].name << std::endl;
    fflush(stdout);
    (*(g[v].fulfill))();
    g[v].fulfilled = true;
  //}
}

} // namespace(cvc)
