#pragma once

#include "enums.hpp"
#include "types.h"

#include <string>
#include <iostream>

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

} // namespace(cvc)
