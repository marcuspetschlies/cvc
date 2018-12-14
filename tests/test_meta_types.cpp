#include "types.h"
#include "meta_types.hpp"

#include <vector>
#include <exception>
#include <iostream>

using namespace cvc;

int main(int argc, char ** argv){
  std::vector< stoch_prop_meta_t > stoch_props;
  std::vector< twopt_oet_meta_t > twopt_correls;

  int zero_mom[3] = {0,0,0};
  int nonzero_mom[3] = {-1,2,3};

  stoch_props.push_back( stoch_prop_meta_t(zero_mom,
                                           5,
                                           "u") );
  stoch_props.push_back( stoch_prop_meta_t(nonzero_mom,
                                           4,
                                           "d") );

  twopt_correls.push_back( twopt_oet_meta_t("u", "u", "bwd", 4, 2, 5, {1.0, 0.0} ) );
  
  // this will throw an exception and leave the twopt_oet_meta_t incompletely constructed
  try{
    twopt_correls.push_back( twopt_oet_meta_t("d", "u", "bleh", 4, 2, 5, {0.2, 22.23}) );
  }
  catch( const std::exception &e ){
    std::cout << e.what() << std::endl;
  }


  return 0;
}
