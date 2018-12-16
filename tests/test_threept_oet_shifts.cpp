#define MAIN_PROGRAM
#include "global.h"
#undef MAIN_PROGRAM

#include "Core.hpp"
#include "ParallelMT19937_64.hpp"
#include "meta_types.hpp"
#include "enums.hpp"
#include "init_g_gauge_field.hpp"
#include "meta_parsing.hpp"

#include "types.h"

#include <vector>
#include <map>

using namespace cvc;

int main(int argc, char ** argv){
  Core core(argc,argv,"test_threept_oet_shifts");
  if( !(core.is_initialised()) ){
    std::cout << "Core initialisation failed!\n";
    return(CVC_EXIT_CORE_INIT_FAILURE);
  }
  int exitstatus = 0;

  ParallelMT19937_64 rng(8908908212ULL); 

#ifdef HAVE_MPI
  Stopwatch sw(g_cart_grid);
#else
  Stopwatch sw(0);
#endif
 
  CHECK_EXITSTATUS_NONZERO(
      exitstatus,
      init_g_gauge_field(),
      "[cpff_invert_contract] Error initialising gauge field!",
      true, 
      CVC_EXIT_GAUGE_INIT_FAILURE);

  const unsigned long long global_volume = g_nproc*VOLUME;

  std::vector<double> ranspinor(24*(VOLUME+RAND));
  rng.gen_z2(ranspinor.data(), 24); 

  std::vector<threept_shifts_oet_meta_t> threept_shift_correls;
  std::vector<shift_t> zero_left_shifts;
  std::vector<shift_t> two_right_fwd0_shifts;
  std::vector<shift_t> xp0p1_left_bwd_shift;
  std::vector<shift_t> xp0p1_right_bwd_shift; 

  // define a shift acting in the forward direction at the origin
  two_right_fwd0_shifts.push_back( { {0,0,0,0},
                                     DIM_T,
                                     DIR_FWD} );
  // and add a second shift on top of that
  two_right_fwd0_shifts.push_back( { {0,0,0,0},
                                     DIM_T,
                                     DIR_FWD} );

  // 3pt function with \bar{psi}(x) U_0(x) U_0(x+0) \psi(x+0+0) insertion
  threept_shift_correls.push_back( 
      threept_shifts_oet_meta_t(
        "u", 
        "u", 
        "d", 
        "fwd", 
        5, 
        5, 
        4, 
        5,
        zero_left_shifts,
        two_right_fwd0_shifts, 
        {1.0, 0.0} )
      );

  // define a shift for this type of construction
  //                               +-----------------------+
  // +-------------------+         |           x'=(x+1+0)  |
  // |          |        |         |           |           |
  // |          |        |         |           |           |
  // |          |        |     =   |           |           |
  // |          v        |         |           v           |
  // |x         psi(x+1) |         |           psi(x'-0)   |
  // +-------------------+         +-----------------------+
  xp0p1_right_bwd_shift.push_back( { {+1,+1,0,0}, /* indicates that the origin should be shifted 
                                                   by one unit in T and one unit in X */
                                      DIM_T,             
                                      DIR_BWD} );       
  
  // define a shift for this type of construction
  //
  // +------------+      +----------------------+
  // |psibar(x+0) |      |  psibar(x'-1)        |
  // |------>     |      |  --------> x'=(x+1+0)|
  // |            |      |                      |
  // |            |   =  |                      |
  // |x           |      |                      |
  // +------------+      +----------------------+
  xp0p1_left_bwd_shift.push_back( { {+1,+1,0,0}, /* indicates that the origin should be shifted 
                                                  by one unit in T and one unit in X */
                                    DIM_X,             
                                    DIR_BWD} );

  // these two can be contracted to produce
  // +-----------------+
  // |psibar(x+0)      |
  // |-------->|       |
  // |         |       |
  // |         |       |
  // |         v       |
  // |x       psi(x+1) |
  // +-----------------+

  // 3pt function with \bar{psi}(x+0) U_1(x+0) U_0^dag(x+1+) \psi(x+1) insertion
  threept_shift_correls.push_back( 
      threept_shifts_oet_meta_t(
        "u", 
        "u", 
        "d", 
        "fwd", 
        5, 
        5, 
        4, 
        5,
        xp0p1_left_bwd_shift,
        xp0p1_right_bwd_shift, 
        {1.0, 0.0} )
      );

  // TODO: 
  // 1) extract required propagators
  // 2) extract level 1 shifts
  // 3) extract level 2 shifts
  // [...]

  std::map<std::string, stoch_prop_meta_t> props_meta;
  std::map<std::string, seq_stoch_prop_meta_t> seq_props_meta;
  std::map<std::string, shifted_prop_meta_t> shifted_props_meta;
 
  int p[3] = {0,0,0};

  get_fwd_bwd_props_meta_from_npt_oet_meta(threept_shift_correls, props_meta);
  get_seq_props_meta_from_npt_oet_meta(threept_shift_correls, p, 24, seq_props_meta);

  for( auto const & prop : props_meta ){
    std::cout << prop.second.key() << std::endl;
  }
  for( auto const & prop : seq_props_meta ){
    std::cout << prop.second.key() << std::endl;
  }

  return 0;
} 
