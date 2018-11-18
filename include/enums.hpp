#ifndef ENUMS_HPP
#define ENUMS_HPP

namespace cvc {

typedef enum latDims_t {
  DIM_T = 0,
  DIM_X,
  DIM_Y,
  DIM_Z
} latDims_t;

typedef enum ExitCode_t {
  CVC_EXIT_SUCCESS = 0,
  CVC_EXIT_CORE_INIT_FAILURE
} ExitCode_t;

}

#endif
