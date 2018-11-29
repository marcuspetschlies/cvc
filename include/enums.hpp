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
  CVC_EXIT_CORE_INIT_FAILURE = 1,
  CVC_EXIT_H5_KEY_LENGTH = 2,
  CVC_EXIT_GAUGE_INIT_FAILE = 3,
  CVC_EXIT_UTIL_FUNCTION_FAILURE = 4,
  CVC_EXIT_MALLOC_FAILURE = 5
} ExitCode_t;

}

#endif
