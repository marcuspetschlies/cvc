#ifndef ENUMS_HPP
#define ENUMS_HPP

namespace cvc {

typedef enum latDim_t {
  DIM_T = 0,
  DIM_X,
  DIM_Y,
  DIM_Z,
  DIM_NDIM
} latDim_t;

extern const char latDim_names[];

typedef enum shift_dir_t {
  DIR_FWD = 0,
  DIR_BWD,
  DIR_NDIR
} shift_dir_t;

extern const char shift_dir_names[];

typedef enum ExitCode_t {
  CVC_EXIT_SUCCESS = 0,
  CVC_EXIT_CORE_INIT_FAILURE = 1,
  CVC_EXIT_H5_KEY_LENGTH = 2,
  CVC_EXIT_GAUGE_INIT_FAILURE = 3,
  CVC_EXIT_UTIL_FUNCTION_FAILURE = 4,
  CVC_EXIT_MALLOC_FAILURE = 5,
  CVC_EXIT_NCODES
} ExitCode_t;

}

#endif
