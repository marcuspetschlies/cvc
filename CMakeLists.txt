cmake_minimum_required (VERSION 3.8)
project (cvc LANGUAGES C CXX Fortran)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(FortranCInterface)
FortranCInterface_HEADER(fortran_name_mangling.h MACRO_NAMESPACE "F_")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# we will collect any libraries that we want to link against to compile
# CVC main programs in this list which we grow below with list(APPEND cvc_libraries [...])
set(cvc_libraries)

set( MPI_HOME CACHE PATH "path to MPI library installation directory" )
if( MPI_HOME )
  add_definitions(-DHAVE_MPI)
  add_definitions(-DOMPI_SKIP_MPICXX)
  include_directories(${MPI_HOME}/include)
  link_directories(${MPI_HOME}/lib)
  list(APPEND cvc_libraries "-lmpi")
else()
  find_package(MPI)
  if( MPI_C_FOUND )
    add_definitions(-DHAVE_MPI)
    # we never want to have the C++ MPI bindings!
    add_definitions(-DOMPI_SKIP_MPICXX)
    include_directories(${MPI_C_INCLUDE_PATH})
    list(APPEND cvc_libraries "${MPI_C_LIBRARIES}")
    message(STATUS "MPI found")
    message(STATUS "  MPI_C_INCLUDE_PATH=${MPI_C_INCLUDE_PATH}")
    message(STATUS "  MPI_C_LIBRARIES=${MPI_C_LIBRARIES}")
  else()
    message( FATAL_ERROR "MPI could not be found automatically and MPI_HOME has not been specified!" )
  endif()
endif()

if( MPI_C_FOUND OR MPI_HOME )
  set(PARALLEL_LEVEL "TXYZ" CACHE STRING "parallelisation level, possible values: T, TX, TXY, TXYZ")
  if( ("D_${PARALLEL_LEVEL}" STREQUAL "D_T") OR 
      ("D_${PARALLEL_LEVEL}" STREQUAL "D_TX") OR 
      ("D_${PARALLEL_LEVEL}" STREQUAL "D_TXY") OR 
      ("D_${PARALLEL_LEVEL}" STREQUAL "D_TXYZ") )
    add_definitions(-DPARALLEL${PARALLEL_LEVEL})
    message( STATUS "Compiling with parallelisation level ${PARALLEL_LEVEL}" )
  else()
    message( FATAL_ERROR "Parallelisation level '${PARALLEL_LEVEL}' is invalid, allowed values T, TX, TXY, TXYZ" )
  endif()
endif()

set( TMLQCD_SRC CACHE PATH "path to tmLQCD source directory" )
if( TMLQCD_SRC )
  include_directories( ${TMLQCD_SRC}/include )
endif()

set( TMLQCD_BUILD CACHE PATH "path to tmLQCD build directory (must be build directory for source in TMLQCD_SRC!)" )
if( TMLQCD_BUILD )
  link_directories(${TMLQCD_BUILD}/lib)
  include_directories(${TMLQCD_BUILD})
  list(APPEND cvc_libraries 
    "-lwrapper" "-lhmc" "-lmonomial"
    "-loperator" "-lmeas" "-linit" 
    "-lhmc" "-lxchange" "-lrational"
    "-lsolver" "-lio" "-llinalg" "-lm")
  add_definitions(-DHAVE_TMLQCD_LIBWRAPPER)
  message( STATUS "tmLQCD wrapper library enabled" ) 
else()
  message( STATUS "tmLQCD wrapper library disabled" )
endif()

set( LIME_HOME CACHE PATH "path to the c-lime library installation" )
if( LIME_HOME )
  link_directories(${LIME_HOME}/lib)
  include_directories(${LIME_HOME}/include)
  list(APPEND cvc_libraries "-llime")
else()
  message(FATAL_ERROR "LIME_HOME must be defined!" )
endif()

set( LEMON_HOME CACHE PATH "path to the LEMON library installation" )
if( LEMON_HOME )
  add_definitions(-DHAVE_LIBLEMON)
  link_directories(${LEMON_HOME}/lib)
  include_directories(${LEMON_HOME}/include)
  list(APPEND cvc_libraries "-llemon")
  # some of the conversions done in LEMON are rather dubious at best
  # let's use -fpermissive to allow them to compile anyway
  #string(APPEND CMAKE_CXX_FLAGS " -fpermissive")
  message( STATUS "LEMON support enabled" )
else()
  message( STATUS "LEMON support disabled" )
endif()


set( DD_ALPHA_AMG_HOME CACHE PATH "path to the DDalphaAMG library installation (optional, depends on how tmLQCD was configured)" )
if( DD_ALPHA_AMG_HOME )
  link_directories(${DD_ALPHA_AMG_HOME}/lib)
  include_directories(${DD_ALPHA_AMG_HOME}/include)
  list(APPEND cvc_libraries "-lDDalphaAMG")
  message( STATUS "DDalphaAMG support enabled" )
endif()

# we prefer static libraries
set(HDF5_USE_STATIC_LIBRARIES)
find_package(HDF5)
if(HDF5_FOUND)
  add_definitions(-DHAVE_HDF5)
  include_directories(${HDF5_INCLUDE_DIRS})
  list(APPEND cvc_libraries ${HDF5_LIBRARIES})
  message(STATUS "HDF5 found")
  message(STATUS "  HDF5_INCLUDE_DIRS=${HDF5_INCLUDE_DIRS}")
  message(STATUS "  HDF5_LIBRARIES=${HDF5_LIBRARIES}")
else()
  message(STATUS "HDF5 could not be found!")
endif()

# attempt to find lhpc-aff-config
set( LHPC_AFF_HOME CACHE PATH "path to LHPC AFF library installation (optional, required only if 'lhpc-aff-config' is not in \$PATH" )
set( USE_AFF CACHE BOOL "whether AFF support should be compiled even though HDF5 is available")
if( (NOT HDF5_FOUND) OR USE_AFF )
  list(APPEND CMAKE_PREFIX_PATH "${LHPC_AFF_HOME}")
  find_program(LHPC_AFF_CONFIG
    lhpc-aff-config)
  if( "X_${LHPC_AFF_CONFIG}" STREQUAL "X_LHPC_AFF_CONFIG-NOTFOUND" )
    message( FATAL_ERROR "lhpc-aff-config could not be found. Did you try specifying -DLHPC_AFF_HOME ?" )
  else()
    add_definitions(-DHAVE_LHPC_AFF)
    # parse the AFF version
    execute_process(
      COMMAND ${LHPC_AFF_CONFIG} --version 
      COMMAND sed s/[^0-9.]//g 
      COMMAND sed s/^.//g
      OUTPUT_VARIABLE LHPC_AFF_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if( "${LHPC_AFF_VERSION}" VERSION_LESS "2.1.0" )
      message( FATAL_ERROR "LHPC AFF version: ${LHPC_AFF_VERSION}, but 2.1.0 required!" )
    endif()
    # the stripping of -I and -L is a bit of a hack and might break...
    execute_process(
      COMMAND ${LHPC_AFF_CONFIG} --cflags
      COMMAND sed s/-I//g
      OUTPUT_VARIABLE LHPC_AFF_INCLUDEDIRS
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${LHPC_AFF_CONFIG} --ldflags
      COMMAND sed s/-L//g
      OUTPUT_VARIABLE LHPC_AFF_LINKDIRS
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${LHPC_AFF_CONFIG} --libs
      OUTPUT_VARIABLE LHPC_AFF_LIBS
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    include_directories( ${LHPC_AFF_INCLUDEDIRS} )
    list(APPEND cvc_libraries "${LHPC_AFF_LIBS}" )
    link_directories( ${LHPC_AFF_LINKDIRS} )
    message( STATUS "AFF support enabled" )
    message( STATUS "  LHPC_AFF_LINKDIRS: ${LHPC_AFF_LINKDIRS}")
    message( STATUS "  LHPC_AFF_LIBS: ${LHPC_AFF_LIBS}")
    message( STATUS "  LHPC_AFF_INCLUDEDIRS: ${LHPC_AFF_INCLUDEDIRS}")
  endif()
endif()

find_package(FLEX REQUIRED)
if( FLEX_FOUND )
  FLEX_TARGET(read_input_parser read_input_parser_cvc.l ${CMAKE_SOURCE_DIR}/read_input_parser_cvc.cpp COMPILE_FLAGS "-i" )
  if( FLEX_read_input_parser_DEFINED )
    message( STATUS "input parser built: ${FLEX_read_input_parser_OUTPUTS}" )
  else()
    message( FATAL_ERROR "input parser failed to build!" )
  endif()
else()
  message( FATAL_ERROR "Unable to find flex!" )
endif()

# currently CVC does not seem to use FFTW3 but some other version
# for now let's remove support
#set( FFTW_HOME CACHE PATH "path to the fftw3 library installation" )
#if( FFTW_HOME )
#  link_directories(${FFTW_HOME}/lib)
#  include_directories(${FFTW_HOME}/include)
#  link_libraries(-lfftw3)
#else()
#  message(FATAL_ERROR "FFTW_HOME must be defined!" )
#endif()

# maybe we might want to use yaml in CVC at some point
#find_package(yaml-cpp REQUIRED)
#message("Found yaml-cpp: ${YAML_CPP_LIBRARIES}")
#include_directories(${YAML_CPP_INCLUDE_DIR})
#link_libraries(${YAML_CPP_LIBRARIES})

set(LAPACK_HOME CACHE PATH "Path to custom LAPACK library")
if(NOT LAPACK_HOME)
  find_package(LAPACK REQUIRED)
  if( LAPACK_FOUND )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
    list(APPEND cvc_libraries "${LAPACK_LIBRARIES}" )
    message(STATUS "Using LAPACK libraries: ${LAPACK_LIBRARIES}")
  else()
    message( FATAL_ERROR "Unable to find LAPACK!" )
  endif()
else()
  link_directories("${LAPACK_HOME}/lib")
  set(LAPACK_LIBRARIES CACHE STRING "libraries for custom LAPACK (e.g. '-lsci_gnu_mp' or '-llapack')")
  if(LAPACK_LIBRARIES)
    list(APPEND cvc_libraries "${LAPACK_LIBRARIES}")
    message(STATUS "Using LAPACK libraries: ${LAPACK_LIBRARIES}")
  else()
    message(FATAL_ERROR "When using custom LAPACK libraries, -DLAPACK_HOME and -DLAPACK_LIBRARIES must be provided")
  endif()
endif()


set(BLAS_HOME CACHE PATH "Path to custom BLAS library")
if(NOT BLAS_HOME)
  find_package(BLAS REQUIRED)
  if( BLAS_FOUND )
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
    list(APPEND cvc_libraries "${BLAS_LIBRARIES}" )
    message(STATUS "Using BLAS libraries: ${BLAS_LIBRARIES}")
  else()
    message( FATAL_ERROR "Unable to find BLAS!" )
  endif()
else()
  link_directories("${BLAS_HOME}/lib")
  set(BLAS_LIBRARIES CACHE STRING "libraries for custom LAPACK (e.g. '-lsci_gnu_61_mp' or '-lblas')")
  if(BLAS_LIBRARIES)
    list(APPEND cvc_libraries "${BLAS_LIBRARIES}")
    message(STATUS "Using BLAS libraries: ${BLAS_LIBRARIES}")
  else()
    message(FATAL_ERROR "When using custom BLAS libraries, -DBLAS_HOME and -DBLAS_LIBRARIES must be provided")
  endif()
endif()

find_package(OpenMP REQUIRED)
if ( OPENMP_FOUND )
  add_definitions(-DHAVE_OPENMP)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  list(APPEND cvc_libraries "${OPENMP_LIBRARIES}" )
else()
  message( FATAL_ERROR "Unable to find OpenMP!" )
endif()

# generate git version information in the file gitversion.cpp in the build directory
# it will be linked into libcvc below
include(GetGitRevisionDescription)
get_git_head_revision(git_refspec git_sha1)
git_local_changes(git_changes)
message(STATUS "git sha1: ${git_sha1}")
message(STATUS "git refspec: ${git_refspec}")
message(STATUS "git state: ${git_changes}")
configure_file("${CMAKE_SOURCE_DIR}/gitversion.cpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/gitversion.cpp"
  @ONLY)
list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/gitversion.cpp")

find_package(Boost
             REQUIRED
             program_options)
if( NOT Boost_FOUND )
  message( FATAL_ERROR "Unable to find Boost. Did you try specifying -DBOOST_ROOT?")
else()
  include_directories(${Boost_INCLUDE_DIRS})
  list(APPEND cvc_libraries "${Boost_LIBRARIES}")
  message(STATUS "  Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
  message(STATUS "  Boost_LIBRARIES: ${Boost_LIBRARIES}")
endif()

# the CUDA libraries must be the last to be set
set( QUDA_HOME CACHE PATH "path to QUDA installation (optional, depends on whether tmLQCD uses QUDA sovers)" ) 
if( QUDA_HOME )
  link_directories(${QUDA_HOME}/lib)
  include_directories(${QUDA_HOME}/include)
  list(APPEND cvc_libraries "-lquda")

  # enable CUDA language support
  enable_language(CUDA)
  # we still use find_package because we want the directories
  find_package(CUDA REQUIRED)
  if( CUDA_TOOLKIT_ROOT_DIR )
    link_directories( "${CUDA_TOOLKIT_ROOT_DIR}/lib" "${CUDA_TOOLKIT_ROOT_DIR}/lib64" )
    include_directories( ${CUDA_INCLUDE_DIRS} )
    # QUDA requires -lcuda for cuMemAlloc in addition to -lcudart
    list(APPEND cvc_libraries "-lcuda" "-lcudart" "-lcublas")
    message(STATUS "CUDA found")
    message(STATUS "  CUDA_SDK_ROOT_DIR=${CUDA_SDK_ROOT_DIR}")
    message(STATUS "  CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "  CUDA_INCLUDE_DIRS=${CUDA_INCLUDE_DIRS}")
  else()
    message( FATAL_ERROR "Unable to find CUDA!")
  endif()
  message( STATUS "compiling against tmLQCD with QUDA" )
endif()

set( GPU_DIRECT_SOLVER FALSE CACHE BOOL "whether to use the invert_quda_direct pathway for inversions" )
if( GPU_DIRECT_SOLVER )
  if( NOT QUDA_HOME )
    message(FATAL_ERROR "The direct GPU solver is only available when tmLQCD has been compiled with QUDA support. See QUDA_HOME.")
  endif()
  add_definitions(-DGPU_DIRECT_SOLVER)
endif()

set( QMP_HOME CACHE PATH "path to the QMP library required by QPhiX (optional, depends on how tmLQCD was configured)" )
if( QMP_HOME )
  link_directories(${QMP_HOME}/lib)
  include_directories(${QMP_HOME}/include)
  list(APPEND cvc_libraries "-lqmp")
endif()

set( QPHIX_HOME CACHE PATH "path to the QPhiX library installation (optional, depends on how tmLQCD was configured)" )
if( QPHIX_HOME )
  ## this is a really ugly hack to make sure that all symbols are found, bit it works!
  ## it can only be done as the very last step because it changes the linker flags
  ## and makes many tests break
  ## the way the QPhiX interface is linked into tmLQCD should be adjusted for this
  ## to become unnecessary
  set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${TMLQCD_BUILD}/qphix_interface.o" )
  link_directories(${QPHIX_HOME}/lib)
  include_directories(${QPHIX_HOME}/include)
  list(APPEND cvc_libraries "-lqphix_codegen" "-lqphix_solver")
  message( STATUS "compiling against tmLQCD with QPhiX" )
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})

add_library(cvc
  ${CMAKE_CURRENT_BINARY_DIR}/gitversion.cpp
  ${FLEX_read_input_parser_OUTPUTS}
  init/init_g_gauge_field.cpp
  mpi_init.cpp
  cvc_geometry.cpp
  set_default.cpp
  cvc_utils.cpp
  clover.cpp
  propagator_io.cpp
  prepare_source.cpp
  project.cpp
  contract_cvc_tensor.cpp
  io.cpp
  get_index.cpp
  ranlxd.cpp
  ranlxs.cpp
  Q_phi.cpp
  Q_clover_phi.cpp
  scalar_products.cpp
  matrix_init.cpp
  dml.cpp
  io_utils.cpp
  DML_crc32.cpp
  gauge_io.cpp
  make_x_orbits.cpp
  rng/ParallelMT19937_64.cpp
)
add_executable(cpff_invert_contract
  cpff_invert_contract.cpp)
target_link_libraries(cpff_invert_contract
  cvc
  ${cvc_libraries})

add_executable(cpff_tbc_invert_contract
  cpff_tbc_invert_contract.cpp)

add_executable(cpff_invert_contract_no_spin_dilution
  cpff_invert_contract_no_spin_dilution.cpp)
target_link_libraries(cpff_invert_contract_no_spin_dilution
  cvc
  ${cvc_libraries})

add_subdirectory(external/googletest)
add_subdirectory(tests)
