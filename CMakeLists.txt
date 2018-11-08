cmake_minimum_required (VERSION 2.8)
project (cvc)

set( MPI_HOME CACHE PATH "path to MPI library installation directory" )
if( MPI_HOME )
  include_directories(${MPI_HOME}/include)
  link_directories(${MPI_HOME}/lib)
  link_libraries(-lmpi)
else()
  find_package(MPI)
  if( MPI_C_FOUND )
    include_directories(${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
    link_libraries(${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
  else()
    message( FATAL_ERROR "MPI could not be found automatically and MPI_HOME has not been specified!" )
  endif()
endif()

set( TMLQCD_SRC CACHE PATH "path to tmLQCD source directory" )
if( TMLQCD_SRC )
  include_directories( ${TMLQCD_SRC}/include )
  include_directories( ${TMLQCD_SRC} )
else()
  message(FATAL_ERROR "TMLQCD_SRC must be defined!" )
endif()

set( TMLQCD_BUILD CACHE PATH "path to tmLQCD build directory (must be build directory for source in TMLQCD_SRC!)" )
if( TMLQCD_BUILD )
  link_directories(${TMLQCD_BUILD}/lib)
  include_directories(${TMLQCD_BUILD})
  link_libraries( -lwrapper -lhmc -lmonomial -loperator -lmeas -linit -lhmc -lxchange -lrational -lsolver -lio -llinalg -lm )
else()
  message(FATAL_ERROR "TMLQCD_BUILD must be defined!" )
endif()

set( LIME_HOME CACHE PATH "path to the c-lime library installation" )
if( LIME_HOME )
  link_directories(${LIME_HOME}/lib)
  include_directories(${LIME_HOME}/include)
  link_libraries(-llime)
else()
  message(FATAL_ERROR "LIME_HOME must be defined!" )
endif()

set( LEMON_HOME CACHE PATH "path to the LEMON library installation (optional, depends on how tmLQCD was configured)" )
if( LEMON_HOME )
  link_directories(${LEMON_HOME}/lib)
  include_directories(${LEMON_HOME}/include)
  link_libraries(-llemon)
endif()

set( QMP_HOME CACHE PATH "path to the QMP library required by QPhiX (optional, depends on how tmLQCD was configured)" )
if( QMP_HOME )
  link_directories(${QMP_HOME}/lib)
  include_directories(${QMP_HOME}/include)
  link_libraries(-lqmp)
endif()

set( QPHIX_HOME CACHE PATH "path to the QPhiX library installation (optional, depends on how tmLQCD was configured)" )
if( QPHIX_HOME )
  ## this is a really ugly hack to make sure that all symbols are found, bit it works!
  set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${TMLQCD_BUILD}/qphix_interface.o" )
  
  link_directories(${QPHIX_HOME}/lib)
  include_directories(${QPHIX_HOME}/include)
  link_libraries(-lqphix_codegen -lqphix_solver)
endif()

set( DD_ALPHA_AMG_HOME CACHE PATH "path to the DDalphaAMG library installation (optional, depends on how tmLQCD was configured)" )
if( DD_ALPHA_AMG_HOME )
  link_directories(${DD_ALPHA_AMG_HOME}/lib)
  include_directories(${DD_ALPHA_AMG_HOME}/include)
  link_libraries(-lDDalphaAMG)
endif()

set( QUDA_HOME CACHE PATH "path to QUDA installation (optional, depends on whether tmLQCD uses QUDA sovers)" ) 
if( QUDA_HOME )
  link_directories(${QUDA_HOME}/lib)
  include_directories(${QUDA_HOME}/include)
  link_libraries(-lquda)

  find_package(CUDA)
  if( CUDA_FOUND )
    link_directories( ${CUDA_LIBRARY_DIRS} )
    include_directories( ${CUDA_INCLUDE_DIRS} )
    # QUDA required -lcuda for cuMemAlloc in addition to -lcudart
    link_libraries( ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} -lcuda )
  else()
    message( FATAL_ERROR "Unable to find CUDA!")
  endif()

endif()


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# maybe we might want to use yaml in CVC at some point
#find_package(yaml-cpp REQUIRED)
#message("Found yaml-cpp: ${YAML_CPP_LIBRARIES}")
#include_directories(${YAML_CPP_INCLUDE_DIR})
#link_libraries(${YAML_CPP_LIBRARIES})

find_package(LAPACK REQUIRED)
if( LAPACK_FOUND )
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
  link_libraries( ${LAPACK_LIBRARIES} )
else()
  message( FATAL_ERROR "Unable to find LAPACK!" )
endif()

find_package(BLAS REQUIRED)
if( BLAS_FOUND )
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
  link_libraries( ${BLAS_LIBRARIES} )
else()
  message( FATAL_ERROR "Unable to find BLAS!" )
endif()

find_package(OpenMP REQUIRED)
if ( OPENMP_FOUND )
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  link_libraries( ${OPENMP_LIBRARIES} )
else()
  message( FATAL_ERROR "Unable to find OpenMP!" )
endif()

# maybe we might need Boost at some point
#find_package(Boost 
#             REQUIRED
#             program_options
#             filesystem)
#if( Boost_FOUND )
#  link_directories( ${Boost_LIBRARY_DIRS} )
#  include_directories( ${Boost_INCLUDE_DIRS} )
#  link_libraries( ${Boost_LIBRARIES} )
#else()
#  message( FATAL_ERROR "Unable to find Boost (or one of the requested components, see CMakeLists.txt!")
#endif()


# base package includes and libraries
include_directories(include)

add_executable(cpff_invert_contract
  cpff_invert_contract.cpp)
