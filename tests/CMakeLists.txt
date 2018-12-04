add_executable(test_Core test_Core.cpp)
target_link_libraries(test_Core cvc ${cvc_libraries})

add_executable(test_ParallelMT19937_64 test_ParallelMT19937_64.cpp)
target_link_libraries(test_ParallelMT19937_64 cvc ${cvc_libraries})

add_executable(test_SequenceOfUnique test_SequenceOfUnique.cpp)
target_link_libraries(test_SequenceOfUnique cvc ${cvc_libraries})

add_executable(benchmark_index_tools benchmark_index_tools.cpp)
target_link_libraries(benchmark_index_tools cvc ${cvc_libraries})

add_executable(test_index_tools test_index_tools.cpp)
target_link_libraries(test_index_tools cvc ${cvc_libraries})

add_executable(test_h5_check_key_exists test_h5_check_key_exists.cpp)
target_link_libraries(test_h5_check_key_exists ${cvc_libraries})

add_executable(test_CHECK_EXITSTATUS test_CHECK_EXITSTATUS.cpp)
target_link_libraries(test_CHECK_EXITSTATUS ${cvc_libraries})

add_executable(test_prepare_gamma_timeslice_oet test_prepare_gamma_timeslice_oet.cpp)
target_link_libraries(test_prepare_gamma_timeslice_oet cvc ${cvc_libraries})

add_executable(test_meta_types test_meta_types.cpp)
target_link_libraries(test_meta_types ${cvc_libraries})

# copy over test file for test of h5utils
configure_file(${CMAKE_SOURCE_DIR}/tests/test.h5
  ${CMAKE_BINARY_DIR}/tests/test.h5
  COPYONLY)

# copy over analysis R script for test of Parallel MT19937_64 generator
configure_file(${CMAKE_SOURCE_DIR}/tests/analysis_test_ParallelMT19937_64.R
  ${CMAKE_BINARY_DIR}/tests/analysis_test_ParallelMT19937_64.R
  COPYONLY)

add_subdirectory(gtest)
