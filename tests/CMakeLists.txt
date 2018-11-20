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

add_subdirectory(gtest)
