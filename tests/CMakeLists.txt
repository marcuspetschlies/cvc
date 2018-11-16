add_executable(test_MT19937_64 test_MT19937_64.cpp)
add_executable(test_ParallelMT19937_64 test_ParallelMT19937_64.cpp)
add_executable(test_SequenceOfUnique test_SequenceOfUnique.cpp)


target_include_directories(test_SequenceOfUnique PUBLIC ${Boost_INCLUDE_DIRS} )
target_link_libraries(test_SequenceOfUnique ${Boost_LIBRARIES} cvc)

target_link_libraries(test_ParallelMT19937_64 ParallelMT19937_64 cvc ${Boost_LIBRARIES})
target_include_directories(test_ParallelMT19937_64 PUBLIC ${Boost_INCLUDE_DIRS} )

if( NOT Boost_FOUND )
  message( FATAL_ERROR "Unable to find Boost (or one of the requested components, see CMakeLists.txt!")
endif()

