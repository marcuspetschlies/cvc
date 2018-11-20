# Gtest
set(gtest_sources
  main.cpp
  test_MT19937_64.cpp
  )
add_executable(main-gtest
  ${gtest_sources})

target_include_directories(main-gtest PUBLIC ${gtest_SOURCE_DIR}/include)
target_link_libraries(main-gtest gtest_main)

enable_testing()
add_test(NAME gtest
  COMMAND ./main-gtest)
