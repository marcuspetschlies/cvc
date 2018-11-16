# Gtest
set(GTEST_ROOT CACHE PATH "Path to Google test framework")
find_package(GTest)

if(${GTEST_FOUND})
  add_executable(main-gtest
    main.cpp
    test_MT19937_64.cpp
    )

  target_link_libraries(main-gtest GTest::GTest)

  enable_testing()
  add_test(NAME gtest
    COMMAND ./main-gtest)
else()
  message(STATUS "gtest was not found, not building gtest tests. Did you try specifying -DGTEST_ROOT?")
endif()
