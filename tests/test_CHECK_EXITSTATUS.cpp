#define MAIN_PROGRAM
#include "global.h"
#undef MAIN_PROGRAM

#include <iostream>

int test_check_exitstatus(bool fail){
  if(fail)
    return -1;
  else
    return 0;
}

int main(int argc, char ** argv){
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  int exitstatus = 0;

  std::cout << "Testing CHECK_EXITSTATUS_NONZERO macro with a function which succeeds" << std::endl;

  CHECK_EXITSTATUS_NONZERO(exitstatus, test_check_exitstatus(false),
                           "[test_CHECK_EXITSTATUS]: failure of test_check_exit!",
                           false, 127);

  std::cout << std::endl;
  std::cout << "Testing CHECK_EXITSTATUS_NONZERO macro with a function which fails without exiting" << std::endl;

  CHECK_EXITSTATUS_NONZERO(exitstatus, test_check_exitstatus(true),
                           "[test_CHECK_EXITSTATUS]: failure of test_check_exit!",
                           false, 127);
  
  std::cout << std::endl;
  std::cout << "Testing CHECK_EXITSTATUS_NEGATIVE macro with a function which fails without exiting" << std::endl;

  CHECK_EXITSTATUS_NEGATIVE(exitstatus, test_check_exitstatus(true),
                           "[test_CHECK_EXITSTATUS]: failure of test_check_exit!",
                           false, 127);

  std::cout << std::endl;
  std::cout << "Testing CHECK_EXITSTATUS_NEGATIVE macro with a function which fails and exit with signal 127" << std::endl;

  CHECK_EXITSTATUS_NEGATIVE(exitstatus, test_check_exitstatus(true),
                            "[test_CHECK_EXITSTATUS]: failure of test_check_exit!",
                            true, 127);

  return 0;
}
