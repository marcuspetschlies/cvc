#define MAIN_PROGRAM
#include "global.h"
#undef MAIN_PROGRAM

#include "Core.hpp"
#include "enums.hpp"

#include <unistd.h>
#include <boost/program_options.hpp>


namespace po=boost::program_options;

void declare_test_cmd_options(po::options_description & cmd_desc)
{
  cmd_desc.add_options()
    ("test,t",
     po::value<std::string>()->default_value( std::string("default test string") ),
     "additional test option (string)");
}

int main(int argc, char ** argv){
  // test with one additional cmdline argument
  cvc::Core core(argc,argv,"test_Core", declare_test_cmd_options);

  if( !(core.is_initialised()) ){
    std::cout << "Core initialisation failed!\n";
    return(cvc::CVC_EXIT_CORE_INIT_FAILURE);
  } 
  const po::variables_map & core_cmd_options = core.get_cmd_options();
  std::cout << "Additional command line argument: " << core_cmd_options["test"].as<std::string>() << std::endl;
  
  return 0;
}
