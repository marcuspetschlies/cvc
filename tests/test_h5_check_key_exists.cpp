#define MAIN_PROGRAM
#include "global.h"
#undef MAIN_PROGRAM

#include <hdf5.h>
#include <h5utils.h>

#include <cstdio>
#include <string>
#include <iostream>

using namespace cvc;

int main(int argc, char ** argv)
{
  hid_t file_id = H5Fopen("test.h5", H5F_ACC_RDWR, H5P_DEFAULT);
  if( file_id < 0 ){
    printf("Error opening 'test.h5' Does the file exist?\n");
    return 1;
  }

  // we will search for two keys in the file, one which exists and one which does not 
  std::string search_keys[2] = {"/d+-dd-sud/t6/s0/dt4/gf4/d0/fwd/d0/fwd/gi0/pfx0pfy0pfz0/px-5py-7pz-4",
                                "/d+-gd-sud/t6/s0/dt4/gf4/gc0/bleh/fwd/gi0/pfx0pfy0pfz0/px-5py-7pz-4"};
  std::string fail_path;
  bool key_exists;

	for( const auto & key : search_keys ){
    key_exists = h5_check_key_exists(file_id,
                                     key.c_str(),
                                     fail_path);
    std::cout << "Search key: " << key << std::endl
      << "Found: " << key_exists << std::endl;
    if( !key_exists ){
      std::cout << "Failed at: " << fail_path << std::endl;
    }
  }

  hid_t status = H5Fclose(file_id);
  if( status < 0 )
  {
    printf("Error closing 'test.h5'\n");
    return 2;
  }

  return 0;
}

