#include "MT19937_64.hpp"
#include <iostream>
#include <vector>

// 2^28
#define N_TEST 268435456ULL

#ifdef HAVE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv){
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  const unsigned long long init_key[4]={0x12345ULL, 0x23456ULL, 0x34567ULL, 0x45678ULL};
  const unsigned long long length_key = 4ULL;
  cvc::MT19937_64 rangen(init_key, length_key);

  std::vector<unsigned long long> testvals(N_TEST);
  for(unsigned long long i = 0; i < N_TEST; ++i){
    testvals[i] = rangen.gen_int64();
  }
  
  // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/mt19937-64.out.txt
  // 7266447313870364031 4946485549665804864 [...]  
  std::cout << "First two values in test sequence: " << testvals[0] << " " << testvals[1] << std::endl;
#ifdef HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
}

