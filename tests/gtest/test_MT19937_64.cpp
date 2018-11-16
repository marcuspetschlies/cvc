#include "MT19937_64.hpp"

#include <gtest/gtest.h>

TEST(MT19937_64, checkFirstTwoOutputs){
  const unsigned long long init_key[4]={0x12345ULL, 0x23456ULL, 0x34567ULL, 0x45678ULL};
  const unsigned long long length_key = 4ULL;
  cvc::MT19937_64 rangen(init_key, length_key);
  
  // http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/mt19937-64.out.txt
  // 7266447313870364031 4946485549665804864 [...]  
  EXPECT_EQ(rangen.gen_int64(), 7266447313870364031ULL);
  EXPECT_EQ(rangen.gen_int64(), 4946485549665804864ULL);
}
