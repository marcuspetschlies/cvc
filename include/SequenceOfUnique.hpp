// Based on https://github.com/preshing/RandomSequence/blob/master/randomsequence.h

#ifndef SEQUENCEOFUNIQUE_HPP
#define SEQUENCEOFUNIQUE_HPP

// we need a 128-bit unsigned integer because we need to square a 64-bit integer
#include <boost/multiprecision/cpp_int.hpp>
using namespace boost::multiprecision;

// 2^64-59 is prime
constexpr unsigned long long SOUprime = 18446744073709551557ULL;

namespace cvc {

/**
 * @brief Class to generate a sequence of unique 64-bit integers
 * This will step through a larger portion of the space in [0,2^64)
 * with random-looking steps before starting over. 
 * The nice thing is that given a seed and an offset, the position
 * in the sequence can always be reset to an exact point.
 */
class SequenceOfUnique
{
private:
  unsigned int m_index;
  unsigned int m_intermediateOffset;

  unsigned long long permuteQPR(unsigned long long x)
  {
    if (x >= SOUprime)
      return x;  // The 59 integers out of range are mapped to themselves.
    // we need to up-convert to uint128_t to take the square and after the modulo, we can
    // safely down-convert
    unsigned long long residue = static_cast<unsigned long long>(((uint128_t) x * (uint128_t)x) % (uint128_t)SOUprime);
    return (x <= SOUprime / 2) ? residue : SOUprime - residue;
  }

public:
  SequenceOfUnique(unsigned long long seedBase, unsigned long long seedOffset)
  {
    m_index = permuteQPR(permuteQPR(seedBase) + 7507824837999787008ULL);
    m_intermediateOffset = permuteQPR(permuteQPR(seedOffset) + 5078388643492239360ULL);
  }

  unsigned long long next()
  {
    return permuteQPR((permuteQPR(m_index++) + m_intermediateOffset) ^ 6624225796869099520ULL);
  }
};

} // namespace(cvc)


#endif // SEQUENCEOFUNIQUE_HPP

