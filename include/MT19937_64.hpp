#ifndef MT19937_64_HPP
#define MT19937_64_HPP

/* 
   A C-program for MT19937-64 (2004/9/29 version).
   Coded by Takuji Nishimura and Makoto Matsumoto.

   This is a 64-bit version of Mersenne Twister pseudorandom number
   generator.

   Before using, initialize the state by using init_genrand64(seed)  
   or init_by_array64(init_key, key_length).

   Copyright (C) 2004, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   References:
   T. Nishimura, ``Tables of 64-bit Mersenne Twisters''
     ACM Transactions on Modeling and 
     Computer Simulation 10. (2000) 348--357.
   M. Matsumoto and T. Nishimura,
     ``Mersenne Twister: a 623-dimensionally equidistributed
       uniform pseudorandom number generator''
     ACM Transactions on Modeling and 
     Computer Simulation 8. (Jan. 1998) 3--30.

   Any feedback is very welcome.
   http://www.math.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove spaces)
*/

/* Class implementation by Bartosz Kostrzewa, 2018 */

namespace cvc {

constexpr unsigned long long MT19937_64_NN = 312;
constexpr unsigned long long MT19937_64_MM = 156;
constexpr unsigned long long MT19937_64_MATRIX_A = 0xB5026F5AA96619E9ULL;
constexpr unsigned long long MT19937_64_UM = 0xFFFFFFFF80000000ULL;
constexpr unsigned long long MT19937_64_LM = 0x7FFFFFFFULL;
constexpr unsigned long long MT19937_64_mag01[2] = {0ULL, MT19937_64_MATRIX_A};

/**
 * @brief class wrapper for 64-bit Mersenne Twister
 */
class MT19937_64 {
public:
  MT19937_64(const unsigned long long seed)
  {
    init(seed);
  }

  MT19937_64(const MT19937_64 & other) 
  {
    for(int i = 0; i<MT19937_64_NN; ++i){
      mt[i] = other.get_mt(i);
    }
    mti = other.get_mti();
    initialised = other.get_init();
  }
  
  /**
   * @brief Constructor which initialises with an array of seeds.
   * The default test program for MT19937_64 initialises the RNG with
   * a seed array of length 4. In order to check for correctness, we
   * need to support this mode as well.
   *
   * @param init_key[]
   * @param key_length
   */
  MT19937_64(const unsigned long long init_key[],
             const unsigned long long key_length)
  {
    init_by_array(init_key, key_length);
  }


  /**
   * @brief Default constructor.
   * WARNING: the default constructor is empty on purpose. If default-constructed,
   * one of the initialisation functions has to be called externally! 
   */
  MT19937_64()
  {
  };

	unsigned long long gen_int64(void)
  {
    int i;
    unsigned long long x;

    if (mti >= MT19937_64_NN) { /* generate NN words at one time */

        for (i=0;i<MT19937_64_NN-MT19937_64_MM;i++) {
            x = (mt[i]&MT19937_64_UM)|(mt[i+1]&MT19937_64_LM);
            mt[i] = mt[i+MT19937_64_MM] ^ (x>>1) ^ MT19937_64_mag01[(int)(x&1ULL)];
        }
        for (;i<MT19937_64_NN-1;i++) {
            x = (mt[i]&MT19937_64_UM)|(mt[i+1]&MT19937_64_LM);
            mt[i] = mt[i+(MT19937_64_MM-MT19937_64_NN)] ^ (x>>1) ^ MT19937_64_mag01[(int)(x&1ULL)];
        }
        x = (mt[MT19937_64_NN-1]&MT19937_64_UM)|(mt[0]&MT19937_64_LM);
        mt[MT19937_64_NN-1] = mt[MT19937_64_MM-1] ^ (x>>1) ^ MT19937_64_mag01[(int)(x&1ULL)];

        mti = 0;
    }
  
    x = mt[mti++];

    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);

    return x;
  }
  
  // generate random number in [0,1] interval
  double gen_real(void)
  {
    return( gen_int64() >> 11) * (1.0/9007199254740991.0);
  }

  unsigned long long get_mt(const int i) const
  {
    return( mt[i] );
  }

  int get_mti(void) const
  {
    return(mti);
  }

  bool get_init(void) const
  {
    return(initialised);
  }

  void init(const unsigned long long seed)
  {
    mt[0] = seed;
    for (mti=1; mti<MT19937_64_NN; mti++)
      mt[mti] =  (6364136223846793005ULL * (mt[mti-1] ^ (mt[mti-1] >> 62)) + mti);
    initialised = true; 
  }

  void init_by_array(const unsigned long long init_key[],
  		               const unsigned long long key_length)
  {
    unsigned long long i, j, k;
    init(19650218ULL);
    i=1; j=0;
    k = (MT19937_64_NN>key_length ? MT19937_64_NN : key_length);
    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 62)) * 3935559000370003845ULL))
          + init_key[j] + j; /* non linear */
        i++; j++;
        if (i>=MT19937_64_NN) { mt[0] = mt[MT19937_64_NN-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=MT19937_64_NN-1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 62)) * 2862933555777941757ULL))
          - i; /* non linear */
        i++;
        if (i>=MT19937_64_NN) { mt[0] = mt[MT19937_64_NN-1]; i=1; }
    }
  
    mt[0] = 1ULL << 63; /* MSB is 1; assuring non-zero initial array */ 
  }

private:

  bool initialised;
  unsigned long long mt[MT19937_64_NN]; /* the 312 64-bit uints which give the state of the RNG */  
  int mti; /* position in the RNG sequence, part of internal state */
};

}

#endif
