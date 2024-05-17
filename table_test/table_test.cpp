/**
 * Test efficiency of table structure alternatives.
 */

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <memory>
#include "table_init_d.h"

using namespace cvc;
using my_clock = std::chrono::high_resolution_clock;
template<typename T1, typename T2>
double to_us(std::chrono::duration<T1, T2> dur) {
  constexpr double BIL = 1e9;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count() / BIL;
}

struct const_dslice_level1 {
  const double* arr;
  size_t i;
  size_t N2;
  const_dslice_level1(const double* arr, size_t i, size_t N2) :
      arr(arr), i(i), N2(N2) {}
  double operator[](size_t k) const {
    return arr[i*N2 + k];
  }
};
struct const_dslice_level2 {
  const double* arr;
  size_t i;
  size_t N1;
  size_t N2;
  const_dslice_level2(const double* arr, size_t i, size_t N1, size_t N2) :
      arr(arr), i(i), N1(N1), N2(N2) {}
  const_dslice_level1 operator[](size_t j) const {
    return const_dslice_level1(arr, i*N1+j, N2);
  }
};
struct dslice_level1 {
  double* arr;
  size_t i;
  size_t N2;
  dslice_level1(double* arr, size_t i, size_t N2) :
      arr(arr), i(i), N2(N2) {}
  double& operator[](size_t k) {
    return arr[i*N2 + k];
  }
};
struct dslice_level2 {
  double* arr;
  size_t i;
  size_t N1;
  size_t N2;
  dslice_level2(double* arr, size_t i, size_t N1, size_t N2) :
      arr(arr), i(i), N1(N1), N2(N2) {}
  dslice_level1 operator[](size_t j) {
    return dslice_level1(arr, i*N1+j, N2);
  }
};


class dtable_3level {
 public:
  dtable_3level(size_t N0, size_t N1, size_t N2) : N0(N0), N1(N1), N2(N2) {
    arr = (double*) malloc(N0 * N1 * N2 * sizeof(double));
  }
  ~dtable_3level() {
    free(arr);
  }

  const_dslice_level2 operator[](size_t i) const {
    return const_dslice_level2(arr, i, N1, N2);
  }
  dslice_level2 operator[](size_t i) {
    return dslice_level2(arr, i, N1, N2);
  }
  
 private:
  double* arr;
  const size_t N0;
  const size_t N1;
  const size_t N2;
};


int main(int argc, char** argv) {
  constexpr size_t VOLUME = 16*16*16*16;
  constexpr size_t N0 = 2;
  constexpr size_t N1 = 12;
  constexpr size_t N2 = 24 * VOLUME;

  /// Version A:
  double *** tableA = init_3level_dtable(N0, N1, N2);
  auto _start = my_clock::now();
  for (int i = 0; i < (int)N0; ++i) {
    for (int j = 0; j < (int)N1; ++j) {
      for (int k = 0; k < (int)N2; ++k) {
        tableA[i][j][k] = 1.234 * (((i*N1 + j)*N2 + k) % 7);
      }
    }
  }
  auto _dur = to_us(my_clock::now() - _start);
  printf("Time A build: %.04f\n", _dur);
  _start = my_clock::now();
  double outA = 0.0;
  for (int i = 0; i < (int)N0; ++i) {
    for (int j = 0; j < (int)N1; ++j) {
      for (int k = 0; k < (int)N2; ++k) {
        if (((i*N1 + j)*N2 + k) % 3 == 0) {
          outA += tableA[i][j][k];
        }
        else {
          outA -= 2*tableA[i][j][k];
        }
      }
    }
  }
  printf("outA = %f\n", outA);
  _dur = to_us(my_clock::now() - _start);
  printf("Time A reduce: %.04f\n", _dur);
  fini_3level_dtable(&tableA);

  /// Version B:
  dtable_3level tableB = dtable_3level(N0, N1, N2);
  _start = my_clock::now();
  for (int i = 0; i < (int)N0; ++i) {
    for (int j = 0; j < (int)N1; ++j) {
      for (int k = 0; k < (int)N2; ++k) {
        tableB[i][j][k] = 1.234 * (((i*N1 + j)*N2 + k) % 7);
      }
    }
  }
  _dur = to_us(my_clock::now() - _start);
  printf("Time B build: %.04f\n", _dur);
  _start = my_clock::now();
  double outB = 0.0;
  for (int i = 0; i < (int)N0; ++i) {
    for (int j = 0; j < (int)N1; ++j) {
      for (int k = 0; k < (int)N2; ++k) {
        if (((i*N1 + j)*N2 + k) % 3 == 0) {
          outB += tableB[i][j][k];
        }
        else {
          outB -= 2*tableB[i][j][k];
        }
      }
    }
  }
  printf("outB = %f\n", outB);
  _dur = to_us(my_clock::now() - _start);
  printf("Time B reduce: %.04f\n", _dur);
}
