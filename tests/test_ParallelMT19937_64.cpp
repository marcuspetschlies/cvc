
#define MAIN_PROGRAM
#include "global.h"
#undef MAIN_PROGRAM

#include "ParallelMT19937_64.hpp"
#include "enums.hpp"
#include "index_tools.hpp"
#include "Stopwatch.hpp"
#include "Core.hpp"
#include "debug_printf.hpp"
#include "loop_tools.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <iostream>
#include <vector>
#include <array>
#include <fstream>


constexpr unsigned int startsite = 0;
constexpr unsigned int no_testsites = 1000;
constexpr unsigned int no_testvals = 50000;

using namespace cvc;

int main(int argc, char** argv)
{
  Core core(argc,argv);
  if( !(core.is_initialised()) ){
    std::cout << "Core initialisation failed!\n";
    return(CVC_EXIT_CORE_INIT_FAILURE);
  } 

#ifdef HAVE_MPI
  Stopwatch sw(g_cart_grid);
#else
  Stopwatch sw(0);
#endif

  debug_printf(0, 0, "\n\n############ TESTING ParallelMT19937_64 ###############\n\n");

  sw.reset();
  ParallelMT19937_64 rangen(982932ULL);
  sw.elapsed_print("ParallelMT19937_64 initialisation");

  ParallelMT19937_64 rangen2(982932ULL);

  ParallelMT19937_64 rangen3(982932ULL);

  MT19937_64 sitegen(877123ULL);

  debug_printf(0, 0, "Generatign RNs for statistical tests\n");
  // output for statistical tests
  std::vector< std::vector<double> > ordered(no_testsites);
  std::vector< std::vector<double> > random(no_testsites);
  std::vector< double > z2(24*VOLUME);

  for(unsigned int li = 0; li < no_testsites; ++li){
    ordered[li].resize(no_testvals);
    random[li].resize(no_testvals);
  }

  // generate a random sequence of test sites 
  std::array<size_t, no_testsites> testsites;
  for(unsigned int li = 0; li < no_testsites; ++li){
    testsites[li] = static_cast<size_t>(sitegen.gen_int64() % VOLUME);
  }

  // generate sequences of random numbers using generators sitting at
  // consecutive lattice sites
  #pragma omp parallel for
  for(unsigned int li = startsite; li < no_testsites+startsite; ++li){
    for(unsigned int ri = 0; ri < no_testvals; ++ri){
      ordered[li][ri] = rangen.gen_real_at_site(li);
    }
  }

  // generate sequences of random numbers using generators randomly
  // chosen on the lattice
  #pragma omp parallel for
  for(unsigned int li = 0; li < no_testsites; ++li){
    for(unsigned int ri = 0; ri < no_testvals; ++ri){
      random[li][ri] = rangen2.gen_real_at_site( testsites[li] );
    }
  }

  // generate some z2 cross z2 random numbers
  rangen3.gen_z2(z2.data(),
                 24);

  debug_printf(0,0, "Writing data for statistical tests\n");
  std::ofstream random_outfile("random.dat",
                               std::ios::out | std::ios::binary);
  std::ofstream ordered_outfile("ordered.dat",
                                std::ios::out | std::ios::binary);
  std::ofstream z2_outfile("z2.dat",
                           std::ios::out | std::ios::binary);

  for(unsigned int li = 0; li < no_testsites; ++li){
    random_outfile.write((char*)random[li].data(), no_testvals*sizeof(double));
    ordered_outfile.write((char*)ordered[li].data(), no_testvals*sizeof(double));
  }
  z2_outfile.write((char*)z2.data(), 24*VOLUME*sizeof(double));

  random_outfile.close();
  ordered_outfile.close();
  z2_outfile.close();
  
  debug_printf(0, 0, "Timing generation of 50*24*VOLUME RNs\n");
  std::vector<double> testvec(24*VOLUME);
  sw.reset();
  // generate 24*VOLUME random numbers 50 times
  for(int irun = 0; irun < 50; irun++){
    rangen.gen_real(testvec.data(), 24);
    std::cout << testvec[989] << std::endl;
  }
  sw.elapsed_print("ParallelMT19937_64 test generation");
  
  return 0;
}
