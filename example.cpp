/****************************************************
 * p2gg_xspace.c
 *
 * Fr 23. Okt 12:56:08 CEST 2015
 *
 * - originally copied from cvc_exact2_xspace.c
 *
 * PURPOSE:
 * - contractions for P -> gamma gamma in position space
 * DONE:
 * TODO:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
}
#endif

namespace cvc {

void usage (void) {
  fprintf(stdout, "# [usage] this is usage reporting from namespace cvc\n");
  return;
}

}

namespace cvc2 {

void usage (void) {
  fprintf(stdout, "# [usage] this is usage reporting from namespace cvc2\n");
  return;
}

}

using namespace cvc2;

int main(int argc, char **argv) {
  
  int cart_id = 0;
  time_t the_time;


  usage();



  if(cart_id==0) {
    the_time = time(NULL);
    fprintf(stdout, "# [example] %s# [example] end of run\n", ctime(&the_time));
  }

  return(0);

}
