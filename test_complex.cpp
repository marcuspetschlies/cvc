/****************************************************
 * test_sp.cpp
 *
 * So 5. Jun 18:15:21 CEST 2016
 *
 * PURPOSE:
 * TODO:
 * DONE:
 * CHANGES:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#define MAIN_PROGRAM
#include "cvc_complex.h"

using namespace cvc;


void usage(void) {
  fprintf(stdout, "usage:\n");
  exit(0);
}

int main(int argc, char **argv) {
  
  int c, verbose=0; 
  double d;
  complex w;
  double _Complex z;

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?v")) != -1) {
    switch (c) {
    case 'v':
      verbose = 1;
      break;
    case 'h':
    case '?':
    default:
      usage();
      break;
    }
  }

  fprintf(stdout, "# [] size of double = %lu\n", sizeof(double));
  fprintf(stdout, "# [] size of complex = %lu\n", sizeof(complex));
  fprintf(stdout, "# [] size of double _Complex = %lu\n", sizeof(double _Complex));

  return(0);
}
