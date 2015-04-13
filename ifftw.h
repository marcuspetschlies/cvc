#ifndef _IFFTW_H
#define _IFFTW_H

#ifdef MPI
#  include <fftw_mpi.h>
#else
#  ifdef OPENMP
#    include <fftw_threads.h>
#  else
#    include <fftw.h>
#  endif
#endif

#endif
