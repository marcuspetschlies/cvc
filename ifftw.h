#ifndef _IFFTW_H
#define _IFFTW_H

#ifdef HAVE_MPI
#  include "fftw_mpi.h"
#  ifdef HAVE_OPENMP
#    include "fftw_threads.h"
#  endif
#else
#  ifdef HAVE_OPENMP
#    include "fftw_threads.h"
#    include "fftw.h"
#  else
#    include "fftw.h"
#  endif
#endif

#endif
