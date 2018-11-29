
#ifdef HAVE_OPENMP
#define PARALLEL_AND_FOR(_loopvar,_start,_end) \
_Pragma("omp parallel for") \
  for(size_t (_loopvar) = (_start); (_loopvar) < (_end); ++(_loopvar))
#else
#define PARALLEL_AND_FOR(_loopvar,_start,_end) \
  for(size_t (_loopvar) = (_start); (_loopvar) < (_end); ++(_loopvar))
#endif

#ifdef HAVE_OPENMP
#define FOR_IN_PARALLEL(_loopvar, _start, _end) \
_Pragma("omp for") \
  for(size_t (_loopvar) = (_start); (_loopvar) < (_end); ++(_loopvar))
#else
#define FOR_IN_PARALLEL(_loopvar, _strt, _end) \
  for(size_t (_loopvar) = (_start); (_loopvar) < (_end); ++(_loopvar))
#endif

