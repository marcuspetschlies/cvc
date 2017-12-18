#ifndef _DUMMY_SOLVER_H
#define _DUMMY_SOLVER_H

#ifdef DUMMY_SOLVER 
#  define _TMLQCD_INVERT dummy_solver
namespace cvc {
int dummy_solver ( double * const, double * const, const int, int);
}  /* end of namespace cvc */
#else
#  define _TMLQCD_INVERT tmLQCD_invert
#endif

#endif
