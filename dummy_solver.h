#ifndef _DUMMY_SOLVER_H
#define _DUMMY_SOLVER_H

#if (defined DUMMY_SOLVER ) || !( defined HAVE_TMLQCD_LIBWRAPPER )
#  define _TMLQCD_INVERT dummy_solver
namespace cvc {
/* int dummy_solver ( double * const, double * const, const int, int); */
int dummy_solver ( double * const, double * const, const int );
}  /* end of namespace cvc */
#else
#  define _TMLQCD_INVERT tmLQCD_invert
#endif

#endif
