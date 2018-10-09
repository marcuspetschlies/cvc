#ifndef _DUMMY_SOLVER_H
#define _DUMMY_SOLVER_H

#ifdef DUMMY_SOLVER
#  define _TMLQCD_INVERT dummy_solver
namespace cvc {
int dummy_solver ( double * const propagator, double * const source, const int op_id, int write_prop);
}  /* end of namespace cvc */
#else
#  define _TMLQCD_INVERT tmLQCD_invert
#endif

#endif
