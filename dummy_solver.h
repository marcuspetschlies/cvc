#ifndef _DUMMY_SOLVER_H
#define _DUMMY_SOLVER_H

#if ( defined DUMMY_SOLVER ) && ( defined GPU_DIRECT_SOLVER )
#error exclusive XXX_SOLVER definitions
#endif

#if ( defined DUMMY_SOLVER ) || ( ! defined HAVE_TMLQCD_LIBWRAPPER ) 
#  define _TMLQCD_INVERT dummy_solver
namespace cvc {
// int dummy_solver ( double * const propagator, double * const source, const int op_id, int write_prop);
int dummy_solver ( double * const propagator, double * const source, int const op_id );
}  /* end of namespace cvc */
#elif ( defined HAVE_TMLQCD_LIBWRAPPER ) 
#if ( defined GPU_DIRECT_SOLVER )
#  define _TMLQCD_INVERT invert_quda_direct
#elif 
#  define _TMLQCD_INVERT tmLQCD_invert
#endif

#endif

#endif
