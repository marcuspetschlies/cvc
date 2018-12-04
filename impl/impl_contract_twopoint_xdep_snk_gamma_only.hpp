#include "loop_tools.h"
#include "cvc_linalg.h"
#include "cvc_complex.h"
#include "global.h"

template <typename T>
void contract_twopoint_xdep_snk_gamma_only(
    T * const contr, const int idsink, 
    T const * chi, T const * psi, 
    int const stride, double const factor)
{
#pragma omp parallel
  {
    unsigned int cidx;
    double spinor1[24], spinor2[24];
    complex w;
    FOR_IN_PARALLEL(ix, 0, VOLUME)
    {
      size_t cidx = ix * stride;
      _fv_eq_gamma_ti_fv(spinor1, idsink, psi+_GSI(ix));
      _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
      _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), spinor2);

      contr[2*cidx  ] = factor * w.re;
      contr[2*cidx+1] = factor * w.im;
    }
  } // parallel section
}

