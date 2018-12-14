#pragma once

#include "loop_tools.h"
#include "cvc_linalg.h"
#include "cvc_complex.h"
#include "global.h"

template <typename TYPE>
void contract_twopoint_xdep_gamma5_gamma_snk_only(
    TYPE * const contr, const int idsink, 
    TYPE const * chi, TYPE const * psi,
    const int stride) 
{
#ifdef HAVE_OPENMP
#pragma omp parallel
#endif
  {
    double spinor1[24], spinor2[24];
    complex w;
    size_t iix, cidx;
    FOR_IN_PARALLEL(ix, 0, VOLUME)
    {
      iix = _GSI(ix);
      cidx = ix * stride;
      _fv_eq_gamma_ti_fv(spinor1, idsink, psi+iix);
      _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
      _co_eq_fv_dag_ti_fv(&w, chi+iix, spinor2);

      contr[2*cidx  ] = (TYPE)(w.re);
      contr[2*cidx+1] = (TYPE)(w.im);
    }
  } // parallel section
}

