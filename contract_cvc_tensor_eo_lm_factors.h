#ifndef _CONTRACT_CVC_TENSOR_EO_LM_FACTORS_H
#define _CONTRACT_CVC_TENSOR_EO_LM_FACTORS_H

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

namespace cvc {

/***********************************************************
 * contract_cvc_tensor_eo_lm_factors
 ***********************************************************/
int contract_cvc_tensor_eo_lm_factors ( double ** const eo_evecs_field, unsigned int const nev, double * const gauge_field, double ** const mzz[2], double ** const mzzinv[2], struct AffWriter_s * affw, char * const tag, const int (*momentum_list)[3], unsigned int const momentum_number, unsigned int const io_proc, unsigned int const block_length );

}  // end of namespace cvc

#endif
