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


/***********************************************************/
/***********************************************************/

/***********************************************************
 *
 ***********************************************************/
inline void contract_cvc_tensor_eo_lm_key ( char * const key, char * const prefix, char * const midfix, int const t, int const mu, int const b ) {

  char midfix_str[20]="";
  char t_str[10]="";
  char mu_str[10]="";

  if ( midfix != NULL ) {
    sprintf ( midfix_str, "%s/", midfix );
  }

  if ( t > -1  ) {
    sprintf ( t_str, "t%.2d/", t );
  }

  if ( mu > -1  ) {
    sprintf ( t_str, "mu%d/", mu );
  }

  sprintf ( key, "%s/%s%s%s/b%.2d", prefix, midfix_str, t_str, mu_str, b );

  return;
}  // end of contract_cvc_tensor_eo_lm_key


}  // end of namespace cvc

#endif
