#ifndef _CONTRACT_CVC_TENSOR_H
#define _CONTRACT_CVC_TENSOR_H
/****************************************************
 * contract_cvc_tensor.h
 *
 * Tue Feb  7 15:28:04 CET 2017
 *
 ****************************************************/

namespace cvc {

void init_contract_cvc_tensor_usource(double *gauge_field, int source_coords[4]);

void contract_cvc_tensor_eo ( double *conn, double *contact_term, double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o  );

}  /* end of namespace cvc */

#endif
