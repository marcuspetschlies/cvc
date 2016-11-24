#ifndef _CONTRACT_CVC_TENSOR_H
#define _CONTRACT_CVC_TENSOR_H
/****************************************************
 * contract_cvc_tensor.h
 *
 * Thu Nov 17 10:37:10 CET 2016
 *
 ****************************************************/

namespace cvc {

void init_contract_cvc_tensor_gperm(void);

void init_contract_cvc_tensor_usource(double *gauge_field, int source_coords[4]);

void contract_cvc_tensor(double *conn, double *contact_term, double*fwd_list[5][12], double*bwd_list[5][12], double*fwd_list_eo[2][5][12], double*bwd_list_eo[2][5][12]);

void contract_cvc_m(double *conn, int gid, double*fwd_list[5][12], double*bwd_list[5][12], double*fwd_list_eo[2][5][12], double*bwd_list_eo[2][5][12]);

void contract_m_m(double *conn, int idsource, int idsink, double*fwd_list[12], double*bwd_list[12], double*fwd_list_eo[2][12], double*bwd_list_eo[2][12]);

void contract_cvc_loop (double*conn, double**field_list, int field_number, int *momentum_list[3], int momentum_number, double*weight, double mass, double**eo_work);

void contract_m_loop (double*conn, double**field_list, int field_number, int *gid_list, int gid_number, int *momentum_list[3], int momentum_number, double*weight, double mass, double**eo_work);

}  /* end of namespace cvc */

#endif
