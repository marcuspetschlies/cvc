#ifndef _CONTRACT_CVC_TENSOR_ALL2ALL_H
#define _CONTRACT_CVC_TENSOR_ALL2ALL_H

/****************************************************
 * contract_cvc_tensor_all2all.h
 ****************************************************/

namespace cvc {

/************************************************************
 * vdag_w_reduce_write
 ************************************************************/
int vdag_w_reduce_write (double _Complex ***contr, double _Complex **V, double _Complex **W, int dimV, int dimW, char *aff_path, struct AffWriter_s *affw, struct AffNode_s *affn, int io_proc, double _Complex **V_ts, double _Complex **W_ts, double *mcontr_buffer);

/************************************************************
 * calculate gsp V^+ Gamma(p) W
 ************************************************************/
int contract_vdag_gloc_w_blocked (double**V, int numV, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size );

/************************************************************
 * calculate gsp V^+ Gamma(p) Phi
 ************************************************************/
int contract_vdag_gloc_phi_blocked (double**V, double**Phi, int numV, int numPhi, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size );

/*******************************************************************************************
 * calculate V^+ cvc-vertex S
 *******************************************************************************************/
int contract_vdag_cvc_w_blocked (double**V, int numV, int momentum_number, int (*momentum_list)[3], struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size );

/*******************************************************************************************
 * calculate V^+ cvc-vertex Phi
 *******************************************************************************************/
int contract_vdag_cvc_phi_blocked (double**V, double**Phi, int numV, int numPhi, int momentum_number, int (*momentum_list)[3], struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size );

}  // end of namespace cvc

#endif
