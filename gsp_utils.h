/***************************************************
 * gsp_utils.h                                     *
 ***************************************************/
#ifndef _GSP_UTILS_H
#define _GSP_UTILS_H
namespace cvc {

int gsp_init (double ******gsp_out, int Np, int Ng, int Nt, int Nv);

int gsp_fini(double******gsp);

int gsp_reset (double ******gsp_out, int Np, int Ng, int Nt, int Nv);

void gsp_make_eo_phase_field (double*phase_e, double*phase_o, int *momentum);

void gsp_make_o_phase_field_sliced3d (double _Complex*phase, int *momentum);

void gsp_make_eo_phase_field_sliced3d (double _Complex**phase, int *momentum, int eo);

int gsp_read_node (double ***gsp, int num, int momentum[3], int gamma_id, char*tag);

int gsp_write_eval(double *eval, int num, char*tag);

int gsp_read_eval(double **eval, int num, char*tag);

void co_eq_tr_gsp_ti_gsp (complex *w, double**gsp1, double**gsp2, double*lambda, int num);

void co_eq_tr_gsp (complex *w, double**gsp1, double*lambda, int num);

int gsp_printf (double ***gsp, int num, char*name, FILE*ofs);

void gsp_pl_eq_gsp (double _Complex **gsp1, double _Complex **gsp2, int num);

void co_eq_gsp_diag_ti_gsp_diag (complex *w, double**gsp1, double**gsp2, double*lambda, int num);

int gsp_calculate_v_w_block_asym(double*gsp_out, double**V, double**W, unsigned int numV, unsigned int numW);

void co_eq_gsp_diag (complex *w, double**gsp1, int num);

void co_pl_eq_gsp_diag (complex *w, double**gsp1, int num);

int gsp_calculate_xv_from_v (double **xv, double **v, double **work, int num, double mass, unsigned int N);

int gsp_calculate_w_from_xv_and_v (double **w, double **xv, double **v, double **work, int num, double mass, unsigned int N);

/***************************************************/
/***************************************************/

}  /* end of namespace cvc */
#endif
