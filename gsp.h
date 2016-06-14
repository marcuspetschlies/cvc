/***************************************************
 * gsp.h                                     *
 ***************************************************/
#ifndef _GSP_H
#define _GSP_H
namespace cvc {

int gsp_init (double ******gsp_out, int Np, int Ng, int Nt, int Nv);

int gsp_fini(double******gsp);

void gsp_make_eo_phase_field (double*phase_e, double*phase_o, int *momentum);

}  /* end of namespace cvc */
#endif
