#ifndef _CONTRACT_CVC_TENSOR_H
#define _CONTRACT_CVC_TENSOR_H
/****************************************************
 * contract_cvc_tensor.h
 *
 * Fri Mar 10 14:39:30 CET 2017
 *
 ****************************************************/

#define _ISIGN(i_) ( (i_)==0 ? 0 : ( (i_)>0 ? 1 : -1 )  )

namespace cvc {

void init_contract_cvc_tensor_usource(double *gauge_field, int source_coords[4], complex *phase);

void contract_cvc_tensor_eo ( double *conn_e, double *conn_o, double *contact_term, double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o , double*gauge_field );

void co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (complex *w, fermion_propagator_type *r, fermion_propagator_type *s, double sign, unsigned int N);

void cvc_tensor_eo_subtract_contact_term (double**eo_tensor, double*contact_term, int gsx[4], int have_source );

int cvc_tensor_eo_momentum_projection (double****tensor_tp, double**tensor_eo, int (*momentum_list)[3], int momentum_number);

int cvc_tensor_tp_write_to_aff_file (double***cvc_tp, struct AffWriter_s*affw, char*tag, int (*momentum_list)[3], int momentum_number, int io_proc );

int cvc_tensor_eo_check_wi_position_space (double **tensor_eo);

int apply_constant_cvc_vertex_at_source (double**s, int mu, int fbwd, const unsigned int N );

int contract_write_to_aff_file (double ** const c_tp, struct AffWriter_s*affw, char*tag, const int (* momentum_list)[3], int const momentum_number, int const io_proc );

int contract_write_to_h5_file (double ** const c_tp, void * file, char*tag, const int (*momentum_list)[3], int const momentum_number, int const io_proc );

int contract_local_local_2pt_eo ( double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o,
    int *gamma_sink_list, int gamma_sink_num, int*gamma_source_list, int gamma_source_num, int (*momentum_list)[3], int momentum_number,  struct AffWriter_s*affw, char*tag,
    int io_proc );

int contract_local_cvc_2pt_eo ( double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o,
  int *gamma_sink_list, int gamma_sink_num, int (*momentum_list)[3], int momentum_number,  struct AffWriter_s*affw, char*tag,
  int io_proc );

int cvc_tensor_eo_write_contact_term_to_aff_file ( double *contact_term, struct AffWriter_s*affw, char *tag, int io_proc );

void co_field_pl_eq_tr_propagator_field (complex *w, fermion_propagator_type *r, double sign, unsigned int N);

void co_field_pl_eq_tr_propagator_field_conj (complex *w, fermion_propagator_type *r, double sign, unsigned int N);

void contract_cvc_loop_eo ( double ***loop, double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o , double*gauge_field );

void contract_cvc_loop_eo_lma ( double ***loop, double**eo_evecs_field, double *eo_evecs_norm, int nev, double*gauge_field, double **mzz[2], double **mzzinv[2]);

void contract_cvc_loop_eo_lma_wi ( double **wi, double**eo_evecs_field, double *eo_evecs_norm, int nev, double*gauge_field, double **mzz[2], double **mzzinv[2]);

int cvc_loop_eo_check_wi_position_space_lma ( double ***wwi, double ***loop_lma, double **eo_evecs_field, double *evecs_norm, int nev, double *gauge_field, double **mzz[2], double **mzzinv[2]  );

int cvc_loop_eo_momentum_projection (double****loop_tp, double***loop_eo, int (*momentum_list)[3], int momentum_number);

int cvc_loop_tp_write_to_aff_file (double***cvc_tp, struct AffWriter_s*affw, char*tag, int (*momentum_list)[3], int momentum_number, int io_proc );

int cvc_loop_eo_check_wi_momentum_space_lma ( double **wi, double ***loop_lma, int (*momentum_list)[3], int momentum_number  );

int contract_cvc_tensor_eo_lm_mee (
  double ** const eo_evecs_field, unsigned int const nev,
  double * const gauge_field, double ** const mzz[2], double ** const mzzinv[2],
  struct AffWriter_s * affw, char * const tag,
  int (* const momentum_list)[3], unsigned int const momentum_number,
  unsigned int const io_proc
);

int contract_cvc_tensor_eo_lm_ct (
  double ** const eo_evecs_field, unsigned int const nev,
  double * const gauge_field, double ** const mzz[2], double ** const mzzinv[2],
  struct AffWriter_s * affw, char * const tag,
  unsigned int const io_proc
);

int contract_cvc_tensor_eo_lm_mee_ct (
  double ** const eo_evecs_field, unsigned int const nev,
  double * const gauge_field, double ** const mzz[2], double ** const mzzinv[2],
  struct AffWriter_s * affw, char * const tag,
  int (* const momentum_list)[3], unsigned int const momentum_number,
  unsigned int const io_proc
);


void contract_cvc_local_tensor_eo ( double * const conn_e, double * const conn_o, double ** const sprop_list_e, double ** const sprop_list_o, double ** const tprop_list_e, double ** const tprop_list_o , double * const gauge_field );


/****************************************************
 * average 4x4 tensor spatial components over orbit
 * with totally antisymmetric 3x3 tensor
 ****************************************************/
static inline void antisymmetric_orbit_average_spatial (double *** const d_out, double ****** const d_in, int const dim[3], int const momentum_num, int  const (*momentum_list)[3], int const reim ) {

  int const epsilon_tensor[3][3] = { {0,1,2}, {1,2,0}, {2,0,1} };

  /* double const norm = 1. / ( 2. * (double)(
      (momentum_list[0][0] != 0) +
      (momentum_list[0][1] != 0) +
      (momentum_list[0][2] != 0) ) * (double)momentum_num );
   */

  double const pnorm[3] = {
    2. * sin( M_PI * momentum_list[0][0] / LX_global ),
    2. * sin( M_PI * momentum_list[0][1] / LY_global ),
    2. * sin( M_PI * momentum_list[0][2] / LZ_global )
  };
  double const pnorm2 = pnorm[0] * pnorm[0] + pnorm[1] * pnorm[1] + pnorm[2] * pnorm[2];
  double const norm = 1. / ( 2. * pnorm2 * (double)momentum_num );

  if ( g_verbose > 3 ) fprintf ( stdout, "# [antisymmetric_orbit_average_spatial] norm = %f\n", norm );
#pragma omp prallel for
  for ( int i = 0; i < dim[0]; i++ ) {
    for ( int k = 0; k < dim[1]; k++ ) {
      for ( int l = 0; l < dim[2]; l++ ) {
        /* sum over momentum orbit */
        for ( int imom = 0; imom < momentum_num; imom++ ) {
          /* int const p[3] = { _ISIGN(momentum_list[imom][0]), _ISIGN(momentum_list[imom][1]), _ISIGN(momentum_list[imom][2]) }; */
          int const p[3] = { 
              2. * sin( M_PI *  momentum_list[imom][0] / LX_global ),
              2. * sin( M_PI *  momentum_list[imom][1] / LY_global ),
              2. * sin( M_PI *  momentum_list[imom][2] / LZ_global ) };

          for ( int ia = 0; ia < 3; ia++ ) {
            d_out[i][k][l] += p[epsilon_tensor[ia][0]] * (
                d_in[i][k][imom][epsilon_tensor[ia][1]+1][epsilon_tensor[ia][2]+1][2*l+reim]
              - d_in[i][k][imom][epsilon_tensor[ia][2]+1][epsilon_tensor[ia][1]+1][2*l+reim]
              );
          }  /* end of loop on permutations */
        }  /* end of loop on momenta */
        /* normalize */
        d_out[i][k][l] *= norm;
      }  /* end of loop on dim[2] */
    }  /* end of loop on dim[1] */
  }  /* end of loop on dim[0] */

  return;
}  /* end of antisymmetric_orbit_average_spatial */


}  /* end of namespace cvc */

#endif
