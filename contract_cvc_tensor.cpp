/****************************************************
 * contract_cvc_tensor.cpp
 *
 * Sun Feb  5 13:23:50 CET 2017
 *
 * PURPOSE:
 * - contractions for cvc-cvc tensor
 * DONE:
 * TODO:
 ****************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

#include "cvc_complex.h"
#include "ilinalg.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"

namespace cvc {

static double *Usource[4], Usourcebuffer[72];
static int source_proc_id;
static unsigned int source_location;
static int source_location_iseven;

/***********************************************************
 * initialize Usource
 *
 * NOTE: the gauge field at source is multiplied with the
 *       boundary phase
 ***********************************************************/
void init_contract_cvc_tensor_usource(double *gauge_field, int source_coords[4]) {

  int gsx[4] = {source_coords[0], source_coords[1], source_coords[2], source_coords[3] };
  int sx[4];
  int exitstatus;
  double ratime, retime;

  /***********************************************************
   * determine source coordinates, find out, if source_location is in this process
   ***********************************************************/
  ratime = _GET_TIME;

  if( ( exitstatus = get_point_source_info (gsx, sx, &source_proc_id) ) != 0  ) {
    fprintf(stderr, "[init_contract_cvc_tensor_usource] Error from get_point_source_info, status was %d\n", exitstatus);
    EXIT(1);
  }

  Usource[0] = Usourcebuffer;
  Usource[1] = Usourcebuffer+18;
  Usource[2] = Usourcebuffer+36;
  Usource[3] = Usourcebuffer+54;

  if( source_proc_id == g_cart_id ) { 
    source_location = g_ipt[sx[0]][sx[1]][sx[2]][sx[3]];
    source_location_iseven = g_iseven[source_location] ;
    fprintf(stdout, "# [init_contract_cvc_tensor_usource] local source coordinates: %u = (%3d,%3d,%3d,%3d), is even = %d\n",
       source_location, sx[0], sx[1], sx[2], sx[3], source_location_iseven);
    _cm_eq_cm_ti_co(Usource[0], &g_gauge_field[_GGI(source_location,0)], &co_phase_up[0]);
    _cm_eq_cm_ti_co(Usource[1], &g_gauge_field[_GGI(source_location,1)], &co_phase_up[1]);
    _cm_eq_cm_ti_co(Usource[2], &g_gauge_field[_GGI(source_location,2)], &co_phase_up[2]);
    _cm_eq_cm_ti_co(Usource[3], &g_gauge_field[_GGI(source_location,3)], &co_phase_up[3]);
  }

#ifdef HAVE_MPI
  MPI_Bcast(Usourcebuffer, 72, MPI_DOUBLE, source_proc_id, g_cart_grid);
#endif
  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [init_contract_cvc_tensor_usource] time for init_contract_cvc_tensor_usource = %e seconds\n", retime-ratime);
}  /* end of init_contract_cvc_tensor_usource */

  /***********************************************************************************************************/
  /***********************************************************************************************************/

/***********************************************************
 * reduction
 *
 * w += tr ( g5 r^+ g5 s )
 ***********************************************************/
void co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (complex *w, fermion_propagator_type *r, fermion_propagator_type *s, double sign, unsigned int N) {

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  unsigned int ix;
  complex *w_ = NULL, wtmp;
  fermion_propagator_type r_=NULL, s_=NULL;
  fermion_propagator_type fp1, fp2;

  create_fp( &fp1 );
  create_fp( &fp2 );

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( ix=0; ix<N; ix++ ) {
    w_ = w + ix;
    r_ = r[ix];
    s_ = s[ix];

    /* multiply fermion propagator from the left and right with gamma_5 */
    _fp_eq_gamma_ti_fp( fp1, 5, r_ );
    _fp_eq_fp_ti_gamma( fp2, 5, fp1 );
    _co_eq_tr_fp_dagger_ti_fp ( &wtmp, fp2,  s_);
    _co_pl_eq_co_ti_re(w_, &wtmp, sign);

  }

  free_fp( &fp1 );
  free_fp( &fp2 );

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field */

  /***********************************************************************************************************/
  /***********************************************************************************************************/

/***********************************************************
 * eo-prec contractions for cvc - cvc tensor
 *
 * NOTE: neither conn_e nor conn_o nor contact_term 
 *       are initialized to zero here
 ***********************************************************/
void contract_cvc_tensor_eo ( double *conn_e, double *conn_o, double *contact_term, double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o , double*gauge_field ) {
  
  const unsigned int Vhalf = VOLUME / 2;
  const size_t sizeof_eo_propagator_field = Vhalf * 288 * sizeof(double);

  int mu, nu;
  int exitstatus;
  double ratime, retime;
  complex *conn_ = NULL;

  /* auxilliary fermion propagator field with halo */
  fermion_propagator_type *fp_aux        = create_fp_field( (VOLUME+RAND)/2 );

  /* fermion propagator fields without halo */
  fermion_propagator_type *gamma_fp_X[4];
  gamma_fp_X[0] = create_fp_field( Vhalf );
  gamma_fp_X[1] = create_fp_field( Vhalf );
  gamma_fp_X[2] = create_fp_field( Vhalf );
  gamma_fp_X[3] = create_fp_field( Vhalf );

  fermion_propagator_type *fp_Y_gamma       = create_fp_field( Vhalf );
  fermion_propagator_type *gamma_fp_Y_gamma = create_fp_field( Vhalf );
  fermion_propagator_type *fp_X             = create_fp_field( Vhalf );

#if 0
  contact_term[0] = 0.; contact_term[1] = 0.;
  contact_term[2] = 0.; contact_term[3] = 0.;
  contact_term[4] = 0.; contact_term[5] = 0.;
  contact_term[6] = 0.; contact_term[7] = 0.;
#endif  /* of if 0 */

  /**********************************************************
   **********************************************************
   **
   ** contractions
   **
   **********************************************************
   **********************************************************/  
  ratime = _GET_TIME;

 
  /**********************************************************
   * (1) X = T^e, Y = S^o
   **********************************************************/  

  /* fp_X = tprop^e */
  exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_X, tprop_list_e+48, Vhalf);
  /* fp_aux = fp_X */
  memcpy(fp_aux[0][0], fp_X[0][0], sizeof_eo_propagator_field );

  /* gamma_fp_X^o = Gamma_mu^f fp_aux^e */
  for(mu=0; mu<4; mu++) {
    apply_cvc_vertex_propagator_eo ( gamma_fp_X[mu], fp_aux, mu, 0, gauge_field, 1);
  }

  /* loop on nu */
  for( nu=0; nu<4; nu++ )
  {

    /* fp_Y_gamma^o = sprop^o , source + nu */
    exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_Y_gamma, sprop_list_o+12*nu, Vhalf);

    /* fp_Y_gamma^o = fp_Y_gamma^o * Gamma_nu^b  */
    apply_propagator_constant_cvc_vertex ( fp_Y_gamma, fp_Y_gamma, nu, 1, Usource[nu], Vhalf );

    for( mu=0; mu<4; mu++ ) {

      conn_ = (complex*)( conn_o + (4*mu+nu)*2*Vhalf );

      /* contract S^o Gamma_nu, Gamma_mu T^e */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, fp_Y_gamma, gamma_fp_X[mu], 1., Vhalf);


      /* fp_aux^o = fp_Y_gamma^o */
      memcpy(fp_aux[0][0], fp_Y_gamma[0][0], sizeof_eo_propagator_field );
      
      /* gamma_fp_Y_gamma^e = Gamma_mu^f fp_aux^o */
      apply_cvc_vertex_propagator_eo ( gamma_fp_Y_gamma, fp_aux, mu, 0, gauge_field, 0);

      conn_ = (complex*)( conn_e + (4*mu+nu)*2*Vhalf );

      /* contract gamma_fp_Y_gamma^e, T^e */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, gamma_fp_Y_gamma, fp_X, -1., Vhalf);

    }
#if 0 
#endif  /* of if 0 */

    /* contribution to contact term */
    if( (source_proc_id == g_cart_id) && !source_location_iseven ) {
      /* gamma_fp_X[nu] = Gamma_nu^f T^e */
      complex w;
      _co_eq_tr_fp ( &w, gamma_fp_X[nu][g_lexic2eosub[source_location]] );
      contact_term[2*nu  ] -= w.re;
      contact_term[2*nu+1] -= w.im;
    }

  }  /* end of loop on nu */

  /**********************************************************
   * (2) X = S^e, Y = T^o
   **********************************************************/  

  /* fp_X = sprop^e */
  exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_X, sprop_list_e+48, Vhalf);
  /* fp_aux = fp_X */
  memcpy(fp_aux[0][0], fp_X[0][0], sizeof_eo_propagator_field );

  /* gamma_fp_X^o = Gamma_mu^f fp_aux^e */
  for(mu=0; mu<4; mu++) {
    apply_cvc_vertex_propagator_eo ( gamma_fp_X[mu], fp_aux, mu, 0, gauge_field, 1);
  }

  /* loop on nu */
  for( nu=0; nu<4; nu++ ) 
  {
  

    /* fp_Y_gamma^o = tprop^o , source + nu */
    exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_Y_gamma, tprop_list_o+12*nu, Vhalf);

    /* fp_Y_gamma^o = fp_Y_gamma^o * Gamma_nu^b  */
    apply_propagator_constant_cvc_vertex ( fp_Y_gamma, fp_Y_gamma, nu, 1, Usource[nu], Vhalf );

    for( mu=0; mu<4; mu++ ) {

      conn_ = (complex*)( conn_o + (4*mu+nu)*2*Vhalf );

      /* contract Gamma_mu^f S^e, T^o Gamma_nu^b */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, gamma_fp_X[mu], fp_Y_gamma, 1., Vhalf);

      /* fp_aux^o = fp_Y_gamma^o */
      memcpy(fp_aux[0][0], fp_Y_gamma[0][0], sizeof_eo_propagator_field );
      
      /* gamma_fp_Y_gamma^e = Gamma_mu^f fp_aux^o */
      apply_cvc_vertex_propagator_eo ( gamma_fp_Y_gamma, fp_aux, mu, 0, gauge_field, 0);

      conn_ = (complex*)( conn_e + (4*mu+nu)*2*Vhalf );

      /* contract S^e , gamma_fp_Y_gamma^e */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, fp_X, gamma_fp_Y_gamma, -1., Vhalf);

    }

#if 0
#endif  /* of if 0 */

    /* contribution to contact term */
    if( (source_proc_id == g_cart_id) && !source_location_iseven ) {
      /* fp_Y_gamma = T^o Gamma_nu^b */
      complex w;
      _co_eq_tr_fp ( &w, fp_Y_gamma[ g_lexic2eosub[source_location] ] );
      contact_term[2*nu  ] += w.re;
      contact_term[2*nu+1] += w.im;
    }

  }  /* end of loop on nu */

  /**********************************************************
   * (3) X = T^o, Y = S^e
   **********************************************************/  

  /* fp_X = tprop^o */
  exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_X, tprop_list_o+48, Vhalf);
  /* fp_aux^o = fp_X^o */
  memcpy(fp_aux[0][0], fp_X[0][0], sizeof_eo_propagator_field );

  /* gamma_fp_X^e = Gamma_mu^f fp_aux^o */
  for(mu=0; mu<4; mu++) {
    apply_cvc_vertex_propagator_eo ( gamma_fp_X[mu], fp_aux, mu, 0, gauge_field, 0);
  }

  /* loop on nu */
  for( nu=0; nu<4; nu++ )
  {
  

    /* fp_Y_gamma^e = sprop^e , source + nu */
    exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_Y_gamma, sprop_list_e+12*nu, Vhalf);

    /* fp_Y_gamma^e = fp_Y_gamma^e * Gamma_nu^b  */
    apply_propagator_constant_cvc_vertex ( fp_Y_gamma, fp_Y_gamma, nu, 1, Usource[nu], Vhalf );

    for( mu=0; mu<4; mu++ ) {

      conn_ = (complex*)( conn_e + (4*mu+nu)*2*Vhalf );

      /* contract S^e Gamma_nu^b, Gamma_mu^f T^o */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, fp_Y_gamma, gamma_fp_X[mu], 1., Vhalf);

      /* fp_aux^e = fp_Y_gamma^e */
      memcpy(fp_aux[0][0], fp_Y_gamma[0][0], sizeof_eo_propagator_field );
      
      /* gamma_fp_Y_gamma^o = Gamma_mu^f fp_aux^e */
      apply_cvc_vertex_propagator_eo ( gamma_fp_Y_gamma, fp_aux, mu, 0, gauge_field, 1);

      conn_ = (complex*)( conn_o + (4*mu+nu)*2*Vhalf );

      /* contract gamma_fp_Y_gamma^o, T^o */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, gamma_fp_Y_gamma, fp_X, -1., Vhalf);

    }
#if 0
#endif  /* of if 0 */

    /* contribution to contact term */
    if( (source_proc_id == g_cart_id) && source_location_iseven ) {
      /* gamma_fp_X = Gamma_nu^f T^o */
      complex w;
      _co_eq_tr_fp ( &w, gamma_fp_X[nu][ g_lexic2eosub[source_location] ] );
      contact_term[2*nu  ] -= w.re;
      contact_term[2*nu+1] -= w.im;
    }

  }  /* end of loop on nu */

  /**********************************************************
   * (4) X = S^o, Y = T^e
   **********************************************************/  

  /* fp_X = sprop^o */
  exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_X, sprop_list_o+48, Vhalf);
  /* fp_aux^o = fp_X^o */
  memcpy(fp_aux[0][0], fp_X[0][0], sizeof_eo_propagator_field );

  /* gamma_fp_X^e = Gamma_mu^f fp_aux^o */
  for(mu=0; mu<4; mu++) {
    apply_cvc_vertex_propagator_eo ( gamma_fp_X[mu], fp_aux, mu, 0, gauge_field, 0);
  }

  /* loop on nu */
  for( nu=0; nu<4; nu++ ) 
  {

    /* fp_Y_gamma^e = tprop^e , source + nu */
    exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_Y_gamma, tprop_list_e+12*nu, Vhalf);

    /* fp_Y_gamma^e = fp_Y_gamma^e * Gamma_nu^b  */
    apply_propagator_constant_cvc_vertex ( fp_Y_gamma, fp_Y_gamma, nu, 1, Usource[nu], Vhalf );

    for( mu=0; mu<4; mu++ ) {

      conn_ = (complex*)( conn_e + (4*mu+nu)*2*Vhalf );

      /* contract Gamma_mu^f S^o, T^e Gamma_nu^b */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, gamma_fp_X[mu], fp_Y_gamma, 1., Vhalf);

      /* fp_aux^e = fp_Y_gamma^e */
      memcpy(fp_aux[0][0], fp_Y_gamma[0][0], sizeof_eo_propagator_field );
      
      /* gamma_fp_Y_gamma^o = Gamma_mu^f fp_aux^e */
      apply_cvc_vertex_propagator_eo ( gamma_fp_Y_gamma, fp_aux, mu, 0, gauge_field, 1);

      conn_ = (complex*)( conn_o + (4*mu+nu)*2*Vhalf );

      /* contract S^o, gamma_fp_Y_gamma^o */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, fp_X, gamma_fp_Y_gamma, -1., Vhalf);

    }
#if 0
#endif  /* of if 0 */

    /* contribution to contact term */
    if( (source_proc_id == g_cart_id) && source_location_iseven ) {
      /* fp_Y_gamma = T^e Gamma_nu^b */
      complex w;
      _co_eq_tr_fp ( &w, fp_Y_gamma[ g_lexic2eosub[source_location] ] );
      contact_term[2*nu  ] += w.re;
      contact_term[2*nu+1] += w.im;
    }

  }  /* end of loop on nu */

  /* normalization */
#if 0
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for(ix=0; ix<32*Vhalf; ix++) conn_e[ix] *= -0.25;
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for(ix=0; ix<32*Vhalf; ix++) conn_o[ix] *= -0.25;
#endif  /* of if 0 */

  /* free auxilliary fields */
  free_fp_field(&fp_aux);
  free_fp_field(&fp_X);
  free_fp_field(&(gamma_fp_X[0]));
  free_fp_field(&(gamma_fp_X[1]));
  free_fp_field(&(gamma_fp_X[2]));
  free_fp_field(&(gamma_fp_X[3]));
  free_fp_field( &fp_Y_gamma );
  free_fp_field( &gamma_fp_Y_gamma );

  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [contract_cvc_tensor_eo] time for contract_cvc_tensor = %e seconds\n", retime-ratime);

#ifdef HAVE_MPI
  if(g_cart_id == source_proc_id) fprintf(stdout, "# [contract_cvc_tensor_eo] broadcasting contact term\n");
  MPI_Bcast(contact_term, 8, MPI_DOUBLE, source_proc_id, g_cart_grid);
  /* TEST */
  /* fprintf(stdout, "[%2d] contact term = "\
      "(%e + I %e, %e + I %e, %e + I %e, %e +I %e)\n",
      g_cart_id, contact_term[0], contact_term[1], contact_term[2], contact_term[3],
      contact_term[4], contact_term[5], contact_term[6], contact_term[7]); */
  if( source_proc_id == g_cart_id ) {
    fprintf(stdout, "# [contract_cvc_tensor_eo] contact term[0] = %25.16e + I %25.16e\n", contact_term[0], contact_term[1]);
    fprintf(stdout, "# [contract_cvc_tensor_eo] contact term[1] = %25.16e + I %25.16e\n", contact_term[2], contact_term[3]);
    fprintf(stdout, "# [contract_cvc_tensor_eo] contact term[2] = %25.16e + I %25.16e\n", contact_term[4], contact_term[5]);
    fprintf(stdout, "# [contract_cvc_tensor_eo] contact term[3] = %25.16e + I %25.16e\n", contact_term[6], contact_term[7]);
  }
#endif

#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif
  return;

}  /* end of contract_cvc_tensor_eo */

}  /* end of namespace cvc */
