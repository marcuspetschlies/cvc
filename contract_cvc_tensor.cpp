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
#include <complex.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#include "cvc_complex.h"
#include "ilinalg.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "matrix_init.h"
#include "project.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"
#include "contract_cvc_tensor.h"

namespace cvc {

static double *Usource[4], Usourcebuffer[72];
static double _Complex UsourceMatrix[4][144];
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

  /***********************************************************
   * initialize UsourceMatrix
   *
   * U = ( gamma_mu + 1 ) U_mu^+
   *
   *
   * ( a + ib ) * (i c) = -bc + i ac
   * re <- -im * c
   * im <-  re * c
   *  
   * imag: re  ->  s * im
   *       im      s * re
   ***********************************************************/
  for(int mu=0; mu<4; mu++ ) {
    int isimag = gamma_permutation[mu][0] % 2;
    int k[4] = { gamma_permutation[mu][6*0] / 6, gamma_permutation[mu][6*1] / 6, gamma_permutation[mu][6*2] / 6, gamma_permutation[mu][6*3] / 6};
    int s[4] = { gamma_sign[mu][6*0], gamma_sign[mu][6*1], gamma_sign[mu][6*2], gamma_sign[mu][6*3] };

    for(int i=0; i<4; i++) {
      /* spin diagonal part */
      for(int a=0; a<3; a++) {
      for(int b=0; b<3; b++) {
        int index1 = 3 * i + a;
        int index2 = 3 * i + b;
          /* (i,a), (i,b) = U_mu(a,b)^* * 1 */
          UsourceMatrix[mu][  12*index1 + index2] = ( Usource[mu][2*(3*a+b)] - Usource[mu][2*(3*a+b)+1] * I ); 
      }}

      /* spin off-diagonal part */

      for(int a=0; a<3; a++) {
      for(int b=0; b<3; b++) {
        int index1 = 3 * i + a;
        int index2 = 3 * k[i] + b;
        UsourceMatrix[mu][  12*index1 + index2] = ( isimag ? -s[i]*I : s[i] ) * ( Usource[mu][2*(3*a+b)] - Usource[mu][2*(3*a+b)+1] * I ); 
      }} 
    }
  }  /* end of loop on mu */


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

/***************************************************************************
 * subtract contact term
 *
 * - only by process that has source location
 ***************************************************************************/
void cvc_tensor_eo_subtract_contact_term (double**tensor_eo, double*contact_term, int gsx[4], int have_source ) {

  const unsigned int Vhalf = VOLUME / 2;

  if( have_source ) {
    int mu, sx[4] = { gsx[0] % T, gsx[1] % LX, gsx[2] % LY, gsx[3] % LZ };
    unsigned int ix   = g_ipt[sx[0]][sx[1]][sx[2]][sx[3]];
    unsigned int ixeo = g_lexic2eosub[ix];

    if( g_verbose > 0 ) fprintf(stdout, "# [cvc_tensor_eo_subtract_contact_term] process %d subtracting contact term\n", g_cart_id);
    if ( g_iseven[ix] ) {
      for(mu=0; mu<4; mu++) {
        tensor_eo[0][_GWI(5*mu,ixeo,Vhalf)    ] -= contact_term[2*mu  ];
        tensor_eo[0][_GWI(5*mu,ixeo,Vhalf) + 1] -= contact_term[2*mu+1];
      }
    } else {
      for(mu=0; mu<4; mu++) {
        tensor_eo[1][_GWI(5*mu,ixeo,Vhalf)    ] -= contact_term[2*mu  ];
        tensor_eo[1][_GWI(5*mu,ixeo,Vhalf) + 1] -= contact_term[2*mu+1];
      }
    }
  }  /* end of if have source */
}  /* end of cvc_tensor_eo_subtract_contact_term */


/***************************************************************************
 * momentum projections
 ***************************************************************************/

int cvc_tensor_eo_momentum_projection (double****tensor_tp, double**tensor_eo, int (*momentum_list)[3], int momentum_number) {

  const unsigned int Vhalf = VOLUME / 2;
  int exitstatus, mu;
  double ***cvc_tp = NULL, *cvc_tensor_lexic=NULL;
  double ratime, retime;

  ratime = _GET_TIME;

  if ( *tensor_tp == NULL ) {
    exitstatus = init_3level_buffer(tensor_tp, momentum_number, 16, 2*T);
    if(exitstatus != 0) {
      fprintf(stderr, "[cvc_tensor_eo_momentum_projection] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(1);
    }
  }
  cvc_tp = *tensor_tp;

  cvc_tensor_lexic = (double*)malloc(32*VOLUME*sizeof(double));
  if( cvc_tensor_lexic == NULL ) {
    fprintf(stderr, "[cvc_tensor_eo_momentum_projection] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(2);
  }
  for ( mu=0; mu<16; mu++ ) {
    complex_field_eo2lexic (cvc_tensor_lexic+2*mu*VOLUME, tensor_eo[0]+2*mu*Vhalf, tensor_eo[1]+2*mu*Vhalf );
  }

  exitstatus = momentum_projection (cvc_tensor_lexic, cvc_tp[0][0], T*16, momentum_number, momentum_list);
  if(exitstatus != 0) {
    fprintf(stderr, "[cvc_tensor_eo_momentum_projection] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }
  retime = _GET_TIME;
  if( g_cart_id == 0 ) fprintf(stdout, "# [cvc_tensor_eo_momentum_projection] time for momentum projection = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);

  return(0);
}  /* end of cvc_tensor_eo_momentum_projection */

/***************************************************************************
 * write tp-tensor results to file
 ***************************************************************************/

int cvc_tensor_tp_write_to_aff_file (double***cvc_tp, struct AffWriter_s*affw, char*tag, int (*momentum_list)[3], int momentum_number, int io_proc ) {

  int exitstatus, i;
  double ratime, retime;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char aff_buffer_path[200];
  double *buffer = NULL;
  double _Complex *aff_buffer = NULL;
  double _Complex *zbuffer = NULL;

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[cvc_tensor_tp_write_to_aff_file] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }

    zbuffer = (double _Complex*)malloc(  momentum_number * 16 * T_global * sizeof(double _Complex) );
    if( zbuffer == NULL ) {
      fprintf(stderr, "[cvc_tensor_tp_write_to_aff_file] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(6);
    }
  }

  ratime = _GET_TIME;

  /* reorder cvc_tp into buffer with order time - munu - momentum */
  buffer = (double*)malloc(  momentum_number * 32 * T * sizeof(double) );
  if( buffer == NULL ) {
    fprintf(stderr, "[cvc_tensor_tp_write_to_aff_file] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
  i = 0;
  for( int it = 0; it < T; it++ ) {
    for( int mu=0; mu<16; mu++ ) {
      for( int ip=0; ip<momentum_number; ip++) {
        buffer[i++] = cvc_tp[ip][mu][2*it  ];
        buffer[i++] = cvc_tp[ip][mu][2*it+1];
      }
    }
  }

#ifdef HAVE_MPI
  i = momentum_number * 32 * T;
#  if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
  if(io_proc>0) {
    exitstatus = MPI_Gather(buffer, i, MPI_DOUBLE, zbuffer, i, MPI_DOUBLE, 0, g_tr_comm);
    if(exitstatus != MPI_SUCCESS) {
      fprintf(stderr, "[cvc_tensor_tp_write_to_aff_file] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(3);
    }
  }
#  else
  exitstatus = MPI_Gather(buffer, i, MPI_DOUBLE, zbuffer, i, MPI_DOUBLE, 0, g_cart_grid);
  if(exitstatus != MPI_SUCCESS) {
    fprintf(stderr, "[cvc_tensor_tp_write_to_aff_file] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }
#  endif

#else
  memcpy(zbuffer, buffer, momentum_number * 32 * T * sizeof(double) );
#endif
  free( buffer );

  if(io_proc == 2) {

    /* reverse the ordering back to momentum - munu - time */
    aff_buffer = (double _Complex*)malloc( momentum_number * 16 * T_global * sizeof(double _Complex) );
    if(aff_buffer == NULL) {
      fprintf(stderr, "[cvc_tensor_tp_write_to_aff_file] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(2);
    }
    i = 0;
    for( int ip=0; ip<momentum_number; ip++) {
      for( int mu=0; mu<16; mu++ ) {
        for( int it = 0; it < T_global; it++ ) {
          int offset = (it * 16 + mu ) * momentum_number + ip;
          aff_buffer[i++] = zbuffer[offset];
        }
      }
    }
    free( zbuffer );

    for(i=0; i < momentum_number; i++) {
      sprintf(aff_buffer_path, "%s/px%.2dpy%.2dpz%.2d", tag, momentum_list[i][0], momentum_list[i][1], momentum_list[i][2] );
      /* fprintf(stdout, "# [cvc_tensor_tp_write_to_aff_file] current aff path = %s\n", aff_buffer_path); */
      affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
      exitstatus = aff_node_put_complex (affw, affdir, aff_buffer+16*T_global*i, (uint32_t)T_global*16);
      if(exitstatus != 0) {
        fprintf(stderr, "[cvc_tensor_tp_write_to_aff_file] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(5);
      }
    }
    free( aff_buffer );
  }  /* if io_proc == 2 */

#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif

  retime = _GET_TIME;
  if(io_proc == 2) fprintf(stdout, "# [cvc_tensor_tp_write_to_aff_file] time for saving momentum space results = %e seconds\n", retime-ratime);

  return(0);

}  /* end of cvc_tensor_tp_write_to_aff_file */


/********************************************
 * check Ward-identity in position space
 *
 *   starting from eo-precon tensor
 ********************************************/

int cvc_tensor_eo_check_wi_position_space (double **tensor_eo) {

  const unsigned int Vhalf = VOLUME / 2;

  int nu;
  int exitstatus;
  double ratime, retime;

  /********************************************
   * check the Ward identity in position space 
   ********************************************/
  ratime = _GET_TIME;

  const unsigned int VOLUMEplusRAND = VOLUME + RAND;
  const unsigned int stride = VOLUMEplusRAND;
  double *conn_buffer = (double*)malloc(32*VOLUMEplusRAND*sizeof(double));
  if(conn_buffer == NULL)  {
    fprintf(stderr, "# [cvc_tensor_eo_check_wi_position_space] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  for(nu=0; nu<16; nu++) {
    complex_field_eo2lexic (conn_buffer+2*nu*VOLUMEplusRAND, tensor_eo[0]+2*nu*Vhalf, tensor_eo[1]+2*nu*Vhalf );
#ifdef HAVE_MPI
    xchange_contraction( conn_buffer+2*nu*VOLUMEplusRAND, 2 );
#endif
  }

  if(g_cart_id == 0 && g_verbose > 1) fprintf(stdout, "# [cvc_tensor_eo_check_wi_position_space] checking Ward identity in position space\n");
  for(nu=0; nu<4; nu++) {
    double norm = 0.;
    complex w;
    unsigned int ix;
    for(ix=0; ix<VOLUME; ix++ ) {
      w.re = conn_buffer[_GWI(4*0+nu,ix          ,stride)  ] + conn_buffer[_GWI(4*1+nu,ix          ,stride)  ]
           + conn_buffer[_GWI(4*2+nu,ix          ,stride)  ] + conn_buffer[_GWI(4*3+nu,ix          ,stride)  ]
           - conn_buffer[_GWI(4*0+nu,g_idn[ix][0],stride)  ] - conn_buffer[_GWI(4*1+nu,g_idn[ix][1],stride)  ]
           - conn_buffer[_GWI(4*2+nu,g_idn[ix][2],stride)  ] - conn_buffer[_GWI(4*3+nu,g_idn[ix][3],stride)  ];

      w.im = conn_buffer[_GWI(4*0+nu,ix          ,stride)+1] + conn_buffer[_GWI(4*1+nu,ix          ,stride)+1]
           + conn_buffer[_GWI(4*2+nu,ix          ,stride)+1] + conn_buffer[_GWI(4*3+nu,ix          ,stride)+1]
           - conn_buffer[_GWI(4*0+nu,g_idn[ix][0],stride)+1] - conn_buffer[_GWI(4*1+nu,g_idn[ix][1],stride)+1]
           - conn_buffer[_GWI(4*2+nu,g_idn[ix][2],stride)+1] - conn_buffer[_GWI(4*3+nu,g_idn[ix][3],stride)+1];
      
      norm += w.re*w.re + w.im*w.im;
    }
#ifdef HAVE_MPI
    double dtmp = norm;
    exitstatus = MPI_Allreduce(&dtmp, &norm, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid);
    if(exitstatus != MPI_SUCCESS) {
      fprintf(stderr, "[cvc_tensor_eo_check_wi_position_space] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(2);
    }
#endif
    if(g_cart_id == 0) fprintf(stdout, "# [cvc_tensor_eo_check_wi_position_space] WI nu = %2d norm = %25.16e %s %d\n", nu, sqrt(norm), __FILE__, __LINE__);
  }  /* end of loop on nu */

  free(conn_buffer);

  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [cvc_tensor_eo_check_wi_position_space] time for checking position space WI = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);

  return(0);
}  /* end of cvc_tensor_eo_check_wi_position_space */


/************************************************************
 * calculate gsp using t-blocks
 *
          subroutine zgemm  (   character   TRANSA,
          V^+ Gamma(p) W
          eo - scalar product over even 0 / odd 1 sites

          V is numV x (12 VOL3half) (C) = (12 VOL3half) x numV (F)

          W is numW x (12 VOL3half) (C) = (12 VOL3half) x numW (F)

          zgemm calculates
          V^H x [ (Gamma(p) x W) ] which is numV x numW (F) = numW x numV (C)

    complex*16 function zdotc   (   integer   N, complex*16, dimension(*)    ZX, integer   INCX, complex*16, dimension(*)    ZY, integer   INCY )   

 *
 ************************************************************/
int contract_vdag_gloc_spinor_field (double *contr, double**prop_list_e, double**prop_list_o,  double**V, int numV, int numW, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, struct AffWriter_s*affw, char*tag ) {
 
  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double);


  int exitstatus, iproc, gamma_id;
  int x0, ievecs, k;
  int i_momentum, i_gamma_id, momentum[3];
  int io_proc = 2;
  unsigned int ix;
  size_t items, offset, bytes;

  double ratime, retime, momentum_ratime, momentum_retime, gamma_ratime, gamma_retime;
  double _Complex **phase = NULL;
  double _Complex *V_buffer = NULL, *W_buffer = NULL, *Z_buffer = NULL;
  double _Complex *zptr=NULL, ztmp;
  double _Complex ***spinor_aux = NULL;
#ifdef HAVE_MPI
  double _Complex *mZ_buffer = NULL;
#endif

  /*variables for blas interface */
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;

  char BLAS_TRANSA, BLAS_TRANSB;
  int BLAS_M, BLAS_N, BLAS_K;
  double _Complex *BLAS_A=NULL, *BLAS_B=NULL, *BLAS_C=NULL;
  int BLAS_LDA, BLAS_LDB, BLAS_LDC;

#ifdef HAVE_MPI
  MPI_Status mstatus;
  int mcoords[4], mrank;
#endif

  /***********************************************
   * even/odd phase field for Fourier phase
   ***********************************************/
  exitstatus = init_2level_buffer( (double***)(&phase), momentum_number, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[] Error from init_2level_buffer %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  exitstatus = init_2level_buffer( (double***)(&Vts), numV, 24*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[] Error from init_2level_buffer %s %d\n", __FILE__, __LINE__ );
    return(3);
  }

  exitstatus = init_2level_buffer( (double***)(&Wts), numW, 24*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[] Error from init_2level_buffer %s %d\n", __FILE__, __LINE__ );
    return(3);
  }


  /* timeslice-wise */
  for ( it = 0; it < T; it++  ) {
    /*copy timeslices of evecs */
    for(i = 0; i< numV, i++ ) {
      memcpy(Vts[i], V[i]+it*_GSI(VOL3half), sizeof_eo_spinor_field_timeslice );
    }

    /* eo phase field for all momenta */
    make_eo_phase_field_timeslice (phase, momentum_number, momentum_list, it, 1);

    /* buffer for results */
    double _Complex *Z_buffer = malloc(nev * 12 * gamma_id_number * 4)

    /* loop on shifts of source location */
    for(mu = 0; mu < 5; mu++ ) {
    
      for(i = 0; i< 12, i++ ) {
        memcpy(Wts[i], prop_o[i]+it*_GSI(VOL3half), sizeof_eo_spinor_field_timeslice );
      }
    
      /***********************************************
       * loop on momenta
       ***********************************************/
      for(i_momentum=0; i_momentum < momentum_number; i_momentum++) {

        momentum[0] = momentum_list[i_momentum][0];
        momentum[1] = momentum_list[i_momentum][1];
        momentum[2] = momentum_list[i_momentum][2];

      if(g_cart_id == 0 && g_verbose > 1) {
        fprintf(stdout, "# [] using source momentum = (%d, %d, %d)\n", momentum[0], momentum[1], momentum[2]);
      }

      /***********************************************
       * loop on gamma id's
       ***********************************************/
      for(i_gamma_id=0; i_gamma_id < gamma_id_number; i_gamma_id++) {

        gamma_id = gamma_id_list[i_gamma_id];
        if(g_cart_id == 0 && g_verbose > 1) fprintf(stdout, "# [] using source gamma id %d\n", gamma_id);

        /* multiply with gamma and phase field */
   
        for(i=0; i<12; i++) {
          spinor_field_eq_gamma_co_ti_spinor_field( Wts[i], gamma_id, phase[i_momentum], Wts[i], VOL3half );
        }
 
        /* scalar products */

        
        /***********************************************
         * Vts is nev x 12Vhalf (C) = 12Vhalf x nev (F)
         * Wts is  12 x 12Vhalf (C) = 12Vhalf x  12 (F)
         *
         * Vts^H Wts is nev x 12 (F) = 12 x nev (C)
         ***********************************************/
        BLAS_TRANSA = "C"
        BLAS_TRANSB = "N"
        BLAS_M = nev;
        BLAS_N = 12; 
        BLAS_K = 12*VOL3half;
        BLAS_ALPHA = 1.;
        BLAS_BETA  = 0.;
        BLAS_A = Vts;
        BLAS_B = Wts;
        BLAS_C = Z_buffer;
        BLAS_LDA = BLAS_K;
        BLAS_LDB = BLAS_K;
        BLAS_LDC = BLAS_M;
        _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC, 1, 1);


      /* buffer for input matrices */
      if( (V_buffer = (double _Complex*)malloc(V_buffer_bytes) ) == NULL ) {
       fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] Error from malloc\n");
       return(5);
      }
      
      if( (W_buffer = (double _Complex*)malloc(W_buffer_bytes)) == NULL ) {
       fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] Error from malloc\n");
       return(6);
      }

      BLAS_A = V_buffer;
      BLAS_B = W_buffer;

      /***********************************************
       * loop on timeslices
       ***********************************************/
      for(x0 = 0; x0 < T; x0++) {

        /* copy to timeslice of V to V_buffer */
#ifdef HAVE_OPENMP
#pragma omp parallel shared(x0)
{
#endif
        bytes  = VOL3half * sizeof_spinor_point;
        offset = 12 * VOL3half;
        items  = 12 * Vhalf;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
        for(ievecs=0; ievecs<numV; ievecs++) {
          memcpy(V_buffer+ievecs*offset, V_ptr+(ievecs*items + x0*offset), bytes );
        }

#ifdef HAVE_OPENMP
#pragma omp for
#endif
        for(ievecs=0; ievecs<numW; ievecs++) {
          memcpy(W_buffer+ievecs*offset, W_ptr+(ievecs*items + x0*offset), bytes );
        }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
        /***********************************************
         * apply Gamma(pvec) to W
         ***********************************************/
        ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel private(zptr,ztmp)
{
#endif
       
        double spinor2[24];
        zptr = W_buffer;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
        for(ix=0; ix<numW*VOL3half; ix++) {
          /* W <- gamma exp ip W */
          ztmp = phase[x0][ix % VOL3half];
          _fv_eq_gamma_ti_fv(spinor2, gamma_id, (double*)zptr);
          zptr[ 0] = ztmp * (spinor2[ 0] + spinor2[ 1] * I);
          zptr[ 1] = ztmp * (spinor2[ 2] + spinor2[ 3] * I);
          zptr[ 2] = ztmp * (spinor2[ 4] + spinor2[ 5] * I);
          zptr[ 3] = ztmp * (spinor2[ 6] + spinor2[ 7] * I);
          zptr[ 4] = ztmp * (spinor2[ 8] + spinor2[ 9] * I);
          zptr[ 5] = ztmp * (spinor2[10] + spinor2[11] * I);
          zptr[ 6] = ztmp * (spinor2[12] + spinor2[13] * I);
          zptr[ 7] = ztmp * (spinor2[14] + spinor2[15] * I);
          zptr[ 8] = ztmp * (spinor2[16] + spinor2[17] * I);
          zptr[ 9] = ztmp * (spinor2[18] + spinor2[19] * I);
          zptr[10] = ztmp * (spinor2[20] + spinor2[21] * I);
          zptr[11] = ztmp * (spinor2[22] + spinor2[23] * I);
          zptr += 12;
        }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
        retime = _GET_TIME;
        if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] time for conj gamma p W = %e seconds\n", retime - ratime);

        /***********************************************
         * scalar products as matrix multiplication
         ***********************************************/
        ratime = _GET_TIME;

        /* output buffer */
        BLAS_C = Z_buffer + x0 * numV*numW;


        retime = _GET_TIME;
        if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] time for zgemm = %e seconds\n", retime - ratime);

      }  /* end of loop on timeslice x0 */
      free(V_buffer); V_buffer = NULL;
      free(W_buffer); W_buffer = NULL;

#ifdef HAVE_MPI
      ratime = _GET_TIME;
      /* reduce within global timeslice */
      items = T * numV * numW;
      bytes = items * sizeof(double _Complex);
      if( (mZ_buffer = (double _Complex*)malloc( bytes )) == NULL ) {
        fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] Error, could not open file %s for writing\n", filename);
        return(5);
      }
      memcpy(mZ_buffer, Z_buffer, bytes);
      MPI_Allreduce(mZ_buffer, Z_buffer, 2*T*numV*numW, MPI_DOUBLE, MPI_SUM, g_ts_comm);
      free(mZ_buffer); mZ_buffer = NULL;
      retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] time for allreduce = %e seconds\n", retime - ratime);
#endif


      /***********************************************
       * write gsp to disk
       ***********************************************/
      if(tag != NULL) {
        if(io_proc == 2) {
          sprintf(filename, "%s.px%.2dpy%.2dpz%.2d.g%.2d", tag, momentum[0], momentum[1], momentum[2], gamma_id);
          ofs = fopen(filename, "w");
          if(ofs == NULL) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] Error, could not open file %s for writing\n", filename);
            return(12);
          }
        }
#ifdef HAVE_MPI
        items = T * numV * numW;
        bytes = items * sizeof(double _Complex);
        if(io_proc == 2) {
          if( (mZ_buffer = (double _Complex*)malloc( bytes )) == NULL ) {
            fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] Error, could not open file %s for writing\n", filename);
            return(7);
          }
        }
#endif

        for(iproc=0; iproc < g_nproc_t; iproc++) {
#ifdef HAVE_MPI
          ratime = _GET_TIME;
          if(iproc > 0) {
            /***********************************************
             * gather at root
             ***********************************************/
            k = 2*T*numV*numW; /* number of items to be sent and received */
            if(io_proc == 2) {
              mcoords[0] = iproc; mcoords[1] = 0; mcoords[2] = 0; mcoords[3] = 0;
              MPI_Cart_rank(g_cart_grid, mcoords, &mrank);

              /* receive gsp with tag iproc; overwrite Z_buffer */
              status = MPI_Recv(mZ_buffer, k, MPI_DOUBLE, mrank, iproc, g_cart_grid, &mstatus);
              if(status != MPI_SUCCESS ) {
                fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] proc%.4d Error from MPI_Recv, status was %d\n", g_cart_id, status);
                return(9);
              }
            } else if (g_proc_coords[0] == iproc && io_proc == 1) {
              /* send correlator with tag 2*iproc */
              status = MPI_Send(Z_buffer, k, MPI_DOUBLE, mrank, iproc, g_cart_grid);
              if(status != MPI_SUCCESS ) {
                fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] proc%.4d Error from MPI_Recv, status was %d\n", g_cart_id, status);
                return(10);
              }
            }
          }  /* end of if iproc > 0 */

          retime = _GET_TIME;
          if(io_proc == 2) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] time for exchange = %e seconds\n", retime-ratime);
#endif  /* of ifdef HAVE_MPI */

          /***********************************************
           * I/O process write to file
           ***********************************************/
          if(io_proc == 2) {
            ratime = _GET_TIME;
#ifdef HAVE_MPI
            zptr = iproc == 0 ? Z_buffer : mZ_buffer;
#else
            zptr = Z_buffer;
#endif
            if( fwrite(zptr, sizeof(double _Complex), items, ofs) != items ) {
              fprintf(stderr, "[gsp_calculate_v_dag_gamma_p_w_block_asym] Error, could not write proper amount of data to file %s\n", filename);
              return(13);
            }

            retime = _GET_TIME;
            fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] time for writing = %e seconds\n", retime-ratime);
          }  /* end of if io_proc == 2 */

        }  /* end of loop on iproc */
        if(io_proc == 2) {
#ifdef HAVE_MPI
          free(mZ_buffer);
#endif
          fclose(ofs);
        }
      }  /* end of if tag != NULL */

      gamma_retime = _GET_TIME;
      if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] time for gamma id %d = %e seconds\n", gamma_id_list[i_gamma_id], gamma_retime - gamma_ratime);

   }  /* end of loop on gamma id */

   momentum_retime = _GET_TIME;
   if(g_cart_id == 0) fprintf(stdout, "# [gsp_calculate_v_dag_gamma_p_w_block_asym] time for momentum (%d, %d, %d) = %e seconds\n",
       momentum[0], momentum[1], momentum[2], momentum_retime - momentum_ratime);

  }   /* end of loop on source momenta */

  fini_2level_buffer( &phase );
  return(0);

}  /* end of contract_v_dag_gloc_spinor_field */

}  /* end of namespace cvc */
