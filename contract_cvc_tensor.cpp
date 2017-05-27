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
#include "iblas.h"
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
 * always backward, since the shift is in +mu direction
 ************************************************************/
int apply_constant_cvc_vertex_at_source (double**s, int mu, int fbwd, const unsigned int N ) {

  int exitstatus;

  /* allocate a fermion propagator field */
  fermion_propagator_type *fp1 = create_fp_field( N );

  /* assign from spinor fields s */
  exitstatus = assign_fermion_propagaptor_from_spinor_field ( fp1, s, N );
  if(exitstatus != 0) {
    fprintf(stderr, "[apply_constant_cvc_vertex_at_source] Error from assign_fermion_propagaptor_from_spinor_field, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /* apply the vertex */
  apply_propagator_constant_cvc_vertex ( fp1, fp1, mu, fbwd, Usource[mu], N );

  /* restore the propagator to spinor fields */
  exitstatus = assign_spinor_field_from_fermion_propagaptor (s, fp1, N);
  if(exitstatus != 0) {
    fprintf(stderr, "[apply_constant_cvc_vertex_at_source] Error from assign_spinor_field_from_fermion_propagaptor, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  free_fp_field( &fp1 );
  return(0);
}  /* end of apply_constant_cvc_vertex_at_source */

/************************************************************
 * calculate gsp using t-blocks
 *
          subroutine zgemm  (   character   TRANSA,
          V^+ Gamma(p) S
          eo - scalar product over even 0 / odd 1 sites

          V is numV x (12 VOL3half) (C) = (12 VOL3half) x numV (F)

          prop is nsf x (12 VOL3half) (C) = (12 VOL3half) x nsf (F)

          zgemm calculates
          V^H x [ (Gamma(p) x prop) ] which is numV x nsf (F) = nsf x numV (C)
 *
 ************************************************************/
int contract_vdag_gloc_spinor_field (
    double**prop_list_e, 
    double**prop_list_o, 
    int nsf,
    double**V, 
    int numV, 
    int momentum_number, 
    int (*momentum_list)[3], 
    int gamma_id_number, 
    int*gamma_id_list, 
    struct AffWriter_s*affw, 
    char*tag, int io_proc, 
    double*gauge_field, 
    double **mzz[2], 
    double**mzzinv[2] ) {
 
  const unsigned int Vhalf    = VOLUME / 2;
  const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  const size_t sizeof_eo_spinor_field           = _GSI(  Vhalf )          * sizeof(double);
  const size_t sizeof_eo_spinor_field_with_halo = _GSI( (VOLUME+RAND)/2 ) * sizeof(double);
  const size_t sizeof_eo_spinor_field_timeslice = _GSI(  VOL3half )       * sizeof(double);

  int exitstatus;
  double *spinor_work = NULL;
  double *spinor_aux  = NULL;
  double _Complex **phase_field = NULL;
  double _Complex **W           = NULL;
  double _Complex **V_ts        = NULL;
  double _Complex **prop_ts     = NULL;
  double _Complex **prop_phase  = NULL;
  double _Complex ***contr      = NULL;
  double _Complex *contr_allt_buffer = NULL;

  double *mcontr_buffer = NULL;
  double ratime, retime;

  struct AffNode_s *affn = NULL, *affdir=NULL;
  char aff_path[400];

  /* BLAS parameters for zgemm */
  char BLAS_TRANSA = 'C';
  char BLAS_TRANSB = 'N';
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  int BLAS_M = numV;
  int BLAS_K = 12*VOL3half;
  int BLAS_N = nsf;
  int BLAS_LDA = BLAS_K;
  int BLAS_LDB = BLAS_K;
  int BLAS_LDC = BLAS_M;

  ratime = _GET_TIME;

  exitstatus = init_2level_buffer ( (double***)(&phase_field), T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_buffer ( (double***)(&V_ts), numV, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  exitstatus = init_2level_buffer ( (double***)(&prop_ts), nsf, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&prop_phase), nsf, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_3level_buffer ( (double****)(&contr), T, nsf, 2*numV );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(5);
  }

#ifdef HAVE_MPI
  mcontr_buffer = (double*)malloc(numV*nsf*2*sizeof(double) ) ;
  if ( mcontr_buffer == NULL ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
#endif

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }

  /************************************************
   ************************************************
   **
   ** V, odd part
   **
   ************************************************
   ************************************************/

  /* loop on momenta */
  for( int im=0; im<momentum_number; im++ ) {
  
    /* make odd phase field */
    make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

    /* calculate the propagators including current Fourier phase */
    for( int i=0; i<nsf; i++) {
      spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_phase[i]), prop_list_o[i], (double*)(phase_field[0]), Vhalf);
    }

    for( int ig=0; ig<gamma_id_number; ig++ ) {

      for( int it=0; it<T; it++ ) {

        /* copy timslice of V  */
        unsigned int offset = _GSI(VOL3half) * it;
        for( int i=0; i<numV; i++ ) memcpy( V_ts[i], V[i]+offset, sizeof_eo_spinor_field_timeslice );

        /* calculate Gamma times spinor field timeslice 
         *
         *  prop_ts = g5 Gamma prop_list_o [it]
         *
         */
        for( int i=0; i<nsf; i++) {
          spinor_field_eq_gamma_ti_spinor_field( (double*)(prop_ts[i]), gamma_id_list[ig], (double*)(prop_phase[i])+offset, VOL3half);
        }
        g5_phi( (double*)(prop_ts[0]), nsf*VOL3half);

        /* matrix multiplication
         *
         * contr[it][i][k] = V_i^+ prop_ts_k
         *
         */

        BLAS_A = V_ts[0];
        BLAS_B = prop_ts[0];
        BLAS_C = contr[it][0];

         _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
         memcpy( mcontr_buffer,  contr[it][0], numV*nsf*sizeof(double _Complex) );
         exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*nsf, MPI_DOUBLE, MPI_SUM, g_ts_comm);
         if( exitstatus != MPI_SUCCESS ) {
           fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
           return(8);
         }
#endif
      }  /* end of loop on timeslices */

      /************************************************
       * write to file
       ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
      if ( io_proc == 2 ) {
        contr_allt_buffer = (double _Complex *)malloc(numV*nsf*T_global*sizeof(double _Complex) );
        if(contr_allt_buffer == NULL ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
          return(9);
        }
      }

      /* gather to root, which must be io_proc = 2 */
      if ( io_proc > 0 ) {
        int count = numV*nsf*2*T;
        exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(10);
         }
      }
#else
      if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
      return(1);
#endif
#else
      contr_allt_buffer = contr[0][0];
#endif
      if ( io_proc == 2 ) {
        sprintf(aff_path, "%s/v_dag_gloc_s/px%.2dpy%.2dpz%.2d/g%.2d", tag, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        affdir = aff_writer_mkpath(affw, affn, aff_path);
        exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*nsf ) );
        if(exitstatus != 0) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(11);
        }

#ifdef HAVE_MPI
        free( contr_allt_buffer );
#endif
      }
    }  /* end of loop on Gamma structures */
  }  /* end of loop on momenta */

  if(g_cart_id == 0) fprintf(stdout, "# [contract_vdag_gloc_spinor_field] end of V part\n");
  fflush(stdout);
#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif
#if 0
#endif  /* of if 0 */


  /************************************************
   ************************************************
   **
   ** Xbar V, even part
   **
   ************************************************
   ************************************************/

  /************************************************
   * allocate auxilliary W
   ************************************************/
  exitstatus = init_2level_buffer ( (double***)(&W), numV, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  /* calculate Xbar V */
  for( int i=0; i<numV; i++ ) {
    /*
     * W_e = Xbar V_o = -M_ee^-1[dn] M_eo V_o
     */
    memcpy( spinor_work, (double*)(V[i]), sizeof_eo_spinor_field );
    X_clover_eo ( (double*)(W[i]), spinor_work, gauge_field, mzzinv[1][0] );
  }
  free ( spinor_work ); spinor_work = NULL;

  /* loop on momenta */
  for( int im=0; im<momentum_number; im++ ) {
  
    /* make even phase field */
    make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

    /* calculate the propagators including current Fourier phase */
    for( int i=0; i<nsf; i++) {
      spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_phase[i]), prop_list_e[i], (double*)(phase_field[0]), Vhalf);
    }

    for( int ig=0; ig<gamma_id_number; ig++ ) {

      for( int it=0; it<T; it++ ) {

        /* copy timslice of V  */
        unsigned int offset = _GSI(VOL3half) * it;
        for( int i=0; i<numV; i++ ) memcpy( V_ts[i], (double*)(W[i])+offset, sizeof_eo_spinor_field_timeslice );

        /* calculate Gamma times spinor field timeslice 
         *
         *  prop_ts = g5 Gamma prop_list_o [it]
         *
         */
        for( int i=0; i<nsf; i++) {
          spinor_field_eq_gamma_ti_spinor_field( (double*)(prop_ts[i]), gamma_id_list[ig], (double*)(prop_phase[i])+offset, VOL3half);
        }
        g5_phi( (double*)(prop_ts[0]), nsf*VOL3half);

        /* matrix multiplication
         *
         * contr[it][i][k] = V_i^+ prop_ts_k
         *
         */

        BLAS_A = V_ts[0];
        BLAS_B = prop_ts[0];
        BLAS_C = contr[it][0];

         _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
         memcpy( mcontr_buffer,  contr[it][0], numV*nsf*sizeof(double _Complex) );
         exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*nsf, MPI_DOUBLE, MPI_SUM, g_ts_comm);
         if( exitstatus != MPI_SUCCESS ) {
           fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
           return(8);
         }
#endif

      }  /* end of loop on timeslices */

      /************************************************
       * write to file
       ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
      if ( io_proc == 2 ) {
        contr_allt_buffer = (double _Complex *)malloc(numV*nsf*T_global*sizeof(double _Complex) );
        if(contr_allt_buffer == NULL ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
          return(9);
        }
      }

      /* gather to root, which must be io_proc = 2 */
      if ( io_proc > 0 ) {
        int count = numV*nsf*2*T;
        exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(10);
         }
      }
#else
      if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
      return(1);
#endif
#else
     contr_allt_buffer = contr[0][0];
#endif
      if ( io_proc == 2 ) {
        sprintf(aff_path, "%s/xv_dag_gloc_s/px%.2dpy%.2dpz%.2d/g%.2d", tag, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        affdir = aff_writer_mkpath(affw, affn, aff_path);
        exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*nsf ) );
        if(exitstatus != 0) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(11);
        }
#ifdef HAVE_MPI
        free( contr_allt_buffer );
#endif
      }

    }  /* end of loop on Gamma structures */


  }  /* end of loop on momenta */


  if(g_cart_id == 0) fprintf(stdout, "# [contract_vdag_gloc_spinor_field] end of Xbar V part\n");
  fflush(stdout);
#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif
#if 0
#endif  /* of if 0 */

  /************************************************
   ************************************************
   **
   ** W, odd part
   **
   ************************************************
   ************************************************/
  /* calculate W from V and Xbar V */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  spinor_aux  = (double*)malloc( sizeof_eo_spinor_field );
  if ( spinor_work == NULL || spinor_aux == NULL ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  for( int i=0; i<numV; i++ ) {
    /*
     * W_o = Cbar V
     */
    memcpy( spinor_work, (double*)(W[i]), sizeof_eo_spinor_field );
    memcpy( (double*)(W[i]), (double*)(V[i]), sizeof_eo_spinor_field );
    C_clover_from_Xeo ( (double*)(W[i]), spinor_work, spinor_aux, gauge_field, mzz[1][1]);
                  
    // memcpy( spinor_aux, (double*)(V[i]), sizeof_eo_spinor_field );
    // C_clover_oo ( (double*)(W[i]), spinor_aux, gauge_field, spinor_work, mzz[1][1], mzzinv[1][0]);

  }
  free ( spinor_work ); spinor_work = NULL;
  free ( spinor_aux  ); spinor_aux = NULL;


  /* loop on momenta */
  for( int im=0; im<momentum_number; im++ ) {
  
    /* make odd phase field */
    make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

    /* calculate the propagators including current Fourier phase */
    for( int i=0; i<nsf; i++) {
      spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_phase[i]), prop_list_o[i], (double*)(phase_field[0]), Vhalf);
    }

    for( int ig=0; ig<gamma_id_number; ig++ ) {

      for( int it=0; it<T; it++ ) {

        /* copy timslice of V  */
        unsigned int offset = _GSI(VOL3half) * it;
        for( int i=0; i<numV; i++ ) memcpy( (double*)(V_ts[i]), (double*)(W[i])+offset, sizeof_eo_spinor_field_timeslice );

        /* calculate Gamma times spinor field timeslice 
         *
         *  prop_ts = g5 Gamma prop_list_o [it]
         *
         */
        for( int i=0; i<nsf; i++) {
          spinor_field_eq_gamma_ti_spinor_field( (double*)(prop_ts[i]), gamma_id_list[ig], (double*)(prop_phase[i])+offset, VOL3half);
        }
        g5_phi( (double*)(prop_ts[0]), nsf*VOL3half);

        /* matrix multiplication
         *
         * contr[it][i][k] = V_i^+ prop_ts_k
         *
         */

        BLAS_A = V_ts[0];
        BLAS_B = prop_ts[0];
        BLAS_C = contr[it][0];

         _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
         memcpy( mcontr_buffer,  contr[it][0], numV*nsf*sizeof(double _Complex) );
         exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*nsf, MPI_DOUBLE, MPI_SUM, g_ts_comm);
         if( exitstatus != MPI_SUCCESS ) {
           fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
           return(8);
         }
#endif
      }  /* end of loop on timeslices */

      /************************************************
       * write to file
       ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
      if ( io_proc == 2 ) {
        contr_allt_buffer = (double _Complex *)malloc(numV*nsf*T_global*sizeof(double _Complex) );
        if(contr_allt_buffer == NULL ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
          return(9);
        }
      }

      /* gather to root, which must be io_proc = 2 */
      if ( io_proc > 0 ) {
        int count = numV*nsf*2*T;
        exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(10);
         }
      }
#else
      if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
      return(1);
#endif
#else
      contr_allt_buffer = contr[0][0];
#endif
      if ( io_proc == 2 ) {
        sprintf(aff_path, "%s/w_dag_gloc_s/px%.2dpy%.2dpz%.2d/g%.2d", tag, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        affdir = aff_writer_mkpath(affw, affn, aff_path);
        exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*nsf ) );
        if(exitstatus != 0) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(11);
        }

#if HAVE_MPI
        free( contr_allt_buffer );
#endif
      }
    }  /* end of loop on Gamma structures */
  }  /* end of loop on momenta */

  if(g_cart_id == 0) fprintf(stdout, "# [contract_vdag_gloc_spinor_field] end of W part\n");
  fflush(stdout);
#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif
#if 0
#endif  /* of if 0 */

  /************************************************
   ************************************************
   **
   ** XW, even part
   **
   ************************************************
   ************************************************/

  /* calculate X W */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  for( int i=0; i<numV; i++ ) {
    /*
     * W_e = X W_o = -M_ee^-1[up] M_eo W_o
     */
    memcpy( spinor_work, (double*)(W[i]), sizeof_eo_spinor_field );
    X_clover_eo ( (double*)(W[i]), spinor_work, gauge_field, mzzinv[0][0] );
  }
  free ( spinor_work ); spinor_work = NULL;

  /* loop on momenta */
  for( int im=0; im<momentum_number; im++ ) {
  
    /* make even phase field */
    make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

    /* calculate the propagators including current Fourier phase */
    for( int i=0; i<nsf; i++) {
      spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_phase[i]), prop_list_e[i], (double*)(phase_field[0]), Vhalf);
    }

    for( int ig=0; ig<gamma_id_number; ig++ ) {

      for( int it=0; it<T; it++ ) {

        /* copy timslice of V  */
        unsigned int offset = _GSI(VOL3half) * it;
        for( int i=0; i<numV; i++ ) memcpy( V_ts[i], (double*)(W[i])+offset, sizeof_eo_spinor_field_timeslice );

        /* calculate Gamma times spinor field timeslice 
         *
         *  prop_ts = g5 Gamma prop_list_o [it]
         *
         */
        for( int i=0; i<nsf; i++) {
          spinor_field_eq_gamma_ti_spinor_field( (double*)(prop_ts[i]), gamma_id_list[ig], (double*)(prop_phase[i])+offset, VOL3half);
        }
        g5_phi( (double*)(prop_ts[0]), nsf*VOL3half);

        /* matrix multiplication
         *
         * contr[it][i][k] = V_i^+ prop_ts_k
         *
         */

        BLAS_A = V_ts[0];
        BLAS_B = prop_ts[0];
        BLAS_C = contr[it][0];

         _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
         memcpy( mcontr_buffer,  contr[it][0], numV*nsf*sizeof(double _Complex) );
         exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*nsf, MPI_DOUBLE, MPI_SUM, g_ts_comm);
         if( exitstatus != MPI_SUCCESS ) {
           fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
           return(8);
         }
#endif
      }  /* end of loop on timeslices */

      /************************************************
       * write to file
       ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
      if ( io_proc == 2 ) {
        contr_allt_buffer = (double _Complex *)malloc(numV*nsf*T_global*sizeof(double _Complex) );
        if(contr_allt_buffer == NULL ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
          return(9);
        }
      }

      /* gather to root, which must be io_proc = 2 */
      if ( io_proc > 0 ) {
        int count = numV*nsf*2*T;
        exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
        if( exitstatus != MPI_SUCCESS ) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(10);
         }
      }
#else
      if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
      return(1);
#endif
#else
      contr_allt_buffer = contr[0][0];
#endif

      if ( io_proc == 2 ) {
        sprintf(aff_path, "%s/xw_dag_gloc_s/px%.2dpy%.2dpz%.2d/g%.2d", tag, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        affdir = aff_writer_mkpath(affw, affn, aff_path);
        exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*nsf ) );
        if(exitstatus != 0) {
          fprintf(stderr, "[contract_vdag_gloc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(11);
        }
#ifdef HAVE_MPI
        free( contr_allt_buffer );
#endif
      }
    }  /* end of loop on Gamma structures */
  }  /* end of loop on momenta */

  if(g_cart_id == 0) fprintf(stdout, "# [contract_vdag_gloc_spinor_field] end of X W part\n");
  fflush(stdout);
#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif

#if 0
#endif  /* of if 0 */

  fini_2level_buffer ( (double***)(&W) );

#ifdef HAVE_MPI
  free ( mcontr_buffer );
#endif
  fini_2level_buffer ( (double***)(&phase_field) );
  fini_2level_buffer ( (double***)(&V_ts) );
  fini_2level_buffer ( (double***)(&prop_ts) );
  fini_2level_buffer ( (double***)(&prop_phase) );
  fini_3level_buffer ( (double****)(&contr) );

  retime = _GET_TIME;
  if ( io_proc  == 2 ) {
    fprintf(stdout, "# [contract_vdag_gloc_spinor_field] time for contract_vdag_gloc_spinor_field = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }

  return(0);

}  /* end of contract_v_dag_gloc_spinor_field */


/*******************************************************************************************
 * calculate V^+ cvc-vertex S
 *
          subroutine zgemm  (   character   TRANSA,
          V^+ Gamma(p) S
          eo - scalar product over even 0 / odd 1 sites

          V is numV x (12 VOL3half) (C) = (12 VOL3half) x numV (F)

          prop is nsf x (12 VOL3half) (C) = (12 VOL3half) x nsf (F)

          zgemm calculates
          V^H x [ (Gamma(p) x prop) ] which is numV x nsf (F) = nsf x numV (C)
 *
 *******************************************************************************************/
int contract_vdag_cvc_spinor_field (double**prop_list_e, double**prop_list_o, int nsf, double**V, int numV, int momentum_number, int (*momentum_list)[3], struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size ) {
 
  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double);

  int exitstatus;
  double **eo_spinor_work = NULL, **eo_spinor_aux = NULL;
  double _Complex **phase_field = NULL;
  double _Complex **W = NULL;
  double _Complex **V_ts = NULL;
  double _Complex **prop_ts = NULL, **prop_vertex = NULL;
  double _Complex ***contr = NULL;
  double _Complex *contr_allt_buffer = NULL;

  double *mcontr_buffer = NULL;
  double ratime, retime;

  struct AffNode_s *affn = NULL, *affdir=NULL;
  char aff_path[200];

  /* BLAS parameters for zgemm */
  char BLAS_TRANSA = 'C';
  char BLAS_TRANSB = 'N';
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  int BLAS_M = numV;
  int BLAS_K = 12*VOL3half;
  int BLAS_N = block_size;
  int BLAS_LDA = BLAS_K;
  int BLAS_LDB = BLAS_K;
  int BLAS_LDC = BLAS_M;

  int block_num = (int)(nsf / block_size);
  if ( block_num * block_size != nsf ) {
    fprintf(stderr, "[contract_v_dag_cvc_spinor_field] Error, nsf must be divisible by block size %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  ratime = _GET_TIME;

  exitstatus = init_2level_buffer ( (double***)(&eo_spinor_work), 1, _GSI( (VOLUME+RAND)/2) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_buffer ( (double***)(&phase_field), T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_buffer ( (double***)(&V_ts), numV, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  exitstatus = init_2level_buffer ( (double***)(&prop_ts), block_size, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&prop_vertex), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_3level_buffer ( (double****)(&contr), T, block_size, 2*numV );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(5);
  }

#ifdef HAVE_MPI
  mcontr_buffer = (double*)malloc(numV * block_size * 2 * sizeof(double) ) ;
  if ( mcontr_buffer == NULL ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
#endif

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }


  /************************************************
   ************************************************
   **
   ** V, odd part
   **
   ************************************************
   ************************************************/

  for ( int iblock=0; iblock < block_num; iblock++ ) {

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* copy propagator to field with halo */
          memcpy( eo_spinor_work[0], prop_list_e[iblock*block_size + i], sizeof_eo_spinor_field );
          /* apply vertex for ODD target field */
          apply_cvc_vertex_eo((double*)(prop_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

          for( int it=0; it<T; it++ ) {

            /* copy timslice of V  */
            unsigned int offset = _GSI(VOL3half) * it;
            for( int i=0; i<numV; i++ ) memcpy( V_ts[i], (double*)(V[i])+offset, sizeof_eo_spinor_field_timeslice );

            /*
             *
             *
             */
            for( int i=0; i<block_size; i++) {
              spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_ts[i]), (double*)(prop_vertex[i])+offset, (double*)(phase_field[it]), VOL3half);
            }
            g5_phi( (double*)(prop_ts[0]), block_size*VOL3half);

            /* matrix multiplication
             *
             * contr[it][i][k] = V_i^+ prop_ts_k
             *
             */

            BLAS_A = V_ts[0];
            BLAS_B = prop_ts[0];
            BLAS_C = contr[it][0];

            _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
            memcpy( mcontr_buffer,  contr[it][0], numV*block_size*sizeof(double _Complex) );
            exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*block_size, MPI_DOUBLE, MPI_SUM, g_ts_comm);
            if( exitstatus != MPI_SUCCESS ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(8);
            }
#endif
          }  /* end of loop on timeslices */

          /************************************************
           * write to file
           ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
          if ( io_proc == 2 ) {
            contr_allt_buffer = (double _Complex *)malloc(numV*block_size*T_global*sizeof(double _Complex) );
            if(contr_allt_buffer == NULL ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
              return(9);
            }
          }

          /* gather to root, which must be io_proc = 2 */
          if ( io_proc > 0 ) {
            int count = numV * block_size * 2 * T;
            exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
            if( exitstatus != MPI_SUCCESS ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(10);
             }
          }
#else
          if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
          return(1);
#endif
#else
          contr_allt_buffer = contr[0][0];
#endif
          if ( io_proc == 2 ) {
            sprintf(aff_path, "%s/v_dag_cvc_s/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);
    
            affdir = aff_writer_mkpath(affw, affn, aff_path);
            exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*block_size ) );
            if(exitstatus != 0) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(11);
            }
#ifdef HAVE_MPI
            free( contr_allt_buffer );
#endif
          }
        }  /* end of loop on momenta */
      }   /* end of loop on momenta */
    }  /* end of loop on shift directions */
  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */

  /************************************************
   ************************************************
   **
   ** Xbar V, even part
   **
   ************************************************
   ************************************************/

  /************************************************
   * allocate auxilliary W
   ************************************************/
  exitstatus = init_2level_buffer ( (double***)(&W), numV, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }


  /* calculate Xbar V */
  for( int i=0; i<numV; i++ ) {
    /*
     * W_e = Xbar V_o = -M_ee^-1[dn] M_eo V_o
     */
    memcpy( eo_spinor_work[0], (double*)(V[i]), sizeof_eo_spinor_field );
    X_clover_eo ( (double*)(W[i]), eo_spinor_work[0], gauge_field, mzzinv[1][0] );
  }

  for( int iblock=0; iblock < block_num; iblock++ ) {

    for ( int mu=0; mu<4; mu++ ) {

      for ( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* copy propagator to field with halo */
          memcpy( eo_spinor_work[0], prop_list_o[iblock*block_size + i], sizeof_eo_spinor_field );
          /* apply vertex for ODD target field */
          apply_cvc_vertex_eo((double*)(prop_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 0);
        }

    
        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
      
          /* make even phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);
    
          for( int it=0; it<T; it++ ) {
    
            /* copy timslice of V  */
            unsigned int offset = _GSI(VOL3half) * it;
            for( int i=0; i<numV; i++ ) memcpy( V_ts[i], (double*)(W[i])+offset, sizeof_eo_spinor_field_timeslice );
    
            /*
             *
             *
             */
            for( int i=0; i<block_size; i++) {
              spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_ts[i]), (double*)(prop_vertex[i])+offset, (double*)(phase_field[it]), VOL3half);
            }
            g5_phi( (double*)(prop_ts[0]), block_size*VOL3half);
    
            /* matrix multiplication
             *
             * contr[it][i][k] = V_i^+ prop_ts_k
             *
             */
    
            BLAS_A = V_ts[0];
            BLAS_B = prop_ts[0];
            BLAS_C = contr[it][0];
    
            _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);
    
#ifdef HAVE_MPI
            memcpy( mcontr_buffer,  contr[it][0], numV*block_size*sizeof(double _Complex) );
            exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*block_size, MPI_DOUBLE, MPI_SUM, g_ts_comm);
            if( exitstatus != MPI_SUCCESS ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(8);
            }
#endif
          }  /* end of loop on timeslices */
    
          /************************************************
           * write to file
           ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
          if ( io_proc == 2 ) {
            contr_allt_buffer = (double _Complex *)malloc(numV*block_size*T_global*sizeof(double _Complex) );
            if(contr_allt_buffer == NULL ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
              return(9);
            }
          }
    
          /* gather to root, which must be io_proc = 2 */
          if ( io_proc > 0 ) {
            int count = numV*block_size*2*T;
            exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
            if( exitstatus != MPI_SUCCESS ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(10);
             }
          }
#else
          if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
          return(1);
#endif
#else
          contr_allt_buffer = contr[0][0];
#endif

          if ( io_proc == 2 ) {
            sprintf(aff_path, "%s/xv_dag_cvc_s/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);
    
            affdir = aff_writer_mkpath(affw, affn, aff_path);
            exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*block_size ) );
            if(exitstatus != 0) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(11);
            }
#ifdef HAVE_MPI
            free( contr_allt_buffer );
#endif
          }
        }  /* end of loop on momenta */  
      }  /* end of loop on fbwd */

    }  /* end of loop on shift directions mu */

  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */


  /************************************************
   ************************************************
   **
   ** W, odd part
   **
   ************************************************
   ************************************************/


  /* calculate W from V and Xbar V */
  exitstatus = init_2level_buffer ( (double***)(&eo_spinor_aux), 1, _GSI( Vhalf ) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }
  for( int i=0; i<numV; i++ ) {
    /*
     * W_o = Cbar V
         */
    memcpy( eo_spinor_work[0], (double*)(W[i]), sizeof_eo_spinor_field );
    memcpy( (double*)(W[i]), (double*)(V[i]), sizeof_eo_spinor_field );
    C_clover_from_Xeo ( (double*)(W[i]), eo_spinor_work[0], eo_spinor_aux[0], gauge_field, mzz[1][1]);
  }
  fini_2level_buffer ( (double***)(&eo_spinor_aux) );

  for( int iblock=0; iblock < block_num; iblock++ ) {

    for ( int mu=0; mu<4; mu++ ) {

      for ( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* copy propagator to field with halo */
          memcpy( eo_spinor_work[0], prop_list_e[iblock*block_size + i], sizeof_eo_spinor_field );
          /* apply vertex for ODD target field */
          apply_cvc_vertex_eo((double*)(prop_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
      
          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);
    
          for( int it=0; it<T; it++ ) {
    
            /* copy timslice of V  */
            unsigned int offset = _GSI(VOL3half) * it;
            for( int i=0; i<numV; i++ ) memcpy( V_ts[i], (double*)(W[i])+offset, sizeof_eo_spinor_field_timeslice );
    
            /* 
             *
             */
            for( int i=0; i<block_size; i++) {
              spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_ts[i]), (double*)(prop_vertex[i])+offset, (double*)(phase_field[it]), VOL3half);
            }
            g5_phi( (double*)(prop_ts[0]), block_size*VOL3half);
    
            /* matrix multiplication
             *
             * contr[it][i][k] = V_i^+ prop_ts_k
             *
             */
    
            BLAS_A = V_ts[0];
            BLAS_B = prop_ts[0];
            BLAS_C = contr[it][0];
    
             _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);
    
#ifdef HAVE_MPI
             memcpy( mcontr_buffer,  contr[it][0], numV*block_size*sizeof(double _Complex) );
             exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*block_size, MPI_DOUBLE, MPI_SUM, g_ts_comm);
             if( exitstatus != MPI_SUCCESS ) {
               fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
               return(8);
             }
#endif
          }  /* end of loop on timeslices */
    
          /************************************************
           * write to file
           ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
          if ( io_proc == 2 ) {
            contr_allt_buffer = (double _Complex *)malloc(numV*block_size*T_global*sizeof(double _Complex) );
            if(contr_allt_buffer == NULL ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
              return(9);
            }
          }
    
          /* gather to root, which must be io_proc = 2 */
          if ( io_proc > 0 ) {
            int count = numV*block_size*2*T;
            exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
            if( exitstatus != MPI_SUCCESS ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(10);
             }
          }

#else
          if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
          return(1);
#endif
#else
          contr_allt_buffer = contr[0][0];
#endif
          if ( io_proc == 2 ) {
            sprintf(aff_path, "%s/w_dag_cvc_s/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);
    
            affdir = aff_writer_mkpath(affw, affn, aff_path);
            exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*block_size ) );
            if(exitstatus != 0) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(11);
            }
    
#ifdef HAVE_MPI
            free( contr_allt_buffer );
#endif
          }

        }  /* end of loop on momenta */ 
      }  /* end of loop on fbwd */
    }  /* end of loop on shift directions mu */
  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */


  /************************************************
   ************************************************
   **
   ** XW, even part
   **
   ************************************************
   ************************************************/

  /* calculate X W */
  for( int i=0; i<numV; i++ ) {
    /*
     * W_e = X W_o = -M_ee^-1[up] M_eo W_o
     */
    memcpy( eo_spinor_work[0], (double*)(W[i]), sizeof_eo_spinor_field );
    X_clover_eo ( (double*)(W[i]), eo_spinor_work[0], gauge_field, mzzinv[0][0] );
  }

  for( int iblock=0; iblock < block_num; iblock++ ) {

    for ( int mu=0; mu<4; mu++ ) {

      for ( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* copy propagator to field with halo */
          memcpy( eo_spinor_work[0], prop_list_o[iblock*block_size + i], sizeof_eo_spinor_field );
          /* apply vertex for ODD target field */
          apply_cvc_vertex_eo((double*)(prop_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 0);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
      
          /* make even phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);
    
          for( int it=0; it<T; it++ ) {
    
            /* copy timslice of V  */
            unsigned int offset = _GSI(VOL3half) * it;
            for( int i=0; i<numV; i++ ) memcpy( V_ts[i], (double*)(W[i])+offset, sizeof_eo_spinor_field_timeslice );
    
            /*
             *
             *
             */
            for( int i=0; i<block_size; i++) {
              spinor_field_eq_spinor_field_ti_complex_field ( (double*)(prop_ts[i]), (double*)(prop_vertex[i])+offset, (double*)(phase_field[it]), VOL3half);
            }
            g5_phi( (double*)(prop_ts[0]), block_size*VOL3half);
    
            /* matrix multiplication
             *
             * contr[it][i][k] = V_i^+ prop_ts_k
             *
             */
    
            BLAS_A = V_ts[0];
            BLAS_B = prop_ts[0];
            BLAS_C = contr[it][0];
    
             _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);
    
#ifdef HAVE_MPI
             memcpy( mcontr_buffer,  contr[it][0], numV*block_size*sizeof(double _Complex) );
             exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*numV*block_size, MPI_DOUBLE, MPI_SUM, g_ts_comm);
             if( exitstatus != MPI_SUCCESS ) {
               fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
               return(8);
             }
#endif
          }  /* end of loop on timeslices */
    
          /************************************************
           * write to file
           ************************************************/
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
          if ( io_proc == 2 ) {
            contr_allt_buffer = (double _Complex *)malloc(numV*block_size*T_global*sizeof(double _Complex) );
            if(contr_allt_buffer == NULL ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from malloc %s %d\n", __FILE__, __LINE__);
              return(9);
            }
          }
    
          /* gather to root, which must be io_proc = 2 */
          if ( io_proc > 0 ) {
            int count = numV*block_size*2*T;
            exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
            if( exitstatus != MPI_SUCCESS ) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(10);
             }
          }
#else
          if (io_proc == 2) fprintf(stderr, "[contract_vdag_gloc_spinor_field] 1-dim MPI gathering currently not implemented\n");
          return(1);
#endif
#else
          contr_allt_buffer = contr[0][0];
#endif

          if ( io_proc == 2 ) {
            sprintf(aff_path, "%s/xw_dag_cvc_s/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);
    
            affdir = aff_writer_mkpath(affw, affn, aff_path);
            exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, (uint32_t)(T_global*numV*block_size ) );
            if(exitstatus != 0) {
              fprintf(stderr, "[contract_vdag_cvc_spinor_field] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
              return(11);
            }
#ifdef HAVE_MPI   
            free( contr_allt_buffer );
#endif
          }

        }  /* end of loop on momenta */  
      }  /* end of loop on fbwd */
    }  /* end of loop on shift directions mu */
  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */

  fini_2level_buffer ( (double***)(&W) );

#ifdef HAVE_MPI
  free ( mcontr_buffer );
#endif
  fini_2level_buffer ( (double***)(&eo_spinor_work) );
  fini_2level_buffer ( (double***)(&phase_field) );
  fini_2level_buffer ( (double***)(&V_ts) );
  fini_2level_buffer ( (double***)(&prop_ts) );
  fini_2level_buffer ( (double***)(&prop_vertex) );
  fini_3level_buffer ( (double****)(&contr) );

  retime = _GET_TIME;
  if ( io_proc  == 2 ) {
    fprintf(stdout, "# [contract_vdag_cvc_spinor_field] time for contract_vdag_gloc_spinor_field = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }

  return(0);

}  /* end of contract_v_dag_cvc_spinor_field */


/************************************************************
 ************************************************************
 **
 ** vdag_w_reduce_write
 **
 ************************************************************
 ************************************************************/
int vdag_w_reduce_write (double _Complex ***contr, double _Complex **V, double _Complex **W, int dimV, int dimW, char *aff_path, struct AffWriter_s *affw, struct AffNode_s *affn, int io_proc, double _Complex **V_ts, double _Complex **W_ts, double *mcontr_buffer) {

  static const unsigned int Vhalf = VOLUME / 2;
  static const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  static const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  static const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double);

  int exitstatus;
  /* BLAS parameters for zgemm */
  char BLAS_TRANSA = 'C';
  char BLAS_TRANSB = 'N';
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;
  int BLAS_M = dimV;
  int BLAS_K = 12*VOL3half;
  int BLAS_N = dimW;
  int BLAS_LDA = BLAS_K;
  int BLAS_LDB = BLAS_K;
  int BLAS_LDC = BLAS_M;
  double _Complex *BLAS_A = V_ts[0];
  double _Complex *BLAS_B = W_ts[0];
  double _Complex *BLAS_C = NULL;
  double _Complex *contr_allt_buffer = NULL;

  for( int it=0; it<T; it++ ) {

    /* copy timslice of V  */
    unsigned int offset = _GSI(VOL3half) * it;
    for( int i=0; i<dimV; i++ ) memcpy( V_ts[i], (double*)(V[i])+offset, sizeof_eo_spinor_field_timeslice );
    for( int i=0; i<dimW; i++ ) memcpy( W_ts[i], (double*)(W[i])+offset, sizeof_eo_spinor_field_timeslice );

    BLAS_C = contr[it][0];

    _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

#ifdef HAVE_MPI
    memcpy( mcontr_buffer,  contr[it][0], dimV * dimW * sizeof(double _Complex) );
    exitstatus = MPI_Allreduce(mcontr_buffer, contr[it][0], 2*dimV*dimW, MPI_DOUBLE, MPI_SUM, g_ts_comm);
    if( exitstatus != MPI_SUCCESS ) {
      fprintf(stderr, "[vdag_w_reduce_write] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(8);
    }
#endif
  }  /* end of loop on timeslices */

  /************************************************
   * write to file
   ************************************************/
#ifdef HAVE_MPI
  if ( io_proc == 2 ) {
    contr_allt_buffer = (double _Complex *)malloc(dimV*dimW * T_global*sizeof(double _Complex) );
    if(contr_allt_buffer == NULL ) {
      fprintf(stderr, "[vdag_w_reduce_write] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(9);
    }
  }
#else
  contr_allt_buffer = contr[0][0];
#endif

#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  /* gather to root, which must be io_proc = 2 */
  if ( io_proc > 0 ) {
    int count = dimV * dimW * 2 * T;
    exitstatus =  MPI_Gather(contr[0][0], count, MPI_DOUBLE, contr_allt_buffer, count, MPI_DOUBLE, 0, g_tr_comm );
    if( exitstatus != MPI_SUCCESS ) {
      fprintf(stderr, "[vdag_w_reduce_write] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(10);
    }
  }
#else
  if (io_proc == 2) {
      fprintf(stderr, "[vdag_w_reduce_write] Error, not implemented yet\n");
  }
  return(1);
#endif
#endif

  if ( io_proc == 2 ) {
    uint32_t count = dimV * dimW * T_global;
    struct AffNode_s *affdir = aff_writer_mkpath(affw, affn, aff_path);
    exitstatus = aff_node_put_complex (affw, affdir, contr_allt_buffer, count );
    if(exitstatus != 0) {
      fprintf(stderr, "[vdag_w_reduce_write] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(11);
    }

#ifdef HAVE_MPI
    free( contr_allt_buffer );
#endif
  }
  return(0);
}  /* end of vdag_w_reduce_write */


/************************************************************
 * calculate gsp V^+ Gamma(p) W
 *
 *   local vertex Gamma(p)
 *   eo - scalar product over even 0 / odd 1 sites
 *
 *   V is numV x (12 VOL3half) (C) = (12 VOL3half) x numV (F)
 *
 *   Wb is block_size x (12 VOL3half) (C) = (12 VOL3half) x block_size (F)
 *
 *   XbarV, W, XW and their blocked versions all the same formats
 *
 *   zgemm calculates
 *   V^H x [ Gamma(p) x Wb ) ] which is numV x block_size (F) = block_size x numV (C)
 *
 ************************************************************/
int contract_vdag_gloc_w_blocked (double**V, int numV, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size ) {
 
  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  const size_t sizeof_eo_spinor_field_with_halo = _GSI( (VOLUME+RAND)/2 ) * sizeof(double);
  /* const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double); */

  int exitstatus;
  double *spinor_work = NULL, *spinor_aux = NULL;
  double _Complex **phase_field = NULL;
  double _Complex **V_ts = NULL;
  double _Complex **W = NULL, **W_phase = NULL, **W_gamma_phase = NULL, **W_ts = NULL, **W_aux = NULL;
  double _Complex ***contr = NULL;

  double *mcontr_buffer = NULL;
  double ratime, retime;

  struct AffNode_s *affn = NULL;
  char aff_path[200];

  int block_num = (int)(numV / block_size);
  if ( block_num * block_size != numV ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error, numV must be divisible by block size %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  ratime = _GET_TIME;

  /**********************************
   * Fourier phase field
   **********************************/
  exitstatus = init_2level_buffer ( (double***)(&phase_field), T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /**********************************
   * auxilliary V-fields
   **********************************/
  exitstatus = init_2level_buffer ( (double***)(&W), numV, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  exitstatus = init_2level_buffer ( (double***)(&V_ts), numV, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  /**********************************
   * auxilliary W-fields
   **********************************/
  exitstatus = init_2level_buffer ( (double***)(&W_ts), block_size, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&W_phase), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_2level_buffer ( (double***)(&W_gamma_phase), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  /**********************************
   * fields for contractions
   **********************************/
  exitstatus = init_3level_buffer ( (double****)(&contr), T, block_size, 2*numV );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(5);
  }

#ifdef HAVE_MPI
  mcontr_buffer = (double*)malloc(numV*block_size*2*sizeof(double) ) ;
  if ( mcontr_buffer == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
#endif

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }


  /************************************************
   ************************************************
   **
   ** V - V (odd part)
   **
   ************************************************
   ************************************************/

  /* loop on blocks of fields */
  for( int iblock=0; iblock < block_num; iblock++ ) {

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make odd phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++ ) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(W_phase[i]), (double*)(V[iblock*block_size + i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {

        /* calculate Gamma times spinor field timeslice 
         *
         *  prop_ts = g5 Gamma prop_list_o [it]
         *
         */
        spinor_field_eq_gamma_ti_spinor_field( (double*)(W_gamma_phase[0]), gamma_id_list[ig], (double*)(W_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(W_gamma_phase[0]), block_size * Vhalf);

        /* prepare the AFF key */
        sprintf(aff_path, "%s/v_dag_gloc_v/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

/************************************************
 * NOTE: this casting may be dangerous
 *   (double _Complex**)V since V is double**
 *
 ************************************************/
        exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, W_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }
      }  /* end of loop on Gamma structures */


    }  /* end of loop on momenta */

  }  /* end of loops on blocks */



  /************************************************
   ************************************************
   **
   ** Xbar V - Xbar V (even part)
   **
   ************************************************
   ************************************************/


  /* calculate Xbar V */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  for( int i=0; i<numV; i++ ) {
    /*
     * W_e = Xbar V_o = -M_ee^-1[dn] M_eo V_o
     */
    memcpy( spinor_work, (double*)(V[i]), sizeof_eo_spinor_field );
    X_clover_eo ( (double*)(W[i]), spinor_work, gauge_field, mzzinv[1][0] );
  }
  free ( spinor_work ); spinor_work = NULL;

  for( int iblock=0; iblock < block_num; iblock++ ) {

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
      /* make even phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(W_phase[i]), (double*)(W[iblock * block_size + i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {
 
        spinor_field_eq_gamma_ti_spinor_field( (double*)(W_gamma_phase[0]), gamma_id_list[ig], (double*)(W_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(W_gamma_phase[0]), block_size * Vhalf);

        sprintf(aff_path, "%s/xv_dag_gloc_xv/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, W_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */  
  }  /* end of loop on blocks */
#if 0 
#endif  /* of if 0 */



  /************************************************
   ************************************************
   **
   ** V - W (odd part) and
   ** W - W (odd part)
   **
   ************************************************
   ************************************************/
  /* calculate W from V and Xbar V */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  spinor_aux  = (double*)malloc( sizeof_eo_spinor_field );
  if ( spinor_work == NULL || spinor_aux == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  for( int i=0; i<numV; i++ ) {
    /*
     * W_o = Cbar V
     */
    memcpy( spinor_work, (double*)(W[i]), sizeof_eo_spinor_field );
    memcpy( (double*)(W[i]), (double*)(V[i]), sizeof_eo_spinor_field );
    C_clover_from_Xeo ( (double*)(W[i]), spinor_work, spinor_aux, gauge_field, mzz[1][1]);
  }
  free ( spinor_work ); spinor_work = NULL;
  free ( spinor_aux  ); spinor_aux = NULL;

  for( int iblock=0; iblock < block_num; iblock++ ) {

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make odd phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(W_phase[i]), (double*)(W[iblock * block_size + i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {

        spinor_field_eq_gamma_ti_spinor_field( (double*)(W_gamma_phase[0]), gamma_id_list[ig], (double*)(W_phase[0]), block_size*Vhalf );
        g5_phi( (double*)(W_gamma_phase[0]), block_size * Vhalf );

        sprintf(aff_path, "%s/v_dag_gloc_w/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

/************************************************
 * NOTE: this casting may be dangerous
 *   (double _Complex**)V since V is double**
 *
 ************************************************/
        exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, W_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

        sprintf(aff_path, "%s/w_dag_gloc_w/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, W_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }


      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */  
  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */


  /************************************************
   ************************************************
   **
   ** XW - XW ( even part)
   **
   ************************************************
   ************************************************/
  /* calculate X W */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  for( int i=0; i<numV; i++ ) {
    /*
     * W_e = X W_o = -M_ee^-1[up] M_eo W_o
     */
    memcpy( spinor_work, (double*)(W[i]), sizeof_eo_spinor_field );
    X_clover_eo ( (double*)(W[i]), spinor_work, gauge_field, mzzinv[0][0] );
  }
  free ( spinor_work ); spinor_work = NULL;

  for( int iblock=0; iblock < block_num; iblock++ ) {

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make even phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(W_phase[i]), (double*)(W[iblock*block_size + i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {

        spinor_field_eq_gamma_ti_spinor_field( (double*)(W_gamma_phase[0]), gamma_id_list[ig], (double*)(W_phase[0]), block_size*Vhalf );
        g5_phi( (double*)(W_gamma_phase[0]), block_size * Vhalf );

        sprintf(aff_path, "%s/xw_dag_gloc_xw/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, W_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }
      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */
  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */

  /************************************************
   ************************************************
   **
   ** XW - XV ( even part)
   **
   ************************************************
   ************************************************/

  /* calculate X W */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }
  exitstatus = init_2level_buffer ( (double***)(&W_aux), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }


  for( int iblock=0; iblock < block_num; iblock++ ) {

    for( int i=0; i<block_size; i++ ) {
      /*
       * Xbar V = Xbar V = -M_ee^-1[dn] M_eo V
       */
      memcpy( spinor_work, (double*)(V[iblock * block_size + i]), sizeof_eo_spinor_field );
      X_clover_eo ( (double*)(W_aux[i]), spinor_work, gauge_field, mzzinv[1][0] );
    }

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make even phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(W_phase[i]), (double*)(W_aux[i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {

        spinor_field_eq_gamma_ti_spinor_field( (double*)(W_gamma_phase[0]), gamma_id_list[ig], (double*)(W_phase[0]), block_size*Vhalf );
        g5_phi( (double*)(W_gamma_phase[0]), block_size * Vhalf );

        sprintf(aff_path, "%s/xw_dag_gloc_xv/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, W_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }
      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */
  }  /* end of loop on blocks */
  free ( spinor_work ); spinor_work = NULL;
  fini_2level_buffer ( (double***)(&W_aux) );
#if 0
#endif  /* of if 0 */

#ifdef HAVE_MPI
  free ( mcontr_buffer );
#endif
  fini_2level_buffer ( (double***)(&phase_field) );
  fini_2level_buffer ( (double***)(&W) );
  fini_2level_buffer ( (double***)(&V_ts) );
  fini_2level_buffer ( (double***)(&W_ts) );
  fini_2level_buffer ( (double***)(&W_phase) );
  fini_2level_buffer ( (double***)(&W_gamma_phase) );
  fini_3level_buffer ( (double****)(&contr) );

  retime = _GET_TIME;
  if ( io_proc  == 2 ) {
    fprintf(stdout, "# [contract_v_dag_gloc_w_blocked] time for contract_v_dag_gloc_w_blocked = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }

  return(0);

}  /* end of contract_v_dag_gloc_w_blocked */


/************************************************************
 * calculate gsp V^+ Gamma(p) Phi
 *
 *   V intended eigenvectors
 *   Phi intended stochastic propagators
 *   local vertex Gamma(p)
 *   eo - scalar product over even 0 / odd 1 sites
 *
 *   V is numV x (12 VOL3half) (C) = (12 VOL3half) x numV (F)
 *
 *   Wb is block_size x (12 VOL3half) (C) = (12 VOL3half) x block_size (F)
 *
 *   XbarV, W, XW and their blocked versions all the same formats
 *
 *   zgemm calculates
 *   V^H x [ Gamma(p) x Wb ) ] which is numV x block_size (F) = block_size x numV (C)
 *
 ************************************************************/
int contract_vdag_gloc_phi_blocked (double**V, double**Phi, int numV, int numPhi, int momentum_number, int (*momentum_list)[3], int gamma_id_number, int*gamma_id_list, struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size ) {
 
  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  const size_t sizeof_eo_spinor_field_with_halo = _GSI( (VOLUME+RAND)/2 ) * sizeof(double);
  /* const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double); */

  int exitstatus;
  double *spinor_work = NULL, *spinor_aux = NULL;
  double _Complex **phase_field = NULL;
  double _Complex **V_ts = NULL;
  double _Complex **W = NULL;
  double _Complex **Phi_phase = NULL, **Phi_gamma_phase = NULL, **Phi_ts = NULL,
         **Phi_aux1 = NULL, **Phi_aux2 = NULL;
  double _Complex ***contr = NULL;

  double *mcontr_buffer = NULL;
  double ratime, retime;

  struct AffNode_s *affn = NULL;
  char aff_path[200];

  int block_num = (int)(numPhi / block_size);
  if ( block_num * block_size != numPhi ) {
    fprintf(stderr, "[contract_vdag_gloc_phi_blocked] Error, numPhi must be divisible by block size %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  ratime = _GET_TIME;

  /**********************************
   * Fourier phase field
   **********************************/
  exitstatus = init_2level_buffer ( (double***)(&phase_field), T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /**********************************
   * auxilliary V-fields
   **********************************/
  exitstatus = init_2level_buffer ( (double***)(&W), numV, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  exitstatus = init_2level_buffer ( (double***)(&V_ts), numV, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  /**********************************
   * auxilliary Phi-fields
   **********************************/
  exitstatus = init_2level_buffer ( (double***)(&Phi_ts), block_size, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&Phi_phase), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_2level_buffer ( (double***)(&Phi_gamma_phase), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_2level_buffer ( (double***)(&Phi_aux1), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_2level_buffer ( (double***)(&Phi_aux2), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  /**********************************
   * fields for contractions
   **********************************/
  exitstatus = init_3level_buffer ( (double****)(&contr), T, block_size, 2*numV );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(5);
  }

#ifdef HAVE_MPI
  mcontr_buffer = (double*)malloc(numV*block_size*2*sizeof(double) ) ;
  if ( mcontr_buffer == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
#endif

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }

  /************************************************
   ************************************************
   **
   ** V, W - Phi, Xi ( all four odd part)
   **
   ************************************************
   ************************************************/

  /* calculate W from V 
   * W = Cbar V
   * */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  if ( spinor_work == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }
  for( int i = 0; i < numV; i++ ) {
    /* Cbar_oo = g5 ( Mbar_oo - M_oe Mbar_ee^-1 M_eo ) */
    C_clover_oo ( (double*)(W[i]), (double*)(V[i]), gauge_field, spinor_work, mzz[1][1], mzzinv[1][0]);
  }

  /* loop on blocks of fields */
  for( int iblock=0; iblock < block_num; iblock++ ) {

    /* copy Phi */
    memcpy( Phi_aux1[0], Phi[iblock*block_size], block_size * sizeof_eo_spinor_field );
    /* calculate Xi  = Cbar Phi */
    for( int i = 0; i < block_size; i++ ) {
      /* Cbar = g5 ( Mbar_oo - M_oe Mbar_ee^-1 M_eo ) */
      C_clover_oo ( (double*)(Phi_aux2[i]), (double*)(Phi_aux1[i]), gauge_field, spinor_work, mzz[1][1], mzzinv[1][0]);
    }

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make odd phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++ ) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_aux1[i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {

        /* calculate Gamma times spinor field timeslice 
         */
        spinor_field_eq_gamma_ti_spinor_field( (double*)(Phi_gamma_phase[0]), gamma_id_list[ig], (double*)(Phi_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(Phi_gamma_phase[0]), block_size * Vhalf);

        /* prepare the AFF key */
        sprintf(aff_path, "%s/v_dag_gloc_phi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

/************************************************
 * NOTE: dangerous type cast
 *   (double _Complex**)V from  double**V
 *
 ************************************************/
        exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

        /* prepare the AFF key */
        sprintf(aff_path, "%s/w_dag_gloc_phi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }
      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */

    for( int im=0; im<momentum_number; im++ ) {
  
      /* make odd phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++ ) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_aux2[i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {

        /* calculate Gamma times spinor field timeslice 
         */
        spinor_field_eq_gamma_ti_spinor_field( (double*)(Phi_gamma_phase[0]), gamma_id_list[ig], (double*)(Phi_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(Phi_gamma_phase[0]), block_size * Vhalf);

        /* prepare the AFF key */
        sprintf(aff_path, "%s/v_dag_gloc_xi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

/************************************************
 * NOTE: dangerous type cast
 *   (double _Complex**)V from  double**V
 *
 ************************************************/
        exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

        /* prepare the AFF key */
        sprintf(aff_path, "%s/w_dag_gloc_xi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }
      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */

  }  /* end of loops on blocks */
  free( spinor_work );

  /************************************************
   ************************************************
   **
   ** Xbar V - Xbar Phi, X Xi (even part)
   **
   ************************************************
   ************************************************/

  /* calculate Xbar V */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  spinor_aux  = (double*)malloc( sizeof_eo_spinor_field );
  if ( spinor_work == NULL || spinor_aux == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  for( int i=0; i<numV; i++ ) {
    /*
     * W <- Xbar V = -M_ee^-1[dn] M_eo V
     */
    memcpy( spinor_work, (double*)(V[i]), sizeof_eo_spinor_field );
    X_clover_eo ( (double*)(W[i]), spinor_work, gauge_field, mzzinv[1][0] );
  }

  for( int iblock=0; iblock < block_num; iblock++ ) {

    /* calculate Xbar Phi 
     * calculate X Xi 
     */
    for( int i = 0; i < block_size; i++ ) {
      memcpy( spinor_work, Phi[iblock * block_size + i], sizeof_eo_spinor_field );
      X_clover_eo ( (double*)(Phi_aux1[i]), spinor_work, gauge_field, mzzinv[1][0] );

      /* calculate Xi  = Cbar from Phi, Xbar Phi */
      memcpy( Phi_aux2[i], Phi[iblock * block_size + i], sizeof_eo_spinor_field );
      memcpy( spinor_work, Phi_aux1[i], sizeof_eo_spinor_field );
      C_clover_from_Xeo ( (double*)(Phi_aux2[i]), spinor_work, spinor_aux, gauge_field, mzz[1][1]);

      /* calculate X Xi */
      memcpy(spinor_work, Phi_aux2[i], sizeof_eo_spinor_field );
      X_clover_eo ( (double*)(Phi_aux2[i]), spinor_work, gauge_field, mzzinv[0][0] );
    }

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make even phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_aux1[i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {
 
        spinor_field_eq_gamma_ti_spinor_field( (double*)(Phi_gamma_phase[0]), gamma_id_list[ig], (double*)(Phi_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(Phi_gamma_phase[0]), block_size * Vhalf);

        sprintf(aff_path, "%s/xv_dag_gloc_xphi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */  

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make even phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_aux2[i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {
 
        spinor_field_eq_gamma_ti_spinor_field( (double*)(Phi_gamma_phase[0]), gamma_id_list[ig], (double*)(Phi_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(Phi_gamma_phase[0]), block_size * Vhalf);

        sprintf(aff_path, "%s/xv_dag_gloc_xxi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */  
  }  /* end of loop on blocks */
  free ( spinor_work );
  free ( spinor_aux  );

  /************************************************
   ************************************************
   **
   ** X W - Xbar Phi, X Xi (even part)
   **
   ************************************************
   ************************************************/

  /* calculate Xbar V */
  spinor_work = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  spinor_aux  = (double*)malloc( sizeof_eo_spinor_field );
  if ( spinor_work == NULL || spinor_aux == NULL ) {
    fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(12);
  }

  for( int i=0; i<numV; i++ ) {
    /* aux <- Cbar V = W */
    C_clover_oo (spinor_aux, (double*)(V[i]), gauge_field, spinor_work, mzz[1][1], mzzinv[1][0] );

    /* work <- aux */
    memcpy( spinor_work, spinor_aux, sizeof_eo_spinor_field );
    /* W <- X aux */
    X_clover_eo ( (double*)(W[i]), spinor_work, gauge_field, mzzinv[0][0] );
  }

  for( int iblock=0; iblock < block_num; iblock++ ) {

    /* calculate Xbar Phi */
    for( int i = 0; i < block_size; i++ ) {
      /* work <- phi */
      memcpy( spinor_work, Phi[iblock * block_size + i], sizeof_eo_spinor_field );
      /* phi <- Xbar work */
      X_clover_eo ( (double*)(Phi_aux1[i]), spinor_work, gauge_field, mzzinv[1][0] );

      /* phi2 <- phi */
      memcpy( Phi_aux2[i], Phi[iblock * block_size + i], sizeof_eo_spinor_field );
      /* work <- Xbar phi */
      memcpy( spinor_work, Phi_aux1[i], sizeof_eo_spinor_field );
      /* phi2 <- Cbar( phi2, work) using aux */
      C_clover_from_Xeo ( (double*)(Phi_aux2[i]), spinor_work, spinor_aux, gauge_field, mzz[1][1]);

      /* work <- phi2 */
      memcpy(spinor_work, Phi_aux2[i], sizeof_eo_spinor_field );
      /* phi2 <- X work */
      X_clover_eo ( (double*)(Phi_aux2[i]), spinor_work, gauge_field, mzzinv[0][0] );
    }

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make even phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_aux1[i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {
 
        spinor_field_eq_gamma_ti_spinor_field( (double*)(Phi_gamma_phase[0]), gamma_id_list[ig], (double*)(Phi_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(Phi_gamma_phase[0]), block_size * Vhalf);

        sprintf(aff_path, "%s/xw_dag_gloc_xphi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */  

    /* loop on momenta */
    for( int im=0; im<momentum_number; im++ ) {
  
      /* make even phase field */
      make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

      /* calculate the propagators including current Fourier phase */
      for( int i=0; i < block_size; i++) {
        spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_aux2[i]), (double*)(phase_field[0]), Vhalf);
      }

      for( int ig=0; ig<gamma_id_number; ig++ ) {
 
        spinor_field_eq_gamma_ti_spinor_field( (double*)(Phi_gamma_phase[0]), gamma_id_list[ig], (double*)(Phi_phase[0]), block_size * Vhalf );
        g5_phi( (double*)(Phi_gamma_phase[0]), block_size * Vhalf);

        sprintf(aff_path, "%s/xw_dag_gloc_xxi/block%.2d/px%.2dpy%.2dpz%.2d/g%.2d", tag, iblock, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2], gamma_id_list[ig]);

        exitstatus = vdag_w_reduce_write ( contr, W, Phi_gamma_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_v_dag_gloc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(2);
        }

      }  /* end of loop on Gamma structures */
    }  /* end of loop on momenta */  
  }  /* end of loop on blocks */
  free ( spinor_work );
  free ( spinor_aux  );

#ifdef HAVE_MPI
  free ( mcontr_buffer );
#endif
  fini_2level_buffer ( (double***)(&phase_field) );
  fini_2level_buffer ( (double***)(&W) );
  fini_2level_buffer ( (double***)(&V_ts) );
  fini_2level_buffer ( (double***)(&Phi_ts) );
  fini_2level_buffer ( (double***)(&Phi_phase) );
  fini_2level_buffer ( (double***)(&Phi_gamma_phase) );
  fini_3level_buffer ( (double****)(&contr) );

  retime = _GET_TIME;
  if ( io_proc  == 2 ) {
    fprintf(stdout, "# [contract_v_dag_gloc_phi_blocked] time for contract_v_dag_gloc_phi_blocked = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }

  return(0);

}  /* end of contract_v_dag_gloc_phi */


/*******************************************************************************************
 * calculate V^+ cvc-vertex S
 *
          subroutine zgemm  (   character   TRANSA,
          V^+ Gamma(p) S
          eo - scalar product over even 0 / odd 1 sites

          V is numV x (12 VOL3half) (C) = (12 VOL3half) x numV (F)

          prop is nsf x (12 VOL3half) (C) = (12 VOL3half) x nsf (F)

          zgemm calculates
          V^H x [ (Gamma(p) x prop) ] which is numV x nsf (F) = nsf x numV (C)
 *
 *******************************************************************************************/
int contract_vdag_cvc_w_blocked (double**V, int numV, int momentum_number, int (*momentum_list)[3], struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size ) {
 
  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  const size_t sizeof_eo_spinor_field_with_halo = _GSI( (VOLUME+RAND)/2 ) * sizeof(double);
  /* const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double); */

  int exitstatus;
  double **eo_spinor_work = NULL;
  double _Complex **phase_field = NULL;

  /* auxilliary V fields */
  double _Complex **V_ts = NULL;

  /* auxilliary W fields */
  double _Complex **W = NULL;
  double _Complex **W_ts = NULL, **W_vertex = NULL,**W_phase = NULL, **W_aux = NULL;

  /* fields for contractions */
  double _Complex ***contr = NULL;

  double *mcontr_buffer = NULL;
  double ratime, retime;

  struct AffNode_s *affn = NULL;
  char aff_path[200];

  int block_num = (int)( numV / block_size);
  if ( block_num * block_size != numV ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error, numV must be divisible by block size %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  ratime = _GET_TIME;

  exitstatus = init_2level_buffer ( (double***)(&eo_spinor_work), 1, _GSI( (VOLUME+RAND)/2) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /* Fourier phases */
  exitstatus = init_2level_buffer ( (double***)(&phase_field), T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /* auxilliary V fields */
  exitstatus = init_2level_buffer ( (double***)(&V_ts), numV, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  /* auxilliary W fields  */
  exitstatus = init_2level_buffer ( (double***)(&W), numV, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&W_aux), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&W_ts), block_size, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&W_vertex), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_2level_buffer ( (double***)(&W_phase), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  /* fields for contractions */
  exitstatus = init_3level_buffer ( (double****)(&contr), T, block_size, 2*numV );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(5);
  }

#ifdef HAVE_MPI
  /* buffer for xchange */
  mcontr_buffer = (double*)malloc(numV * block_size * 2 * sizeof(double) ) ;
  if ( mcontr_buffer == NULL ) {
    fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
#endif

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }

  /************************************************
   ************************************************
   **
   ** calculate W from V
   **
   ************************************************
   ************************************************/
  for( int i=0; i < numV; i++ ) {
    /* W = Cbar_oo V */
    C_clover_oo ( (double*)(W[i]), (double*)(V[i]), gauge_field, eo_spinor_work[0], mzz[1][1], mzzinv[1][0]);
  }

  /************************************************
   ************************************************
   **
   ** V - Xbar V
   ** V - X W
   ** W - Xbar V
   ** W - X W
   **
   ************************************************
   ************************************************/

  for ( int iblock=0; iblock < block_num; iblock++ ) {

    /* calculate a block of Xbar V */
    for( int i=0; i < block_size; i++ ) {
      /* spinor_work = V */
      memcpy( eo_spinor_work[0], V[iblock*block_size + i], sizeof_eo_spinor_field );
      /* W_aux = Xbar V */
      X_clover_eo ( (double*)(W_aux[i]), eo_spinor_work[0], gauge_field, mzzinv[1][0]);
    }

    /************************************************
     *
     * V - Xbar V and
     * W - Xbar V
     *
     ************************************************/

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = W_aux = Xbar V */
          memcpy( eo_spinor_work[0], (double*)(W_aux[i]), sizeof_eo_spinor_field );
          /* W_vertex = CVC(mu, fbwd) Xbar V
           *   ODD target field
           * */
          apply_cvc_vertex_eo( (double*)(W_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(W_phase[i]), (double*)(W_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(W_phase[0]), block_size*Vhalf);

          /************************************************
           * (1) V - Xbar V
           ************************************************/
          sprintf(aff_path, "%s/v_dag_cvc_xv/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

/******************************************
 * NOTE: DANGEROUS TYPECAST
 ******************************************/
          exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, W_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

          /************************************************
           * (2) W - Xbar V
           ************************************************/
          sprintf(aff_path, "%s/w_dag_cvc_xv/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, W_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        } /* end of loop on momenta */ 
      }  /* end of loop on fwd, bwd */ 
    }  /* end of loop on shift directions */

    /************************************************
     *
     * V - X W and
     * W - X W
     *
     ************************************************/

    /* calculate a block of X W */
    for( int i=0; i < block_size; i++ ) {
      /* spinor_work = W */
      memcpy( eo_spinor_work[0], W[iblock*block_size + i], sizeof_eo_spinor_field );
      /* W_aux = X W */
      X_clover_eo ( (double*)(W_aux[i]), eo_spinor_work[0], gauge_field, mzzinv[0][0]);
    }

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = W_aux = X W */
          memcpy( eo_spinor_work[0], (double*)(W_aux[i]), sizeof_eo_spinor_field );
          /* W_vertex = CVC(mu, fbwd) X W
           *   ODD target field
           * */
          apply_cvc_vertex_eo( (double*)(W_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(W_phase[i]), (double*)(W_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(W_phase[0]), block_size*Vhalf);

          /************************************************
           * (3) V - X W
           ************************************************/
          sprintf(aff_path, "%s/v_dag_cvc_xw/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

/******************************************
 * NOTE: DANGEROUS TYPECAST
 ******************************************/
          exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, W_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

          /************************************************
           * (4) W - X W
           ************************************************/
          sprintf(aff_path, "%s/w_dag_cvc_xw/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, W_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, W_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_w_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        } /* end of loop on momenta */ 
      }  /* end of loop on fwd, bwd */ 
    }  /* end of loop on shift directions */

  }  /* end of loop on blocks */

#ifdef HAVE_MPI
  free ( mcontr_buffer );
#endif
  fini_2level_buffer ( (double***)(&eo_spinor_work) );
  fini_2level_buffer ( (double***)(&phase_field) );
  fini_2level_buffer ( (double***)(&V_ts) );
  fini_2level_buffer ( (double***)(&W) );
  fini_2level_buffer ( (double***)(&W_ts) );
  fini_2level_buffer ( (double***)(&W_aux) );
  fini_2level_buffer ( (double***)(&W_vertex) );
  fini_2level_buffer ( (double***)(&W_phase) );
  fini_3level_buffer ( (double****)(&contr) );

  retime = _GET_TIME;
  if ( io_proc  == 2 ) {
    fprintf(stdout, "# [contract_vdag_cvc_w_blocked] time for contract_vdag_cvc_w_blocked = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }

  return(0);

}  /* end of contract_v_dag_cvc_w_blocked */


/*******************************************************************************************
 * calculate V^+ cvc-vertex Phi
 *
          V^+ point-split vertex Phi
          eo - scalar product over even 0 / odd 1 sites
 *
 *******************************************************************************************/
int contract_vdag_cvc_phi_blocked (double**V, double**Phi, int numV, int numPhi, int momentum_number, int (*momentum_list)[3], struct AffWriter_s*affw, char*tag, int io_proc, double*gauge_field, double **mzz[2], double**mzzinv[2], int block_size ) {
 
  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  /* const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double); */

  int exitstatus;
  double **eo_spinor_work = NULL;
  double _Complex **phase_field = NULL;

  /* auxilliary V fields */
  double _Complex **V_ts = NULL;
  double _Complex **W = NULL;

  /* auxilliary Phi fields */
  double _Complex **Phi_ts = NULL, **Phi_vertex = NULL,**Phi_phase = NULL, **Phi_aux = NULL;

  /* fields for contractions */
  double _Complex ***contr = NULL;

  double *mcontr_buffer = NULL;
  double ratime, retime;

  struct AffNode_s *affn = NULL;
  char aff_path[200];

  int block_num = (int)( numPhi / block_size);
  if ( block_num * block_size != numPhi ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error, numPhi must be divisible by block size %s %d\n", __FILE__, __LINE__);
    return(1);
  }

  ratime = _GET_TIME;

  exitstatus = init_2level_buffer ( (double***)(&eo_spinor_work), 2, _GSI( (VOLUME+RAND)/2) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /* Fourier phases */
  exitstatus = init_2level_buffer ( (double***)(&phase_field), T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /* auxilliary V fields */
  exitstatus = init_2level_buffer ( (double***)(&V_ts), numV, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  /* auxilliary W fields  */
  exitstatus = init_2level_buffer ( (double***)(&W), numV, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  /* auxilliary fields for Phi */
  exitstatus = init_2level_buffer ( (double***)(&Phi_aux), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&Phi_ts), block_size, _GSI(VOL3half) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  exitstatus = init_2level_buffer ( (double***)(&Phi_vertex), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  exitstatus = init_2level_buffer ( (double***)(&Phi_phase), block_size, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }

  /* fields for contractions */
  exitstatus = init_3level_buffer ( (double****)(&contr), T, block_size, 2*numV );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(5);
  }

#ifdef HAVE_MPI
  /* buffer for xchange */
  mcontr_buffer = (double*)malloc(numV * block_size * 2 * sizeof(double) ) ;
  if ( mcontr_buffer == NULL ) {
    fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
#endif

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }

  /************************************************
   ************************************************
   **
   ** calculate Xbarm V from V
   **
   **   all vectors
   **
   ************************************************
   ************************************************/
  for( int i=0; i < numV; i++ ) {
    /* spinor_work = V */
    memcpy( eo_spinor_work[0], (double*)(V[i]), sizeof_eo_spinor_field );
    /* W = Xbar V */
    X_clover_eo ( (double*)(W[i]), eo_spinor_work[0], gauge_field, mzzinv[1][0]);
  }


  /************************************************
   ************************************************
   **
   ** Xbar V - Phi
   ** V      - Xbar Phi
   ** Xbar V - Sigma
   ** V      - X Sigma
   **
   ************************************************
   ************************************************/

  for ( int iblock=0; iblock < block_num; iblock++ ) {

    /* calculate a block of Xbar Phi */
    for( int i=0; i < block_size; i++ ) {
      /* spinor_work = Phi */
      memcpy( eo_spinor_work[0], (double*)(Phi[iblock*block_size + i]), sizeof_eo_spinor_field );
      /* Phi_aux = Xbar spinor_work = Xbar Phi */
      X_clover_eo ( (double*)(Phi_aux[i]), eo_spinor_work[0], gauge_field, mzzinv[1][0]);
    }

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /************************************************
         * (1) Xbar V - Phi
         ************************************************/

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi */
          memcpy( eo_spinor_work[0], (double*)(Phi[iblock*block_size + i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) Phi 
           *   EVEN target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 0);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make even phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);


          sprintf(aff_path, "%s/xv_dag_cvc_phi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        } /* end of loop on momenta */ 

        /************************************************
         * (2) V - Xbar Phi
         ************************************************/

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi_aux = Xbar Phi */
          memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) Xbar Phi
           *   ODD target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);

          sprintf(aff_path, "%s/v_dag_cvc_xphi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

/************************************************
 * NOTE: DANGEROUS TYPECAST
 ************************************************/
          exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        } /* end of loop on momenta */ 

      }  /* end of loop on fwd, bwd */ 
    }  /* end of loop on shift directions */

    /************************************************
     * (3) Xbar V - Xi
     ************************************************/

    /* calculate block of Sigma from Xbar Phi and Phi */
    for( int i = 0; i < block_size; i++  ) {
      /* eo_spinor = Xbar Phi */
      memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
      /* Phi_aux = Phi */
      memcpy( (double*)(Phi_aux[i]), (double*)(Phi[iblock*block_size+i]) , sizeof_eo_spinor_field );
      /* Phi_aux = Cbar_oo ( Phi_aux, Phi ) = Sigma */
      C_clover_from_Xeo ( (double*)(Phi_aux[i]), eo_spinor_work[0], eo_spinor_work[1], gauge_field, mzz[1][1]);
    }

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi_aux = Sigma */
          memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) Sigma
           *   EVEN target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 0);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make even phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);


          sprintf(aff_path, "%s/xv_dag_cvc_xi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        } /* end of loop on momenta */ 

      }  /* end of loop on fwd, bwd */
    }  /* end of loop on shift directions */
#if 0
#endif  /* of if 0*/


    /************************************************
     * (4) V - X Xi
     ************************************************/

    /* calculate block of X Sigma from Sigma */
    for( int i = 0; i < block_size; i++  ) {
      /* eo_spinor = Sigma */
      memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
      /* Phi_aux = X Sigma */
      X_clover_eo ( (double*)(Phi_aux[i]), eo_spinor_work[0], gauge_field, mzzinv[0][0]);
    }


    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi_aux = X Sigma */
          memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) X Sigma
           *   ODD target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);


          sprintf(aff_path, "%s/v_dag_cvc_xxi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

/************************************************
 * NOTE: DANGEROUS TYPECAST
 ************************************************/
          exitstatus = vdag_w_reduce_write ( contr, (double _Complex**)V, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        } /* end of loop on momenta */ 

      }  /* end of loop on fwd, bwd */
    }  /* end of loop on shift directions */
#if 0
#endif  /* of if 0 */
  }  /* end of loop on blocks */



  /************************************************
   * W - Xbar Phi and
   * W - X Xi
   ************************************************/

  /* calculate W from V, Xbar V */
  for( int i = 0; i < numV; i++ ) {
    /* spinor_work = W = Xbar V */
    memcpy( eo_spinor_work[0], (double*)(W[i]), sizeof_eo_spinor_field );
    /* W = V */
    memcpy( (double*)(W[i]), (double*)(V[i]), sizeof_eo_spinor_field );
    /* W = Cbar_oo ( W, spinor_work )*/
    C_clover_from_Xeo ( (double*)(W[i]), eo_spinor_work[0], eo_spinor_work[1], gauge_field, mzz[1][1]);
  }

  /* loop on blocks of fields */
  for ( int iblock=0; iblock < block_num; iblock++ ) {

    /* calculate a block of Xbar Phi */
    for( int i=0; i < block_size; i++ ) {
      /* spinor_work = Phi */
      memcpy( eo_spinor_work[0], (double*)(Phi[iblock*block_size + i]), sizeof_eo_spinor_field );
      /* Phi_aux = Xbar Phi */
      X_clover_eo ( (double*)(Phi_aux[i]), eo_spinor_work[0], gauge_field, mzzinv[1][0]);
    }

    /************************************************
     * (5) W - Xbar Phi
     ************************************************/

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi_aux = Xbar Phi */
          memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) Xbar Phi 
           *   ODD target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {
  
          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);

          sprintf(aff_path, "%s/w_dag_cvc_xphi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        }  /* end of loop on momenta */
      }  /* end of loop on fwd, bwd */
    }  /* end of loop on shift directions */

    /************************************************
     * (6) W - X Xi
     ************************************************/

    /* calculate a block of X Sigma */
    for( int i = 0; i < block_size; i++  ) {
      /* spinor_work = Phi_aux = Xbar Phi */
      memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
      /* Phi_aux = Phi */
      memcpy( (double*)(Phi_aux[i]), (double*)(Phi[iblock*block_size+i]) , sizeof_eo_spinor_field );
      /* Phi_aux = C_oo ( Phi_aux, Phi ) = Sigma */
      C_clover_from_Xeo ( (double*)(Phi_aux[i]), eo_spinor_work[0], eo_spinor_work[1], gauge_field, mzz[1][1]);
      /* spinor_work = Phi_aux = Sigma */
      memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
      /* Phi_aux = X spinor_work  = */
      X_clover_eo ( (double*)(Phi_aux[i]), eo_spinor_work[0], gauge_field, mzzinv[0][0]);
    }

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi_aux = X Sigma */
          memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) X Sigma
           *   ODD target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 1);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {

          /* make odd phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 1);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);

          sprintf(aff_path, "%s/w_dag_cvc_xxi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }


        }  /* end of loop on momenta */
      }  /* end of loop on fwd, bwd */
    }  /* end of loop on shift directions */

  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */


  /************************************************
   * X W - Phi and
   * X W - Xi
   ************************************************/

  /* calculate W <- X W from W */
  for( int i=0; i < numV; i++ ) {
    /* spinor_work = W */
    memcpy( eo_spinor_work[0], (double*)(W[i]), sizeof_eo_spinor_field );
    /* W = X spinor_work = X W */
    X_clover_eo ( (double*)(W[i]), eo_spinor_work[0], gauge_field, mzzinv[0][0]);
  }

  /* loop on blocks of fields */
  for ( int iblock=0; iblock < block_num; iblock++ ) {

    /* calculate a block of Sigma */
    for( int i = 0; i < block_size; i++  ) {
      /* Phi_aux = Cbar_oo Phi */
      C_clover_oo ( (double*)(Phi_aux[i]), (double*)(Phi[iblock * block_size + i]), gauge_field, eo_spinor_work[0], mzz[1][1], mzzinv[1][0] );
    }

    /************************************************
     * (7) X W - Phi
     ************************************************/

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi */
          memcpy( eo_spinor_work[0], (double*)(Phi[iblock*block_size+i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) Phi
           *   EVEN target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 0);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {

          /* make even phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);

          sprintf(aff_path, "%s/xw_dag_cvc_phi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        }  /* end of loop on momenta */
      }  /* end of loop on fwd, bwd */
    }  /* end of loop on shift directions */

    /************************************************
     * (8) X W - Sigma
     ************************************************/

    /* loop on directions mu */
    for( int mu=0; mu<4; mu++ ) {

      /* loop on fwd / bwd */
      for( int fbwd=0; fbwd<2; fbwd++ ) {

        /* apply the cvc vertex in direction mu, fbwd to current block of fields */
        for( int i=0; i < block_size; i++) {
          /* spinor_work = Phi_aux = Sigma */
          memcpy( eo_spinor_work[0], (double*)(Phi_aux[i]), sizeof_eo_spinor_field );
          /* Phi_vertex = CVC(mu, fbwd) Sigma
           *   EVEN target field
           * */
          apply_cvc_vertex_eo( (double*)(Phi_vertex[i]), eo_spinor_work[0], mu, fbwd, gauge_field, 0);
        }

        /* loop on momenta */
        for( int im=0; im<momentum_number; im++ ) {

          /* make even phase field */
          make_eo_phase_field_sliced3d ( phase_field, momentum_list[im], 0);

          /* multiply by momentum phase */
          for( int i=0; i < block_size; i++) {
            spinor_field_eq_spinor_field_ti_complex_field ( (double*)(Phi_phase[i]), (double*)(Phi_vertex[i]), (double*)(phase_field[0]), Vhalf);
          }
          g5_phi( (double*)(Phi_phase[0]), block_size*Vhalf);

          sprintf(aff_path, "%s/xw_dag_cvc_xi/block%.2d/mu%d/fbwd%d/px%.2dpy%.2dpz%.2d", tag, iblock, mu, fbwd, momentum_list[im][0], momentum_list[im][1], momentum_list[im][2]);

          exitstatus = vdag_w_reduce_write ( contr, W, Phi_phase, numV, block_size, aff_path, affw, affn, io_proc, V_ts, Phi_ts, mcontr_buffer);
          if ( exitstatus != 0 ) {
            fprintf(stderr, "[contract_vdag_cvc_phi_blocked] Error from vdag_w_reduce_write, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
            return(2);
          }

        }  /* end of loop on momenta */
      }  /* end of loop on fwd, bwd */
    }  /* end of loop on shift directions */
  }  /* end of loop on blocks */
#if 0
#endif  /* of if 0 */

#ifdef HAVE_MPI
  free ( mcontr_buffer );
#endif
  fini_2level_buffer ( (double***)(&eo_spinor_work) );
  fini_2level_buffer ( (double***)(&phase_field) );
  fini_2level_buffer ( (double***)(&V_ts) );
  fini_2level_buffer ( (double***)(&W) );
  fini_2level_buffer ( (double***)(&Phi_ts) );
  fini_2level_buffer ( (double***)(&Phi_aux) );
  fini_2level_buffer ( (double***)(&Phi_vertex) );
  fini_2level_buffer ( (double***)(&Phi_phase) );
  fini_3level_buffer ( (double****)(&contr) );

  retime = _GET_TIME;
  if ( io_proc  == 2 ) {
    fprintf(stdout, "# [contract_vdag_cvc_phi_blocked] time for contract_v_dag_cvc_phi_blocked = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }

  return(0);

}  /* end of contract_v_dag_cvc_phi_blocked */
#if 0
#endif  /* of if 0 */

}  /* end of namespace cvc */
