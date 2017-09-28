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
#include "scalar_products.h"

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
void init_contract_cvc_tensor_usource(double *gauge_field, int source_coords[4], complex *phase ) {

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
    if ( phase != NULL ) {
      _cm_eq_cm_ti_co(Usource[0], &gauge_field[_GGI(source_location,0)], &phase[0]);
      _cm_eq_cm_ti_co(Usource[1], &gauge_field[_GGI(source_location,1)], &phase[1]);
      _cm_eq_cm_ti_co(Usource[2], &gauge_field[_GGI(source_location,2)], &phase[2]);
      _cm_eq_cm_ti_co(Usource[3], &gauge_field[_GGI(source_location,3)], &phase[3]);
    } else {
      _cm_eq_cm(Usource[0], &gauge_field[_GGI(source_location,0)] );
      _cm_eq_cm(Usource[1], &gauge_field[_GGI(source_location,1)] );
      _cm_eq_cm(Usource[2], &gauge_field[_GGI(source_location,2)] );
      _cm_eq_cm(Usource[3], &gauge_field[_GGI(source_location,3)] );
    }
  }

#ifdef HAVE_MPI
  MPI_Bcast(Usourcebuffer, 72, MPI_DOUBLE, source_proc_id, g_cart_grid);
#endif

#if 0
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
#endif  /* of if 0 */

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
  free ( cvc_tensor_lexic );
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

/***************************************************************************
 * write time-momentum-dependent contraction  results to AFF file
 *
 * p runs slower than t, i.e. c_tp[momentum][time]
 *
 ***************************************************************************/

int contract_write_to_aff_file (double **c_tp, struct AffWriter_s*affw, char*tag, int (*momentum_list)[3], int momentum_number, int io_proc ) {

  int exitstatus, i;
  double ratime, retime;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char aff_buffer_path[200];
  double *buffer = NULL;
  double _Complex *aff_buffer = NULL;
  double _Complex *zbuffer = NULL;

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[contract_write_to_aff_file] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }

    zbuffer = (double _Complex*)malloc(  momentum_number * T_global * sizeof(double _Complex) );
    if( zbuffer == NULL ) {
      fprintf(stderr, "[contract_write_to_aff_file] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(6);
    }
  }

  ratime = _GET_TIME;

  /* reorder c_tp into buffer with order time - munu - momentum */
  buffer = (double*)malloc(  momentum_number * 2 * T * sizeof(double) );
  if( buffer == NULL ) {
    fprintf(stderr, "[contract_write_to_aff_file] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
  i = 0;
  for( int it = 0; it < T; it++ ) {
    for( int ip=0; ip<momentum_number; ip++) {
      buffer[i++] = c_tp[ip][2*it  ];
      buffer[i++] = c_tp[ip][2*it+1];
    }
  }

#ifdef HAVE_MPI
  i = momentum_number * 2 * T;
#  if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
  if(io_proc>0) {
    exitstatus = MPI_Gather(buffer, i, MPI_DOUBLE, zbuffer, i, MPI_DOUBLE, 0, g_tr_comm);
    if(exitstatus != MPI_SUCCESS) {
      fprintf(stderr, "[contract_write_to_aff_file] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(3);
    }
  }
#  else
  exitstatus = MPI_Gather(buffer, i, MPI_DOUBLE, zbuffer, i, MPI_DOUBLE, 0, g_cart_grid);
  if(exitstatus != MPI_SUCCESS) {
    fprintf(stderr, "[contract_write_to_aff_file] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }
#  endif

#else
  memcpy(zbuffer, buffer, momentum_number * 2 * T * sizeof(double) );
#endif
  free( buffer );

  if(io_proc == 2) {

    /* reverse the ordering back to momentum - munu - time */
    aff_buffer = (double _Complex*)malloc( momentum_number * T_global * sizeof(double _Complex) );
    if(aff_buffer == NULL) {
      fprintf(stderr, "[contract_write_to_aff_file] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(2);
    }
    i = 0;
    for( int ip=0; ip<momentum_number; ip++) {
      for( int it = 0; it < T_global; it++ ) {
        int offset = it * momentum_number + ip;
        aff_buffer[i++] = zbuffer[offset];
      }
    }
    free( zbuffer );

    for(i=0; i < momentum_number; i++) {
      sprintf(aff_buffer_path, "%s/px%.2dpy%.2dpz%.2d", tag, momentum_list[i][0], momentum_list[i][1], momentum_list[i][2] );
      /* fprintf(stdout, "# [contract_write_to_aff_file] current aff path = %s\n", aff_buffer_path); */
      affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
      exitstatus = aff_node_put_complex (affw, affdir, aff_buffer+T_global*i, (uint32_t)T_global);
      if(exitstatus != 0) {
        fprintf(stderr, "[contract_write_to_aff_file] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(5);
      }
    }
    free( aff_buffer );
  }  /* if io_proc == 2 */

#ifdef HAVE_MPI
  if ( MPI_Barrier( g_cart_grid ) != MPI_SUCCESS ) {
   fprintf(stderr, "[] Error from MPI_Barrier %s %d\n", __FILE__, __LINE__);
   return(2);
  }
#endif

  retime = _GET_TIME;
  if(io_proc == 2) fprintf(stdout, "# [contract_write_to_aff_file] time for saving momentum space results = %e seconds\n", retime-ratime);

  return(0);

}  /* end of contract_write_to_aff_file */

/********************************************
 * write a contact term to AFF file
 ********************************************/
int cvc_tensor_eo_write_contact_term_to_aff_file ( double *contact_term, struct AffWriter_s*affw, char *tag, int io_proc ) {

  int exitstatus;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char aff_buffer_path[200];
  double _Complex aff_buffer[1];

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[cvc_tensor_eo_write_contact_term_to_aff_file] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }

    for ( int mu = 0; mu < 4; mu++ ) {
      sprintf( aff_buffer_path, "%s/contact_term/mu%d", tag, mu);
      aff_buffer[0] = contact_term[2*mu] + contact_term[2*mu+1] * I;

      affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
      exitstatus = aff_node_put_complex (affw, affdir, aff_buffer, (uint32_t)1 );
      if(exitstatus != 0) {
        fprintf(stderr, "[cvc_tensor_eo_write_contact_term_to_aff_file] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(5);
      }
    }
  }  /* end of if io_proc == 2 */

  return(0);
}  /* end of cvc_tensor_eo_write_contact_term_to_aff_file */
            

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


#if 0

This is not yet correct; for full time dilution on
the odd subspace, one needs to sum over the time projector
indices t-1, t, t+1 to get the even part of the local
loop at time t; this is due to the appearance of
Xbar^+ xi

/************************************************
 *
 ************************************************/
int contract_local_loop_stochastic_clover (double***eo_stochastic_propagator, double***eo_stochastic_source, int nsample,
    int *gid_list, int gid_number,
    int *momentum_list[3], int momentum_number,
    double**mzz, double**mzzinv, double**eo_spinor_work) {

  const unsigned int VOL3half = LX*LY*LZ/2;
  const size_t sizeof_eo_spinor_field_timeslice = _GSI(VOL3half) * sizeof(double);

  double _Complex ***loop = NULL;
  double _Complex **phase = NULL;

  char BLAS_TRANSA, BLAS_TRANSB;
  int BLAS_M, BLAS_K, BLAS_N, BLAS_LDA, BLAS_LDB, BLAS_LDC;
  double _Complex *BLAS_A = NULL, *BLAS_B = NULL, *BLAS_C = NULL;
  double _Complex BLAS_ALPHA = 1.;
  double _Complex BLAS_BETA  = 0.;

  exitstatus = init_3level_zbuffer ( &loop, T, nsample, VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_local_loop_stochastic_clover] Error from init_3level_zbuffer, status was %d\n");
    return(1);
  }

  exitstatus = init_3level_zbuffer ( &ploop, T, momentum_number, nsample );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_local_loop_stochastic_clover] Error from init_3level_zbuffer, status was %d\n");
    return(1);
  }

  exitstatus = init_2level_zbuffer ( &phase, momentum_number, VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_local_loop_stochastic_clover] Error from init_2level_zbuffer, status was %d\n");
    return(1);
  }

  for ( int igamma = 0; igamma < gid_number; igamma++ ) {
    int gid = gid_list[igamma];

    /************************************************
     * odd part 
     ************************************************/
    for ( int isample = 0; isample < nsample; isample++) {
      for ( int it = 0; it < T; it++) {
        unsigned int offset_timeslice = _GSI(VOL3half) * it;
        /* apply gamma gid to stochastic propagator for global timeslice it + g_proc_coords[0]*T for timeslice it */
        spinor_field_eq_gamma_ti_spinor_field ( eo_spinor_work[0], gid, eo_stochastic_propagator[timeslice+g_proc_coords[0]*T][isample]+offset_timeslice, VOL3half );
        g5_phi(eo_spinor_work[0], VOL3half);

        co_field_eq_fv_dag_ti_fv ( (double*)(loop[it][isample]), eo_stochastic_source[it][isample], eo_spinor_work[0], VOL3half );

      }
    }  /* end of loop on stochastic samples */

    for ( int it= 0; it < T; it++ ) {
      int timeslice = it + g_proc_coords[0] * T;
      make_eo_phase_field_timeslice ( phase, momentum_number, momentum_list, timeslice, 0 );

      BLAS_TRANSA = 'T';
      BLAS_TRANSB = 'N';
      BLAS_M     = nsample;
      BLAS_K     = VOL3half;
      BLAS_N     = momentum_number;
      BLAS_A     = loop[it][0];
      BLAS_B     = phase[0];
      BLAS_C     = ploop[it][0];
      BLAS_LDA   = BLAS_K;
      BLAS_LDB   = BLAS_K;
      BLAS_LDC   = BLAS_M;

      _F(zgemm) ( &BLAS_TRANSA, &BLAS_TRANSB, &BLAS_M, &BLAS_N, &BLAS_K, &BLAS_ALPHA, BLAS_A, &BLAS_LDA, BLAS_B, &BLAS_LDB, &BLAS_BETA, BLAS_C, &BLAS_LDC,1,1);

    }  /* end of loop on timeslices */

    /************************************************/
    /************************************************/

    /************************************************
     * even part 
     ************************************************/

  }  /* end of loop on vertex gamma */

  fini_3level_zbuffer ( &loop );
  fini_3level_zbuffer ( &ploop );
  fini_2level_zbuffer ( &phase);


}  /* end of contract_local_loop_stochastic_clover */
#endif  /* of if 0 */


/***********************************************************
 * eo-prec contractions for local - local 2-point function
 *
 * NOTE: neither conn_e nor conn_o nor contact_term 
 *       are initialized to zero here
 ***********************************************************/
int contract_local_local_2pt_eo ( double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o, 
    int *gamma_sink_list, int gamma_sink_num, int*gamma_source_list, int gamma_source_num, int (*momentum_list)[3], int momentum_number,  struct AffWriter_s*affw, char*tag,
    int io_proc ) {
  
  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3 = LX * LY * LZ;
  const unsigned int VOL3half = VOL3 / 2;

  int exitstatus;
  double ratime, retime;
  double **conn_e = NULL, **conn_o = NULL, **conn_lexic = NULL;
  double **conn_p = NULL;
  char aff_tag[200];

  /* auxilliary fermion propagator fields ( without halo ) */
  fermion_propagator_type *fp_S_e = create_fp_field( Vhalf );
  fermion_propagator_type *fp_S_o = create_fp_field( Vhalf );
  fermion_propagator_type *fp_T_e = create_fp_field( Vhalf );
  fermion_propagator_type *fp_T_o = create_fp_field( Vhalf );
  fermion_propagator_type *fp_aux = create_fp_field( Vhalf );

  /**********************************************************
   **********************************************************
   **
   ** contractions
   **
   **********************************************************
   **********************************************************/  
  ratime = _GET_TIME;

  exitstatus = init_2level_buffer ( &conn_e, T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_local_local_2pt_eo] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_buffer ( &conn_o, T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_local_local_2pt_eo] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_buffer ( &conn_lexic, T, 2*VOL3 );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_local_local_2pt_eo] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_buffer ( &conn_p, momentum_number, 2*T );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_local_local_2pt_eo] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }
 
  /* fp_S_e = sprop^e */
  exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_S_e, sprop_list_e, Vhalf);
  /* fp_S_o = sprop^o */
  exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_S_o, sprop_list_o, Vhalf);
  /* fp_T_e = tprop^e */
  exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_T_e, tprop_list_e, Vhalf);
  /* fp_T_o = tprop^o */
  exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_T_o, tprop_list_o, Vhalf);

  /* loop on gamma structures at sink */
  for ( int idsink = 0; idsink < gamma_sink_num; idsink++ ) {
    /* loop on gamma structures at source */
    for ( int idsource = 0; idsource < gamma_source_num; idsource++ ) {

      memset( conn_e[0], 0, 2*Vhalf*sizeof(double) );
      memset( conn_o[0], 0, 2*Vhalf*sizeof(double) );

      /**********************************************************
       * even part
       **********************************************************/
      /* fp_aux <- Gamma_f x fp_T_e */
      fermion_propagator_field_eq_gamma_ti_fermion_propagator_field (fp_aux, gamma_sink_list[idsink], fp_T_e, Vhalf );
      /* fp_aux <- fp_aux x Gamma_i */
      fermion_propagator_field_eq_fermion_propagator_field_ti_gamma (fp_aux, gamma_source_list[idsource], fp_aux, Vhalf );
      /* contract g5 fp_S_e^+ g5 fp_aux = g5 S^e^+ g5 Gamma_f T^e Gamma_i */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field ( (complex *)(conn_e[0]), fp_S_e, fp_aux, -1., Vhalf);

      /**********************************************************
       * odd part
       **********************************************************/
      /* fp_aux <- Gamma_f x fp_T_o */
      fermion_propagator_field_eq_gamma_ti_fermion_propagator_field (fp_aux, gamma_sink_list[idsink], fp_T_o, Vhalf );
      /* fp_aux <- fp_aux x Gamma_i */
      fermion_propagator_field_eq_fermion_propagator_field_ti_gamma (fp_aux, gamma_source_list[idsource], fp_aux, Vhalf );
      /* contract g5 fp_S_o^+ g5 fp_aux = g5 S^o^+ g5 Gamma_f T^o Gamma_i */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field ( (complex *)(conn_o[0]), fp_S_o, fp_aux, -1., Vhalf);

      /**********************************************************
       * Fourier transform
       **********************************************************/
      complex_field_eo2lexic ( conn_lexic[0], conn_e[0], conn_o[0] );

      exitstatus = momentum_projection ( conn_lexic[0], conn_p[0], T, momentum_number, momentum_list);
      if(exitstatus != 0) {
        fprintf(stderr, "[contract_local_local_2pt_eo] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(3);
      }

      sprintf(aff_tag, "%s/gf%.2d/gi%.2d", tag, gamma_sink_list[idsink], gamma_source_list[idsource] );
      exitstatus = contract_write_to_aff_file ( conn_p, affw, aff_tag, momentum_list, momentum_number, io_proc );
      if(exitstatus != 0) {
        fprintf(stderr, "[contract_local_local_2pt_eo] Error from contract_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(3);
      }

    }  /* end of loop on Gamma_i */
  }  /* end of loop on Gamma_f */

  /* free auxilliary fields */
  free_fp_field(&fp_aux);
  free_fp_field(&fp_S_e);
  free_fp_field(&fp_S_o);
  free_fp_field(&fp_T_e);
  free_fp_field(&fp_T_o);

  fini_2level_buffer ( &conn_e );
  fini_2level_buffer ( &conn_o );
  fini_2level_buffer ( &conn_lexic );
  fini_2level_buffer ( &conn_p );

  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [contract_local_local_2pt_eo] time for contract_cvc_tensor = %e seconds\n", retime-ratime);

  return(0);

}  /* end of contract_local_local_2pt_eo */

/***********************************************************
 * eo-prec contractions for local - cvc 2-point function
 *
 * NOTE: neither conn_e nor conn_o nor contact_term 
 *       are initialized to zero here
 ***********************************************************/
int contract_local_cvc_2pt_eo ( double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o,
    int *gamma_sink_list, int gamma_sink_num, int (*momentum_list)[3], int momentum_number,  struct AffWriter_s*affw, char*tag,
    int io_proc ) {

  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3 = LX * LY * LZ;
  const unsigned int VOL3half = VOL3 / 2;

  int exitstatus;
  double ratime, retime;
  double **conn_e = NULL, **conn_o = NULL, **conn_lexic = NULL;
  double **conn_p = NULL;
  char aff_tag[200];

  /* auxilliary fermion propagator fields ( without halo ) */
  fermion_propagator_type *fp_aux     = create_fp_field( Vhalf );
  fermion_propagator_type *fp_S_e     = create_fp_field( Vhalf );
  fermion_propagator_type *fp_S_o     = create_fp_field( Vhalf );
  fermion_propagator_type *fp_T_e     = create_fp_field( Vhalf );
  fermion_propagator_type *fp_T_o     = create_fp_field( Vhalf );
  fermion_propagator_type *fp_S_e_mu  = create_fp_field( Vhalf );
  fermion_propagator_type *fp_S_o_mu  = create_fp_field( Vhalf );
  fermion_propagator_type *fp_T_e_mu  = create_fp_field( Vhalf );
  fermion_propagator_type *fp_T_o_mu  = create_fp_field( Vhalf );

  /**********************************************************
   **********************************************************
   **
   ** contractions
   **
   **********************************************************
   **********************************************************/  
  ratime = _GET_TIME;

  exitstatus = init_2level_buffer ( &conn_e, T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_local_cvc_2pt_eo] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_buffer ( &conn_o, T, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_local_cvc_2pt_eo] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_buffer ( &conn_lexic, T, 2*VOL3 );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_local_cvc_2pt_eo] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  exitstatus = init_2level_buffer ( &conn_p, momentum_number, 2*T );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_local_cvc_2pt_eo] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }
 
  /* fp_S_e = sprop^e */
  exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_S_e, &(sprop_list_e[48]), Vhalf);
  /* fp_S_o = sprop^o */
  exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_S_o, &(sprop_list_o[48]), Vhalf);
  /* fp_T_e = tprop^e */
  exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_T_e, &(tprop_list_e[48]), Vhalf);
  /* fp_T_o = tprop^o */
  exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_T_o, &(tprop_list_o[48]), Vhalf);

  /* loop on vetor index mu at source */
  for ( int mu = 0; mu < 4; mu++ ) {

    exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_aux, &(sprop_list_e[mu*12]), Vhalf);
    apply_propagator_constant_cvc_vertex ( fp_S_e_mu, fp_aux, mu, 1, Usource[mu], Vhalf );

    exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_aux, &(sprop_list_o[mu*12]), Vhalf);
    apply_propagator_constant_cvc_vertex ( fp_S_o_mu, fp_aux, mu, 1, Usource[mu], Vhalf );

    exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_aux, &(tprop_list_e[mu*12]), Vhalf);
    apply_propagator_constant_cvc_vertex ( fp_T_e_mu, fp_aux, mu, 1, Usource[mu], Vhalf );

    exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_aux, &(tprop_list_o[mu*12]), Vhalf);
    apply_propagator_constant_cvc_vertex ( fp_T_o_mu, fp_aux, mu, 1, Usource[mu], Vhalf );

    /* loop on gamma structures at sink */
    for ( int idsink = 0; idsink < gamma_sink_num; idsink++ ) {

      memset( conn_e[0], 0, 2*Vhalf*sizeof(double) );
      memset( conn_o[0], 0, 2*Vhalf*sizeof(double) );

      /**********************************************************
       * even part
       **********************************************************/
   
      /* fp_aux <- Gamma_f x fp_T_e */
      fermion_propagator_field_eq_gamma_ti_fermion_propagator_field (fp_aux, gamma_sink_list[idsink], fp_T_e, Vhalf );
      /* contract g5 fp_aux^+ g5 fp_aux2 = g5 ( fp_S_e Gamma_cvc )^+ g5 Gamma_f x fp_T_e */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field ( (complex *)(conn_e[0]), fp_S_e_mu, fp_aux, 1., Vhalf);

      apply_propagator_constant_cvc_vertex ( fp_aux, fp_T_e, mu, 1, Usource[mu], Vhalf );
      /* fp_aux <- Gamma_f x fp_T_e_mu */
      fermion_propagator_field_eq_gamma_ti_fermion_propagator_field (fp_aux, gamma_sink_list[idsink], fp_T_e_mu, Vhalf );
      /* contract g5 fp_S_e^+ g5 fp_aux = g5 S^e^+ g5 Gamma_f T^e Gamma_cvc */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field ( (complex *)(conn_e[0]), fp_S_e, fp_aux, -1., Vhalf);

      /**********************************************************
       * odd part
       **********************************************************/
      /* fp_aux <- Gamma_f x fp_T_o */
      fermion_propagator_field_eq_gamma_ti_fermion_propagator_field (fp_aux, gamma_sink_list[idsink], fp_T_o, Vhalf );
      /* contract g5 ( fp_S_o Gamma_cvc)^+ g5 fp_T_o */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field ( (complex *)(conn_o[0]), fp_S_o_mu, fp_aux, 1., Vhalf);

      /* fp_aux <- Gamma_f x fp_T_o_mu */
      fermion_propagator_field_eq_gamma_ti_fermion_propagator_field (fp_aux, gamma_sink_list[idsink], fp_T_o_mu, Vhalf );
      /* contract g5 fp_S_o^+ g5 fp_aux = g5 S^o^+ g5 Gamma_f T^o Gamma_cvc */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field ( (complex *)(conn_o[0]), fp_S_o, fp_aux, -1., Vhalf);

      /**********************************************************
       * Fourier transform
       **********************************************************/
      complex_field_eo2lexic ( conn_lexic[0], conn_e[0], conn_o[0] );

      exitstatus = momentum_projection ( conn_lexic[0], conn_p[0], T, momentum_number, momentum_list);
      if(exitstatus != 0) {
        fprintf(stderr, "[contract_local_cvc_2pt_eo] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(3);
      }

      sprintf(aff_tag, "%s/gf%.2d/mu%.2d", tag, gamma_sink_list[idsink], mu );
      exitstatus = contract_write_to_aff_file ( conn_p, affw, aff_tag, momentum_list, momentum_number, io_proc );
      if(exitstatus != 0) {
        fprintf(stderr, "[contract_local_cvc_2pt_eo] Error from contract_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(3);
      }

    }  /* end of loop on mu */
  }  /* end of loop on Gamma_f */

  /* free auxilliary fields */
  free_fp_field( &fp_aux );
  free_fp_field( &fp_S_e );
  free_fp_field( &fp_S_o );
  free_fp_field( &fp_T_e );
  free_fp_field( &fp_T_o );
  free_fp_field( &fp_S_e_mu );
  free_fp_field( &fp_S_o_mu );
  free_fp_field( &fp_T_e_mu );
  free_fp_field( &fp_T_o_mu );

  fini_2level_buffer ( &conn_e );
  fini_2level_buffer ( &conn_o );
  fini_2level_buffer ( &conn_lexic );
  fini_2level_buffer ( &conn_p );

  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [contract_local_cvc_2pt_eo] time for contract_cvc_tensor = %e seconds\n", retime-ratime);

  return(0);

}  /* end of contract_local_cvc_2pt_eo */

/***********************************************************
 * reduction
 *
 * w += tr ( r )
 ***********************************************************/
void co_field_pl_eq_tr_propagator_field (complex *w, fermion_propagator_type *r, double sign, unsigned int N) {

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  complex *w_ = NULL, wtmp;
  fermion_propagator_type r_=NULL;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( unsigned int ix=0; ix<N; ix++ ) {
    w_ = w + ix;
    r_ = r[ix];
    _co_eq_tr_fp ( &wtmp, r_ );
    _co_pl_eq_co_ti_re(w_, &wtmp, sign);
  }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of co_field_pl_eq_tr_propagator_field */

/***********************************************************
 * reduction
 *
 * w += tr ( r )^*
 ***********************************************************/
void co_field_pl_eq_tr_propagator_field_conj (complex *w, fermion_propagator_type *r, double sign, unsigned int N) {

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  complex *w_ = NULL, wtmp;
  fermion_propagator_type r_=NULL;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for( unsigned int ix=0; ix<N; ix++ ) {
    w_ = w + ix;
    r_ = r[ix];
    _co_eq_tr_fp ( &wtmp, r_ );
    _co_pl_eq_co_conj_ti_re(w_, &wtmp, sign);
  }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of co_field_pl_eq_tr_propagator_field */


/***********************************************************
 *
 ***********************************************************/
void contract_cvc_loop_eo ( double ***loop, double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o , double*gauge_field ) {

  const unsigned int Vhalf = VOLUME / 2;
  int exitstatus;

  fermion_propagator_type *fp       = create_fp_field( (VOLUME+RAND)/2 );
  fermion_propagator_type *gamma_fp = create_fp_field( Vhalf );
  

  /* even part of sprop */
  exitstatus = assign_fermion_propagaptor_from_spinor_field ( fp, sprop_list_e, Vhalf);
  for ( int mu = 0; mu < 4; mu++ ) {
    /* input field is even, output field is odd */
    apply_cvc_vertex_propagator_eo ( gamma_fp, fp, mu, 0, gauge_field, 1);
    co_field_pl_eq_tr_propagator_field ( (complex*)loop[1][mu], gamma_fp, 1., Vhalf);
  }

  /* odd part of sprop */
  exitstatus = assign_fermion_propagaptor_from_spinor_field ( fp, sprop_list_o, Vhalf);
  for ( int mu = 0; mu < 4; mu++ ) {
    /* input field is odd, output field is even */
    apply_cvc_vertex_propagator_eo ( gamma_fp, fp, mu, 0, gauge_field, 0);
    co_field_pl_eq_tr_propagator_field ( (complex*)loop[0][mu], gamma_fp, 1., Vhalf);
  }

  /* even part of tprop */
  exitstatus = assign_fermion_propagaptor_from_spinor_field ( fp, tprop_list_e, Vhalf);
  for ( int mu = 0; mu < 4; mu++ ) {
    /* input field is even, output field is odd */
    apply_cvc_vertex_propagator_eo ( gamma_fp, fp, mu, 0, gauge_field, 1);
    co_field_pl_eq_tr_propagator_field_conj ( (complex*)loop[1][mu], gamma_fp, -1., Vhalf);
  }

  /* odd part of tprop */
  exitstatus = assign_fermion_propagaptor_from_spinor_field ( fp, tprop_list_o, Vhalf);
  for ( int mu = 0; mu < 4; mu++ ) {
    /* input field is odd, output field is even */
    apply_cvc_vertex_propagator_eo ( gamma_fp, fp, mu, 0, gauge_field, 0);
    co_field_pl_eq_tr_propagator_field_conj ( (complex*)loop[0][mu], gamma_fp, -1., Vhalf);
  }
#if 0
#endif  /* of if 0 */

#if 0
  const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  double **eo_spinor_work = NULL, **eo_spinor_field = NULL;

  exitstatus = init_2level_buffer ( &eo_spinor_work, 1, _GSI( (VOLUME+RAND)/2 ) );
  if ( exitstatus != 0 ) {
    fprintf(stdout, "[contract_cvc_loop_eo] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }
  exitstatus = init_2level_buffer ( &eo_spinor_field, 12, _GSI( Vhalf ) );
  if ( exitstatus != 0 ) {
    fprintf(stdout, "[contract_cvc_loop_eo] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  /* even part of sprop */
  for ( int mu = 0; mu < 4; mu++ ) {
    for ( int i = 0; i < 12; i++) {
      memcpy ( eo_spinor_work[0], sprop_list_e[i], sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( eo_spinor_field[i], eo_spinor_work[0], mu, 0, gauge_field, 1);
    }
    exitstatus = assign_fermion_propagaptor_from_spinor_field ( gamma_fp, eo_spinor_field, Vhalf);
    co_field_pl_eq_tr_propagator_field ( (complex*)loop[1][mu], gamma_fp, 1., Vhalf);
  }

  /* odd part of sprop */
  for ( int mu = 0; mu < 4; mu++ ) {
    for ( int i = 0; i < 12; i++) {
      memcpy ( eo_spinor_work[0], sprop_list_o[i], sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( eo_spinor_field[i], eo_spinor_work[0], mu, 0, gauge_field, 0);
    }
    exitstatus = assign_fermion_propagaptor_from_spinor_field ( gamma_fp, eo_spinor_field, Vhalf);
    co_field_pl_eq_tr_propagator_field ( (complex*)loop[0][mu], gamma_fp, 1., Vhalf);
  }

  /* even part of tprop */
  for ( int mu = 0; mu < 4; mu++ ) {
    for ( int i = 0; i < 12; i++) {
      memcpy ( eo_spinor_work[0], tprop_list_e[i], sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( eo_spinor_field[i], eo_spinor_work[0], mu, 0, gauge_field, 1);
    }
    exitstatus = assign_fermion_propagaptor_from_spinor_field ( gamma_fp, eo_spinor_field, Vhalf);
    co_field_pl_eq_tr_propagator_field_conj ( (complex*)loop[1][mu], gamma_fp, -1., Vhalf);
  }

  /* odd part of tprop */
  for ( int mu = 0; mu < 4; mu++ ) {
    for ( int i = 0; i < 12; i++) {
      memcpy ( eo_spinor_work[0], tprop_list_o[i], sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( eo_spinor_field[i], eo_spinor_work[0], mu, 0, gauge_field, 0);
    }
    exitstatus = assign_fermion_propagaptor_from_spinor_field ( gamma_fp, eo_spinor_field, Vhalf);
    co_field_pl_eq_tr_propagator_field_conj ( (complex*)loop[0][mu], gamma_fp, -1., Vhalf);
  }

  fini_2level_buffer ( &eo_spinor_work );
  fini_2level_buffer ( &eo_spinor_field );
#endif  /* of if 0 */

  free_fp_field(&fp);
  free_fp_field(&gamma_fp);
  return;

}  /* end of contract_cvc_loop_eo */


/***********************************************************
 *
 ***********************************************************/
void contract_cvc_loop_eo_lma ( double ***loop, double**eo_evecs_field, double *eo_evecs_norm, int nev, double*gauge_field, double **mzz[2], double **mzzinv[2]) {

  const unsigned int Vhalf = VOLUME / 2;
  const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  int exitstatus;

  double ratime, retime;
  double **eo_spinor_work = NULL, **eo_spinor_field = NULL;
  double *v = NULL, *w = NULL, *xv = NULL, *xw = NULL;

  ratime= _GET_TIME;

  exitstatus = init_2level_buffer ( &eo_spinor_work, 2, _GSI( (VOLUME+RAND)/2 ) );
  if ( exitstatus != 0 ) {
    fprintf(stdout, "[contract_cvc_loop_eo] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }
  exitstatus = init_2level_buffer ( &eo_spinor_field, 4, _GSI( Vhalf ) );
  if ( exitstatus != 0 ) {
    fprintf(stdout, "[contract_cvc_loop_eo] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }
  v  = eo_spinor_field[0];
  xv = eo_spinor_field[1];
  w  = eo_spinor_field[2];
  xw = eo_spinor_field[3];

  /* loop on eigenvectors */
  for ( int i = 0; i < nev; i++) {
    /* V */
    memcpy ( v, eo_evecs_field[i], sizeof_eo_spinor_field );

    /* double norm;
    spinor_scalar_product_re( &norm, v, v, Vhalf);
    fprintf(stdout, "# [contract_cvc_loop_eo_lma] V norm %3d %25.16e\n", i, norm); */

    /* Xbar V */
    memcpy ( eo_spinor_work[0], v, sizeof_eo_spinor_field );
    X_clover_eo ( xv, eo_spinor_work[0], gauge_field, mzzinv[1][0]);

    /* W from V and Xbar V */
    memcpy ( w, v,  sizeof_eo_spinor_field );
    memcpy ( eo_spinor_work[0],  xv, sizeof_eo_spinor_field );
    C_clover_from_Xeo ( w, eo_spinor_work[0], eo_spinor_work[1], gauge_field, mzz[1][1]);
    spinor_field_ti_eq_re ( w, eo_evecs_norm[i], Vhalf);

    /* spinor_scalar_product_re( &norm, w, w, Vhalf);
    norm /= sqrt(eo_evecs_norm[i]);
    fprintf(stdout, "# [contract_cvc_loop_eo_lma] W norm %3d %25.16e\n", i, norm); */


    /* X W from W */
    memcpy ( eo_spinor_work[0], w, sizeof_eo_spinor_field );
    X_clover_eo ( xw, eo_spinor_work[0], gauge_field, mzzinv[0][0]);

    for ( int mu = 0; mu < 4; mu++ ) {

      /* V^+ g5 Gamma_mu^f X W */
      memcpy ( eo_spinor_work[0], xw, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( eo_spinor_work[1], eo_spinor_work[0], mu, 0, gauge_field, 1);
      g5_phi( eo_spinor_work[1], Vhalf);
      co_field_pl_eq_fv_dag_ti_fv ( loop[1][mu], v, eo_spinor_work[1], Vhalf );

      /* (Xbar V)^+ g5 Gamma_mu^f W */
      memcpy ( eo_spinor_work[0], w, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( eo_spinor_work[1], eo_spinor_work[0], mu, 0, gauge_field, 0);
      g5_phi( eo_spinor_work[1], Vhalf);
      co_field_pl_eq_fv_dag_ti_fv ( loop[0][mu], xv, eo_spinor_work[1], Vhalf );

      /* W^+ g5 Gamma_mu^f ( Xbar V ) */
      memcpy ( eo_spinor_work[0], xv, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( eo_spinor_work[1], eo_spinor_work[0], mu, 0, gauge_field, 1);
      g5_phi( eo_spinor_work[1], Vhalf);
      co_field_mi_eq_fv_dag_ti_fv ( loop[1][mu], eo_spinor_work[1], w, Vhalf );

      /* (X W)^+ g5 Gamma_mu^f V */
      memcpy ( eo_spinor_work[0], v, sizeof_eo_spinor_field );
      apply_cvc_vertex_eo( eo_spinor_work[1], eo_spinor_work[0], mu, 0, gauge_field, 0);
      g5_phi( eo_spinor_work[1], Vhalf);
      co_field_mi_eq_fv_dag_ti_fv ( loop[0][mu], eo_spinor_work[1], xw, Vhalf );

    }  /* end of loop on mu */

  }  /* end of loop on eigenvectors */

  fini_2level_buffer ( &eo_spinor_work );
  fini_2level_buffer ( &eo_spinor_field );

  retime = _GET_TIME;
  if (g_cart_id == 0 ) fprintf(stdout, "# [contract_cvc_loop_eo_lma] time for contract_cvc_loop_eo_lma_wi = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);

  return;

}  /* end of contract_cvc_loop_eo_lma */

/***********************************************************
 *
 ***********************************************************/
void contract_cvc_loop_eo_lma_wi ( double **wi, double**eo_evecs_field, double *eo_evecs_norm, int nev, double*gauge_field, double **mzz[2], double **mzzinv[2]) {

  const unsigned int Vhalf = VOLUME / 2;
  const size_t sizeof_eo_spinor_field = _GSI( Vhalf ) * sizeof(double);
  int exitstatus;

  double ratime, retime;
  double **eo_spinor_work = NULL, **eo_spinor_field = NULL, *v = NULL, *w = NULL;

  ratime = _GET_TIME;

  exitstatus = init_2level_buffer ( &eo_spinor_work, 2, _GSI( (VOLUME+RAND)/2 ) );
  if ( exitstatus != 0 ) {
    fprintf(stdout, "[contract_cvc_loop_eo_lma_wi] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(1);
  }
  exitstatus = init_2level_buffer ( &eo_spinor_field, 2, _GSI( Vhalf ) );
  if ( exitstatus != 0 ) {
    fprintf(stdout, "[contract_cvc_loop_eo_lma_wi] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    EXIT(2);
  }

  v = eo_spinor_field[0];
  w = eo_spinor_field[1];

  /* even part */
  /* memset ( wi[0], 0, Vhalf*2*sizeof(double) ); */
  /* memset ( wi[1], 0, Vhalf*2*sizeof(double) ); */
 
  /* odd part */
  for ( int i = 0; i < nev; i++) {
    /* V */
    memcpy ( v, eo_evecs_field[i], sizeof_eo_spinor_field );

    /* W = Cbar V */
    C_clover_oo ( w, v, gauge_field, eo_spinor_work[0], mzz[1][1], mzzinv[1][0]);
    spinor_field_ti_eq_re ( w, sqrt( eo_evecs_norm[i] ), Vhalf);

    /* double norm;
    spinor_scalar_product_re( &norm, w, w, Vhalf);
    fprintf(stdout, "# [contract_cvc_loop_eo_lma_wi] Wtilde norm %3d %25.16e\n", i, norm);*/

    co_field_mi_eq_fv_dag_ti_fv ( wi[1] , w, w, Vhalf );
    co_field_pl_eq_fv_dag_ti_fv ( wi[1] , v, v, Vhalf );

  }

  fini_2level_buffer ( &eo_spinor_work );
  fini_2level_buffer ( &eo_spinor_field );

  retime = _GET_TIME;
  if (g_cart_id == 0 ) fprintf(stdout, "# [contract_cvc_loop_eo_lma_wi] time for contract_cvc_loop_eo_lma_wi = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);

  return;
}  /* contract_cvc_loop_eo_lma_wi */

/***************************************************************************
 * check position space WI for a loop
 ***************************************************************************/
int cvc_loop_eo_check_wi_position_space_lma ( double ***wwi, double ***loop_lma, double **eo_evecs_field, double *evecs_norm, int nev, double *gauge_field, double **mzz[2], double **mzzinv[2]  ) {

  const unsigned int Vhalf = VOLUME / 2;

  int exitstatus;
  double **conn_buffer = NULL, norm_accum = 0.;
  double **wi = NULL;
#ifdef HAVE_OPENMP
  omp_lock_t writelock;
#endif

  exitstatus = init_2level_buffer ( &conn_buffer, 4, 2*(VOLUME+RAND) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[cvc_loop_eo_check_wi_position_space_lma] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  if ( *wwi == NULL ) {
    if ( g_cart_id == 0 ) fprintf(stdout, "# [cvc_loop_eo_check_wi_position_space_lma] allocating new wi contraction field\n");
    exitstatus = init_2level_buffer ( &wi, 2, 2*Vhalf );
    if ( exitstatus != 0 ) {
      fprintf(stderr, "[cvc_loop_eo_check_wi_position_space_lma] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(1);
    }
    *wwi = wi;
  } else {
    if ( g_cart_id == 0 ) fprintf(stdout, "# [cvc_loop_eo_check_wi_position_space_lma] using existing wi contraction field\n");
    wi = *wwi;
  }

  contract_cvc_loop_eo_lma_wi ( wi, eo_evecs_field, evecs_norm, nev, gauge_field, mzz, mzzinv );

  for ( int mu = 0; mu < 4; mu++ ) {
    complex_field_eo2lexic (conn_buffer[mu], loop_lma[0][mu], loop_lma[1][mu] );
#ifdef HAVE_MPI
    xchange_contraction( conn_buffer[mu], 2 );
#endif
  }

#ifdef HAVE_OPENMP
  omp_init_lock(&writelock);
#pragma omp parallel shared( norm_accum, conn_buffer, wi )
{
#endif
  double normt = 0.;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for ( unsigned int ix = 0; ix < VOLUME; ix++ ) {
    int ieo = g_iseven[ix] == 1 ? 0 : 1;
    unsigned int ixeosub = g_lexic2eosub[ix];
    double dnormr = conn_buffer[0][2*ix] 
                  + conn_buffer[1][2*ix] 
                  + conn_buffer[2][2*ix] 
                  + conn_buffer[3][2*ix]
                  - conn_buffer[0][2*g_idn[ix][0]]
                  - conn_buffer[1][2*g_idn[ix][1]]
                  - conn_buffer[2][2*g_idn[ix][2]]
                  - conn_buffer[3][2*g_idn[ix][3]] - wi[ieo][2*ixeosub];

    double dnormi = conn_buffer[0][2*ix+1]
                  + conn_buffer[1][2*ix+1]   
                  + conn_buffer[2][2*ix+1]  
                  + conn_buffer[3][2*ix+1]
                  - conn_buffer[0][2*g_idn[ix][0]+1]
                  - conn_buffer[1][2*g_idn[ix][1]+1]
                  - conn_buffer[2][2*g_idn[ix][2]+1]
                  - conn_buffer[3][2*g_idn[ix][3]+1] - wi[ieo][2*ixeosub+1]; 


    if ( g_verbose > 4 ) fprintf(stdout, "# [cvc_loop_eo_check_wi_position_space_lma] proc%.4d %3d %3d %3d %3d\t\t\t%25.16e %25.16e\n",
        g_cart_id, ix/(LX*LY*LZ) + g_proc_coords[0]*T, (ix%(LX*LY*LZ))/(LY*LZ) + g_proc_coords[1]*LX,
        (ix%(LY*LZ))/LZ + g_proc_coords[2]*LY, ix%LZ + g_proc_coords[3]*LZ, dnormr, dnormi );
      normt += dnormr * dnormr + dnormi * dnormi;
  }
#ifdef HAVE_OPENMP
  omp_set_lock(&writelock);
#endif
  norm_accum += normt;

#ifdef HAVE_OPENMP
  omp_unset_lock(&writelock);
}  /* end of parallel region */
  omp_destroy_lock(&writelock);
#endif

  norm_accum = sqrt ( norm_accum );
#ifdef HAVE_MPI
  double dtmp = norm_accum;
  exitstatus = MPI_Allreduce(&dtmp, &norm_accum, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid);
  if(exitstatus != MPI_SUCCESS) {
    fprintf(stderr, "[cvc_loop_eo_check_wi_position_space_lma] Error from MPI_Allreduce, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }
#endif
  if (g_cart_id == 0) fprintf(stdout, "# [cvc_loop_eo_check_wi_position_space_lma] WI %25.16e\n", norm_accum );

  /* fini_2level_buffer ( &wi ); */
  fini_2level_buffer ( &conn_buffer );

  return(0);
}  /* end of cvc_loop_eo_check_wi_position_space_lma */


/***************************************************************************
 * momentum projections
 ***************************************************************************/

int cvc_loop_eo_momentum_projection (double****loop_tp, double***loop_eo, int (*momentum_list)[3], int momentum_number) {

  int exitstatus;
  double ***cvc_tp = NULL, *cvc_loop_lexic=NULL;
  double ratime, retime;

  ratime = _GET_TIME;

  if ( *loop_tp == NULL ) {
    exitstatus = init_3level_buffer( loop_tp, momentum_number, 4, 2*T);
    if(exitstatus != 0) {
      fprintf(stderr, "[cvc_loop_eo_momentum_projection] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(1);
    }
  }
  cvc_tp = *loop_tp;

  cvc_loop_lexic = (double*)malloc( 8 * VOLUME * sizeof(double));
  if( cvc_loop_lexic == NULL ) {
    fprintf(stderr, "[cvc_loop_eo_momentum_projection] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(2);
  }
  for ( int mu = 0; mu < 4; mu++ ) {
    complex_field_eo2lexic ( cvc_loop_lexic+2*mu*VOLUME, loop_eo[0][mu], loop_eo[1][mu] );
  }

  exitstatus = momentum_projection (cvc_loop_lexic, cvc_tp[0][0], T*4, momentum_number, momentum_list);
  if(exitstatus != 0) {
    fprintf(stderr, "[cvc_loop_eo_momentum_projection] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }
  free ( cvc_loop_lexic );

  for ( int ip = 0; ip < momentum_number; ip++) {
    double dtmp[2], phase[2];
    double pvec[3] = { 
      M_PI * momentum_list[ip][0] / (double)LX_global, 
      M_PI * momentum_list[ip][1] / (double)LY_global,
      M_PI * momentum_list[ip][2] / (double)LZ_global };

    for ( int mu = 1; mu < 4; mu++ ) {
      phase[0] = cos ( pvec[mu-1] );
      phase[1] = sin ( pvec[mu-1] );

      for ( int it = 0; it < T; it++ ) {
        dtmp[0] = cvc_tp[ip][mu][2*it  ];
        dtmp[1] = cvc_tp[ip][mu][2*it+1];

        cvc_tp[ip][mu][2*it  ] = dtmp[0] * phase[0] - dtmp[1] * phase[1];
        cvc_tp[ip][mu][2*it+1] = dtmp[0] * phase[1] + dtmp[1] * phase[0];
      }
    }
  }

  retime = _GET_TIME;
  if( g_cart_id == 0 ) fprintf(stdout, "# [cvc_loop_eo_momentum_projection] time for momentum projection = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);

  return(0);
}  /* end of cvc_loop_eo_momentum_projection */


/***************************************************************************
 * write tp-loop results to file
 ***************************************************************************/

int cvc_loop_tp_write_to_aff_file (double***cvc_tp, struct AffWriter_s*affw, char*tag, int (*momentum_list)[3], int momentum_number, int io_proc ) {

  int exitstatus, i;
  double ratime, retime;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char aff_buffer_path[200];
  double *buffer = NULL;
  double _Complex *aff_buffer = NULL;
  double _Complex *zbuffer = NULL;

  if ( io_proc == 2 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[cvc_loop_tp_write_to_aff_file] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }

    zbuffer = (double _Complex*)malloc(  momentum_number * 4 * T_global * sizeof(double _Complex) );
    if( zbuffer == NULL ) {
      fprintf(stderr, "[cvc_loop_tp_write_to_aff_file] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(6);
    }
  }

  ratime = _GET_TIME;

  /* reorder cvc_tp into buffer with order time - munu - momentum */
  buffer = (double*)malloc(  momentum_number * 8 * T * sizeof(double) );
  if( buffer == NULL ) {
    fprintf(stderr, "[cvc_loop_tp_write_to_aff_file] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(6);
  }
  i = 0;
  for( int it = 0; it < T; it++ ) {
    for( int mu=0; mu<4; mu++ ) {
      for( int ip=0; ip < momentum_number; ip++) {
        buffer[i++] = cvc_tp[ip][mu][2*it  ];
        buffer[i++] = cvc_tp[ip][mu][2*it+1];
      }
    }
  }

#ifdef HAVE_MPI
  i = momentum_number * 8 * T;
#  if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ) 
  if(io_proc>0) {
    exitstatus = MPI_Gather(buffer, i, MPI_DOUBLE, zbuffer, i, MPI_DOUBLE, 0, g_tr_comm);
    if(exitstatus != MPI_SUCCESS) {
      fprintf(stderr, "[cvc_loop_tp_write_to_aff_file] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
      return(3);
    }
  }
#  else
  exitstatus = MPI_Gather(buffer, i, MPI_DOUBLE, zbuffer, i, MPI_DOUBLE, 0, g_cart_grid);
  if(exitstatus != MPI_SUCCESS) {
    fprintf(stderr, "[cvc_loop_tp_write_to_aff_file] Error from MPI_Gather, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(4);
  }
#  endif

#else
  memcpy(zbuffer, buffer, momentum_number * 8 * T * sizeof(double) );
#endif
  free( buffer );

  if(io_proc == 2) {

    /* reverse the ordering back to momentum - munu - time */
    aff_buffer = (double _Complex*)malloc( momentum_number * 8 * T_global * sizeof(double _Complex) );
    if(aff_buffer == NULL) {
      fprintf(stderr, "[cvc_loop_tp_write_to_aff_file] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(2);
    }
    i = 0;
    for( int ip=0; ip<momentum_number; ip++) {
      for( int mu=0; mu < 4; mu++ ) {
        for( int it = 0; it < T_global; it++ ) {
          int offset = (it * 4 + mu ) * momentum_number + ip;
          aff_buffer[i++] = zbuffer[offset];
        }
      }
    }
    free( zbuffer );

    for(i=0; i < momentum_number; i++) {
      for ( int mu = 0; mu < 4; mu++ ) {
        sprintf(aff_buffer_path, "%s/px%.2dpy%.2dpz%.2d/mu%d", tag, momentum_list[i][0], momentum_list[i][1], momentum_list[i][2], mu );
        /* fprintf(stdout, "# [cvc_loop_tp_write_to_aff_file] current aff path = %s\n", aff_buffer_path); */
        affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);
        exitstatus = aff_node_put_complex (affw, affdir, aff_buffer+T_global*(4 * i + mu), (uint32_t)T_global);
        if(exitstatus != 0) {
          fprintf(stderr, "[cvc_loop_tp_write_to_aff_file] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(5);
        }
      }  /* end of loop on mu */
    }  /* end of loop on momenta  */
    free( aff_buffer );
  }  /* if io_proc == 2 */

#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif

  retime = _GET_TIME;
  if(io_proc == 2) fprintf(stdout, "# [cvc_loop_tp_write_to_aff_file] time for saving momentum space results = %e seconds\n", retime-ratime);

  return(0);

}  /* end of cvc_loop_tp_write_to_aff_file */

/***************************************************************************
 * check Ward identity in momentum space
 ***************************************************************************/
int cvc_loop_eo_check_wi_momentum_space_lma ( double **wi, double ***loop_lma, int (*momentum_list)[3], int momentum_number  ) {

  int exitstatus;
  double ratime, retime;
  double *wi_lexic = NULL, **wi_tp = NULL;

  ratime = _GET_TIME;

  wi_lexic = (double*)malloc( 2*VOLUME*sizeof(double) );
  if ( wi_lexic == NULL ) {
    fprintf(stderr, "[cvc_loop_eo_check_wi_momentum_space_lma] Error from malloc %s %d\n", __FILE__, __LINE__);
    return(1);
  }
  exitstatus = init_2level_buffer ( &wi_tp, momentum_number, 2*T );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[cvc_loop_eo_check_wi_momentum_space_lma] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  complex_field_eo2lexic ( wi_lexic, wi[0], wi[1] );

  exitstatus = momentum_projection ( wi_lexic, wi_tp[0], T, momentum_number, momentum_list);
  if(exitstatus != 0) {
    fprintf(stderr, "[cvc_loop_eo_check_wi_momentum_space_lma] Error from momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(3);
  }

  free ( wi_lexic );

  for ( int ip = 0; ip < momentum_number; ip++ ) {
    double dtmp[2], phase[2];

    for ( int ip0 = 0; ip0 < T_global; ip0 ++ ) {
      
      double p[4] = {
          M_PI * ip0 / (double)T_global,
          M_PI * momentum_list[ip][0] / (double)LX_global,
          M_PI * momentum_list[ip][1] / (double)LY_global,
          M_PI * momentum_list[ip][2] / (double)LZ_global };

      double sinp[4] = { 2*sin( p[0] ), 2*sin( p[1] ), 2*sin( p[2] ), 2*sin( p[3] ) }; 

      double jjp[8];
      for ( int mu = 0; mu < 4; mu++ ) {
        jjp[2*mu  ] = 0.;
        jjp[2*mu+1] = 0.;
        for ( int it = 0; it < T; it++ ) {
          double phase = p[0] * ( 2 * ( it + g_proc_coords[0]*T) + (int)(mu == 0) );
          double ephase[2] = { cos ( phase ), sin ( phase ) };
          jjp[2*mu  ] += loop_lma[ip][mu][2*it  ] * ephase[0] - loop_lma[ip][mu][2*it+1] * ephase[1];
          jjp[2*mu+1] += loop_lma[ip][mu][2*it+1] * ephase[0] + loop_lma[ip][mu][2*it  ] * ephase[1];
        }
      }
#ifdef HAVE_MPI
      double buffer[8];
      exitstatus = MPI_Allreduce( jjp, buffer, 8, MPI_DOUBLE, MPI_SUM, g_tr_comm );
      if ( exitstatus != MPI_SUCCESS ) {
        fprintf(stderr, "[cvc_loop_eo_check_wi_momentum_space_lma] Error from MPI_Allreduce, status %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(3);
      }
      memcpy ( jjp, buffer, 8*sizeof(double) );
#endif

      double ww[2];
      ww[0] = 0.;
      ww[1] = 0.;
      for ( int it = 0; it < T; it++ ) {
        double phase = p[0] * 2 * ( it + g_proc_coords[0] * T);
        double ephase[2] = { cos ( phase ), sin ( phase ) };
        ww[0] += wi_tp[ip][2*it  ] * ephase[0] - wi_tp[ip][2*it+1] * ephase[1];
        ww[1] += wi_tp[ip][2*it+1] * ephase[0] + wi_tp[ip][2*it  ] * ephase[1];
      }
#ifdef HAVE_MPI
      exitstatus = MPI_Allreduce( ww, buffer, 2, MPI_DOUBLE, MPI_SUM, g_tr_comm );
      if ( exitstatus != MPI_SUCCESS ) {
        fprintf(stderr, "[cvc_loop_eo_check_wi_momentum_space_lma] Error from MPI_Allreduce, status %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(3);
      }
      ww[0] = buffer[0];
      ww[1] = buffer[1];
#endif

      double pJi = -( sinp[0] * jjp[0] + sinp[1] * jjp[2] + sinp[2] * jjp[4] + sinp[3] * jjp[6] ); 
      double pJr =    sinp[0] * jjp[1] + sinp[1] * jjp[3] + sinp[2] * jjp[5] + sinp[3] * jjp[7]; 

      if ( g_cart_id == 0 ) {
        fprintf(stdout, "# [cvc_loop_eo_check_wi_momentum_space_lma] p = %3d %3d %3d %3d pJ = %25.16e %25.16e    ww = %25.16e %25.16e\n", 
            ip0, momentum_list[ip][0], momentum_list[ip][1], momentum_list[ip][2],
            pJr, pJi, ww[0], ww[1] );
      }

    }  /* end of loop on ip0  */

  }  /* end of loop on momenta */

  fini_2level_buffer ( &wi_tp );

  retime = _GET_TIME;
  if( g_cart_id == 0 ) fprintf(stdout, "# [cvc_loop_eo_check_wi_momentum_space_lma] time for saving momentum space results = %e seconds\n", retime-ratime);

  return(0);
}  /* end of cvc_loop_eo_check_wi_momentum_space_lma */


/***********************************************************
 * contractions for eo-precon lm cvc - cvc tensor
 *
 * NOTE:
 *
 ***********************************************************/
int contract_cvc_tensor_eo_lm_factors ( double**eo_evecs_field, int nev, double*gauge_field, double **mzz[2], double **mzzinv[2],
    struct AffWriter_s **affw, char*tag, 
    int (*momentum_list)[3], int momentum_number, int io_proc, int block_length ) {

  const unsigned int Vhalf = VOLUME / 2;
  const unsigned int VOL3half = ( LX * LY * LZ ) / 2;
  const size_t sizeof_eo_spinor_field           = _GSI( Vhalf    ) * sizeof(double);
  const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double);


  int exitstatus;

  double **v = NULL, **xv = NULL, ***eo_block_field = NULL, **w = NULL, **xw = NULL;
  double ***contr_x = NULL, **eo_spinor_work = NULL;
  double _Complex ***contr_p = NULL;
  char aff_tag[200];

  int block_number = nev / block_length;
  if (io_proc == 2 && g_verbose > 3 ) {
    fprintf(stdout, "# [contract_cvc_tensor_eo_lm_factors] number of blocks = %d\n", block_length);
  }
  if ( nev - block_number * block_length != 0 ) {
    if ( io_proc == 2 ) fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error, nev not divisible by block_length\n");
    return(1);
  }

  /***********************************************************
   * auxilliary eo spinor fields with halo
   ***********************************************************/
  exitstatus = init_2level_buffer ( &eo_spinor_work, 4, _GSI( (VOLUME+RAND)/2 )  );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  /***********************************************************
   * set V
   ***********************************************************/
  v = eo_evecs_field;

  /***********************************************************
   * XV
   ***********************************************************/
  exitstatus = init_2level_buffer ( &xv, nev, _GSI( Vhalf )  );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  exitstatus = init_3level_buffer ( &contr_x, nev, block_length, 2*VOL3half );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  exitstatus = init_3level_zbuffer ( &contr_p, momentum_number, nev, block_length );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from init_3level_zbuffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  /***********************************************************
   * set xv
   *
   * Xbar using Mbar_{ee}^{-1}, i.e. mzzinv[1][0] and 
   ***********************************************************/
  for ( int iev = 0; iev < nev; iev++ ) {
    memcpy( eo_spinor_work[0], v[iev], sizeof_eo_spinor_field );
    X_clover_eo ( xv[iev], eo_spinor_work[0], gauge_field, mzzinv[1][0] );
  }  /* end of loop on eigenvectors */

  /***********************************************************
   * auxilliary block field
   ***********************************************************/
  exitstatus = init_3level_buffer ( &eo_block_field, 4, block_length, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from init_3level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  /***********************************************************
   * W block field
   ***********************************************************/
  exitstatus = init_2level_buffer ( &w, block_length, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  /***********************************************************
   * XW block field
   ***********************************************************/
  exitstatus = init_2level_buffer ( &xw, block_length, _GSI(Vhalf) );
  if ( exitstatus != 0 ) {
    fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(2);
  }

  /***********************************************************
   * loop on evec blocks
   ***********************************************************/
  for ( int iblock = 0; iblock < block_number; iblock++ ) {

    /***********************************************************
     * calculate w, xw for the current block,
     * w = Cbar ( v, xv )
     * xw = X w
     ***********************************************************/
    for ( int iev = 0; iev < block_length; iev++ ) {
      memcpy( w[iev], v[iblock*block_length+iev], sizeof_eo_spinor_field );
      memcpy( eo_spinor_work[0],  xv[iblock*block_length+iev], sizeof_eo_spinor_field );
      C_clover_from_Xeo ( w[iev], eo_spinor_work[0], eo_spinor_work[1], gauge_field, mzz[1][1] );

      memcpy( eo_spinor_work[0],  w[iev], sizeof_eo_spinor_field );
      X_clover_eo ( xw[iev], w[iev], gauge_field, mzzinv[0][0] );
    }

    /***********************************************************
     * loop vector index mu
     ***********************************************************/
    for ( int mu = 0; mu < 4; mu++ ) {

      /***********************************************************
       * apply  g5 Gmu W
       ***********************************************************/
      for ( int iev = 0; iev < block_length; iev++ ) {
        memcpy ( eo_spinor_work[0], w[iev], sizeof_eo_spinor_field );
        /* Gmufwdr */
        apply_cvc_vertex_eo( eo_block_field[0][iev], eo_spinor_work[0], mu, 0, gauge_field, 0 );
        /* Gmubwdr */
        apply_cvc_vertex_eo( eo_block_field[1][iev], eo_spinor_work[0], mu, 1, gauge_field, 0 );
      }
      g5_phi ( eo_block_field[0][0], 2 * Vhalf * block_length );


      /***********************************************************
       * apply  g5 Gmu XW
       ***********************************************************/
      for ( int iev = 0; iev < block_length; iev++ ) {
        memcpy ( eo_spinor_work[0], xw[iev], sizeof_eo_spinor_field );
        /* Gmufwdr */
        apply_cvc_vertex_eo( eo_block_field[2][iev], eo_spinor_work[0], mu, 0, gauge_field, 1 );
        /* Gmubwdr */
        apply_cvc_vertex_eo( eo_block_field[3][iev], eo_spinor_work[0], mu, 1, gauge_field, 1 );
      }
      g5_phi ( eo_block_field[2][0], 2 * Vhalf * block_length );

      /***********************************************************
       * loop on timeslices
       ***********************************************************/
      for ( int it = 0; it < T; it++ ) {

        /***********************************************************
         * XV^+ x Gmufwd W
         ***********************************************************/

        /***********************************************************
         * spin-color reduction
         ***********************************************************/
        exitstatus = vdag_w_spin_color_reduction ( contr_x, xv, eo_block_field[0], nev, block_length, it );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_spin_color_reduction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * momentum projection
         ***********************************************************/
        exitstatus = vdag_w_momentum_projection ( contr_p, contr_x, nev, block_length, momentum_list, momentum_number, it, 0, mu );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * V^+ x Gmubwd XW
         ***********************************************************/

        /***********************************************************
         * spin-color reduction
         ***********************************************************/
        exitstatus = vdag_w_spin_color_reduction ( contr_x, v, eo_block_field[3], nev, block_length, it+1 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_spin_color_reduction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * momentum projection
         ***********************************************************/
        exitstatus = vdag_w_momentum_projection ( contr_p, contr_x, nev, block_length, momentum_list, momentum_number, it, 1, mu );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * XV^+ x Gmubwd W
         ***********************************************************/

        /***********************************************************
         * spin-color reduction
         ***********************************************************/
        exitstatus = vdag_w_spin_color_reduction ( contr_x, xv, eo_block_field[2], nev, block_length, it+1 );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_spin_color_reduction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * momentum projection
         ***********************************************************/
        exitstatus = vdag_w_momentum_projection ( contr_p, contr_x, nev, block_length, momentum_list, momentum_number, it, 0, mu );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * V^+ x Gmufwd XW
         ***********************************************************/

        /***********************************************************
         * spin-color reduction
         ***********************************************************/
        exitstatus = vdag_w_spin_color_reduction ( contr_x, xv, eo_block_field[1], nev, block_length, it );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_spin_color_reduction, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }

        /***********************************************************
         * momentum projection
         ***********************************************************/
        exitstatus = vdag_w_momentum_projection ( contr_p, contr_x, nev, block_length, momentum_list, momentum_number, it, 1, mu );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_momentum_projection, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }


        /***********************************************************
         * write to file
         ***********************************************************/
        sprintf ( aff_tag, "%s/t%.2d/mu%d/b%.2d", tag, it+g_proc_coords[0]*T, mu, iblock );
        exitstatus = vdag_w_write_to_aff_file ( contr_p, nev, block_length, affw[it], aff_tag, momentum_list, momentum_number, io_proc );
        if ( exitstatus != 0 ) {
          fprintf(stderr, "[contract_cvc_tensor_eo_lm_factors] Error from vdag_w_write_to_aff_file, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
          return(4);
        }


      }  /* end of loop on timeslices */

    }  /* end of loop on vector index mu */

  }  /* end of loop on evec blocks */

  fini_2level_buffer ( &eo_spinor_work );
  fini_2level_buffer ( &xv );

  fini_3level_buffer ( &eo_block_field );
  fini_2level_buffer ( &w  );
  fini_2level_buffer ( &xw );

  fini_3level_buffer ( &contr_x );
  fini_3level_zbuffer ( &contr_p );

  return(0);
}  /* end of contract_cvc_tensor_eo_lm_factors */

/***********************************************************
 *
 * dimV must be integer multiple of dimW
 ***********************************************************/
int vdag_w_spin_color_reduction ( double ***contr, double**V, double**W, int dimV, int dimW, int t ) {

  const unsigned int VOL3half = LX*LY*LZ/2;
  const size_t sizeof_eo_spinor_field_timeslice = _GSI( VOL3half ) * sizeof(double);

  int exitstatus;
  double **v_ts = NULL, **w_ts = NULL;
  double ratime, retime;

  ratime = _GET_TIME;

  if ( (exitstatus = init_2level_buffer ( &v_ts, dimV, _GSI( VOL3half) ) ) != 0 ) {
    fprintf( stderr, "[vdag_w_spin_color_reduction] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  if ( (exitstatus = init_2level_buffer ( &w_ts, dimW, _GSI( VOL3half) ) ) != 0 ) {
    fprintf( stderr, "[vdag_w_spin_color_reduction] Error from init_2level_buffer, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  /***********************************************************
   * copy the timeslices t % T
   ***********************************************************/
  unsigned int offset = ( t % T) * _GSI( VOL3half );
  for ( int i = 0; i < dimV; i++ ) {
    memcpy( v_ts[i], V[i] + offset, sizeof_eo_spinor_field_timeslice );
  }

  for ( int i = 0; i < dimW; i++ ) {
    memcpy( w_ts[i], W[i] + offset, sizeof_eo_spinor_field_timeslice );
  }

#ifdef HAVE_MPI
  /***********************************************************
   * if t = T, exchange
   * receive from g_nb_t_up
   * send    to   g_nb_t_dn
   ***********************************************************/
  if ( t == T ) {
    MPI_Status mstatus;
    double *buffer = NULL;
    /***********************************************************
     * exchange v_ts
     ***********************************************************/
    if ( (buffer = (double*)malloc ( dimV * sizeof_eo_spinor_field_timeslice ) ) == NULL ) {
      fprintf(stderr, "[vdag_w_spin_color_reduction] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(2);
    }
    memcpy ( buffer, v_ts[0], dimV * sizeof_eo_spinor_field_timeslice );
    if ( (exitstatus = MPI_Send( buffer, dimV*_GSI(VOL3half), MPI_DOUBLE, g_nb_t_dn, 101, g_cart_grid) ) != MPI_SUCCESS ) {
      fprintf(stderr, "[vdag_w_spin_color_reduction] Error from MPI_Send, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      return(3);
    }
    if ( (exitstatus = MPI_Recv( v_ts[0], dimV*_GSI(VOL3half), MPI_DOUBLE, g_nb_t_up, 101, g_cart_grid, &mstatus)) != MPI_SUCCESS ) {
      fprintf(stderr, "[vdag_w_spin_color_reduction] Error from MPI_Recv, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      return(3);
    }
    free ( buffer ); buffer = NULL;

    /***********************************************************
     * exchange w_ts
     ***********************************************************/
    if ( (buffer = (double*)malloc ( dimW * sizeof_eo_spinor_field_timeslice ) ) == NULL ) {
      fprintf(stderr, "[vdag_w_spin_color_reduction] Error from malloc %s %d\n", __FILE__, __LINE__);
      return(2);
    }
    memcpy ( buffer, w_ts[0], dimW * sizeof_eo_spinor_field_timeslice );
    if ( (exitstatus = MPI_Send( buffer, dimW*_GSI(VOL3half), MPI_DOUBLE, g_nb_t_dn, 102, g_cart_grid) ) != MPI_SUCCESS ) {
      fprintf(stderr, "[vdag_w_spin_color_reduction] Error from MPI_Send, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      return(3);
    }
    if ( (exitstatus = MPI_Recv( w_ts[0], dimW*_GSI(VOL3half), MPI_DOUBLE, g_nb_t_up, 102, g_cart_grid, &mstatus)) != MPI_SUCCESS ) {
      fprintf(stderr, "[vdag_w_spin_color_reduction] Error from MPI_Recv, status was %d %s %d\n", exitstatus, __FILE__, __LINE__ );
      return(3);
    }
    free ( buffer ); buffer = NULL;
  }  /* end of if t == T */
#endif

  int nb = dimV / dimW;
  for ( int ib = 0; ib < nb; ib++ ) {
    co_field_eq_fv_dag_ti_fv ( contr[ib*dimW][0], v_ts[ib*dimW], w_ts[0], dimW*VOL3half );
  }

  fini_2level_buffer ( &v_ts );
  fini_2level_buffer ( &w_ts );

  retime = _GET_TIME;
  if ( g_cart_id == 0 && g_verbose > 0) {
    fprintf(stdout, "# [vdag_w_spin_color_reduction] time for vdag_w_spin_color_reduction = %e seconds %s %d\n", retime-ratime, __FILE__, __LINE__);
  }
  return(0);
}  /* end of vdag_w_spin_color_reduction */


/***********************************************************
 * momentum projection
 ***********************************************************/
int vdag_w_momentum_projection ( double _Complex ***contr_p, double ***contr_x, int dimV, int dimW, int (*momentum_list)[3], int momentum_number, int t, int ieo, int mu ) {

  int exitstatus;
  double momentum_shift[3] = {0.,0.,0.};

  if ( mu > 0 ) {
    momentum_shift[mu-1] = -1.;
  }

  if ( (exitstatus = momentum_projection_eo_timeslice ( contr_x[0][0], (double*)(contr_p[0][0]), dimV*dimW, momentum_number, momentum_list, t, ieo, momentum_shift, 1 )) != 0 ) {
    fprintf(stderr, "[vdag_w_momentum_projection] Error from momentum_projection_eo_timeslice, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
    return(1);
  }

  return(0);
}  /* end of vdag_w_momentum_projection */

/***********************************************************
 *
 ***********************************************************/
int vdag_w_write_to_aff_file ( double _Complex ***contr_tp, int nv, int nw, struct AffWriter_s*affw, char*tag, int (*momentum_list)[3], int momentum_number, int io_proc ) {

  const uint32_t items = nv * nw;

  int exitstatus;
  double ratime, retime;
  struct AffNode_s *affn = NULL, *affdir=NULL;
  char aff_buffer_path[200];

  ratime = _GET_TIME;

  if ( io_proc == 1 ) {
    if( (affn = aff_writer_root(affw)) == NULL ) {
      fprintf(stderr, "[vdag_w_write_to_aff_file] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
      return(1);
    }
  }

  if(io_proc == 1) {

    for( int i = 0; i < momentum_number; i++ ) {

      sprintf(aff_buffer_path, "%s/px%.2dpy%.2dpz%.2d", tag, momentum_list[i][0], momentum_list[i][1], momentum_list[i][2] );
      /* fprintf(stdout, "# [vdag_w_write_to_aff_file] current aff path = %s\n", aff_buffer_path); */

      affdir = aff_writer_mkpath(affw, affn, aff_buffer_path);

      exitstatus = aff_node_put_complex (affw, affdir, contr_tp[i][0], items );
      if(exitstatus != 0) {
        fprintf(stderr, "[vdag_w_write_to_aff_file] Error from aff_node_put_double, status was %d %s %d\n", exitstatus, __FILE__, __LINE__);
        return(5);
      }
    }
  }  /* if io_proc == 2 */

#ifdef HAVE_MPI
  MPI_Barrier( g_cart_grid );
#endif

  retime = _GET_TIME;
  if(io_proc == 2 && g_verbose > 0) fprintf(stdout, "# [vdag_w_write_to_aff_file] time for saving momentum space results = %e seconds\n", retime-ratime);

  return(0);
}  /* end of vdag_w_write_to_aff_file */

}  /* end of namespace cvc */