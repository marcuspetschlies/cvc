/****************************************************
 * contract_cvc_tensor.cpp
 *
 * Sun Feb  5 13:23:50 CET 2017
 *
 * - originally copied from p2gg_xspace.c
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
 ***********************************************************/
void init_contract_cvc_tensor_usource(double *gauge_field, int source_coords[4]) {

  int gsx0 = source_coords[0], gsx1 = source_coords[1], gsx2 = source_coords[2], gsx3 = source_coords[3];
  int sx0, sx1, sx2, sx3;
  int source_proc_coords[4];
  int exitstatus;
  double ratime, retime;

  /***********************************************************
   * determine source coordinates, find out, if source_location is in this process
   ***********************************************************/
  ratime = _GET_TIME;
#ifdef HAVE_MPI
  source_proc_coords[0] = gsx0 / T;
  source_proc_coords[1] = gsx1 / LX;
  source_proc_coords[2] = gsx2 / LY;
  source_proc_coords[3] = gsx3 / LZ;

  if(g_cart_id == 0) {
    fprintf(stdout, "# [init_contract_cvc_tensor_usource] global source coordinates: (%3d,%3d,%3d,%3d)\n",  gsx0, gsx1, gsx2, gsx3);
    fprintf(stdout, "# [init_contract_cvc_tensor_usource] source proc coordinates: (%3d,%3d,%3d,%3d)\n",  source_proc_coords[0], source_proc_coords[1], source_proc_coords[2], source_proc_coords[3]);
  }

  exitstatus = MPI_Cart_rank(g_cart_grid, source_proc_coords, &source_proc_id);
  if(exitstatus != MPI_SUCCESS) {
    fprintf(stderr, "[init_contract_cvc_tensor_usource] Error from MPI_Cart_rank, status was %d\n", exitstatus);
    EXIT(1);
  }
  if( source_proc_id == g_cart_id ) {
    fprintf(stdout, "# [init_contract_cvc_tensor_usource] process %2d has source location\n", source_proc_id);
  }
#endif

  sx0 = gsx0 % T;
  sx1 = gsx1 % LX;
  sx2 = gsx2 % LY;
  sx3 = gsx3 % LZ;

  Usource[0] = Usourcebuffer;
  Usource[1] = Usourcebuffer+18;
  Usource[2] = Usourcebuffer+36;
  Usource[3] = Usourcebuffer+54;

  if( source_proc_id == g_cart_id ) { 
    source_location = g_ipt[sx0][sx1][sx2][sx3];
    source_location_iseven = g_iseven[source_location] ;
    fprintf(stdout, "# [init_contract_cvc_tensor_usource] local source coordinates: (%3d,%3d,%3d,%3d), is even = %d\n", sx0, sx1, sx2, sx3, source_location_iseven);
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


/***********************************************************
 * reduction
 *
 * w = tr ( g5 r^+ g5 s )
 ***********************************************************/
void co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (complex *w, fermion_propagator_type *r, fermion_propagator_type *s, unsigned int N) {

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  unsigned int ix;
  complex *w_ = NULL, wtmp;
  fermion_propagator_type *r_=NULL, *s_=NULL;
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

    _fp_eq_gamma_ti_fp( fp1, 5, r_ );
    _fp_eq_fp_ti_gamma( fp2, 5, fp1 );
    _co_eq_tr_fp_dagger_ti_fp ( &wtmp, fp2,  s_);
    _co_pl_eq_co( w_, &wtmp);
  }

  free_fp( &fp1 );
  free_fp( &fp2 );

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field */

void co_field_mi_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (complex *w, fermion_propagator_type *r, fermion_propagator_type *s, unsigned int N) {

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  unsigned int ix;
  complex *w_ = NULL, wtmp;
  fermion_propagator_type *r_=NULL, *s_=NULL;
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

    _fp_eq_gamma_ti_fp( fp1, 5, r_ );
    _fp_eq_fp_ti_gamma( fp2, 5, fp1 );
    _co_eq_tr_fp_dagger_ti_fp ( &wtmp, fp2,  s_);
    _co_mi_eq_co( w_, &wtmp);
  }

  free_fp( &fp1 );
  free_fp( &fp2 );

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of co_field_mi_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field */


/***********************************************************
 * contractions for cvc - cvc tensor
 ***********************************************************/
void contract_cvc_tensor_eo ( double *conn, double *contact_term, double**sprop_list_e, double**sprop_list_o, double**tprop_list_e, double**tprop_list_o  ) {
  
  const unsigned int Vhalf = VOLUME / 2;
  const size_t sizeof_eo_propagator_field = Vhalf * 288 * sizeof(double);

  int mu, nu, ir, ia, ib, imunu;
  unsigned int ix;
  double ratime, retime;
#ifndef HAVE_OPENMP
  double spinor1[24], spinor2[24], U_[18];
  complex w, w1;
#endif
  double *phi=NULL, *chi=NULL;

  /* auxilliary fermion propagator field with halo */
  fermion_propagator_type fp_aux        = create_fp_field( (VOLUME+RAND)/2 );

  /* fermion propagator fields without halo */
  fermion_propagator_type gamma_fp_X[0] = create_fp_field( VOLUME/2 );
  fermion_propagator_type gamma_fp_X[1] = create_fp_field( VOLUME/2 );
  fermion_propagator_type gamma_fp_X[2] = create_fp_field( VOLUME/2 );
  fermion_propagator_type gamma_fp_X[3] = create_fp_field( VOLUME/2 );

  fermion_propagator_type fp_Y_gamma    = create_fp_field( VOLUME/2 );
  fermion_propagator_type fp_X          = create_fp_field( VOLUME/2 );

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
  for( nu=0; n<4; mu++ ) {
  
    /* fp_Y_gamma^o = sprop^o , source + nu */
    exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_Y_gamma, sprop_list_o+12*nu, Vhalf);

    /* fp_Y_gamma^o = fp_Y_gamma^o * Gamma_nu^b  */
    apply_propagator_constant_cvc_vertex ( fp_Y_gamma, fp_Y_gamma, nu, 1, Usource[nu] );

    for( mu=0; mu<4; mu++ ) {

      complex *conn_ = (complex*)( conn + (4*mu+nu)*2*VOLUME );

      /* contract S^o Gamma_nu, Gamma_mu T^e */
      co_field_mi_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, fp_Y_gamma, gamma_fp_X[mu], Vhalf);


      /* fp_aux^o = fp_Y_gamma^o */
      memcpy(fp_aux[0][0], fp_Y_gamma[0][0], sizeof_eo_propagator_field );
      
      /* gamma_fp_Y_gamma^e = Gamma_mu^f fp_aux^o */
      apply_cvc_vertex_propagator_eo ( gamma_fp_Y_gamma, fp_aux, mu, 0, gauge_field, 0);

      /* contract gamma_fp_Y_gamma^e, T^e */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, gamma_fp_Y_gamma, fp_X, Vhalf);

    }

    /* contribution to contact term */
    if( (source_proc_id == g_cart_id) && !source_location_iseven ) {
      /* gamma_fp_X[nu] = Gamma_nu^f T^e */
      complex w;
      _co_eq_tr_fp ( &w, gamma_fp_X[nu][source_location] );
      contact_term[2*nu  ] += -0.5 * w.re;
      contact_term[2*nu+1] += -0.5 * w.im;
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
  for( nu=0; n<4; mu++ ) {
  
    /* fp_Y_gamma^o = sprop^o , source + nu */
    exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_Y_gamma, tprop_list_o+12*nu, Vhalf);

    /* fp_Y_gamma^o = fp_Y_gamma^o * Gamma_nu^b  */
    apply_propagator_constant_cvc_vertex ( fp_Y_gamma, fp_Y_gamma, nu, 1, Usource[nu] );

    for( mu=0; mu<4; mu++ ) {

      complex *conn_ = (complex*)( conn + (4*mu+nu)*2*VOLUME );

      /* contract Gamma_mu^f S^e, T^o Gamma_nu^b */
      co_field_mi_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, gamma_fp_X[mu], fp_Y_gamma, Vhalf);

      /* fp_aux^o = fp_Y_gamma^o */
      memcpy(fp_aux[0][0], fp_Y_gamma[0][0], sizeof_eo_propagator_field );
      
      /* gamma_fp_Y_gamma^e = Gamma_mu^f fp_aux^o */
      apply_cvc_vertex_propagator_eo ( gamma_fp_Y_gamma, fp_aux, mu, 0, gauge_field, 0);

      /* contract gamma_fp_Y_gamma^e, T^e */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, fp_X, gamma_fp_Y_gamma, Vhalf);

    }

    /* contribution to contact term */
    if( (source_proc_id == g_cart_id) && !source_location_iseven ) {
      /* fp_Y_gamma = T^o Gamma_nu^b */
      complex w;
      _co_eq_tr_fp ( &w, fp_Y_gamma[source_location] );
      contact_term[2*nu  ] += 0.5 * w.re;
      contact_term[2*nu+1] += 0.5 * w.im;
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
  for( nu=0; n<4; mu++ ) {
  
    /* fp_Y_gamma^e = sprop^e , source + nu */
    exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_Y_gamma, sprop_list_e+12*nu, Vhalf);

    /* fp_Y_gamma^e = fp_Y_gamma^e * Gamma_nu^b  */
    apply_propagator_constant_cvc_vertex ( fp_Y_gamma, fp_Y_gamma, nu, 1, Usource[nu] );

    for( mu=0; mu<4; mu++ ) {

      complex *conn_ = (complex*)( conn + (4*mu+nu)*2*VOLUME );

      /* contract S^e Gamma_nu^b, Gamma_mu^f T^o */
      co_field_mi_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, fp_Y_gamma, gamma_fp_X[mu], Vhalf);

      /* fp_aux^e = fp_Y_gamma^e */
      memcpy(fp_aux[0][0], fp_Y_gamma[0][0], sizeof_eo_propagator_field );
      
      /* gamma_fp_Y_gamma^o = Gamma_mu^f fp_aux^e */
      apply_cvc_vertex_propagator_eo ( gamma_fp_Y_gamma, fp_aux, mu, 0, gauge_field, 1);

      /* contract gamma_fp_Y_gamma^o, T^o */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, gamma_fp_Y_gamma, fp_X, Vhalf);

    }

    /* contribution to contact term */
    if( (source_proc_id == g_cart_id) && source_location_iseven ) {
      /* gamma_fp_X = Gamma_nu^f T^o */
      complex w;
      _co_eq_tr_fp ( &w, gamma_fp_X[nu][source_location] );
      contact_term[2*nu  ] -= 0.5 * w.re;
      contact_term[2*nu+1] -= 0.5 * w.im;
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
  for( nu=0; n<4; mu++ ) {
  
    /* fp_Y_gamma^e = tprop^e , source + nu */
    exitstatus = assign_fermion_propagaptor_from_spinor_field (fp_Y_gamma, tprop_list_e+12*nu, Vhalf);

    /* fp_Y_gamma^e = fp_Y_gamma^e * Gamma_nu^b  */
    apply_propagator_constant_cvc_vertex ( fp_Y_gamma, fp_Y_gamma, nu, 1, Usource[nu] );

    for( mu=0; mu<4; mu++ ) {

      complex *conn_ = (complex*)( conn + (4*mu+nu)*2*VOLUME );

      /* contract Gamma_mu^f S^o, T^e Gamma_nu^b */
      co_field_mi_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, gamma_fp_X[mu], fp_Y_gamma, Vhalf);

      /* fp_aux^e = fp_Y_gamma^e */
      memcpy(fp_aux[0][0], fp_Y_gamma[0][0], sizeof_eo_propagator_field );
      
      /* gamma_fp_Y_gamma^o = Gamma_mu^f fp_aux^e */
      apply_cvc_vertex_propagator_eo ( gamma_fp_Y_gamma, fp_aux, mu, 0, gauge_field, 1);

      /* contract S^o, gamma_fp_Y_gamma^o */
      co_field_pl_eq_tr_g5_ti_propagator_field_dagger_ti_g5_ti_propagator_field (conn_, fp_X, gamma_fp_Y_gamma, Vhalf);

    }

    /* contribution to contact term */
    if( (source_proc_id == g_cart_id) && source_location_iseven ) {
      /* fp_Y_gamma = T^e Gamma_nu^b */
      complex w;
      _co_eq_tr_fp ( &w, fp_Y_gamma[source_location] );
      contact_term[2*nu  ] = 0.5 * w.re;
      contact_term[2*nu+1] = 0.5 * w.im;
    }

  }  /* end of loop on nu */

  free_fp_field(&fp_aux);
  free_fp_field(&fp_X);
  free_fp_field(&(gamma_fp_X[0]));
  free_fp_field(&(gamma_fp_X[1]));
  free_fp_field(&(gamma_fp_X[2]));
  free_fp_field(&(gamma_fp_X[3]));
  free_fp_field( &fp_Y_gamma );
  free_fp_field( &gamma_fp_Y_gamma );

  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [contract_cvc_tensor] time for contract_cvc_tensor = %e seconds\n", retime-ratime);

#ifdef HAVE_MPI
  if(g_cart_id == source_proc_id) fprintf(stdout, "# [contract_cvc_tensor] broadcasting contact term ...\n");
  MPI_Bcast(contact_term, 8, MPI_DOUBLE, source_proc_id, g_cart_grid);
  /* TEST */
  /* fprintf(stdout, "[%2d] contact term = "\
      "(%e + I %e, %e + I %e, %e + I %e, %e +I %e)\n",
      g_cart_id, contact_term[0], contact_term[1], contact_term[2], contact_term[3],
      contact_term[4], contact_term[5], contact_term[6], contact_term[7]); */
#endif

  return;

}  /* end of contract_cvc_tensor_eo */


/***********************************************************
 * contractions for cvc - m
 *   cvc at source, m at sink
 ***********************************************************/
void contract_cvc_m(double *conn, int gid, double*fwd_list[5][12], double*bwd_list[5][12], double*fwd_list_eo[2][5][12], double*bwd_list_eo[2][5][12]) {
  const  size_t sizeof_spinor_field = _GSI(VOLUME) * sizeof(double);
  int nu, ir, ia, ib;
  unsigned int ix;
  double ratime, retime;
#ifndef HAVE_OPENMP
  complex w, w1;
#endif
  double **spinor_work = NULL;

  spinor_work = (double**)malloc(24*sizeof(double*));
  for(nu=0; nu<24; nu++ ) alloc_spinor_field(&spinor_work[nu], VOLUME);

  /**********************************************************
   **********************************************************
   **
   ** contractions
   **
   **********************************************************
   **********************************************************/  
  
    ratime = _GET_TIME;
  
    /**********************************************************
     * first contribution
     **********************************************************/  
  
    
    if(fwd_list != NULL) {
      for(ia=0; ia<12; ia++) {
        memcpy( spinor_work[ia], fwd_list[4][ia], sizeof_spinor_field);
      }
    } else {
      for(ia=0; ia<12; ia++) {
        spinor_field_eo2lexic( spinor_work[ia], fwd_list_eo[0][4][ia], fwd_list_eo[1][4][ia] );
      }
    }
    for(ia=0; ia<12; ia++) {
      spinor_field_eq_gamma_ti_spinor_field(spinor_work[ia], gid, spinor_work[ia], VOLUME);
      g5_phi(spinor_work[ia], VOLUME);
    }

    /* loop on the Lorentz index nu at source */
    for(nu=0; nu<4; nu++) 
    {
      if(bwd_list != NULL) {
        for(ib=0; ib<12; ib++) {
          memcpy( spinor_work[12+ib], bwd_list[nu][ib], sizeof_spinor_field);
        }
      } else {
        for(ib=0; ib<12; ib++) {
          spinor_field_eo2lexic( spinor_work[12+ib], bwd_list_eo[0][nu][ib], bwd_list_eo[1][nu][ib] );
        }
      }
  
      for(ir=0; ir<4; ir++) {
  
        for(ia=0; ia<3; ia++) {
          /* phi = g_spinor_field[      4*12 + 3*ir + ia]; */
          /* phi = g_spinor_field[120 + 4*12 + 3*ir + ia]; */
          double *phi = spinor_work[3*ir+ia];
  
        for(ib=0; ib<3; ib++) {
          /* chi = g_spinor_field[60 + nu*12 + 3*gperm[nu][ir] + ib]; */
          /* chi = g_spinor_field[(1 - g_propagator_position) * 60 + nu*12 + 3*gperm[nu][ir] + ib]; */
          double *chi = spinor_work[12 + 3*gperm[nu][ir] + ib];
  
          /* 1) gamma_nu gamma_5 x U */
  
#ifdef HAVE_OPENMP
#pragma omp parallel shared(ia, ib, nu, ir)
{
            complex w, w1;
#pragma omp for
#endif
            for(ix=0; ix<VOLUME; ix++) {
              _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), phi+_GSI(ix));
              _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
              if(!isimag[nu]) {
                conn[_GWI(nu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.re;
                conn[_GWI(nu,ix,VOLUME)+1] += gperm_sign[nu][ir] * w1.im;
              } else {
                conn[_GWI(nu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.im;
                conn[_GWI(nu,ix,VOLUME)+1] -= gperm_sign[nu][ir] * w1.re;
              }
            }  /* of ix */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

        }      /* of ib */
        }      /* of ia */
  
        for(ia=0; ia<3; ia++) {
          /* phi = g_spinor_field[      4*12 + 3*ir            + ia]; */
          /* phi = g_spinor_field[120 +      4*12 + 3*ir            + ia]; */
          double *phi = spinor_work[3*ir + ia];
  
        for(ib=0; ib<3; ib++) {
          /* chi = g_spinor_field[60 + nu*12 + 3*gperm[ 4][ir] + ib]; */
          /* chi = g_spinor_field[(1 - g_propagator_position) * 60 + nu*12 + 3*gperm[ 4][ir] + ib]; */
          double *chi = spinor_work[12 + 3*gperm[ 4][ir] + ib];

  
          /* -gamma_5 x U */
#ifdef HAVE_OPENMP
#pragma omp parallel shared(ia, ib, nu, ir)
{
            complex w, w1;

#pragma omp for
#endif
            for(ix=0; ix<VOLUME; ix++) {
              _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), phi+_GSI(ix));
              _co_eq_co_ti_co(&w1, &w, (complex*)(Usource[nu]+2*(3*ia+ib)));
              conn[_GWI(nu,ix,VOLUME)  ] -= gperm_sign[4][ir] * w1.re;
              conn[_GWI(nu,ix,VOLUME)+1] -= gperm_sign[4][ir] * w1.im;
            }  /* of ix */
  
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif 

        }      /* of ib */
        }  /* of ia */
      }    /* of ir */
  
    }  // of nu
  
    /**********************************************************
     * second contribution
     **********************************************************/  
  
    /* loop on the Lorentz index nu at source */
    if(bwd_list != NULL) {
      for(ib=0; ib<12; ib++) {
        memcpy( spinor_work[12+ib], bwd_list[4][ib], sizeof_spinor_field);
      }
    } else {
      for(ib=0; ib<12; ib++) {
        spinor_field_eo2lexic( spinor_work[12+ib], bwd_list_eo[0][4][ib], bwd_list_eo[1][4][ib] );
      }
    }

    for(nu=0; nu<4; nu++) 
    {
      if(fwd_list != NULL) {
        for(ia=0; ia<12; ia++) {
          memcpy( spinor_work[ia], fwd_list[nu][ia], sizeof_spinor_field);
        }
      } else {
        for(ia=0; ia<12; ia++) {
          spinor_field_eo2lexic( spinor_work[ia], fwd_list_eo[0][nu][ia], fwd_list_eo[1][nu][ia] );
        }
      }
      for(ia=0; ia<12; ia++) {
        spinor_field_eq_gamma_ti_spinor_field(spinor_work[ia], gid, spinor_work[ia], VOLUME);
        g5_phi(spinor_work[ia], VOLUME);
      }

      for(ir=0; ir<4; ir++) {
  
        for(ia=0; ia<3; ia++) {
          /* phi = g_spinor_field[     nu*12 + 3*ir            + ia]; */
          /* phi = g_spinor_field[120 +     nu*12 + 3*ir            + ia]; */
          double *phi = spinor_work[3*ir + ia];
  
        for(ib=0; ib<3; ib++) {
          /* chi = g_spinor_field[60 +  4*12 + 3*gperm[nu][ir] + ib]; */
          /* chi = g_spinor_field[(1 - g_propagator_position) * 60 +  4*12 + 3*gperm[nu][ir] + ib]; */
          double *chi = spinor_work[12 + 3*gperm[nu][ir] + ib];
  
      
          /* 1) gamma_nu gamma_5 x U^dagger */
  
#ifdef HAVE_OPENMP
#pragma omp parallel shared(ia, ib, nu, ir)
{
            complex w, w1;
#pragma omp for
#endif
            for(ix=0; ix<VOLUME; ix++) {
              _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), phi+_GSI(ix));
              _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
              if(!isimag[nu]) {
                conn[_GWI(nu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.re;
                conn[_GWI(nu,ix,VOLUME)+1] += gperm_sign[nu][ir] * w1.im;
              } else {
                conn[_GWI(nu,ix,VOLUME)  ] += gperm_sign[nu][ir] * w1.im;
                conn[_GWI(nu,ix,VOLUME)+1] -= gperm_sign[nu][ir] * w1.re;
              }
            }  /* of ix */
  
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif


  
        } /* of ib */
        } /* of ia */
  
        for(ia=0; ia<3; ia++) {
          /* phi = g_spinor_field[     nu*12 + 3*ir            + ia]; */
          /* phi = g_spinor_field[120 +     nu*12 + 3*ir            + ia]; */
          double *phi = spinor_work[3*ir + ia];
  
        for(ib=0; ib<3; ib++) {
          /* chi = g_spinor_field[60 +  4*12 + 3*gperm[ 4][ir] + ib]; */
          /* chi = g_spinor_field[(1 - g_propagator_position) * 60 +  4*12 + 3*gperm[ 4][ir] + ib]; */
          double *chi = spinor_work[12 + 3*gperm[ 4][ir] + ib];
  
          /* -gamma_5 x U */
  
#ifdef HAVE_OPENMP
#pragma omp parallel shared(ia, ib, nu, ir)
{
            complex w, w1;
#pragma omp for
#endif
            for(ix=0; ix<VOLUME; ix++) {
              _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), phi+_GSI(ix));
              _co_eq_co_ti_co_conj(&w1, &w, (complex*)(Usource[nu]+2*(3*ib+ia)));
              conn[_GWI(nu,ix,VOLUME)  ] += gperm_sign[4][ir] * w1.re;
              conn[_GWI(nu,ix,VOLUME)+1] += gperm_sign[4][ir] * w1.im;
          
            }  /* of ix */
  
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
        }      /* of ib */
        }  /* of ia */
      }    /* of ir */
    }      /* of nu */


  retime = _GET_TIME;
  if(g_cart_id==0) fprintf(stdout, "# [contract_cvc_tensor] time for contract_cvc_m = %e seconds\n", retime-ratime);

  for(nu=0; nu<24; nu++ ) free(spinor_work[nu]);
  free( spinor_work );

  return;

}  /* end of contract_cvc_m */

/***********************************************************
 * contractions for m - m
 *   m at source and sink
 ***********************************************************/
void contract_m_m(double *conn, int idsource, int idsink, double*fwd_list[12], double*bwd_list[12], double*fwd_list_eo[2][12], double*bwd_list_eo[2][12]) {
  
  const size_t sizeof_spinor_field = _GSI(VOLUME) * sizeof(double);

  int ir, ia;
  double ratime, retime;
#ifndef HAVE_OPENMP
  complex w;
  unsigned int ix;
#endif
  double *spinor_work[2];

  int psource[4] = { gamma_permutation[idsource][ 0] / 6,  gamma_permutation[idsource][ 6] / 6, 
                     gamma_permutation[idsource][12] / 6, gamma_permutation[idsource][18] / 6 };
  int jre = gamma_permutation[idsource][ 0] % 2;
  int jim = (jre + 1) % 2;
  double sre = 1 - 2*jre;
  /* sign from the source gamma matrix; the minus sign
   * in the lower two lines is the action of gamma_5 */
  double ssource[4] =  { (double)(gamma_sign[idsource][ 0] * gamma_sign[5][gamma_permutation[idsource][ 0]]),
                         (double)(gamma_sign[idsource][ 6] * gamma_sign[5][gamma_permutation[idsource][ 6]]),
                         (double)(gamma_sign[idsource][12] * gamma_sign[5][gamma_permutation[idsource][12]]),
                         (double)(gamma_sign[idsource][18] * gamma_sign[5][gamma_permutation[idsource][18]]) };


  ratime = _GET_TIME;
  alloc_spinor_field(&spinor_work[0], VOLUME);
  alloc_spinor_field(&spinor_work[1], VOLUME);

  memset(conn, 0, 2*VOLUME*sizeof(double));
  /**********************************************************
   * contraction
   **********************************************************/  
  for(ir=0; ir<4; ir++) {
    for(ia=0; ia<3; ia++) {
      if(fwd_list != NULL) {
        memcpy( spinor_work[0], fwd_list[3*ir+ia], sizeof_spinor_field);
      } else {
        spinor_field_eo2lexic( spinor_work[0], fwd_list_eo[0][3*ir+ia], fwd_list_eo[1][3*ir+ia] );
      }
      spinor_field_eq_gamma_ti_spinor_field(spinor_work[0], idsink, spinor_work[0], VOLUME);
      g5_phi(spinor_work[0], VOLUME);
      double *phi = spinor_work[0];

      if(bwd_list != NULL) {
        memcpy(spinor_work[1], bwd_list[3*+psource[ir] + ia], sizeof_spinor_field);
      } else {
        spinor_field_eo2lexic( spinor_work[1], bwd_list_eo[0][3*+psource[ir] + ia], bwd_list_eo[1][3*+psource[ir] + ia]);
      }
      double *chi = spinor_work[1];

#ifdef HAVE_OPENMP
#pragma omp parallel shared(ia, ir)
{
      complex w;
      unsigned int ix;
#pragma omp for
#endif
      for(ix=0; ix<VOLUME; ix++) {
        _co_eq_fv_dag_ti_fv(&w, chi+_GSI(ix), phi+_GSI(ix));
#if 0
        if(!isimag[nu]) {
          conn[2*ix  ] += gperm_sign[nu][ir] * w.re;
          conn[2*ix+1] += gperm_sign[nu][ir] * w.im;
        } else {
          conn[2*ix  ] += gperm_sign[nu][ir] * w.im;
          conn[2*ix+1] -= gperm_sign[nu][ir] * w.re;
        }
#endif
        conn[2*ix+jre] += sre * ssource[ir] * w.re;
        conn[2*ix+jim] +=       ssource[ir] * w.im;

      }  /* of ix */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
    }  /* end of loop on ir */
  }    /* end of loop on ia */

  free( spinor_work[0] );
  free( spinor_work[1] );

  retime = _GET_TIME;
  if(g_cart_id == 0) fprintf(stdout, "# [contract_m_m] time for contract_m_m = %e seconds\n", retime-ratime);

  return;
}  /* end of contract_m_m */


/*************************************************************************************************
 * contract loop with eigenvectors of stochastic propagators
 *  - contraction for twisted mass (only important for X_eo and C_from_Xeo)
 *************************************************************************************************/
void contract_m_loop (double*conn, double**field_list, int field_number, int *gid_list, int gid_number, int *momentum_list[3], int momentum_number, double*weight, double mass, double**eo_work) {

  typedef struct {
    int x[4];
  } point;

  const unsigned int Vhalf = VOLUME/2;
  const size_t sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);

  int i, ig, imom;
  unsigned int ix, iy;
  int x0, x1, x2, x3;
  point *eo2lexic_coords;
  double *buffer;

  eo2lexic_coords = (point*)malloc(VOLUME*sizeof(point));
  if(eo2lexic_coords == NULL) {
    fprintf(stderr, "[contract_m_loop] Error from malloc\n");
    EXIT(1);
  }
  for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      iy = g_lexic2eosub[ix];
      if(g_iseven[ix]) {
        eo2lexic_coords[iy].x[0] = x0;
        eo2lexic_coords[iy].x[1] = x1;
        eo2lexic_coords[iy].x[2] = x2;
        eo2lexic_coords[iy].x[3] = x3;
      } else {
        eo2lexic_coords[iy+Vhalf].x[0] = x0;
        eo2lexic_coords[iy+Vhalf].x[1] = x1;
        eo2lexic_coords[iy+Vhalf].x[2] = x2;
        eo2lexic_coords[iy+Vhalf].x[3] = x3;
      }
    }}}
  }

  buffer = (double*)malloc(2*VOLUME*sizeof(double));
  if( buffer == NULL ) {
    fprintf(stderr, "[contract_m_loop] Error from malloc\n");
    EXIT(2);
  }

  memset( conn, 0, 2*T*momentum_number*gid_number*sizeof(double) );

  /* loop on fields */
  for(i=0; i<field_number; i++) {
    memcpy(eo_work[3], field_list[i], sizeof_eo_spinor_field);
    /* Xbar V */
    X_eo (eo_work[0], eo_work[3], -mass, g_gauge_field);
    /* Wtilde  = Cbar V */
    C_from_Xeo (eo_work[1], eo_work[0], field_list[i], g_gauge_field, -mass);
    /* XW = X W*/
    X_eo (eo_work[2], eo_work[1], mass, g_gauge_field);

    for(ig=0; ig<gid_number; ig++) {
      int gid = gid_list[ig];
      memset(buffer, 0, 2*VOLUME*sizeof(double));
#ifdef HAVE_OPENMP
#pragma omp parallel
{         
#endif
      double spinor1[24];
#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for(ix=0; ix<Vhalf; ix++) {
        /* even part */
        /* sp1 = gamma x X Wtilde  */
        _fv_eq_gamma_ti_fv( spinor1, gid, eo_work[2]+_GSI(ix) );
        /* sp1 = g5 sp1 */
        _fv_ti_eq_g5(spinor1);
        /* buffer = (Xbar V)^+ sp1 */
        _co_eq_fv_dag_ti_fv((complex*)(buffer+2*ix), eo_work[0]+_GSI(ix), spinor1);
      }

#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for(ix=0; ix<Vhalf; ix++) {
        /* odd part */
        /* sp1 = gamma x Wtilde  */
        _fv_eq_gamma_ti_fv( spinor1, gid, eo_work[1]+_GSI(ix) );
        /* sp1 = g5 sp1 */
        _fv_ti_eq_g5(spinor1);
        /* buffer = V^+ sp1 */
        _co_eq_fv_dag_ti_fv((complex*)(buffer+2*(Vhalf+ix)), field_list[i]+_GSI(ix), spinor1);
      }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

    for(imom=0; imom<momentum_number; imom++) {
      double q[3] = { 2.*M_PI * momentum_list[imom][0] / LX_global, 2.*M_PI * momentum_list[imom][1] / LY_global, 2.*M_PI * momentum_list[imom][2] / LZ_global };
      double q_offset = q[0] * g_proc_coords[1]*LX + q[1] * g_proc_coords[2]*LY + q[2] * g_proc_coords[3]*LZ;
#ifdef HAVE_OPENMP
#pragma omp parallel
{         
#endif
        complex w1;
        complex *zconn_ = (complex*)(conn + 2* ( ig * momentum_number + imom ) * T);
        double q_phase;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
        for(ix=0; ix<Vhalf; ix++) {
          q_phase = q_offset + q[0] * eo2lexic_coords[ix].x[1] + q[1] * eo2lexic_coords[ix].x[2] + q[2] * eo2lexic_coords[ix].x[3];
          x0 = eo2lexic_coords[ix].x[0];
          /* even part */
          w1.re = cos(q_phase)*weight[i]; w1.im = sin(q_phase)*weight[i];
          _co_pl_eq_co_ti_co( zconn_+x0 , (complex*)(buffer+2*ix), &w1 );
        }
#ifdef HAVE_OPENMP
#pragma omp for
#endif
        for(ix=Vhalf; ix<VOLUME; ix++) {
          q_phase = q_offset + q[0] * eo2lexic_coords[ix].x[1] + q[1] * eo2lexic_coords[ix].x[2] + q[2] * eo2lexic_coords[ix].x[3];
          x0 = eo2lexic_coords[ix].x[0];
          /* even part */
          w1.re = cos(q_phase)*weight[i]; w1.im = sin(q_phase)*weight[i];
          _co_pl_eq_co_ti_co( zconn_+x0 , (complex*)(buffer+2*ix), &w1 );
        }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
      }  /* end of loop on gamma */
    }  /* end of loop on momenta */
  }  /* end of loop on fields */


  free(eo2lexic_coords);
  free(buffer);
  return;
}  /* end of contract_m_loop */


/*************************************************************************************************
 * contract meson loop with eigenvectors of stochastic propagators
 * - tm-clover version
 *
 *************************************************************************************************/
void contract_m_loop_clover (double*conn, double**field_list, int field_number, int *gid_list, int gid_number,
    int *momentum_list[3], int momentum_number, double*weight, double*mzz_dn, double*mzzinv_up, double*mzzinv_dn, double**eo_work) {

  typedef struct {
    int x[4];
  } point;

  const unsigned int Vhalf = VOLUME/2;
  const size_t sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);

  int i, ig, imom;
  unsigned int ix, iy;
  int x0, x1, x2, x3;
  point *eo2lexic_coords;
  double *buffer;

  eo2lexic_coords = (point*)malloc(VOLUME*sizeof(point));
  if(eo2lexic_coords == NULL) {
    fprintf(stderr, "[contract_m_loop] Error from malloc\n");
    EXIT(1);
  }
  for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      iy = g_lexic2eosub[ix];
      if(g_iseven[ix]) {
        eo2lexic_coords[iy].x[0] = x0;
        eo2lexic_coords[iy].x[1] = x1;
        eo2lexic_coords[iy].x[2] = x2;
        eo2lexic_coords[iy].x[3] = x3;
      } else {
        eo2lexic_coords[iy+Vhalf].x[0] = x0;
        eo2lexic_coords[iy+Vhalf].x[1] = x1;
        eo2lexic_coords[iy+Vhalf].x[2] = x2;
        eo2lexic_coords[iy+Vhalf].x[3] = x3;
      }
    }}}
  }

  buffer = (double*)malloc(2*VOLUME*sizeof(double));
  if( buffer == NULL ) {
    fprintf(stderr, "[contract_m_loop] Error from malloc\n");
    EXIT(2);
  }

  memset( conn, 0, 2*T*momentum_number*gid_number*sizeof(double) );

  /* loop on fields */
  for(i=0; i<field_number; i++) {
    memcpy(eo_work[3], field_list[i], sizeof_eo_spinor_field);
    /* Xbar V */
    X_clover_eo (eo_work[0], eo_work[3], g_gauge_field, mzzinv_dn);
    /* Wtilde  = Cbar V */
    C_clover_from_Xeo (eo_work[1], eo_work[0], field_list[i], g_gauge_field, mzz_dn);
    /* XW = X W*/
    X_clover_eo (eo_work[2], eo_work[1], g_gauge_field, mzzinv_up);

    for(ig=0; ig<gid_number; ig++) {
      int gid = gid_list[ig];
      memset(buffer, 0, 2*VOLUME*sizeof(double));
#ifdef HAVE_OPENMP
#pragma omp parallel
{         
#endif
      double spinor1[24];
#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for(ix=0; ix<Vhalf; ix++) {
        /* even part */
        /* sp1 = gamma x X Wtilde  */
        _fv_eq_gamma_ti_fv( spinor1, gid, eo_work[2]+_GSI(ix) );
        /* sp1 = g5 sp1 */
        _fv_ti_eq_g5(spinor1);
        /* buffer = (Xbar V)^+ sp1 */
        _co_eq_fv_dag_ti_fv((complex*)(buffer+2*ix), eo_work[0]+_GSI(ix), spinor1);
      }

#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for(ix=0; ix<Vhalf; ix++) {
        /* odd part */
        /* sp1 = gamma x Wtilde  */
        _fv_eq_gamma_ti_fv( spinor1, gid, eo_work[1]+_GSI(ix) );
        /* sp1 = g5 sp1 */
        _fv_ti_eq_g5(spinor1);
        /* buffer = V^+ sp1 */
        _co_eq_fv_dag_ti_fv((complex*)(buffer+2*(Vhalf+ix)), field_list[i]+_GSI(ix), spinor1);
      }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

    for(imom=0; imom<momentum_number; imom++) {
      double q[3] = { 2.*M_PI * momentum_list[imom][0] / LX_global, 2.*M_PI * momentum_list[imom][1] / LY_global, 2.*M_PI * momentum_list[imom][2] / LZ_global };
      double q_offset = q[0] * g_proc_coords[1]*LX + q[1] * g_proc_coords[2]*LY + q[2] * g_proc_coords[3]*LZ;
#ifdef HAVE_OPENMP
#pragma omp parallel
{         
#endif
        complex w1;
        complex *zconn_ = (complex*)(conn + 2* ( ig * momentum_number + imom ) * T);
        double q_phase;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
        for(ix=0; ix<Vhalf; ix++) {
          q_phase = q_offset + q[0] * eo2lexic_coords[ix].x[1] + q[1] * eo2lexic_coords[ix].x[2] + q[2] * eo2lexic_coords[ix].x[3];
          x0 = eo2lexic_coords[ix].x[0];
          /* even part */
          w1.re = cos(q_phase)*weight[i]; w1.im = sin(q_phase)*weight[i];
          _co_pl_eq_co_ti_co( zconn_+x0 , (complex*)(buffer+2*ix), &w1 );
        }
#ifdef HAVE_OPENMP
#pragma omp for
#endif
        for(ix=Vhalf; ix<VOLUME; ix++) {
          q_phase = q_offset + q[0] * eo2lexic_coords[ix].x[1] + q[1] * eo2lexic_coords[ix].x[2] + q[2] * eo2lexic_coords[ix].x[3];
          x0 = eo2lexic_coords[ix].x[0];
          /* even part */
          w1.re = cos(q_phase)*weight[i]; w1.im = sin(q_phase)*weight[i];
          _co_pl_eq_co_ti_co( zconn_+x0 , (complex*)(buffer+2*ix), &w1 );
        }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
      }  /* end of loop on gamma */
    }  /* end of loop on momenta */
  }  /* end of loop on fields */


  free(eo2lexic_coords);
  free(buffer);
  return;
}  /* end of contract_m_loop_clover */


/*************************************************************************************************
 * contract loop with eigenvectors of stochastic propagators
 * - contractions for tm-clover (only for Xeo and C_from_Xeo)
 *************************************************************************************************/
void contract_cvc_loop (double*conn, double**field_list, int field_number, int *momentum_list[3], int momentum_number, double*weight, double mass, double**eo_work) {

  typedef struct {
    int x[4];
  } point;

  const unsigned int Vhalf = VOLUME/2;
  const unsigned int VpRhalf = (VOLUME+RAND)/2;
  const size_t sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);

  int i, x0, x1, x2, x3, mu, imom;
  unsigned int ix;
  point *lexic_coords;
  double *buffer;

  lexic_coords = (point*)malloc(VOLUME*sizeof(point));
  if(lexic_coords == NULL) {
    fprintf(stderr, "[contract_cvc_loop] Error from malloc\n");
    EXIT(1);
  }
  for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      lexic_coords[ix].x[0] = x0;
      lexic_coords[ix].x[1] = x1;
      lexic_coords[ix].x[2] = x2;
      lexic_coords[ix].x[3] = x3;
    }}}
  }

  buffer = (double*)malloc(2*VOLUME*sizeof(double));
  if( buffer == NULL ) {
    fprintf(stderr, "[contract_m_loop] Error from malloc\n");
    EXIT(2);
  }

  memset( conn, 0, 8*T*momentum_number*sizeof(double) );

  /* loop on fields */
  for(i=0; i<field_number; i++) {
    memcpy(eo_work[3], field_list[i], sizeof_eo_spinor_field);
    /* Xbar V */
    X_eo (eo_work[0], eo_work[3], -mass, g_gauge_field);
    /* Wtilde  = Cbar V */
    C_from_Xeo (eo_work[1], eo_work[0], field_list[i], g_gauge_field, -mass);
    /* XW = X W*/
    X_eo (eo_work[2], eo_work[1], mass, g_gauge_field);

    memcpy(eo_work[3], field_list[i], sizeof_eo_spinor_field);

    /* xchange even fields */
    xchange_eo_field(eo_work[0], 0);
    xchange_eo_field(eo_work[2], 0);
    /* xchange odd fields */
    xchange_eo_field(eo_work[1], 1);
    xchange_eo_field(eo_work[3], 1);

    for(mu=0; mu<4; mu++) {

      memset(buffer, 0, 2*VOLUME*sizeof(double));
#ifdef HAVE_OPENMP
#pragma omp parallel
{         
#endif
      double spinor1[24], spinor2[24], U_[18];
      unsigned int ixeo, ix, ixpmueo;
      complex *zbuffer_ = NULL;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for(ixeo=0; ixeo<Vhalf; ixeo++) {
        /*************
         * even part *
         *************/
        ix = g_eo2lexic[ixeo];
        zbuffer_ = (complex*)(buffer + 2*ix);
        /* odd ix + mu */
        ixpmueo = g_lexic2eosub[g_iup[ix][mu]];

        _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &(co_phase_up[mu]) );
        _fv_eq_cm_ti_fv(spinor1, U_, eo_work[1]+_GSI(ixpmueo));
        _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        _fv_mi_eq_fv(spinor2, spinor1);
        _fv_ti_eq_g5(spinor2);
        _co_eq_fv_dag_ti_fv( zbuffer_, eo_work[0]+_GSI(ixeo), spinor2);

        _fv_eq_cm_dag_ti_fv(spinor1, U_, eo_work[2]+_GSI(ixeo));
        _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        _fv_pl_eq_fv(spinor2, spinor1);
        _fv_ti_eq_g5(spinor2);
        _co_pl_eq_fv_dag_ti_fv( zbuffer_, eo_work[3]+_GSI(ixpmueo), spinor2);

        _co_ti_eq_re( zbuffer_, weight[i]);
      }

#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for(ixeo=0; ixeo<Vhalf; ixeo++) {
        /*************
         * odd part *
         *************/
        /* odd ix */
        ix = g_eo2lexic[ixeo + VpRhalf];
        zbuffer_ = (complex*)(buffer + 2 * ix);
        /* even ix + mu */
        ixpmueo = g_lexic2eosub[g_iup[ix][mu]];

        _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &(co_phase_up[mu]) );
        _fv_eq_cm_ti_fv(spinor1, U_, eo_work[2]+_GSI(ixpmueo));
        _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        _fv_mi_eq_fv(spinor2, spinor1);
        _fv_ti_eq_g5(spinor2);
        _co_eq_fv_dag_ti_fv( zbuffer_, eo_work[3]+_GSI(ixeo), spinor2);

        _fv_eq_cm_dag_ti_fv(spinor1, U_, eo_work[1]+_GSI(ixeo));
        _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        _fv_pl_eq_fv(spinor2, spinor1);
        _fv_ti_eq_g5(spinor2);
        _co_pl_eq_fv_dag_ti_fv( zbuffer_, eo_work[0]+_GSI(ixpmueo), spinor2);

        _co_ti_eq_re( zbuffer_, weight[i]);
      }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

    for(imom=0; imom<momentum_number; imom++) {
      const double q[3] = { 2.*M_PI * momentum_list[imom][0] / LX_global, 2.*M_PI * momentum_list[imom][1] / LY_global, 2.*M_PI * momentum_list[imom][2] / LZ_global };
      const double q_offset = q[0] * g_proc_coords[1]*LX + q[1] * g_proc_coords[2]*LY + q[2] * g_proc_coords[3]*LZ;
#ifdef HAVE_OPENMP
#pragma omp parallel
{         
#endif
        complex w1;
        complex *zconn_ = (complex*)(conn + 2* ( mu * momentum_number + imom ) * T);
        double q_phase;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
        for(ix=0; ix<VOLUME; ix++) {
          q_phase = q_offset + q[0] * lexic_coords[ix].x[1] + q[1] * lexic_coords[ix].x[2] + q[2] * lexic_coords[ix].x[3];
          x0 = lexic_coords[ix].x[0];
          /* even part */
          w1.re = cos(q_phase); w1.im = sin(q_phase);
          _co_pl_eq_co_ti_co( zconn_+x0 , (complex*)(buffer+2*ix), &w1 );
        }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
      }  /* end of loop on momenta */  
    }  /* end of loop on mu */
  }  /* end of loop on fields */


  free(lexic_coords);
  free(buffer);
  return;
}  /* end of contract_cvc_loop */


/*************************************************************************************************
 * contract loop with eigenvectors of stochastic propagators
 * - contractions for tm-clover (only for Xeo and C_from_Xeo)
 *   what should be passed:
 *   mzz_dn[1] (odd part) for C_from_Xeo
 *   mzzinv_dn[0] (even part) for Xbar
 *   mzzinv_up[0] (even part) for X
 *************************************************************************************************/
void contract_cvc_loop_clover (double*conn, double**field_list, int field_number, int *momentum_list[3], int momentum_number, double*weight,
   double*mzz_dn, double*mzzinv_up, double*mzzinv_dn, double**eo_work) {

  typedef struct {
    int x[4];
  } point;

  const unsigned int Vhalf = VOLUME/2;
  const unsigned int VpRhalf = (VOLUME+RAND)/2;
  const size_t sizeof_eo_spinor_field = _GSI(Vhalf) * sizeof(double);

  int i, x0, x1, x2, x3, mu, imom;
  unsigned int ix;
  point *lexic_coords;
  double *buffer;

  lexic_coords = (point*)malloc(VOLUME*sizeof(point));
  if(lexic_coords == NULL) {
    fprintf(stderr, "[contract_cvc_loop] Error from malloc\n");
    EXIT(1);
  }
  for(x0=0; x0<T; x0++) {
    for(x1=0; x1<LX; x1++) {
    for(x2=0; x2<LY; x2++) {
    for(x3=0; x3<LZ; x3++) {
      ix = g_ipt[x0][x1][x2][x3];
      lexic_coords[ix].x[0] = x0;
      lexic_coords[ix].x[1] = x1;
      lexic_coords[ix].x[2] = x2;
      lexic_coords[ix].x[3] = x3;
    }}}
  }

  buffer = (double*)malloc(2*VOLUME*sizeof(double));
  if( buffer == NULL ) {
    fprintf(stderr, "[contract_m_loop] Error from malloc\n");
    EXIT(2);
  }

  memset( conn, 0, 8*T*momentum_number*sizeof(double) );

  /* loop on fields */
  for(i=0; i<field_number; i++) {
    memcpy(eo_work[3], field_list[i], sizeof_eo_spinor_field);
    /* Xbar V */
    X_clover_eo (eo_work[0], eo_work[3], g_gauge_field, mzzinv_dn);
    /* Wtilde  = Cbar V */
    C_clover_from_Xeo (eo_work[1], eo_work[0], field_list[i], g_gauge_field, mzz_dn);
    /* XW = X W*/
    X_clover_eo (eo_work[2], eo_work[1], g_gauge_field, mzzinv_up);

    memcpy(eo_work[3], field_list[i], sizeof_eo_spinor_field);

    /* xchange even fields */
    xchange_eo_field(eo_work[0], 0);
    xchange_eo_field(eo_work[2], 0);
    /* xchange odd fields */
    xchange_eo_field(eo_work[1], 1);
    xchange_eo_field(eo_work[3], 1);

    for(mu=0; mu<4; mu++) {

      memset(buffer, 0, 2*VOLUME*sizeof(double));
#ifdef HAVE_OPENMP
#pragma omp parallel
{         
#endif
      double spinor1[24], spinor2[24], U_[18];
      unsigned int ixeo, ix, ixpmueo;
      complex *zbuffer_ = NULL;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for(ixeo=0; ixeo<Vhalf; ixeo++) {
        /*************
         * even part *
         *************/
        ix = g_eo2lexic[ixeo];
        zbuffer_ = (complex*)(buffer + 2*ix);
        /* odd ix + mu */
        ixpmueo = g_lexic2eosub[g_iup[ix][mu]];

        _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &(co_phase_up[mu]) );
        _fv_eq_cm_ti_fv(spinor1, U_, eo_work[1]+_GSI(ixpmueo));
        _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        _fv_mi_eq_fv(spinor2, spinor1);
        _fv_ti_eq_g5(spinor2);
        _co_eq_fv_dag_ti_fv( zbuffer_, eo_work[0]+_GSI(ixeo), spinor2);

        _fv_eq_cm_dag_ti_fv(spinor1, U_, eo_work[2]+_GSI(ixeo));
        _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        _fv_pl_eq_fv(spinor2, spinor1);
        _fv_ti_eq_g5(spinor2);
        _co_pl_eq_fv_dag_ti_fv( zbuffer_, eo_work[3]+_GSI(ixpmueo), spinor2);

        _co_ti_eq_re( zbuffer_, weight[i]);
      }

#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for(ixeo=0; ixeo<Vhalf; ixeo++) {
        /*************
         * odd part *
         *************/
        /* odd ix */
        ix = g_eo2lexic[ixeo + VpRhalf];
        zbuffer_ = (complex*)(buffer + 2 * ix);
        /* even ix + mu */
        ixpmueo = g_lexic2eosub[g_iup[ix][mu]];

        _cm_eq_cm_ti_co(U_, g_gauge_field+_GGI(ix,mu), &(co_phase_up[mu]) );
        _fv_eq_cm_ti_fv(spinor1, U_, eo_work[2]+_GSI(ixpmueo));
        _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        _fv_mi_eq_fv(spinor2, spinor1);
        _fv_ti_eq_g5(spinor2);
        _co_eq_fv_dag_ti_fv( zbuffer_, eo_work[3]+_GSI(ixeo), spinor2);

        _fv_eq_cm_dag_ti_fv(spinor1, U_, eo_work[1]+_GSI(ixeo));
        _fv_eq_gamma_ti_fv(spinor2, mu, spinor1);
        _fv_pl_eq_fv(spinor2, spinor1);
        _fv_ti_eq_g5(spinor2);
        _co_pl_eq_fv_dag_ti_fv( zbuffer_, eo_work[0]+_GSI(ixpmueo), spinor2);

        _co_ti_eq_re( zbuffer_, weight[i]);
      }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

    for(imom=0; imom<momentum_number; imom++) {
      const double q[3] = { 2.*M_PI * momentum_list[imom][0] / LX_global, 2.*M_PI * momentum_list[imom][1] / LY_global, 2.*M_PI * momentum_list[imom][2] / LZ_global };
      const double q_offset = q[0] * g_proc_coords[1]*LX + q[1] * g_proc_coords[2]*LY + q[2] * g_proc_coords[3]*LZ;
#ifdef HAVE_OPENMP
#pragma omp parallel
{         
#endif
        complex w1;
        complex *zconn_ = (complex*)(conn + 2* ( mu * momentum_number + imom ) * T);
        double q_phase;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
        for(ix=0; ix<VOLUME; ix++) {
          q_phase = q_offset + q[0] * lexic_coords[ix].x[1] + q[1] * lexic_coords[ix].x[2] + q[2] * lexic_coords[ix].x[3];
          x0 = lexic_coords[ix].x[0];
          /* even part */
          w1.re = cos(q_phase); w1.im = sin(q_phase);
          _co_pl_eq_co_ti_co( zconn_+x0 , (complex*)(buffer+2*ix), &w1 );
        }

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
      }  /* end of loop on momenta */  
    }  /* end of loop on mu */
  }  /* end of loop on fields */


  free(lexic_coords);
  free(buffer);
  return;
}  /* end of contract_cvc_loop_clover */

}  /* end of namespace cvc */
