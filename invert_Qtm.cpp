/*********************************************
 * invert_Qtm
 * 
 * PURPOSE:
 * - solve phi = D_tm xi for given phi with D_tm
 *   the twisted-mass Dirac operator
 * - phi should be properly initialized by
 *   calling function
 *********************************************/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#include "global.h"
#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "mpi_init.h"
#include "Q_phi.h"
#include "cvc_utils.h"
#include "invert_Qtm.h"

namespace cvc {
void spinor_scalar_product_co(complex *w, double *xi, double *phi, int V) {

  int ix, iix;
  complex p, p2;
#ifdef HAVE_MPI
  complex pall;
#endif

  p2.re = 0.;
  p2.im = 0.;

  iix=0;
  for(ix=0; ix<V; ix++) {
    _co_eq_fv_dag_ti_fv(&p, xi+iix, phi+iix);
    p2.re += p.re;
    p2.im += p.im;
    iix+=24;
  }

  //fprintf(stdout, "# [spinor_scalar_product_co] %d local: %e %e\n", g_cart_id, p2.re, p2.im);

#ifdef HAVE_MPI
  pall.re=0.; pall.im=0.;
  MPI_Allreduce(&p2, &pall, 2, MPI_DOUBLE, MPI_SUM, g_cart_grid);
  w->re = pall.re;
  w->im = pall.im;
#else
  w->re = p2.re;
  w->im = p2.im;
#endif
}

void spinor_scalar_product_re(double *r, double *xi, double *phi, int V) {

  int ix, iix;
  double w;
  complex p;
#ifdef HAVE_MPI
  double wall;
#endif
  
  w = 0.;
  iix=0;
  for(ix=0; ix<V; ix++) {
    _co_eq_fv_dag_ti_fv(&p, xi+iix, phi+iix);
    w += p.re;
    iix+=24;
  }
  /* fprintf(stdout, "# [spinor_scalar_product_re] %d local: %e\n", g_cart_id, w); */
#ifdef HAVE_MPI
  wall = 0.;
  MPI_Allreduce(&w, &wall, 1, MPI_DOUBLE, MPI_SUM, g_cart_grid);
  *r = wall;
#else
  *r = w;
#endif
}  /* end of spinor_scalar_product_re */

/*************************************************
 * eo = 0 --- even subfield
 * eo = 1 --- odd subfield
 *************************************************/
void eo_spinor_spatial_scalar_product_co(complex *w, double *xi, double *phi, int eo) {

  int ix, iix, it;
  complex p[T];
  unsigned int N = VOLUME / 2;

  memset(p, 0, T*sizeof(complex));
  
  for(ix=0; ix<N; ix++) {
    iix = _GSI(ix);
    it  = g_eosub2t[eo][ix];
    _co_pl_eq_fv_dag_ti_fv( (p+it), xi+iix, phi+iix);
  }

  /* fprintf(stdout, "# [spinor_scalar_product_co] %d local: %e %e\n", g_cart_id, p2.re, p2.im); */

#ifdef HAVE_MPI
  memset(w, 0, T*sizeof(complex));
  MPI_Allreduce(p, w, 2*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
#else
  memcpy(w, p, T*sizeof(complex));
#endif
}  /* eo_spinor_spatial_scalar_product_co */

void eo_spinor_dag_gamma_spinor(complex*gsp, double*xi, int gid, double*phi) {

  unsigned int ix, iix;
  unsigned int N = VOLUME / 2;
  double spinor1[24];

  for(ix=0; ix<N; ix++) {
    iix = _GSI(ix);
    _fv_eq_gamma_ti_fv(spinor1, gid, phi+iix);
    _co_eq_fv_dag_ti_fv(gsp+ix, xi+iix, spinor1);
  }
}  /* end of eo_spinor_dag_gamma_spinor */

/*************************************************************
 * Note, that the phase field must be the even phase field for eo = 0
 * and the odd phase field for eo = 1
 *************************************************************/
void eo_gsp_momentum_projection (complex *gsp_p, complex *gsp_x, complex *phase, int eo) {
  
  unsigned int ix, it;
  unsigned int N = VOLUME / 2;
  complex p[T];

  memset(p, 0, T*sizeof(complex));

  for(ix=0; ix<N; ix++) {
    it = g_eosub2t[eo][ix];
    _co_pl_eq_co_ti_co(p+it, gsp_x+ix, phase+ix );
  }

#ifdef HAVE_MPI
  MPI_Allreduce(p, gsp_p, 2*T, MPI_DOUBLE, MPI_SUM, g_ts_comm);
#else
  memcpy(gsp_p, p, 2*T*sizeof(double));
#endif
}  /* end of eo_gsp_momentum_projection */


int invert_Qtm(double *xi, double *phi, int kwork) {

  int ix, niter, iix;
  double *r1_ptr = (double*)NULL; 
  double *r2_ptr = (double*)NULL;
  double *s_ptr  = (double*)NULL;
  double *t_ptr  = (double*)NULL;
  double *x_ptr  = (double*)NULL;
  double *p_ptr  = (double*)NULL;
  double *p2_ptr = (double*)NULL;
  double u, norm, normb;
  double spinor1[24], spinor2[24];
  complex alpha, beta, omega;
  complex w, w2, w3, r0rn;

  /*************************
   * set the fields
   *************************/
  r1_ptr = g_spinor_field[kwork];
  r2_ptr = g_spinor_field[kwork+1];
  s_ptr  = g_spinor_field[kwork+2];
  t_ptr  = g_spinor_field[kwork+3];
  p_ptr  = g_spinor_field[kwork+4];
  p2_ptr = g_spinor_field[kwork+5];
  x_ptr  = xi;

  if( r1_ptr==(double*)NULL ||  r2_ptr==(double*)NULL || s_ptr==(double*)NULL || t_ptr==(double*)NULL || 
      p_ptr==(double*)NULL || x_ptr==(double*)NULL || phi==(double*)NULL ) return(-2);


  /*************************
   * initialize
   *************************/
  
  alpha.re = 0.; alpha.im = 0.;
  beta.re  = 0.; beta.im  = 0.;
  omega.re = 0.; omega.im = 0.;
  w.re     = 0.; w.im     = 0.;
  w2.re    = 0.; w2.im    = 0.;
  w3.re    = 0.; w3.im    = 0.;
  u        = 0.;

  /* normb */
  spinor_scalar_product_re(&normb, phi, phi, VOLUME);
  if(g_cart_id==0) fprintf(stdout, "# norm of r.-h. side: %e\n", normb);

  /* p = phi - D xi */
  Q_phi_tbc(p_ptr, xi);
  iix=0;
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_fv_mi_fv(p_ptr+iix, phi+iix, p_ptr+iix);
    iix+=24;
  }
  xchange_field(p_ptr);

  /* check the norm */
  spinor_scalar_product_re(&norm, p_ptr, p_ptr, VOLUME);
/*  if(g_cart_id==0) fprintf(stdout, "norm = %25.16e\n", norm); */
  if(norm<=solver_precision*normb) {
    if(g_cart_id==0) fprintf(stdout, "start spinor solves to requested precision\n");
    return(0);
  }

  /* r1 = p = r2 */
  memcpy((void*)r1_ptr, (void*)p_ptr, 24*VOLUME*sizeof(double));
  memcpy((void*)r2_ptr, (void*)p_ptr, 24*VOLUME*sizeof(double));
  
  /* p2 = D p */
  Q_phi_tbc(p2_ptr, p_ptr);

  /*************************
   * start iteration
   *************************/
  for(niter=0; niter<=niter_max; niter++) {
        
    spinor_scalar_product_co(&r0rn, r2_ptr, r1_ptr, VOLUME);
    spinor_scalar_product_co(&w, r2_ptr, p2_ptr, VOLUME);
    _co_eq_co_ti_co_inv(&alpha, &r0rn, &w);

/*
    if(g_cart_id==0) fprintf(stdout, "\nr0rn =\t%25.16e +i %25.16e\n"\
      "w =\t%25.16e +i %25.16e\nalpha =\t%25.16e +i %25.16e\n", r0rn.re, r0rn.im, w.re, w.im, alpha.re, alpha.im);
*/

    /* the new complete s */
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_co(spinor1, p2_ptr+iix, &alpha);
      _fv_eq_fv_mi_fv(s_ptr+iix, r1_ptr+iix, spinor1);
      iix+=24;
    }
    xchange_field(s_ptr);

    /* the new t */
    Q_phi_tbc(t_ptr, s_ptr);

    spinor_scalar_product_co(&w, t_ptr, s_ptr, VOLUME);
    spinor_scalar_product_re(&u, t_ptr, t_ptr, VOLUME);
    _co_eq_co_ti_re(&omega, &w, 1./u);
/*
    if(g_cart_id==0) fprintf(stdout, "i\nw =\t%25.16e +i %25.16e\n"\
      "u =\t%25.16e\nomega =\t%25.16e +i %25.16e\n", w.re, w.im, u, omega.re, omega.im);
*/
    

    /* the new r1 */
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_co(spinor1, t_ptr+iix, &omega);
      _fv_eq_fv_mi_fv(r1_ptr+iix, s_ptr+iix, spinor1);
      iix+=24;
    }

    spinor_scalar_product_re(&norm, r1_ptr, r1_ptr, VOLUME);
    if(g_cart_id==0) fprintf(stdout, "# [%d] residuum after iteration %d: %25.16e\n", g_cart_id, niter, norm);
    if(norm<=solver_precision*normb) break;

    /* the new x */
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_co(spinor1, s_ptr+iix, &omega);
      _fv_pl_eq_fv(x_ptr+iix, spinor1);
      _fv_eq_fv_ti_co(spinor1, p_ptr+iix, &alpha);
      _fv_pl_eq_fv(x_ptr+iix, spinor1);
      iix+=24;
    }

    spinor_scalar_product_co(&w, r2_ptr, r1_ptr, VOLUME);
    _co_eq_co_ti_co_inv(&w2, &w, &r0rn);
    _co_eq_co_ti_co_inv(&w3, &alpha, &omega);
    _co_eq_co_ti_co(&beta, &w2, &w3);
    r0rn.re = w.re; r0rn.im = w.im;
/*
    if(g_cart_id==0) fprintf(stdout, "\nw =\t%25.16e +i %25.16e\n"\
      "w2 =\t%25.16e +i %25.16e\nw3 =\t%25.16e +i %25.16e\nbeta =\t%25.16e +i %25.16e\n", w.re, w.im, w2.re, w2.im, w3.re, w3.im, beta.re, beta.im);
*/

    /* the new p */
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_co(spinor1, p2_ptr+iix, &omega);
      _fv_eq_fv_mi_fv(spinor1, p_ptr+iix, spinor1);
      _fv_eq_fv_ti_co(spinor2, spinor1, &beta);
      _fv_eq_fv_pl_fv(p_ptr+iix, r1_ptr+iix, spinor2);
      iix+=24;
    }
    xchange_field(p_ptr);

    /* the new p2 */
    Q_phi_tbc(p2_ptr, p_ptr);

  }

  /*************************
   * output
   *************************/
  if(norm<=solver_precision*normb && niter<=niter_max) {
    if(g_cart_id==0) {
      fprintf(stdout, "# BiCGStab converged after %d steps with relative residuum %e\n", niter, norm/normb);
    }
  } else {
    if(g_cart_id==0) {
      fprintf(stdout, "# No convergence in BiCGStab; after %d steps relative residuum is %e\n", niter, norm/normb);
    }
    return(-3);
  }


  /*************************
   * check the solution
   *************************/
  xchange_field(x_ptr);
  Q_phi_tbc(p_ptr, x_ptr);
  iix=0;
  for(ix=0; ix<VOLUME; ix++) {
    _fv_mi_eq_fv(p_ptr+iix, phi+iix);
    iix+=24;
  }
  spinor_scalar_product_re(&norm, p_ptr, p_ptr, VOLUME);
  if(g_cart_id==0) {
    fprintf(stdout, "# true relative squared residuum is %e\n", norm/normb);
  }

  return(niter);
}

/****************************************************************
 * invert_Qtm_her
 ****************************************************************/

int invert_Qtm_her(double *xi, double *phi, int kwork) {

  int ix, niter, iix;
  double *r1_ptr = (double*)NULL; 
  double *r2_ptr = (double*)NULL;
  double *s_ptr  = (double*)NULL;
  double *t_ptr  = (double*)NULL;
  double *x_ptr  = (double*)NULL;
  double *p_ptr  = (double*)NULL;
  double *p2_ptr = (double*)NULL;
  double *aux    = (double*)NULL;
  double u, norm, normb;
  double spinor1[24], spinor2[24];
  complex alpha, beta, omega;
  complex w, w2, w3, r0rn;

  /*************************
   * set the fields
   *************************/
  r1_ptr = g_spinor_field[kwork];
  r2_ptr = g_spinor_field[kwork+1];
  s_ptr  = g_spinor_field[kwork+2];
  t_ptr  = g_spinor_field[kwork+3];
  p_ptr  = g_spinor_field[kwork+4];
  p2_ptr = g_spinor_field[kwork+5];
  x_ptr  = xi;
  aux    = g_spinor_field[kwork+6];

  if( r1_ptr==(double*)NULL ||  r2_ptr==(double*)NULL || s_ptr==(double*)NULL || t_ptr==(double*)NULL || 
      p_ptr==(double*)NULL || x_ptr==(double*)NULL || phi==(double*)NULL ) return(-2);


  /*************************
   * initialize
   *************************/
  
  alpha.re = 0.; alpha.im = 0.;
  beta.re  = 0.; beta.im  = 0.;
  omega.re = 0.; omega.im = 0.;
  w.re     = 0.; w.im     = 0.;
  w2.re    = 0.; w2.im    = 0.;
  w3.re    = 0.; w3.im    = 0.;
  u        = 0.;

  /* normb */
  spinor_scalar_product_re(&normb, phi, phi, VOLUME);
  if(g_cart_id==0) fprintf(stdout, "# norm of r.-h. side: %e\n", normb);

  /* p = gamma5 phi - g5 Du g5 Dd xi */
  Qf5(aux, xi, -g_mu);
  xchange_field(aux);
  Qf5(p_ptr, aux, g_mu);
  iix=0;
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_gamma_ti_fv(spinor1, 5, phi+iix);
    _fv_eq_fv_mi_fv(p_ptr+iix, spinor1, p_ptr+iix);
    iix+=24;
  }
  xchange_field(p_ptr);

  /* check the norm */
  spinor_scalar_product_re(&norm, p_ptr, p_ptr, VOLUME);
/*  if(g_cart_id==0) fprintf(stdout, "norm = %25.16e\n", norm); */
  if(norm<=solver_precision*normb) {
    if(g_cart_id==0) fprintf(stdout, "start spinor solves to requested precision\n");
    return(0);
  }

  /* r1 = p = r2 */
  memcpy((void*)r1_ptr, (void*)p_ptr, 24*VOLUME*sizeof(double));
  memcpy((void*)r2_ptr, (void*)p_ptr, 24*VOLUME*sizeof(double));
  
  /* p2 = D p */
  Qf5(aux, p_ptr, -g_mu);
  xchange_field(aux);
  Qf5(p2_ptr, aux, g_mu);

  /*************************
   * start iteration
   *************************/
  for(niter=0; niter<=niter_max; niter++) {
        
    spinor_scalar_product_co(&r0rn, r2_ptr, r1_ptr, VOLUME);
    spinor_scalar_product_co(&w, r2_ptr, p2_ptr, VOLUME);
    _co_eq_co_ti_co_inv(&alpha, &r0rn, &w);

/*
    if(g_cart_id==0) fprintf(stdout, "\nr0rn =\t%25.16e +i %25.16e\n"\
      "w =\t%25.16e +i %25.16e\nalpha =\t%25.16e +i %25.16e\n", r0rn.re, r0rn.im, w.re, w.im, alpha.re, alpha.im);
*/

    /* the new complete s */
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_co(spinor1, p2_ptr+iix, &alpha);
      _fv_eq_fv_mi_fv(s_ptr+iix, r1_ptr+iix, spinor1);
      iix+=24;
    }
    xchange_field(s_ptr);

    /* the new t */
    Qf5(aux, s_ptr, -g_mu);
    xchange_field(aux);
    Qf5(t_ptr, aux, g_mu);

    spinor_scalar_product_co(&w, t_ptr, s_ptr, VOLUME);
    spinor_scalar_product_re(&u, t_ptr, t_ptr, VOLUME);
    _co_eq_co_ti_re(&omega, &w, 1./u);
/*
    if(g_cart_id==0) fprintf(stdout, "i\nw =\t%25.16e +i %25.16e\n"\
      "u =\t%25.16e\nomega =\t%25.16e +i %25.16e\n", w.re, w.im, u, omega.re, omega.im);
*/
    

    /* the new r1 */
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_co(spinor1, t_ptr+iix, &omega);
      _fv_eq_fv_mi_fv(r1_ptr+iix, s_ptr+iix, spinor1);
      iix+=24;
    }

    spinor_scalar_product_re(&norm, r1_ptr, r1_ptr, VOLUME);
    if(g_cart_id==0) fprintf(stdout, "# [%d] residuum after iteration %d: %25.16e\n", g_cart_id, niter, norm);
    if(norm<=solver_precision*normb) break;

    /* the new x */
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_co(spinor1, s_ptr+iix, &omega);
      _fv_pl_eq_fv(x_ptr+iix, spinor1);
      _fv_eq_fv_ti_co(spinor1, p_ptr+iix, &alpha);
      _fv_pl_eq_fv(x_ptr+iix, spinor1);
      iix+=24;
    }

    spinor_scalar_product_co(&w, r2_ptr, r1_ptr, VOLUME);
    _co_eq_co_ti_co_inv(&w2, &w, &r0rn);
    _co_eq_co_ti_co_inv(&w3, &alpha, &omega);
    _co_eq_co_ti_co(&beta, &w2, &w3);
    r0rn.re = w.re; r0rn.im = w.im;
/*
    if(g_cart_id==0) fprintf(stdout, "\nw =\t%25.16e +i %25.16e\n"\
      "w2 =\t%25.16e +i %25.16e\nw3 =\t%25.16e +i %25.16e\nbeta =\t%25.16e +i %25.16e\n", w.re, w.im, w2.re, w2.im, w3.re, w3.im, beta.re, beta.im);
*/

    /* the new p */
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_co(spinor1, p2_ptr+iix, &omega);
      _fv_eq_fv_mi_fv(spinor1, p_ptr+iix, spinor1);
      _fv_eq_fv_ti_co(spinor2, spinor1, &beta);
      _fv_eq_fv_pl_fv(p_ptr+iix, r1_ptr+iix, spinor2);
      iix+=24;
    }
    xchange_field(p_ptr);

    /* the new p2 */
    Qf5(aux, p_ptr, -g_mu);
    xchange_field(aux);
    Qf5(p2_ptr, aux, g_mu);

  }

  /*******************************
   * get final solution
   *******************************/
  xchange_field(x_ptr);
  Qf5(aux, x_ptr, -g_mu);
  memcpy((void*)x_ptr, (void*)aux, 24*VOLUME*sizeof(double));
  xchange_field(x_ptr);

  /*************************
   * output
   *************************/
  if(norm<=solver_precision*normb && niter<=niter_max) {
    if(g_cart_id==0) {
      fprintf(stdout, "# BiCGStab converged after %d steps with relative residuum %e\n", niter, norm/normb);
    }
  } else {
    if(g_cart_id==0) {
      fprintf(stdout, "# No convergence in BiCGStab; after %d steps relative residuum is %e\n", niter, norm/normb);
    }
    return(-3);
  }


  /*************************
   * check the solution
   *************************/
  Q_phi_tbc(p_ptr, x_ptr);
  iix=0;
  for(ix=0; ix<VOLUME; ix++) {
    _fv_mi_eq_fv(p_ptr+iix, phi+iix);
    iix+=24;
  }
  spinor_scalar_product_re(&norm, p_ptr, p_ptr, VOLUME);
  if(g_cart_id==0) {
    fprintf(stdout, "# true relative squared residuum is %e\n", norm/normb);
  }

  return(niter);
}


int invert_Q_Wilson(double *xi, double *phi, int kwork) {

  int ix, niter, iix;
  double *r1_ptr = (double*)NULL; 
  double *r2_ptr = (double*)NULL;
  double *s_ptr  = (double*)NULL;
  double *t_ptr  = (double*)NULL;
  double *x_ptr  = (double*)NULL;
  double *p_ptr  = (double*)NULL;
  double *p2_ptr = (double*)NULL;
  double u, norm, normb;
  double spinor1[24], spinor2[24];
  complex alpha, beta, omega;
  complex w, w2, w3, r0rn;

  /*************************
   * set the fields
   *************************/
  r1_ptr = g_spinor_field[kwork];
  r2_ptr = g_spinor_field[kwork+1];
  s_ptr  = g_spinor_field[kwork+2];
  t_ptr  = g_spinor_field[kwork+3];
  p_ptr  = g_spinor_field[kwork+4];
  p2_ptr = g_spinor_field[kwork+5];
  x_ptr  = xi;

  if( r1_ptr==(double*)NULL ||  r2_ptr==(double*)NULL || s_ptr==(double*)NULL || t_ptr==(double*)NULL || 
      p_ptr==(double*)NULL || x_ptr==(double*)NULL || phi==(double*)NULL ) return(-2);


  /*************************
   * initialize
   *************************/
  
  alpha.re = 0.; alpha.im = 0.;
  beta.re  = 0.; beta.im  = 0.;
  omega.re = 0.; omega.im = 0.;
  w.re     = 0.; w.im     = 0.;
  w2.re    = 0.; w2.im    = 0.;
  w3.re    = 0.; w3.im    = 0.;
  u        = 0.;

  /* normb */
  spinor_scalar_product_re(&normb, phi, phi, VOLUME);
  if(g_cart_id==0) fprintf(stdout, "# norm of r.-h. side: %e\n", normb);

  /* p = phi - D xi */
  Q_Wilson_phi(p_ptr, xi);
  iix=0;
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_fv_mi_fv(p_ptr+iix, phi+iix, p_ptr+iix);
    iix+=24;
  }
  xchange_field(p_ptr);

  /* check the norm */
  spinor_scalar_product_re(&norm, p_ptr, p_ptr, VOLUME);
  //if(g_cart_id==0) fprintf(stdout, "norm = %25.16e\n", norm);
  if(norm<=solver_precision*normb) {
    if(g_cart_id==0) fprintf(stdout, "start spinor solves to requested precision\n");
      return(0);
  }

  /* r1 = p = r2 */
  memcpy((void*)r1_ptr, (void*)p_ptr, 24*VOLUME*sizeof(double));
  memcpy((void*)r2_ptr, (void*)p_ptr, 24*VOLUME*sizeof(double));
  
  /* p2 = D p */
  Q_Wilson_phi(p2_ptr, p_ptr);

  /*************************
   * start iteration
   *************************/
  for(niter=0; niter<=niter_max; niter++) {
        
    spinor_scalar_product_co(&r0rn, r2_ptr, r1_ptr, VOLUME);
    spinor_scalar_product_co(&w, r2_ptr, p2_ptr, VOLUME);
    _co_eq_co_ti_co_inv(&alpha, &r0rn, &w);


    //if(g_cart_id==0) fprintf(stdout, "\nr0rn =\t%25.16e +i %25.16e\n"\
    //  "w =\t%25.16e +i %25.16e\nalpha =\t%25.16e +i %25.16e\n", r0rn.re, r0rn.im, w.re, w.im, alpha.re, alpha.im);


    /* the new complete s */
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_co(spinor1, p2_ptr+iix, &alpha);
      _fv_eq_fv_mi_fv(s_ptr+iix, r1_ptr+iix, spinor1);
      iix+=24;
    }
    xchange_field(s_ptr);

    /* the new t */
    Q_Wilson_phi(t_ptr, s_ptr);

    spinor_scalar_product_co(&w, t_ptr, s_ptr, VOLUME);
    spinor_scalar_product_re(&u, t_ptr, t_ptr, VOLUME);
    _co_eq_co_ti_re(&omega, &w, 1./u);

    //if(g_cart_id==0) fprintf(stdout, "i\nw =\t%25.16e +i %25.16e\n"\
    //  "u =\t%25.16e\nomega =\t%25.16e +i %25.16e\n", w.re, w.im, u, omega.re, omega.im);

    

    /* the new r1 */
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_co(spinor1, t_ptr+iix, &omega);
      _fv_eq_fv_mi_fv(r1_ptr+iix, s_ptr+iix, spinor1);
      iix+=24;
    }

    spinor_scalar_product_re(&norm, r1_ptr, r1_ptr, VOLUME);
    if(g_cart_id==0) {
      fprintf(stdout, "# [%d] residuum after iteration %d: %25.16e\n", g_cart_id, niter, norm);
      fflush(stdout);
    }
    if(norm<=solver_precision*normb) break;

    /* the new x */
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_co(spinor1, s_ptr+iix, &omega);
      _fv_pl_eq_fv(x_ptr+iix, spinor1);
      _fv_eq_fv_ti_co(spinor1, p_ptr+iix, &alpha);
      _fv_pl_eq_fv(x_ptr+iix, spinor1);
      iix+=24;
    }

    spinor_scalar_product_co(&w, r2_ptr, r1_ptr, VOLUME);
    _co_eq_co_ti_co_inv(&w2, &w, &r0rn);
    _co_eq_co_ti_co_inv(&w3, &alpha, &omega);
    _co_eq_co_ti_co(&beta, &w2, &w3);
    r0rn.re = w.re; r0rn.im = w.im;


    //spinor_scalar_product_re(&norm, r1_ptr, r1_ptr, VOLUME);
    //if(g_cart_id==0) fprintf(stdout, "# [%d] (1) norm of r1 = %25.16e\n", g_cart_id, norm);
    //spinor_scalar_product_re(&norm, r2_ptr, r2_ptr, VOLUME);
    //if(g_cart_id==0) fprintf(stdout, "# [%d] (1) norm of r2 = %25.16e\n", g_cart_id, norm);
    //if(g_cart_id==0) fprintf(stdout, "\n(1) w =\t%25.16e +i %25.16e\n"\
    //  "(1) w2 =\t%25.16e +i %25.16e\n(1) w3 =\t%25.16e +i %25.16e\n(1) beta =\t%25.16e +i %25.16e\n", w.re, w.im, w2.re, w2.im, w3.re, w3.im, beta.re, beta.im);


    /* the new p */
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_co(spinor1, p2_ptr+iix, &omega);
      _fv_eq_fv_mi_fv(spinor1, p_ptr+iix, spinor1);
      _fv_eq_fv_ti_co(spinor2, spinor1, &beta);
      _fv_eq_fv_pl_fv(p_ptr+iix, r1_ptr+iix, spinor2);
      iix+=24;
    }
    xchange_field(p_ptr);

    /* the new p2 */
    Q_Wilson_phi(p2_ptr, p_ptr);

  }

  /*************************
   * output
   *************************/
  if(norm<=solver_precision*normb && niter<=niter_max) {
    if(g_cart_id==0) {
      fprintf(stdout, "# BiCGStab converged after %d steps with relative residuum %e\n", niter, norm/normb);
    }
  } else {
    if(g_cart_id==0) {
      fprintf(stdout, "# No convergence in BiCGStab; after %d steps relative residuum is %e\n", niter, norm/normb);
    }
    return(-3);
  }


  /*************************
   * check the solution
   *************************/
  xchange_field(x_ptr);
  Q_Wilson_phi(p_ptr, x_ptr);
  iix=0;
  for(ix=0; ix<VOLUME; ix++) {
    _fv_mi_eq_fv(p_ptr+iix, phi+iix);
    iix+=24;
  }
  spinor_scalar_product_re(&norm, p_ptr, p_ptr, VOLUME);
  if(g_cart_id==0) {
    fprintf(stdout, "# true relative squared residuum is %e\n", norm/normb);
  }

  return(niter);
}


/****************************************************************
 * invert_Q_Wilson_her
 ****************************************************************/

int invert_Q_Wilson_her(double *xi, double *phi, int kwork) {

  int ix, niter, iix;
  double *r_ptr = (double*)NULL; 
  double *s_ptr  = (double*)NULL;
  double *x_ptr  = (double*)NULL;
  double *p_ptr  = (double*)NULL;
  double norm, normb;
  double spinor1[24], spinor2[24];
  complex alpha, w;
  double beta, r0r0, rnrn;

  /*************************
   * set the fields
   *************************/
  r_ptr = g_spinor_field[kwork];
  s_ptr  = g_spinor_field[kwork+1];
  p_ptr  = g_spinor_field[kwork+2];
  x_ptr  = xi;

  if( r_ptr==(double*)NULL || s_ptr==(double*)NULL || p_ptr==(double*)NULL || x_ptr==(double*)NULL || phi==(double*)NULL ) return(-2);

  /*************************
   * initialize
   *************************/
  
  alpha.re = 0.; alpha.im = 0.;
  beta     = 0.;
  w.re     = 0.; w.im     = 0.;

  /* normb */
  spinor_scalar_product_re(&normb, phi, phi, VOLUME);
  if(g_cart_id==0) fprintf(stdout, "# norm of r.-h. side: %e\n", normb);

  /* p = gamma5 phi - g5 D_W xi */
  Q_g5_Wilson_phi(p_ptr, xi);
  iix=0;
  for(ix=0; ix<VOLUME; ix++) {
    _fv_eq_gamma_ti_fv(spinor1, 5, phi+iix);
    _fv_eq_fv_mi_fv(p_ptr+iix, spinor1, p_ptr+iix);
    iix+=24;
  }
  xchange_field(p_ptr);

  /* check the norm */
  spinor_scalar_product_re(&norm, p_ptr, p_ptr, VOLUME);
/*  if(g_cart_id==0) fprintf(stdout, "norm = %25.16e\n", norm); */
  if(norm<=solver_precision*normb) {
    if(g_cart_id==0) fprintf(stdout, "start spinor solves to requested precision\n");
    return(0);
  }

  /* r1 = p */
  memcpy((void*)r_ptr, (void*)p_ptr, 24*VOLUME*sizeof(double));
  
  /* r0p0 = <r0, r0> */
  spinor_scalar_product_re(&rnrn, r_ptr, r_ptr, VOLUME);

  Q_g5_Wilson_phi(s_ptr, p_ptr);

  /*************************
   * start iteration
   *************************/
  for(niter=0; niter<=niter_max; niter++) {
       
    r0r0 = rnrn;

    // w = (p, A p)
    spinor_scalar_product_co(&w, p_ptr, s_ptr, VOLUME);
    alpha.re  = rnrn / ( w.re*w.re + w.im*w.im );
    alpha.im  = -alpha.re * w.im;
    alpha.re *= w.re;

    /* the new r */
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_co(spinor1, s_ptr+iix, &alpha);
      _fv_eq_fv_mi_fv(r_ptr+iix, r_ptr+iix, spinor1);
      iix+=24;
    }


    /* the new x */
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_co(spinor1, p_ptr+iix, &alpha);
      _fv_pl_eq_fv(x_ptr+iix, spinor1);
      iix+=24;
    }

    spinor_scalar_product_re(&rnrn, r_ptr, r_ptr, VOLUME);
    if(g_cart_id==0) fprintf(stdout, "# [%d] residuum after iteration %d: %25.16e\n", g_cart_id, niter, rnrn);
    if(rnrn<=solver_precision*normb) {
      norm = rnrn;
      break;
    }

    /* the new p */
    beta = rnrn / r0r0;
    iix=0;
    for(ix=0; ix<VOLUME; ix++) {
      _fv_eq_fv_ti_re(spinor1, p_ptr+iix, beta);
      _fv_eq_fv_pl_fv(p_ptr+iix, r_ptr+iix, spinor1);
      iix+=24;
    }
    xchange_field(p_ptr);

    /* the new s */
    Q_g5_Wilson_phi(s_ptr, p_ptr);

  }  // of loop on niter

  /*******************************
   * exchane final solution
   *******************************/
  xchange_field(x_ptr);

  /*************************
   * output
   *************************/
  if(norm<=solver_precision*normb && niter<=niter_max) {
    if(g_cart_id==0) {
      fprintf(stdout, "# CGher converged after %d steps with relative residuum %e\n", niter, norm/normb);
    }
  } else {
    if(g_cart_id==0) {
      fprintf(stdout, "# No convergence in CGher; after %d steps relative residuum is %e\n", niter, norm/normb);
    }
    return(-3);
  }

  /*************************
   * check the solution
   *************************/
  Q_Wilson_phi(p_ptr, x_ptr);
  iix=0;
  for(ix=0; ix<VOLUME; ix++) {
    _fv_mi_eq_fv(p_ptr+iix, phi+iix);
    iix+=24;
  }
  spinor_scalar_product_re(&norm, p_ptr, p_ptr, VOLUME);
  if(g_cart_id==0) {
    fprintf(stdout, "# true relative squared residuum is %e\n", norm/normb);
  }

  return(niter);
}

}  /* end of namespace cvc */
