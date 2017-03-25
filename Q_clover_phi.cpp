/********************
 * Q_clover_phi.c
 *
 * Mi 15. Jun 16:17:29 CEST 2016
 *
 ********************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include "cvc_complex.h"
#include "global.h"
#include "cvc_linalg.h"
#include "laplace_linalg.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "Q_phi.h"
#include "Q_clover_phi.h"

#ifdef F_
#define _F(s) s##_
#else
#define _F(s) s
#endif

#define ZGESV  _F(zgesv)

extern "C" void ZGESV(int* n, int* nrhs, _Complex double a[], int* lda, int ipivot[], _Complex double b[], int* ldb, int *info);


/* So 19. Jun 12:23:23 CEST 2016 */
/* pack 6x6 matrix in _Complex double from three 3x3 in cvc complex */
#define _PACK_MATRIX(a_,b_,c_,d_) {\
  (a_)[ 0] = (b_)[ 0] + (b_)[ 1] * I; \
  (a_)[ 1] = (b_)[ 2] + (b_)[ 3] * I; \
  (a_)[ 2] = (b_)[ 4] + (b_)[ 5] * I; \
  (a_)[ 3] = (c_)[ 0] + (c_)[ 1] * I; \
  (a_)[ 4] = (c_)[ 2] + (c_)[ 3] * I; \
  (a_)[ 5] = (c_)[ 4] + (c_)[ 5] * I; \
  (a_)[ 6] = (b_)[ 6] + (b_)[ 7] * I; \
  (a_)[ 7] = (b_)[ 8] + (b_)[ 9] * I; \
  (a_)[ 8] = (b_)[10] + (b_)[11] * I; \
  (a_)[ 9] = (c_)[ 6] + (c_)[ 7] * I; \
  (a_)[10] = (c_)[ 8] + (c_)[ 9] * I; \
  (a_)[11] = (c_)[10] + (c_)[11] * I; \
  (a_)[12] = (b_)[12] + (b_)[13] * I; \
  (a_)[13] = (b_)[14] + (b_)[15] * I; \
  (a_)[14] = (b_)[16] + (b_)[17] * I; \
  (a_)[15] = (c_)[12] + (c_)[13] * I; \
  (a_)[16] = (c_)[14] + (c_)[15] * I; \
  (a_)[17] = (c_)[16] + (c_)[17] * I; \
  (a_)[18] = (c_)[ 0] - (c_)[ 1] * I; \
  (a_)[19] = (c_)[ 6] - (c_)[ 7] * I; \
  (a_)[20] = (c_)[12] - (c_)[13] * I; \
  (a_)[21] = (d_)[ 0] + (d_)[ 1] * I; \
  (a_)[22] = (d_)[ 2] + (d_)[ 3] * I; \
  (a_)[23] = (d_)[ 4] + (d_)[ 5] * I; \
  (a_)[24] = (c_)[ 2] - (c_)[ 3] * I; \
  (a_)[25] = (c_)[ 8] - (c_)[ 9] * I; \
  (a_)[26] = (c_)[14] - (c_)[15] * I; \
  (a_)[27] = (d_)[ 6] + (d_)[ 7] * I; \
  (a_)[28] = (d_)[ 8] + (d_)[ 9] * I; \
  (a_)[29] = (d_)[10] + (d_)[11] * I; \
  (a_)[30] = (c_)[ 4] - (c_)[ 5] * I; \
  (a_)[31] = (c_)[10] - (c_)[11] * I; \
  (a_)[32] = (c_)[16] - (c_)[17] * I; \
  (a_)[33] = (d_)[12] + (d_)[13] * I; \
  (a_)[34] = (d_)[14] + (d_)[15] * I; \
  (a_)[35] = (d_)[16] + (d_)[17] * I; \
}


/* So 19. Jun 12:34:57 CEST 2016 */
/* unpack 6x6 matrix in _Complex double to four 3x3 in cvc complex */
#define _UNPACK_MATRIX(a_,b_,c_,d_,e_) {\
  (b_)[ 0] = creal( (a_)[ 0] );  (b_)[ 1] = cimag( (a_)[ 0] ); \
  (b_)[ 2] = creal( (a_)[ 1] );  (b_)[ 3] = cimag( (a_)[ 1] ); \
  (b_)[ 4] = creal( (a_)[ 2] );  (b_)[ 5] = cimag( (a_)[ 2] ); \
  (b_)[ 6] = creal( (a_)[ 6] );  (b_)[ 7] = cimag( (a_)[ 6] ); \
  (b_)[ 8] = creal( (a_)[ 7] );  (b_)[ 9] = cimag( (a_)[ 7] ); \
  (b_)[10] = creal( (a_)[ 8] );  (b_)[11] = cimag( (a_)[ 8] ); \
  (b_)[12] = creal( (a_)[12] );  (b_)[13] = cimag( (a_)[12] ); \
  (b_)[14] = creal( (a_)[13] );  (b_)[15] = cimag( (a_)[13] ); \
  (b_)[16] = creal( (a_)[14] );  (b_)[17] = cimag( (a_)[14] ); \
  (c_)[ 0] = creal( (a_)[ 3] );  (c_)[ 1] = cimag( (a_)[ 3] ); \
  (c_)[ 2] = creal( (a_)[ 4] );  (c_)[ 3] = cimag( (a_)[ 4] ); \
  (c_)[ 4] = creal( (a_)[ 5] );  (c_)[ 5] = cimag( (a_)[ 5] ); \
  (c_)[ 6] = creal( (a_)[ 9] );  (c_)[ 7] = cimag( (a_)[ 9] ); \
  (c_)[ 8] = creal( (a_)[10] );  (c_)[ 9] = cimag( (a_)[10] ); \
  (c_)[10] = creal( (a_)[11] );  (c_)[11] = cimag( (a_)[11] ); \
  (c_)[12] = creal( (a_)[15] );  (c_)[13] = cimag( (a_)[15] ); \
  (c_)[14] = creal( (a_)[16] );  (c_)[15] = cimag( (a_)[16] ); \
  (c_)[16] = creal( (a_)[17] );  (c_)[17] = cimag( (a_)[17] ); \
  (d_)[ 0] = creal( (a_)[18] );  (d_)[ 1] = cimag( (a_)[18] ); \
  (d_)[ 2] = creal( (a_)[19] );  (d_)[ 3] = cimag( (a_)[19] ); \
  (d_)[ 4] = creal( (a_)[20] );  (d_)[ 5] = cimag( (a_)[20] ); \
  (d_)[ 6] = creal( (a_)[24] );  (d_)[ 7] = cimag( (a_)[24] ); \
  (d_)[ 8] = creal( (a_)[25] );  (d_)[ 9] = cimag( (a_)[25] ); \
  (d_)[10] = creal( (a_)[26] );  (d_)[11] = cimag( (a_)[26] ); \
  (d_)[12] = creal( (a_)[30] );  (d_)[13] = cimag( (a_)[30] ); \
  (d_)[14] = creal( (a_)[31] );  (d_)[15] = cimag( (a_)[31] ); \
  (d_)[16] = creal( (a_)[32] );  (d_)[17] = cimag( (a_)[32] ); \
  (e_)[ 0] = creal( (a_)[21] );  (e_)[ 1] = cimag( (a_)[21] ); \
  (e_)[ 2] = creal( (a_)[22] );  (e_)[ 3] = cimag( (a_)[22] ); \
  (e_)[ 4] = creal( (a_)[23] );  (e_)[ 5] = cimag( (a_)[23] ); \
  (e_)[ 6] = creal( (a_)[27] );  (e_)[ 7] = cimag( (a_)[27] ); \
  (e_)[ 8] = creal( (a_)[28] );  (e_)[ 9] = cimag( (a_)[28] ); \
  (e_)[10] = creal( (a_)[29] );  (e_)[11] = cimag( (a_)[29] ); \
  (e_)[12] = creal( (a_)[33] );  (e_)[13] = cimag( (a_)[33] ); \
  (e_)[14] = creal( (a_)[34] );  (e_)[15] = cimag( (a_)[34] ); \
  (e_)[16] = creal( (a_)[35] );  (e_)[17] = cimag( (a_)[35] ); \
}

namespace cvc {

/***********************************************************
 * calculate the clover term
 *
 * input gauge_field, N
 * output s
 *
 * ATTENTION: s = i G_munu is calculated; s is antisymmetric in (mu,nu)
 *            but hermitean
 ***********************************************************/

void clover_term_fini(double***s) {
  if(*s != NULL) {
    if((*s)[0] != NULL) {
      free( (*s)[0] );
    }
    free(*s);
  }
}  /* end of clover_term_fini */

void clover_term_init (double***s, int nmat) {

  /* why VOLUME+RAND ? VOLUME should be enough 
   * Yes, but the index layout in g_lexic2eosub, which is used below,
   * is even interior - even halo - odd interior - odd halo.
   * IF this indexing is NOT used later on in application
   * of g_clover, it can be changed to g_lexic2eosub + g_iseven during
   * construction
   */
  /* const size_t N = (size_t)(VOLUME+RAND) / 2; */
  const size_t N = (size_t)VOLUME / 2;
  const size_t sizeof_su3 = 18;

  if ( (*s) == NULL) {
    if(g_cart_id == 0) fprintf(stdout, "# [clover_term_init] allocating clover term\n");

    (*s) = (double**)malloc(2*sizeof(double*));
    if( (*s)==NULL) {
      fprintf(stderr, "[clover_term_init] Error from malloc\n");
      EXIT(1);
    }
    (*s)[0] = (double*)malloc(2*N*nmat*sizeof_su3*sizeof(double));
    if( (*s)[0]==NULL) {
      fprintf(stderr, "[clover_term_init] Error from malloc\n");
      EXIT(2);
    }
    (*s)[1] = (*s)[0] + N * sizeof_su3 * nmat;
  }
}  /* end of clover_term_init */

void clover_term_eo (double**s, double*gauge_field) {

  const double norm  = 0.25;

#ifdef HAVE_OPENMP
#pragma omp parallel shared(s,gauge_field)
{
#endif
  unsigned int ix;
  unsigned int ix_pl_mu, ix_mi_mu, ix_pl_nu, ix_mi_nu, ix_mi_mu_mi_nu, ix_mi_mu_pl_nu, ix_pl_mu_mi_nu;
  int imu, inu, imunu;
  double *s_ptr;
  double U1[18], U2[18], U3[18];
  int ieo;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix = 0; ix < VOLUME; ix++ )
  {
    /* ieo = 0 for ix even and 1 for ix odd */
    ieo = 1 - g_iseven[ix];
    imunu = 0;
    for(imu = 0; imu<3; imu++) {

      ix_pl_mu = g_iup[ix][imu];
      ix_mi_mu = g_idn[ix][imu];

      for(inu = imu+1; inu<4; inu++) {

        /* s_ptr = s[0] + _GSWI( g_lexic2eo[ix],imunu); */
        s_ptr = s[ieo] + _GSWI( g_lexic2eosub[ix],imunu);

        ix_pl_nu = g_iup[ix][inu];
        ix_mi_nu = g_idn[ix][inu];

        ix_mi_mu_mi_nu = g_idn[ix_mi_mu][inu];
        ix_mi_mu_pl_nu = g_iup[ix_mi_mu][inu];
        ix_pl_mu_mi_nu = g_idn[ix_pl_mu][inu];

        _cm_eq_zero(s_ptr);

        /* fprintf(stdout, "# [clover_term] xle=%8d xeo=%8d imu=%d inu=%d imunu=%d\n", ix, g_lexic2eo[ix], imu, inu, imunu); */

        /********************************
         *    x + nu
         *     ____  x + mu+nu
         *    |    |
         *    |    | ^
         *   _|____|
         *    |x     x + mu
         *
         ********************************/
        _cm_eq_cm_ti_cm(U1, gauge_field+_GGI(ix,imu), gauge_field+_GGI(ix_pl_mu,inu) );
        _cm_eq_cm_ti_cm(U2, gauge_field+_GGI(ix,inu), gauge_field+_GGI(ix_pl_nu,imu) );
        _cm_eq_cm_ti_cm_dag( U3 , U1, U2 );
        _cm_pl_eq_cm( s_ptr , U3 );

        /********************************
         *    x 
         *   _|____  x + mu
         *    |    |
         *    |    | ^
         *    |____|
         *           x + mu - nu
         *    x - nu
         *
         ********************************/
      /*
        _cm_eq_cm_ti_cm_dag(U1, gauge_field+_GGI(ix,imu), gauge_field+_GGI(ix_pl_mu_mi_nu, inu) );
        _cm_eq_cm_dag_ti_cm(U2, gauge_field+_GGI(ix_mi_nu, imu), gauge_field+_GGI(ix_mi_nu,inu) );
       */
        _cm_eq_cm_dag_ti_cm(U1, gauge_field+_GGI(ix_mi_nu,inu), gauge_field+_GGI(ix_mi_nu, imu) );
        _cm_eq_cm_ti_cm_dag(U2, gauge_field+_GGI(ix_pl_mu_mi_nu, inu), gauge_field+_GGI(ix, imu) );
        _cm_eq_cm_ti_cm(U3, U1, U2 );
        _cm_pl_eq_cm( s_ptr , U3 );


        /********************************
         *    x-mu+nu 
         *     ____  x + nu
         *    |    |
         *    |    | ^
         *    |____|_
         *         | x
         *    x-mu
         *
         ********************************/
        _cm_eq_cm_ti_cm_dag(U1, gauge_field+_GGI(ix, inu), gauge_field+_GGI(ix_mi_mu_pl_nu, imu) );
        _cm_eq_cm_dag_ti_cm(U2, gauge_field+_GGI(ix_mi_mu,inu), gauge_field+_GGI(ix_mi_mu, imu) );
        _cm_eq_cm_ti_cm(U3, U1, U2 );
        _cm_pl_eq_cm( s_ptr , U3 );

        /********************************
         *    x-mu
         *     ____|_  x
         *    |    |
         *    |    | ^
         *    |____| x-nu
         *  x-mu-nu 
         *    
         ********************************/
        /*
        _cm_eq_cm_ti_cm(U1, gauge_field+_GGI(ix_mi_mu_mi_nu, imu), gauge_field+_GGI(ix_mi_nu, inu) );
        _cm_eq_cm_ti_cm(U2, gauge_field+_GGI(ix_mi_mu_mi_nu, inu),  gauge_field+_GGI(ix_mi_mu, imu) );
        */
        _cm_eq_cm_ti_cm(U1, gauge_field+_GGI(ix_mi_mu_mi_nu, inu), gauge_field+_GGI(ix_mi_mu, imu) );
        _cm_eq_cm_ti_cm(U2, gauge_field+_GGI(ix_mi_mu_mi_nu, imu),  gauge_field+_GGI(ix_mi_nu, inu) );
        _cm_eq_cm_dag_ti_cm(U3, U1, U2 );
        _cm_pl_eq_cm( s_ptr , U3 );

        _cm_eq_antiherm_cm(U3, s_ptr);

        /* TEST */
        _cm_eq_cm_ti_re( s_ptr , U3, norm );
        /* _cm_eq_cm_ti_im( s_ptr , U3, norm ); */

        imunu++;
      }  /* end of loop on nu */
    }    /* end of loop on mu */
  }      /* end of loop on ix */

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of clover_term */

void clover_mzz_matrix (double**mzz, double**cl, double mu, double csw) {

  const double mutilde = 2. * g_kappa * mu;
  const double cswtilde = g_kappa * csw;
  const int incrcl = _GSWI(0,1);
  const unsigned int N = VOLUME/2;

#ifdef HAVE_OPENMP
#pragma omp parallel shared(mzz,cl)
{
#endif

  unsigned int ix, ieo;
  double U1[18];
  double *mzz11_, *mzz12_, *mzz22_, *mzz33_, *mzz34_, *mzz44_;
  double *cl01_, *cl02_, *cl03_, *cl12_, *cl13_, *cl23_;
  complex cone_pl_imutilde, cone_mi_imutilde;

  cone_pl_imutilde.re = 1.;
  cone_pl_imutilde.im =  mutilde;
  cone_mi_imutilde.re = 1.;
  cone_mi_imutilde.im = -mutilde;

  for(ieo = 0; ieo<2; ieo++) {
#ifdef HAVE_OPENMP
#pragma omp for
#endif
    for(ix = 0; ix < N; ix++ ) {
  
      cl01_ = cl[ieo] + _GSWI(ix,0);
      cl02_ = cl01_ + incrcl;
      cl03_ = cl02_ + incrcl;
      cl12_ = cl03_ + incrcl;
      cl13_ = cl12_ + incrcl;
      cl23_ = cl13_ + incrcl;
  
      mzz11_ = mzz[ieo] + _GSWI(ix,0);
      mzz12_ = mzz11_ + incrcl;
      mzz22_ = mzz12_ + incrcl;
      mzz33_ = mzz22_ + incrcl;
      mzz34_ = mzz33_ + incrcl;
      mzz44_ = mzz34_ + incrcl;
  
      /* T_11 */
      _cm_eq_cm_mi_cm(U1, cl03_, cl12_ );
      _cm_eq_id_ti_co(mzz11_, cone_pl_imutilde);
      _cm_pl_eq_cm_ti_im(mzz11_, U1, cswtilde );
  
      /* T_22 */
      _cm_eq_id_ti_co(mzz22_, cone_pl_imutilde);
      _cm_pl_eq_cm_ti_im(mzz22_, U1, -cswtilde );
  
      /* T_12 */
      _cm_eq_cm_mi_cm(U1, cl01_, cl23_);
      _cm_eq_cm_pl_cm(mzz12_, cl02_, cl13_);
      _cm_pl_eq_cm_ti_im(mzz12_, U1, 1.);
      _cm_ti_eq_re(mzz12_, cswtilde);
  
      /* T_33 */
      _cm_eq_cm_pl_cm(U1, cl03_, cl12_ );
      _cm_eq_id_ti_co(mzz33_, cone_mi_imutilde);
      _cm_pl_eq_cm_ti_im(mzz33_, U1, -cswtilde );
  
      /* T_44 */
      _cm_eq_id_ti_co(mzz44_, cone_mi_imutilde);
      _cm_pl_eq_cm_ti_im(mzz44_, U1,  cswtilde );
  
      /* T_34 */
      _cm_eq_cm_pl_cm(U1, cl01_, cl23_);
      _cm_eq_cm_mi_cm(mzz34_, cl02_, cl13_);
      _cm_pl_eq_cm_ti_im(mzz34_, U1, 1.);
      _cm_ti_eq_re(mzz34_, -cswtilde);
  
    }  /* end of loop on ix */
  }  /* end of loop on ieo */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of clover_mzz_maxtrix */


/***********************************************************
 * invert the M_zz matrix
 ***********************************************************/
void clover_mzz_inv_matrix (double**mzzinv, double**mzz) {
  
  const int incrcl = _GSWI(0,1);
  const unsigned int Vhalf = VOLUME / 2;

  unsigned int ix, ieo;
  /* int i; */
  _Complex double A[36], B[36];
  int N     = 6;
  int NRHS  = 6;
  int LDA   = 6;
  int IPIV[6];
  int LDB   = 6;
  int INFO;
  double *mzz11_, *mzz12_, *mzz22_, *mzz33_, *mzz34_, *mzz44_;
  double *mzzinv11_, *mzzinv12_, *mzzinv21_, *mzzinv22_, *mzzinv33_, *mzzinv34_, *mzzinv43_, *mzzinv44_;

  for(ieo=0; ieo<2; ieo++) {
    for(ix = 0; ix < Vhalf; ix++) {
  
      /* set matrix A */
      mzz11_ = mzz[ieo] + _GSWI(ix,0);
      mzz12_ = mzz11_ + incrcl;
      mzz22_ = mzz12_ + incrcl;
      mzz33_ = mzz22_ + incrcl;
      mzz34_ = mzz33_ + incrcl;
      mzz44_ = mzz34_ + incrcl;
  
      mzzinv11_ = mzzinv[ieo] + 8 * incrcl * ix;
      mzzinv12_ = mzzinv11_ + incrcl;
      mzzinv21_ = mzzinv12_ + incrcl;
      mzzinv22_ = mzzinv21_ + incrcl;
      mzzinv33_ = mzzinv22_ + incrcl;
      mzzinv34_ = mzzinv33_ + incrcl;
      mzzinv43_ = mzzinv34_ + incrcl;
      mzzinv44_ = mzzinv43_ + incrcl;
  
      /*********************************
       * upper left block
       *********************************/
      _PACK_MATRIX(A,mzz11_,mzz12_,mzz22_);
  
      /* TEST */
  /*    
      fprintf(stdout, "A <- array(dim=c(6,6))\n");
      for(i=0; i<6; i++) {
        fprintf(stdout, "A[%d,] = c( %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i)\n", i+1, 
            creal(A[6*i]), cimag(A[6*i]), creal(A[6*i+1]), cimag(A[6*i+1]), creal(A[6*i+2]), cimag(A[6*i+2]), creal(A[6*i+3]), cimag(A[6*i+3]), creal(A[6*i+4]), cimag(A[6*i+4]), creal(A[6*i+5]), cimag(A[6*i+5]) );
      }
  */
      /* set matrix B */
      memset(B, 0, 36*sizeof(_Complex double));
      B[ 0] = 1.0; B[ 7] = 1.0; B[14] = 1.0; B[21] = 1.0; B[28] = 1.0; B[35] = 1.0;
  
      /* SUBROUTINE ZGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO ) */
      ZGESV (&N, &NRHS, A, &LDA, IPIV, B, &LDB, &INFO);
      if(INFO != 0) {
        fprintf(stderr, "[clover_mzz_inv_matrix] Error from zgesv, status was %d\n", INFO);
        EXIT(2);
      }
      /* TEST */
  /*
      fprintf(stdout, "B <- array(dim=c(6,6))\n");
      for(i=0; i<6; i++) {
        fprintf(stdout, "B[%d,] <- c( %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i)\n", i+1, 
            creal(B[6*i]), cimag(B[6*i]), creal(B[6*i+1]), cimag(B[6*i+1]), creal(B[6*i+2]), cimag(B[6*i+2]), creal(B[6*i+3]), cimag(B[6*i+3]), creal(B[6*i+4]), cimag(B[6*i+4]), creal(B[6*i+5]), cimag(B[6*i+5]) );
      }
  */
      _UNPACK_MATRIX(B, mzzinv11_, mzzinv12_, mzzinv21_, mzzinv22_);
  
      /*********************************
       * lower right block
       *********************************/
  
      /* set matrix A*/
      _PACK_MATRIX(A,mzz33_,mzz34_,mzz44_);
  
      /* TEST */
  /*
      fprintf(stdout, "A <- array(dim=c(6,6))\n");
      for(i=0; i<6; i++) {
        fprintf(stdout, "A[%d,] = c( %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i)\n", i+1, 
            creal(A[6*i]), cimag(A[6*i]), creal(A[6*i+1]), cimag(A[6*i+1]), creal(A[6*i+2]), cimag(A[6*i+2]), creal(A[6*i+3]), cimag(A[6*i+3]), creal(A[6*i+4]), cimag(A[6*i+4]), creal(A[6*i+5]), cimag(A[6*i+5]) );
      }
  */
  
      /* set matrix B */
      memset(B, 0, 36*sizeof(_Complex double));
      B[ 0] = 1.0; B[ 7] = 1.0; B[14] = 1.0; B[21] = 1.0; B[28] = 1.0; B[35] = 1.0;
  
      /* SUBROUTINE ZGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO ) */
      ZGESV (&N, &NRHS, A, &LDA, IPIV, B, &LDB, &INFO);
      if(INFO != 0) {
        fprintf(stderr, "[clover_mzz_inv_matrix] Error from zgesv, status was %d\n", INFO);
        EXIT(3);
      }
  
      /* TEST */
  /*
      fprintf(stdout, "B <- array(dim=c(6,6))\n");
      for(i=0; i<6; i++) {
        fprintf(stdout, "B[%d,] <- c( %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i, %25.16e + %25.16e*1.i)\n", i+1, 
            creal(B[6*i]), cimag(B[6*i]), creal(B[6*i+1]), cimag(B[6*i+1]), creal(B[6*i+2]), cimag(B[6*i+2]), creal(B[6*i+3]), cimag(B[6*i+3]), creal(B[6*i+4]), cimag(B[6*i+4]), creal(B[6*i+5]), cimag(B[6*i+5]) );
      }
  */
  
      _UNPACK_MATRIX(B, mzzinv33_, mzzinv34_, mzzinv43_, mzzinv44_);
  
    }  /* end of loop on ix */
  }  /* end of loop on ieo */

}  /* end of clover_mzz_inv_maxtrix */


/***********************************************************
 * apply clover term as M_zz matrix
 * safe, when s = r (r copied before modification of s)
 ***********************************************************/
void M_clover_zz_matrix (double*s, double*r, double*mzz) {

  const unsigned int N = VOLUME/2;
  const int incrcl = _GSWI(0,1);
  const int vincr  = _GVI(1);
  const double one_over_two_kappa = 0.5/g_kappa;

#ifdef HAVE_OPENMP
#pragma omp parallel shared(s,r,mzz)
{
#endif

  unsigned int ix;
  double *r1_, *r2_, *r3_, *r4_;
  double *s1_, *s2_, *s3_, *s4_;
  double v1[6], v2[6];
  double *mzz11_, *mzz12_, *mzz22_, *mzz33_, *mzz34_, *mzz44_;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix = 0; ix < N; ix++ ) {
    mzz11_ = mzz + _GSWI(ix,0);
    mzz12_ = mzz11_ + incrcl;
    mzz22_ = mzz12_ + incrcl;
    mzz33_ = mzz22_ + incrcl;
    mzz34_ = mzz33_ + incrcl;
    mzz44_ = mzz34_ + incrcl;

    s1_ = s + _GSI(ix);
    s2_ = s1_ +  vincr;
    s3_ = s2_ +  vincr;
    s4_ = s3_ +  vincr;

    r1_ = r + _GSI(ix);
    r2_ = r1_ +  vincr;
    r3_ = r2_ +  vincr;
    r4_ = r3_ +  vincr;

    _cv_eq_cv(v1, r1_);
    _cv_eq_cv(v2, r2_);

    _cv_eq_cm_ti_cv    ( s1_ , mzz11_, v1);
    _cv_pl_eq_cm_ti_cv ( s1_ , mzz12_, v2);

    _cv_eq_cm_ti_cv        ( s2_ , mzz22_, v2);
    _cv_pl_eq_cm_dag_ti_cv ( s2_ , mzz12_, v1);

    _cv_eq_cv(v1, r3_);
    _cv_eq_cv(v2, r4_);

    _cv_eq_cm_ti_cv    ( s3_ , mzz33_, v1);
    _cv_pl_eq_cm_ti_cv ( s3_ , mzz34_, v2);

    _cv_eq_cm_ti_cv        ( s4_ , mzz44_, v2);
    _cv_pl_eq_cm_dag_ti_cv ( s4_ , mzz34_, v1);
    
    /* s_ *= 1/2kappa */
    _fv_ti_eq_re(s1_, one_over_two_kappa);

  }  /* end of loop on ix */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of M_clover_zz_matrix */


/***********************************************************
 * Dirac operator with clover term
 * M_zz in block matrix form
 *
 * - e_old and o_old MUST have halo
 ***********************************************************/
void Q_clover_phi_matrix_eo (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double *aux, double**mzz) {

  const unsigned int N = VOLUME / 2;

  /* e_new = M_ee e_old + M_eo o_old */
  Hopping_eo(e_new, o_old, gauge_field, 0);
  /* aux = M_ee e_old */
  M_clover_zz_matrix (aux, e_old, mzz[0]);

  /* e_new = e_new + aux = M_ee e_old + M_eo o_old */
  spinor_field_pl_eq_spinor_field ( e_new, aux, N);

  /* o_new = M_oo o_old + M_oe e_old */
  Hopping_eo(o_new, e_old, gauge_field, 1);
  /* aux = M_oo o_old*/
  M_clover_zz_matrix (aux, o_old, mzz[1]);
  /* o_new  = o_new + aux = M_oe e_old + M_oo o_old */
  spinor_field_pl_eq_spinor_field ( o_new, aux, N );

}  /* end of Q_clover_phi_matrix_eo */


/***********************************************************
 * Dirac operator with clover term
 ***********************************************************/
void Q_clover_phi_eo (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double mass, double *aux, double**cl) {

  const unsigned int N = VOLUME / 2;

  /* e_new = M_ee e_old + M_eo o_old */
  Hopping_eo(e_new, o_old, gauge_field, 0);
  /* aux = M_ee e_old */
  M_clover_zz (aux, e_old, mass, cl[0]);

  /* e_new = e_new + aux = M_ee e_old + M_eo o_old */
  spinor_field_pl_eq_spinor_field ( e_new, aux, N);

  /* o_new = M_oo o_old + M_oe e_old */
  Hopping_eo(o_new, e_old, gauge_field, 1);
  /* aux = M_oo o_old*/
  M_clover_zz (aux, o_old, mass, cl[1]);
  /* o_new  = o_new + aux = M_oe e_old + M_oo o_old */
  spinor_field_pl_eq_spinor_field ( o_new, aux, N);

}  /* end of Q_clover_phi_eo */


/***********************************************************
 * M_ee and M_oo
 ***********************************************************/
void M_clover_zz (double*s, double*r, double mass, double*cl) {

  const double mutilde            = 2. * g_kappa * mass;
  const double one_over_two_kappa = 0.5/g_kappa;
  const double csw_coeff = -g_csw * g_kappa;
  const int clover_term_gamma_id[] = {10,11,12,13,14,15};
  const int incrcl2 = _GSWI(0,1);
  const unsigned int N = VOLUME/2;

#ifdef HAVE_OPENMP
#pragma omp parallel default(shared) shared(s,r,cl)
{
#endif
  unsigned int ix;
  double *cl_, sp1[24], sp2[24], sp3[24], *s_= NULL, *r_ = NULL;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix = 0; ix < N; ix++) {
    s_  = s  + _GSI(ix);
    r_  = r  + _GSI(ix);
    cl_ = cl + _GSWI(ix,0);;

    /* sp1 = g5 r_ */
    _fv_eq_gamma_ti_fv(sp1, 5, r_);
    /* s_ = i mass sp1 */
    _fv_eq_fv_ti_im(s_, sp1, mutilde);
    /* s_ += (1 + i mass g5) r_ */
    _fv_pl_eq_fv(s_, r_);

    /* application of clover term */

    _fv_eq_cm_ti_fv(sp1, cl_, r_ );
    _fv_eq_gamma_ti_fv(sp3, clover_term_gamma_id[0], sp1);
    cl_ += incrcl2;


    _fv_eq_cm_ti_fv(sp1, cl_, r_ );
    _fv_eq_gamma_ti_fv(sp2, clover_term_gamma_id[1], sp1);
    _fv_pl_eq_fv(sp3, sp2);
    cl_ += incrcl2;


    _fv_eq_cm_ti_fv(sp1, cl_, r_ );
    _fv_eq_gamma_ti_fv(sp2, clover_term_gamma_id[2], sp1);
    _fv_pl_eq_fv(sp3, sp2);
    cl_ += incrcl2;


    _fv_eq_cm_ti_fv(sp1, cl_, r_ );
    _fv_eq_gamma_ti_fv(sp2, clover_term_gamma_id[3], sp1);
    _fv_pl_eq_fv(sp3, sp2);
    cl_ += incrcl2;


    _fv_eq_cm_ti_fv(sp1, cl_, r_ );
    _fv_eq_gamma_ti_fv(sp2, clover_term_gamma_id[4], sp1);
    _fv_pl_eq_fv(sp3, sp2);
    cl_ += incrcl2;


    _fv_eq_cm_ti_fv(sp1, cl_, r_ );
    _fv_eq_gamma_ti_fv(sp2, clover_term_gamma_id[5], sp1);
    _fv_pl_eq_fv(sp3, sp2);

    _fv_ti_eq_re(sp3, csw_coeff);
    _fv_pl_eq_fv(s_, sp3);

    /* s_ *= 1/2kappa */
    _fv_ti_eq_re(s_, one_over_two_kappa);

  }  /* end of loop in ix over VOLUME/2 */

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of M_clover_zz */


/***********************************************************
 * apply M_clover_zz_inv as matrix
 * - safe, when s = r (r is copied before modification of s)
 ***********************************************************/
void M_clover_zz_inv_matrix (double*s, double*r, double *mzzinv) {

  const unsigned int N = VOLUME/2;
  const int incrcl     = _GSWI(0,1);
  const int vincr      = _GVI(1);
  const double two_kappa = 2.*g_kappa;

#ifdef HAVE_OPENMP
#pragma omp parallel shared(r,s,mzzinv)
{
#endif
  unsigned int ix;
  double *s1_, *s2_, *r1_, *r2_;
  double v1[6], v2[6];
  double *u11_, *u12_, *u21_, *u22_;

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix = 0; ix < N; ix++ ) {
    s1_ = s   + _GSI(ix);
    s2_ = s1_ + vincr;
    r1_ = r   + _GSI(ix);
    r2_ = r1_ + vincr;

    u11_ = mzzinv + 8 * incrcl * ix;
    u12_ = u11_ + incrcl;
    u21_ = u12_ + incrcl;
    u22_ = u21_ + incrcl;

    _cv_eq_cv(v1, r1_);
    _cv_eq_cv(v2, r2_);
    _cv_eq_cm_ti_cv(s1_,u11_,v1);
    _cv_pl_eq_cm_ti_cv(s1_,u12_,v2);
    _cv_ti_eq_re(s1_, two_kappa);

    _cv_eq_cm_ti_cv(s2_,u22_,v2);
    _cv_pl_eq_cm_ti_cv(s2_,u21_,v1);
    _cv_ti_eq_re(s2_, two_kappa);

    s1_ = s2_ + vincr;
    s2_ = s1_ + vincr;
    r1_ = r2_ + vincr;
    r2_ = r1_ + vincr;

    u11_ = u22_ + incrcl;
    u12_ = u11_ + incrcl;
    u21_ = u12_ + incrcl;
    u22_ = u21_ + incrcl;
    
    _cv_eq_cv(v1, r1_);
    _cv_eq_cv(v2, r2_);
    _cv_eq_cm_ti_cv(s1_,u11_,v1);
    _cv_pl_eq_cm_ti_cv(s1_,u12_,v2);
    _cv_ti_eq_re(s1_, two_kappa);

    _cv_eq_cm_ti_cv(s2_,u22_,v2);
    _cv_pl_eq_cm_ti_cv(s2_,u21_,v1);
    _cv_ti_eq_re(s2_, two_kappa);

  }  /* end of loop in ix over VOLUME/2 */
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

}  /* end of M_clover_zz_inv_matrix */


/***********************************************************
 * s = C_oo r
 *   in    : r (remains unchanged), gauge_field, mass
 *   out   : s (changed)
 *   in/out: space: s_aux (changed)
 *   r, s do not need halo
 *   s MUST NOT be ne same memory region as r
 *
 *   s_aux MUST HAVE halo
 ***********************************************************/
void C_clover_oo (double*s, double*r, double *gauge_field, double *s_aux, double*mzz, double*mzzinv) {

  const unsigned int N = VOLUME / 2;
  const size_t sizeof_field = _GSI(N) * sizeof(double);

  if(s == r ) {
    fprintf(stderr, "[C_clover_oo] Error, in and out pointer coincide\n");
    EXIT(1);
  }

  /* s_aux = r */
  memcpy(s_aux, r, sizeof_field);
  /* s = M_eo s_aux */
  Hopping_eo(s, s_aux, gauge_field, 0);
  /* s = M_ee^-1 M_eo s_aux */
  M_clover_zz_inv_matrix(s, s, mzzinv);

  /* s_aux = s */
  memcpy(s_aux, s, sizeof_field);
  /* s = M_oe s_aux = M_oe M_ee^-1 M_eo r */
  Hopping_eo(s, s_aux, gauge_field, 1);

  /* s_aux = M_oo r */
  M_clover_zz_matrix(s_aux, r, mzz);

  /* s = s_aux - s = M_oo r - M_oe M_ee^-1 M_eo r */
  spinor_field_eq_spinor_field_mi_spinor_field(s, s_aux, s, N);
  /* s = g5 s */
  g5_phi(s, N);
 
}  /* end of C_clover_oo */

/********************************************************************
 * X_eo    = -M_ee^-1 M_eo,    mu > 0
 * Xbar_eo = -Mbar_ee^-1 M_eo, mu < 0
 * the input field is always odd, the output field is always even
 *
 * even does not need halo sites
 * odd MUST HAVE halo sites
 * even MUST NOT be same memory region as odd
 ********************************************************************/
void X_clover_eo (double *even, double *odd, double *gauge_field, double*mzzinv) {

  const unsigned int N = VOLUME/2;

  if(even == odd ) {
    fprintf(stderr, "[X_clover_eo] Error, in and out pointer coincide\n");
    EXIT(1);
  }

  /* even = M_eo odd */
  Hopping_eo(even, odd, gauge_field, 0);
  /* even = M_ee^-1 even = M_ee^-1 M_eo odd */
  M_clover_zz_inv_matrix (even, even, mzzinv);
  /* even *= -1 */
  spinor_field_ti_eq_re(even, -1., N);
}  /* end of X_clover_eo */


/********************************************************************
 * C_from_Xeo
 * - apply C = g5 ( M_oo + M_oe X_eo )
 *
 * out t
 * in     s = X_eo v, unchanged
 * in/out t = v, modified
 *        r = auxilliary, modified
 * 
 * t does not need halo
 * s MUST HAVE halo
 ********************************************************************/
void C_clover_from_Xeo (double *t, double *s, double *r, double *gauge_field, double*mzz) {

  const unsigned int N = VOLUME / 2;

  /* r = M_oe s = M_oe X_eo t  */
  Hopping_eo(r, s, gauge_field, 1);
  /* t = M_zz t */
  M_clover_zz_matrix (t, t, mzz);
  /* t = t + r = M_oo^-1 t + M_oe X_eo t*/
  spinor_field_pl_eq_spinor_field (t,r,N);
  /* t = g5 t */
  g5_phi(t,N);

}  /* C_clover_from_Xeo */

/********************************************************************
 * ( g5 M_ee & 0 )
 * ( g5 M_oe & 1 )
 *
 * safe, if e_new = e_old or o_new = o_old
 * e/o_new/old do not need halo
 * aux MUST HAVE halo
 ********************************************************************/
void Q_clover_eo_SchurDecomp_A (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double *mzz, double *aux) {

  const unsigned int N = VOLUME / 2;
  const size_t  sizeof_field = _GSI(N) * sizeof(double);

  /* aux <- e_old */
  memcpy(aux, e_old, sizeof_field);

  /* o_new = M_oe aux = M_oe e_old */
  Hopping_eo(e_new, aux, gauge_field, 1);

  /* e_new = g5 e_new = g5 M_oe e_old */
  spinor_field_eq_gamma_ti_spinor_field(e_new, 5, e_new, N);

  /* o_new = o_old + e_new = o_old + g5 M_oe e_old */
  spinor_field_eq_spinor_field_pl_spinor_field(o_new, o_old, e_new, N);

  /* e_new = M_zz aux = M_zz e_old */
  M_clover_zz_matrix (e_new, aux, mzz);

  /* e_new = g5 aux */
  spinor_field_eq_gamma_ti_spinor_field(e_new, 5, e_new, N);

}  /* end of Q_clover_eo_SchurDecomp_A */



/********************************************************************
 * apply A^-1
 *
 * from Schur-decomposition Q = g5 A B g5 times the Dirac operator
 *
 * A^-1 =
 *   ( M_ee^-1 g5         & 0 )
 *   ( -g5 M_oe M_ee^-1 g5 & 1 )
 *
 * safe, if e_new = e_old or o_new = o_old
 *
 * e/o_new/old do not need halo
 * aux MUST HAVE halo
 ********************************************************************/
void Q_clover_eo_SchurDecomp_Ainv (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double*mzzinv, double *aux) {

  const unsigned int N = VOLUME / 2;
  const size_t sizeof_field = _GSI(N)*sizeof(double);

  /* aux = g5 e_old */
  spinor_field_eq_gamma_ti_spinor_field(aux, 5, e_old, N);

  /* aux <- M_ee^-1 aux = M_ee^-1 g5 e_old */
  M_clover_zz_inv_matrix (aux, aux, mzzinv);

  /* e_new = M_oe aux; e_new is auxilliary field here */
  Hopping_eo(e_new, aux, gauge_field, 1);

#ifdef HAVE_OPENMP
#pragma omp parallel shared(o_new, o_old, e_new)
{
#endif
  unsigned int ix, iix;
  double spinor1[24], *r_=NULL, *s_=NULL, *t_=NULL;
#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix=0; ix<N; ix++) {
    iix = _GSI(ix);
    r_ = o_new + iix;
    s_ = e_new + iix;
    t_ = o_old + iix;
    /* sp1 = g5 s_ = g5 e_new = g5 M_oe M_ee^-1 g5 e_old */
    _fv_eq_gamma_ti_fv(spinor1, 5, s_);
    /* r_ = t_ - sp1 <=>  o_new = -g5 M_oe M_ee^-1 g5 e_old + o_old */
    _fv_eq_fv_mi_fv(r_, t_, spinor1 );
  }
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
  /* e_new = aux = M_ee^-1 e_old */
  memcpy(e_new, aux, sizeof_field);

}  /* end of Q_clover_SchurDecomp_Ainv */

/********************************************************************
 * apply B
 *
 * B =
 *   ( 1 & M_ee^(-1) M_eo )
 *   ( 0 &        C       )
 *
 * - o_old, o_new do not need halo
 *
 * - aux MUST have halo
 ********************************************************************/
void Q_clover_eo_SchurDecomp_B (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double*mzz, double*mzzinv, double *aux) {

  const unsigned int N = VOLUME / 2;
  const size_t sizeof_field = _GSI(N)*sizeof(double);

  if(o_new == o_old ) {
    fprintf(stderr, "[Q_clover_eo_SchurDecomp_B] Error, o_old = o_new\n");
    EXIT(1);
  }

  /* aux = o_old */
  memcpy(aux, o_old, sizeof_field);
 
  /* o_new = M_eo o_old, o_new auxilliary field */
  Hopping_eo(o_new, aux, gauge_field, 0);

  /* o_new = M_ee^(-1) o_new */
  M_clover_zz_inv_matrix(o_new, o_new, mzzinv);

  /* e_new = e_old + o_new = e_old + M_ee^-1 M_eo o_old */
  spinor_field_eq_spinor_field_pl_spinor_field(e_new, e_old, o_new, N);

  /* o_new = C_oo o_old */
  C_clover_oo (o_new, o_old, gauge_field, aux, mzz, mzzinv);

}  /* end of Q_clover_eo_SchurDecomp_B */

/********************************************************************
 * apply B^-1
 *
 * from Schur-decomposition Q = g5 A B of g5 times the Dirac operator
 *
 * ( 1 & -M_ee^(-1) M_eo 2kappa )
 * ( 0 &        2kappa          )
 *
 * safe, if e_new = e_old or o_new = o_old
 * e/o_new/old do not need halo
 * aux MUST HAVE halo
 *
 * NOTE: THIS IS B^-1 WITHOUT the C^-1 ! o_old is supposed to be some 
 *   o_old = C^-1 psi
 ********************************************************************/
void Q_clover_eo_SchurDecomp_Binv (double *e_new, double *o_new, double *e_old, double *o_old, double *gauge_field, double*mzzinv, double *aux) {

  const double twokappa = 2. * g_kappa;
  const unsigned int N = VOLUME / 2;
  const size_t sizeof_field = _GSI(N) * sizeof(double);

  spinor_field_eq_spinor_field_ti_re (aux, o_old, twokappa, N);


  /* o_new = M_eo aux */
  Hopping_eo(o_new, aux, gauge_field, 0);

  /* o_new = M_ee^(-1) o_new */
  M_clover_zz_inv_matrix(o_new, o_new, mzzinv);

  /* e_new = e_old - o_new */
  spinor_field_eq_spinor_field_pl_spinor_field_ti_re(e_new, e_old, o_new, -1, N);

  /* o_new <- aux */
  memcpy(o_new, aux, sizeof_field);

}  /* end of Q_clover_eo_SchurDecomp_Binv */

/********************************************************************
 * prop and source full spinor fields
 ********************************************************************/
int Q_clover_invert (double*prop, double*source, double*gauge_field, double *mzzinv, int op_id) {

#ifdef HAVE_TMLQCD_LIBWRAPPER

  const size_t sizeof_eo_spinor_field_with_halo = _GSI(VOLUME+RAND)/2;

  int exitstatus;
  double *eo_spinor_work[3];

  eo_spinor_work[0]  = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  eo_spinor_work[1]  = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  eo_spinor_work[2]  = (double*)malloc( sizeof_eo_spinor_field_with_halo );
  
  spinor_field_lexic2eo (source, eo_spinor_work[0], eo_spinor_work[1] );

  Q_clover_eo_SchurDecomp_Ainv (eo_spinor_work[0], eo_spinor_work[1], eo_spinor_work[0], eo_spinor_work[1], gauge_field, mzzinv, eo_spinor_work[2]);
  exitstatus = tmLQCD_invert_eo(eo_spinor_work[2], eo_spinor_work[1], op_id);
  if(exitstatus != 0) {
    fprintf(stderr, "[Q_clover_invert] Error from tmLQCD_invert_eo, status was %d\n", exitstatus);
    return(1);
  }
  Q_clover_eo_SchurDecomp_Binv (eo_spinor_work[0], eo_spinor_work[2], eo_spinor_work[0], eo_spinor_work[2], gauge_field, mzzinv, eo_spinor_work[1]);
  spinor_field_eo2lexic (prop, eo_spinor_work[0], eo_spinor_work[2] );
  free( eo_spinor_work[0] );
  free( eo_spinor_work[1] );
  free( eo_spinor_work[2] );
  return(0);
#else
  if( g_cart_id == 0 ) fprintf(stderr, "[Q_clover_invert] Error, no inverter\n");
  return(2);
#endif
}  /* Q_clover_invert */

}  /* end of namespace cvc */
