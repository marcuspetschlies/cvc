#include "stdlib.h"
#include "string.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_linalg.h"
#include "cvc_utils.h"
#include "hyp_smear.h"

#define _CM_PROJ(_U_OUT, _U_IN) { cm_proj(_U_IN); _cm_eq_cm(_U_OUT, _U_IN); }
/* #define _CM_PROJ(_U_OUT, _U_IN) cm_proj_iterate( (_U_OUT), (_U_IN), accu, imax) */
/* #define _CM_PROJ(_U_OUT, _U_IN) { reunitarize_Givens_rotations( _U_IN); _cm_eq_cm( (_U_OUT), (_U_IN) ); } */

namespace cvc {


int index_tab_inv[4][4][4];
int index_tab_inv2[4][4];
int index_tab3_inv[3][3];

void init_tab_inv(void) {
  /* first index 0 */
  index_tab_inv[0][1][2] = 0;
  index_tab_inv[0][2][1] = 0;
  index_tab_inv[0][2][3] = 1;
  index_tab_inv[0][3][2] = 1;
  index_tab_inv[0][1][3] = 2;
  index_tab_inv[0][3][1] = 2;
  /* first index 1 */
  index_tab_inv[1][0][3] = 0;
  index_tab_inv[1][3][0] = 0;
  index_tab_inv[1][2][3] = 1;
  index_tab_inv[1][3][2] = 1;
  index_tab_inv[1][0][2] = 2;
  index_tab_inv[1][2][0] = 2;
  /* first index 2 */
  index_tab_inv[2][0][1] = 0;
  index_tab_inv[2][1][0] = 0;
  index_tab_inv[2][1][3] = 1;
  index_tab_inv[2][3][1] = 1;
  index_tab_inv[2][0][3] = 2;
  index_tab_inv[2][3][0] = 2;
  /* first index 3 */
  index_tab_inv[3][0][2] = 0;
  index_tab_inv[3][2][0] = 0;
  index_tab_inv[3][0][1] = 1;
  index_tab_inv[3][1][0] = 1;
  index_tab_inv[3][1][2] = 2;
  index_tab_inv[3][2][1] = 2;
  
  index_tab_inv2[0][1] = 0;
  index_tab_inv2[0][2] = 1;
  index_tab_inv2[0][3] = 2;
  index_tab_inv2[1][2] = 0;
  index_tab_inv2[1][3] = 1;
  index_tab_inv2[1][0] = 2;
  index_tab_inv2[2][0] = 0;
  index_tab_inv2[2][1] = 1;
  index_tab_inv2[2][3] = 2;
  index_tab_inv2[3][0] = 0;
  index_tab_inv2[3][1] = 1;
  index_tab_inv2[3][2] = 2;

  index_tab3_inv[0][1] = 0;
  index_tab3_inv[0][2] = 1;
  index_tab3_inv[1][0] = 0;
  index_tab3_inv[1][2] = 1;
  index_tab3_inv[2][0] = 0;
  index_tab3_inv[2][1] = 1;
}


int index_tab[4][3][4] = {
  { {0,1,2,3},
    {0,2,3,1},
    {0,3,1,2} },
  { {1,2,3,0},
    {1,3,0,2},
    {1,0,2,3} },
  { {2,0,1,3},
    {2,1,3,0},
    {2,3,0,1} },
  { {3,0,2,1},
    {3,1,0,2},
    {3,2,1,0}}
};

int index_tab3[3][2][3] = {
  { {0,1,2}, {0,2,1} },
  { {1,0,2}, {1,2,0} },
  { {2,0,1}, {2,1,0} }
};


// modified APE smearing step 3
int hyp_smear_step( double *u_out, double *u_in, double A[3], double accu, unsigned int imax) {
  const int dim = 4;
  const unsigned int VOL3 = LX*LY*LZ;
  const double OneOverSix = 0.1666666666666667;
  int mu, nu, rho, eta, i, iperm, ix;
  double *vbar[3], *vtilde[3];
  size_t shift;
  double U[18], U2[18], U3[18];

  alloc_gauge_field(vbar,   VOLUMEPLUSRAND);
  alloc_gauge_field(vbar+1, VOLUMEPLUSRAND);
  alloc_gauge_field(vbar+2, VOLUMEPLUSRAND);

  alloc_gauge_field(vtilde,   VOLUMEPLUSRAND);
  alloc_gauge_field(vtilde+1, VOLUMEPLUSRAND);
  alloc_gauge_field(vtilde+2, VOLUMEPLUSRAND);

  /* fprintf(stdout, "# [hyp_smear_step] A = %e, %e, %e\n", A[0], A[1], A[2]); */

  if( vbar[0] == NULL || vbar[1] == NULL || vbar[2] == NULL) return(2);
  if( vtilde[0] == NULL || vtilde[1] == NULL || vtilde[2] == NULL) return(1);

  // calculate v1[mu][nu][rho] = \bar{V}_{\mu; \nu\rho}
  for(mu=0; mu<4; mu++) {
    for(iperm=0; iperm<3; iperm++) {
      nu  = index_tab[mu][iperm][1];
      rho = index_tab[mu][iperm][2];
      eta = index_tab[mu][iperm][3];

      /* fprintf(stdout, "# [hyp_smear_step] mu = %d, nu = %d, rho = %d, eta = %d\n", mu, nu, rho, eta); */

      for(ix=0; ix<VOLUME; ix++) {

        // set initial v to (1 - alpha_3) u for each triple (mu, nu, rho)
        /* U = (1 - A[3]) * u_in[mu] */
        _cm_eq_cm_ti_re(U, u_in+_GGI(ix,mu), 1.-A[2]);

        /* positive eta direction */
        _cm_eq_cm_ti_cm(U2, u_in+_GGI(ix,eta), u_in+_GGI(g_iup[ix][eta],mu));
        _cm_eq_cm_ti_cm_dag(U3, U2, u_in+_GGI(g_iup[ix][mu],eta));
        _cm_ti_eq_re(U3, 0.5*A[2]);
        _cm_pl_eq_cm(U, U3);

        /* negative eta-direction */
        _cm_eq_cm_dag(U3, u_in+_GGI(g_idn[ix][eta],eta));
        _cm_eq_cm_ti_cm(U2, U3, u_in+_GGI(g_idn[ix][eta],mu));
        _cm_eq_cm_ti_cm(U3, U2, u_in+_GGI(g_idn[g_iup[ix][mu]][eta],eta));
        _cm_ti_eq_re(U3, 0.5*A[2]);
        _cm_pl_eq_cm(U, U3);

        /* TEST */
        /* _cm_fprintf(U, "vbar", stdout); */

        _CM_PROJ( U3, U );

        /* TEST */
        /* _cm_fprintf(U3, "Pvbar", stdout); */

        _cm_eq_cm( vbar[iperm]+_GGI(ix,mu), U3);
      }

    }  /* end of permutations */

  }  /* end of construction of Vbar */

  /* exchange the fields vbar */
  xchange_gauge_field(vbar[0]);
  xchange_gauge_field(vbar[1]);
  xchange_gauge_field(vbar[2]);

  /* construct Vtilde from Vbar */
  for(mu=0; mu<4; mu++) {
    for(iperm=0; iperm<3; iperm++) {
      nu = index_tab[mu][iperm][1];

   
      for(ix=0; ix<VOLUME; ix++) {
        _cm_eq_cm_ti_re(U, u_in+_GGI(ix,mu), (1.-A[1]));

        /* first rho */
        rho = index_tab[mu][iperm][2];

        /* fprintf(stdout, "# [hyp_smear_step] ix = %d, mu = %d, nu = %d, rho(1) = %d\n", ix, mu, nu, rho); */


        /* first rho, positive direction */
        _cm_eq_cm_ti_cm( U2, vbar[index_tab_inv[rho][nu][mu]]+_GGI(ix,rho), vbar[index_tab_inv[mu][rho][nu]]+_GGI(g_iup[ix][rho],mu) );
        _cm_eq_cm_ti_cm_dag( U3, U2, vbar[index_tab_inv[rho][nu][mu]]+_GGI(g_iup[ix][mu],rho) );
        _cm_ti_eq_re(U3, 0.25*A[1]);
        _cm_pl_eq_cm(U, U3);

        /* first rho, negative direction */
        _cm_eq_cm_dag(U3, vbar[index_tab_inv[rho][nu][mu]]+_GGI(g_idn[ix][rho],rho));
        _cm_eq_cm_ti_cm(U2, U3, vbar[index_tab_inv[mu][rho][nu]]+_GGI(g_idn[ix][rho],mu));
        _cm_eq_cm_ti_cm(U3, U2, vbar[index_tab_inv[rho][nu][mu]]+_GGI(g_idn[g_iup[ix][mu]][rho],rho));
        _cm_ti_eq_re(U3, 0.25*A[1]);
        _cm_pl_eq_cm(U, U3);

        /* second rho */
        rho = index_tab[mu][iperm][3];
        /* fprintf(stdout, "# [hyp_smear_step] ix = %d, mu = %d, nu = %d, rho(2) = %d\n", ix, mu, nu, rho); */

        /* second rho, positive direction */
        _cm_eq_cm_ti_cm( U2, vbar[index_tab_inv[rho][nu][mu]]+_GGI(ix,rho), vbar[index_tab_inv[mu][rho][nu]]+_GGI(g_iup[ix][rho],mu) );
        _cm_eq_cm_ti_cm_dag( U3, U2, vbar[index_tab_inv[rho][nu][mu]]+_GGI(g_iup[ix][mu],rho) );
        _cm_ti_eq_re(U3, 0.25*A[1]);
        _cm_pl_eq_cm(U, U3);

        /* second rho, negative direction */
        _cm_eq_cm_dag(U3, vbar[index_tab_inv[rho][nu][mu]]+_GGI(g_idn[ix][rho],rho));
        _cm_eq_cm_ti_cm(U2, U3, vbar[index_tab_inv[mu][rho][nu]]+_GGI(g_idn[ix][rho],mu));
        _cm_eq_cm_ti_cm(U3, U2, vbar[index_tab_inv[rho][nu][mu]]+_GGI(g_idn[g_iup[ix][mu]][rho],rho));
        _cm_ti_eq_re(U3, 0.25*A[1]);
        _cm_pl_eq_cm(U, U3);
        
        /* TEST */
        /* _cm_fprintf(U, "vtilde", stdout); */

        _CM_PROJ(U3, U);

        /* TEST */
        /* _cm_fprintf(U3, "Pvtilde", stdout); */

        _cm_eq_cm(vtilde[iperm]+_GGI(ix,mu), U3);
      }  /* end of loop on ix */
    }    /* end of loop on perm */
  }      /* end of loop on mu */
  
  xchange_gauge_field(vtilde[0]);
  xchange_gauge_field(vtilde[1]);
  xchange_gauge_field(vtilde[2]);
  
  /* construct V from Vtilde */
  for(mu=0; mu<4; mu++) {

    for(ix=0; ix<VOLUME; ix++) {

      _cm_eq_cm_ti_re(U, u_in+_GGI(ix,mu), (1.-A[0]));

      for(iperm=0; iperm<3; iperm++) {
        nu = index_tab[mu][iperm][1];

        /* fprintf(stdout, "# [hyp_smear_step] ix = %d, mu = %d, nu = %d\n", ix, mu, nu); */

        /* positive direction */
        _cm_eq_cm_ti_cm( U2, vtilde[index_tab_inv2[nu][mu]]+_GGI(ix,nu), vtilde[index_tab_inv2[mu][nu]]+_GGI(g_iup[ix][nu],mu) );
        _cm_eq_cm_ti_cm_dag(U3, U2, vtilde[index_tab_inv2[nu][mu]]+_GGI(g_iup[ix][mu],nu));
        _cm_ti_eq_re(U3, OneOverSix*A[0]);
        _cm_pl_eq_cm(U, U3);

        _cm_eq_cm_dag(U3, vtilde[index_tab_inv2[nu][mu]]+_GGI(g_idn[ix][nu],nu));
        _cm_eq_cm_ti_cm(U2, U3, vtilde[index_tab_inv2[mu][nu]]+_GGI(g_idn[ix][nu],mu));
        _cm_eq_cm_ti_cm(U3, U2, vtilde[index_tab_inv2[nu][mu]]+_GGI(g_idn[g_iup[ix][mu]][nu], nu));
        _cm_ti_eq_re(U3, OneOverSix*A[0]);
        _cm_pl_eq_cm(U, U3);
      }  /* of loop on perm */

      /* TEST */
      /* _cm_fprintf(U, "v", stdout); */

      _CM_PROJ(U3, U);

      /* TEST */
      /* _cm_fprintf(U3, "Pv", stdout); */

      _cm_eq_cm( u_out+_GGI(ix,mu), U3);

    }  /* of loop on ix */
  }    /* of loop on mu*/
  xchange_gauge_field(u_out);
  
  /* free auxilliary gauge fields */
  free(vbar[0]); vbar[0] = NULL;
  free(vbar[1]); vbar[1] = NULL;
  free(vbar[2]); vbar[2] = NULL;
  free(vtilde[0]); vtilde[0] = NULL;
  free(vtilde[1]); vtilde[1] = NULL;
  free(vtilde[2]); vtilde[2] = NULL;
  
  return(0);
}  /* of hyp_smear_step */


/*****************************************************************************
 * 3-dimensional HYP smearing step
 *****************************************************************************/
int hyp_smear_step_3d( double *u_out, double *u_in, double A[2], double accu, unsigned int imax) {
  const int dim = 4;
  const unsigned int VOL3 = LX*LY*LZ;
  int mu, nu, rho, i, iperm, ix;
  int imu, inu, irho;
  double *vtilde[2];
  size_t shift;
  double U[18], U2[18], U3[18];
  const double one_mi_a0    = 1. - A[0];
  const double one_mi_a1    = 1. - A[1];
  const double a0_over_four = 0.25 * A[0];
  const double a1_over_two  = 0.5  * A[1];

  /* TEST */
  /* fprintf(stdout, "# [hyp_smear_step_3d] A = %e, %e\n", A[0], A[1]); */

  alloc_gauge_field(vtilde,   VOLUMEPLUSRAND);
  alloc_gauge_field(vtilde+1, VOLUMEPLUSRAND);

  if( vtilde[0] == NULL || vtilde[1] == NULL) return(1);

  /* construct Vtilde from U */
  for(mu=1; mu<4; mu++) {
    for(iperm=0; iperm<2; iperm++) {
      nu  = index_tab3[mu-1][iperm][1]+1;
      rho = index_tab3[mu-1][iperm][2]+1;

      /* TEST */
      /* fprintf(stdout, "# [hyp_smear_step_3d] vtilde mu=%d, nu=%d, rho=%d\n", mu, nu, rho); */

      for(ix=0; ix<VOLUME; ix++) {

        /* TEST */
        /* fprintf(stdout, "# [hyp_smear_step_3d] ix = %d\n", ix); */

        _cm_eq_cm_ti_re(U, u_in+_GGI(ix,mu), one_mi_a1);

        /* first rho, positive direction */
        _cm_eq_cm_ti_cm    ( U2, u_in+_GGI(ix,rho), u_in+_GGI(g_iup[ix][rho],mu ) );
        _cm_eq_cm_ti_cm_dag( U3, U2               , u_in+_GGI(g_iup[ix][mu],rho) );
        _cm_ti_eq_re(U3, a1_over_two );
        _cm_pl_eq_cm(U, U3);

        /* first rho, negative direction */
        _cm_eq_cm_dag(U3, u_in+_GGI(g_idn[ix][rho],rho));
        _cm_eq_cm_ti_cm(U2, U3, u_in+_GGI(g_idn[ix][rho],mu)            );
        _cm_eq_cm_ti_cm(U3, U2, u_in+_GGI(g_idn[g_iup[ix][mu]][rho],rho));
        _cm_ti_eq_re(U3, a1_over_two );
        _cm_pl_eq_cm(U, U3);

        /* TEST */
        /* _cm_fprintf(U, "vtilde", stdout); */

        _CM_PROJ(U3, U);

        /* TEST */
        /* _cm_fprintf(U3, "Pvtilde", stdout); */

        _cm_eq_cm(vtilde[iperm]+_GGI(ix,mu), U3);
      }  /* end of loop on ix */
    }    /* end of loop on perm */
  }      /* end of loop on mu */
  
  xchange_gauge_field(vtilde[0]);
  xchange_gauge_field(vtilde[1]);
  
  /* construct V from Vtilde */
  for(mu=1; mu<4; mu++) {
    imu = mu-1;

    for(ix=0; ix<VOLUME; ix++) {

      _cm_eq_cm_ti_re(U, u_in+_GGI(ix,mu), one_mi_a0 );

      for(iperm=0; iperm<2; iperm++) {
        inu = index_tab3[mu-1][iperm][1];
        nu  = inu + 1;

        /* TEST */
        /* fprintf(stdout, "# [hyp_smear_step_3d] ix = %d, mu = %d / %d, nu = %d / %d\n", ix, imu, mu, inu, nu); */

        /* positive direction */
        _cm_eq_cm_ti_cm( U2, vtilde[index_tab3_inv[inu][imu]]+_GGI(ix,nu), vtilde[index_tab3_inv[imu][inu]]+_GGI(g_iup[ix][nu],mu) );
        _cm_eq_cm_ti_cm_dag(U3, U2, vtilde[index_tab3_inv[inu][imu]]+_GGI(g_iup[ix][mu],nu));
        _cm_ti_eq_re(U3, a0_over_four );
        _cm_pl_eq_cm(U, U3);

        _cm_eq_cm_dag(U3, vtilde[index_tab3_inv[inu][imu]]+_GGI(g_idn[ix][nu],nu));
        _cm_eq_cm_ti_cm(U2, U3, vtilde[index_tab3_inv[imu][inu]]+_GGI(g_idn[ix][nu],mu));
        _cm_eq_cm_ti_cm(U3, U2, vtilde[index_tab3_inv[inu][imu]]+_GGI(g_idn[g_iup[ix][mu]][nu], nu));
        _cm_ti_eq_re(U3, a0_over_four );
        _cm_pl_eq_cm(U, U3);
      }  /* of loop on perm */

      /* TEST */
      /* _cm_fprintf(U, "v", stdout); */

      _CM_PROJ(U3, U);
      _cm_eq_cm( u_out+_GGI(ix,mu), U3);

      /* TEST */
      /* _cm_fprintf(U3, "Pv", stdout); */

    }  /* of loop on ix */
  }    /* of loop on mu*/
  xchange_gauge_field(u_out);
  
  /* free auxilliary gauge fields */
  free(vtilde[0]); vtilde[0] = NULL;
  free(vtilde[1]); vtilde[1] = NULL;
  
  return(0);
}  /* of hyp_smear_step_3d */

/*****************************************************************************
 * HYP smearing function
 *****************************************************************************/
int hyp_smear (double *u, unsigned int N, double accu, unsigned int imax) {
  int i, status;
  double A[] =  { 0.75, 0.6, 0.3 };
  double *u_aux = NULL;

  init_tab_inv();

  status = alloc_gauge_field(&u_aux, VOLUMEPLUSRAND);
  if(u_aux == NULL) {
    fprintf(stderr, "[hyp_smear] Error from alloc_gauge_field, u_aux was NULL\n");
    EXIT(1);
  }

  for(i = 1; i<=N; i++) {
    memcpy(u_aux, u, 72*VOLUMEPLUSRAND*sizeof(double));
    status = hyp_smear_step(u, u_aux, A, accu, imax);

    if(status != 0) {
      fprintf(stderr, "[hyp_smear] Error from hyp_smear_step, status was %d\n", status);
      EXIT(status);
    }
  }
  return(0);
}  /* of hyp_smear */

/*****************************************************************************
 * HYP smearing function
 *****************************************************************************/
int hyp_smear_3d (double *u, unsigned int N, double *A, double accu, unsigned int imax) {
  int i, status;
  double *u_aux = NULL;

  init_tab_inv();

  status = alloc_gauge_field(&u_aux, VOLUMEPLUSRAND);
  if(u_aux == NULL) {
    fprintf(stderr, "[hyp_smear] Error from alloc_gauge_field, u_aux was NULL\n");
    EXIT(1);
  }

  for(i = 1; i<=N; i++) {
    fprintf(stdout, "# [hyp_smear_3d] hyp smear step %d\n", i);
    memcpy(u_aux, u, 72*VOLUMEPLUSRAND*sizeof(double));
    status = hyp_smear_step_3d(u, u_aux, A, accu, imax);

    if(status != 0) {
      fprintf(stderr, "[hyp_smear] Error from hyp_smear_step_3d, status was %d\n", status);
      EXIT(status);
    }
  }
  if(u_aux != NULL) free(u_aux);
  return(0);
}  /* of hyp_smear_3d */

}
