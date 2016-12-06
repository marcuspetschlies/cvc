/****************************************************
 * contract_piN2piN.c
 * 
 * Mon Dec  5 15:04:00 CET 2016
 *
 * PURPOSE:
 *   pi N - pi N 2-point function contractions
 *   with point-source propagators, sequential
 *   propagators and stochastic propagagtors
 * TODO:
 * DONE:
 *
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
#include <omp.h>
#endif

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#include "cvc_complex.h"
#include "ilinalg.h"
#include "icontract.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "matrix_init.h"
#include "project.h"

namespace cvc {

int contract_piN2piN (spinor_propagator_type *res, double**uprop_list, double**dprop_list, double**tfii_list, double**tffi_list, double**pffii_list, 
    int ncomp, int(*comp_list)[2], double*comp_list_sign) {

  double ratime, retime;

      /******************************************************
       * contractions
       *
       * REMEMBER:
       *
       *   uprop = S_u
       *   dprop = S_d  // not needed for Wilson fermions (u and d degenerate)
       *   tfii  = sequential propagator t_f <- t_i <- t_i
       *   tffi  = sequential propagator t_f <- t_f <- t_i
       *   pffii = sequential^2 propagator t_f <- t_f <- t_i <- t_i
       *   fp1   = C Gamma_1 S_u
       *   fp2   = C Gamma_1 S_u C Gamma_2
       *   fp3   =           S_u C Gamma_2
       *   fp4   = C Gamma_1 tffi
       *   fp5   = C Gamma_1 S_d C Gamma_2
       *   fp6   = C Gamma_1 tffi C Gamma_2
       ******************************************************/
      ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel shared(res)
{
#endif
      unsigned int ix;
      int icomp;
      fermion_propagator_type fp1, fp2, fp3, fpaux, fp4, fp5, fp6, uprop, dprop;
      fermion_propagator_type tfii, tffi, pffii, pfifi;
      spinor_propagator_type sp_c1, sp_c2, sp_c3, sp1, sp2;

      create_fp(&uprop);
      create_fp(&dprop);
      create_fp(&tfii);
      create_fp(&tffi);
      create_fp(&pfifi);
      create_fp(&pffii);
      create_fp(&fp1);
      create_fp(&fp2);
      create_fp(&fp3);
      create_fp(&fp4);
      create_fp(&fp5);
      create_fp(&fp6);
      create_fp(&fpaux);

      create_sp(&sp_c1);
      create_sp(&sp_c2);
      create_sp(&sp_c3);
      create_sp(&sp1);
      create_sp(&sp2);

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for(ix=0; ix<VOLUME; ix++)
      {
        /* assign the propagator points */

        _assign_fp_point_from_field(uprop, uprop_list, ix);
        _assign_fp_point_from_field(dprop, dprop_list, ix);

        _assign_fp_point_from_field(tfii,  tfii_list,  ix);

        _assign_fp_point_from_field(tffi,  tffi_list,  ix);

        _assign_fp_point_from_field(pffii, pffii_list, ix);

        for(icomp=0; icomp<ncomp; icomp++) {

          _sp_eq_zero( res[ix*ncomp+icomp]);

          /******************************************************
           * prepare fermion propagators
           ******************************************************/
          _fp_eq_zero(fp1);
          _fp_eq_zero(fp2);
          _fp_eq_zero(fp3);
          _fp_eq_zero(fp4);
          _fp_eq_zero(fp5);
          _fp_eq_zero(fp6);
          _fp_eq_zero(fpaux);

          /* fp1 = C Gamma_1 x S_u = g0 g2 Gamma_1 S_u */
          _fp_eq_gamma_ti_fp(fp1, comp_list[icomp][0], uprop);
          _fp_eq_gamma_ti_fp(fpaux, 2, fp1);
          _fp_eq_gamma_ti_fp(fp1,   0, fpaux);

          /* fp2 = C Gamma_1 x S_u x C Gamma_2 = fp1 x g0 g2 Gamma_2 */
          _fp_eq_fp_ti_gamma(fp2, 0, fp1);
          _fp_eq_fp_ti_gamma(fpaux, 2, fp2);
          _fp_eq_fp_ti_gamma(fp2, comp_list[icomp][1], fpaux);

          /* fp5 = C Gamma_1 x S_d x C Gamma_2 */
          if(g_fermion_type == _TM_FERMION) {
            _fp_eq_gamma_ti_fp(fp5, comp_list[icomp][0], dprop);
            _fp_eq_gamma_ti_fp(fpaux, 2, fp5);
            _fp_eq_gamma_ti_fp(fp5,   0, fpaux);
            _fp_eq_fp_ti_gamma(fpaux, 0, fp5);
            _fp_eq_fp_ti_gamma(fp5, 2, fpaux);
            _fp_eq_fp_ti_gamma(fpaux, comp_list[icomp][1], fp5);
            _fp_eq_fp(fp5, fpaux);
          } else {
            _fp_eq_fp(fp5, fp2);
          }

          /* fp3 = S_u x C Gamma_2 = uprop x g0 g2 Gamma_2 */
          _fp_eq_fp_ti_gamma(fp3,   0, uprop);
          _fp_eq_fp_ti_gamma(fpaux, 2, fp3);
          _fp_eq_fp_ti_gamma(fp3, comp_list[icomp][1], fpaux);

          /* fp4 = [ C Gamma_1 x tffi ] x C Gamma_2 */
          _fp_eq_gamma_ti_fp(fp4, comp_list[icomp][0], tffi);
          _fp_eq_gamma_ti_fp(fpaux, 2, fp4);
          _fp_eq_gamma_ti_fp(fp4,   0, fpaux);

          _fp_eq_fp_ti_gamma(fp6, 0, fp4);
          _fp_eq_fp_ti_gamma(fpaux, 2, fp6);
          _fp_eq_fp_ti_gamma(fp6, comp_list[icomp][1], fpaux);

          /*********************
           * C_1
           *********************/
          _sp_eq_zero( sp_c1 );
//#if 0  /* pffii */
          /* (1) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, pffii, fp1);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract23_fp(sp1, fp3, fpaux);
          _sp_pl_eq_sp(sp_c1, sp1);
//#endif
//#if 0  /* tr */
          /* (2) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, pffii, fp2);
          /* reduce to spin propagator */
          /* _sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract34_fp(sp1, uprop, fpaux);
          _sp_pl_eq_sp(sp_c1, sp1);
//#endif
          /* add and assign */
          _sp_pl_eq_sp_ti_re( res[ix*ncomp+icomp], sp_c1, comp_list_sign[icomp]);

          /*********************
           * C_2
           *********************/
          _sp_eq_zero( sp_c2 );
//#if 0
          /* (1) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp3, fp4);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract23_fp(sp1, tfii, fpaux);
          _sp_pl_eq_sp(sp_c2, sp1);
//#endif
//#if 0
          /* (2) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp6, uprop);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract23_fp(sp1, tfii, fpaux);
          _sp_pl_eq_sp(sp_c2, sp1);
//#endif
//#if 0
          /* (3) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, tfii, fp4);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract23_fp(sp1, fp3, fpaux);
          _sp_pl_eq_sp(sp_c2, sp1);
//#endif
//#if 0  /* tr */
          /* (4) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, tfii, fp6);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract34_fp(sp1, uprop, fpaux);
          _sp_pl_eq_sp(sp_c2, sp1);
//#endif
          /* add and assign */
          _sp_pl_eq_sp_ti_re( res[ix*ncomp+icomp], sp_c2, -comp_list_sign[icomp]);

        }  /* end of loop on components */

      }    /* end of loop on ix */

      free_fp(&uprop);
      free_fp(&dprop);
      free_fp(&tfii);
      free_fp(&tffi);
      free_fp(&pfifi);
      free_fp(&pffii);
      free_fp(&fp1);
      free_fp(&fp2);
      free_fp(&fp3);
      free_fp(&fp4);
      free_fp(&fp5);
      free_fp(&fp6);
      free_fp(&fpaux);

      free_sp(&sp_c1);
      free_sp(&sp_c2);
      free_sp(&sp_c3);
      free_sp(&sp1);
      free_sp(&sp2);

#ifdef OPENMP
}
#endif
      retime = _GET_TIME;
      if(g_cart_id == 0)  fprintf(stdout, "# [contract_piN2piN] time for contractions = %e seconds\n", retime-ratime);
  
  return(0);
}  /* end of contract_piN2piN */







/******************************************************
 * contrcations for Z-diagrams
 * using the one-end-trick
 ******************************************************/
int contract_piN2piN_oet (spinor_propagator_type *res, double**uprop_list, double**dprop_list, double**phi_0_list, double**phi_p_list) {

  double ratime, retime;

      /******************************************************
       * contractions
       *
       * REMEMBER:
       *
       *   uprop = S_u
       *   dprop = S_d  // not needed for Wilson fermions (u and d degenerate)
       *   tfii  = sequential propagator t_f <- t_i <- t_i
       *   tffi  = sequential propagator t_f <- t_f <- t_i
       *   pffii = sequential^2 propagator t_f <- t_f <- t_i <- t_i
       *   pfifi = sequential^2 propagator t_f <- t_i <- t_f <- t_i
       *   fp1   = C Gamma_1 S_u
       *   fp2   = C Gamma_1 S_u C Gamma_2
       *   fp3   =           S_u C Gamma_2
       *   fp4   = C Gamma_1 tffi
       *   fp5   = C Gamma_1 S_d C Gamma_2
       *   fp6   = C Gamma_1 tffi C Gamma_2
       ******************************************************/
      ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel private (ix,icomp) shared(res)
{
#endif
      fermion_propagator_type fp1, fp2, fp3, fpaux, fp4, fp5, fp6, uprop, dprop;
      fermion_propagator_type tfii, tffi, pffii, pfifi;
      spinor_propagator_type sp_c1, sp_c2, sp_c3, sp1, sp2;

      create_fp(&uprop);
      create_fp(&dprop);
      create_fp(&tfii);
      create_fp(&tffi);
      create_fp(&pfifi);
      create_fp(&pffii);
      create_fp(&fp1);
      create_fp(&fp2);
      create_fp(&fp3);
      create_fp(&fp4);
      create_fp(&fp5);
      create_fp(&fp6);
      create_fp(&fpaux);

      create_sp(&sp_c1);
      create_sp(&sp_c2);
      create_sp(&sp_c3);
      create_sp(&sp1);
      create_sp(&sp2);

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for(ix=0; ix<VOLUME; ix++)
      {
        /* assign the propagator points */

        _assign_fp_point_from_field(uprop, uprop_list,    ix);
        _assign_fp_point_from_field(dprop, dprop_list,    ix);

        _assign_fp_point_from_field(tfii,  tfii_list,  ix);

        _assign_fp_point_from_field(tffi,  tffi_list,  ix);

        _assign_fp_point_from_field(pffii, pffii_list, ix);

        _assign_fp_point_from_field(pfifi, g_spinor_field + _PFIFI, ix);



        /* flavor rotation for twisted mass fermions */
        if(g_fermion_type == _TM_FERMION) {
          /* S_u */
          _fp_eq_rot_ti_fp(fp1, uprop, +1, g_fermion_type, fp2);
          _fp_eq_fp_ti_rot(uprop, fp1, +1, g_fermion_type, fp2);

          /* S_d */
          _fp_eq_rot_ti_fp(fp1, dprop, -1, g_fermion_type, fp2);
          _fp_eq_fp_ti_rot(dprop, fp1, -1, g_fermion_type, fp2);

          /* T_fii^ud */
          _fp_eq_rot_ti_fp(fp1, tfii, +1, g_fermion_type, fp2);
          _fp_eq_fp_ti_rot(tfii, fp1, -1, g_fermion_type, fp2);

          /* T_ffi^du */
          _fp_eq_rot_ti_fp(fp1, tffi, -1, g_fermion_type, fp2);
          _fp_eq_fp_ti_rot(tffi, fp1, +1, g_fermion_type, fp2);

          /* P_ffii^dud */
          _fp_eq_rot_ti_fp(fp1, pffii, -1, g_fermion_type, fp2);
          _fp_eq_fp_ti_rot(pffii, fp1, -1, g_fermion_type, fp2);

          /* P_fifi^udu */
          _fp_eq_rot_ti_fp(fp1, pfifi, +1, g_fermion_type, fp2);
          _fp_eq_fp_ti_rot(pfifi, fp1, +1, g_fermion_type, fp2);
        }

        for(icomp=0; icomp<ncomp; icomp++) {

          _sp_eq_zero( res[ix*ncomp+icomp]);

          /******************************************************
           * prepare fermion propagators
           ******************************************************/
          _fp_eq_zero(fp1);
          _fp_eq_zero(fp2);
          _fp_eq_zero(fp3);
          _fp_eq_zero(fp4);
          _fp_eq_zero(fp5);
          _fp_eq_zero(fp6);
          _fp_eq_zero(fpaux);

          /* fp1 = C Gamma_1 x S_u = g0 g2 Gamma_1 S_u */
          _fp_eq_gamma_ti_fp(fp1, comp_list[icomp][0], uprop);
          _fp_eq_gamma_ti_fp(fpaux, 2, fp1);
          _fp_eq_gamma_ti_fp(fp1,   0, fpaux);

          /* fp2 = C Gamma_1 x S_u x C Gamma_2 = fp1 x g0 g2 Gamma_2 */
          _fp_eq_fp_ti_gamma(fp2, 0, fp1);
          _fp_eq_fp_ti_gamma(fpaux, 2, fp2);
          _fp_eq_fp_ti_gamma(fp2, comp_list[icomp][1], fpaux);

          /* fp5 = C Gamma_1 x S_d x C Gamma_2 */
          if(g_fermion_type == _TM_FERMION) {
            _fp_eq_gamma_ti_fp(fp5, comp_list[icomp][0], dprop);
            _fp_eq_gamma_ti_fp(fpaux, 2, fp5);
            _fp_eq_gamma_ti_fp(fp5,   0, fpaux);
            _fp_eq_fp_ti_gamma(fpaux, 0, fp5);
            _fp_eq_fp_ti_gamma(fp5, 2, fpaux);
            _fp_eq_fp_ti_gamma(fpaux, comp_list[icomp][1], fp5);
            _fp_eq_fp(fp5, fpaux);
          } else {
            _fp_eq_fp(fp5, fp2);
          }

          /* fp3 = S_u x C Gamma_2 = uprop x g0 g2 Gamma_2 */
          _fp_eq_fp_ti_gamma(fp3,   0, uprop);
          _fp_eq_fp_ti_gamma(fpaux, 2, fp3);
          _fp_eq_fp_ti_gamma(fp3, comp_list[icomp][1], fpaux);

          /* fp4 = [ C Gamma_1 x tffi ] x C Gamma_2 */
          _fp_eq_gamma_ti_fp(fp4, comp_list[icomp][0], tffi);
          _fp_eq_gamma_ti_fp(fpaux, 2, fp4);
          _fp_eq_gamma_ti_fp(fp4,   0, fpaux);

          _fp_eq_fp_ti_gamma(fp6, 0, fp4);
          _fp_eq_fp_ti_gamma(fpaux, 2, fp6);
          _fp_eq_fp_ti_gamma(fp6, comp_list[icomp][1], fpaux);

          /*********************
           * C_1
           *********************/
          _sp_eq_zero( sp_c1 );
//#if 0  /* pffii */
          /* (1) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, pffii, fp1);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract23_fp(sp1, fp3, fpaux);
          _sp_pl_eq_sp(sp_c1, sp1);
//#endif
//#if 0  /* tr */
          /* (2) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, pffii, fp2);
          /* reduce to spin propagator */
          /* _sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract34_fp(sp1, uprop, fpaux);
          _sp_pl_eq_sp(sp_c1, sp1);
//#endif
          /* add and assign */
          _sp_pl_eq_sp_ti_re( res[ix*ncomp+icomp], sp_c1, comp_list_sign[icomp]);

          /*********************
           * C_2
           *********************/
          _sp_eq_zero( sp_c2 );
//#if 0
          /* (1) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp3, fp4);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract23_fp(sp1, tfii, fpaux);
          _sp_pl_eq_sp(sp_c2, sp1);
//#endif
//#if 0
          /* (2) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp6, uprop);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract23_fp(sp1, tfii, fpaux);
          _sp_pl_eq_sp(sp_c2, sp1);
//#endif
//#if 0
          /* (3) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, tfii, fp4);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract23_fp(sp1, fp3, fpaux);
          _sp_pl_eq_sp(sp_c2, sp1);
//#endif
//#if 0  /* tr */
          /* (4) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, tfii, fp6);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract34_fp(sp1, uprop, fpaux);
          _sp_pl_eq_sp(sp_c2, sp1);
//#endif
          /* add and assign */
          _sp_pl_eq_sp_ti_re( res[ix*ncomp+icomp], sp_c2, -comp_list_sign[icomp]);

          /*********************
           * C_3
           *********************/
          _sp_eq_zero( sp_c3 );
//#if 0  /* tr */
          /* (1) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp5, uprop);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract34_fp(sp1, pfifi, fpaux);
          _sp_pl_eq_sp(sp_c3, sp1);
//#endif
//#if 0
          /* (2) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp5, uprop);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 );*/
          _sp_eq_fp_del_contract23_fp(sp1, pfifi, fpaux);
          _sp_pl_eq_sp(sp_c3, sp1);
//#endif
//#if 0
          /* (3) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp5, pfifi);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract23_fp(sp1, uprop, fpaux);
          _sp_pl_eq_sp(sp_c3, sp1);
//#endif
//#if 0  /* tr */
          /* (4) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp5, pfifi);
          /* reduce to spin propagator */
          /*_sp_eq_zero( sp1 ); */
          _sp_eq_fp_del_contract34_fp(sp1, uprop, fpaux);
          _sp_pl_eq_sp(sp_c3, sp1);
//#endif
          _sp_pl_eq_sp_ti_re( res[ix*ncomp+icomp], sp_c3, comp_list_sign[icomp]);
        }  /* of icomp */

      }    /* of ix */

      free_fp(&uprop);
      free_fp(&dprop);
      free_fp(&tfii);
      free_fp(&tffi);
      free_fp(&pfifi);
      free_fp(&pffii);
      free_fp(&fp1);
      free_fp(&fp2);
      free_fp(&fp3);
      free_fp(&fp4);
      free_fp(&fp5);
      free_fp(&fp6);
      free_fp(&fpaux);

      free_sp(&sp_c1);
      free_sp(&sp_c2);
      free_sp(&sp_c3);
      free_sp(&sp1);
      free_sp(&sp2);

#ifdef OPENMP
}
#endif
      retime = _GET_TIME;
      if(g_cart_id == 0)  fprintf(stdout, "# [contract_piN2piN] time for contractions = %e seconds\n", retime-ratime);
  
  return(0);
}  /* end of contract_piN2piN_oet */

}  /* end of namespace cvc */
