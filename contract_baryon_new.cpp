/****************************************************
 * contract_baryon.c
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

#include "basic_types.h"

namespace cvc {

void allocate_memory_for_piN_piN_contractions(storage_for_piN_piN_contractions_type *storage_for_piN_piN_contractions){
  create_fp(&storage_for_piN_piN_contractions->uprop);
  create_fp(&storage_for_piN_piN_contractions->dprop);
  create_fp(&storage_for_piN_piN_contractions->tfii);
  create_fp(&storage_for_piN_piN_contractions->tffi);
  create_fp(&storage_for_piN_piN_contractions->pffii);
  create_fp(&storage_for_piN_piN_contractions->fp1);
  create_fp(&storage_for_piN_piN_contractions->fp2);
  create_fp(&storage_for_piN_piN_contractions->fp3);
  create_fp(&storage_for_piN_piN_contractions->fp4);
  create_fp(&storage_for_piN_piN_contractions->fp6);
  create_fp(&storage_for_piN_piN_contractions->fpaux);

  create_sp(&storage_for_piN_piN_contractions->sp1);
}

void prepare_fermion_propagators_for_piN_piN(

/******************************************************
 ******************************************************
 **
 ** pi+ x proton - pi+ x proton
 **
 ******************************************************
 ******************************************************/
int contract_piN_piN (spinor_propagator_type **res, double**uprop_list, double**dprop_list, double**tfii_list, double**tffi_list, double**pffii_list, 
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
      unsigned int ix, iix;
      int icomp;
      storage_for_piN_piN_contractions_type storage_for_piN_piN_contractions;

      allocate_memory_for_piN_piN_contractions(&storage_for_piN_piN_contractions);

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for(ix=0; ix<VOLUME; ix++)
      {
        /* assign the propagator points */

        assign_propagators_for_piN_piN_contractions_from_point(&storage_for_piN_piN_contractions,ix);

        _assign_fp_point_from_field(uprop, uprop_list, ix);
        _assign_fp_point_from_field(dprop, dprop_list, ix);

        _assign_fp_point_from_field(tfii,  tfii_list,  ix);

        _assign_fp_point_from_field(tffi,  tffi_list,  ix);

        _assign_fp_point_from_field(pffii, pffii_list, ix);

        for(icomp=0; icomp<ncomp; icomp++) {

          iix = ix*ncomp+icomp;

          /******************************************************
           * prepare fermion propagators
           ******************************************************/
          prepare_fermion_propagators_for_piN_piN(&storage_for_piN_piN_contractions,iix,comp_list[icomp]);
          _fp_eq_zero(fp1);
          _fp_eq_zero(fp2);
          _fp_eq_zero(fp3);
          _fp_eq_zero(fp4);
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

// box diagram with one trace

//#if 0  /* pffii */
          /* (1) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, pffii, fp1);
          /* reduce to spin propagator */
          _sp_eq_zero( sp1 );
          _sp_eq_fp_del_contract23_fp(sp1, fp3, fpaux);
          _sp_eq_sp_ti_re( res[0][iix], sp1, comp_list_sign[icomp]);
//#endif
// box diagram with two traces

//#if 0  /* tr */
          /* (2) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, pffii, fp2);
          /* reduce to spin propagator */
          _sp_eq_zero( sp1 );
          _sp_eq_fp_del_contract34_fp(sp1, uprop, fpaux);
          _sp_eq_sp_ti_re( res[1][iix], sp1, comp_list_sign[icomp]);
//#endif

          /*********************
           * C_2
           *********************/
//#if 0
          /* (1) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp3, fp4);
          /* reduce to spin propagator */
          _sp_eq_zero( sp1 );
          _sp_eq_fp_del_contract23_fp(sp1, tfii, fpaux);
          _sp_eq_sp_ti_re( res[2][iix], sp1, -comp_list_sign[icomp]);
//#endif
//#if 0
          /* (2) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp6, uprop);
          /* reduce to spin propagator */
          _sp_eq_zero( sp1 );
          _sp_eq_fp_del_contract23_fp(sp1, tfii, fpaux);
          _sp_eq_sp_ti_re( res[3][iix], sp1, -comp_list_sign[icomp]);
//#endif
//#if 0
          /* (3) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, tfii, fp4);
          /* reduce to spin propagator */
          _sp_eq_zero( sp1 );
          _sp_eq_fp_del_contract23_fp(sp1, fp3, fpaux);
          _sp_eq_sp_ti_re( res[4][iix], sp1, -comp_list_sign[icomp]);
//#endif
//#if 0  /* tr */
          /* (4) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, tfii, fp6);
          /* reduce to spin propagator */
          _sp_eq_zero( sp1 );
          _sp_eq_fp_del_contract34_fp(sp1, uprop, fpaux);
//#endif
          _sp_eq_sp_ti_re( res[5][iix], sp1, -comp_list_sign[icomp]);

        }  /* end of loop on components */

      }    /* end of loop on ix */

      free_fp(&uprop);
      free_fp(&dprop);
      free_fp(&tfii);
      free_fp(&tffi);
      free_fp(&pffii);
      free_fp(&fp1);
      free_fp(&fp2);
      free_fp(&fp3);
      free_fp(&fp4);
      free_fp(&fp6);
      free_fp(&fpaux);

      free_sp(&sp1);

#ifdef HAVE_OPENMP
}
#endif
      retime = _GET_TIME;
      if(g_cart_id == 0)  fprintf(stdout, "# [contract_piN_piN] time for contractions = %e seconds\n", retime-ratime);
  
  return(0);
}  /* end of contract_piN_piN */




/******************************************************
 ******************************************************
 **
 ** pi+ x proton - pi+ x proton
 **
 **   contrcations for Z-diagrams
 **   using the one-end-trick
 ******************************************************
 ******************************************************/
int contract_piN_piN_oet (spinor_propagator_type **res, double**uprop_list, double**dprop_list, double**pfifi_list, int ncomp, int(*comp_list)[2], double*comp_list_sign) {

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
#pragma omp parallel shared(res)
{
#endif
  unsigned int ix, iix;
  int icomp;
  fermion_propagator_type fp1, fp2, fpaux, fp5, uprop, dprop;
  fermion_propagator_type pfifi;
  spinor_propagator_type sp1;

  create_fp(&uprop);
  create_fp(&dprop);
  create_fp(&pfifi);
  create_fp(&fp1);
  create_fp(&fp2);
  create_fp(&fp5);
  create_fp(&fpaux);

  create_sp(&sp1);

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
      for(ix=0; ix<VOLUME; ix++)
      {
        /* assign the propagator points */

        _assign_fp_point_from_field(uprop, uprop_list, ix);
        _assign_fp_point_from_field(dprop, dprop_list, ix);
        _assign_fp_point_from_field(pfifi, pfifi_list, ix);


        for(icomp=0; icomp<ncomp; icomp++) {

          iix = ix * ncomp + icomp;

          /******************************************************
           * prepare fermion propagators
           ******************************************************/
          _fp_eq_zero(fp1);
          _fp_eq_zero(fp2);
          _fp_eq_zero(fp5);
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

          /*********************
           * C_3
           *********************/
//#if 0  /* tr */
          /* (1) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp5, uprop);
          /* reduce to spin propagator */
          _sp_eq_zero( sp1 );
          _sp_eq_fp_del_contract34_fp(sp1, pfifi, fpaux);
          _sp_eq_sp_ti_re( res[0][iix], sp1, comp_list_sign[icomp]);
//#endif
//#if 0
          /* (2) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp5, uprop);
          /* reduce to spin propagator */
          _sp_eq_zero( sp1 );
          _sp_eq_fp_del_contract23_fp(sp1, pfifi, fpaux);
          _sp_eq_sp_ti_re( res[1][iix], sp1, comp_list_sign[icomp]);
//#endif
//#if 0
          /* (3) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp5, pfifi);
          /* reduce to spin propagator */
          _sp_eq_zero( sp1 );
          _sp_eq_fp_del_contract23_fp(sp1, uprop, fpaux);
          _sp_eq_sp_ti_re( res[2][iix], sp1, comp_list_sign[icomp]);
//#endif
//#if 0  /* tr */
          /* (4) */
          /* reduce */
          /*_fp_eq_zero(fpaux); */
          _fp_eq_fp_eps_contract13_fp(fpaux, fp5, pfifi);
          /* reduce to spin propagator */
          _sp_eq_zero( sp1 );
          _sp_eq_fp_del_contract34_fp(sp1, uprop, fpaux);
          _sp_pl_eq_sp_ti_re( res[3][iix], sp1, comp_list_sign[icomp]);
//#endif
        }  /* of icomp */

      }    /* of ix */

      free_fp(&uprop);
      free_fp(&dprop);
      free_fp(&pfifi);
      free_fp(&fp1);
      free_fp(&fp2);
      free_fp(&fp5);
      free_fp(&fpaux);

      free_sp(&sp1);

#ifdef HAVE_OPENMP
}
#endif
      retime = _GET_TIME;
      if(g_cart_id == 0)  fprintf(stdout, "# [contract_piN_piN_oet] time for oet contractions = %e seconds\n", retime-ratime);
  
  return(0);
}  /* end of contract_piN_piN_oet */

/******************************************************
 ******************************************************
 **
 ** proton - proton
 **
 ******************************************************
 ******************************************************/
int contract_N_N (spinor_propagator_type **res, double**uprop_list, double**dprop_list, int ncomp, int(*comp_list)[2], double*comp_list_sign) {
  /******************************************************
   * contractions
   ******************************************************/
  double ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  /* variables */
  unsigned int ix, iix;
  int icomp;
  fermion_propagator_type fp1, fp2, fp3, fpaux, uprop, dprop;
  spinor_propagator_type sp1, sp2;

  create_fp(&fp1);
  create_fp(&fp2);
  create_fp(&fp3);
  create_fp(&uprop);
  create_fp(&dprop);

  create_sp(&sp1);
  create_sp(&sp2);

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix=0; ix<VOLUME; ix++) {

    for(icomp=0; icomp<ncomp; icomp++) {
      iix = ix*ncomp+icomp;

      // assign the propagators
      _assign_fp_point_from_field(uprop, uprop_list, ix);

      if(g_fermion_type==_TM_FERMION) {
        _assign_fp_point_from_field(dprop, dprop_list, ix);
      } else {
        _fp_eq_fp(dprop, uprop);
      }

      // S_u x Cg5
      //_fp_eq_fp_ti_Cg5(fp1, uprop, fp3);
      //
      // S_u x C x Gamma_1
      _fp_eq_zero(fp1);
      _fp_eq_zero(fp3);
      _fp_eq_fp_ti_gamma(fp1, 0, uprop);
      _fp_eq_fp_ti_gamma(fp3, 2, fp1);
      _fp_eq_fp_ti_gamma(fp1, comp_list[icomp][0], fp3);

      // Cg5 x S_d
      // _fp_eq_Cg5_ti_fp(fp2, dprop, fp3);
      //
      // C x Gamma_2 x S_d
      _fp_eq_zero(fp2);
      _fp_eq_zero(fp3);
      _fp_eq_gamma_ti_fp(fp2, comp_list[icomp][1], dprop);
      _fp_eq_gamma_ti_fp(fp3, 2, fp2);
      _fp_eq_gamma_ti_fp(fp2, 0, fp3);
      
      /******************************************************
       * first contribution
       ******************************************************/

      // reduce
      _fp_eq_zero(fp3);
      _fp_eq_fp_eps_contract13_fp(fp3, fp1, fp2);

      // reduce to spin propagator
      _sp_eq_zero( sp1 );
      _sp_eq_fp_del_contract34_fp(sp1, uprop, fp3);

      _sp_eq_sp_ti_re( res[0][iix], sp1,comp_list_sign[icomp]);

      /******************************************************
       * second contribution
       ******************************************************/

      // reduce
      _fp_eq_zero(fp3);
      _fp_eq_fp_eps_contract24_fp(fp3, fp1, fp2);

      // reduce to spin propagator
      _sp_eq_zero( sp2 );
      _sp_eq_fp_del_contract23_fp(sp2, fp3, uprop);

      _sp_eq_sp_ti_re( res[1][iix], sp2,comp_list_sign[icomp]);
    }  /* of icomp */

  }  // end of loop on VOLUME

 free_fp(&fp1);
 free_fp(&fp2);
 free_fp(&fp3);
 free_fp(&uprop);
 free_fp(&dprop);

 free_sp(&sp1);
 free_sp(&sp2);

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

  double retime = _GET_TIME;
  if(g_cart_id == 0)  fprintf(stdout, "# [contract_N_N] time for contractions = %e seconds\n", retime-ratime);

  return(0);
}  /* end of contract_N_N */


/******************************************************
 ******************************************************
 **
 ** Delta++ - Delta++
 **
 ******************************************************
 ******************************************************/
int contract_D_D (spinor_propagator_type **res, double**uprop_list, double**dprop_list, int ncomp, int(*comp_list)[2], double*comp_list_sign) {
  /******************************************************
   * contractions
   ******************************************************/
  double ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif
  /* variables */
  unsigned int ix, iix;
  int icomp;
  fermion_propagator_type fp1, fp2, fp3, fpaux, uprop;
  spinor_propagator_type sp1, sp2;

  create_fp(&fp1);
  create_fp(&fp2);
  create_fp(&fp3);
  create_fp(&fpaux);
  create_fp(&uprop);

  create_sp(&sp1);
  create_sp(&sp2);

#ifdef HAVE_OPENMP
#pragma omp for
#endif
  for(ix=0; ix<VOLUME; ix++)
  {
    /* assign the propagators */
    _assign_fp_point_from_field(uprop, uprop_list, ix);

    for(icomp=0; icomp<ncomp; icomp++) {
      iix = ix*ncomp+icomp;

      /* _sp_eq_zero( connq[ix*ncomp+icomp]); */

      /******************************************************
       * prepare propagators
       ******************************************************/
      /* fp1 = C Gamma_1 x S_u = g0 g2 Gamma_1 S_u */
      _fp_eq_zero(fp1);
      _fp_eq_zero(fpaux);
      _fp_eq_gamma_ti_fp(fp1, comp_list[icomp][0], uprop);
      _fp_eq_gamma_ti_fp(fpaux, 2, fp1);
      _fp_eq_gamma_ti_fp(fp1, 0, fpaux);
      /* fp2 = C Gamma_1 x S_u x C Gamma_2 */
      _fp_eq_zero(fp2);
      _fp_eq_zero(fpaux);
      _fp_eq_fp_ti_gamma(fp2, 0, fp1);
      _fp_eq_fp_ti_gamma(fpaux, 2, fp2);
      _fp_eq_fp_ti_gamma(fp2, comp_list[icomp][1], fpaux);
      /* fp3 = S_u x C Gamma_2 = S_u g0 g2 Gamma_2 */
      _fp_eq_zero(fp3);
      _fp_eq_zero(fpaux);
      _fp_eq_fp_ti_gamma(fp3, 0, uprop);
      _fp_eq_fp_ti_gamma(fpaux, 2, fp3);
      _fp_eq_fp_ti_gamma(fp3, comp_list[icomp][1], fpaux);


      /******************************************************
       * contractions
       ******************************************************/
      /* (1) */
      /* reduce */
      _fp_eq_zero(fpaux);
      _fp_eq_fp_eps_contract13_fp(fpaux, fp1, uprop);
      /* reduce to spin propagator */
      _sp_eq_zero( sp1 );
      _sp_eq_fp_del_contract23_fp(sp1, fp3, fpaux);
      _sp_eq_sp_ti_re(res[0][iix], sp1, -comp_list_sign[icomp]);

      /* (2) */
      /* reduce to spin propagator */
      _sp_eq_zero( sp2 );
      _sp_eq_fp_del_contract24_fp(sp2, fp3, fpaux);
      _sp_eq_sp_ti_re(res[1][iix], sp2, -comp_list_sign[icomp]);


      /* (3) */
      /* reduce */
      _fp_eq_zero(fpaux);
      _fp_eq_fp_eps_contract13_fp(fpaux, fp2, uprop);
      /* reduce to spin propagator */
      _sp_eq_zero( sp1 );
      _sp_eq_fp_del_contract23_fp(sp1, uprop, fpaux);
      _sp_eq_sp_ti_re(res[2][iix], sp1, -comp_list_sign[icomp]);


      /* (4) */
      /* reduce */
      _fp_eq_zero(fpaux);
      _fp_eq_fp_eps_contract13_fp(fpaux, fp1, fp3);
      /* reduce to spin propagator */
      _sp_eq_zero( sp2 );
      _sp_eq_fp_del_contract24_fp(sp2, uprop, fpaux);
      _sp_eq_sp_ti_re(res[3][iix], sp2, -comp_list_sign[icomp]);

      /* (5) */
      /* reduce */
      _fp_eq_zero(fpaux);
      _fp_eq_fp_eps_contract13_fp(fpaux, fp2, uprop);
      /* reduce to spin propagator */
      _sp_eq_zero( sp1 );
      _sp_eq_fp_del_contract34_fp(sp1, uprop, fpaux);
      _sp_eq_sp_ti_re(res[4][iix], sp1, -comp_list_sign[icomp]);

      /* (6) */
      /* reduce */
      _fp_eq_zero(fpaux);
      _fp_eq_fp_eps_contract13_fp(fpaux, fp1, fp3);
      /* reduce to spin propagator */
      _sp_eq_zero( sp2 );
      _sp_eq_fp_del_contract34_fp(sp2, uprop, fpaux);
      _sp_eq_sp_ti_re(res[5][iix], sp2, -comp_list_sign[icomp]);
    }  /* of icomp */

  }    /* of ix */


  free_fp(&fp1);
  free_fp(&fp2);
  free_fp(&fp3);
  free_fp(&fpaux);
  free_fp(&uprop);

  free_sp(&sp1);
  free_sp(&sp2);

#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

  double retime = _GET_TIME;
  if(g_cart_id == 0)  fprintf(stdout, "# [contract_D_D] time for contractions = %e seconds\n", retime-ratime);

  return(0);
}  /* end of contract_D_D */

/******************************************************
 ******************************************************
 **
 ** pi+ x proton - Delta++
 **
 ******************************************************
 ******************************************************/
int contract_piN_D (spinor_propagator_type **res, double**uprop_list, double**dprop_list, double**tfii_list, int ncomp, int(*comp_list)[2], double*comp_list_sign) {

    /******************************************************
     * contractions
     ******************************************************/
    double ratime = _GET_TIME;

#ifdef HAVE_OPENMP
#pragma omp parallel
{
#endif

    /* variables */
    unsigned int ix, iix;
    int icomp;
    fermion_propagator_type fp1, fp2, fp3, fp4, fpaux, uprop, dprop;
    spinor_propagator_type sp1, sp2;
  
    create_fp(&fp1);
    create_fp(&fp2);
    create_fp(&fp3);
    create_fp(&fp4);
    create_fp(&fpaux);
    create_fp(&uprop);
    create_fp(&dprop);
  
    create_sp(&sp1);
    create_sp(&sp2);

#ifdef HAVE_OPENMP
#pragma omp for
#endif

    for(ix=0; ix<VOLUME; ix++) {
      /* int seq_prop_id = ( (int)( fermion_type == _TM_FERMION ) + 1 ) * n_s*n_c; */
      /* assign the propagators */
      _assign_fp_point_from_field(uprop, uprop_list, ix);
      _assign_fp_point_from_field(dprop, tfii_list, ix);

      for(icomp=0; icomp<ncomp; icomp++) {
        iix = ix*ncomp+icomp;

        /******************************************************
         * prepare fermion propagators
         ******************************************************/
        _fp_eq_zero(fp1);
        _fp_eq_zero(fp2);
        _fp_eq_zero(fp3);
        _fp_eq_zero(fp4);
        _fp_eq_zero(fpaux);
        /* fp1 = C Gamma_1 x S_u = g0 g2 Gamma_1 S_u */
        _fp_eq_gamma_ti_fp(fp1, comp_list[icomp][0], uprop);
        _fp_eq_gamma_ti_fp(fpaux, 2, fp1);
        _fp_eq_gamma_ti_fp(fp1,   0, fpaux);

        /*  fp2 = C Gamma_1 x S_u x C Gamma_2 = fp1 x g0 g2 Gamma_2 */
        _fp_eq_fp_ti_gamma(fp2, 0, fp1);
        _fp_eq_fp_ti_gamma(fpaux, 2, fp2);
        _fp_eq_fp_ti_gamma(fp2, comp_list[icomp][1], fpaux);
 
        /* fp3 = S_u x C Gamma_2 = uprop x g0 g2 Gamma_2 */
        _fp_eq_fp_ti_gamma(fp3,   0, uprop);
        _fp_eq_fp_ti_gamma(fpaux, 2, fp3);
        _fp_eq_fp_ti_gamma(fp3, comp_list[icomp][1], fpaux);
 
        /* fp4 = C Gamma_1 x S_seq = g0 g2 Gamma_1 dprop  */
        _fp_eq_gamma_ti_fp(fp4, comp_list[icomp][0], dprop);
        _fp_eq_gamma_ti_fp(fpaux, 2, fp4);
        _fp_eq_gamma_ti_fp(fp4,   0, fpaux);

        /* (1) */
        /* reduce */
        _fp_eq_zero(fpaux);
        _fp_eq_fp_eps_contract13_fp(fpaux, fp2, uprop);
        /* reduce to spin propagator */
        _sp_eq_zero( sp1 );
        _sp_eq_fp_del_contract23_fp(sp1, dprop, fpaux);
        _sp_eq_sp_ti_re(res[0][iix], sp1, -comp_list_sign[icomp]);


        /* (2) */
        /* reduce */
        _fp_eq_zero(fpaux);
        _fp_eq_fp_eps_contract13_fp(fpaux, fp1, fp3);
        /* reduce to spin propagator */
        _sp_eq_zero( sp2 );
        _sp_eq_fp_del_contract24_fp(sp2, dprop, fpaux);
        _sp_eq_sp_ti_re(res[1][iix], sp2, -comp_list_sign[icomp]);
  
        /* (3) */
        /* reduce */
        _fp_eq_zero(fpaux);
        _fp_eq_fp_eps_contract13_fp(fpaux, fp4, uprop);
        /* reduce to spin propagator */
        _sp_eq_zero( sp1 );
        _sp_eq_fp_del_contract23_fp(sp1, fp3, fpaux);
        _sp_eq_sp_ti_re(res[2][iix], sp1, -comp_list_sign[icomp]);

        /* (4) */
        /* reduce */
        _fp_eq_zero(fpaux);
        _fp_eq_fp_eps_contract13_fp(fpaux, fp1, dprop);
        /* reduce to spin propagator */
        _sp_eq_zero( sp2 );
        _sp_eq_fp_del_contract24_fp(sp2, fp3, fpaux);
        _sp_eq_sp_ti_re(res[3][iix], sp2, -comp_list_sign[icomp]);
  
        /* (5) */
        /* reduce */
        _fp_eq_zero(fpaux);
        _fp_eq_fp_eps_contract13_fp(fpaux, fp4, fp3);
        /* reduce to spin propagator */
        _sp_eq_zero( sp1 );
        _sp_eq_fp_del_contract34_fp(sp1, uprop, fpaux);
        _sp_eq_sp_ti_re(res[4][iix], sp1, -comp_list_sign[icomp]);
        
        /* (6) */
        /* reduce */
        _fp_eq_zero(fpaux);
        _fp_eq_fp_eps_contract13_fp(fpaux, fp2, dprop);
        /* reduce to spin propagator */
        _sp_eq_zero( sp2 );
        _sp_eq_fp_del_contract34_fp(sp2, uprop, fpaux);
        _sp_eq_sp_ti_re(res[5][iix], sp2, -comp_list_sign[icomp]);

      }  /* of icomp */

    }    /* of ix */
  
   free_fp(&fp1);
   free_fp(&fp2);
   free_fp(&fp3);
   free_fp(&fp4);
   free_fp(&fpaux);
   free_fp(&uprop);
   free_fp(&dprop);
  
   free_sp(&sp1);
   free_sp(&sp2);


#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif

    double retime = _GET_TIME;
    if(g_cart_id == 0)  fprintf(stdout, "# [contract_piN_D] time for contractions = %e seconds\n", retime-ratime);

    return(0);
}  /* end of contract_piN_D */


/***********************************************
 * multiply x-space spinor propagator field
 *   with boundary phase
 ***********************************************/
int add_baryon_boundary_phase (spinor_propagator_type*sp, int tsrc, int ncomp) {

  const unsigned int VOL3 = LX * LY * LZ;

  int it;
  double ratime = _GET_TIME;
  if(g_propagator_bc_type == 0) {
    // multiply with phase factor
    fprintf(stdout, "# [add_baryon_boundary_phase] multiplying with boundary phase factor\n");
    for(it=0;it<T;it++) {
      int ir = (it + g_proc_coords[0] * T - tsrc + T_global) % T_global;
      const complex w1 = { cos( 3. * M_PI*(double)ir / (double)T_global ), sin( 3. * M_PI*(double)ir / (double)T_global ) };
#ifdef HAVE_OPENMP
#pragma omp parallel shared(sp,it)
{
#endif
      unsigned int ix;
      int icomp;
      spinor_propagator_type sp1;
      create_sp(&sp1);
#ifdef HAVE_OPENMP
#pragma omp for
#endif
      for(ix=0;ix<VOL3;ix++) {
        unsigned int iix = (it * VOL3 + ix) * ncomp;
        for(icomp=0; icomp<ncomp; icomp++) {
          _sp_eq_sp(sp1, sp[iix] );
          _sp_eq_sp_ti_co( sp[iix], sp1, w1);
          iix++;
        }
      }
      free_sp(&sp1);
#ifdef HAVE_OPENMP
}  
#endif
    }
  } else if (g_propagator_bc_type == 1) {
    // multiply with step function
    int ir;
    fprintf(stdout, "# [add_baryon_boundary_phase] multiplying with boundary step function\n");
    for(ir=0; ir<T; ir++) {
      it = ir + g_proc_coords[0] * T;  // global t-value, 0 <= t < T_global
      if(it < tsrc) {
#ifdef HAVE_OPENMP
#pragma omp parallel shared(it, sp)
{
#endif
        unsigned int ix;
        int icomp;
        spinor_propagator_type sp1;
        create_sp(&sp1);
#ifdef HAVE_OPENMP
#pragma omp for
#endif
        for(ix=0;ix<VOL3;ix++) {
          unsigned int iix = (it * VOL3 + ix) * ncomp;
          for(icomp=0; icomp<ncomp; icomp++) {
            _sp_eq_sp(sp1, sp[iix] );
            _sp_eq_sp_ti_re( sp[iix], sp1, -1.);
            iix++;
          }
        }
  
        free_sp(&sp1);
#ifdef HAVE_OPENMP
}  /* end of parallel region */
#endif
      }  /* end of if it < tsrc */
    }  /* end of loop on ir */
  }
  double retime = _GET_TIME;
  if(g_cart_id == 0)  fprintf(stdout, "# [add_baryon_boundary_phase] time for boundary phase = %e seconds\n", retime-ratime);

  return(0);
}  /* end of add_boundary_phase */


/***********************************************
 * multiply with phase from source location
 * - using pi1 + pi2 = - ( pf1 + pf2 ), so
 *   pi1 = - ( pi2 + pf1 + pf2 )
 ***********************************************/
int add_source_phase (double****connt, int pi2[3], int pf2[3], int source_coords[3], int ncomp) {

  const double TWO_MPI = 2. * M_PI;

  double ratime, retime;
  int it;

  ratime = _GET_TIME;
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for(it=0; it<T; it++) {
    int k, icomp;
    double phase;
    complex w;
    int pi2_[3] = {0,0,0}, pf2_[3] = {0,0,0};
    spinor_propagator_type sp1;
    create_sp(&sp1);
    if( pi2 != NULL ) memcpy(pi2_, pi2, 3*sizeof(int));
    if( pf2 != NULL ) memcpy(pf2_, pf2, 3*sizeof(int));

    for(k=0; k<g_sink_momentum_number; k++) {
      int pf1_[3] = {  g_sink_momentum_list[k][0], g_sink_momentum_list[k][1], g_sink_momentum_list[k][2]};
      phase = -TWO_MPI * (
          (double)(pf1_[0] + pi2_[0] + pf2_[0] ) / (double)LX_global * source_coords[0]
        + (double)(pf1_[1] + pi2_[1] + pf2_[1] ) / (double)LY_global * source_coords[1]
        + (double)(pf1_[2] + pi2_[2] + pf2_[2] ) / (double)LZ_global * source_coords[2]
                                                                                                          );
      w.re = cos(phase);
      w.im = sin(phase);
      for(icomp=0; icomp<ncomp; icomp++) {
        spinor_propagator_type connt_sp = &(connt[it][k][icomp*g_sv_dim]);
        _sp_eq_sp(sp1, connt_sp );
        _sp_eq_sp_ti_co(connt_sp, sp1, w);
      }  /* end of loop on components */
    }  /* end of loop on sink momenta  / pf1 */
    free_sp(&sp1);
  }  /* end of loop on T */

  retime = _GET_TIME;
  if(g_cart_id == 0)  fprintf(stdout, "# [add_source_phase] time for source phase = %e seconds\n", retime-ratime);
  return(0);
}  /* end of add_source_phase */

}  /* end of namespace cvc */
