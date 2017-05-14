
int epsilon[][] = {{0,1,2,+1},{0,2,1,-1},{1,0,2,-1},{1,2,0,+1},{2,0,1,+1},{2,1,0,-1}}

int f_index(int dirac,int color){
  return dirac*3+color;
}

int V1_index(int alpha2,int a,int m){
  return (alpha2*3+m)*3+a;
}

int V2_index(int alpha1,int alpha2,int alpha3,int n){
  return ((alpha1*4+alpha2)*4+alpha3)*3+n;
}

void V1_eq_epsilon_fv_ti_fp(double *V1,fermion_vector_type fv,fermion_propagator_type fp){
  for(int m = 0;m < 3;m++){
  for(int alpha1 = 0;alpha1 < 3;alpha1++){
  for(int alpha2 = 0;alpha2 < 3;alpha2++){
  for(int i = 0;i < 6;i++){
    int c = epsilon[i][0];
    int b = epsilon[i][1];
    int a = epsilon[i][2];
    int sign = epsilon[i][3];
    V1[V1_index(alpha2,a,m)] = sign*fv[f_index(alpha1,c)]*fp[f_index(alpha1,b)][f_index(alpha2,m)];
  }}}}
}

void V2_eq_epsilon_V1_ti_fp(double *V2,double *V1,fermion_propagator_type fp){
  for(int a = 0;a < 3;a++){
  for(int alpha1 = 0;alpha1 < 3;alpha1++){
  for(int alpha2 = 0;alpha2 < 3;alpha2++){
  for(int alpha3 = 0;alpha3 < 3;alpha3++){
  for(int i = 0;i < 6;i++){
    int n = epsilon[i][0];
    int m = epsilon[i][1];
    int l = epsilon[i][2];
    int sign = epsilon[i][3];
    V2[V2_index(alpha1,alpha2,alpha3,n)] = sign*V1[V1_index(alpha1,a,m)]*fp[f_index(alpha2,a)][f_index(alpha3,l)];
  }}}} 
}

void compute_all_b_phis_and_all_w_phis(int i_src,int i_coherent,b_all_phis_type *b_all_phis,w_all_phis_type *w_all_phis,program_instruction_type *program_instructions,forward_propagators_type *forward_propagators,sequential_propagators_type *sequential_propagators,stochastic_sources_and_propagators_type *stochastic_sources_and_propagators){

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
    fermion_propagator_type fp1, fp2, fp3, fpaux, fp4, fp6, uprop, dprop;
    fermion_propagator_type tfii, tffi, pffii;
    spinor_propagator_type sp1;

    create_fp(&uprop);
    create_fp(&dprop);
    create_fp(&tfii);
    create_fp(&tffi);
    create_fp(&pffii);
    create_fp(&fp1);
    create_fp(&fp2);
    create_fp(&fp3);
    create_fp(&fp4);
    create_fp(&fp6);
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

      _assign_fp_point_from_field(tfii,  tfii_list,  ix);

      _assign_fp_point_from_field(tffi,  tffi_list,  ix);

      _assign_fp_point_from_field(pffii, pffii_list, ix);

      for(icomp=0; icomp<ncomp; icomp++) {

        iix = ix*ncomp+icomp;

        /******************************************************
         * prepare fermion propagators
         ******************************************************/
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

}



