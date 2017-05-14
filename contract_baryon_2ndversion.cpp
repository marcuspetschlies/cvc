
typedef double* V1_type;
typedef double _Complex*** V2_x_type;

int epsilon[][] = {{0,1,2,+1},{0,2,1,-1},{1,0,2,-1},{1,2,0,+1},{2,0,1,+1},{2,1,0,-1}}

int f_complex_index(int dirac,int color){
  return dirac*3+color;
}

int V1_complex_index(int alpha2,int a,int m){
  return (alpha2*3+m)*3+a;
}

int V2_complex_index(int alpha1,int alpha2,int alpha3,int n){
  return ((alpha1*4+alpha2)*4+alpha3)*3+n;
}

int V2_double_size(){
  return 4*4*4*3*2;
}

void V1_eq_epsilon_fv_ti_fp(double _Complex *V1,fermion_vector_type fv,fermion_propagator_type fp){
  for(int m = 0;m < 3;m++){
  for(int alpha1 = 0;alpha1 < 3;alpha1++){
  for(int alpha2 = 0;alpha2 < 3;alpha2++){
  for(int i = 0;i < 6;i++){
    int c = epsilon[i][0];
    int b = epsilon[i][1];
    int a = epsilon[i][2];
    int sign = epsilon[i][3];
    V1[V1_complex_index(alpha2,a,m)] = sign*fv[f_complex_index(alpha1,c)]*((double _Complex*)fp[f_complex_index(alpha1,b)])[f_complex_index(alpha2,m)];
  }}}}
}

void V2_eq_epsilon_V1_ti_fp(double _Complex *V2,double _Complex *V1,fermion_propagator_type fp){
  for(int a = 0;a < 3;a++){
  for(int alpha1 = 0;alpha1 < 3;alpha1++){
  for(int alpha2 = 0;alpha2 < 3;alpha2++){
  for(int alpha3 = 0;alpha3 < 3;alpha3++){
  for(int i = 0;i < 6;i++){
    int n = epsilon[i][0];
    int m = epsilon[i][1];
    int l = epsilon[i][2];
    int sign = epsilon[i][3];
    V2[V2_complex_index(alpha1,alpha2,alpha3,n)] = sign*V1[V1_complex_index(alpha1,a,m)]*((double _Complex*)fp[f_complex_index(alpha2,a)])[f_complex_index(alpha3,l)];
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

  V2_x_type V2_x;

  init_3level_buffer(((double****)&V2_x),3,VOL3*ncomp,V2_double_size());

  int it;
  for(it=0; it<T;it++){
#ifdef HAVE_OPENMP
#pragma omp parallel shared(res)
{
#endif
    unsigned int ix, ivx, iivx;
    int icomp;
    fermion_propagator_type uprop,tfii;
    fermion_vector_type fv_phi,fv1,fv2;
    V1_type V1;

    create_fp(&uprop);
    create_fp(&tfii);
    create_fv(&fv_phi);
    create_fv(&fv1);
    create_fv(&fv2);
    create_V1(&V1);

#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
    for(ivx=0;ivx<VOL3;ivx++){
      ix = it*VOL3+ivx;
      /* assign the propagator points */
      _assign_fp_point_from_field(uprop, uprop_list, ix);
      _assign_fp_point_from_field(tfii,  tfii_list,  ix);
      assign_fv_point_from_field(fv_phi, phi_list, ix);

      for(icomp=0; icomp<ncomp; icomp++) {

        iivx = ivx*ncomp+icomp;

        /******************************************************
         * prepare fermion vectors
         ******************************************************/
        _fv_eq_gamma_ti_fv(fv1,comp_list[icomp][0],fv_phi)
        _fv_eq_gamma_ti_fv(fv2,2,fv1)
        _fv_eq_gamma_ti_fv(fv1,0,fv2)

        /******************************************************
         * B-diagrams
         ******************************************************/

        V1_eq_epsilon_fv_ti_fp(V1,fv1,uprop);
        V2_eq_epsilon_V1_ti_fp(V2_x[0][iivx],V1,uprop);

        /******************************************************
         * W-diagrams
         ******************************************************/

        V1_eq_epsilon_fv_ti_fp(V1,fv1,uprop);
        V2_eq_epsilon_V1_ti_fp(V2_x[1][iivx],V1,tfii);

        V1_eq_epsilon_fv_ti_fp(V1,fv1,tfii);
        V2_eq_epsilon_V1_ti_fp(V2_x[2][iivx],V1,uprop);

      }  /* end of loop on components */

    }    /* end of loop on ix */

    free_fp(&uprop);
    free_fp(&tfii);
    free_fv(&fv_phi);
    free_fv(&fv1);
    free_fv(&fv2);
    free_V1(&V1);

#ifdef HAVE_OPENMP
}
#endif

    // local fourier transformation
    V2_eq_fft_V2_x(&V2,V2_x,3,g_sink_momentum_number,g_sink_momentum_list);
  }

  free_3level_buffer(V2_x);

  retime = _GET_TIME;
  if(g_cart_id == 0)  fprintf(stdout, "# [contract_piN_piN] time for contractions = %e seconds\n", retime-ratime);

}



