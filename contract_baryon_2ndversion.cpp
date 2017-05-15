
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
#include "types.h"
#include "global.h"

using namespace cvc;

#include "basic_types.h"
#include "contract_baryon_2ndversion.h"

namespace cvc {

  typedef double _Complex* V1_type;
  typedef double _Complex*** V2_x_type;
  typedef double _Complex**** V2_type;

  int epsilon[6][4] = {{0,1,2,+1},{0,2,1,-1},{1,0,2,-1},{1,2,0,+1},{2,0,1,+1},{2,1,0,-1}};

  int f_complex_index(int dirac,int color){
    return dirac*3+color;
  }

  int V1_complex_index(int alpha2,int a,int m){
    return (alpha2*3+m)*3+a;
  }

  unsigned int V1_double_size(){
    return 4*4*3*2;
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
    }}}} }
  }

  void V2_eq_fft_V2_x(double **V2_local_fft,V2_x_type V2_x,int io_proc,int num_combinations,int ncomp,int momentum_number, int (*momentum_list)[3]){
    int icombination;
    double **buffer=NULL;
    for(icombination=0;icombination<num_combinations;icombination++){
      momentum_projection2((double*)V2_x[icombination][0],&(V2_local_fft[0][0]),ncomp*V2_double_size()/2,momentum_number,momentum_list,NULL);
    }
  }

void V2_local_fft_to_V2_global_fft(double ***V2_for_b_and_w_diagrams,double ***V2_local_fft,int num_combinations,int ncomp,int momentum_number,program_instruction_type *program_instructions){
      int k = T*momentum_number*ncomp*V2_double_size();
 #ifdef HAVE_MPI
      if(program_instructions->io_proc>0) {
        int exitstatus = MPI_Allgather(&(V2_local_fft[0][0][0]), k, MPI_DOUBLE, &(V2_for_b_and_w_diagrams[0][0][0]), k, MPI_DOUBLE, g_tr_comm);
        if(exitstatus != MPI_SUCCESS) {
          fprintf(stderr, "[piN2piN] Error from MPI_Allgather, status was %d\n", exitstatus);
          EXIT(124);
        }
      }
#else
      memcpy(&(V2_for_b_and_w_diagrams[0][0][0]),&(V2_local_fft[0][0][0]),k);
#endif
}

  void create_fv(fermion_vector_type *fv){
    *fv = (fermion_vector_type)malloc(16*2*sizeof(double));
  }

  void free_fv(fermion_vector_type *fv){
    free(*fv);
    *fv = NULL;
  }

  void create_V1(V1_type *V1){
    *V1 = (V1_type)malloc(V1_double_size()*sizeof(double));
  }

  void free_V1(V1_type *V1){
    free(*V1);
    *V1 = NULL;
  }

  void assign_fv_point_from_field(fermion_vector_type fv,double *stoch_prop,unsigned int ix){
    _fv_eq_fv(fv,stoch_prop+_GSI(ix));
  } 

  void compute_V2_for_b_and_w_diagrams(int i_src,int i_coherent,int ncomp,int *comp_list,double*comp_list_sign,int nsample,V2_for_b_and_w_diagrams_type *V2_for_b_and_w_diagrams,program_instruction_type *program_instructions,double **uprop_list,double **tfii_list,double **phi_list){

    double ratime, retime;

    ratime = _GET_TIME;

    unsigned int VOL3 = program_instructions->VOL3;

    V2_x_type V2_x = NULL;

    init_3level_buffer((double****)&V2_x,3,VOL3*ncomp,V2_double_size());

    double ***V2_local_fft = NULL;
    init_3level_buffer(&V2_local_fft,T,g_sink_momentum_number,ncomp*V2_double_size());

    unsigned int it,isample;
    for(isample=0;isample < nsample;isample++){
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
      printf("test2\n");

  #ifdef HAVE_OPENMP
  #pragma omp parallel for
  #endif
      for(ivx=0;ivx<VOL3;ivx++){
        ix = it*VOL3+ivx;
        /* assign the propagator points */
        _assign_fp_point_from_field(uprop, uprop_list, ix);
        _assign_fp_point_from_field(tfii,  tfii_list,  ix);
        assign_fv_point_from_field(fv_phi, phi_list[isample], ix);

        for(icomp=0; icomp<ncomp; icomp++) {

          iivx = ivx*ncomp+icomp;

          /******************************************************
           * prepare fermion vectors
           ******************************************************/
          _fv_eq_gamma_ti_fv(fv1,comp_list[icomp],fv_phi)
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
   
      printf("t: %d\n" ,it);
      // local fourier transformation
      V2_eq_fft_V2_x((double**)V2_local_fft[it],V2_x,program_instructions->io_proc,3,ncomp,g_sink_momentum_number,g_sink_momentum_list);
    }
      
      V2_local_fft_to_V2_global_fft(&((*V2_for_b_and_w_diagrams)[isample*T]),V2_local_fft,3,ncomp,g_sink_momentum_number,program_instructions);
    }

    fini_3level_buffer((double****)&V2_x);
    fini_3level_buffer(&V2_local_fft);

    retime = _GET_TIME;
    if(g_cart_id == 0)  fprintf(stdout, "# [contract_piN_piN] time for contractions = %e seconds\n", retime-ratime);

  }
}

void gammas_eq_gammas_ti_gammas(int dest_permutation[24],int dest_sign[24],int gamma_permutation_1[24],int gamma_sign_1[24],int gamma_permutation_2[24],int gamma_sign_2[24]){
  for(int i = 0;i < 24;i++){
    dest_permutation[i] = gamma_permutation_1[gamma_permutation_2[i]];
    dest_sign[i] = gamma_sign_2[i]*gamma_sign_1[gamma_permutation_2[i]];
  }
}

int get_icomp_from_single_comps(int icomp_f1,int icomp_i1,int num_components,int *component[2]){
  for(int i=0;i<num_components;i++){
    if(component[i][0] == icomp_f1 && component[i][1] == icomp_i1)
      return i;
  }
  return 0;
}

void compute_b_or_w_diagram_from_V2(gathered_FT_WDc_contractions_type *gathered_FT_WDc_contractions,int diagram,b_1_xi_type *b_1_xi,w_1_xi_type *w_1_xi,V2_for_b_and_w_diagrams_type *V2_for_b_and_w_diagrams,program_instruction_type *program_instructions,int num_component_f1,int *component_f1,int num_component_i1,int *component_i1,int num_components,int *component[2]){

  int gamma_permutation_1[24],gamma_permutation_2[24];
  int gamma_sign_1[24],gamma_sign_2[24];

  for(int icomp_f1=0;icomp_f1<num_component_f1;icomp_f1++){
  for(int icomp_i1=0;icomp_i1<num_component_i1;icomp_i1++){
    int icomp=get_icomp_from_single_comps(icomp_f1,icomp_i1,num_components,component);
    gammas_eq_gammas_ti_gammas(gamma_permutation_2,gamma_sign_2,gamma_permutation[0],gamma_sign[0],gamma_permutation[2],gamma_sign[2]);
    gammas_eq_gammas_ti_gammas(gamma_permutation_1,gamma_sign_1,gamma_permutation_2,gamma_sign_2,gamma_permutation[icomp_i1],gamma_sign[icomp_i1]);
    
//    for(int it=0;it<T;it++){
//    for(

//    }

  }}

}


