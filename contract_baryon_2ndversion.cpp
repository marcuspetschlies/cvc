
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

  void print_2level_buffer(double **buffer,int n1,int n2){
    for(int i1=0;i1<n1;i1++){
    for(int i2=0;i2<n2;i2++){
      printf("%d %d %e\n",i1,i2,buffer[i1][i2]);
    }}
  }

  void print_3level_buffer(double ***buffer,int n1,int n2,int n3){
    for(int i1=0;i1<n1;i1++){
    for(int i2=0;i2<n2;i2++){
    for(int i3=0;i3<n3;i3++){
      printf("%d %d %d %e\n",i1,i2,i3,buffer[i1][i2][i3]);
    }}}
  }

  double comp_fp_sum(fermion_propagator_type fp){
    double sum = 0;
    for(int i1 = 0;i1 < 12; i1++){
    for(int i2 = 0;i2 < 24; i2++){
      sum += fp[i1][i2];
    }}
    return sum;
  }

  double comp_fv_sum(fermion_vector_type fv){
    double sum = 0;
    for(int i1 = 0;i1 < 24; i1++){
      sum += fv[i1];
    }
    return sum;
  }

  typedef double _Complex* V1_type;
  typedef double _Complex*** V2_x_type;
  typedef double _Complex**** V2_type;

  int epsilon[6][4] = {{0,1,2,+1},{0,2,1,-1},{1,0,2,-1},{1,2,0,+1},{2,0,1,+1},{2,1,0,-1}};

  int f_complex_index(int dirac,int color){
    return dirac*3+color;
  }

  int fp_complex_index(int dirac1,int color1,int dirac2,int color2){
    return ((dirac2*3+color2)*4+dirac1)*3+color1;
  }

  double _Complex get_cmplx_from_fp_at(fermion_propagator_type fp,int dirac1,int color1,int dirac2,int color2){
    return ((double _Complex*)fp[0])[fp_complex_index(dirac1,color1,dirac2,color2)];
  }

  double _Complex get_cmplx_from_fv_at(fermion_vector_type fv,int dirac,int color){
    return ((double _Complex*)fv)[f_complex_index(dirac,color)];
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

  int V2_complex_size(){
    return 4*4*4*3;
  }

  int V2_double_size(){
    return V2_complex_size()*2;
  }

  void V1_eq_epsilon_fv_ti_fp(double _Complex *V1,fermion_vector_type fv,fermion_propagator_type fp){
    memset((double*)V1,0,V1_double_size());

    for(int m = 0;m < 3;m++){
    for(int alpha1 = 0;alpha1 < 3;alpha1++){
    for(int alpha2 = 0;alpha2 < 3;alpha2++){
    for(int i = 0;i < 6;i++){
      int c = epsilon[i][0];
      int b = epsilon[i][1];
      int a = epsilon[i][2];
      int sign = epsilon[i][3];
      V1[V1_complex_index(alpha2,a,m)] += sign*get_cmplx_from_fv_at(fv,alpha1,c)*get_cmplx_from_fp_at(fp,alpha1,b,alpha2,m);
    }}}}
  }

  void V2_eq_epsilon_V1_ti_fp(double _Complex *V2,double _Complex *V1,fermion_propagator_type fp){
    memset((double*)V2,0,V2_double_size());

    for(int a = 0;a < 3;a++){
    for(int alpha1 = 0;alpha1 < 3;alpha1++){
    for(int alpha2 = 0;alpha2 < 3;alpha2++){
    for(int alpha3 = 0;alpha3 < 3;alpha3++){
    for(int i = 0;i < 6;i++){
      int n = epsilon[i][0];
      int m = epsilon[i][1];
      int l = epsilon[i][2];
      int sign = epsilon[i][3];
      V2[V2_complex_index(alpha1,alpha2,alpha3,n)] += sign*V1[V1_complex_index(alpha1,a,m)]*get_cmplx_from_fp_at(fp,alpha2,a,alpha3,l);
    }}}} }
  }

  void V2_eq_fft_V2_x(double **V2_fft,V2_x_type V2_x,int io_proc,int num_combinations,int ncomp,int momentum_number, int (*momentum_list)[3]){
    int icombination;
    for(icombination=0;icombination<num_combinations;icombination++){
      momentum_projection2((double*)V2_x[icombination][0],&(V2_fft[icombination*momentum_number][0]),ncomp*V2_complex_size(),momentum_number,momentum_list,NULL);
    }
  }

void gather_V2_ffts(double ***V2_for_b_and_w_diagrams,double ***V2_fft,int num_combinations,int ncomp,int momentum_number,program_instruction_type *program_instructions){
      int k = T*num_combinations*momentum_number*ncomp*V2_double_size();
 #ifdef HAVE_MPI
      if(program_instructions->io_proc>0) {
        int exitstatus = MPI_Allgather(&(V2_fft[0][0][0]), k, MPI_DOUBLE, &(V2_for_b_and_w_diagrams[0][0][0]), k, MPI_DOUBLE, g_tr_comm);
        if(exitstatus != MPI_SUCCESS) {
          fprintf(stderr, "[piN2piN] Error from MPI_Allgather, status was %d\n", exitstatus);
          EXIT(124);
        }
      }
#else
      memcpy(&(V2_for_b_and_w_diagrams[0][0][0]),&(V2_fft[0][0][0]),k);
#endif
}

  void create_fv(fermion_vector_type *fv){
    *fv = (fermion_vector_type)malloc(12*2*sizeof(double));
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

    double ***V2_fft = NULL;
    init_3level_buffer(&V2_fft,T,3*g_sink_momentum_number,ncomp*V2_double_size());

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
        _assign_fp_point_from_field(uprop, uprop_list, ix);  // should work
        _assign_fp_point_from_field(tfii,  tfii_list,  ix); // should work
        assign_fv_point_from_field(fv_phi, phi_list[isample], ix); // should work

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
/*      if(g_cart_id == 0)
        print_3level_buffer((double***)V2_x,3,3*ncomp,20);*/
      // local fourier transformation
      V2_eq_fft_V2_x((double**)V2_fft[it],V2_x,program_instructions->io_proc,3,ncomp,g_sink_momentum_number,g_sink_momentum_list);
      //printf("V2_fft:\n");
      //print_2level_buffer((double**)V2_fft[it],g_sink_momentum_number,ncomp*V2_double_size());
    }
      
      gather_V2_ffts(&((*V2_for_b_and_w_diagrams)[isample*T_global]),V2_fft,3,ncomp,g_sink_momentum_number,program_instructions);
    }

    //memset(&((*V2_for_b_and_w_diagrams)[0][0][0]),0,g_nsample*T_global*3*g_sink_momentum_number*ncomp*V2_double_size());

    fini_3level_buffer((double****)&V2_x);
    fini_3level_buffer(&V2_fft);

    retime = _GET_TIME;
    if(g_cart_id == 0)  fprintf(stdout, "# [contract_piN_piN] time for contractions = %e seconds\n", retime-ratime);

  }

  void dmat_eq_dmat_ti_dmat(double _Complex** dest,double _Complex** left,double _Complex** right){
    memset((double*)&(dest[0][0]),0,4*4*2);
    for(int a=0;a<4;a++){
    for(int b=0;b<4;b++){
    for(int c=0;c<4;c++){
      dest[a][c] += left[a][b]*right[b][c];
    }}}
  }

  void init_gammas(double _Complex**** gammas){

    (*gammas) = NULL;
    init_3level_buffer((double****)gammas,16,4,4*2);

    memset((double*)&((*gammas)[0][0][0]),0,16*4*4*2);

    (*gammas)[0][0][2] = -1;
    (*gammas)[0][1][3] = -1;
    (*gammas)[0][2][0] = -1;
    (*gammas)[0][3][1] = -1;

    (*gammas)[2][0][3] = -1;
    (*gammas)[2][1][2] = +1;
    (*gammas)[2][2][1] = +1;
    (*gammas)[2][3][0] = -1;

    (*gammas)[4][0][0] = +1;
    (*gammas)[4][1][1] = +1;
    (*gammas)[4][2][2] = +1;
    (*gammas)[4][3][3] = +1;

    (*gammas)[5][0][0] = +1;
    (*gammas)[5][1][1] = +1;
    (*gammas)[5][2][2] = -1;
    (*gammas)[5][3][3] = -1;

    dmat_eq_dmat_ti_dmat((*gammas)[6],(*gammas)[0],(*gammas)[5]);
  }

  void fini_gammas(double _Complex**** gammas){
    fini_3level_buffer((double****)gammas);
  }

  void print_dmat(double _Complex **dmat){
    printf("---------------------------------------\n");
    for(int a=0;a<4;a++){
    for(int b=0;b<4;b++){
      printf("%+.2f %+.2fi  ",creal(dmat[a][b]),cimag(dmat[a][b]));
    }
    printf("\n");
    }
    printf("---------------------------------------\n");
  }

  void gammas_eq_gammas_ti_gammas(int dest_permutation[24],int dest_sign[24],int gamma_permutation_1[24],int gamma_sign_1[24],int gamma_permutation_2[24],int gamma_sign_2[24]){
    for(int i = 0;i < 24;i++){
      dest_permutation[i] = gamma_permutation_1[gamma_permutation_2[i]];
      dest_sign[i] = gamma_sign_2[i]*gamma_sign_1[gamma_permutation_2[i]];
    }
  }

  int get_icomp_from_single_comps(int icomp_f1,int icomp_i1,int num_components,int(*component)[2]){
    for(int i=0;i<num_components;i++){
      if(component[i][0] == icomp_f1 && component[i][1] == icomp_i1)
        return i;
    }
    fprintf(stderr,"comp not found %d %d\n\n",icomp_f1,icomp_i1);
    return 0;
  }

  void init_dmat(double _Complex*** dmat){
    *dmat = NULL;
    init_2level_buffer((double***)dmat,4,4*2);
  }

  void fini_dmat(double _Complex*** dmat){
    fini_2level_buffer((double***)dmat);
  }

  void print_V2_for_b_diagrams(V2_for_b_and_w_diagrams_type *V2_for_b_and_w_diagrams,int num_component_f1,int icomp_f1){
    for(int it=0;it<T_global;it++){
    for(int isample=0;isample<g_nsample;isample++){
    for(int i_sink_mom=0;i_sink_mom<g_sink_momentum_number;i_sink_mom++){
//    for(int alpha=0;alpha<4;alpha++){
//    for(int beta=0;beta<4;beta++){
//    for(int delta=0;delta<4;delta++){
//    for(int m=0;m<3;m++){
      int alpha=1,beta=1,delta=1,m=1;
      double _Complex c = ((double _Complex ***)*V2_for_b_and_w_diagrams)[isample*T_global+it][0][(i_sink_mom*num_component_f1+icomp_f1)*V2_double_size()/2+V2_complex_index(beta,alpha,delta,m)];
      printf("isample: %d, it: %d, i_sink_mom: %d, icomp_f1: %d alpha: %d, beta: %d, delta: %d, m: %d, %e %e\n",isample,it,i_sink_mom,icomp_f1,alpha,beta,delta,m,creal(c),cimag(c));
    }}}//}}}}
  }

  void compute_b_or_w_diagram_from_V2(gathered_FT_WDc_contractions_type *gathered_FT_WDc_contractions,int diagram,b_1_xi_type *b_1_xi,w_1_xi_type *w_1_xi,V2_for_b_and_w_diagrams_type *V2_for_b_and_w_diagrams,program_instruction_type *program_instructions,int num_component_f1,int *component_f1,int num_component_i1,int *component_i1,int num_components,int(*component)[2],int tsrc){

    if(program_instructions->io_proc<=0) return;

    double _Complex ***gammas;
    init_gammas(&gammas);

    for(int icomp_f1=0;icomp_f1<num_component_f1;icomp_f1++){
    for(int icomp_i1=0;icomp_i1<num_component_i1;icomp_i1++){
      int icomp=get_icomp_from_single_comps(component_f1[icomp_f1],component_i1[icomp_i1],num_components,component);

      double _Complex **dmat1,**dmat2;
      init_dmat(&dmat1);
      init_dmat(&dmat2);
      dmat_eq_dmat_ti_dmat(dmat1,gammas[0],gammas[2]);
      dmat_eq_dmat_ti_dmat(dmat2,dmat1,gammas[component_i1[icomp_i1]]);
  
      if(diagram == 0){

        if(g_cart_id == 0){
          printf("comp_i1=%d\n",component_i1[icomp_i1]);
          //print_dmat((double _Complex**)dmat2);
          //print_V2_for_b_diagrams(V2_for_b_and_w_diagrams,num_component_f1,icomp_f1);
        }

        for(int it=0;it<T_global;it++){
          int ir = (it - tsrc + T_global) % T_global;
          double _Complex w1 = cos( 3. * M_PI*(double)ir / (double)T_global )+sin( 3. * M_PI*(double)ir / (double)T_global )*_Complex_I;
        for(int isample=0;isample<g_nsample;isample++){
        for(int i_sink_mom=0;i_sink_mom<g_sink_momentum_number;i_sink_mom++){
        for(int alpha=0;alpha<4;alpha++){
        for(int beta=0;beta<4;beta++){
        ((double _Complex ****)*gathered_FT_WDc_contractions)[it][i_sink_mom][icomp*4+alpha][beta] = 0;
        for(int delta=0;delta<4;delta++){
        for(int gamma=0;gamma<4;gamma++){
        for(int m=0;m<3;m++){
          ((double _Complex ****)*gathered_FT_WDc_contractions)[it][i_sink_mom][icomp*4+alpha][beta] += -((double _Complex ***)*V2_for_b_and_w_diagrams)[isample*T_global+it][0][(i_sink_mom*num_component_f1+icomp_f1)*V2_complex_size()+V2_complex_index(beta,alpha,delta,m)]*dmat2[gamma][delta]*((double _Complex ***)*b_1_xi)[it][isample][gamma*3+m]*w1;
        }}}}}}}}

      }
      else{
        for(int it=0;it<T_global;it++){
        for(int isample=0;isample<g_nsample;isample++){
        for(int i_sink_mom=0;i_sink_mom<g_sink_momentum_number;i_sink_mom++){
        for(int alpha=0;alpha<4;alpha++){
        for(int beta=0;beta<4;beta++){
        ((double _Complex ****)*gathered_FT_WDc_contractions)[it][i_sink_mom][icomp*4+alpha][beta] = 0;
        }}}}}
      }

      fini_dmat(&dmat1);
      fini_dmat(&dmat2);

    }}

    fini_gammas(&gammas);

  }

}

