
int epsilon[][] = {{0,1,2,+1},{0,2,1,-1},{1,0,2,-1},{1,2,0,+1},{2,0,1,+1},{2,1,0,-1}}

int f_index(int dirac,int color){
  return dirac*3+color;
}

int V1_index(int dirac1,int color1,int color2){
  return (dirac1*3+color2)*3+color1;
}

int V2_index(int dirac1,int dirac2,int dirac3,int color1){
  return ((dirac1*4+dirac2)*4+dirac3)*3+color1;
}

void V1_eq_epsilon_fv_ti_fp(double *V1,fermion_vector_type fv,fermion_propagator_type fp){
  for(int n = 0;n < 3;n++){
  for(int alpha = 0;alpha < 3;alpha++){
  for(int beta = 0;beta < 3;beta++){
  for(int i = 0;i < 6;i++){
    int a = epsilon[i][0];
    int b = epsilon[i][1];
    int c = epsilon[i][2];
    int sign = epsilon[i][3];
    V1[V1_index(beta,a,n)] = sign*fv[f_index(alpha,b)]*fp[f_index(alpha,c)][f_index(beta,n)];
  }}}}
}

void V2_eq_epsilon_V1_ti_fp(double *V2,double *V1,fermion_propagator_type fp){
  for(int a = 0;a < 3;a++){
  for(int alpha = 0;alpha < 3;alpha++){
  for(int beta = 0;beta < 3;beta++){
  for(int delta = 0;delta < 3;delta++){
  for(int i = 0;i < 6;i++){
    int m = epsilon[i][0];
    int n = epsilon[i][1];
    int l = epsilon[i][2];
    int sign = epsilon[i][3];
    V2[V2_index(beta,alpha,delta,m)] = sign*V1[V1_index(beta,a,n)]*fp[f_index(alpha,a)][f_index(delta,l)];
  }}}} 
}






