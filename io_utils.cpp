/* $Id: io_utils.c,v 1.2 2007/11/22 15:57:38 urbach Exp $ */

#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<time.h>
#include<sys/time.h> 
#include<sys/types.h>
#include<math.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#  include <unistd.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#  include"lime.h" 

#ifdef __cplusplus
}
#endif

namespace cvc {

#include"io_utils.h"


int isnan_f  (float       x) { return x != x; }
int isnan_d  (double      x) { return x != x; }
int isnan_ld (long double x) { return x != x; }

int big_endian(){
  union{
    int l;
    char c[sizeof(int)];
  } u;

  u.l=1;
  return(u.c[sizeof(int) - 1] == 1);
}

void byte_swap(void * ptr, int nmemb){
  int j;
  char char_in[4];
  char * in_ptr;
  int * int_ptr;

  for(j = 0, int_ptr = (int *) ptr; j < nmemb; j++, int_ptr++) {
    in_ptr = (char *) int_ptr;
    
    char_in[0] = in_ptr[0];
    char_in[1] = in_ptr[1];
    char_in[2] = in_ptr[2];
    char_in[3] = in_ptr[3];

    in_ptr[0] = char_in[3];
    in_ptr[1] = char_in[2];
    in_ptr[2] = char_in[1];
    in_ptr[3] = char_in[0];
  }
}

void byte_swap_double(void * ptr, int nmemb){
  int j;
  char char_in[8];
  char * in_ptr;
  int * int_ptr;

  for(j = 0, int_ptr = (int *) ptr; j < nmemb; j++, int_ptr++) {
    in_ptr = (char *) int_ptr;
    
    char_in[0] = in_ptr[0];
    char_in[1] = in_ptr[1];
    char_in[2] = in_ptr[2];
    char_in[3] = in_ptr[3];
    char_in[4] = in_ptr[4];
    char_in[5] = in_ptr[5];
    char_in[6] = in_ptr[6];
    char_in[7] = in_ptr[7];

    in_ptr[0] = char_in[7];
    in_ptr[1] = char_in[6];
    in_ptr[2] = char_in[5];
    in_ptr[3] = char_in[4];
    in_ptr[4] = char_in[3];
    in_ptr[5] = char_in[2];
    in_ptr[6] = char_in[1];
    in_ptr[7] = char_in[0];
  }
}

void byte_swap64(void * ptr, int nmemb){
  int j;
  char char_in[8];
  char * in_ptr;
  double * double_ptr;

  double_ptr = (double *) ptr;
  for(j = 0; j < nmemb; j++, double_ptr++) {
    in_ptr = (char *) double_ptr;
    
    char_in[0] = in_ptr[0];
    char_in[1] = in_ptr[1];
    char_in[2] = in_ptr[2];
    char_in[3] = in_ptr[3];
    char_in[4] = in_ptr[4];
    char_in[5] = in_ptr[5];
    char_in[6] = in_ptr[6];
    char_in[7] = in_ptr[7];

    in_ptr[0] = char_in[7];
    in_ptr[1] = char_in[6];
    in_ptr[2] = char_in[5];
    in_ptr[3] = char_in[4];
    in_ptr[4] = char_in[3];
    in_ptr[5] = char_in[2];
    in_ptr[6] = char_in[1];
    in_ptr[7] = char_in[0];
  }
}

void byte_swap64_v2(double *ptr, unsigned int nmemb){
  unsigned int j;
  char char_in[8];
  char * in_ptr;

  for(j = 0; j < nmemb; j++) {
    in_ptr = (char *)(ptr+j);
    
    char_in[0] = in_ptr[0];
    char_in[1] = in_ptr[1];
    char_in[2] = in_ptr[2];
    char_in[3] = in_ptr[3];
    char_in[4] = in_ptr[4];
    char_in[5] = in_ptr[5];
    char_in[6] = in_ptr[6];
    char_in[7] = in_ptr[7];

    in_ptr[0] = char_in[7];
    in_ptr[1] = char_in[6];
    in_ptr[2] = char_in[5];
    in_ptr[3] = char_in[4];
    in_ptr[4] = char_in[3];
    in_ptr[5] = char_in[2];
    in_ptr[6] = char_in[1];
    in_ptr[7] = char_in[0];
  }
}

void * byte_swap_assign(void * out_ptr, void * in_ptr, int nmemb){
  int j;
  char * char_in_ptr, * char_out_ptr;
  double * double_in_ptr, * double_out_ptr;

  double_in_ptr = (double *) in_ptr;
  double_out_ptr = (double *) out_ptr;
  for(j = 0; j < nmemb; j++){
    char_in_ptr = (char *) double_in_ptr;
    char_out_ptr = (char *) double_out_ptr;
    
    char_out_ptr[7] = char_in_ptr[0];
    char_out_ptr[6] = char_in_ptr[1];
    char_out_ptr[5] = char_in_ptr[2];
    char_out_ptr[4] = char_in_ptr[3];
    char_out_ptr[3] = char_in_ptr[4];
    char_out_ptr[2] = char_in_ptr[5];
    char_out_ptr[1] = char_in_ptr[6];
    char_out_ptr[0] = char_in_ptr[7];
    double_in_ptr++;
    double_out_ptr++;
  }
  return(out_ptr);
}

//void * byte_swap_assign_singleprec(void * out_ptr, void * in_ptr, int nmemb){
//  int j;
//  char * char_in_ptr, * char_out_ptr;
//  float * float_in_ptr, * float_out_ptr;
//
//  float_in_ptr = (float *) in_ptr;
//  float_out_ptr = (float *) out_ptr;
//  for(j = 0; j < nmemb; j++){
//    char_in_ptr = (char *) float_in_ptr;
//    char_out_ptr = (char *) float_out_ptr;
//    
//    char_out_ptr[3] = char_in_ptr[0];
//    char_out_ptr[2] = char_in_ptr[1];
//    char_out_ptr[1] = char_in_ptr[2];
//    char_out_ptr[0] = char_in_ptr[3];
//    float_in_ptr++;
//    float_out_ptr++;
//  }
//  return(out_ptr);
//}

//void * single2double(void * out_ptr, void * in_ptr, int nmemb) {
//  int i;
//  float * float_ptr = (float*) in_ptr;
//  double * double_ptr = (double*) out_ptr;
//
//  for(i = 0; i < nmemb; i++) {
//    (*double_ptr) = (double) (*float_ptr);
//
//    float_ptr++;
//    double_ptr++;
//  }
//  return(out_ptr);
//}

//void * double2single(void * out_ptr, void * in_ptr, int nmemb) {
//  int i;
//  float * float_ptr = (float*) out_ptr;
//  double * double_ptr = (double*) in_ptr;
//
//  for(i = 0; i < nmemb; i++) {
//    (*float_ptr) = (float) (*double_ptr);
//
//    float_ptr++;
//    double_ptr++;
//  }
//  return(out_ptr);
//}

//inline void * byte_swap_assign_single2double(void * out_ptr, void * in_ptr, int nmemb){
//  int j;
//  char * char_in_ptr, * char_out_ptr;
//  double * double_out_ptr;
//  float * float_in_ptr;
//  float tmp;
//
//  float_in_ptr = (float *) in_ptr;
//  double_out_ptr = (double *) out_ptr;
//  char_out_ptr = (char *) &tmp;
//  for(j = 0; j < nmemb; j++){
//    char_in_ptr = (char *) float_in_ptr;
//    
//    char_out_ptr[3] = char_in_ptr[0];
//    char_out_ptr[2] = char_in_ptr[1];
//    char_out_ptr[1] = char_in_ptr[2];
//    char_out_ptr[0] = char_in_ptr[3];
//    (*double_out_ptr) = (double) tmp;
//    float_in_ptr++;
//    double_out_ptr++;
//  }
//  return(out_ptr);
//}

//void * byte_swap_assign_double2single(void * out_ptr, void * in_ptr, int nmemb){
//  int j;
//  char * char_in_ptr, * char_out_ptr;
//  double * double_in_ptr;
//  float * float_out_ptr;
//  float tmp;
//
//  float_out_ptr = (float *) out_ptr;
//  double_in_ptr = (double *) in_ptr;
//  char_in_ptr = (char *) &tmp;
//  for(j = 0; j < nmemb; j++){
//    tmp = (float) (*double_in_ptr);
//    char_out_ptr = (char*) float_out_ptr;
//
//    char_out_ptr[3] = char_in_ptr[0];
//    char_out_ptr[2] = char_in_ptr[1];
//    char_out_ptr[1] = char_in_ptr[2];
//    char_out_ptr[0] = char_in_ptr[3];
//
//    float_out_ptr++;
//    double_in_ptr++;
//  }
//  return(out_ptr);
//}

}
