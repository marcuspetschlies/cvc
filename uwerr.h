#ifndef _UWERR_H
#define _UWERR_H

#include <math.h>
#include "global.h"

#define MAX_NO_REPLICA 100
#define _SQR(_a)   ( (_a) * (_a) )
#define _num_bin(_n) (1 + (int)rint( log( (double)(_n) ) / log(2.0) ))
#define TINY (1e-15)

typedef int (*dquant)(void*, void*, double*);
 
typedef struct {
  double value;
  double dvalue;
  double ddvalue;
  double tauint;
  double dtauint;
  size_t Wopt;
  double *tau;
  double *gamma;
  size_t nalpha;
  size_t nreplica;
  size_t n_r[100];
  size_t Wmax;
  size_t npara;
  void *para;
  char obsname[400];
  int write_flag;
  dquant func;
  dquant dfunc;
  double s_tau;
  double Qval;
  double *p_r;
  double p_r_mean;
  double p_r_var;
  double *bins;
  double *binbd;
  size_t ipo;
} uwerr;

extern int uwerr_verbose;

static inline void ARITHMEAN ( double * const data_pointer, size_t const data_stride, size_t const data_number, double * const res_pointer) {
  size_t i;
  *res_pointer = 0.0;
  for(i=0; i<data_number; i++) {
    *res_pointer += *(data_pointer + i*data_stride);
  }
  *res_pointer /= (double)data_number;
}

static inline void WEIGHEDMEAN ( double * const data_pointer, size_t const data_stride, size_t const data_number, double * const res_pointer,
  size_t * const weights_pointer) {
  size_t i;
  double n=0.0;
  *res_pointer = 0.0;
  for(i=0; i<data_number; i++) {
    *res_pointer += *(data_pointer + i*data_stride) * (double)(*(weights_pointer + i));
    n += (double)(*(weights_pointer + i));
  }
  *res_pointer /= n;
}

static inline void STDDEV ( double * const data_pointer, size_t const data_stride, size_t const data_number, double * const res_pointer) {
  size_t j;
  double mean=0.0;
  *res_pointer = 0.0;
  ARITHMEAN(data_pointer, data_stride, data_number, &mean);
  for(j=0; j<data_number; j++) {
    *res_pointer += _SQR(*(data_pointer + j*data_stride) - mean);
  }
  *res_pointer = sqrt( *res_pointer / (double)(data_number-1) / (double)(data_number) );
}

static inline void VARIANCE ( double * const data_pointer, size_t const data_stride, size_t const data_number, double * const res_pointer) {
  size_t j;
  double mean=0.0, d;
  *res_pointer = 0.0;
  ARITHMEAN(data_pointer, data_stride, data_number, &mean);
/*  fprintf(stdout, "# mean = %25.16e\n", mean); */
  for(j=0; j<data_number; j++) {
    d = *(data_pointer + j*data_stride) - mean;
    *res_pointer += d * d;
  }
  *res_pointer /= (double)(data_number - 1);
}

static inline void VARIANCE_FIXED_MEAN ( double * const data_pointer, double const mean, size_t const data_stride, size_t const data_number,
    double * const res_pointer) {
  size_t j;
  double d;
  *res_pointer = 0.0;
  for(j=0; j<data_number; j++) {
    d = *(data_pointer + j*data_stride) - mean;
    *res_pointer += d * d;
  }
  *res_pointer /= (double)(data_number);
}

static inline void SUM ( double * const data_pointer, size_t const data_stride, size_t const data_number, double * const res_pointer) {
  size_t i;
  *res_pointer = 0.0;
  for(i=0; i<data_number; i++) {
    *res_pointer += *(data_pointer + i*data_stride);
  }
}

static inline void ASSIGN ( double * const data_pointer1, double * const data_pointer2, size_t const data_number) {
  memcpy((void*)data_pointer1, (void*)data_pointer2, data_number*sizeof(double));
}

static inline void COPY_STRIDED ( double * const data_pointer1, double * const data_pointer2, size_t const data_stride, size_t const data_number) {
  size_t i;
  for(i=0; i<data_number; i++) {
    data_pointer1[i] =  data_pointer2[i*data_stride];
  }
}

static inline void ADD ( double * const data_pointer1, double * const data_pointer2, size_t const data_number) {
  size_t i;
  for(i=0; i<data_number; i++) {
    *(data_pointer1 + i) += *(data_pointer2 + i);
  }
}

static inline void ADD_ASSIGN ( double * const data_pointer1, double * const data_pointer2, double * const data_pointer3,
                      size_t const data_number) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( size_t i=0; i<data_number; i++) {
    *(data_pointer1 + i) = *(data_pointer2 + i) + *(data_pointer3 + i);
  }
}

static inline void SUB ( double * const data_pointer1, double * const data_pointer2, size_t const data_number) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( size_t i=0; i<data_number; i++) {
    *(data_pointer1 + i) -= *(data_pointer2 + i);
  }
}
static inline void SUB_ASSIGN ( double * const data_pointer1, double * const data_pointer2, double * const data_pointer3,
                      size_t const data_number) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( size_t i=0; i<data_number; i++) {
    *(data_pointer1 + i) = *(data_pointer2 + i) - *(data_pointer3 + i);
  }
}

static inline void PROD ( double * const data_pointer1, double * const data_pointer2, size_t const data_number) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( size_t i=0; i<data_number; i++) {
    *(data_pointer1 + i) *= *(data_pointer2 + i);
  }
}

static inline void PROD_ASSIGN ( double * const data_pointer1, double * const data_pointer2, double * const data_pointer3, size_t const data_number) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( size_t i=0; i<data_number; i++) {
    *(data_pointer1 + i) = *(data_pointer2 + i) * *(data_pointer3 + i);
  }
}

static inline void NORM2 ( double * const data_pointer1, size_t const data_number, double * const res_pointer) {
  size_t i;
  double d;
  *res_pointer = 0.0;
  for(i=0; i<data_number; i++) {
    d = *(data_pointer1+i);
    *res_pointer += d * d;
  }
}

static inline void DIFFNORM2 ( double * const data_pointer1, double * const data_pointer2, size_t const data_number, double * const res_pointer) {
  size_t i;
  double d;
  *res_pointer = 0.0;
  for(i=0; i<data_number; i++) {
    d = *(data_pointer1+i) - *(data_pointer2+i);
    *res_pointer += d * d;
  }
}

static inline void SET_TO ( double * const data_pointer, size_t const data_number, double const value) {
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
  for ( size_t i=0; i<data_number; i++) {
    *(data_pointer + i) = value;
  }
}

static inline void MIN_UINT ( size_t * const int_pointer, size_t const int_number, size_t * const res_pointer) {
              size_t i;
              *res_pointer = *int_pointer;
              for(i=1; i<int_number; i++) {
                 if(*(int_pointer+i) < *res_pointer) {
		    *res_pointer = *(int_pointer+i);
                 }
              }
}

static inline void ABS_MAX_DBL ( double * const dbl_pointer, size_t const dbl_number, double * const res_pointer) {   \
              size_t i;
              *res_pointer = fabs(*(dbl_pointer));
              for(i=1; i<dbl_number; i++) {
                if(*res_pointer < fabs(*(dbl_pointer + i))) {
		  *res_pointer = fabs(*(dbl_pointer + i));
                }
              }
}

int uwerr_init ( uwerr * const u );

int uwerr_calloc ( uwerr * const u );

int uwerr_free ( uwerr * const u );

int uwerr_printf ( uwerr const u );

int uwerr_analysis (double * const data, uwerr * const u);

#endif
