#ifndef _DERIVED_QUANTITIES_H
#define _DERIVED_QUANTITIES_H

#define _POW2(_a) (      (_a) * (_a) )
#define _POW3(_a) ( _POW2(_a) * (_a) )
#define _POW4(_a) ( _POW3(_a) * (_a) )
#define _POW5(_a) ( _POW4(_a) * (_a) )
#define _POW6(_a) ( _POW5(_a) * (_a) )

namespace cvc {

int ratio_1_1 ( void * param , void * v_in, double * v_out);

int dratio_1_1 ( void *param , void *v_in, double *v_out);

int ratio_1_1_sub ( void * param , void * v_in, double * v_out);

int dratio_1_1_sub ( void *param , void *v_in, double *v_out);

int log_ratio_1_1 ( void * param , void * v_in, double * v_out);

int dlog_ratio_1_1 ( void * param , void * v_in, double * v_out);

int acosh_ratio ( void * param , void * v_in, double * v_out);

int dacosh_ratio ( void * param , void * v_in, double * v_out);

int acosh_ratio_deriv  ( void * param , void * v_in, double * v_out);
int dacosh_ratio_deriv ( void * param , void * v_in, double * v_out);

int a_mi_b_ti_c ( void * param , void * v_in, double * v_out);

int da_mi_b_ti_c ( void * param , void * v_in, double * v_out);

int  cumulant_1  ( void * param , void * v_in, double * v_out);
int dcumulant_1  ( void * param , void * v_in, double * v_out);
int  cumulant_2  ( void * param , void * v_in, double * v_out);
int dcumulant_2  ( void * param , void * v_in, double * v_out);
int  cumulant_3  ( void * param , void * v_in, double * v_out);
int dcumulant_3  ( void * param , void * v_in, double * v_out);
int  cumulant_4  ( void * param , void * v_in, double * v_out);
int dcumulant_4  ( void * param , void * v_in, double * v_out);

int ratio_1_2_mi_3 ( void * param , void * v_in, double * v_out);
int dratio_1_2_mi_3 ( void * param , void * v_in, double * v_out);

int sqrt_ab_over_cd ( void * param , void * v_in, double * v_out);
int dsqrt_ab_over_cd ( void * param , void * v_in, double * v_out);


}

#endif
