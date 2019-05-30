#ifndef _DERIVED_QUANTITIES_H
#define _DERIVED_QUANTITIES_H

namespace cvc {

int ratio_1_1 ( void * param , void * v_in, double * v_out);

int dratio_1_1 ( void *param , void *v_in, double *v_out);

int ratio_1_1_sub ( void * param , void * v_in, double * v_out);

int dratio_1_1_sub ( void *param , void *v_in, double *v_out);

int log_ratio_1_1 ( void * param , void * v_in, double * v_out);

int dlog_ratio_1_1 ( void * param , void * v_in, double * v_out);

int acosh_ratio ( void * param , void * v_in, double * v_out);

int dacosh_ratio ( void * param , void * v_in, double * v_out);

int acosh_ratio_deriv ( void * param , void * v_in, double * v_out);

}

#endif
