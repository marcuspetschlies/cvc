/*************************************************************
 * distillation_utils.h
 *************************************************************/
#ifndef _DISTILLATION_PREPARE_SOURCE_H
#define _DISTILLATION_PREPARE_SOURCE_H

namespace cvc {

int distillation_prepare_source ( double *s, double**v, int ievec, int ispin, int timeslice );

int distillation_reduce_propagator ( double ***p, double *s, double ***v , int numV );

int colorvector_field_from_spinor_field ( double *r, double *s  );

int distillation_write_perambulator (double ****p, int numV, struct AffWriter_s*affw, char *tag, int io_proc );

void colorvector_field_norm_diff_timeslice (double*d, double *r, double *s, int it, unsigned int N);

int read_eigensystem_timeslice ( double **v, int numV, char*filename);

int write_eigensystem_timeslice ( double **v, int numV, char*filename);

}  /* end of namespace cvc */
#endif
