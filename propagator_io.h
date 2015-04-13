/* $Id: propagator_io.h,v 1.2 2007/11/24 14:37:24 urbach Exp $ */

#ifndef _PROPAGATOR_IO_H
#define _PROPAGATOR_IO_H

#include "dml.h"
#include "lime.h"

int write_propagator(double * const s, char * filename, const int append, const int prec);

int write_propagator_format(char * filename, const int prec, const int no_flavours);

int write_propagator_type(const int type, char * filename);

int get_propagator_type(char * filename);

int write_lime_spinor(double * const s, char * filename, const int append, const int prec);

int write_checksum(char * filename, DML_Checksum *checksum); 

/* int write_binary_spinor_data(double * const s, LimeWriter * limewriter,
  const int prec, DML_Checksum * ans); */

int read_lime_spinor(double * const s, char * filename, const int position);

int read_cmi(double *v, const char * filename);
int write_binary_spinor_data_timeslice(double * const s, LimeWriter * limewriter,
  const int prec, int timeslice, DML_Checksum * ans);

int write_lime_spinor_timeslice(double * const s, char * filename,
   const int prec, int timeslice, DML_Checksum *checksum);
int prepare_propagator(int timeslice, int iread, int is_mms, int no_mass, double sign, double mass, int isave, double *work, int pos);
int prepare_propagator2(int *source_coords, int iread, int sign, double *work, int pos, int format);
int rotate_propagator_ETMC_UKQCD (double *spinor, long unsigned int V);

#endif
