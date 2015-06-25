/* $Id: io.h,v 1.2 2006/04/18 15:29:27 urbach Exp $ */
#ifndef _IO_H
#define _IO_H

#include "dml.h"
#include "lime.h"

namespace cvc {

int read_lime_gauge_field_doubleprec(const char * filename);
int read_lime_gauge_field_singleprec(const char * filename);
int read_lime_gauge_field_doubleprec_timeslice(double *gfield, const char * filename, const int timeslice, DML_Checksum *checksum);
int read_lime_gauge_field_singleprec_timeslice(double *gfield, const char * filename, const int timeslice, DML_Checksum *checksum);
int read_ildg_nersc_gauge_field(const double * gauge, const char * filename);

}
#endif
