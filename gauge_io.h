#ifndef _GAUGE_IO_H
#define _GAUGE_IO_H


#include"dml.h"
#include "lime.h"

namespace cvc {
int write_lime_gauge_field(char * filename, const double plaq, const int counter, const int prec);
int write_xlf_info(const double plaq, const int counter, char * filename, const int append, char * data_buf);
int write_ildg_format_xml(char *filename, LimeWriter * limewriter, const int prec);
n_uint64_t file_size(FILE *fp);
int read_nersc_gauge_field(double*s, char*filename, double *plaq);
int read_nersc_gauge_field_timeslice(double*s, char*filename, int timeslice, uint32_t *checksum);
int read_nersc_gauge_binary_data_3col(FILE*ifs, double*s, DML_Checksum*ans);
int read_nersc_gauge_field_3x3(double*s, char*filename, double *plaq);

}
#endif
