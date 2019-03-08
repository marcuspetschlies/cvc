#ifndef _CONTRACTION_IO_H
#define _CONTRACTION_IO_H

#include "dml.h"

#ifdef __cplusplus
extern "C"
{
#endif

#include "lime.h"
#ifdef HAVE_LIBLEMON
#include "lemon.h"
#endif

#ifdef __cplusplus
}
#endif

namespace cvc {

int write_contraction_format(char * filename, const int prec, const int N, char * type, const int gid, const int append);


int write_lime_contraction(double * const s, char * filename, const int prec, const int N, char * type, const int gid, const int append);
#ifdef HAVE_LIBLEMON
int write_binary_contraction_data(double * const s, LemonWriter * writer, const int prec, const int N, DML_Checksum * ans);
#else
int write_binary_contraction_data(double * const s, LimeWriter * limewriter, const int prec, const int N, DML_Checksum * ans);
#endif

int read_lime_contraction(double * const s, char * filename, const int N, const int position);

int read_binary_contraction_data(double * const s, LimeReader * limereader,
  const int prec, const int N, DML_Checksum * ans);

#if HAVE_HDF5
int read_from_h5_file ( void * const buffer, void * file, char*tag,  int const io_proc );
#endif

}
#endif
