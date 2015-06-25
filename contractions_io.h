#ifndef _CONTRACTION_IO_H
#define _CONTRACTION_IO_H

#include "dml.h"
#include "lime.h"
#ifdef HAVE_LIBLEMON
#  include "lemon.h"
#endif

namespace cvc {

int write_contraction_format(char * filename, const int prec, const int N, char * type, const int gid, const int sid);


int write_lime_contraction(double * const s, char * filename, const int prec, const int N, char * type, const int gid, const int sid);
#ifdef HAVE_LIBLEMON
int write_binary_contraction_data(double * const s, LemonWriter * writer, const int prec, const int N, DML_Checksum * ans);
#else
int write_binary_contraction_data(double * const s, LimeWriter * limewriter, const int prec, const int N, DML_Checksum * ans);
#endif

int read_lime_contraction(double * const s, char * filename, const int N, const int position);

int read_binary_contraction_data(double * const s, LimeReader * limereader,
  const int prec, const int N, DML_Checksum * ans);


/* different versions: */
int write_lime_contraction_v2(double * const s, char * filename, const int prec, const int N, char * type, const int gid, const int sid);
#ifdef HAVE_LIBLEMON
int write_binary_contraction_data_v2(double * const s, LemonWriter * writer, const int prec, const int N, DML_Checksum * ans);
#else
int write_binary_contraction_data_v2(double * const s, LimeWriter * limewriter,
  const int prec, const int N, DML_Checksum * ans);
#endif

int read_binary_contraction_data_v2(double * const s, LimeReader * limereader, const int prec, const int N, DML_Checksum * ans);

int read_lime_contraction_v2(double * const s, char * filename, const int N, const int position);


/* 3-dim. versions without MPI support */
int write_binary_contraction_data_3d(double * const s, LimeWriter * limewriter, const int prec, const int N, DML_Checksum * ans);
int read_binary_contraction_data_3d(double * const s, LimeReader * limereader, const int prec, const int N, DML_Checksum *ans);
int write_lime_contraction_3d(double * const s, char * filename, const int prec, const int N, char * type, const int gid, const int sid);
int read_lime_contraction_3d(double * const s, char * filename, const int N, const int position);

/***********************************************************
 * write_lime_contraction timeslice-wise
 ***********************************************************/
int write_binary_contraction_data_timeslice(double * const s, LimeWriter * limewriter, const int prec, const int N, DML_Checksum * ans, int timeslice);
int write_lime_contraction_timeslice(double * const s, char * filename, const int prec, const int N, char * type, const int gid, const int sid, DML_Checksum *checksum, int timeslice);


int read_lime_contraction_timeslice(double * const s, char * filename, const int N, const int position, DML_Checksum*checksum, int timeslice);
int read_binary_contraction_data_timeslice(double * const s, LimeReader * limereader, const int prec, const int N, DML_Checksum *ans, int timeslice);
}
#endif
