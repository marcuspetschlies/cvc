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

#ifdef HAVE_LHPC_AFF
int write_aff_contraction ( void * const contr, void * const awriter, void * const afilename, char * tag, unsigned int const nc, const char * data_type);
#endif

#if HAVE_HDF5
int read_from_h5_file ( void * const buffer, void * file, char*tag,  const char * data_type, int const io_proc );

/* int write_h5_contraction ( void * const contr, void * const awriter, void * const afilename, char * tag, unsigned int const nc, const char * data_type ); */
int write_h5_contraction ( void * const contr, void * const awriter, void * const afilename, char * tag, unsigned int const nc, const char * data_type, int const ncdim, const int * const cdim );

int write_h5_attribute ( const char * filename, const char * attr_name, const char * info );

#endif

int write_vdag_gloc_v_to_file ( double _Complex ***** vv, int const nv, int const momentum_number, int  (* const momentum_list)[3] , int const gamma_id_number, int * const gamma_id_list, void * writer, void * file, char * tag , int const io_proc );

}
#endif
