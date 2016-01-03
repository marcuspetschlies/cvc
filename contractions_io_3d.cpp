/*****************************************************************************
 * contractions_io_3d.c
 *
 * PURPOSE
 * - functions for i/o of contractions, derived from propagator_io
 * - uses lime and the DML checksum
 * - specifically for 3-dim. fields
 * TODO:
 * CHANGES:
 *
 *****************************************************************************/


#define _FILE_OFFSET_BITS 64

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h> 
#include <sys/types.h>
#include <math.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#  include <unistd.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#  include "lime.h" 
#  ifdef HAVE_LIBLEMON
#    include "lemon.h"
#  endif

#ifdef __cplusplus
}
#endif


#include "cvc_complex.h"
#include "global.h"
#include "cvc_geometry.h"
#include "dml.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "contractions_io.h"
#include "cvc_utils.h"

namespace cvc {

/* write an N-comp. contraction to file */

/****************************************************
 * write_binary_contraction_data
 ****************************************************/
int write_binary_contraction_data_3d(double * const s, LimeWriter * limewriter, const int prec, const int N, DML_Checksum * ans) {
#ifdef HAVE_MPI
  fprintf(stderr, "[write_binary_contraction_data_3d] No mpi version.\n");
  return(1);
#else
  int x, y, z, i=0, mu, status=0;
  double *tmp;
  float  *tmp2;
  int proc_coords[4], tloc,xloc,yloc,zloc, proc_id;
  n_uint64_t bytes;
  DML_SiteRank rank;
  int words_bigendian = big_endian();
  unsigned int VOL3 = LX*LY*LZ;
  DML_checksum_init(ans);

  tmp = (double*)malloc(2*N*sizeof(double));
  tmp2 = (float*)malloc(2*N*sizeof(float));

  if(prec == 32) bytes = (n_uint64_t)2*N*sizeof(float);
  else bytes = (n_uint64_t)2*N*sizeof(double);

  if(g_cart_id==0) {
      for(x = 0; x < LX; x++) {
      for(y = 0; y < LY; y++) {
      for(z = 0; z < LZ; z++) {
        /* Rank should be computed by proc 0 only */
        rank = (DML_SiteRank) (( x * LY + y)*LZ + z);
        for(mu=0; mu<N; mu++) {
          i = _GWI(mu, g_ipt[0][x][y][z], VOL3);
          if(!words_bigendian) {
            if(prec == 32) {
              byte_swap_assign_double2single( (tmp2+2*mu), (s + i), 2);
            } else {
              byte_swap_assign( (tmp+2*mu), (s + i), 2);
            }
          } else {
            if(prec == 32) {
              double2single((float*)(tmp2+2*mu), (s + i), 2);
            } else {
              tmp[2*mu  ] = s[i  ];
              tmp[2*mu+1] = s[i+1];
            }
          }
        }
        if(prec == 32) {
          DML_checksum_accum(ans,rank,(char *) tmp2,2*N*sizeof(float));
          status = limeWriteRecordData((void*)tmp2, &bytes, limewriter);
        }
        else {
          status = limeWriteRecordData((void*)tmp, &bytes, limewriter);
          DML_checksum_accum(ans,rank,(char *) tmp, 2*N*sizeof(double));
        }
      }}}
  }
  free(tmp2);
  free(tmp);
  return(0);
#endif
}

/************************************************************
 * read_binary_contraction_data
 ************************************************************/

int read_binary_contraction_data_3d(double * const s, LimeReader * limereader, const int prec, const int N, DML_Checksum *ans) {
#ifdef HAVE_MPI
  fprintf(stderr, "[read_binary_contraction_data_3d] No mpi version.\n");
  return(1);
#else
  int status=0, mu;
  n_uint64_t bytes, ix;
  double *tmp;
  DML_SiteRank rank;
  float *tmp2;
  int x, y, z;
  int words_bigendian = big_endian();
  unsigned int VOL3 = LX * LY * LZ;

  DML_checksum_init(ans);
  rank = (DML_SiteRank) 0;
 
  if( (tmp = (double*)malloc(2*N*sizeof(double))) == (double*)NULL ) {
    exit(500);
  }
  if( (tmp2 = (float*)malloc(2*N*sizeof(float))) == (float*)NULL ) {
    exit(501);
  }
 
 
  if(prec == 32) bytes = 2*N*sizeof(float);
  else bytes = 2*N*sizeof(double);
  for(x = 0; x < LX; x++){
  for(y = 0; y < LY; y++){
  for(z = 0; z < LZ; z++){
    ix = g_ipt[0][x][y][z];
    rank = (DML_SiteRank) (( LXstart + x)*(LY*g_nproc_y) + LYstart + y)*LZ + z;
    if(prec == 32) {
      status = limeReaderReadData(tmp2, &bytes, limereader);
      DML_checksum_accum(ans,rank,(char *) tmp2, bytes);	    
    }
    else {
      status = limeReaderReadData(tmp, &bytes, limereader);
      DML_checksum_accum(ans,rank,(char *) tmp, bytes);
    }
 
    for(mu=0; mu<N; mu++) {
      if(!words_bigendian) {
        if(prec == 32) {
          byte_swap_assign_single2double(s + _GWI(mu,ix,VOL3), (float*)(tmp2+2*mu), 2);
        } else {
          byte_swap_assign(s + _GWI(mu,ix,VOL3), (float*)(tmp+2*mu), 2);
        }
      } else {  // words_bigendian true
        if(prec == 32) {
          single2double(s + _GWI(mu,ix,VOL3), (float*)(tmp2+2*mu), 2);
        }
        else {
          s[_GWI(mu, ix,VOL3)  ] = tmp[2*mu  ];
          s[_GWI(mu, ix,VOL3)+1] = tmp[2*mu+1];
        }
      }
    }

    if(status < 0 && status != LIME_EOR) {
      return(-1);
    }
  }}}
  if(g_cart_id == 0) printf("\n# [read_binary_contraction_data] The final checksum is %#lx %#lx\n", (*ans).suma, (*ans).sumb);

  free(tmp2); free(tmp);
  return(0);
#endif
}

/***********************************************************
 * write_lime_contraction
 ***********************************************************/
int write_lime_contraction_3d(double * const s, char * filename, const int prec, const int N, char * type, const int gid, const int sid) {
#ifdef HAVE_MPI
  fprintf(stderr, "[write_lime_contraction_3d] No mpi version.\n");
  return(1);
#else

  FILE * ofs = NULL;
  LimeWriter * limewriter = NULL;
  LimeRecordHeader * limeheader = NULL;
  int status = 0;
  int ME_flag=0, MB_flag=0;
  n_uint64_t bytes;
  DML_Checksum checksum;

  write_contraction_format(filename, prec, N, type, gid, sid);

  if(g_cart_id==0) {
    ofs = fopen(filename, "a");
    if(ofs == (FILE*)NULL) {
      fprintf(stderr, "Could not open file %s for writing!\n Aborting...\n", filename);
      exit(500);
    }
  
    limewriter = limeCreateWriter( ofs );
    if(limewriter == (LimeWriter*)NULL) {
      fprintf(stderr, "LIME error in file %s for writing!\n Aborting...\n", filename);
      exit(500);
    }

    bytes = LX*g_nproc_x*LY*g_nproc_y*LZ*(n_uint64_t)2*N*sizeof(double)*prec/64;
    MB_flag=0; ME_flag=1;
    limeheader = limeCreateHeader(MB_flag, ME_flag, "scidac-binary-data", bytes);
    status = limeWriteRecordHeader( limeheader, limewriter);
    if(status < 0 ) {
      fprintf(stderr, "LIME write header (scidac-binary-data) error %d\n", status);
      exit(500);
    }
    limeDestroyHeader( limeheader );
  }
  
  status = write_binary_contraction_data_3d(s, limewriter, prec, N, &checksum);
  if(g_cart_id==0) {
    printf("# Final check sum is (%#lx  %#lx)\n", checksum.suma, checksum.sumb);
    if(ferror(ofs)) {
      fprintf(stderr, "Warning! Error while writing to file %s \n", filename);
    }
    limeDestroyWriter( limewriter );
    fflush(ofs);
    fclose(ofs);
  }
  write_checksum(filename, &checksum);
  return(0);
#endif
}

/***********************************************************
 * read_lime_contraction
 ***********************************************************/

int read_lime_contraction_3d(double * const s, char * filename, const int N, const int position) {
#ifdef HAVE_MPI
  fprintf(stderr, "[read_lime_contraction_3d] No mpi version.\n");
  return(1);
#else
  FILE *ifs=(FILE*)NULL;
  int status=0, getpos=-1;
  n_uint64_t bytes;
  char * header_type;
  LimeReader * limereader;
  int prec = 32;
  DML_Checksum checksum;

  if((ifs = fopen(filename, "r")) == (FILE*)NULL) {
    if(g_proc_id == 0) {
      fprintf(stderr, "Error opening file %s\n", filename);
    }
    return(106);
  }

  limereader = limeCreateReader( ifs );
  if( limereader == (LimeReader *)NULL ) {
    if(g_proc_id == 0) {
      fprintf(stderr, "Unable to open LimeReader\n");
    }
    return(-1);
  }
  while( (status = limeReaderNextRecord(limereader)) != LIME_EOF ) {
    if(status != LIME_SUCCESS ) {
      fprintf(stderr, "limeReaderNextRecord returned error with status = %d!\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = limeReaderType(limereader);
    if(strcmp("scidac-binary-data",header_type) == 0) getpos++;
    if(getpos == position) break;
  }
  if(status == LIME_EOF) {
    if(g_proc_id == 0) {
      fprintf(stderr, "no scidac-binary-data record found in file %s\n",filename);
    }
    limeDestroyReader(limereader);
    fclose(ifs);
    if(g_proc_id == 0) {
      fprintf(stderr, "try to read in non-lime format\n");
    }
    return(read_contraction(s, NULL, filename, N));
  }
  bytes = limeReaderBytes(limereader);
  if((int)bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*LZ*2*N*sizeof(double)) prec = 64;
  else if((int)bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*LZ*2*N*sizeof(float)) prec = 32;
  else {
    if(g_proc_id == 0) {
      fprintf(stderr, "wrong length in contraction: bytes = %lu, not %d. Aborting read!\n", bytes, (LX*g_nproc_x)*(LY*g_nproc_y)*LZ*2*N*(int)sizeof(float));
    }
    return(-1);
  }
  if(g_proc_id == 0) {
    printf("# %d Bit precision read\n", prec);
  }

  status = read_binary_contraction_data_3d(s, limereader, prec, N, &checksum);

  if(g_proc_id == 0) {
    printf("# checksum for contractions in file %s position %d is %#x %#x\n",
           filename, position, checksum.suma, checksum.sumb);
  }

  if(status < 0) {
    fprintf(stderr, "LIME read error occured with status = %d while reading file %s!\n Aborting...\n",
            status, filename);
    exit(500);
  }

  limeDestroyReader(limereader);
  fclose(ifs);
  return(0);
#endif
}

}
