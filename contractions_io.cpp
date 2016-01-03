/*****************************************************************************
 * contractions_io.c
 *
 * PURPOSE
 * - functions for i/o of contractions, derived from propagator_io
 * - uses lime and the DML checksum
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
#include "lime.h" 
#ifdef HAVE_LIBLEMON
#include "lemon.h"
#endif

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
#ifndef HAVE_LIBLEMON
int write_binary_contraction_data(double * const s, LimeWriter * limewriter, const int prec, const int N, DML_Checksum * ans) {

  fprintf(stderr, "[] Error, lime version not implemented yet\n");
  return(-1);
}

#else  // HAVE_LIBLEMON
int write_binary_contraction_data(double * const s, LemonWriter * writer, const int prec, const int N, DML_Checksum * ans) {

  int x, y, z, t, ix=0, mu, status=0;
  n_uint64_t bytes;
  DML_SiteRank rank;
  double *buffer;
  int words_bigendian = big_endian();
  int latticeSize[] = {T_global, LX_global, LY_global, LZ_global};
  int scidacMapping[] = {0, 1, 2, 3};
  unsigned long bufoffset = 0;
  char *filebuffer = NULL;
  DML_checksum_init(ans);

  if(g_cart_id == 0) fprintf(stdout, "\n# [] words_bigendian = %d\n", words_bigendian);

  if(prec == 32) bytes = (n_uint64_t)2*N*sizeof(float);   // single precision 
  else           bytes = (n_uint64_t)2*N*sizeof(double);  // double precision

  if((void*)(filebuffer = (char*)malloc(VOLUME * bytes)) == NULL) {
    fprintf (stderr, "\n[write_binary_contraction_data] malloc error in write_binary_contraction_datad\n");
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 114);
    MPI_Finalize();
    exit(114);
  }

  if((void*)(buffer = (double*)malloc(2*N*sizeof(double))) == NULL) {
    fprintf (stderr, "\n[write_binary_contraction_data] malloc error\n");
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 115);
    MPI_Finalize();
    exit(115);
  }

  for(t = 0; t < T; t++) {
  for(x = 0; x < LX; x++) {
  for(y = 0; y < LY; y++) {
  for(z = 0; z < LZ; z++) {
    rank = (DML_SiteRank) ((((Tstart + t)*LX_global + x+LXstart)*LY_global + LYstart + y)*((DML_SiteRank)LZ_global) + LZstart + z);
    ix  = g_ipt[t][x][y][z];
    for(mu=0;mu<N;mu++) {
      buffer[2*mu  ] = s[ _GWI(mu,ix,VOLUME)   ];
      buffer[2*mu+1] = s[ _GWI(mu,ix,VOLUME)+1 ];
    }

    if(!words_bigendian) {
      if(prec == 32) {
        byte_swap_assign_double2single((float*)(filebuffer + bufoffset), buffer, 2*N);
      } else {
        byte_swap_assign(filebuffer + bufoffset, buffer, 2*N);
      }
    } else {  // words_bigendian true
      if(prec == 32) {
        double2single((float*)(filebuffer + bufoffset), buffer, 2*N);
      } else {
        memcpy(filebuffer + bufoffset, buffer, bytes);
      }
    }
    DML_checksum_accum(ans, rank, (char*) filebuffer + bufoffset, bytes);
    bufoffset += bytes;
  }}}}

  status = lemonWriteLatticeParallelMapped(writer, filebuffer, bytes, latticeSize, scidacMapping);

  if (status != LEMON_SUCCESS) {
    free(filebuffer);
    fprintf(stderr, "\n[write_binary_contraction_data] LEMON write error occurred with status = %d\n", status);
    return(-2);
  }

  lemonWriterCloseRecord(writer);
  DML_global_xor(&(ans->suma));
  DML_global_xor(&(ans->sumb));
  free(filebuffer);
  free(buffer);
  return(0);
}
#endif  // HAVE_LIBLEMON



/************************************************************
 * read_binary_contraction_data
 ************************************************************/

int read_binary_contraction_data(double * const s, LimeReader * limereader, const int prec, const int N, DML_Checksum *ans) {

  int status=0, mu;
  n_uint64_t bytes, ix;
  double *tmp;
  DML_SiteRank rank;
  float *tmp2;
  int t, x, y, z;
  int words_bigendian = big_endian();

  DML_checksum_init(ans);
  rank = (DML_SiteRank) 0;
 
  if( (tmp = (double*)malloc(2*N*sizeof(double))) == (double*)NULL ) {
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(500);
  }
  if( (tmp2 = (float*)malloc(2*N*sizeof(float))) == (float*)NULL ) {
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(501);
  }
 
 
  if(prec == 32) bytes = 2*N*sizeof(float);
  else bytes = 2*N*sizeof(double);
#ifdef HAVE_MPI
  limeReaderSeek(limereader,(n_uint64_t) (Tstart*(LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z) + LXstart*(LY*g_nproc_y)*(LZ*g_nproc_z) + LYstart*(LZ*g_nproc_z) + LZstart)*bytes, SEEK_SET);
#endif
  for(t = 0; t < T; t++){
  for(x = 0; x < LX; x++){
  for(y = 0; y < LY; y++){
  for(z = 0; z < LZ; z++){
    ix = g_ipt[t][x][y][z];
    rank = (DML_SiteRank) (((t+Tstart)*(LX*g_nproc_x) + LXstart + x)*(LY*g_nproc_y) + LYstart + y)*(LZ*g_nproc_z) + LZstart + z;
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
          byte_swap_assign_single2double(s + _GWI(mu,ix,VOLUME), (float*)(tmp2+2*mu), 2);
        } else {
          byte_swap_assign( (s + _GWI(mu,ix,VOLUME)), (tmp+2*mu), 2);
        }
      } else {  // words_bigendian true
        if(prec == 32) {
          single2double( (s + _GWI(mu,ix,VOLUME)), (float*)(tmp2+2*mu), 2);
        }
        else {
          s[_GWI(mu, ix,VOLUME)  ] = tmp[2*mu  ];
          s[_GWI(mu, ix,VOLUME)+1] = tmp[2*mu+1];
        }
      }
    }

    if(status < 0 && status != LIME_EOR) {
      return(-1);
    }
  }}}}
#ifdef HAVE_MPI
  DML_checksum_combine(ans);
#endif
  if(g_cart_id == 0) printf("\n# [read_binary_contraction_data] The final checksum is %#x %#x\n", (*ans).suma, (*ans).sumb);

  free(tmp2); free(tmp);
  return(0);
}

/**************************************************************
 * write_contraction_format
 **************************************************************/

int write_contraction_format(char * filename, const int prec, const int N, char * type, const int gid, const int sid) {
  FILE * ofs = NULL;
  LimeWriter * limewriter = NULL;
  LimeRecordHeader * limeheader = NULL;
  int status = 0;
  int ME_flag=0, MB_flag=1;
  char message[500];
  n_uint64_t bytes;

  if(g_cart_id==0) {
    ofs = fopen(filename, "w");
  
    if(ofs == (FILE*)NULL) {
      fprintf(stderr, "Could not open file %s for writing!\n Aborting...\n", filename);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }
    limewriter = limeCreateWriter( ofs );
    if(limewriter == (LimeWriter*)NULL) {
      fprintf(stderr, "LIME error in file %s for writing!\n Aborting...\n", filename);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }
  
    sprintf(message, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<cvcFormat>\n<type>%s</type>\n<precision>%d</precision>\n<components>%d</components>\n<lx>%d</lx>\n<ly>%d</ly>\n<lz>%d</lz>\n<lt>%d</lt>\n<nconf>%d</nconf>\n<source>%d</source>\n</cvcFormat>", type, prec, N, LX*g_nproc_x, LY*g_nproc_y, LZ*g_nproc_z, T_global, gid, sid);
    bytes = strlen( message );
    limeheader = limeCreateHeader(MB_flag, ME_flag, "cvc-contraction-format", bytes);
    status = limeWriteRecordHeader( limeheader, limewriter);
    if(status < 0 ) {
      fprintf(stderr, "LIME write header error %d\n", status);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }
    limeDestroyHeader( limeheader );
    limeWriteRecordData(message, &bytes, limewriter);
  
    limeDestroyWriter( limewriter );
    fflush(ofs);
    fclose(ofs);
  }
  return(0);
}


/***********************************************************
 * write_lime_contraction
 ***********************************************************/
#ifndef HAVE_LIBLEMON
int write_lime_contraction(double * const s, char * filename, const int prec, const int N, char * type, const int gid, const int sid) {

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
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }
  
    limewriter = limeCreateWriter( ofs );
    if(limewriter == (LimeWriter*)NULL) {
      fprintf(stderr, "LIME error in file %s for writing!\n Aborting...\n", filename);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }

    bytes = LX*g_nproc_x*LY*g_nproc_y*LZ*g_nproc_z*T_global*(n_uint64_t)2*N*sizeof(double)*prec/64;
    MB_flag=0; ME_flag=1;
    limeheader = limeCreateHeader(MB_flag, ME_flag, "scidac-binary-data", bytes);
    status = limeWriteRecordHeader( limeheader, limewriter);
    if(status < 0 ) {
      fprintf(stderr, "LIME write header (scidac-binary-data) error %d\n", status);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }
    limeDestroyHeader( limeheader );
  }
  
  status = write_binary_contraction_data(s, limewriter, prec, N, &checksum);
  if(status < 0 ) {
    fprintf(stderr, "[write_lime_contraction] Error from write_binary_contraction_data, status was %d\n", status);
    exit(502);
  }
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
}
#else  // HAVE_LIBLEMON
int write_lime_contraction(double * const s, char * filename, const int prec, const int N, char * type, const int gid, const int sid) {

  MPI_File * ifs = NULL;
  LemonWriter * writer = NULL;
  LemonRecordHeader * header = NULL;
  int status = 0;
  int ME_flag=0, MB_flag=0;

  // n_uint64_t bytes;
  n_uint64_t bytes;

  DML_Checksum checksum;
  char *message;

  if(g_cart_id == 0) fprintf(stdout, "\n# [write_lime_contraction] constructing lemon writer for file %s\n", filename);

  ifs = (MPI_File*)malloc(sizeof(MPI_File));
  status = MPI_File_open(g_cart_grid, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, ifs);
  if(status == MPI_SUCCESS) status = MPI_File_set_size(*ifs, 0);
  status = (status == MPI_SUCCESS) ? 0 : 1;
  writer = lemonCreateWriter(ifs, g_cart_grid);
  status = status || (writer == NULL);

  if(status) {
    fprintf(stderr, "[write_lime_contraction] Error, could not open file for writing\n");
    MPI_Abort(MPI_COMM_WORLD, 120);
    MPI_Finalize();
    exit(120);
  }

  // format message
  message = (char*)malloc(2048);
  sprintf(message, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<cvcFormat>\n<type>%s</type>\n<precision>%d</precision>\n<components>%d</components>\n<lx>%d</lx>\n<ly>%d</ly>\n<lz>%d</lz>\n<lt>%d</lt>\n<nconf>%d</nconf>\n<source>%d</source>\n</cvcFormat>\ndate %s", type, prec, N, LX*g_nproc_x, LY*g_nproc_y, LZ*g_nproc_z, T_global, gid, sid, ctime(&g_the_time));
  bytes = strlen(message);
  MB_flag=1, ME_flag=1;
  header = lemonCreateHeader(MB_flag, ME_flag, "cvc_contraction-format", bytes);
  status = lemonWriteRecordHeader(header, writer);
  lemonDestroyHeader(header);
  lemonWriteRecordData( message, (MPI_Offset*)&bytes, writer);
  lemonWriterCloseRecord(writer);
  free(message);

  // binary data message
  bytes = (n_uint64_t)LX_global * LY_global * LZ_global * T_global * (n_uint64_t) (2*N*sizeof(double) * prec / 64);
  MB_flag=1, ME_flag=0;
  header = lemonCreateHeader(MB_flag, ME_flag, "scidac-binary-data", bytes);
  status = lemonWriteRecordHeader(header, writer);
  lemonDestroyHeader(header);
  write_binary_contraction_data(s, writer, prec, N, &checksum);

  // checksum message
  message = (char*)malloc(512);
  sprintf(message, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                   "<scidacChecksum>\n"
                   "  <version>1.0</version>\n"
                   "  <suma>%08x</suma>\n"
                   "  <sumb>%08x</sumb>\n"
                   "</scidacChecksum>", checksum.suma, checksum.sumb);
  bytes = strlen(message);
  MB_flag=0, ME_flag=1;
  header = lemonCreateHeader(MB_flag, ME_flag, "scidac-checksum", bytes);
  status = lemonWriteRecordHeader(header, writer);
  lemonDestroyHeader(header);
  lemonWriteRecordData( message, (MPI_Offset*)&bytes, writer);
  lemonWriterCloseRecord(writer);
  free(message);

  lemonDestroyWriter(writer);
  MPI_File_close(ifs);
  free(ifs);
  return(0);
}
#endif

/***********************************************************
 * read_lime_contraction
 ***********************************************************/

int read_lime_contraction(double * const s, char * filename, const int N, const int position) {
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
  if(bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*(n_uint64_t)(LZ*g_nproc_z)*T_global*2*N*sizeof(double)) prec = 64;
  else if(bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*(n_uint64_t)(LZ*g_nproc_z)*T_global*2*N*sizeof(float)) prec = 32;
  else {
    if(g_proc_id == 0) {
      fprintf(stderr, "wrong length in contraction: bytes = %lu, not %d. Aborting read!\n", bytes, (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*2*N*(int)sizeof(float));
    }
    return(-1);
  }
  if(g_proc_id == 0) {
    printf("# %d Bit precision read\n", prec);
  }

  status = read_binary_contraction_data(s, limereader, prec, N, &checksum);

  if(g_proc_id == 0) {
    printf("# checksum for contractions in file %s position %d is %#x %#x\n",
           filename, position, checksum.suma, checksum.sumb);
  }

  if(status < 0) {
    fprintf(stderr, "LIME read error occured with status = %d while reading file %s!\n Aborting...\n",
            status, filename);
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(500);
  }

  limeDestroyReader(limereader);
  fclose(ifs);
  return(0);
}  /* end of read_lime_contraction */

}  /* end of namespace cvc */
