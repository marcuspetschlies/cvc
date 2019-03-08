/*****************************************************************************
 * contractions_io
 *
 * Tue Nov 15 12:04:09 CET 2016
 *
 * PURPOSE
 * - functions for i/o of contractions, derived from propagator_io
 * - read / write N-component complex (i.e. 2N real components) field to file
 * - expected memory layout
 *     [2N doubles] [2N doubles] ... [2N doubles]
 *     0            1            ... VOLUME-1
 * - space-time indices running (slowest to fastest) t,x,y,z 
 * - uses lime/lemon and the DML checksum
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
#include <sys/stat.h>
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

#ifdef HAVE_HDF5
#include <hdf5.h>
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

  int x, y, z, t, status=0;
  double *tmp;
  float  *tmp2;
  int proc_coords[4], tloc,xloc,yloc,zloc, proc_id;
  n_uint64_t bytes, idx, idx2;
  DML_SiteRank rank;
  int words_bigendian = big_endian();

#ifdef HAVE_MPI
  int iproc, tag;
  int tgeom[2];
  double *buffer;
  MPI_Status mstatus;
#endif
  DML_checksum_init(ans);

#if !(defined PARALLELTX) && !(defined PARALLELTXY) && !(defined PARALLELTXYZ)
  tmp = (double*)malloc(2*N*sizeof(double));
  tmp2 = (float*)malloc(2*N*sizeof(float));

  if(prec == 32) {
    bytes = (n_uint64_t)2*N*sizeof(float);
  } else {
    bytes = (n_uint64_t)2*N*sizeof(double);
  }

  if(g_cart_id==0) {
    for(t = 0; t < T; t++) {
      for(x = 0; x < LX; x++) {
      for(y = 0; y < LY; y++) {
      for(z = 0; z < LZ; z++) {
        /* Rank should be computed by proc 0 only */
        rank = (DML_SiteRank) ((( (t+Tstart)*LX + x)*LY + y)*LZ + z);
        idx = (n_uint64_t)2 * N * g_ipt[t][x][y][z];

        if(!words_bigendian) {
          if(prec == 32) {
            byte_swap_assign_double2single( tmp2, (s + idx), 2*N);
          } else {
            byte_swap_assign( tmp, (s + idx), 2*N);
          }
        } else {
          if(prec == 32) {
            double2single( tmp2, (s + idx), 2*N);
          } else {
            memcpy(tmp, (s+idx), bytes);
          }
        }

        if(prec == 32) {
          DML_checksum_accum(ans,rank, (char *) tmp2, bytes );
          status = limeWriteRecordData((void*)tmp2, &bytes, limewriter);
        }
        else {
          status = limeWriteRecordData((void*)tmp, &bytes, limewriter);
          DML_checksum_accum(ans,rank,(char *) tmp, 2*N*sizeof(double));
        }
      }
      }
      }
    }    /* of loop on T */
  }
#ifdef HAVE_MPI
  tgeom[0] = g_proc_coords[0] * T;
  tgeom[1] = T;
  if( (buffer = (double*)malloc(2*N*LX*LY*LZ*sizeof(double))) == (double*)NULL ) {
    fprintf(stderr, "[write_binary_contraction_data] Error from malloc for buffer\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
  }
  for(iproc=1; iproc<g_nproc; iproc++) {
    if(g_cart_id==0) {
      tag = 2 * iproc;
      MPI_Recv((void*)tgeom, 2, MPI_INT, iproc, tag, g_cart_grid, &mstatus);
      fprintf(stdout, "# [write_binary_contraction_data] iproc = %d; Tstart = %d, T = %d\n", iproc, tgeom[0], tgeom[1]);
       
      for(t=0; t<tgeom[1]; t++) {
        tag = 2 * ( t*g_nproc + iproc ) + 1;
        MPI_Recv((void*)buffer, 2*N*LX*LY*LZ, MPI_DOUBLE,  iproc, tag, g_cart_grid, &mstatus);

        idx = 0;
        for(x=0; x<LX; x++) {
        for(y=0; y<LY; y++) {
        for(z=0; z<LZ; z++) {
          /* Rank should be computed by proc 0 only */
          rank = (DML_SiteRank) ((( (t+tgeom[0])*LX + x)*LY + y)*LZ + z);
          if(!words_bigendian) {
            if(prec == 32) {
              byte_swap_assign_double2single((float*)tmp2, (buffer + idx), 2*N);
              DML_checksum_accum(ans,rank,(char *) tmp2,2*N*sizeof(float));
              status = limeWriteRecordData((void*)tmp2, &bytes, limewriter);
            } else {
              byte_swap_assign(tmp, (buffer + idx), 2*N);
              status = limeWriteRecordData((void*)tmp, &bytes, limewriter);
              DML_checksum_accum(ans,rank,(char *)tmp, 2*N*sizeof(double));
            }
          } else {
            if(prec == 32) {
              double2single((float*)tmp2, (buffer + idx), 2*N);
              DML_checksum_accum(ans,rank,(char *) tmp2,2*N*sizeof(float));
              status = limeWriteRecordData((void*)tmp2, &bytes, limewriter);
            } else {
              status = limeWriteRecordData((void*)(buffer+idx), &bytes, limewriter);
              DML_checksum_accum(ans,rank,(char *) (buffer+idx), 2*N*sizeof(double));
            }
          }
          idx += 2*N;
        }
        }
        }
      }
    }
    if(g_cart_id==iproc) {
      tag = 2 * iproc;
      MPI_Send((void*)tgeom, 2, MPI_INT, 0, tag, g_cart_grid);
      for(t=0; t<T; t++) {
        idx2 = 0;
        for(x=0; x<LX; x++) {  
        for(y=0; y<LY; y++) {  
        for(z=0; z<LZ; z++) {
          idx = 2 * (n_uint64_t)N * g_ipt[t][x][y][z];
          memcpy( buffer+idx2, (s+idx), 2*N*sizeof(double) );
          idx2 += 2*N;
        }}}
        tag = 2 * ( t*g_nproc + iproc) + 1;
        MPI_Send((void*)buffer, 2*N*LX*LY*LZ, MPI_DOUBLE, 0, tag, g_cart_grid);
      }
    }
    MPI_Barrier(g_cart_grid);

  } /* of iproc = 1, ..., g_nproc-1 */
  free(buffer);
#endif  /* of ifdef HAVE_MPI */

#elif (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  if(g_cart_id == 0) {
    fprintf(stderr, "[write_binary_contraction_data] this is at least 2-dim. parallel; use lemon\n");
    return(4);
  }
#endif

  free(tmp2);
  free(tmp);

#ifdef HAVE_MPI
  MPI_Barrier(g_cart_grid);
#endif

//#ifdef HAVE_MPI
//  DML_checksum_combine(ans);
//#endif
//   if(g_cart_id == 0) printf("\n# [write_binary_contraction_data] The final checksum is %#lx %#lx\n", (*ans).suma, (*ans).sumb);
   
  return(0);
}  /* end of function write_binary_contraction_data */

#else  /* HAVE_LIBLEMON */
int write_binary_contraction_data(double * const s, LemonWriter * writer, const int prec, const int N, DML_Checksum * ans) {

  int x, y, z, t, status=0;
  unsigned int ix=0;
  n_uint64_t bytes, idx;
  DML_SiteRank rank;
  double *buffer;
  int words_bigendian = big_endian();
  int latticeSize[] = {T_global, LX_global, LY_global, LZ_global};
  int scidacMapping[] = {0, 1, 2, 3};
  unsigned long bufoffset = 0;
  char *filebuffer = NULL;
  DML_checksum_init(ans);

  if(g_cart_id == 0) fprintf(stdout, "\n# [write_binary_contraction_data] words_bigendian = %d\n", words_bigendian);

  if(prec == 32) bytes = (n_uint64_t)2*N*sizeof(float);   // single precision 
  else           bytes = (n_uint64_t)2*N*sizeof(double);  // double precision

  if((void*)(filebuffer = (char*)malloc(VOLUME * bytes)) == NULL) {
    fprintf (stderr, "\n[write_binary_contraction_data] malloc error in write_binary_contraction_datad\n");
    fflush(stderr);
    EXIT(114);
  }

  if((void*)(buffer = (double*)malloc(2*N*sizeof(double))) == NULL) {
    fprintf (stderr, "\n[write_binary_contraction_data] malloc error\n");
    fflush(stderr);
    EXIT(115);
  }

  for(t = 0; t < T; t++) {
  for(x = 0; x < LX; x++) {
  for(y = 0; y < LY; y++) {
  for(z = 0; z < LZ; z++) {
    rank = (DML_SiteRank) ((((Tstart + t)*LX_global + x+LXstart)*LY_global + LYstart + y)*((DML_SiteRank)LZ_global) + LZstart + z);
    ix  = g_ipt[t][x][y][z];
    idx = 2 * (n_uint64_t)N * ix;
    memcpy( buffer, s+idx, 2*N*sizeof(double) );

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
}  /* end of function write_binary_contraction_data */
#endif  // HAVE_LIBLEMON



/************************************************************
 * read_binary_contraction_data
 ************************************************************/

int read_binary_contraction_data(double * const s, LimeReader * limereader, const int prec, const int N, DML_Checksum *ans) {

  int status=0;
  n_uint64_t bytes, ix, idx;
  double *tmp;
  DML_SiteRank rank;
  float *tmp2;
  int t, x, y, z;
  int words_bigendian = big_endian();

  DML_checksum_init(ans);
  rank = (DML_SiteRank) 0;
 
  if( (tmp = (double*)malloc(2*N*sizeof(double))) == (double*)NULL ) {
    EXIT(500);
  }
  if( (tmp2 = (float*)malloc(2*N*sizeof(float))) == (float*)NULL ) {
    EXIT(501);
  }
 
 
  if(prec == 32) bytes = 2 * (n_uint64_t)N * sizeof(float);
  else           bytes = 2 * (n_uint64_t)N * sizeof(double);
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
    idx = 2 * N * ix;
    if(!words_bigendian) {
      if(prec == 32) {
        byte_swap_assign_single2double( (s + idx), tmp2, 2*N);
      } else {
        byte_swap_assign( (s + idx), tmp, 2*N);
      }
    } else {  // words_bigendian true
      if(prec == 32) {
        single2double( (s + idx), tmp2, 2*N);
      }
      else {
        memcpy(s+idx, tmp, bytes);
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

int write_contraction_format(char * filename, const int prec, const int N, char * type, const int gid, const int append) {
  FILE * ofs = NULL;
  LimeWriter * limewriter = NULL;
  LimeRecordHeader * limeheader = NULL;
  int status = 0;
  int ME_flag=0, MB_flag=1;
  char message[3000];
  n_uint64_t bytes;

  if(g_cart_id==0) {
    if( append ) {
      ofs = fopen(filename, "a");
      fseek(ofs, 0, SEEK_END);
    } else {
      ofs = fopen(filename, "w");
    }
  
    if(ofs == (FILE*)NULL) {
      fprintf(stderr, "[write_contraction_format] Could not open file %s for writing!\n Aborting...\n", filename);
      EXIT(500);
    }
    limewriter = limeCreateWriter( ofs );
    if(limewriter == (LimeWriter*)NULL) {
      fprintf(stderr, "[write_contraction_format] LIME error in file %s for writing!\n Aborting...\n", filename);
      EXIT(500);
    }
  
    sprintf(message, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<cvcFormat>\n<type>%s</type>\n<precision>%d</precision>\n<components>%d</components>\n<lx>%d</lx>\n<ly>%d</ly>\n<lz>%d</lz>\n<lt>%d</lt>\n<nconf>%d</nconf>\n</cvcFormat>", 
        type, prec, N, LX*g_nproc_x, LY*g_nproc_y, LZ*g_nproc_z, T_global, gid);
    bytes = strlen( message );
    limeheader = limeCreateHeader(MB_flag, ME_flag, "cvc-contraction-format", bytes);
    status = limeWriteRecordHeader( limeheader, limewriter);
    if(status < 0 ) {
      fprintf(stderr, "[write_contraction_format] LIME write header error %d\n", status);
      EXIT(500);
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
int write_lime_contraction(double * const s, char * filename, const int prec, const int N, char * type, const int gid, const int append) {

  FILE * ofs = NULL;
  LimeWriter * limewriter = NULL;
  LimeRecordHeader * limeheader = NULL;
  int status = 0;
  int ME_flag=0, MB_flag=0;
  n_uint64_t bytes;
  DML_Checksum checksum;

  write_contraction_format(filename, prec, N, type, gid, append);

  if(g_cart_id==0) {
    ofs = fopen(filename, "a");
    if(ofs == (FILE*)NULL) {
      fprintf(stderr, "[write_lime_contraction] Could not open file %s for writing!\n Aborting...\n", filename);
      EXIT(500);
    }
  
    limewriter = limeCreateWriter( ofs );
    if(limewriter == (LimeWriter*)NULL) {
      fprintf(stderr, "[write_lime_contraction] LIME error in file %s for writing!\n Aborting...\n", filename);
      EXIT(500);
    }

    bytes = LX*g_nproc_x*LY*g_nproc_y*LZ*g_nproc_z*T_global*(n_uint64_t)2*N*sizeof(double)*prec/64;
    MB_flag=0; ME_flag=1;
    limeheader = limeCreateHeader(MB_flag, ME_flag, "scidac-binary-data", bytes);
    status = limeWriteRecordHeader( limeheader, limewriter);
    if(status < 0 ) {
      fprintf(stderr, "[write_lime_contraction] LIME write header (scidac-binary-data) error %d\n", status);
      EXIT(500);
    }
    limeDestroyHeader( limeheader );
  }
  
  status = write_binary_contraction_data(s, limewriter, prec, N, &checksum);
  if(status < 0 ) {
    fprintf(stderr, "[write_lime_contraction] Error from write_binary_contraction_data, status was %d\n", status);
    EXIT(502);
  }
  if(g_cart_id==0) {
    printf("# [write_lime_contraction] Final check sum is (%#lx  %#lx)\n", checksum.suma, checksum.sumb);
    if(ferror(ofs)) {
      fprintf(stderr, "[write_lime_contraction] Warning! Error while writing to file %s \n", filename);
    }
    limeDestroyWriter( limewriter );
    fflush(ofs);
    fclose(ofs);
  }
  write_checksum(filename, &checksum);
  return(0);
}
#else  // HAVE_LIBLEMON
int write_lime_contraction(double * const s, char * filename, const int prec, const int N, char * type, const int gid, const int append) {

  MPI_File * ifs = NULL;
  LemonWriter * writer = NULL;
  LemonRecordHeader * header = NULL;
  int status = 0;
  int ME_flag=0, MB_flag=0;

/* #ifdef HAVE_MPI
  MPI_Offset bytes;
#else */
  n_uint64_t bytes;
/* #endif */

  DML_Checksum checksum;
  char *message;

  if(g_cart_id == 0) fprintf(stdout, "\n# [write_lime_contraction] constructing lemon writer for file %s\n", filename);

  ifs = (MPI_File*)malloc(sizeof(MPI_File));
  if(append) {
    status = MPI_File_open(g_cart_grid, filename, MPI_MODE_WRONLY | MPI_MODE_APPEND, MPI_INFO_NULL, ifs);
    /* if(status == MPI_SUCCESS) status = MPI_File_seek (*ifs, 0, MPI_SEEK_END); */
  } else {
    status = MPI_File_open(g_cart_grid, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, ifs);
    if(status == MPI_SUCCESS) status = MPI_File_set_size(*ifs, 0);
  }
  status = (status == MPI_SUCCESS) ? 0 : 1;
  if(status) {
    fprintf(stderr, "[write_lime_contraction] Error, could not open file %s\n", filename);
    EXIT(119);
  }

  writer = lemonCreateWriter(ifs, g_cart_grid);
  status = status || (writer == NULL);

  if(status) {
    fprintf(stderr, "[write_lime_contraction] Error from lemonCreateWriter\n");
    EXIT(120);
  }

  // format message
  message = (char*)malloc(2048);
  sprintf(message, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<cvcFormat>\n<type>%s</type>\n<precision>%d</precision>\n<components>%d</components>\n<lx>%d</lx>\n<ly>%d</ly>\n<lz>%d</lz>\n<lt>%d</lt>\n<nconf>%d</nconf>\n</cvcFormat>\ndate %s", type, prec, N, LX*g_nproc_x, LY*g_nproc_y, LZ*g_nproc_z, T_global, gid, ctime(&g_the_time));
  bytes = strlen(message);
  MB_flag=1, ME_flag=1;
  header = lemonCreateHeader(MB_flag, ME_flag, "cvc_contraction-format", bytes);
  status = lemonWriteRecordHeader(header, writer);
  lemonDestroyHeader(header);
  lemonWriteRecordData( message, &bytes, writer);
  lemonWriterCloseRecord(writer);
  free(message);

  // binary data message
#ifdef HAVE_MPI
  bytes = (MPI_Offset)LX_global * LY_global * LZ_global * T_global * (MPI_Offset) (2*N*sizeof(double) * prec / 64);
#else
  bytes = (n_uint64_t)LX_global * LY_global * LZ_global * T_global * (n_uint64_t) (2*N*sizeof(double) * prec / 64);
#endif
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
  lemonWriteRecordData( message, &bytes, writer);
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
      fprintf(stderr, "[read_lime_contraction] Error opening file %s\n", filename);
    }
    return(106);
  }

  limereader = limeCreateReader( ifs );
  if( limereader == (LimeReader *)NULL ) {
    if(g_proc_id == 0) {
      fprintf(stderr, "[read_lime_contraction] Unable to open LimeReader\n");
    }
    return(-1);
  }
  while( (status = limeReaderNextRecord(limereader)) != LIME_EOF ) {
    if(status != LIME_SUCCESS ) {
      fprintf(stderr, "[read_lime_contraction] limeReaderNextRecord returned error with status = %d!\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = limeReaderType(limereader);
    if(strcmp("scidac-binary-data",header_type) == 0) getpos++;
    if(getpos == position) break;
  }
  if(status == LIME_EOF) {
    if(g_proc_id == 0) {
      fprintf(stderr, "[read_lime_contraction] no scidac-binary-data record found in file %s\n",filename);
    }
    limeDestroyReader(limereader);
    fclose(ifs);
    if(g_proc_id == 0) {
      fprintf(stderr, "[read_lime_contraction] try to read in non-lime format\n");
    }
    return(read_contraction(s, NULL, filename, N));
  }
  bytes = limeReaderBytes(limereader);
  if(bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*(n_uint64_t)(LZ*g_nproc_z)*T_global*2*N*sizeof(double)) prec = 64;
  else if(bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*(n_uint64_t)(LZ*g_nproc_z)*T_global*2*N*sizeof(float)) prec = 32;
  else {
    if(g_proc_id == 0) {
      fprintf(stderr, "[read_lime_contraction] wrong length in contraction: bytes = %lu, not %d. Aborting read!\n", bytes, (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*2*N*(int)sizeof(float));
    }
    return(-1);
  }
  if(g_proc_id == 0) {
    printf("# [read_lime_contraction] %d Bit precision read\n", prec);
  }

  status = read_binary_contraction_data(s, limereader, prec, N, &checksum);

  if(g_proc_id == 0) {
    printf("# [read_lime_contraction] checksum for contractions in file %s position %d is %#x %#x\n",
           filename, position, checksum.suma, checksum.sumb);
  }

  if(status < 0) {
    fprintf(stderr, "[read_lime_contraction] LIME read error occured with status = %d while reading file %s!\n Aborting...\n",
            status, filename);
    EXIT(500);
  }

  limeDestroyReader(limereader);
  fclose(ifs);
  return(0);
}  /* end of read_lime_contraction */

/***************************************************************************/
/***************************************************************************/

#ifdef HAVE_HDF5

/***************************************************************************
 * read time-momentum-dependent accumulated loop data from HDF5 file
 *
 * OUT: buffer          : date set
 * IN : file            : here filename
 * IN : tag             : here group
 * IN : io_proc         : I/O id
 *
 ***************************************************************************/
int read_from_h5_file ( void * const buffer, void * file, char*tag,  int const io_proc ) {

  if ( io_proc > 0 ) {

    char * filename = (char *)file;

    /***************************************************************************
     * time measurement
     ***************************************************************************/
    struct timeval ta, tb;
    gettimeofday ( &ta, (struct timezone *)NULL );

    /***************************************************************************
     * io_proc 2 is origin of Cartesian grid and does the write to disk
     ***************************************************************************/
    if(io_proc == 2) {
  
      /***************************************************************************
       * create or open file
       ***************************************************************************/

      hid_t   file_id = -1;
      herr_t  status;

      struct stat fileStat;
      if ( stat( filename, &fileStat) < 0 ) {
        fprintf ( stderr, "[read_from_h5_file] Error, file %s does not exist %s %d\n", filename, __FILE__, __LINE__ );
        return ( 1 );
      } else {
        /* open an existing file. */
        if ( g_verbose > 1 ) fprintf ( stdout, "# [read_from_h5_file] open existing file\n" );
  
        unsigned flags = H5F_ACC_RDONLY;  /* IN: File access flags. Allowable values are:
                                             H5F_ACC_RDWR   --- Allow read and write access to file.
                                             H5F_ACC_RDONLY --- Allow read-only access to file.
  
                                             H5F_ACC_RDWR and H5F_ACC_RDONLY are mutually exclusive; use exactly one.
                                             An additional flag, H5F_ACC_DEBUG, prints debug information.
                                             This flag can be combined with one of the above values using the bit-wise OR operator (`|'),
                                             but it is used only by HDF5 Library developers; it is neither tested nor supported for use in applications. */
        hid_t fapl_id = H5P_DEFAULT;
        /*  hid_t H5Fopen ( const char *name, unsigned flags, hid_t fapl_id ) */
        file_id = H5Fopen (         filename,         flags,        fapl_id );

        if ( file_id < 0 ) {
          fprintf ( stderr, "[read_from_h5_file] Error from H5Fopen %s %d\n", __FILE__, __LINE__ );
          return ( 2 );
        }
      }
  
      if ( g_verbose > 1 ) fprintf ( stdout, "# [read_from_h5_file] file_id = %ld\n", file_id );
  
      /***************************************************************************
       * open H5 data set
       ***************************************************************************/
      hid_t dapl_id       = H5P_DEFAULT;

      hid_t dataset_id = H5Dopen2 ( file_id, tag, dapl_id );
      if ( dataset_id < 0 ) {
        fprintf ( stderr, "[read_from_h5_file] Error from H5Dopen2 %s %d\n", __FILE__, __LINE__ );
        return ( 3 );
      }

      /***************************************************************************
       * some default settings for H5Dread
       ***************************************************************************/
      hid_t mem_type_id   = H5T_NATIVE_DOUBLE;
      hid_t mem_space_id  = H5S_ALL;
      hid_t file_space_id = H5S_ALL;
      hid_t xfer_plist_id = H5P_DEFAULT;

      /***************************************************************************
       * read data set
       ***************************************************************************/
      status = H5Dread ( dataset_id, mem_type_id, mem_space_id, file_space_id, xfer_plist_id, buffer );
      if ( status < 0 ) {
        fprintf ( stderr, "[read_from_h5_file] Error from H5Dread %s %d\n", __FILE__, __LINE__ );
        return ( 4 );
      }

      /***************************************************************************
       * close data set
       ***************************************************************************/
      status = H5Dclose ( dataset_id );
      if ( status < 0 ) {
        fprintf ( stderr, "[read_from_h5_file] Error from H5Dclose %s %d\n", __FILE__, __LINE__ );
        return ( 5 );
      }

      /***************************************************************************
       * close the file
       ***************************************************************************/
      status = H5Fclose ( file_id );
      if( status < 0 ) {
        fprintf(stderr, "[read_from_h5_file] Error from H5Fclose, status was %d %s %d\n", status, __FILE__, __LINE__);
        return(6);
      } 

    }  /* if io_proc == 2 */

    /***************************************************************************
     * time measurement
     ***************************************************************************/
    gettimeofday ( &tb, (struct timezone *)NULL );
  
    show_time ( &ta, &tb, "read_from_h5_file", "write h5", 1 );

  }  /* end of of if io_proc > 0 */
  
  return(0);

}  /* end of read_from_h5_file */

#endif  /* of if HAVE_HDF5 */

}  /* end of namespace cvc */
