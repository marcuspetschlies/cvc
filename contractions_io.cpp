/*****************************************************************************
 * contractions_io
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

#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

#ifdef HAVE_HDF5
#include "hdf5.h"
#endif

#include "cvc_complex.h"
#include "global.h"
#include "cvc_geometry.h"
#include "dml.h"
#include "io_utils.h"
#include "propagator_io.h"
#include "contractions_io.h"
#include "cvc_utils.h"
#include "cvc_timer.h"

#define MAX_SUBGROUP_NUMBER 20

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

  n_uint64_t const LLsize[4] = { ( LX * g_nproc_x ) * ( LY * g_nproc_y ) * ( LZ * g_nproc_z ),
                                                   ( LY * g_nproc_y ) * ( LZ * g_nproc_z ), 
                                                                        ( LZ * g_nproc_z ),
                                                                                            1 };

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

  for(t = 0; t < T; t++){
  for(x = 0; x < LX; x++){
  for(y = 0; y < LY; y++){
  for(z = 0; z < LZ; z++){

#ifdef HAVE_MPI
    limeReaderSeek(limereader,(n_uint64_t) (
			    ( Tstart  + t ) * LLsize[0] 
	                  + ( LXstart + x ) * LLsize[1] 
		          + ( LYstart + y ) * LLsize[2] 
			  + ( LZstart + z ) * LLsize[3] ) * bytes, SEEK_SET);
#endif

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

  // n_uint64_t bytes;
  LEMON_OFFSET_TYPE bytes;

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

/***********************************************************/
/***********************************************************/

#ifdef HAVE_LHPC_AFF
/***********************************************************
 * read AFF contraction
 ***********************************************************/
int read_aff_contraction ( void * const contr, void * const areader, void * const afilename, char * tag, unsigned int const nc) {

  struct AffReader_s *affr = NULL;
  struct AffNode_s *affn = NULL, *affdir = NULL;
  uint32_t items = nc;
  int exitstatus;

  if ( areader != NULL ) {
    affr = (struct AffReader_s *) areader;
  } else if ( afilename != NULL ) {
    char * filename = (char*) afilename;
    if (g_verbose > 2 ) fprintf ( stdout, "# [read_aff_contraction] new AFF reader for file %s %s %d\n", filename, __FILE__, __LINE__ );

    affr = aff_reader (filename);
    if( const char * aff_status_str = aff_reader_errstr(affr) ) {
      fprintf(stderr, "[read_aff_contraction] Error from aff_reader for file %s, status was %s %s %d\n", filename, aff_status_str, __FILE__, __LINE__);
      return( 4 );
    } else {
      if (g_verbose > 2 ) fprintf(stdout, "# [read_aff_contraction] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
    }
  } else { 
    fprintf ( stderr, "[read_aff_contraction] Error, neither reader nor filename\n" );
    return ( 1 );
  }

  if( (affn = aff_reader_root( affr )) == NULL ) {
    fprintf(stderr, "[read_aff_contraction] Error, aff reader is not initialized %s %d\n", __FILE__, __LINE__);
    return( 2 );
  }

  affdir = aff_reader_chpath ( affr, affn, tag );
  if ( affdir == NULL ) {
    fprintf(stderr, "[read_aff_contraction] Error from affdir %s %d\n", __FILE__, __LINE__);
    return( 2 );
  }

  /* fprintf ( stdout, "# [read_aff_contraction] items = %u path = %s\n", items , tag); */

  exitstatus = aff_node_get_complex ( affr, affdir, (double _Complex*) contr, items );
  if( exitstatus != 0 ) {
    fprintf(stderr, "[read_aff_contraction] Error from aff_node_get_complex for key \"%s\", status was %d errmsg %s %s %d\n", tag, exitstatus,
       aff_reader_errstr ( affr ), __FILE__, __LINE__);
    return ( 105 );
  }

  if ( areader == NULL )  {
    /* in that case affr was reader within the scope of this function */
    aff_reader_close ( affr );
  }

  return ( 0 );

}  /* end of read_aff_contraction */
#endif


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
  
    show_time ( &ta, &tb, "read_from_h5_file", "read h5", 1 );

  }  /* end of of if io_proc > 0 */
  
  return(0);

}  /* end of read_from_h5_file */

#endif  /* of if HAVE_HDF5 */


#ifdef HAVE_HDF5

/***************************************************************************
 * write attribute
 *
 ***************************************************************************/
int write_h5_attribute ( const char * filename, const char * attr_name, const char * info ) {

  herr_t status;

  /* hid_t obj_id = H5Oopen ( , , ); */
  hid_t file_id = H5Fopen ( filename, H5F_ACC_RDWR, H5P_DEFAULT );
  if ( file_id < 0 ) {
    fprintf( stderr, "[write_h5_attribute] Error from H5FOpen, status was %ld %s %d\n", file_id, __FILE__, __LINE__ );
    return(1);
  }

  hid_t attrdat_id = H5Screate(H5S_SCALAR);
  if ( attrdat_id < 0 ) {
    fprintf( stderr, "[write_h5_attribute] Error from H5Screate, status was %ld %s %d\n", attrdat_id, __FILE__, __LINE__ );
    return(1);
  }

  hid_t type_id = H5Tcopy(H5T_C_S1);

  H5Tset_size(type_id, strlen( info ) );

  hid_t attr_id = H5Acreate2 ( file_id, attr_name, type_id, attrdat_id, H5P_DEFAULT, H5P_DEFAULT);
  if ( attr_id < 0 ) {
    fprintf( stderr, "[write_h5_attribute] Error from H5Acreate2, status was %ld %s %d\n", attr_id, __FILE__, __LINE__ );
    return(1);
  }

  status = H5Awrite ( attr_id, type_id, info );
  if ( status < 0 ) {
    fprintf( stderr, "[write_h5_attribute] Error from H5Awrite, status was %d %s %d\n", status, __FILE__, __LINE__ );
    return(1);
  }

  H5Aclose(attr_id);

  H5Tclose(type_id);

  H5Sclose(attrdat_id);

  status = H5Fclose ( file_id );
  if ( status < 0 ) {
    fprintf( stderr, "[write_h5_attribute] Error from H5Awrite, status was %d %s %d\n", status, __FILE__, __LINE__ );
    return(1);
  }

  return (0);
}  /* end of write_h5_attribute */



/***************************************************************************
 * write contraction data to file
 *
 ***************************************************************************/

int write_h5_contraction ( void * const contr, void * const awriter, void * const afilename, char * tag, unsigned int const nc, const char * data_type, int const ncdim, const int * const cdim ) {

  char * filename = (char *)afilename;
  if ( filename == NULL ) {
    fprintf( stderr, "[write_h5_contraction] Error, need filename %s %d\n", __FILE__, __LINE__ );
    return(1);
  }

  /***************************************************************************
   * create or open file
   ***************************************************************************/

    hid_t   file_id;
    herr_t  status;

    struct stat fileStat;
    if ( stat( filename, &fileStat) < 0 ) {
      /* creat a new file */

      if ( g_verbose > 2 ) fprintf ( stdout, "# [write_h5_contraction] create new file\n" );

      unsigned flags = H5F_ACC_TRUNC; /* IN: File access flags. Allowable values are:
                                         H5F_ACC_TRUNC --- Truncate file, if it already exists, erasing all data previously stored in the file.
                                         H5F_ACC_EXCL  --- Fail if file already exists.

                                         H5F_ACC_TRUNC and H5F_ACC_EXCL are mutually exclusive; use exactly one.
                                         An additional flag, H5F_ACC_DEBUG, prints debug information.
                                         This flag can be combined with one of the above values using the bit-wise OR operator (`|'),
                                         but it is used only by HDF5 Library developers; it is neither tested nor supported for use in applications.  */
      hid_t fcpl_id = H5P_DEFAULT; /* IN: File creation property list identifier, used when modifying default file meta-data.
                                      Use H5P_DEFAULT to specify default file creation properties. */
      hid_t fapl_id = H5P_DEFAULT; /* IN: File access property list identifier. If parallel file access is desired,
                                      this is a collective call according to the communicator stored in the fapl_id.
                                      Use H5P_DEFAULT for default file access properties. */

      /*  hid_t H5Fcreate ( const char *name, unsigned flags, hid_t fcpl_id, hid_t fapl_id ) */
      file_id = H5Fcreate (         filename,          flags,       fcpl_id,       fapl_id );

    } else {
      /* open an existing file. */
      if ( g_verbose > 2 ) fprintf ( stdout, "# [write_h5_contraction] open existing file\n" );

      unsigned flags = H5F_ACC_RDWR;  /* IN: File access flags. Allowable values are:
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
        fprintf ( stderr, "[write_h5_contraction] Error from H5Fopen for filename %s %s %d\n", filename, __FILE__, __LINE__ );
        return( 1 );
      }
    }

    if ( g_verbose > 2 ) fprintf ( stdout, "# [write_h5_contraction] file_id = %ld\n", file_id );

    /***************************************************************************
     * some default settings for H5Dwrite
     ***************************************************************************/
    hid_t dtype_id;
    hid_t mem_type_id;
    if ( strcmp( data_type , "double" ) == 0 ) {
      dtype_id = H5Tcopy( H5T_NATIVE_DOUBLE );
      mem_type_id   = H5T_NATIVE_DOUBLE;
    } else if ( strcmp( data_type , "int" ) == 0 ) {
      dtype_id = H5Tcopy( H5T_NATIVE_INT );
      mem_type_id   = H5T_NATIVE_INT;
    } else if ( strcmp( data_type , "char" ) == 0 ) {
      dtype_id = H5Tcopy( H5T_NATIVE_CHAR );
      mem_type_id   = H5T_NATIVE_CHAR;
    } else {
      fprintf ( stderr, "[write_h5_contraction] Error, unrecognized data_type %s %s %d\n", data_type, __FILE__, __LINE__ );
      return ( 8 );
    }
    hid_t mem_space_id  = H5S_ALL;
    hid_t file_space_id = H5S_ALL;
    hid_t xfer_plit_id  = H5P_DEFAULT;
    hid_t lcpl_id       = H5P_DEFAULT;
    hid_t dcpl_id       = H5P_DEFAULT;
    hid_t dapl_id       = H5P_DEFAULT;
    hid_t gcpl_id       = H5P_DEFAULT;
    hid_t gapl_id       = H5P_DEFAULT;
    /* size_t size_hint    = 0; */

  /***************************************************************************
   * H5 data space and data type
   ***************************************************************************/
  status = H5Tset_order ( dtype_id, H5T_ORDER_LE );
  /* big_endian() ?  H5T_IEEE_F64BE : H5T_IEEE_F64LE; */

  /* hsize_t dims[1];
  dims[0] = nc;*/    /* number of double elements */

  hsize_t * dims = (hsize_t*) malloc ( ncdim * sizeof ( hsize_t ) );
  for( int i=0; i<ncdim;i++ ) {
    dims[i] = cdim[i];
  }

  /*
             int rank                             IN: Number of dimensions of dataspace.
             const hsize_t * current_dims         IN: Array specifying the size of each dimension.
             const hsize_t * maximum_dims         IN: Array specifying the maximum size of each dimension.
             hid_t H5Screate_simple( int rank, const hsize_t * current_dims, const hsize_t * maximum_dims )
   */
  hid_t space_id = H5Screate_simple(    ncdim,                         dims,                          NULL);

  free ( dims );

  /***************************************************************************
   * create the target (sub-)group and all
   * groups in hierarchy above if they don't exist
   ***************************************************************************/
  hid_t grp_list[MAX_SUBGROUP_NUMBER];
  int grp_list_nmem = 0;
  char grp_name[400], grp_name_tmp[400];
  char * grp_ptr = NULL;
  char grp_sep[] = "/";
  char * dataset_name = NULL;
  strcpy ( grp_name, tag );
  strcpy ( grp_name_tmp, grp_name );
  if ( g_verbose > 2 ) fprintf ( stdout, "# [write_h5_contraction] full grp_name = %s\n", grp_name );
  grp_ptr = strtok ( grp_name_tmp, grp_sep );
  
  while ( grp_ptr != NULL ) {
    grp_ptr = strtok(NULL, grp_sep );
    grp_list_nmem++;
  }

  if ( g_verbose > 4 ) {
    fprintf ( stdout, "# [write_h5_contraction] grp_name %s grp_name_tmp %s  grp_list_nmem %d %s %d\n", grp_name, grp_name_tmp, grp_list_nmem, __FILE__, __LINE__ );
  }

  strcpy ( grp_name_tmp, grp_name );
  grp_ptr = strtok ( grp_name_tmp, grp_sep );
  int imem = 0;
  while ( grp_ptr != NULL ) {
    if ( imem == grp_list_nmem - 1 ) {
      dataset_name = grp_ptr;
    } else {
      hid_t grp;
      hid_t loc_id = ( imem == 0 ) ? file_id : grp_list[imem-1];
      if ( g_verbose > 2 ) fprintf ( stdout, "# [write_h5_contraction] grp_ptr = %s\n", grp_ptr );

      grp = H5Gopen2( loc_id, grp_ptr, gapl_id );
      if ( grp < 0 ) {
        fprintf ( stderr, "[write_h5_contraction] Error from H5Gopen2 for group %s, status was %ld %s %d\n", grp_ptr, grp, __FILE__, __LINE__ );
        grp = H5Gcreate2 (       loc_id,         grp_ptr,       lcpl_id,       gcpl_id,       gapl_id );
        if ( grp < 0 ) {
          fprintf ( stderr, "[write_h5_contraction] Error from H5Gcreate2 for group %s, status was %ld %s %d\n", grp_ptr, grp, __FILE__, __LINE__ );
          return ( 6 );
        } else {
          if ( g_verbose > 2 ) fprintf ( stdout, "# [write_h5_contraction] created group %s %ld %s %d\n", grp_ptr, grp, __FILE__, __LINE__ );
        }
      } else {
        if ( g_verbose > 2 ) fprintf ( stdout, "# [write_h5_contraction] opened group %s %ld %s %d\n", grp_ptr, grp, __FILE__, __LINE__ );
      }
      grp_list[imem] = grp;
    }

    grp_ptr = strtok(NULL, grp_sep );
    imem++;
  }  /* end of loop on sub-groups */

  if ( g_verbose > 4 ) {
    fprintf ( stdout, "# [write_h5_contraction] dataset_name %s  %s %d\n", dataset_name, __FILE__, __LINE__ );
  }

  /***************************************************************************
   * write data set
   ***************************************************************************/

  /* hid_t loc_id = ( grp_list_nmem == 0 ) ? file_id : grp_list[grp_list_nmem - 1 ]; */
  hid_t loc_id = ( grp_list_nmem == 1 ) ? file_id : grp_list[grp_list_nmem - 2 ];
  if ( loc_id < 0 ) {
    fprintf(stderr, "[write_h5_contraction] Error , loc_id < 0 %s %d\n", __FILE__, __LINE__);
    return(18);
  }

  /***************************************************************************
   * create a data set
   ***************************************************************************/
      /*
                   hid_t loc_id         IN: Location identifier
                   const char *name     IN: Dataset name
                   hid_t dtype_id       IN: Datatype identifier
                   hid_t space_id       IN: Dataspace identifier
                   hid_t lcpl_id        IN: Link creation property list
                   hid_t dcpl_id        IN: Dataset creation property list
                   hid_t dapl_id        IN: Dataset access property list
                   hid_t H5Dcreate2 ( hid_t loc_id, const char *name, hid_t dtype_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id )

                   hid_t H5Dcreate ( hid_t loc_id, const char *name, hid_t dtype_id, hid_t space_id, hid_t lcpl_id, hid_t dcpl_id, hid_t dapl_id ) 
       */
  /* hid_t dataset_id = H5Dcreate (       loc_id,             tag,       dtype_id,       space_id,       lcpl_id,       dcpl_id,       dapl_id ); */
  hid_t dataset_id = H5Dcreate (       loc_id,         dataset_name,       dtype_id,       space_id,       lcpl_id,       dcpl_id,       dapl_id );
  if ( dataset_id < 0 ) {
    fprintf(stderr, "[write_h5_contraction] Error from H5Dcreate %s %d\n", __FILE__, __LINE__);
    return(19);
  }

  /***************************************************************************
   * write the current data set
   ***************************************************************************/
      /*
               hid_t dataset_id           IN: Identifier of the dataset to write to.
               hid_t mem_type_id          IN: Identifier of the memory datatype.
               hid_t mem_space_id         IN: Identifier of the memory dataspace.
               hid_t file_space_id        IN: Identifier of the dataset's dataspace in the file.
               hid_t xfer_plist_id        IN: Identifier of a transfer property list for this I/O operation.
               const void * buf           IN: Buffer with data to be written to the file.
        herr_t H5Dwrite ( hid_t dataset_id, hid_t mem_type_id, hid_t mem_space_id, hid_t file_space_id, hid_t xfer_plist_id, const void * buf )
       */
  status = H5Dwrite (       dataset_id,       mem_type_id,       mem_space_id,       file_space_id,        xfer_plit_id,    contr );

  if( status < 0 ) {
    fprintf(stderr, "[write_h5_contraction] Error from H5Dwrite, status was %d %s %d\n", status, __FILE__, __LINE__);
    return(8);
  }

  /***************************************************************************
   * close the current data set
   ***************************************************************************/
  status = H5Dclose ( dataset_id );
  if( status < 0 ) {
    fprintf(stderr, "[write_h5_contraction] Error from H5Dclose, status was %d %s %d\n", status, __FILE__, __LINE__);
    return(9);
  }

  /***************************************************************************
   * close the data space
   ***************************************************************************/
  status = H5Sclose ( space_id );
  if( status < 0 ) {
    fprintf(stderr, "[write_h5_contraction] Error from H5Sclose, status was %d %s %d\n", status, __FILE__, __LINE__);
    return(10);
  }

  /***************************************************************************
   * close all (sub-)groups in reverse order
   ***************************************************************************/
  /* for ( int i = grp_list_nmem - 1; i>= 0; i-- ) */
  for ( int i = grp_list_nmem - 2; i>= 0; i-- )
  {
    status = H5Gclose ( grp_list[i] );
    if( status < 0 ) {
      fprintf(stderr, "[write_h5_contraction] Error from H5Gclose, status was %d %s %d\n", status, __FILE__, __LINE__);
      return(11);
    } else {
      if ( g_verbose > 2 ) fprintf(stdout, "# [write_h5_contraction] closed group %ld %s %d\n", grp_list[i], __FILE__, __LINE__);
    }
  }

  /***************************************************************************
   * close the data type
   ***************************************************************************/
  status = H5Tclose ( dtype_id );
  if( status < 0 ) {
    fprintf(stderr, "[write_h5_contraction] Error from H5Tclose, status was %d %s %d\n", status, __FILE__, __LINE__);
    return(12);
  }

  /***************************************************************************
   * close the file
   ***************************************************************************/
  status = H5Fclose ( file_id );
  if( status < 0 ) {
    fprintf(stderr, "[write_h5_contraction] Error from H5Fclose, status was %d %s %d\n", status, __FILE__, __LINE__);
    return(13);
  } 
  
  return(0);

}  /* end of write_h5_contraction */

#endif  /* if def HAVE_HDF5 */



#ifdef HAVE_LHPC_AFF
/***********************************************************
 * write AFF contraction
 ***********************************************************/
int write_aff_contraction ( void * const contr, void * const awriter, void * const afilename, char * tag, unsigned int const nc, const char * data_type) {

  struct AffWriter_s *affw = NULL;
  struct AffNode_s *affn = NULL, *affdir = NULL;
  uint32_t items = nc;
  int exitstatus = 0;

  if ( awriter != NULL ) {
    affw = (struct AffWriter_s *) awriter;
  } else if ( afilename != NULL ) {
    char * filename = (char*) afilename;
    if (g_verbose > 2 ) fprintf ( stdout, "# [write_aff_contraction] new AFF reader for file %s %s %d\n", filename, __FILE__, __LINE__ );

    affw = aff_writer (filename);
    if( const char * aff_status_str = aff_writer_errstr(affw) ) {
      fprintf(stderr, "[write_aff_contraction] Error from aff_writer, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      return( 4 );
    } else {
      if (g_verbose > 2 ) fprintf(stdout, "# [write_aff_contraction] reading data from aff file %s %s %d\n", filename, __FILE__, __LINE__);
    }
  } else { 
    fprintf ( stderr, "[write_aff_contraction] Error, neither reader nor filename\n" );
    return ( 1 );
  }

  if( (affn = aff_writer_root( affw )) == NULL ) {
    fprintf(stderr, "[read_aff_contraction] Error, aff writer is not initialized %s %d\n", __FILE__, __LINE__);
    return( 2 );
  }

  affdir = aff_writer_mkpath ( affw, affn, tag );
  if ( affdir == NULL ) {
    fprintf(stderr, "[write_aff_contraction] Error from aff_writer_mkpath %s %d\n", __FILE__, __LINE__);
    return( 2 );
  }

  /* fprintf ( stdout, "# [write_aff_contraction] items = %u path = %s\n", items , tag); */

  if ( strcmp( data_type , "complex" ) == 0 ) {
    exitstatus = aff_node_put_complex ( affw, affdir, (double _Complex*) contr, items );
  } else if ( strcmp( data_type , "double" ) == 0 ) {
    exitstatus = aff_node_put_double ( affw, affdir, (double*) contr, items );
  }

  if( exitstatus != 0 ) {
    fprintf(stderr, "[write_aff_contraction] Error from aff_node_put_complex for key \"%s\", status was %d errmsg %s %s %d\n", tag, exitstatus,
       aff_writer_errstr ( affw ), __FILE__, __LINE__);
    return ( 105 );
  }

  if ( awriter == NULL )  {
    const char * aff_status_str = (char*)aff_writer_close (affw);
    if( aff_status_str != NULL ) {
      fprintf(stderr, "[write_aff_contraction] Error from aff_writer_close, status was %s %s %d\n", aff_status_str, __FILE__, __LINE__);
      return(32);
    }
  }

  return ( 0 );

}  /* end of write_aff_contraction */

#endif



}  /* end of namespace cvc */
