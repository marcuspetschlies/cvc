/*
 * TODO:
 * - change all writing routines to enable use of MPI
 * - _ATTENTION_ prec=32 will not work properly -> MPI-message is still 64 even
 *   if writing in 32 !!!
 * CHANGES:
 * - I do not write the propagator type -> changed file-opening mode 
 *   write_propagator_format to "w"
 */


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
#include "cvc_linalg.h"
#include "io_utils.h"
#include "cvc_utils.h"
#include "Q_phi.h"
#include "propagator_io.h"

namespace cvc {

/* write a one flavour propagator to file */
int write_propagator(double * const s, char * filename, 
		     const int append, const int prec) {
  int err = 0;

  write_propagator_format(filename, prec, 1);
#ifdef HAVE_LIBLEMON
  err = write_lemon_spinor(s, filename, 1, prec);
#else
  err = write_lime_spinor(s, filename, 1, prec);
#endif
  return(err);
}  /* end of write_propagator */

#ifdef HAVE_LIBLEMON
int write_binary_spinor_data(double * const s, LemonWriter * writer, const int prec, DML_Checksum * ans) {
                                   
  int x, y, z, t, xG, yG, zG, tG, status = 0;
  unsigned int i = 0;
  int latticeSize[] = {T_global, g_nproc_x*LX, g_nproc_y*LY, g_nproc_z*LZ};
  int scidacMapping[] = {0, 3, 2, 1};
  unsigned long bufoffset = 0;
  char *filebuffer = NULL;
  uint64_t bytes;
  DML_SiteRank rank;
  double tick = 0, tock = 0;
  char measure[64];
  double *p = NULL;
  size_t sizeof_spinor = 24*sizeof(double);

  int words_bigendian = big_endian();


  DML_checksum_init(ans);
  bytes = (uint64_t)sizeof_spinor;
  if (prec == 32) {
    bytes /= 2;
  }
  if((void*)(filebuffer = (char*)malloc(VOLUME * bytes)) == NULL) {
    fprintf (stderr, "[write_binary_spinor_data] Error from malloc\n");
    fflush(stderr);
    return(1);
  }

  /* p = s always, s is ordered lexicographically */
  p = s;

  tG = g_proc_coords[0]*T;
  zG = g_proc_coords[3]*LZ;
  yG = g_proc_coords[2]*LY;
  xG = g_proc_coords[1]*LX;
  for(t = 0; t < T; t++) {
    for(z = 0; z < LZ; z++) {
      for(y = 0; y < LY; y++) {
        for(x = 0; x < LX; x++) {
          rank = (DML_SiteRank) ((((tG + t)*LZ_global + zG + z)*LY_global + yG + y)*LX_global + xG + x);
          i = 24 * g_ipt[t][x][y][z];
   
          if( !words_bigendian ) {

            if (prec == 32) {
              byte_swap_assign_double2single( (void*)(filebuffer + bufoffset), (void*)(p + i), 24);
            } else {
              byte_swap_assign( (void*)(filebuffer + bufoffset), (void*)(p+i), 24);
            }

          } else {

            if (prec == 32) {
              double2single( (void*)(filebuffer + bufoffset), (void*)(p + i), 24);
            } else {
              memcpy( (filebuffer + bufoffset), (p + i), sizeof_spinor );
            }

          }

          DML_checksum_accum(ans, rank, (char*) filebuffer + bufoffset, bytes);
          bufoffset += bytes;

        }
      }
    }
  }

  if (g_verbose > 0) {
    MPI_Barrier(g_cart_grid);
    tick = MPI_Wtime();
  }

  status = lemonWriteLatticeParallelMapped(writer, filebuffer, bytes, latticeSize, scidacMapping);

  if (status != LEMON_SUCCESS)
  {
    free(filebuffer);
    fprintf(stderr, "[write_binary_spinor_data] LEMON write error occurred with status = %d, while in write_binary_spinor_data!\n", status);
    return(-2);
  }

  if (g_verbose > 0) {
    MPI_Barrier(g_cart_grid);
    tock = MPI_Wtime();

    if (g_cart_id == 0) {
      fprintf(stdout, "# [write_binary_spinor_data] time spent writing %lu bytes was %e seconds\n",
          latticeSize[0] * latticeSize[1] * latticeSize[2] * latticeSize[3] * bytes, tock - tick);

      fprintf(stdout, "# [write_binary_spinor_data] global writing speed: %e bytes / second\n",
          latticeSize[0] * latticeSize[1] * latticeSize[2] * latticeSize[3] * bytes / (tock - tick));

      fprintf(stdout, "# [write_binary_spinor_data] local writing speed: %e bytes / second\n",
          latticeSize[0] * latticeSize[1] * latticeSize[2] * latticeSize[3] * bytes / ( g_nproc * (tock - tick) ));
      fflush(stdout);
    }
  }

  lemonWriterCloseRecord(writer);

  DML_global_xor( &(ans->suma) );
  DML_global_xor( &(ans->sumb) );

  free(filebuffer);
  return(0);
}
#else
int write_binary_spinor_data(double * const s, LimeWriter * limewriter,
                                      const int prec, DML_Checksum * ans) {

  int x, y, z, t, i=0, status=0;
  double tmp[24];
  float  tmp2[24];
  int iproc;
  n_uint64_t bytes;
  DML_SiteRank rank;
#ifdef HAVE_MPI
  int tgeom[2];
  double *buffer;
  MPI_Status mstatus;
#endif
  int words_bigendian = big_endian();
  DML_checksum_init(ans);

  if(prec == 32) bytes = (n_uint64_t)24*sizeof(float);
  else bytes = (n_uint64_t)24*sizeof(double);
  if(g_cart_id==0) {
    for(t = 0; t < T; t++) {
      for(z = 0; z < LZ; z++) {
      for(y = 0; y < LY; y++) {
      for(x = 0; x < LX; x++) {
        /* Rank should be computed by proc 0 only */
        rank = (DML_SiteRank) ((( (t+Tstart)*LZ + z)*LY + y)*LX + x);
        i = 24 * g_ipt[t][x][y][z];
        if(!words_bigendian) {
          if(prec == 32) {
            byte_swap_assign_double2single((float*)tmp2, s + i, 24);
            DML_checksum_accum(ans,rank,(char *) tmp2,24*sizeof(float));
            status = limeWriteRecordData((void*)tmp2, &bytes, limewriter);
          }
          else {
            byte_swap_assign(tmp, s + i , 24);
            DML_checksum_accum(ans,rank,(char *) tmp,24*sizeof(double));
            status = limeWriteRecordData((void*)tmp, &bytes, limewriter);
          }
        }
        else {
          if(prec == 32) {
            double2single((float*)tmp2, (s + i), 24);
            DML_checksum_accum(ans,rank,(char *) tmp2,24*sizeof(float));
            status = limeWriteRecordData((void*)tmp2, &bytes, limewriter);
          }
          else {
            status = limeWriteRecordData((void*)(s + i), &bytes, limewriter);
            DML_checksum_accum(ans,rank,(char *) (s + i), 24*sizeof(double));
          }
        }
      }
      }
      }
    }
  }  /* of if g_cart_id == 0 */
#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  return(1);
#else
  tgeom[0] = Tstart;
  tgeom[1] = T;
  for(iproc=1; iproc<g_nproc; iproc++) {
    if(g_cart_id==0) {
        MPI_Recv((void*)tgeom, 2, MPI_INT, iproc, 100+iproc, g_cart_grid, &mstatus);
        fprintf(stdout, "# [write_binary_spinor_data] iproc = %d; Tstart = %d, T = %d\n", iproc, tgeom[0], tgeom[1]);     

        buffer = (double*)malloc(24*LX*LY*LZ*tgeom[1]*sizeof(double));
 
        if(buffer==(double*)NULL) {
          fprintf(stderr, "[write_binary_spinor_data] error using malloc for buffer\n Aborting...\n");
          MPI_Abort(MPI_COMM_WORLD, 1);
          MPI_Finalize();
          exit(500);
        }

        MPI_Recv((void*)buffer, 24*LX*LY*LZ*tgeom[1], MPI_DOUBLE, iproc, 200+iproc, g_cart_grid, &mstatus);

        for(t=0; t<tgeom[1]; t++) {
          for(z=0; z<LZ; z++) {
          for(y=0; y<LY; y++) {
          for(x=0; x<LX; x++) {
            rank = (DML_SiteRank) ((( (t+tgeom[0])*LZ + z)*LY + y)*LX + x);
            i = 24 * ( ((t*LX + x)*LY + y)*LZ + z );
            if(!words_bigendian) {
              if(prec == 32) {
                byte_swap_assign_double2single(tmp2, buffer + i, 24);
                DML_checksum_accum(ans,rank,(char *) tmp2,24*sizeof(float));
                status = limeWriteRecordData((void*)tmp2, &bytes, limewriter);
              }
              else {
                byte_swap_assign(tmp, buffer + i , 24);
                DML_checksum_accum(ans,rank,(char *) tmp,24*sizeof(double));
                status = limeWriteRecordData((void*)tmp, &bytes, limewriter);
              }
            } else {
              if(prec == 32) {
                double2single(tmp2, buffer+i, 24);
                DML_checksum_accum(ans,rank,(char *)tmp2, 24*sizeof(float));
                status = limeWriteRecordData((void*)tmp2, &bytes, limewriter);
              }
              else {
                status = limeWriteRecordData((void*)(buffer + i), &bytes, limewriter);
                DML_checksum_accum(ans,rank,(char *) (buffer + i), 24*sizeof(double));
              }
            }
          }
          }
          }
        }
        free(buffer);
    }

    if(g_cart_id==iproc) {
    
      MPI_Send((void*)tgeom, 2, MPI_INT, 0, 100+iproc, g_cart_grid);

      MPI_Send((void*) s, 24*LX*LY*LZ*T, MPI_DOUBLE, 0, 200+iproc, g_cart_grid);

    }
    MPI_Barrier(g_cart_grid);
    fprintf(stdout, " [write_binary_spinor_data %d] finished iproc = %d\n", g_cart_id, iproc);
  }
#endif  /* if PARALLELTX || PARALLELTXY || PARALLELTXYZ */
#endif


#ifdef HAVE_MPI
  MPI_Barrier(g_cart_grid);
#endif

  return(0);
}
#endif /* HAVE_LIBLEMON */

#ifdef HAVE_LIBLEMON
int read_binary_spinor_data(double * const s, LemonReader * reader, 
			    const int prec, DML_Checksum *checksum) {

  int t, x, y , z, i = 0, status = 0;
  int latticeSize[] = {T_global, LX_global, LY_global, LZ_global};
  int scidacMapping[] = {0, 3, 2, 1};
  n_uint64_t bytes;
  double *p = NULL;
  char *filebuffer = NULL, *current = NULL;
  double tick = 0, tock = 0;
  DML_SiteRank rank;
  uint64_t fbspin;
  char measure[64];
  int words_bigendian = big_endian();

  DML_checksum_init(checksum);

  fbspin = 24*sizeof(double);
  if (prec == 32) fbspin /= 2;
  bytes = fbspin;

  if((void*)(filebuffer = (char*)malloc(VOLUME * bytes)) == NULL) {
    fprintf (stderr, "[read_binary_spinor_data] malloc errno in read_binary_spinor_data_parallel\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(501);
  }

  status = lemonReadLatticeParallelMapped(reader, filebuffer, bytes, latticeSize, scidacMapping);

  if (status < 0 && status != LEMON_EOR) {
    fprintf(stderr, "[read_binary_spinor_data] LEMON read error occured with status = %d while reading!\nPanic! Aborting...\n", status);
    MPI_File_close(reader->fp);
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(502);
  }

  for (t = 0; t <  T; t++) {
  for (z = 0; z < LZ; z++) {
  for (y = 0; y < LY; y++) {
  for (x = 0; x < LX; x++) {
    rank = (DML_SiteRank)( LXstart + (((Tstart  + t) * (LZ*g_nproc_z) + LZstart + z) * (LY*g_nproc_y) + LYstart + y) * ((DML_SiteRank) LX * g_nproc_x) + x);
    current = filebuffer + bytes * (x + (y + (t * LZ + z) * LY) * LX);
    DML_checksum_accum(checksum, rank, current, bytes);

    i = g_ipt[t][x][y][z];
    p = s + _GSI(i);
    if(!words_bigendian) {
      if (prec == 32)
        byte_swap_assign_single2double(p, current, 24*sizeof(double) / 8);
      else
        byte_swap_assign(p, current, 24*sizeof(double) / 8);
    } else {
      if (prec == 32)
        single2double(p, current, 24*sizeof(double) / 8);
      else
        memcpy(p, current, 24*sizeof(double));
    }
  }}}}

  DML_global_xor(&checksum->suma);
  DML_global_xor(&checksum->sumb);

  free(filebuffer);
  return(0);
}
#else 
int read_binary_spinor_data(double * const s, LimeReader * limereader, 
			    const int prec, DML_Checksum *ans) {

  int status=0;
  n_uint64_t bytes, ix;
  double tmp[24];
  DML_SiteRank rank;
  float tmp2[24];
  int words_bigendian;
  unsigned int t, x, y, z;
  words_bigendian = big_endian();

  DML_checksum_init(ans);
  rank = (DML_SiteRank) 0;
  
  if(prec == 32) bytes = 24*sizeof(float);
  else bytes = 24*sizeof(double);
  for(t = 0; t < T; t++){
    for(z = 0; z < LZ; z++){
      for(y = 0; y < LY; y++){
#if (defined HAVE_MPI)
      limeReaderSeek(limereader,(n_uint64_t) ( (((Tstart+t)*(LZ*g_nproc_z) + LZstart + z)*(LY*g_nproc_y)+LYstart+y)*(LX*g_nproc_x) +LXstart )*bytes, SEEK_SET);
#endif
	for(x = 0; x < LX; x++){
	  ix = g_ipt[t][x][y][z]*(n_uint64_t)12;
	  rank = (DML_SiteRank) ((((Tstart+t)*(LZ*g_nproc_z)+LZstart + z)*(LY*g_nproc_y) + LYstart + y)*(DML_SiteRank)(LX*g_nproc_x) + LXstart + x);
	  if(prec == 32) {
	    status = limeReaderReadData(tmp2, &bytes, limereader);
	    DML_checksum_accum(ans,rank,(char *) tmp2, bytes);	    
	  }
	  else {
	    status = limeReaderReadData(tmp, &bytes, limereader);
	    DML_checksum_accum(ans,rank,(char *) tmp, bytes);
	  }
	  if(!words_bigendian) {
	    if(prec == 32) {
	      byte_swap_assign_single2double(&s[2*ix], (float*)tmp2, 24);
	    }
	    else {
	      byte_swap_assign(&s[2*ix], tmp, 24);
	    }
	  }
	  else {
	    if(prec == 32) {
	      single2double(&s[2*ix], (float*)tmp2, 24);
	    }
	    else memcpy(&s[2*ix], tmp, bytes);
	  }
	  if(status < 0 && status != LIME_EOR) {
	    return(-1);
	  }
	}
      }
    }
  }
#ifdef HAVE_MPI
  DML_checksum_combine(ans);
#endif
  if(g_cart_id == 0) printf("# [read_binary_spinor_data] The final checksum is %#lx %#lx\n", (*ans).suma, (*ans).sumb);
  return(0);
}
#endif /* HAVE_LIBLEMON */

/************************************************************
 *
 ************************************************************/
int write_checksum(char * filename, DML_Checksum * checksum) {
  FILE * ofs = NULL;
  LimeWriter * limewriter = NULL;
  LimeRecordHeader * limeheader = NULL;
  int status = 0;
  int ME_flag=0, MB_flag=1;
  char message[500];
  n_uint64_t bytes;
  /*   char * message; */


  if(g_cart_id == 0) {
    ofs = fopen(filename, "a");

    if(ofs == (FILE*)NULL) {
      fprintf(stderr, "[write_checksum] Could not open file %s for writing!\n Aborting...\n", filename);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }
    limewriter = limeCreateWriter( ofs );
    if(limewriter == (LimeWriter*)NULL) {
      fprintf(stderr, "[write_checksum] LIME error in file %s for writing!\n Aborting...\n", filename);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }

    sprintf(message, "<?xml version=\"1.0\" encoding=\"UTF-8\"?><scidacChecksum><version>1.0</version><suma>%#010x</suma><sumb>%#010x</sumb></scidacChecksum>", (*checksum).suma, (*checksum).sumb);
    bytes = strlen( message );
    limeheader = limeCreateHeader(MB_flag, ME_flag, "scidac-checksum", bytes);
    status = limeWriteRecordHeader( limeheader, limewriter);
    if(status < 0 ) {
      fprintf(stderr, "[write_checksum] LIME write header error %d\n", status);
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
#ifdef HAVE_MPI
  MPI_Barrier(g_cart_grid);
#endif
  return(0);
}


int write_propagator_type(const int type, char * filename) {

  FILE * ofs = NULL;
  LimeWriter * limewriter = NULL;
  LimeRecordHeader * limeheader = NULL;
  int status = 0;
  int ME_flag=1, MB_flag=1;
  char message[500];
  n_uint64_t bytes;

  ofs = fopen(filename, "w");
  
  if(ofs == (FILE*)NULL) {
    fprintf(stderr, "[write_propagator_type] Could not open file %s for writing!\n Aboring...\n", filename);
    exit(500);
  }
  limewriter = limeCreateWriter( ofs );
  if(limewriter == (LimeWriter*)NULL) {
    fprintf(stderr, "[write_propagator_type] LIME error in file %s for writing!\n Aborting...\n", filename);
    exit(500);
  }
  
  if(type == 0) {
    sprintf(message,"DiracFermion_Sink");
    bytes = strlen( message );
  }
  else if (type == 1) {
    sprintf(message,"DiracFermion_Source_Sink_Pairs");
    bytes = strlen( message );
  }
  else if (type == 2) {
    sprintf(message,"DiracFermion_ScalarSource_TwelveSink");
    bytes = strlen( message );
  }
  else if (type == 3) {
    sprintf(message,"DiracFermion_ScalarSource_FourSink");
    bytes = strlen( message );
  }
  
  limeheader = limeCreateHeader(MB_flag, ME_flag, "etmc-propagator-type", bytes);
  status = limeWriteRecordHeader( limeheader, limewriter);
  if(status < 0 ) {
    fprintf(stderr, "[write_propagator_type] LIME write header error %d\n", status);
    exit(500);
  }
  limeDestroyHeader( limeheader );
  limeWriteRecordData(message, &bytes, limewriter);
  
  limeDestroyWriter( limewriter );
  fflush(ofs);
  fclose(ofs);
  return(0);
}

int write_propagator_format(char * filename, const int prec, const int no_flavours) {
  FILE * ofs = NULL;
  LimeWriter * limewriter = NULL;
  LimeRecordHeader * limeheader = NULL;
  int status = 0;
  int ME_flag=0, MB_flag=1;
  char message[500];
  n_uint64_t bytes;
  /*   char * message; */

  if(g_cart_id==0) {
/*    ofs = fopen(filename, "a"); */
    ofs = fopen(filename, "w");
  
    if(ofs == (FILE*)NULL) {
      fprintf(stderr, "[write_propagator_format] Could not open file %s for writing!\n Aborting...\n", filename);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }
    limewriter = limeCreateWriter( ofs );
    if(limewriter == (LimeWriter*)NULL) {
      fprintf(stderr, "[write_propagator_format] LIME error in file %s for writing!\n Aborting...\n", filename);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }
  
    sprintf(message, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<etmcFormat>\n<field>diracFermion</field>\n<precision>%d</precision>\n<flavours>%d</flavours>\n<lx>%d</lx>\n<ly>%d</ly>\n<lz>%d</lz>\n<lt>%d</lt>\n</etmcFormat>", prec, no_flavours, LX_global, LY_global, LZ_global, T_global);
    bytes = strlen( message );
    limeheader = limeCreateHeader(MB_flag, ME_flag, "etmc-propagator-format", bytes);
    status = limeWriteRecordHeader( limeheader, limewriter);
    if(status < 0 ) {
      fprintf(stderr, "[write_propagator_format] LIME write header error %d\n", status);
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


#ifdef HAVE_LIBLEMON
int write_lemon_spinor(double * const s, char * filename, const int append, const int prec) {

  MPI_File * ofs = NULL;
  LemonWriter * writer = NULL;
  LemonRecordHeader * header = NULL;
  int status = 0;
  int ME_flag=0, MB_flag=0;

  LEMON_OFFSET_TYPE bytes;

  DML_Checksum checksum;
  char *message;

  if(g_cart_id == 0) fprintf(stdout, "\n# [write_lemon_spinor] constructing lemon writer for file %s\n", filename);

  ofs = (MPI_File*)malloc(sizeof(MPI_File));
  status = MPI_File_open(g_cart_grid, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, ofs);
  if(status == MPI_SUCCESS) status = MPI_File_set_size(*ofs, 0);
  status = (status == MPI_SUCCESS) ? 0 : 1;
  writer = lemonCreateWriter(ofs, g_cart_grid);
  status = status || (writer == NULL);

  if(status) {
    fprintf(stderr, "[write_lemon_spinor] Error, could not open file for writing\n");
    MPI_Abort(MPI_COMM_WORLD, 120);
    MPI_Finalize();
    exit(120);
  }

  // format message
  message = (char*)malloc(2048);
  sprintf(message, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<cvcFormat>\n<precision>%d</precision>\n"\
      "<lx>%d</lx>\n<ly>%d</ly>\n<lz>%d</lz>\n<lt>%d</lt>\n</cvcFormat>\ndate %s", prec, LX*g_nproc_x, LY*g_nproc_y, LZ*g_nproc_z, T_global, ctime(&g_the_time));
  bytes = strlen(message);
  MB_flag=1, ME_flag=1;
  header = lemonCreateHeader(MB_flag, ME_flag, "cvc_contraction-format", bytes);
  status = lemonWriteRecordHeader(header, writer);
  lemonDestroyHeader(header);
  lemonWriteRecordData( message, &bytes, writer);
  lemonWriterCloseRecord(writer);
  free(message);

  // binary data message
  bytes = (n_uint64_t)LX_global * LY_global * LZ_global * T_global * (n_uint64_t) (24*sizeof(double) * prec / 64);
  MB_flag=1, ME_flag=0;
  header = lemonCreateHeader(MB_flag, ME_flag, "scidac-binary-data", bytes);
  status = lemonWriteRecordHeader(header, writer);
  lemonDestroyHeader(header);

  status = write_binary_spinor_data(s, writer, prec, &checksum);
  if(status != 0) {
    if(g_cart_id == 0) fprintf(stderr, "[write_lemon_spinor] Error from write_binary_spinor_data; status was %d\n", status);
    EXIT(10);
  }

  if(g_cart_id==0) {
    printf("# [write_lemon_spinor] Final check sum is (%#x  %#x)\n", checksum.suma, checksum.sumb);
  }

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
  MPI_File_close(ofs);
  free(ofs);
  return(0);
}
#else
int write_lime_spinor(double * const s, char * filename, 
		      const int append, const int prec) {

  FILE * ofs = NULL;
  LimeWriter * limewriter = NULL;
  LimeRecordHeader * limeheader = NULL;
  int status = 0;
  int ME_flag=0, MB_flag=0;
  n_uint64_t bytes;
  DML_Checksum checksum;

  if(g_cart_id==0) {
    if(append) {
      ofs = fopen(filename, "a");
    }
    else {
      ofs = fopen(filename, "w");
    }
    if(ofs == (FILE*)NULL) {
      fprintf(stderr, "[write_lime_spinor] Could not open file %s for writing!\n Aborting...\n", filename);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }
  
    limewriter = limeCreateWriter( ofs );
    if(limewriter == (LimeWriter*)NULL) {
      fprintf(stderr, "[write_lime_spinor] LIME error in file %s for writing!\n Aborting...\n", filename);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }

    bytes = (LX*g_nproc_x)*LY*LZ*T_global*(n_uint64_t)24*sizeof(double)*prec/64;
    MB_flag=0; ME_flag=1;
    limeheader = limeCreateHeader(MB_flag, ME_flag, "scidac-binary-data", bytes);
    status = limeWriteRecordHeader( limeheader, limewriter);
    if(status < 0 ) {
      fprintf(stderr, "[write_lime_spinor] LIME write header (scidac-binary-data) error %d\n", status);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }
    limeDestroyHeader( limeheader );
  }
  
  status = write_binary_spinor_data(s, limewriter, prec, &checksum);
  if(g_cart_id==0) {
    printf("# [write_lime_spinor] Final check sum is (%#lx  %#lx)\n", checksum.suma, checksum.sumb);
    if(ferror(ofs)) {
      fprintf(stderr, "[write_lime_spinor] Warning! Error while writing to file %s \n", filename);
    }
    limeDestroyWriter( limewriter );
    fflush(ofs);
    fclose(ofs);
  }
  write_checksum(filename, &checksum);
  return(0);
}
#endif  // of if HAVE_LIBLEMON 

int get_propagator_type(char * filename) {
  FILE * ifs;
  int status=0, ret=-1;
  n_uint64_t bytes;
  char * tmp;
  LimeReader * limereader;
  
  if((ifs = fopen(filename, "r")) == (FILE*)NULL) {
    fprintf(stderr, "[get_propagator_type] Error opening file %s\n", filename);
    return(ret);
  }
  
  limereader = limeCreateReader( ifs );
  if( limereader == (LimeReader *)NULL ) {
    fprintf(stderr, "[get_propagator_type] Unable to open LimeReader\n");
    return(ret);
  }
  while( (status = limeReaderNextRecord(limereader)) != LIME_EOF ) {
    if(status != LIME_SUCCESS ) {
      fprintf(stderr, "[get_propagator_type] limeReaderNextRecord returned error with status = %d!\n", status);
      status = LIME_EOF;
      break;
    }
    if(strcmp("etmc-propagator-type", limeReaderType(limereader)) == 0) break;
  }
  if(status == LIME_EOF) {
    fprintf(stderr, "[get_propagator_type] no etmc-propagator-type record found in file %s\n",filename);
    limeDestroyReader(limereader);
    fclose(ifs);
    return(ret);
  }
  tmp = (char*) calloc(500, sizeof(char));
  bytes = limeReaderBytes(limereader);
  status = limeReaderReadData(tmp, &bytes, limereader);
  limeDestroyReader(limereader);
  fclose(ifs);
  if(strcmp("DiracFermion_Sink", tmp) == 0) ret = 0;
  else if(strcmp("DiracFermion_Source_Sink_Pairs", tmp) == 0) ret = 1;
  else if(strcmp("DiracFermion_ScalarSource_TwelveSink", tmp) == 0) ret = 2;
  else if(strcmp("DiracFermion_ScalarSource_FourSink", tmp) == 0) ret = 3;
  free(tmp);
  return(ret);
}

#ifdef HAVE_LIBLEMON
int read_lime_spinor(double * const s, char * filename, const int position) {
  MPI_File *ifs;
  int status = 0, getpos = 0, prec = 0, prop_type;
  char *header_type = NULL;
  LemonReader *reader = NULL;
  DML_Checksum checksum;
  n_uint64_t bytes = 0;

  if(g_cart_id==0)
    fprintf(stdout, "# [read_lime_spinor] reading prop in LEMON format from file %s at pos %d\n", filename, position);

  ifs = (MPI_File*)malloc(sizeof(MPI_File));
  status = MPI_File_open(g_cart_grid, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, ifs);
  status = (status == MPI_SUCCESS) ? 0 : 1;
  if(status) {
    fprintf(stderr, "[read_lime_spinor] Err, could not open file for reading\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(500);
  }
  
  if( (reader = lemonCreateReader(ifs, g_cart_grid))==NULL ) {
    fprintf(stderr, "[read_lime_spinor] Error, could not create lemon reader.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(502);
  }

  while ((status = lemonReaderNextRecord(reader)) != LIME_EOF) {
    if (status != LIME_SUCCESS) {
      fprintf(stderr, "[read_lime_spinor] lemonReaderNextRecord returned status %d.\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = (char*)lemonReaderType(reader);
    if (strcmp("scidac-binary-data", header_type) == 0) {
      if (getpos == position)
        break;
      else
        ++getpos;
    }
  }

  if (status == LIME_EOF) {
    fprintf(stderr, "[read_lime_spinor] Error, no scidac-binary-data record found in file.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(500);
  } 

  bytes = lemonReaderBytes(reader);
  if (bytes == (n_uint64_t)LX * g_nproc_x * LY * g_nproc_y * LZ * g_nproc_z * T_global * 24*sizeof(double)) {
    prec = 64;
  } else {
    if (bytes == (n_uint64_t)LX * g_nproc_x * LY * g_nproc_y * LZ * g_nproc_z * T_global * 24* sizeof(double) / 2) {
      prec = 32;
    } else {
      if(g_cart_id==0) fprintf(stderr, "[read_lime_spinor] Error, wrong length in spinor. Aborting read!\n");
       MPI_Abort(MPI_COMM_WORLD, 1);
       MPI_Finalize();
       exit(501);
    }
  }
  if(g_cart_id==0) fprintf(stdout, "# [read_lime_spinor] %d bit precision read.\n", prec);

  read_binary_spinor_data(s, reader, prec, &checksum);

  if (g_cart_id == 0) fprintf(stdout, "# [read_lime_spinor] checksum for DiracFermion field in file %s position %d is %#x %#x\n", 
    filename, position, checksum.suma, checksum.sumb);

  lemonDestroyReader(reader);
  MPI_File_close(ifs);
  free(ifs);
  
  return(0);
}
#else
int read_lime_spinor(double * const s, char * filename, const int position) {
  FILE * ifs;
  int status=0, getpos=-1;
  n_uint64_t bytes;
  char * header_type;
  LimeReader * limereader;
  n_uint64_t prec = 32;
  DML_Checksum checksum;
  
  if((ifs = fopen(filename, "r")) == (FILE*)NULL) {
    fprintf(stderr, "[read_lime_spinor] Error opening file %s\n", filename);
    return(-1);
  }
  if(g_proc_id==0) fprintf(stdout, "# [read_lime_spinor] Reading Dirac-fermion field in LIME format from %s\n", filename);

  limereader = limeCreateReader( ifs );
  if( limereader == (LimeReader *)NULL ) {
    fprintf(stderr, "[read_lime_spinor] Unable to open LimeReader\n");
    return(-1);
  }
  while( (status = limeReaderNextRecord(limereader)) != LIME_EOF ) {
    if(status != LIME_SUCCESS ) {
      fprintf(stderr, "[read_lime_spinor] limeReaderNextRecord returned error with status = %d!\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = limeReaderType(limereader);
    if(strcmp("scidac-binary-data",header_type) == 0) getpos++;
    if(getpos == position) break;
  }
  if(status == LIME_EOF) {
    fprintf(stderr, "[read_lime_spinor] no scidac-binary-data record found in file %s\n",filename);
    limeDestroyReader(limereader);
    fclose(ifs);
    if(g_proc_id==0) fprintf(stderr, "[read_lime_spinor] try to read in CMI format\n");
    return(read_cmi(s, filename));
  }
  bytes = limeReaderBytes(limereader);
  if(bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(24*sizeof(double))) prec = 64;
  else if(bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(24*sizeof(float))) prec = 32;
  else {
    fprintf(stderr, "[read_lime_spinor] wrong length in eospinor: bytes = %llu, not %llu. Aborting read!\n", 
	    bytes, (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(24*sizeof(double)));
    return(-1);
  }
  if(g_cart_id == 0) printf("# [read_lime_spinor] %llu Bit precision read\n", prec);

  status = read_binary_spinor_data(s, limereader, prec, &checksum);

  if(status < 0) {
    fprintf(stderr, "[read_lime_spinor] LIME read error occured with status = %d while reading file %s!\n Aborting...\n", 
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
}
#endif  /* HAVE_LIBLEMON */

/************************************************************************************/
/************************************************************************************/

#ifdef HAVE_LIBLEMON
/************************************************************************************
 * read lemon binary propagator
 ************************************************************************************/
int read_binary_propagator_data(double * const s, LemonReader * reader, const int prec, DML_Checksum *checksum) {

  n_uint64_t const real_block_size = 288;

  int  status = 0;
  int latticeSize[] = {T_global, LX_global, LY_global, LZ_global};
  int scidacMapping[] = {0, 3, 2, 1};
  n_uint64_t bytes;
  double *p = NULL;
  char *filebuffer = NULL, *current = NULL;
  double tick = 0, tock = 0;
  DML_SiteRank rank;
  uint64_t fbspin;
  char measure[64];
  int words_bigendian = big_endian();

  DML_checksum_init(checksum);

  fbspin = real_block_size * sizeof(double);
  if (prec == 32) fbspin /= 2;
  bytes = fbspin;

  if((void*)(filebuffer = (char*)malloc(VOLUME * bytes)) == NULL) {
    fprintf (stderr, "[read_binary_propagator_data] malloc errno in read_binary_spinor_data_parallel %s %d\n", __FILE__, __LINE__);
    EXIT(501);
  }

  status = lemonReadLatticeParallelMapped(reader, filebuffer, bytes, latticeSize, scidacMapping);

  if (status < 0 && status != LEMON_EOR) {
    fprintf(stderr, "[read_binary_propagator_data] LEMON read error occured with status = %d while reading!\nPanic! Aborting... %s %d\n",
        status, __FILE__, __LINE__);
    MPI_File_close(reader->fp);
    EXIT(502);
  }

  for ( int t = 0; t <  T; t++) {
  for ( int z = 0; z < LZ; z++) {
  for ( int y = 0; y < LY; y++) {
  for ( int x = 0; x < LX; x++) {
    rank = (DML_SiteRank)( LXstart + (((Tstart  + t) * (LZ*g_nproc_z) + LZstart + z) * (LY*g_nproc_y) + LYstart + y) * ((DML_SiteRank) LX * g_nproc_x) + x);
    current = filebuffer + bytes * (x + (y + (t * LZ + z) * LY) * LX);
    DML_checksum_accum(checksum, rank, current, bytes);

    unsigned int const i = g_ipt[t][x][y][z];
    p = s + real_block_size * i;
    if(!words_bigendian) {
      if (prec == 32)
        byte_swap_assign_single2double(p, current, real_block_size * sizeof(double) / 8);
      else
        byte_swap_assign(p, current, real_block_size * sizeof(double) / 8);
    } else {
      if (prec == 32)
        single2double(p, current, real_block_size * sizeof(double) / 8);
      else
        memcpy(p, current, real_block_size * sizeof(double));
    }
  }}}}

  DML_global_xor(&checksum->suma);
  DML_global_xor(&checksum->sumb);

  free(filebuffer);
  return(0);
}  /* end of read_binary_propagator_data */

/************************************************************************************/
/************************************************************************************/

/************************************************************************************
 * read propagator using lemon
 ************************************************************************************/
int read_lime_propagator(double * const s, char * filename, const int position) {

  n_uint64_t const real_block_size = 288;

  MPI_File *ifs;
  int status = 0, getpos = 0, prec = 0, prop_type;
  char *header_type = NULL;
  LemonReader *reader = NULL;
  DML_Checksum checksum;
  n_uint64_t bytes = 0;

  if(g_cart_id==0)
    fprintf(stdout, "# [read_lime_propagator] reading prop in LEMON format from file %s at pos %d\n", filename, position);

  ifs = (MPI_File*)malloc(sizeof(MPI_File));
  status = MPI_File_open(g_cart_grid, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, ifs);
  status = (status == MPI_SUCCESS) ? 0 : 1;
  if(status) {
    fprintf(stderr, "[read_lime_propagator] Err, could not open file for reading\n");
    EXIT(500);
  }
  
  if( (reader = lemonCreateReader(ifs, g_cart_grid))==NULL ) {
    fprintf(stderr, "[read_lime_propagator] Error, could not create lemon reader.\n");
    EXIT(502);
  }

  while ((status = lemonReaderNextRecord(reader)) != LIME_EOF) {
    if (status != LIME_SUCCESS) {
      fprintf(stderr, "[read_lime_propagator] lemonReaderNextRecord returned status %d.\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = (char*)lemonReaderType(reader);
    if (strcmp("scidac-binary-data", header_type) == 0) {
      if (getpos == position)
        break;
      else
        ++getpos;
    }
  }

  if (status == LIME_EOF) {
    fprintf(stderr, "[read_lime_propagator] Error, no scidac-binary-data record found in file.\n");
    EXIT(500);
  } 

  bytes = lemonReaderBytes(reader);
  if (bytes == (n_uint64_t)LX * g_nproc_x * LY * g_nproc_y * LZ * g_nproc_z * T_global * real_block_size * sizeof(double)) {
    prec = 64;
  } else {
    if (bytes == (n_uint64_t)LX * g_nproc_x * LY * g_nproc_y * LZ * g_nproc_z * T_global * real_block_size * sizeof(double) / 2) {
      prec = 32;
    } else {
      if(g_cart_id==0) fprintf(stderr, "[read_lime_propagator] Error, wrong length in spinor. Aborting read!\n");
       EXIT(501);
    }
  }
  if(g_cart_id==0) fprintf(stdout, "# [read_lime_propagator] %d bit precision read.\n", prec);

  read_binary_propagator_data(s, reader, prec, &checksum);

  if (g_cart_id == 0) fprintf(stdout, "# [read_lime_propagator] checksum for DiracFermion field in file %s position %d is %#x %#x\n", 
    filename, position, checksum.suma, checksum.sumb);

  lemonDestroyReader(reader);
  MPI_File_close(ifs);
  free(ifs);
  
  return(0);
}  /* read_lime_propagator */

#else  /* ! LEMON but LIME */
/************************************************************************************
 * read lime binary propagator
 ************************************************************************************/
int read_binary_propagator_data(double * const s, LimeReader * limereader, const int prec, DML_Checksum *ans) {

  n_uint64_t const real_block_size = 288;

  int status=0;
  n_uint64_t bytes, ix;
  double tmp[real_block_size];
  float  tmp2[real_block_size];
  DML_SiteRank rank;
  int words_bigendian;
  int t, x, y, z;
  words_bigendian = big_endian();

  DML_checksum_init(ans);
  rank = (DML_SiteRank) 0;
  
  if ( prec == 32 ) bytes = real_block_size * sizeof(float);
  else              bytes = real_block_size * sizeof(double);
  for(t = 0; t < T; t++){
    for(z = 0; z < LZ; z++){
      for(y = 0; y < LY; y++){
#if (defined HAVE_MPI)
      limeReaderSeek(limereader,(n_uint64_t) ( (((Tstart+t)*(LZ*g_nproc_z) + LZstart + z)*(LY*g_nproc_y)+LYstart+y)*(LX*g_nproc_x) +LXstart )*bytes, SEEK_SET);
#endif
	for(x = 0; x < LX; x++){
	  ix = g_ipt[t][x][y][z] * real_block_size;
	  rank = (DML_SiteRank) ((((Tstart+t)*(LZ*g_nproc_z)+LZstart + z)*(LY*g_nproc_y) + LYstart + y)*(DML_SiteRank)(LX*g_nproc_x) + LXstart + x);
	  if(prec == 32) {
	    status = limeReaderReadData(tmp2, &bytes, limereader);
	    DML_checksum_accum(ans,rank,(char *) tmp2, bytes);	    
	  }
	  else {
	    status = limeReaderReadData(tmp, &bytes, limereader);
	    DML_checksum_accum(ans,rank,(char *) tmp, bytes);
	  }
	  if(!words_bigendian) {
	    if(prec == 32) {
	      byte_swap_assign_single2double(&s[ix], (float*)tmp2, (int)real_block_size);
	    }
	    else {
	      byte_swap_assign(&s[ix], tmp, (int)real_block_size);
	    }
	  }
	  else {
	    if(prec == 32) {
	      single2double(&s[ix], (float*)tmp2, (int)real_block_size );
	    }
	    else memcpy(&s[ix], tmp, bytes);
	  }
	  if(status < 0 && status != LIME_EOR) {
	    return(-1);
	  }
	}
      }
    }
  }
#ifdef HAVE_MPI
  DML_checksum_combine(ans);
#endif
  if(g_cart_id == 0) printf("# [read_binary_propagator_data] The final checksum is %#lx %#lx\n", (*ans).suma, (*ans).sumb);
  return(0);
}  /* end of read_binary_propagator_data */

/************************************************************************************/
/************************************************************************************/

/************************************************************************************
 * read lime propagator
 ************************************************************************************/
int read_lime_propagator(double * const s, char * filename, const int position) {

  uint64_t const real_block_size = 288;

  FILE * ifs;
  int status=0, getpos=-1;
  n_uint64_t bytes;
  char * header_type;
  LimeReader * limereader;
  n_uint64_t prec = 32;
  DML_Checksum checksum;
  
  if((ifs = fopen(filename, "r")) == (FILE*)NULL) {
    fprintf(stderr, "[read_lime_propagator] Error opening file %s\n", filename);
    return(-1);
  }
  if(g_proc_id==0) fprintf(stdout, "# [read_lime_propagator] Reading Dirac-fermion field in LIME format from %s\n", filename);

  limereader = limeCreateReader( ifs );
  if( limereader == (LimeReader *)NULL ) {
    fprintf(stderr, "[read_lime_propagator] Unable to open LimeReader\n");
    return(-1);
  }
  while( (status = limeReaderNextRecord(limereader)) != LIME_EOF ) {
    if(status != LIME_SUCCESS ) {
      fprintf(stderr, "[read_lime_propagator] limeReaderNextRecord returned error with status = %d!\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = limeReaderType(limereader);
    if(strcmp("scidac-binary-data",header_type) == 0) getpos++;
    if(getpos == position) break;
  }
  if(status == LIME_EOF) {
    fprintf(stderr, "[read_lime_propagator] no scidac-binary-data record found in file %s\n",filename);
    limeDestroyReader(limereader);
    fclose(ifs);
    return(-2);
  }
  bytes = limeReaderBytes(limereader);
  if(bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(real_block_size*sizeof(double))) prec = 64;
  else if(bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(real_block_size*sizeof(float))) prec = 32;
  else {
    fprintf(stderr, "[read_lime_propagator] wrong length in eoprop: bytes = %lu, not %lu. Aborting read!\n", 
	    bytes, (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(real_block_size*sizeof(double)));
    return(-1);
  }
  if(g_cart_id == 0) printf("# [read_lime_propagator] %llu Bit precision read\n", prec);

  status = read_binary_propagator_data(s, limereader, prec, &checksum);

  if(status < 0) {
    fprintf(stderr, "[read_lime_propagator] LIME read error occured with status = %d while reading file %s!\n Aborting...\n", 
	    status, filename);
    EXIT(500);
  }

  limeDestroyReader(limereader);
  fclose(ifs);
  return(0);
}  /* end of read_lime_propagator  */
#endif  /* HAVE_LIBLEMON */


/**************************************************
 * read propagator in CMI format
 *
 * - What about the phase factor?
 *
 **************************************************/
int read_cmi(double *v, const char * filename) {

  int x, y, z, t, ix, idx, i;
  FILE * ifs;
  float tmp[24];
  double _2_kappa;
  ifs = fopen(filename, "r");
  if(ifs==(FILE*)NULL) {
    fprintf(stderr, "[read_cmi] could not open file %s for reading\n", filename);
    return(-1);
  }
  if(g_cart_id==0) fprintf(stdout, "[read_cmi] Reading Dirac-fermion field in CMI format from %s\n", filename);
  _2_kappa = 2.0 * g_kappa;

  for(x = 0; x < LX; x++) {
  for(y = 0; y < LY; y++) {
  for(z = 0; z < LZ; z++) {
#ifdef HAVE_MPI
    fseek(ifs, (Tstart + (( (x + LXstart)*(LY*g_nproc_y) + LYstart + y) * (LZ*g_nproc_z) + LZstart + z )*T_global) * 24*sizeof(float), SEEK_SET);
#endif
    for(t = 0; t < T; t++) {
      ix = (t*LX*LY*LZ + x*LY*LZ + y*LZ + z)*12;
      fread(tmp, 24*sizeof(float), 1, ifs);

#ifdef BYTE_SWAP
/* 
      fprintf(stderr, "before byte swap: %f\n", tmp[0]);
*/
      byte_swap(tmp, 24*sizeof(float) / 4);
/*
      fprintf(stderr, "after byte swap: %f\n", tmp[0]);
*/
#endif

      for(i = 0; i < 12; i++) {
        idx = (ix+i)*2;
        v[idx  ] = _2_kappa * tmp[2*i  ];
        v[idx+1] = _2_kappa * tmp[2*i+1];
      }
    }
  }
  }
  }

  fclose(ifs);
/*
  if(g_rotate_EMTC_UKQCD) {
    if(g_cart_id==0) fprintf(stdout, "# [read_cmi] rotating propagator UKQCD -> ETMC\n");
    rotate_propagator_ETMC_UKQCD(v, VOLUME);
  }
*/
  return(0);
}

/************************************************************
 *
 ************************************************************/
int write_binary_spinor_data_timeslice(double * const s, LimeWriter * limewriter,
  const int prec, int timeslice, DML_Checksum * ans) {
#ifndef HAVE_MPI
  int x, y, z, t, i=0, status=0;
  double tmp[24];
  float  tmp2[24];
  n_uint64_t bytes;
  DML_SiteRank rank;
  int words_bigendian = big_endian();

  if(timeslice==0) {
    fprintf(stdout, "# [write_binary_spinor_data_timeslice] initializing checksum for timeslice %d\n", timeslice);
    DML_checksum_init(ans);
  }

  if(prec == 32) bytes = (n_uint64_t)24*sizeof(float);
  else bytes = (n_uint64_t)24*sizeof(double);
/*
  if(timeslice>0) {
    limeWriterSeek(limewriter, (n_uint64_t)0, SEEK_SET);
  }
*/
  for(z = 0; z < LZ; z++) {
  for(y = 0; y < LY; y++) {
  for(x = 0; x < LX; x++) {
    rank = (DML_SiteRank) ((( timeslice*LZ + z)*LY + y)*LX + x);
    i = 24 * g_ipt[0][x][y][z];
/* #ifndef WORDS_BIGENDIAN */
    if(!words_bigendian) {
      if(prec == 32) {
        byte_swap_assign_double2single((float*)tmp2, s + i, 24);
        DML_checksum_accum(ans,rank,(char *) tmp2,24*sizeof(float));
        status = limeWriteRecordData((void*)tmp2, &bytes, limewriter);
      }
      else {
        byte_swap_assign(tmp, s + i , 24);
        DML_checksum_accum(ans,rank,(char *) tmp,24*sizeof(double));
        status = limeWriteRecordData((void*)tmp, &bytes, limewriter);
      }
    } else {
/* #else */
      if(prec == 32) {
        double2single((float*)tmp2, (s + i), 24);
        DML_checksum_accum(ans,rank,(char *) tmp2,24*sizeof(float));
        status = limeWriteRecordData((void*)tmp2, &bytes, limewriter);
      }
      else {
        status = limeWriteRecordData((void*)(s + i), &bytes, limewriter);
        DML_checksum_accum(ans,rank,(char *) (s + i), 24*sizeof(double));
      }
    }
/* #endif */
  }}}
  return(0);
#else
  return(-1);
#endif
}

/****************************************************************
 *
 ****************************************************************/
int write_lime_spinor_timeslice(double * const s, char * filename, 
   const int prec, int timeslice, DML_Checksum *checksum) {
#ifndef HAVE_MPI
  FILE * ofs = NULL;
  LimeWriter * limewriter = NULL;
  LimeRecordHeader * limeheader = NULL;
  int status = 0;
  int ME_flag=0, MB_flag=0;
  n_uint64_t bytes;

  if(timeslice==0) {
    write_source_type(0, filename);
  }

  ofs = fopen(filename, "a");
  if(ofs == (FILE*)NULL) {
    fprintf(stderr, "[write_lime_spinor_timeslice] Could not open file %s for writing!\n Aborting...\n", filename);
    exit(500);
  }
  
  limewriter = limeCreateWriter( ofs );
  if(limewriter == (LimeWriter*)NULL) {
    fprintf(stderr, "[write_lime_spinor_timeslice] LIME error in file %s for writing!\n Aborting...\n", filename);
    exit(500);
  }

  bytes = LX*LY*LZ*T_global*(n_uint64_t)24*sizeof(double)*prec/64;
  if(timeslice==0) {
    MB_flag=0; ME_flag=1;
    limeheader = limeCreateHeader(MB_flag, ME_flag, "scidac-binary-data", bytes);
    status = limeWriteRecordHeader( limeheader, limewriter);
    if(status < 0 ) {
      fprintf(stderr, "[write_lime_spinor_timeslice] LIME write header (scidac-binary-data) error %d\n", status);
      exit(500);
    }
    limeDestroyHeader( limeheader );
  }
/*************************************/
  else {
    limewriter->first_record = 0;
    limewriter->last_written = 0;
    limewriter->header_nextP = 0;
    limewriter->bytes_total  = bytes; 
    limewriter->bytes_left   = LX*LY*LZ*(T_global-timeslice)*(n_uint64_t)24*sizeof(double)*prec/64;
    limewriter->rec_ptr      = 0;
    limewriter->rec_start    = 0;
    limewriter->bytes_pad    = 0;
    limewriter->isLastP      = 0;
  }
  if(timeslice==T_global-1) limewriter->isLastP = 1;
/*************************************/
/*
  fprintf(stdout, "\n# [write_lime_spinor_timeslice] ========================================\n");
  fprintf(stdout, "# [write_lime_spinor_timeslice] info on the limewriter:\n");
  fprintf(stdout, "# [write_lime_spinor_timeslice] first record   = %d\n", limewriter->first_record);
  fprintf(stdout, "# [write_lime_spinor_timeslice] last written   = %d\n", limewriter->last_written);
  fprintf(stdout, "# [write_lime_spinor_timeslice] write h/d as next = %d\n", limewriter->header_nextP);
  fprintf(stdout, "# [write_lime_spinor_timeslice] bytes total    = %llu\n", limewriter->bytes_total);
  fprintf(stdout, "# [write_lime_spinor_timeslice] bytes left     = %llu\n", limewriter->bytes_left);
  fprintf(stdout, "# [write_lime_spinor_timeslice] record pointer = %llu\n", limewriter->rec_ptr);
  fprintf(stdout, "# [write_lime_spinor_timeslice] pointer at start of record payload = %llu\n", limewriter->rec_start);
  fprintf(stdout, "# [write_lime_spinor_timeslice] bytes pad      = %d\n", limewriter->bytes_pad);
  fprintf(stdout, "# [write_lime_spinor_timeslice] last record in massage = %d\n", limewriter->isLastP);
  fprintf(stdout, "# [write_lime_spinor_timeslice] ========================================\n");
*/
  status = write_binary_spinor_data_timeslice(s, limewriter, prec, timeslice, checksum);
  if(ferror(ofs)) {
    fprintf(stderr, "[write_lime_spinor_timeslice] Warning! Error while writing to file %s \n", filename);
  }
  limeDestroyWriter( limewriter );
  fflush(ofs);
  fclose(ofs);
  if(timeslice==T_global-1) {
    printf("# [write_lime_spinor_timeslice] Final check sum for file %s is (%#lx  %#lx)\n", filename, (*checksum).suma, (*checksum).sumb);
    write_checksum(filename, checksum); 
  }
  return(0);
#else
  return(-1);
#endif
}

int write_source_type(const int type, char * filename) {

  FILE * ofs                    = NULL;
  LimeWriter * limewriter       = NULL;
  LimeRecordHeader * limeheader = NULL;
  int status = 0;
  int ME_flag=1, MB_flag=1;
  char message[500];
  n_uint64_t bytes;

  ofs = fopen(filename, "w");
  
  if(ofs == (FILE*)NULL) {
    fprintf(stderr, "[write_source_type] Could not open file %s for writing!\n Aboring...\n", filename);
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(500);
  }
  limewriter = limeCreateWriter( ofs );
  if(limewriter == (LimeWriter*)NULL) {
    fprintf(stderr, "[write_source_type] LIME error in file %s for writing!\n Aborting...\n", filename);
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(500);
  }
  
  if(type == 0) {
    sprintf(message,"DiracFermion_Source");
    bytes = strlen( message );
  }
  
  limeheader = limeCreateHeader(MB_flag, ME_flag, "source-type", bytes);
  status = limeWriteRecordHeader( limeheader, limewriter);
  if(status < 0 ) {
    fprintf(stderr, "[write_source_type] LIME write header error %d\n", status);
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
  return(0);
}

/**************************************************
 *
 * prepare_propagator
 *
 **************************************************/
int prepare_propagator(int timeslice, int iread, int is_mms, int no_mass, double sign, double mass, int isave, double *work, int pos, double*gauge_field) {

  char filename[200];
  int status;
  double signed_mass;

  if(is_mms) {
    sprintf(filename, "%s.%.4d.%.2d.%.2d.cgmms.%.2d.inverted", filename_prefix, Nconf, timeslice, iread, no_mass);
    signed_mass = sign * mass;
    status = read_lime_spinor(work, filename, pos);
    xchange_field(work);
    /* Qf5(g_spinor_field[isave], work, signed_mass); */
    Q_phi(g_spinor_field[isave], work, gauge_field, signed_mass);
    g5_phi(g_spinor_field[isave], VOLUME);

    if(g_check_inversion==1) {
      check_source(g_spinor_field[isave], work, gauge_field, -signed_mass, g_source_location, iread);
      // work contains D iread, reread original solution to work
      status = read_lime_spinor(work, filename, pos);
      xchange_field(work);
    }
  } else {
    if(no_mass == -1) {
      if(sign==-1.) {
        sprintf(filename, "%s/source.%.4d.%.2d.%.2d.inverted", filename_prefix2, Nconf, timeslice, iread);
      } else {
        sprintf(filename, "%s/msource.%.4d.%.2d.%.2d.inverted", filename_prefix2, Nconf, timeslice, iread);
      }
    } else {
      if(sign==-1.) {
        sprintf(filename, "%s/mass%.2d/source.%.4d.%.2d.%.2d.inverted", filename_prefix2, no_mass, Nconf, timeslice, iread);
      } else {
        sprintf(filename, "%s/mass%.2d/msource.%.4d.%.2d.%.2d.inverted", filename_prefix2, no_mass, Nconf, timeslice, iread);
      }
    }
    if(g_cart_id==0) fprintf(stdout, "\n# [prepare_propagator] reading fermion field from file %s at position %d\n", filename, pos);
    status = read_lime_spinor(g_spinor_field[isave], filename, pos);
  }

  if(status != 0) {
    fprintf(stderr, "\n[prepare_propagator] Error, reading spinor field returned %d\n", status);
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(500);
  }
  return(0);
}

/**************************************************
 *
 * rotate propagator FROM UKQCD TO ETMC gamma
 * matrix convention
 *
 **************************************************/
int rotate_propagator_ETMC_UKQCD (double *spinor, long unsigned int V) {

  long unsigned int ix;
  double sp[24], sp2[24];
  double norm = -1. / sqrt(2.);

  for(ix=0; ix<V; ix++) {
    _fv_eq_gamma_ti_fv(sp,  0, spinor+_GSI(ix));
    _fv_eq_gamma_ti_fv(sp2, 5, spinor+_GSI(ix));
    _fv_eq_fv_pl_fv(spinor+_GSI(ix), sp, sp2);
    _fv_ti_eq_re(spinor+_GSI(ix), norm);
  }
  return(0);
}

/**************************************************
 *
 * prepare_propagator2
 *
 **************************************************/
int prepare_propagator2(int *source_coords, int iread, int sign, void*work, int pos, int propfilename_format, size_t prec_out) {

  char filename[400];
  int status;

  if(propfilename_format==0) {
    if(sign==+1) {
      //sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.pmass.pre07.%.2d.inverted", filename_prefix, Nconf,
      //source_coords[0], source_coords[1], source_coords[2], source_coords[3], iread);
sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix, Nconf, 0, iread);
    } else {
      //sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2d.nmass.pre07.%.2d.inverted", filename_prefix, Nconf,
      //  source_coords[0], source_coords[1], source_coords[2], source_coords[3], iread);
      sprintf(filename, "%s.%.4d.%.2d.%.2d.inverted", filename_prefix2, Nconf, 0, iread);
    }
  }
  if(propfilename_format==1) {
    if(sign==+1) {
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2ds%.2dc%.2d.pmass.pre07.inverted", 
        filename_prefix, Nconf, source_coords[0], source_coords[1], 
        source_coords[2], source_coords[3], iread/3, iread%3);
    } else {
      sprintf(filename, "%s.%.4d.t%.2dx%.2dy%.2dz%.2ds%.2dc%.2d.nmass.pre07.inverted", 
        filename_prefix, Nconf, source_coords[0], source_coords[1], 
        source_coords[2], source_coords[3], iread/3, iread%3);
    }
  }
  if(propfilename_format==2) {
    if(sign==+1) {
      sprintf(filename, "%s.%.4d.00.%.2d.inverted", filename_prefix, Nconf, iread);
    } else {
      sprintf(filename, "%s.%.4d.00.%.2d.inverted", filename_prefix2, Nconf, iread);
    }
  }

  if(prec_out == 64) {
    status = read_lime_spinor((double*)work, filename, pos);
  } else {
    status = read_lime_spinor_single((float*)work, filename, pos);
  }
  if(status != 0) {
    fprintf(stderr, "[prepare_propagator2] Error, reading spinor field from file %s returned %d\n", filename, status);
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(500);
  }
  return(0);
}


int read_binary_spinor_data_single(float* const s, LimeReader * limereader, 
			    const int prec, DML_Checksum *ans) {

  int status=0;
  n_uint64_t bytes, ix;
  double tmp[24];
  DML_SiteRank rank;
  float tmp2[24];
  int words_bigendian;
  unsigned int t, x, y, z;
  words_bigendian = big_endian();

  DML_checksum_init(ans);
  rank = (DML_SiteRank) 0;
  
  if(prec == 32) bytes = 24*sizeof(float);
  else bytes = 24*sizeof(double);
  for(t = 0; t < T; t++){
    for(z = 0; z < LZ; z++){
      for(y = 0; y < LY; y++){
#if (defined HAVE_MPI)
      limeReaderSeek(limereader,(n_uint64_t) ( (((Tstart+t)*(LZ*g_nproc_z)+LZstart+z)*(LY*g_nproc_y)+LYstart+y)*(LX*g_nproc_x) +LXstart )*bytes, SEEK_SET);
#endif
	for(x = 0; x < LX; x++){
	  ix = g_ipt[t][x][y][z]*(n_uint64_t)12;
	  rank = (DML_SiteRank) ((((Tstart+t)*(LZ*g_nproc_z)+LZstart + z)*(LY*g_nproc_y) + LYstart + y)*(DML_SiteRank)(LX*g_nproc_x) + LXstart + x);
	  if(prec == 32) {
	    status = limeReaderReadData(tmp2, &bytes, limereader);
	    DML_checksum_accum(ans,rank,(char *) tmp2, bytes);	    
	  }
	  else {
	    status = limeReaderReadData(tmp, &bytes, limereader);
	    DML_checksum_accum(ans,rank,(char *) tmp, bytes);
	  }
	  if(!words_bigendian) {
	    if(prec == 32) {
	      byte_swap_assign_singleprec(&s[2*ix], tmp2, 24);
	    }
	    else {
	      byte_swap_assign_double2single(&s[2*ix], (double*)tmp, 24);
	    }
	  }
	  else {
	    if(prec == 32) {
              memcpy(&s[2*ix], tmp2, bytes);
	    }
	    else double2single(&s[2*ix], (double*)tmp, 24); 
	  }
	  if(status < 0 && status != LIME_EOR) {
	    return(-1);
	  }
	}
      }
    }
  }
#ifdef HAVE_MPI
  DML_checksum_combine(ans);
#endif
  if(g_cart_id == 0) printf("# [read_binary_spinor_data_single] The final checksum is %#x %#x\n", (*ans).suma, (*ans).sumb);
  return(0);
}

int read_lime_spinor_single(float * const s, char * filename, const int position) {
  FILE * ifs;
  int status=0, getpos=-1;
  n_uint64_t bytes;
  char * header_type;
  LimeReader * limereader;
  n_uint64_t prec = 32;
  DML_Checksum checksum;
  
  if((ifs = fopen(filename, "r")) == (FILE*)NULL) {
    fprintf(stderr, "[read_lime_spinor_single] Error opening file %s\n", filename);
    return(-1);
  }
  if(g_proc_id==0) fprintf(stdout, "# [read_lime_spinor_single] Reading Dirac-fermion field in LIME format from %s, single prec output\n", filename);

  limereader = limeCreateReader( ifs );
  if( limereader == (LimeReader *)NULL ) {
    fprintf(stderr, "[read_lime_spinor_single] Unable to open LimeReader\n");
    return(-1);
  }
  while( (status = limeReaderNextRecord(limereader)) != LIME_EOF ) {
    if(status != LIME_SUCCESS ) {
      fprintf(stderr, "[read_lime_spinor_single] limeReaderNextRecord returned error with status = %d!\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = limeReaderType(limereader);
    if(strcmp("scidac-binary-data",header_type) == 0) getpos++;
    if(getpos == position) break;
  }
  if(status == LIME_EOF) {
    fprintf(stderr, "[read_lime_spinor_single] no scidac-binary-data record found in file %s\n",filename);
    limeDestroyReader(limereader);
    fclose(ifs);
    if(g_proc_id==0) fprintf(stderr, "[read_lime_spinor_single] try to read in CMI format\n");
    // return(read_cmi(s, filename));
    return(-1);
  }
  bytes = limeReaderBytes(limereader);
  if(bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(24*sizeof(double))) prec = 64;
  else if(bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(24*sizeof(float))) prec = 32;
  else {
    fprintf(stderr, "[read_lime_spinor_single] wrong length in eospinor: bytes = %lu, not %lu. Aborting read!\n", 
	    bytes, (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(24*sizeof(double)));
    return(-1);
  }
  if(g_cart_id == 0) printf("# [read_lime_spinor_single] %lu Bit precision read\n", prec);

  status = read_binary_spinor_data_single(s, limereader, prec, &checksum);

  if(status < 0) {
    fprintf(stderr, "[read_lime_spinor_single] LIME read error occured with status = %d while reading file %s!\n Aborting...\n", 
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
}

/******************************************************************
 * read a timeslice from a spinor field
 ******************************************************************/
#ifdef HAVE_LIBLEMON
int read_lime_spinor_timeslice(double * const s, int timeslice, char * filename, const int position, DML_Checksum*checksum) {
  if (g_cart_id == 0) fprintf(stderr, "[read_lime_spinor_timeslice] Error, no version for lemon so far\n");
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
  return(1);
}
#else
int read_lime_spinor_timeslice(double * const s, int timeslice, char * filename, const int position, DML_Checksum*checksum) {
#ifdef HAVE_MPI
  if (g_cart_id == 0) fprintf(stderr, "[read_lime_spinor_timeslice] Error, no version for MPI so far\n");
  MPI_Abort(MPI_COMM_WORLD, 2);
  MPI_Finalize();
  return(2);
#else
  FILE * ifs;
  int status=0, getpos=-1;
  n_uint64_t bytes;
  char * header_type;
  LimeReader * limereader;
  n_uint64_t prec = 32;
  
  if((ifs = fopen(filename, "r")) == (FILE*)NULL) {
    fprintf(stderr, "[read_lime_spinor_timeslice] Error opening file %s\n", filename);
    return(-1);
  }
  if(g_proc_id==0) fprintf(stdout, "# [read_lime_spinor_timeslice] Reading timeslice no. %d of Dirac-fermion field in LIME format from %s\n", timeslice, filename);

  limereader = limeCreateReader( ifs );
  if( limereader == (LimeReader *)NULL ) {
    fprintf(stderr, "[read_lime_spinor_timeslice] Unable to open LimeReader\n");
    return(-1);
  }
  while( (status = limeReaderNextRecord(limereader)) != LIME_EOF ) {
    if(status != LIME_SUCCESS ) {
      fprintf(stderr, "[read_lime_spinor_timeslice] limeReaderNextRecord returned error with status = %d!\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = limeReaderType(limereader);
    if(strcmp("scidac-binary-data",header_type) == 0) getpos++;
    if(getpos == position) break;
  }
  if(status == LIME_EOF) {
    fprintf(stderr, "[read_lime_spinor_timeslice] no scidac-binary-data record found in file %s\n",filename);
    limeDestroyReader(limereader);
    fclose(ifs);
    if(g_proc_id==0) fprintf(stderr, "[read_lime_spinor_timeslice] try to read in CMI format\n");
    return(read_cmi(s, filename));
  }
  bytes = limeReaderBytes(limereader);
  if(bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(24*sizeof(double))) prec = 64;
  else if(bytes == (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(24*sizeof(float))) prec = 32;
  else {
    fprintf(stderr, "[read_lime_spinor_timeslice] wrong length in eospinor: bytes = %llu, not %llu. Aborting read!\n", 
	    bytes, (LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*(uint64_t)(24*sizeof(double)));
    return(-1);
  }
  if(g_cart_id == 0) printf("# [read_lime_spinor_timeslice] %llu Bit precision read\n", prec);

  status = read_binary_spinor_data_timeslice(s, timeslice, limereader, prec, checksum);

  if(status < 0) {
    fprintf(stderr, "[read_lime_spinor_timeslice] LIME read error occured with status = %d while reading file %s!\n Aborting...\n", 
	    status, filename);
    exit(500);
  }
  if(timeslice == T_global - 1) printf("# [read_lime_spinor_timeslice] The final checksum for prop file %s is %#lx %#lx\n", filename, checksum->suma, checksum->sumb);
  limeDestroyReader(limereader);
  fclose(ifs);
  return(0);
#endif  // of ifdef HAVE_MPI
}
#endif  // of ifdef HAVE_LIBLEMON

int read_binary_spinor_data_timeslice(double * const s, int timeslice, LimeReader * limereader, const int prec, DML_Checksum *ans) {
#ifdef HAVE_MPI
  if(g_cart_id == 0) fprintf(stderr, "[read_binary_spinor_data_timeslice] Error, no MPI version so far\n");
  return(1);
#else
  int status=0;
  n_uint64_t bytes, ix;
  double tmp[24];
  DML_SiteRank rank;
  float tmp2[24];
  int words_bigendian;
  unsigned int t, x, y, z;
  words_bigendian = big_endian();

  if(timeslice == 0) {
    // if(g_cart_id==0) fprintf(stdout, "# [] initializing checksum for timeslice %d\n", timeslice);
    DML_checksum_init(ans);
  }
  rank = (DML_SiteRank) 0;
  
  if(prec == 32) bytes = 24*sizeof(float);
  else bytes = 24*sizeof(double);
  t = timeslice;
  if(t>0) {
    limeReaderSeek(limereader,(n_uint64_t) (t*LX*LY*LZ)*bytes, SEEK_SET);
  }

  for(z = 0; z < LZ; z++){
  for(y = 0; y < LY; y++){
  for(x = 0; x < LX; x++){
    ix = g_ipt[0][x][y][z]*(n_uint64_t)12;
    rank = (DML_SiteRank) ((((Tstart+t)*(LZ*g_nproc_z)+LZstart + z)*(LY*g_nproc_y) + LYstart + y)*(DML_SiteRank)(LX*g_nproc_x) + LXstart + x);
    if(prec == 32) {
      status = limeReaderReadData(tmp2, &bytes, limereader);
      DML_checksum_accum(ans,rank,(char *) tmp2, bytes);	    
    } else {
      status = limeReaderReadData(tmp, &bytes, limereader);
       DML_checksum_accum(ans,rank,(char *) tmp, bytes);
    }
    if(!words_bigendian) {
      if(prec == 32) {
        byte_swap_assign_single2double(&s[2*ix], (float*)tmp2, 24);
      } else {
        byte_swap_assign(&s[2*ix], tmp, 24);
      }
    } else {
      if(prec == 32) {
        single2double(&s[2*ix], (float*)tmp2, 24);
      } else memcpy(&s[2*ix], tmp, bytes);
    }
    if(status < 0 && status != LIME_EOR) {
      return(-1);
    }
  }}}
  // if(timeslice == T_global - 1) printf("# [] The final checksum is %#lx %#lx\n", (*ans).suma, (*ans).sumb);
  return(0);
#endif  // of ifdef HAVE_MPI
}

}
