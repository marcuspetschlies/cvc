/****************************************************
 * IO routines:
 *
 * read_lime_gauge_field_doubleprec
 *
 * read_lime_gauge_field_singleprec
 *
 * Autor: 
 *        Carsten Urbach <urbach@ifh.de>
 *
 *        Marcus Petschlies <marcuspe@physik.hu-berlin.de>
 *        - added read_lime_gauge_field_timeslice_doubleprec,
 *          derived from read_lime_gauge_field_doubleprec:wq
 *        - added changes related to an MPI-version
 *
 ****************************************************/

/*
 * Note:
 * Required version of lime: >= 1.2.3
 * n_uint64_t is a lime defined type!!
 *
 */

#define _FILE_OFFSET_BITS 64

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#include "gauge_io.h"
#include "io.h"
#include "io_utils.h"

namespace cvc {
/* #define MAXBUF 1048576 */

#ifdef HAVE_LIBLEMON
int read_lime_gauge_field_doubleprec(const char * filename) {
  MPI_File *ifs=NULL;
  int t, x, y, z, status;
  n_uint64_t bytes;
  int latticeSize[] = {T_global, LX_global, LY_global, LZ_global};
  int scidacMapping[] = {0, 3, 2, 1};
  int prec;
  char * header_type;
  LemonReader * reader=NULL;
  double tmp[72], tmp2[72];
  int words_bigendian, mu, i, j;
  int gauge_read_flag = 0;
  DML_Checksum checksum;
  DML_SiteRank rank;
  uint64_t fbsu3;
  char * filebuffer = NULL, * current = NULL;

  DML_checksum_init(&checksum);
  words_bigendian = big_endian();

  ifs = (MPI_File*)malloc(sizeof(MPI_File));
  status = MPI_File_open(g_cart_grid, (char*)filename, MPI_MODE_RDONLY, MPI_INFO_NULL, ifs);
  status = (status == MPI_SUCCESS) ? 0 : 1;
  if(status) {
    fprintf(stderr, "[read_lime_gauge_field_doubleprec] Could not open file for reading. Aborting...\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(505);
  }

  if( (reader = lemonCreateReader(ifs, g_cart_grid)) == NULL ) {
    fprintf(stderr, "[read_lime_gauge_field_doubleprec] Could not create reader. Aborting...\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(503);
  }

  while ((status = lemonReaderNextRecord(reader)) != LIME_EOF) {
    if (status != LIME_SUCCESS) {
      fprintf(stderr, "[read_lime_gauge_field_doubleprec] lemonReaderNextRecord returned status %d.\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = (char*)lemonReaderType(reader);
    if (strcmp("ildg-binary-data", header_type) == 0) break;
  }
  if(status==LIME_EOF) {
    fprintf(stderr, "[read_lime_gauge_field_doubleprec] no ildg-binary-data record found in file %s\n",filename);
    lemonDestroyReader(reader);
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(502);
  } 

  bytes = lemonReaderBytes(reader);

  if (bytes == (n_uint64_t)g_nproc * (n_uint64_t)VOLUME * 72 * (n_uint64_t)sizeof(double)) {
    prec = 64;
  } else {
    if (bytes == (n_uint64_t)g_nproc * (n_uint64_t)VOLUME * 72 * (n_uint64_t)sizeof(float)) {
      prec = 32;
    } else {
      fprintf(stderr, "[read_lime_gauge_field_doubleprec] Probably wrong lattice size or precision (bytes=%lu).\n", (unsigned long)bytes);
      fprintf(stderr, "[read_lime_gauge_field_doubleprec] Panic! Aborting...\n");
      fflush(stdout);
      MPI_File_close(reader->fp);
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
      exit(501);
    }
  }
  
  if(g_cart_id==0) fprintf(stdout, "# [read_lime_gauge_field_doubleprec] %d Bit precision read.\n", prec);

  fbsu3 = 18*sizeof(double);
  if (prec == 32) fbsu3 /= 2;
  bytes = 4 * fbsu3;

  if((void*)(filebuffer = (char*)malloc(VOLUME * bytes)) == NULL) {
    printf ("[read_lime_gauge_field_doubleprec] malloc errno in read_binary_gauge_data_parallel\n");
    return 1;
  }
  status = lemonReadLatticeParallelMapped(reader, filebuffer, bytes, latticeSize, scidacMapping);

  if (status < 0 && status != LEMON_EOR)  {
    fprintf(stderr, "[read_lime_gauge_field_doubleprec] LEMON read error occured with status = %d while reading!\nPanic! Aborting...\n", status);
    MPI_File_close(reader->fp);
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(500);
  }

  for (t = 0; t <  T; t++) {
  for (z = 0; z < LZ; z++) {
  for (y = 0; y < LY; y++) {
  for (x = 0; x < LX; x++) {
    rank = (DML_SiteRank)(LXstart + (((Tstart + t) * LZ *g_nproc_z + LZstart + z) * LY*g_nproc_y + LYstart+y) * ((DML_SiteRank)LX * g_nproc_x) + x);
    current = filebuffer + bytes * (x + (y + (t * LZ + z) * LY) * LX);
    DML_checksum_accum(&checksum, rank, current, bytes);
    if(!words_bigendian) {
      if (prec == 32) {
        byte_swap_assign_single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 1)], current            , 18*sizeof(double) / 8);
        byte_swap_assign_single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 2)], current +     fbsu3, 18*sizeof(double) / 8);
        byte_swap_assign_single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 3)], current + 2 * fbsu3, 18*sizeof(double) / 8);
        byte_swap_assign_single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 0)], current + 3 * fbsu3, 18*sizeof(double) / 8);
      } else  {
        byte_swap_assign(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 1)], current            , 18*sizeof(double) / 8);
        byte_swap_assign(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 2)], current +     fbsu3, 18*sizeof(double) / 8);
        byte_swap_assign(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 3)], current + 2 * fbsu3, 18*sizeof(double) / 8);
        byte_swap_assign(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 0)], current + 3 * fbsu3, 18*sizeof(double) / 8);
      }
    } else {
      if (prec == 32) {
        single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 1)], current            , 18*sizeof(double) / 8);
        single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 2)], current +     fbsu3, 18*sizeof(double) / 8);
        single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 3)], current + 2 * fbsu3, 18*sizeof(double) / 8);
        single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 0)], current + 3 * fbsu3, 18*sizeof(double) / 8);
      } else {
        memcpy(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 1)], current            , 18*sizeof(double));
        memcpy(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 2)], current +     fbsu3, 18*sizeof(double));
        memcpy(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 3)], current + 2 * fbsu3, 18*sizeof(double));
        memcpy(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 0)], current + 3 * fbsu3, 18*sizeof(double));
      }
    }
  }}}}
  DML_global_xor(&checksum.suma);
  DML_global_xor(&checksum.sumb);
  if(g_cart_id==0) {
    fprintf(stdout, "# [read_lime_gauge_field_doubleprec] checksum for gaugefield %s\n is %#x %#x.\n", filename, checksum.suma, checksum.sumb);
    fflush(stdout); 
  }

  lemonDestroyReader(reader);
  MPI_File_close(ifs);
  free(ifs);

  free(filebuffer);
  return(0);
}
#else
int read_lime_gauge_field_doubleprec(const char * filename) {
  FILE * ifs;
  int t, x, y, z, status;
  n_uint64_t bytes;
  char * header_type;
  LimeReader * limereader;
  double tmp[72], tmp2[72];
  int words_bigendian, mu, i, j;
  int ix;
  DML_Checksum checksum;
  DML_SiteRank rank;

  n_uint64_t file_seek = 0;

  DML_checksum_init(&checksum); 

  words_bigendian = big_endian();
  ifs = fopen(filename, "r");
  if(ifs == (FILE *)NULL) {
    fprintf(stderr, "[read_lime_gauge_field_doubleprec] Could not open file %s\n Aborting...\n", filename);
    return(500);
  }
  limereader = limeCreateReader( ifs );
  if( limereader == (LimeReader *)NULL ) {
    fprintf(stderr, "[read_lime_gauge_field_doubleprec] Unable to open LimeReader\n");
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(500);
  }
  while( (status = limeReaderNextRecord(limereader)) != LIME_EOF ) {
    if(status != LIME_SUCCESS ) {
      fprintf(stderr, "[read_lime_gauge_field_doubleprec] limeReaderNextRecord returned error with status = %d!\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = limeReaderType(limereader);
    if(strcmp("ildg-binary-data",header_type) == 0) break;
  }
  if(status == LIME_EOF) {
    fprintf(stderr, "[read_lime_gauge_field_doubleprec] no ildg-binary-data record found in file %s\n",filename);
    limeDestroyReader(limereader);
    fclose(ifs);
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(-2);
  }
  bytes = limeReaderBytes(limereader);
  //fprintf(stdout, "[%d] lime_reader bytes = %llu\n", g_cart_id, bytes);
  if(bytes != (n_uint64_t)(LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*72*(n_uint64_t)sizeof(double)) {
    if(bytes != (n_uint64_t)(LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*72*(n_uint64_t)sizeof(double)/2) {
      fprintf(stderr, "[read_lime_gauge_field_doubleprec] Probably wrong lattice size or precision (bytes=%llu) in file %s expected %llu\n", 
	      (n_uint64_t)bytes, filename, (n_uint64_t)(LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T_global*72*(n_uint64_t)sizeof(double));
      fprintf(stderr, "[read_lime_gauge_field_doubleprec] Aborting...!\n");
      fflush( stdout );
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(501);
    }
    else {
      fclose(ifs);
      fprintf(stderr, "[read_lime_gauge_field_doubleprec] single precision read!\n");
      return( read_lime_gauge_field_singleprec(filename) );
    }
  }

  bytes = (n_uint64_t)72 * (n_uint64_t)sizeof(double);
  // fprintf(stdout, "[%d] size of n_uint64_t = %lu\n", g_cart_id, sizeof(n_uint64_t));
  // fprintf(stdout, "[%d] bytes = %llu\n", g_cart_id, bytes);

  for(t = 0; t < T; t++) {
    for(z = 0; z < LZ; z++) {
      for(y = 0; y < LY; y++) {
#ifdef HAVE_MPI
        file_seek = (n_uint64_t) ( (((Tstart+t)*(LZ*g_nproc_z) + LZstart + z)*(LY*g_nproc_y) + LYstart + y)*(n_uint64_t)(LX*g_nproc_x) + LXstart) * bytes;
        //fprintf(stdout, "[%d] file_seek = %llu\n", g_cart_id, file_seek);
          
        limeReaderSeek(limereader, file_seek, SEEK_SET);
#endif
	for(x = 0; x < LX; x++) {
	  n_uint64_t p = (((t*LX+x)*LY+y)*LZ+z)*(n_uint64_t)12;
	  rank = (DML_SiteRank) ( (((Tstart+t)*(LZ*g_nproc_z) + LZstart +z)*(LY*g_nproc_y)+LYstart+y)*(DML_SiteRank)(LX*g_nproc_x)+x+LXstart ); 
	  if(!words_bigendian) {
	    status = limeReaderReadData(tmp, &bytes, limereader);
	    DML_checksum_accum(&checksum, rank, (char *)tmp, bytes); 
	    byte_swap_assign(tmp2, tmp, 72);
	  }
	  else {
	    status = limeReaderReadData(tmp2, &bytes, limereader);
	    DML_checksum_accum(&checksum, rank, (char *)tmp2, bytes); 
	  }
          //for(ix=0;ix<36;ix++) fprintf(stdout, "[%3d] %6d%6d%3d%3d\t%e\t%e\n", g_cart_id, (int)rank, (int)p/12, ix/9, ix%9, tmp2[2*ix], tmp2[2*ix+1]);

	  n_uint64_t k =0;
	  /* ILDG has mu-order: x,y,z,t */
	  for(mu = 1; mu < 4; mu++) {
	    for(i = 0; i < 3; i++) {
	      for(j = 0; j < 3; j++) {
		/* config (p+mu*3+i, j) = complex<double> (tmp2[2*k], tmp2[2*k+1]); */

		n_uint64_t index = ((p+mu*3+i) * 3 + j) * 2;
		g_gauge_field[index  ] = tmp2[2*k];
		g_gauge_field[index+1] = tmp2[2*k+1];

		k++;
	      }
	    }
	  }
 	  for(i = 0; i < 3; i++) {
 	    for(j = 0; j < 3; j++) {
 	      /* config (p+i, j) = complex<double> (tmp2[2*k], tmp2[2*k+1]); */

	      n_uint64_t index = ((p+i) * 3 + j) * 2;
	      g_gauge_field[index  ] = tmp2[2*k];
	      g_gauge_field[index+1] = tmp2[2*k+1];

 	      k++; 	    
	    }
 	  }
	  if(status < 0 && status != LIME_EOR) {
	    fprintf(stderr, "[read_lime_gauge_field_doubleprec] LIME read error occured with status = %d while reading file %s!\n Aborting...\n", 
		    status, filename);
#ifdef HAVE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
	    MPI_Finalize();
#endif
	    exit(500);
	  }
	}
      }
    }
  }
  limeDestroyReader(limereader);
  fclose(ifs);
#ifdef HAVE_MPI
  DML_checksum_combine(&checksum); 
#endif
  if(g_cart_id==0) fprintf(stdout, "# [read_lime_gauge_field_doubleprec] checksum for gaugefield %s is %#x %#x\n",
            filename, checksum.suma, checksum.sumb); 
  return(0);
}
#endif

#ifdef HAVE_LIBLEMON
int read_lime_gauge_field_singleprec(const char * filename) {
  MPI_File *ifs=NULL;
  int t, x, y, z, status;
  n_uint64_t bytes;
  int latticeSize[] = {T_global, LX_global, LY_global, LZ_global};
  int scidacMapping[] = {0, 3, 2, 1};
  int prec;
  char * header_type;
  LemonReader * reader=NULL;
  double tmp[72], tmp2[72];
  int words_bigendian, mu, i, j;
  int gauge_read_flag = 0;
  DML_Checksum checksum;
  DML_SiteRank rank;
  uint64_t fbsu3;
  char * filebuffer = NULL, * current = NULL;

  DML_checksum_init(&checksum);
  words_bigendian = big_endian();

  ifs = (MPI_File*)malloc(sizeof(MPI_File));
  status = MPI_File_open(g_cart_grid, (char*)filename, MPI_MODE_RDONLY, MPI_INFO_NULL, ifs);
  status = (status == MPI_SUCCESS) ? 0 : 1;
  if(status) {
    fprintf(stderr, "[read_lime_gauge_field_singleprec] Could not open file for reading. Aborting...\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(505);
  }

  if( (reader = lemonCreateReader(ifs, g_cart_grid)) == NULL ) {
    fprintf(stderr, "[read_lime_gauge_field_singleprec] Could not create reader. Aborting...\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(503);
  }

  while ((status = lemonReaderNextRecord(reader)) != LIME_EOF) {
    if (status != LIME_SUCCESS) {
      fprintf(stderr, "[read_lime_gauge_field_singleprec] lemonReaderNextRecord returned status %d.\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = (char*)lemonReaderType(reader);
    if (strcmp("ildg-binary-data", header_type) == 0) break;
  }
  if(status==LIME_EOF) {
    fprintf(stderr, "[read_lime_gauge_field_singleprec] no ildg-binary-data record found in file %s\n",filename);
    lemonDestroyReader(reader);
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(502);
  } 

  bytes = lemonReaderBytes(reader);

  if (bytes == (n_uint64_t)g_nproc * (n_uint64_t)VOLUME * 72 * (n_uint64_t)sizeof(double)) {
    prec = 64;
  } else {
    if (bytes == (n_uint64_t)g_nproc * (n_uint64_t)VOLUME * 72 * (n_uint64_t)sizeof(float)) {
      prec = 32;
    } else {
      fprintf(stderr, "[read_lime_gauge_field_singleprec] Probably wrong lattice size or precision (bytes=%lu).\n", (unsigned long)bytes);
      fprintf(stderr, "[read_lime_gauge_field_singleprec] Panic! Aborting...\n");
      fflush(stdout);
      MPI_File_close(reader->fp);
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
      exit(501);
    }
  }
  
  if(g_cart_id==0) fprintf(stdout, "# [read_lime_gauge_field_singleprec] %d Bit precision read.\n", prec);

  fbsu3 = 18*sizeof(double);
  if (prec == 32) fbsu3 /= 2;
  bytes = 4 * fbsu3;

  if((void*)(filebuffer = (char*)malloc(VOLUME * bytes)) == NULL) {
    printf ("[read_lime_gauge_field_singleprec] malloc errno in read_binary_gauge_data_parallel\n");
    return 1;
  }
  status = lemonReadLatticeParallelMapped(reader, filebuffer, bytes, latticeSize, scidacMapping);

  if (status < 0 && status != LEMON_EOR)  {
    fprintf(stderr, "[read_lime_gauge_field_singleprec] LEMON read error occured with status = %d while reading!\nPanic! Aborting...\n", status);
    MPI_File_close(reader->fp);
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
    exit(500);
  }

  for (t = 0; t <  T; t++) {
  for (z = 0; z < LZ; z++) {
  for (y = 0; y < LY; y++) {
  for (x = 0; x < LX; x++) {
    rank = (DML_SiteRank)(LXstart + (((Tstart + t) * LZ*g_nproc_z + LZstart + z) * LY*g_nproc_y + LYstart+y) * ((DML_SiteRank)LX * g_nproc_x) + x);
    current = filebuffer + bytes * (x + (y + (t * LZ + z) * LY) * LX);
    DML_checksum_accum(&checksum, rank, current, bytes);
    if(!words_bigendian) {
      if (prec == 32) {
        byte_swap_assign_single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 1)], current            , 18*sizeof(double) / 8);
        byte_swap_assign_single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 2)], current +     fbsu3, 18*sizeof(double) / 8);
        byte_swap_assign_single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 3)], current + 2 * fbsu3, 18*sizeof(double) / 8);
        byte_swap_assign_single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 0)], current + 3 * fbsu3, 18*sizeof(double) / 8);
      } else  {
        byte_swap_assign(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 1)], current            , 18*sizeof(double) / 8);
        byte_swap_assign(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 2)], current +     fbsu3, 18*sizeof(double) / 8);
        byte_swap_assign(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 3)], current + 2 * fbsu3, 18*sizeof(double) / 8);
        byte_swap_assign(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 0)], current + 3 * fbsu3, 18*sizeof(double) / 8);
      }
    } else {
      if (prec == 32) {
        single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 1)], current            , 18*sizeof(double) / 8);
        single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 2)], current +     fbsu3, 18*sizeof(double) / 8);
        single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 3)], current + 2 * fbsu3, 18*sizeof(double) / 8);
        single2double(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 0)], current + 3 * fbsu3, 18*sizeof(double) / 8);
      } else {
        memcpy(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 1)], current            , 18*sizeof(double));
        memcpy(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 2)], current +     fbsu3, 18*sizeof(double));
        memcpy(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 3)], current + 2 * fbsu3, 18*sizeof(double));
        memcpy(&g_gauge_field[ _GGI(g_ipt[t][x][y][z], 0)], current + 3 * fbsu3, 18*sizeof(double));
      }
    }
  }}}}
  DML_global_xor(&checksum.suma);
  DML_global_xor(&checksum.sumb);
  if(g_cart_id==0) {
    fprintf(stdout, "# [read_lime_gauge_field_singleprec] checksum for gaugefield %s\n is %#x %#x.\n", filename, checksum.suma, checksum.sumb);
    fflush(stdout); 
  }

  lemonDestroyReader(reader);
  MPI_File_close(ifs);
  free(ifs);

  free(filebuffer);
  return(0);
}
#else
int read_lime_gauge_field_singleprec(const char * filename) {

  FILE * ifs;
  int t, x, y, z, status;
  n_uint64_t bytes;
  char * header_type;
  LimeReader * limereader;
  float tmp[72], tmp2[72];
  int words_bigendian, mu, i, j;
  DML_Checksum checksum;
  DML_SiteRank rank;

  DML_checksum_init(&checksum);

  words_bigendian = big_endian();
  ifs = fopen(filename, "r");
  if(ifs == (FILE *)NULL) {
    fprintf(stderr, "[read_lime_gauge_field_singleprec] Could not open file %s\n Aborting...\n", filename);
    return(500);
  }
  limereader = limeCreateReader( ifs );
  if( limereader == (LimeReader *)NULL ) {
    fprintf(stderr, "[read_lime_gauge_field_singleprec] Unable to open LimeReader\n");
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(500);
  }
  while( (status = limeReaderNextRecord(limereader)) != LIME_EOF ) {
    if(status != LIME_SUCCESS ) {
      fprintf(stderr, "[read_lime_gauge_field_singleprec] limeReaderNextRecord returned error with status = %d!\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = limeReaderType(limereader);
    if( strcmp("ildg-binary-data",header_type) == 0) break;
  }
  if(status == LIME_EOF) {
    fprintf(stderr, "[read_lime_gauge_field_singleprec] no ildg-binary-data record found in file %s\n",filename);
    limeDestroyReader(limereader);
    fclose(ifs);
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(-2);
  }
  bytes = limeReaderBytes(limereader);
  if(bytes != (n_uint64_t)(LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*(T*g_nproc_t)*72*(n_uint64_t)sizeof(float)) {
    fprintf(stderr, "[read_lime_gauge_field_singleprec] Probably wrong lattice size or precision (bytes=%d) in file %s\n", (int)bytes, filename);
    fprintf(stderr, "[read_lime_gauge_field_singleprec] Aborting...!\n");
    fflush( stdout );
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(501);
  }

  bytes = (n_uint64_t)72*sizeof(float);
  for(t = 0; t < T; t++){
    for(z = 0; z < LZ; z++){
      for(y = 0; y < LY; y++){
#if (defined HAVE_MPI)
        limeReaderSeek(limereader,
	  (n_uint64_t) ( (((Tstart+t)*LZ*g_nproc_z + LZstart + z)*(LY*g_nproc_y) + LYstart + y)*(LX*g_nproc_x)+LXstart ) * bytes, SEEK_SET);
#endif
	for(x = 0; x < LX; x++) {
	  n_uint64_t p = (((t*LX+x)*LY+y)*LZ+z)*(n_uint64_t)12;
	  rank = (DML_SiteRank) ( (((Tstart+t)*LZ*g_nproc_z + LZstart +z)*(LY*g_nproc_y) + LYstart + y)*(DML_SiteRank)(LX*g_nproc_x)+x+LXstart );
	  if(!words_bigendian) {
	    status = limeReaderReadData(tmp, &bytes, limereader);
	    DML_checksum_accum(&checksum, rank, (char *)tmp, bytes);
	    byte_swap_assign_singleprec(tmp2, tmp, 72);
	  }
	  else {
	    status = limeReaderReadData(tmp2, &bytes, limereader);
	    DML_checksum_accum(&checksum, rank, (char *)tmp2, bytes);
	  }
	  n_uint64_t k =0;
	  /* ILDG has mu-order: x,y,z,t */
	  for(mu = 1; mu < 4; mu++) {
	    for(i = 0; i < 3; i++) {
	      for(j = 0; j < 3; j++) {
		/*config (p+mu*3+i, j) = complex<double> (tmp2[2*k], tmp2[2*k+1]); */
		n_uint64_t index = ((p+mu*3+i) * 3 + j) * 2;
		g_gauge_field[index  ] = (double)tmp2[2*k];
		g_gauge_field[index+1] = (double)tmp2[2*k+1];

		k++;
	      }
 	    }
	  }
 	  for(i = 0; i < 3; i++) {
	    for(j = 0; j < 3; j++) {
	      /*  config (p+i, j) = complex<double> (tmp2[2*k], tmp2[2*k+1]); */
	      n_uint64_t index = ((p+i) * 3 + j) * 2;
	      g_gauge_field[index  ] = tmp2[2*k];
	      g_gauge_field[index+1] = tmp2[2*k+1];

	      k++;
	    }
	  }

	  if(status < 0 && status != LIME_EOR) {
	    fprintf(stderr, "[read_lime_gauge_field_singleprec] LIME read error occured with status = %d while reading file %s!\n Aborting...\n", 
		    status, filename);
#ifdef HAVE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
            MPI_Finalize();
#endif
	    exit(500);
	  }
	} 
      }
    }
  }
  limeDestroyReader(limereader);
  fclose(ifs);
#ifdef HAVE_MPI
  DML_checksum_combine(&checksum);
#endif
  if(g_cart_id==0) fprintf(stdout, "# [read_lime_gauge_field_singleprec] checksum for gaugefield %s is %#x %#x\n",
             filename, checksum.suma, checksum.sumb);
  return(0);
}
#endif


/*********************************************************************************************
 *
 *
 *********************************************************************************************/
int read_lime_gauge_field_doubleprec_timeslice(double *gfield, const char * filename, const int timeslice, DML_Checksum *checksum) {
#ifndef HAVE_MPI
  FILE * ifs;
  int t, x, y, z, status;
  n_uint64_t bytes;
  char * header_type;
  LimeReader * limereader;
  double tmp[72], tmp2[72];
  int words_bigendian, mu, i, j;
/*  DML_Checksum checksum; */
  DML_SiteRank rank;

  if(timeslice==0) {
    DML_checksum_init(checksum); 
  }

  words_bigendian = big_endian();
  ifs = fopen(filename, "r");
  if(ifs == (FILE *)NULL) {
    fprintf(stderr, "[read_lime_gauge_field_doubleprec_timeslice] Could not open file %s\n Aborting...\n", filename);
    return(500);
  }
  limereader = limeCreateReader( ifs );
  if( limereader == (LimeReader *)NULL ) {
    fprintf(stderr, "[read_lime_gauge_field_doubleprec_timeslice] Unable to open LimeReader\n");
    exit(500);
  }
  while( (status = limeReaderNextRecord(limereader)) != LIME_EOF ) {
    if(status != LIME_SUCCESS ) {
      fprintf(stderr, "[read_lime_gauge_field_doubleprec_timeslice] limeReaderNextRecord returned error with status = %d!\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = limeReaderType(limereader);
    if(strcmp("ildg-binary-data",header_type) == 0) break;
  }
  if(status == LIME_EOF) {
    fprintf(stderr, "[read_lime_gauge_field_doubleprec_timeslice] no ildg-binary-data record found in file %s\n",filename);
    limeDestroyReader(limereader);
    fclose(ifs);
    exit(-2);
  }
  bytes = limeReaderBytes(limereader);
  if(bytes != (n_uint64_t)LX*LY*LZ*T_global*72*(n_uint64_t)sizeof(double)) {
    if(bytes != (n_uint64_t)LX*LY*LZ*T_global*72*(n_uint64_t)sizeof(double)/2) {
      fprintf(stderr, "[read_lime_gauge_field_doubleprec_timeslice] Probably wrong lattice size or precision (bytes=%llu) in file %s expected %llu\n", 
	      (n_uint64_t)bytes, filename, (n_uint64_t)LX*LY*LZ*T_global*72*(n_uint64_t)sizeof(double));
      fprintf(stderr, "Aborting...!\n");
      fflush( stdout );
      exit(501);
    }
    else {
      fclose(ifs);
      fprintf(stderr, "[read_lime_gauge_field_doubleprec_timeslice] Warning: trying to use single precision reading\n");
      return( read_lime_gauge_field_singleprec_timeslice(gfield, filename, timeslice, checksum) );
    }
  }

  bytes = (n_uint64_t)72*sizeof(double);

  t = 0;
  for(z = 0; z < LZ; z++) {
  for(y = 0; y < LY; y++) {
    limeReaderSeek(limereader,
      (n_uint64_t) ( ((timeslice*LZ + z)*LY + y)*LX ) * bytes, SEEK_SET);
    for(x = 0; x < LX; x++) {
      n_uint64_t p = (((t*LX+x)*LY+y)*LZ+z)*(n_uint64_t)12;
      rank = (DML_SiteRank) ( ( (timeslice*LZ+z)*LY+y)*(DML_SiteRank)LX+x ); 
      if(!words_bigendian) {
        status = limeReaderReadData(tmp, &bytes, limereader);
        DML_checksum_accum(checksum, rank, (char *)tmp, bytes); 
        byte_swap_assign(tmp2, tmp, 72);
      }
      else {
        status = limeReaderReadData(tmp2, &bytes, limereader);
        DML_checksum_accum(checksum, rank, (char *)tmp2, bytes); 
      }
      n_uint64_t k =0;
      /* ILDG has mu-order: x,y,z,t */
      for(mu = 1; mu < 4; mu++) {
        for(i = 0; i < 3; i++) {
        for(j = 0; j < 3; j++) {
          /* config (p+mu*3+i, j) = complex<double> (tmp2[2*k], tmp2[2*k+1]); */

          n_uint64_t index = ((p+mu*3+i) * 3 + j) * 2;
	  gfield[index  ] = tmp2[2*k];
	  gfield[index+1] = tmp2[2*k+1];
          k++;
        }}
      }
      for(i = 0; i < 3; i++) {
      for(j = 0; j < 3; j++) {
        /* config (p+i, j) = complex<double> (tmp2[2*k], tmp2[2*k+1]); */

	n_uint64_t index = ((p+i) * 3 + j) * 2;
	gfield[index  ] = tmp2[2*k];
	gfield[index+1] = tmp2[2*k+1];

        k++; 	    
      }}
      if(status < 0 && status != LIME_EOR) {
        fprintf(stderr, "[read_lime_gauge_field_doubleprec_timeslice] LIME read error occured with status = %d while reading file %s!\n Aborting...\n", 
          status, filename);
	exit(500);
      }
    }
  }}
  
  limeDestroyReader(limereader);
  fclose(ifs);
  if(timeslice==T_global-1) fprintf(stdout, "# [read_lime_gauge_field_doubleprec_timeslice] checksum for gaugefield %s is %#x %#x\n",
            filename, (*checksum).suma, (*checksum).sumb); 
  return(0);
#else
  if(g_cart_id==0) fprintf(stderr, "[read_lime_gauge_field_doubleprec_timeslice] Error, no MPI version\n ");
  return(-1);
#endif
}

/*************************************************************************************************
 *
 *
 *************************************************************************************************/
int read_lime_gauge_field_singleprec_timeslice(double *gfield, const char * filename, const int timeslice, DML_Checksum *checksum) {
#ifndef HAVE_MPI
  FILE * ifs;
  int t, x, y, z, status;
  n_uint64_t bytes;
  char * header_type;
  LimeReader * limereader;
  float tmp[72];
  double tmp2[72];
  int words_bigendian, mu, i, j;
/*  DML_Checksum checksum; */
  DML_SiteRank rank;

  if(timeslice==0) {
    DML_checksum_init(checksum); 
  }

  words_bigendian = big_endian();
  ifs = fopen(filename, "r");
  if(ifs == (FILE *)NULL) {
    fprintf(stderr, "[read_lime_gauge_field_singleprec_timeslice] Could not open file %s\n Aborting...\n", filename);
    return(500);
  }
  limereader = limeCreateReader( ifs );
  if( limereader == (LimeReader *)NULL ) {
    fprintf(stderr, "[read_lime_gauge_field_singleprec_timeslice] Unable to open LimeReader\n");
    exit(500);
  }
  while( (status = limeReaderNextRecord(limereader)) != LIME_EOF ) {
    if(status != LIME_SUCCESS ) {
      fprintf(stderr, "[read_lime_gauge_field_singleprec_timeslice] limeReaderNextRecord returned error with status = %d!\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = limeReaderType(limereader);
    if(strcmp("ildg-binary-data",header_type) == 0) break;
  }
  if(status == LIME_EOF) {
    fprintf(stderr, "[read_lime_gauge_field_singleprec_timeslice] no ildg-binary-data record found in file %s\n",filename);
    limeDestroyReader(limereader);
    fclose(ifs);
    exit(-2);
  }
  bytes = limeReaderBytes(limereader);
  if(bytes != (n_uint64_t)LX*LY*LZ*T_global*72*(n_uint64_t)sizeof(float)) {
    if(bytes != (n_uint64_t)LX*LY*LZ*T_global*72*(n_uint64_t)sizeof(double)) {
      fprintf(stderr, "[read_lime_gauge_field_singleprec_timeslice] Error, Probably wrong lattice size or precision (bytes=%llu) in file %s expected %llu\n", 
	      (n_uint64_t)bytes, filename, (n_uint64_t)LX*LY*LZ*T_global*72*(n_uint64_t)sizeof(double));
      fprintf(stderr, "[read_lime_gauge_field_singleprec_timeslice] Aborting...!\n");
      fflush( stdout );
      exit(501);
    }
    else {
      fclose(ifs);
      fprintf(stderr, "[read_lime_gauge_field_singleprec_timeslice] Error, use double precision reading\n");
      exit(501);
    }
  }

  bytes = (n_uint64_t)72*sizeof(float);

  t = 0;
  for(z = 0; z < LZ; z++) {
  for(y = 0; y < LY; y++) {
    limeReaderSeek(limereader,
      (n_uint64_t) ( ((timeslice*LZ + z)*LY + y)*LX ) * bytes, SEEK_SET);
    for(x = 0; x < LX; x++) {
      n_uint64_t p = (((t*LX+x)*LY+y)*LZ+z)*(n_uint64_t)12;
      rank = (DML_SiteRank) ( ( (timeslice*LZ+z)*LY+y)*(DML_SiteRank)LX+x ); 
      if(!words_bigendian) {
        status = limeReaderReadData(tmp, &bytes, limereader);
        DML_checksum_accum(checksum, rank, (char *)tmp, bytes); 
        byte_swap_assign_single2double(tmp2, tmp, 72);
      }
      else {
        status = limeReaderReadData(tmp, &bytes, limereader);
        DML_checksum_accum(checksum, rank, (char *)tmp, bytes);
        single2double(tmp2, tmp, 72);
      }
      n_uint64_t k =0;
      /* ILDG has mu-order: x,y,z,t */
      for(mu = 1; mu < 4; mu++) {
        for(i = 0; i < 3; i++) {
        for(j = 0; j < 3; j++) {
          /* config (p+mu*3+i, j) = complex<double> (tmp2[2*k], tmp2[2*k+1]); */

          n_uint64_t index = ((p+mu*3+i) * 3 + j) * 2;
	  gfield[index  ] = tmp2[2*k];
	  gfield[index+1] = tmp2[2*k+1];
          k++;
        }}
      }
      for(i = 0; i < 3; i++) {
      for(j = 0; j < 3; j++) {
        /* config (p+i, j) = complex<double> (tmp2[2*k], tmp2[2*k+1]); */

	n_uint64_t index = ((p+i) * 3 + j) * 2;
	gfield[index  ] = tmp2[2*k];
	gfield[index+1] = tmp2[2*k+1];

        k++; 	    
      }}
      if(status < 0 && status != LIME_EOR) {
        fprintf(stderr, "[read_lime_gauge_field_singleprec_timeslice] LIME read error occured with status = %d while reading file %s!\n Aborting...\n", 
          status, filename);
	exit(500);
      }
    }
  }}
  
  limeDestroyReader(limereader);
  fclose(ifs);
  if(timeslice==T_global-1) fprintf(stdout, "# [read_lime_gauge_field_singleprec_timeslice] checksum for gaugefield %s is %#x %#x\n",
            filename, (*checksum).suma, (*checksum).sumb); 
  return(0);
#else
  if(g_cart_id==0) fprintf(stderr, "[read_lime_gauge_field_singleprec_timeslice] Error, no MPI version\n ");
  return(-1);
#endif
}

/*************************************************************************************************
 *
 *
 *************************************************************************************************/
int read_ildg_nersc_gauge_field(const double * gauge, const char * filename) {

  FILE * ifs;
  int t, x, y, z, status;
  n_uint64_t bytes;
  char * header_type;
  LimeReader * limereader;
  float tmp[72], tmp2[72];
  int words_bigendian, mu, i, j;
  DML_Checksum checksum;
  DML_SiteRank rank;

  DML_checksum_init(&checksum);

  words_bigendian = big_endian();
  ifs = fopen(filename, "r");
  if(ifs == (FILE *)NULL) {
    fprintf(stderr, "[read_ildg_nersc_gauge_field] Could not open file %s\n Aborting...\n", filename);
    return(500);
  }
  limereader = limeCreateReader( ifs );
  if( limereader == (LimeReader *)NULL ) {
    fprintf(stderr, "[read_ildg_nersc_gauge_field] Unable to open LimeReader\n");
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(500);
  }
  while( (status = limeReaderNextRecord(limereader)) != LIME_EOF ) {
    if(status != LIME_SUCCESS ) {
      fprintf(stderr, "[read_ildg_nersc_gauge_field] limeReaderNextRecord returned error with status = %d!\n", status);
      status = LIME_EOF;
      break;
    }
    header_type = limeReaderType(limereader);
    fprintf(stdout, "# [read_ildg_nersc_gauge_field] header: %s\n", header_type);
    if( strcmp("ildg-binary-data",header_type) == 0) break;
  }
  if(status == LIME_EOF) {
    fprintf(stderr, "[read_ildg_nersc_gauge_field] no ildg-binary-data record found in file %s\n",filename);
    limeDestroyReader(limereader);
    fclose(ifs);
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(-2);
  }
  bytes = limeReaderBytes(limereader);
  if(bytes != (n_uint64_t)(LX*g_nproc_x)*(LY*g_nproc_y)*(LZ*g_nproc_z)*T*72*(n_uint64_t)sizeof(float)) {
    fprintf(stderr, "[read_ildg_nersc_gauge_field] Probably wrong lattice size or precision (bytes=%d) in file %s\n", (int)bytes, filename);
    fprintf(stderr, "[read_ildg_nersc_gauge_field] Aborting...!\n");
    fflush( stdout );
#ifdef HAVE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Finalize();
#endif
    exit(501);
  }

  status = read_nersc_gauge_binary_data_3col(limereader->fp, (double*)gauge, &checksum);
  if(status != 0) {
    fprintf(stderr, "[read_ildg_nersc_gauge_field] Error from read_nersc_gauge_binary_data_3col, status was %d\n", status);
    return(511);
  }

  limeDestroyReader(limereader);
  fclose(ifs);
#ifdef HAVE_MPI
  DML_checksum_combine(&checksum);
#endif
  if(g_cart_id==0) fprintf(stdout, "# [read_ildg_nersc_gauge_field] checksum for gaugefield %s is %#x %#x\n",
             filename, checksum.suma, checksum.sumb);
  return(0);
}

}
