

#define _FILE_OFFSET_BITS 64

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<time.h>
#include<sys/time.h>
#ifdef HAVE_MPI
# include<mpi.h>
# include<unistd.h>
#endif
#include<math.h>
#include"global.h"
#include"cvc_complex.h"
#include"cvc_linalg.h"

#ifdef __cplusplus
extern "C"
{
#endif

#  include"lime.h"

#ifdef __cplusplus
}
#endif


#include"read_input_parser.h"
#include"io_utils.h"
#include"dml.h"
#include"io.h"
#include"propagator_io.h"
#include"gauge_io.h"
#include "cvc_utils.h"

#define MAXBUF 1048576

namespace cvc {

int write_binary_gauge_data(LimeWriter * limewriter,
			    const int prec, DML_Checksum * ans) {

  int x, y, z, t, i, status=0;
  int tgeom[2];
  double tmp[72];
  double tmp3[72];
  float tmp2[72];
  double tick = 0, tock = 0;
  char measure[64];
  n_uint64_t bytes;
  DML_SiteRank rank;
#ifdef HAVE_MPI
  int tag=0, iproc;
  double *buffer;
  MPI_Status mpi_status;
#endif

  DML_checksum_init(ans);

#ifdef HAVE_MPI
  MPI_Barrier(g_cart_grid);
  tick = MPI_Wtime();
#endif

  if(prec == 32) bytes = (n_uint64_t)72*sizeof(float);
  else bytes = (n_uint64_t)72*sizeof(double);
  if(g_cart_id==0) {
    for(t = 0; t < T; t++) {
      for(z = 0; z < LZ; z++) {
      for(y = 0; y < LY; y++) {
      for(x = 0; x < LX; x++) {
        /* Rank should be computed by proc 0 only */
	rank = (DML_SiteRank) ((( (t+Tstart)*LZ + z)*LY + y)*LX + x);
	memcpy(tmp3   , g_gauge_field + _GGI(g_ipt[t][x][y][z],1), 18*sizeof(double));
	memcpy(tmp3+18, g_gauge_field + _GGI(g_ipt[t][x][y][z],2), 18*sizeof(double));
	memcpy(tmp3+36, g_gauge_field + _GGI(g_ipt[t][x][y][z],3), 18*sizeof(double));
	memcpy(tmp3+54, g_gauge_field + _GGI(g_ipt[t][x][y][z],0), 18*sizeof(double));

#ifndef WORDS_BIGENDIAN
	if(prec == 32) {
          byte_swap_assign_double2single(tmp2, tmp3, 72);
	  DML_checksum_accum(ans, rank, (char*) tmp2, 72*sizeof(float));
	  status = limeWriteRecordData((void*)&tmp2, &bytes, limewriter);
	}
	else {
	  byte_swap_assign(tmp, tmp3, 72);
	  DML_checksum_accum(ans, rank, (char*) tmp, 72*sizeof(double));
	  status = limeWriteRecordData((void*)&tmp, &bytes, limewriter);
	}
#else
        if(prec == 32) {
          double2single(tmp2, tmp3, 72);
	  DML_checksum_accum(ans, rank, (char*) tmp2, 72*sizeof(float));
	  status = limeWriteRecordData((void*)&tmp2, &bytes, limewriter);
	}
        else {
          DML_checksum_accum(ans, rank, (char*) tmp3, 72*sizeof(double));
	  status = limeWriteRecordData((void*)&tmp3, &bytes, limewriter);
	}
#endif
      }
      }
      }
    }
  }
#ifdef HAVE_MPI

  tgeom[0] = Tstart;
  tgeom[1] = T;
  for(iproc=1; iproc<g_nproc; iproc++) {
    if(g_cart_id==0) {

      tag = 2 * iproc;
      MPI_Recv((void*)tgeom, 2, MPI_INT, iproc, tag, g_cart_grid, &mpi_status);
      fprintf(stdout, "# tgeom[%d] = (%d, %d)\n", iproc, tgeom[0], tgeom[1]);
      if( (buffer = (double*)malloc(72*LX*LY*LZ*sizeof(double)) ) == (double*)NULL ) {
        fprintf(stderr, "Error from malloc for buffer\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Finalize();
        exit(500);
      }

      for(t=0; t<tgeom[1]; t++) {

        tag = 2* ( t*g_nproc + iproc ) + 1;
        MPI_Recv((void*)buffer, LX*LY*LZ*72, MPI_DOUBLE, iproc, tag, g_cart_grid, &mpi_status);

        for(z=0; z<LZ; z++) {
        for(y=0; y<LY; y++) {
        for(x=0; x<LX; x++) {

          /* Rank should be computed by proc 0 only */
          rank = (DML_SiteRank) ((( (t+tgeom[0])*LZ + z)*LY + y)*LX + x);
          i = ( x*LY + y )*LZ + z;
          memcpy(tmp3   , buffer + _GGI(i,1), 18*sizeof(double));
          memcpy(tmp3+18, buffer + _GGI(i,2), 18*sizeof(double));
          memcpy(tmp3+36, buffer + _GGI(i,3), 18*sizeof(double));
          memcpy(tmp3+54, buffer + _GGI(i,0), 18*sizeof(double));

#  ifndef WORDS_BIGENDIAN
          if(prec == 32) {
            byte_swap_assign_double2single(tmp2, tmp3, 72);
            DML_checksum_accum(ans, rank, (char*) tmp2, 72*sizeof(float));
            status = limeWriteRecordData((void*)&tmp2, &bytes, limewriter);
          }
          else {
            byte_swap_assign(tmp, tmp3, 72);
            DML_checksum_accum(ans, rank, (char*) tmp, 72*sizeof(double));
            status = limeWriteRecordData((void*)&tmp, &bytes, limewriter);
          }
#  else
          if(prec == 32) {
            double2single(tmp2, tmp3, 72);
            DML_checksum_accum(ans, rank, (char*) tmp2, 72*sizeof(float));
            status = limeWriteRecordData((void*)&tmp2, &bytes, limewriter);
          }
          else {
            DML_checksum_accum(ans, rank, (char*) tmp3, 72*sizeof(double));
            status = limeWriteRecordData((void*)&tmp3, &bytes, limewriter);
          }
#  endif
        }
        }
        }
      }
      free(buffer);
    }  /* of if g_cart_id == 0 */

    if(g_cart_id==iproc) {
      tag = 2 * iproc;
      MPI_Send((void*)tgeom, 2, MPI_INT, 0, tag, g_cart_grid);
      for(t=0; t<T; t++) {
        tag = 2 * ( t*g_nproc + iproc ) + 1;
        MPI_Send((void*)(g_gauge_field + _GGI(g_ipt[t][0][0][0],0)), 72*LX*LY*LZ, MPI_DOUBLE, 0, tag, g_cart_grid);
      }
    }
    
    MPI_Barrier(g_cart_grid);
    if(g_cart_id==0) {
      tock = MPI_Wtime();
      fprintf(stdout, "# time spent writing: %e\n", tock-tick);
    }
  }

#endif

  return(0);

}


int write_lime_gauge_field(char * filename, const double plaq, const int counter, const int prec) {
  FILE * ofs = NULL;
  LimeWriter * limewriter = NULL;
  LimeRecordHeader * limeheader = NULL;
  /* Message end and Message begin flag */
  int ME_flag=0, MB_flag=0, status=0;
  n_uint64_t bytes;
  DML_Checksum checksum;

  write_xlf_info(plaq, counter, filename, 0, (char*)NULL);

  if(g_cart_id == 0) {
    ofs = fopen(filename, "a");
    if(ofs == (FILE*)NULL) {
      fprintf(stderr, "Could not open file %s for writing!\n Aboring...\n", filename);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }
    limewriter = limeCreateWriter( ofs );
    if(limewriter == (LimeWriter*)NULL) {
      fprintf(stderr, "LIME error in file %s for writing!\n Aboring...\n", filename);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }
    write_ildg_format_xml("temp.xml", limewriter, prec); 

    bytes = ((n_uint64_t)LX)*((n_uint64_t)LY)*((n_uint64_t)LZ)*((n_uint64_t)T_global)*((n_uint64_t)72*sizeof(double));
    if(prec == 32) bytes = bytes/((n_uint64_t)2);
    MB_flag=0; ME_flag=0;
    limeheader = limeCreateHeader(MB_flag, ME_flag, "ildg-binary-data", bytes);
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
  }

  write_binary_gauge_data(limewriter, prec, &checksum);
  if(g_proc_id == 0) {
    printf("# checksum for Gauge field written to file %s is %#x %#x\n",
	   filename, checksum.suma, checksum.sumb);
  }

  if(g_cart_id == 0) {
    limeDestroyWriter( limewriter );
    fflush(ofs);
    fclose(ofs);
  }
  write_checksum(filename, &checksum);

  return(0);
}

/***********************************************************************
 * write_ildg_format_xml
 ***********************************************************************/

int write_ildg_format_xml(char *filename, LimeWriter * limewriter, const int prec){
  FILE * ofs;
  n_uint64_t bytes, bytes_left, bytes_to_copy;
  int MB_flag=1, ME_flag=0, status=0;
  LimeRecordHeader * limeheader;
  char buf[MAXBUF];

  ofs = fopen(filename, "w");

  fprintf(ofs, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
  fprintf(ofs, "<ildgFormat xmlns=\"http://www.lqcd.org/ildg\"\n");
  fprintf(ofs, "            xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
  fprintf(ofs, "            xsi:schemaLocation=\"http://www.lqcd.org/ildg filefmt.xsd\">\n");
  fprintf(ofs, "  <version> 1.0 </version>\n");
  fprintf(ofs, "  <field> su3gauge </field>\n");
  fprintf(ofs, "  <precision> %d </precision>\n", prec);
  fprintf(ofs, "  <lx> %d </lx>\n", LX);
  fprintf(ofs, "  <ly> %d </ly>\n", LY);
  fprintf(ofs, "  <lz> %d </lz>\n", LZ);
  fprintf(ofs, "  <lt> %d </lt>\n", T_global);
  fprintf(ofs, "</ildgFormat>");
  fclose( ofs );
  ofs = fopen(filename, "r");
  bytes = file_size(ofs);

  limeheader = limeCreateHeader(MB_flag, ME_flag, "ildg-format", bytes);
  if(limeheader == (LimeRecordHeader*)NULL) {
    fprintf(stderr, "LIME create header ildg-format error\n Aborting...\n");
    exit(500);
  }
  status = limeWriteRecordHeader( limeheader, limewriter);
  if(status < 0 ) {
    fprintf(stderr, "LIME write header ildg-format error %d\n Aborting...\n", status);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
    exit(500);
  }
  limeDestroyHeader( limeheader );

  /* Buffered copy */
  bytes_left = bytes;
  while(bytes_left > 0){
    if(MAXBUF < bytes_left) {
      bytes_to_copy = MAXBUF;
    }
    else bytes_to_copy = bytes_left;
    if( bytes_to_copy != fread(buf,1,bytes_to_copy,ofs))
      {
        fprintf(stderr, "Error reading %s\n", filename);
        return EXIT_FAILURE;
      }

    status = limeWriteRecordData(buf, &bytes_to_copy, limewriter);
    if (status != 0) {
      fprintf(stderr, "LIME error writing ildg-format status = %d\n Aborting...\n", status);
#ifdef HAVE_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Finalize();
#endif
      exit(500);
    }
    bytes_left -= bytes_to_copy;
  }


  fclose( ofs );
  return(0);
}


/***********************************************************************
 * write_xlf_info
 ***********************************************************************/

int write_xlf_info(const double plaq, const int counter, char * filename,
                   const int append, char * data_buf) {
  FILE * ofs;
  LimeWriter * limewriter = NULL;
  LimeRecordHeader * limeheader = NULL;
  /* Message end and Message begin flag */
  int ME_flag=1, MB_flag=1, status=0;
#ifdef HAVE_MPI
  MPI_Status mpi_status;
#endif
  char message[5000];
  char * buf;
  n_uint64_t bytes;
  struct timeval t1;


  gettimeofday(&t1,NULL);
  sprintf(message,"\n plaquette = %e\n Nconf = %d\n kappa = %f, mu = %f\n time = %ld\n date = %s",
            plaq, counter, g_kappa, g_mu, t1.tv_sec, ctime(&t1.tv_sec));

  if(data_buf != (char*)NULL) {
    bytes = strlen( data_buf );
    buf = data_buf;
  }
  else {
    bytes = strlen( message );
    buf = message;
  }
  if(g_cart_id == 0) {
    if(append) {
      ofs = fopen(filename, "a");
    }
    else ofs = fopen(filename, "w");
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

    limeheader = limeCreateHeader(MB_flag, ME_flag, "xlf-info", bytes);
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
    limeWriteRecordData(buf, &bytes, limewriter);

    limeDestroyWriter( limewriter );
    fflush(ofs);
    fclose(ofs);

  }
  return(0);
}

/***************************************
 * file_size
 ***************************************/
n_uint64_t file_size(FILE *fp)
{
  n_uint64_t oldpos = ftello(fp);
  n_uint64_t length;

  if (fseeko(fp, 0L,SEEK_END) == -1)
    return -1;

  length = ftello(fp);

  return ( fseeko(fp,oldpos,SEEK_SET) == -1 ) ? -1 : length;

}

int read_nersc_gauge_field(double*s, char*filename, double *plaq) {
#if (defined PARALLELTX) || (defined PARALLELTXY)  || (defined PARALLELTXYZ) 
  EXIT_WITH_MSG(1, "[read_nersc_gauge_field] >1-dim. parallel version not yet implemented\n");
#endif
/*
  this is the NERSC header format:
  BEGIN_HEADER
  HDR_VERSION = 1.0
  DATATYPE = 4D_SU3_GAUGE
  STORAGE_FORMAT = 1.0
  DIMENSION_1 = 32
  DIMENSION_2 = 32
  DIMENSION_3 = 32
  DIMENSION_4 = 64
  PLAQUETTE = 0.59371401
  LINK_TRACE = -0.00007994
  CHECKSUM = 5523B964
  BOUNDARY_1 = PERIODIC
  BOUNDARY_2 = PERIODIC
  BOUNDARY_3 = PERIODIC
  BOUNDARY_4 = PERIODIC
  ENSEMBLE_ID = Nf0_b6p00_L32T64
  SEQUENCE_NUMBER = 10000
  FLOATING_POINT = IEEE32BIG
  CREATION_DATE = 06:58:49 11/07/2011 
  CREATOR = hopper06
  END_HEADER
*/

  char line[100], item[50], value[50];
  int linelength=99;
  int l1=0, l2=0, l3=0, l4=0;
  int x1, x2, x3, x4;
  unsigned int lvol, iy, step, mu, idx;
  int ix;
  int end_flag=0;
  int length;
  float *ftmp=NULL;
  FILE *ifs=NULL;
  size_t iread;
  uint64_t bytes;
  uint32_t checksum, *iptr, utmp, cks;
  int words_bigendian = big_endian();
  int nu[4];
  double u[12], U_[18];
  // double myplaq=0.;
#ifdef HAVE_MPI
  int ibuf;
#endif

  ifs = fopen(filename, "r");
  if(ifs == NULL) {
    EXIT_WITH_MSG(1, "[read_nersc_gauge_field] Error, could not open file for reading\n");
  }

  if( fgets(line, linelength, ifs) == NULL ) {
    EXIT_WITH_MSG(2, "[read_nersc_gauge_field] Error, could not read first line\n");
  }
  length=strlen(line);
  line[length-1] = '\0';

  if(strcmp(line, "BEGIN_HEADER") != 0 ) {
    EXIT_WITH_MSG(3, "[read_nersc_gauge_field] Error, could not find beginning of header\n");
  }

  while( fgets(line, linelength, ifs) != NULL ) {
    length=strlen(line);
    line[length-1] = '\0';
    if(strcmp(line, "END_HEADER") ==0 ) {
      end_flag=1;
      break;
    }
    sscanf(line, "%s = %s", item, value);

    if(g_cart_id==0) fprintf(stdout, "# [read_nersc_gauge_field] item=%s; value=%s\n", item, value);

    if     (strcmp(item, "DIMENSION_1") == 0) { l1 = atoi(value); }
    else if(strcmp(item, "DIMENSION_2") == 0) { l2 = atoi(value); }
    else if(strcmp(item, "DIMENSION_3") == 0) { l3 = atoi(value); }
    else if(strcmp(item, "DIMENSION_4") == 0) { l4 = atoi(value); }
    else if(strcmp(item, "PLAQUETTE")   == 0) { sscanf(line, "%s = %lf", item, plaq); }
    else if(strcmp(item, "CHECKSUM")    == 0) { sscanf(line, "%s = %x", value, &cks); }
  }

  if(!end_flag) { EXIT_WITH_MSG(4, "[read_nersc_gauge_field] Error, could not reach end of header\n"); }

  if(g_cart_id==0) fprintf(stdout, "# [read_nersc_gauge_field] Reached end of header, read binary section\n");
  
  if(g_cart_id==0) fprintf(stdout, "# [read_nersc_gauge_field] l1=%d, l2=%d, l3=%d, l4=%d\n",l1, l2, l3, l4);

//  lvol = l1 * l2 * l3 * l4;
  lvol = l1 * l2 * l3 * T;

  if(lvol==0) { EXIT_WITH_MSG(5, "[read_nersc_gauge_field] Error, zero volume\n"); }
 
  ftmp = (float*)malloc(lvol*48*sizeof(float));
  if(ftmp == NULL) { EXIT_WITH_MSG(6, "[read_nersc_gauge_field] Error, could not alloc ftmp\n"); }

#ifdef HAVE_MPI
  bytes = l1*l2*l3*g_proc_coords[0]*T*48*sizeof(float);
  fseek(ifs,bytes, SEEK_CUR);
#endif

  iread = fread(ftmp, sizeof(float), 48*lvol, ifs);
  if(iread != 48*lvol) { EXIT_WITH_MSG(7, "[read_nersc_gauge_field] Error, could not read proper amount of data\n"); }
  fclose(ifs);

  if(!words_bigendian) {
    byte_swap(ftmp, 48*lvol);
  }

  bytes = 48*lvol*sizeof(float);
  step=sizeof(uint32_t);
  if(g_cart_id == 0) fprintf(stdout, "# [read_nersc_gauge_field] step size = %u bytes\n", step);
  iptr = (uint32_t*)ftmp;
  checksum = 0;
  for(ix=0;ix<bytes;ix+=step) {
    checksum += *iptr;
    iptr++;
  }
#ifdef HAVE_MPI
  MPI_Allreduce(&checksum, &ibuf, 1, MPI_INT, MPI_SUM, g_cart_grid);
  checksum = ibuf;
#endif
  if(g_cart_id == 0) fprintf(stdout, "# [read_nersc_gauge_field] checksum: read  %#x; calculated %#x\n", cks, checksum);
  if(cks != checksum) { EXIT_WITH_MSG(8, "[read_nersc_gauge_field] Error, checksums do not match\n"); }

  nu[0] = 1;
  nu[1] = 2;
  nu[2] = 3;
  nu[3] = 0;
  // reconstruct gauge field
  // - assume index formula idx = (((t*LX+x)*LY+y)*LZ+z)*(4*3*2*2) + mu*(3*2*2) + 2*(3*u+c)+r
  //   with mu=0,1,2,3; u=0,1; c=0,1,2, r=0,1
  iy = 0;
  
  for(x4=0; x4<T; x4++) {  // t
  for(x3=0; x3<l3; x3++) {  // z
  for(x2=0; x2<l2; x2++) {  // y
  for(x1=0; x1<l1; x1++) {  // x
    ix = g_ipt[x4][x1][x2][x3];

    for(mu=0;mu<4;mu++) {
      idx = 48*iy + 12*mu;

      single2double(u, (float*)(ftmp+idx), 12);

/*
      if(iy>=200000 && iy<=400000) {
        fprintf(stdout, "# [] ix = %u\n", ix);
        fprintf(stdout, "\tu = %f +I %f; %f +I %f; %f +I %f\n", u[0], u[1], u[2], u[3], u[4], u[5]);
        fprintf(stdout, "\tv = %f +I %f; %f +I %f; %f +I %f\n", u[6], u[7], u[8], u[9], u[10], u[11]);
      }
*/
      memcpy(U_, u, 12*sizeof(double));

      U_[12] =  (u[ 2]*u[10] - u[ 3]*u[11]) - (u[ 4]*u[ 8] - u[ 5]*u[ 9]);
      U_[13] = -(u[ 2]*u[11] + u[ 3]*u[10]) + (u[ 4]*u[ 9] + u[ 5]*u[ 8]);
      U_[14] =  (u[ 4]*u[ 6] - u[ 5]*u[ 7]) - (u[ 0]*u[10] - u[ 1]*u[11]);
      U_[15] = -(u[ 4]*u[ 7] + u[ 5]*u[ 6]) + (u[ 0]*u[11] + u[ 1]*u[10]);
      U_[16] =  (u[ 0]*u[ 8] - u[ 1]*u[ 9]) - (u[ 2]*u[ 6] - u[ 3]*u[ 7]);
      U_[17] = -(u[ 0]*u[ 9] + u[ 1]*u[ 8]) + (u[ 2]*u[ 7] + u[ 3]*u[ 6]);

      // fprintf(stdout, "\tix = %d, mu = %d, nu = %d, offset = %d\n", ix, mu, nu[mu], _GGI(ix,nu[mu]));
      // fprintf(stdout, "ix = %u, mu = \n", ix);
      //fflush(stdout);

      memcpy(s+_GGI(ix,nu[mu]), U_, 18*sizeof(double));

/*
      if(iy>=0 && iy<=10000) {
        fprintf(stdout, "# iy = %u\n", iy);
        fprintf(stdout, "\tU[1,1] <- %f + %f*1.i\n", U_[ 0], U_[ 1]);
        fprintf(stdout, "\tU[1,2] <- %f + %f*1.i\n", U_[ 2], U_[ 3]);
        fprintf(stdout, "\tU[1,3] <- %f + %f*1.i\n", U_[ 4], U_[ 5]);
        fprintf(stdout, "\tU[2,1] <- %f + %f*1.i\n", U_[ 6], U_[ 7]);
        fprintf(stdout, "\tU[2,2] <- %f + %f*1.i\n", U_[ 8], U_[ 9]);
        fprintf(stdout, "\tU[2,3] <- %f + %f*1.i\n", U_[10], U_[11]);
        fprintf(stdout, "\tU[3,1] <- %f + %f*1.i\n", U_[12], U_[13]);
        fprintf(stdout, "\tU[3,2] <- %f + %f*1.i\n", U_[14], U_[15]);
        fprintf(stdout, "\tU[3,3] <- %f + %f*1.i\n", U_[16], U_[17]);
      }
*/
    }

    iy++;
  }}}}

  free(ftmp);
  return(0);

}

int read_nersc_gauge_field_timeslice(double*s, char*filename, int timeslice, uint32_t *checksum) {
/*
  this is the NERSC header format:
  BEGIN_HEADER
  HDR_VERSION = 1.0
  DATATYPE = 4D_SU3_GAUGE
  STORAGE_FORMAT = 1.0
  DIMENSION_1 = 32
  DIMENSION_2 = 32
  DIMENSION_3 = 32
  DIMENSION_4 = 64
  PLAQUETTE = 0.59371401
  LINK_TRACE = -0.00007994
  CHECKSUM = 5523B964
  BOUNDARY_1 = PERIODIC
  BOUNDARY_2 = PERIODIC
  BOUNDARY_3 = PERIODIC
  BOUNDARY_4 = PERIODIC
  ENSEMBLE_ID = Nf0_b6p00_L32T64
  SEQUENCE_NUMBER = 10000
  FLOATING_POINT = IEEE32BIG
  CREATION_DATE = 06:58:49 11/07/2011 
  CREATOR = hopper06
  END_HEADER
*/

  char line[100], item[50], value[50];
  int linelength=99;
  int l1=0, l2=0, l3=0, l4=0;
  int x1, x2, x3, x4;
  unsigned int lvol, ix, iy, step, mu, idx, lvol3;
  int end_flag=0;
  int length;
  float *ftmp=NULL;
  FILE *ifs=NULL;
  size_t iread;
  uint64_t bytes;
  uint32_t *iptr, utmp, cks;
  int words_bigendian = big_endian();
  int nu[4];
  int return_value=0;
  double u[12], U_[18], plaq;

  ifs = fopen(filename, "r");
  if(ifs == NULL) {
    fprintf(stderr, "[] Error, could not open file %s for reading\n", filename);
    return(1);
  }

  if( fgets(line, linelength, ifs) == NULL ) {
    fprintf(stderr, "[] Error, could not read first line\n");
    return(2);
  }
  length=strlen(line);
  line[length-1] = '\0';

  if(strcmp(line, "BEGIN_HEADER") != 0 ) {
    fprintf(stderr, "[] Error, could not find beginning of header\n");
    return(3);
  }

  while( fgets(line, linelength, ifs) != NULL ) {
    length=strlen(line);
    line[length-1] = '\0';
    if(strcmp(line, "END_HEADER") ==0 ) {
      end_flag=1;
      break;
    }
    sscanf(line, "%s = %s", item, value);

    // fprintf(stdout, "# [] item=%s; value=%s\n", item, value);

    if     (strcmp(item, "DIMENSION_1") == 0) { l1 = atoi(value); }
    else if(strcmp(item, "DIMENSION_2") == 0) { l2 = atoi(value); }
    else if(strcmp(item, "DIMENSION_3") == 0) { l3 = atoi(value); }
    else if(strcmp(item, "DIMENSION_4") == 0) { l4 = atoi(value); }
    else if(strcmp(item, "PLAQUETTE")   == 0) { sscanf(line, "%s = %lf", item, &plaq); }
    else if(strcmp(item, "CHECKSUM")    == 0) { sscanf(line, "%s = %x", value, &cks); }
  }

  if(!end_flag) {
    fprintf(stderr, "[] Error, could not reach end of header\n");
    return(4);
  }

  // fprintf(stdout, "# [] Reached end of header, read binary section\n");
  
  // fprintf(stdout, "# [] l1=%d, l2=%d, l3=%d, l4=%d\n",l1, l2, l3, l4);

  lvol  = l1 * l2 * l3 * l4;
  lvol3 = l1 * l2 * l3;

  if(lvol==0) {
    fprintf(stderr, "[] Error, zero volume\n");
    return(5);
  }
 
  ftmp = (float*)malloc(lvol3*48*sizeof(float));
  if(ftmp == NULL) {
    fprintf(stderr, "[] Error, could not alloc ftmp\n");
    return(6);
  }

  bytes = timeslice*48*lvol3*sizeof(float);
  if( fseek(ifs, bytes, SEEK_CUR) != 0 ) {
    fprintf(stderr, "[] Error, could not seek to file position\n");
    return(7);
  }

  iread = fread(ftmp, sizeof(float), 48*lvol3, ifs);
  fclose(ifs);
  if(iread != 48*lvol3) {
    fprintf(stderr, "[] Error, could not read proper amount of data\n");
    return(7);
  }

  if(!words_bigendian) {
    byte_swap(ftmp, 48*lvol3);
  }

  bytes = 48*lvol3*sizeof(float);
  step=sizeof(uint32_t);
  //fprintf(stdout, "# [] step size = %u bytes\n", step);
  iptr = (uint32_t*)ftmp;
  if(timeslice==0) *checksum = 0;
  for(ix=0;ix<bytes;ix+=step) {
    *checksum += *iptr;
    iptr++;
  }

  if(timeslice==T_global-1) {
    fprintf(stdout, "# [] checksum: read  %#x; calculated %#x\n", cks, *checksum);
    if(cks != *checksum) {
      fprintf(stderr, "[] Warning, checksums do not match\n");
      return_value = 8;
    }
  }

  nu[0] = 1;
  nu[1] = 2;
  nu[2] = 3;
  nu[3] = 0;
  // reconstruct gauge field
  // - assume index formula idx = (((t*LX+x)*LY+y)*LZ+z)*(4*3*2*2) + mu*(3*2*2) + 2*(3*u+c)+r
  //   with mu=0,1,2,3; u=0,1; c=0,1,2, r=0,1
  iy = 0;
  for(x3=0; x3<l3; x3++) {  // z
  for(x2=0; x2<l2; x2++) {  // y
  for(x1=0; x1<l1; x1++) {  // x
    ix = g_ipt[0][x1][x2][x3];
    for(mu=0;mu<4;mu++) {
      idx = 48*iy + 12*mu;
      single2double(u, (float*)(ftmp+idx), 12);
/*
      if(iy>=200000 && iy<=400000) {
        fprintf(stdout, "# [] ix = %u\n", ix);
        fprintf(stdout, "\tu = %f +I %f; %f +I %f; %f +I %f\n", u[0], u[1], u[2], u[3], u[4], u[5]);
        fprintf(stdout, "\tv = %f +I %f; %f +I %f; %f +I %f\n", u[6], u[7], u[8], u[9], u[10], u[11]);
      }
*/
      memcpy(U_, u, 12*sizeof(double));

      U_[12] =  (u[ 2]*u[10] - u[ 3]*u[11]) - (u[ 4]*u[ 8] - u[ 5]*u[ 9]);
      U_[13] = -(u[ 2]*u[11] + u[ 3]*u[10]) + (u[ 4]*u[ 9] + u[ 5]*u[ 8]);
      U_[14] =  (u[ 4]*u[ 6] - u[ 5]*u[ 7]) - (u[ 0]*u[10] - u[ 1]*u[11]);
      U_[15] = -(u[ 4]*u[ 7] + u[ 5]*u[ 6]) + (u[ 0]*u[11] + u[ 1]*u[10]);
      U_[16] =  (u[ 0]*u[ 8] - u[ 1]*u[ 9]) - (u[ 2]*u[ 6] - u[ 3]*u[ 7]);
      U_[17] = -(u[ 0]*u[ 9] + u[ 1]*u[ 8]) + (u[ 2]*u[ 7] + u[ 3]*u[ 6]);
      memcpy(s+_GGI(ix,nu[mu]), U_, 18*sizeof(double));
/*
      if(iy>=0 && iy<=10000) {
        fprintf(stdout, "# iy = %u\n", iy);
        fprintf(stdout, "\tU[1,1] <- %f + %f*1.i\n", U_[ 0], U_[ 1]);
        fprintf(stdout, "\tU[1,2] <- %f + %f*1.i\n", U_[ 2], U_[ 3]);
        fprintf(stdout, "\tU[1,3] <- %f + %f*1.i\n", U_[ 4], U_[ 5]);
        fprintf(stdout, "\tU[2,1] <- %f + %f*1.i\n", U_[ 6], U_[ 7]);
        fprintf(stdout, "\tU[2,2] <- %f + %f*1.i\n", U_[ 8], U_[ 9]);
        fprintf(stdout, "\tU[2,3] <- %f + %f*1.i\n", U_[10], U_[11]);
        fprintf(stdout, "\tU[3,1] <- %f + %f*1.i\n", U_[12], U_[13]);
        fprintf(stdout, "\tU[3,2] <- %f + %f*1.i\n", U_[14], U_[15]);
        fprintf(stdout, "\tU[3,3] <- %f + %f*1.i\n", U_[16], U_[17]);
      }
*/
    }
    iy++;
  }}}

  free(ftmp);
  return(return_value);

}


int read_nersc_gauge_binary_data_3col(FILE*ifs, double*s, DML_Checksum*ans) {
/*
  this is the NERSC header format:
  BEGIN_HEADER
  HDR_VERSION = 1.0
  DATATYPE = 4D_SU3_GAUGE
  STORAGE_FORMAT = 1.0
  DIMENSION_1 = 32
  DIMENSION_2 = 32
  DIMENSION_3 = 32
  DIMENSION_4 = 64
  PLAQUETTE = 0.59371401
  LINK_TRACE = -0.00007994
  CHECKSUM = 5523B964
  BOUNDARY_1 = PERIODIC
  BOUNDARY_2 = PERIODIC
  BOUNDARY_3 = PERIODIC
  BOUNDARY_4 = PERIODIC
  ENSEMBLE_ID = Nf0_b6p00_L32T64
  SEQUENCE_NUMBER = 10000
  FLOATING_POINT = IEEE32BIG
  CREATION_DATE = 06:58:49 11/07/2011 
  CREATOR = hopper06
  END_HEADER
*/

  char line[100], item[50], value[50];
  int linelength=99;
  int x1, x2, x3, x4;
  unsigned int lvol=VOLUME, ix, iy, step, mu, idx;
  int end_flag=0;
  int length;
  float *ftmp=NULL, res;
  size_t iread;
  uint64_t bytes;
  uint32_t checksum, *iptr, utmp, cks;
  int words_bigendian = big_endian();
  int nu[4];
  double u[12], U_[18];
  float *buffer=NULL;

  if(ifs == NULL) {
    fprintf(stderr, "[] fp is NULL\n");
    return(1);
  }

  ftmp = (float*)malloc(lvol*72*sizeof(float));
  if(ftmp == NULL) {
    fprintf(stderr, "[] Error, could not alloc ftmp\n");
    return(6);
  }

  iread = fread(ftmp, sizeof(float), 72*lvol, ifs);
  if(iread != 72*lvol) {
    fprintf(stderr, "[] Error, could not read proper amount of data\n");
    return(7);
  }

/*
  buffer = (float*)malloc(48*lvol*sizeof(float));
  for(ix=0;ix<lvol;ix++) {
    for(mu=0;mu<4;mu++) {
      memcpy(buffer+48*ix+12*mu, ftmp+72*ix+18*mu, 12*sizeof(float));
    }
  }
*/
  if(!words_bigendian) {
    fprintf(stdout, "# [] performing byte swap\n");
    byte_swap(ftmp, 72*lvol);
//    byte_swap(buffer, 48*lvol);
  }
/*
  for(ix=0;ix<lvol;ix++) {
   for(mu=0;mu<4;mu++) {
     check_F_SU3(ftmp+72*ix+18*mu, &res);
     fprintf(stdout, "%10d%16.7e\n", ix, res);
   }
  }
  fflush(stdout);
*/
/*
  bytes = 48*lvol*sizeof(float);
  step=sizeof(uint32_t);
  fprintf(stdout, "# [] step size = %u bytes\n", step);
  iptr = (uint32_t*)buffer;
  checksum = 0;
  for(ix=0;ix<bytes;ix+=step) {
    checksum += *iptr;
    iptr++;
  }
  fprintf(stdout, "# [] checksum: read  %#lx; calculated %#lx\n", cks, checksum);
  if(cks != checksum) {
    fprintf(stderr, "[] Error, checksums do not match\n");
    return(8);
  }
*/

  nu[0] = 1;
  nu[1] = 2;
  nu[2] = 3;
  nu[3] = 0;
  // reconstruct gauge field
  // - assume index formula idx = (((t*LX+x)*LY+y)*LZ+z)*(4*3*2*2) + mu*(3*2*2) + 2*(3*u+c)+r
  //   with mu=0,1,2,3; u=0,1; c=0,1,2, r=0,1
  iy = 0;
  for(x4=0; x4<T; x4++) {  // t
  for(x3=0; x3<LZ; x3++) {  // z
  for(x2=0; x2<LY; x2++) {  // y
  for(x1=0; x1<LX; x1++) {  // x
    ix = g_ipt[x4][x1][x2][x3];
    for(mu=0;mu<4;mu++) {
      idx = 72*iy + 18*mu;
      single2double(U_, (float*)(ftmp+idx), 18);
      memcpy(s+_GGI(ix,nu[mu]), U_, 18*sizeof(double));
/*
      s[_GGI(ix,nu[mu]) +  0 ] = U_[ 0];
      s[_GGI(ix,nu[mu]) +  1 ] = U_[ 1];
      s[_GGI(ix,nu[mu]) +  2 ] = U_[ 6];
      s[_GGI(ix,nu[mu]) +  3 ] = U_[ 7];
      s[_GGI(ix,nu[mu]) +  4 ] = U_[12];
      s[_GGI(ix,nu[mu]) +  5 ] = U_[13];
      s[_GGI(ix,nu[mu]) +  6 ] = U_[ 2];
      s[_GGI(ix,nu[mu]) +  7 ] = U_[ 3];
      s[_GGI(ix,nu[mu]) +  8 ] = U_[ 8];
      s[_GGI(ix,nu[mu]) +  9 ] = U_[ 9];
      s[_GGI(ix,nu[mu]) + 10 ] = U_[14];
      s[_GGI(ix,nu[mu]) + 11 ] = U_[15];
      s[_GGI(ix,nu[mu]) + 12 ] = U_[ 4];
      s[_GGI(ix,nu[mu]) + 13 ] = U_[ 5];
      s[_GGI(ix,nu[mu]) + 14 ] = U_[10];
      s[_GGI(ix,nu[mu]) + 15 ] = U_[11];
      s[_GGI(ix,nu[mu]) + 16 ] = U_[16];
      s[_GGI(ix,nu[mu]) + 17 ] = U_[17];
*/
    }
    iy++;
  }}}}

  free(ftmp);
  return(0);
}


int read_nersc_gauge_field_3x3(double*s, char*filename, double *plaq) {
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  EXIT_WITH_MSG(1, "\n[read_nersc_gauge_field_3x3] Error, 2- and 3-dim. parallel versions not yet implemented; exit\n");
#endif
/*
  this is the NERSC header format:
  BEGIN_HEADER
  HDR_VERSION = 1.0
  DATATYPE = 4D_SU3_GAUGE
  STORAGE_FORMAT = 1.0
  DIMENSION_1 = 32
  DIMENSION_2 = 32
  DIMENSION_3 = 32
  DIMENSION_4 = 64
  PLAQUETTE = 0.59371401
  LINK_TRACE = -0.00007994
  CHECKSUM = 5523B964
  BOUNDARY_1 = PERIODIC
  BOUNDARY_2 = PERIODIC
  BOUNDARY_3 = PERIODIC
  BOUNDARY_4 = PERIODIC
  ENSEMBLE_ID = Nf0_b6p00_L32T64
  SEQUENCE_NUMBER = 10000
  FLOATING_POINT = IEEE32BIG
  CREATION_DATE = 06:58:49 11/07/2011 
  CREATOR = hopper06
  END_HEADER
*/

  char line[100], item[50], value[50];
  int linelength=99;
  int l1=0, l2=0, l3=0, l4=0;
  int x1, x2, x3, x4;
  unsigned int lvol, ix, iy, step, mu, idx;
  int end_flag=0;
  int length;
  double *ftmp=NULL;
  FILE *ifs=NULL;
  size_t iread;
  uint64_t bytes;
  uint32_t checksum, *iptr, utmp, cks;
  int words_bigendian = big_endian();
  int nu[4];
  double u[12], U_[18];
#ifdef HAVE_MPI
  int ibuf;
  long int offset;
#endif

  ifs = fopen(filename, "r");
  if(ifs == NULL) {
    if(g_cart_id==0) fprintf(stderr, "[] Error, could not open file %s for reading\n", filename);
    EXIT(1);
  }

  if( fgets(line, linelength, ifs) == NULL ) {
    if(g_cart_id==0) fprintf(stderr, "[] Error, could not read first line\n");
    EXIT(2);
  }
  length=strlen(line);
  line[length-1] = '\0';

  if(strcmp(line, "BEGIN_HEADER") != 0 ) {
    if(g_cart_id==0) fprintf(stderr, "[] Error, could not find beginning of header\n");
    EXIT(3);
  }

  while( fgets(line, linelength, ifs) != NULL ) {
    length=strlen(line);
    line[length-1] = '\0';
    if(strcmp(line, "END_HEADER") ==0 ) {
      end_flag=1;
      break;
    }
    sscanf(line, "%s = %s", item, value);

    fprintf(stdout, "# [] item=%s; value=%s\n", item, value);

    if     (strcmp(item, "DIMENSION_1") == 0) { l1 = atoi(value); }
    else if(strcmp(item, "DIMENSION_2") == 0) { l2 = atoi(value); }
    else if(strcmp(item, "DIMENSION_3") == 0) { l3 = atoi(value); }
    else if(strcmp(item, "DIMENSION_4") == 0) { l4 = atoi(value); }
    else if(strcmp(item, "PLAQUETTE")   == 0) { sscanf(line, "%s = %lf", item, plaq); }
    else if(strcmp(item, "CHECKSUM")    == 0) { sscanf(line, "%s = %x", value, &cks); }
  }

  if(!end_flag) {
    if(g_cart_id==0) fprintf(stderr, "[] Error, could not reach end of header\n");
    EXIT(4);
  }

  if(g_cart_id==0) fprintf(stdout, "# [] Reached end of header, read binary section\n");
  
  if(g_cart_id==0) fprintf(stdout, "# [] l1=%d, l2=%d, l3=%d, l4=%d\n",l1, l2, l3, l4);

//  lvol = l1 * l2 * l3 * l4;
  lvol = l1 * l2 * l3 * T;

  if(lvol==0) {
    if(g_cart_id==0) fprintf(stderr, "[] Error, zero volume\n");
    EXIT(5);
  }
 
  ftmp = (double*)malloc(lvol*72*sizeof(double));
  if(ftmp == NULL) {
    if(g_cart_id==0) fprintf(stderr, "[] Error, could not alloc ftmp\n");
    EXIT(6);
  }
#ifdef HAVE_MPI
    offset = 72*(long int)l1*l2*l3*g_proc_coords[0]*T*sizeof(double);
    //fprintf(stdout, "# [%d] offset = %ld\n", g_cart_id, offset);
    if(fseek(ifs, offset, SEEK_CUR) != 0) {
      EXIT(10);
    }
#endif
  iread = fread(ftmp, sizeof(double), 72*lvol, ifs);
  if(iread != 72*lvol) {
    if(g_cart_id==0) fprintf(stderr, "[] Error, could not read proper amount of data\n");
    EXIT(7);
  }
  fclose(ifs);

  if(!words_bigendian) {
    // fprintf(stdout, "# [] performing byte swap\n");
    byte_swap64_v2(ftmp, 72*lvol);
  }

  bytes = 72*lvol*sizeof(double);
  step=sizeof(uint32_t);
  if(g_cart_id==0) fprintf(stdout, "# [] step size = %u bytes\n", step);
  iptr = (uint32_t*)ftmp;
  checksum = 0;
  for(ix=0;ix<bytes;ix+=step) {
    checksum += *iptr;
    iptr++;
  }

#ifdef HAVE_MPI
    MPI_Allreduce(&checksum, &ibuf, 1, MPI_INT, MPI_SUM, g_cart_grid);
    checksum = ibuf;
#endif
  if(g_cart_id==0) fprintf(stdout, "# [] checksum: read  %#x; calculated %#x\n", cks, checksum);

  if(cks != checksum) {
    if(g_cart_id==0) fprintf(stderr, "[] Error, checksums do not match\n");
    EXIT(8);
  }

  nu[0] = 1;
  nu[1] = 2;
  nu[2] = 3;
  nu[3] = 0;
  // reconstruct gauge field
  // - assume index formula idx = (((t*LX+x)*LY+y)*LZ+z)*(4*3*3*2) + mu*(3*3*2) + 2*(3*u+c)+r
  //   with mu=0,1,2,3; u=0,1,2; c=0,1,2, r=0,1
  iy = 0;
  for(x4=0; x4<T; x4++) {  // t
  for(x3=0; x3<l3; x3++) {  // z
  for(x2=0; x2<l2; x2++) {  // y
  for(x1=0; x1<l1; x1++) {  // x
    ix = g_ipt[x4][x1][x2][x3];
    for(mu=0;mu<4;mu++) {
      idx = 72*iy + 18*mu;
      memcpy(s+_GGI(ix,nu[mu]), (ftmp+idx), 18*sizeof(double));

    }
    iy++;
  }}}}

  free(ftmp);
  return(0);

}

}
