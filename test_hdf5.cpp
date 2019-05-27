/****************************************************
 * test_zm4x4.cpp
 *
 * So 29. Jul 21:30:12 CEST 2018
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <ctype.h>
#include <sys/stat.h>
#include <sys/types.h>
 
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include <getopt.h>

#include <hdf5.h>

#define MAIN_PROGRAM

#include "cvc_complex.h"
#include "cvc_linalg.h"
#include "global.h"
#include "cvc_geometry.h"
#include "cvc_utils.h"
#include "mpi_init.h"
#include "io.h"
#include "read_input_parser.h"

#define FILE "groups.h5"

using namespace cvc;

void usage() {
#ifdef HAVE_MPI
  MPI_Abort(MPI_COMM_WORLD, 1);
  MPI_Finalize();
#endif
  exit(0);
}

int main(int argc, char **argv) {


  int c;
  int filename_set = 0;
  char filename[100];
  int exitstatus;


#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  while ((c = getopt(argc, argv, "h?f:")) != -1) {
    switch (c) {
      case 'f':
        strcpy(filename, optarg);
        filename_set=1;
        break;
      case 'h':
      case '?':
      default:
        fprintf(stdout, "# [test_zm4x4] exit\n");
        exit(1);
      break;
    }
  }


  /* set the default values */
  if(filename_set==0) strcpy(filename, "cvc.input");
  fprintf(stdout, "# [test_zm4x4] Reading input from file %s\n", filename);
  read_input_parser(filename);

  /*********************************
   * initialize MPI parameters for cvc
   *********************************/
   mpi_init(argc, argv);

 /*********************************
  * set number of openmp threads
  *********************************/
  fprintf(stdout, "# [test_zm4x4] test :  g_num_threads = %d\n", g_num_threads);
#ifdef HAVE_OPENMP
 if(g_cart_id == 0) fprintf(stdout, "# [test_zm4x4] setting omp number of threads to %d\n", g_num_threads);
 omp_set_num_threads(g_num_threads);
#pragma omp parallel
{
  fprintf(stdout, "# [test_zm4x4] proc%.4d thread%.4d using %d threads\n", g_cart_id, omp_get_thread_num(), omp_get_num_threads());
}
#else
  if(g_cart_id == 0) fprintf(stdout, "[test_zm4x4] Warning, resetting global thread number to 1\n");
  g_num_threads = 1;
#endif

  if(init_geometry() != 0) {
    fprintf(stderr, "[test_zm4x4] Error from init_geometry\n");
    EXIT(1);
  }

  geometry();

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Board of Trustees of the University of Illinois.         *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://support.hdfgroup.org/ftp/HDF5/releases.  *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 *  This example illustrates how to create a dataset in a group.
 *  It is used in the HDF5 Tutorial.
 */


  hid_t       file_id, group_id, dataset_id, dataspace_id;  /* identifiers */
  hsize_t     dims[2];
  herr_t      status;
  int         i, j, dset1_data[3][3], dset2_data[2][10];

  /* Initialize the first dataset. */
  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      dset1_data[i][j] = j + 1;

  /* Initialize the second dataset. */
  for (i = 0; i < 2; i++)
    for (j = 0; j < 10; j++)
      dset2_data[i][j] = j + 1;

  struct stat fileStat;
  if(stat( FILE, &fileStat) < 0 ) {
    /* Open an existing file. */
    fprintf ( stdout, "# [test_hdf5] create new file\n" );
    file_id = H5Fcreate(FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  } else {
    fprintf ( stdout, "# [test_hdf5] open existing file\n" );
    file_id = H5Fopen(FILE, H5F_ACC_RDWR, H5P_DEFAULT);
  }

  /* Create a group named "/MyGroup" in the file. */
  group_id = H5Gcreate2(file_id, "/MyGroup", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  /* Close the group. */
  status = H5Gclose(group_id);

  /* Create the data space for the first dataset. */
  dims[0] = 3;
  dims[1] = 3;
  dataspace_id = H5Screate_simple(2, dims, NULL);

  /* Create a dataset in group "MyGroup". */
  dataset_id = H5Dcreate2(file_id, "/MyGroup/dset1", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  /* Write the first dataset. */
  status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset1_data);

  /* Close the data space for the first dataset. */
  status = H5Sclose(dataspace_id);

  /* Close the first dataset. */
  status = H5Dclose(dataset_id);

  /* Close the group. */
  // status = H5Gclose(group_id);
#if 0
#endif

#if 0
  /* Open an existing group of the specified file. */
  group_id = H5Gopen2(file_id, "/MyGroup/Group_A", H5P_DEFAULT);

  /* Create the data space for the second dataset. */
  dims[0] = 2;
  dims[1] = 10;
  dataspace_id = H5Screate_simple(2, dims, NULL);

  /* Create the second dataset in group "Group_A". */
  dataset_id = H5Dcreate2(group_id, "dset2", H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  /* Write the second dataset. */
  status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dset2_data);

  /* Close the data space for the second dataset. */
  status = H5Sclose(dataspace_id);

  /* Close the second dataset */
  status = H5Dclose(dataset_id);

  /* Close the group. */
  status = H5Gclose(group_id);
#endif
  /* Close the file. */
  status = H5Fclose(file_id);

  return(0);
}
