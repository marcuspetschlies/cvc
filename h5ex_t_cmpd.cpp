/************************************************************

  This example shows how to read and write compound
  datatypes to a dataset.  The program first writes
  compound structures to a dataset with a dataspace of DIM0,
  then closes the file.  Next, it reopens the file, reads
  back the data, and outputs it to the screen.

  This file is intended for use with HDF5 Library version 1.8

 ************************************************************/

#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#define FILE            "h5ex_t_cmpd.h5"
#define DATASET         "DS1"
#define DIM0            4

typedef struct {
    double  re;
    double  im;
} complex;                                 /* Compound type */

int
main (void)
{
    hid_t       file, filetype, memtype, strtype, space, dset;
                                            /* Handles */
    herr_t      status;
    hsize_t     dims[1] = {DIM0};
    double _Complex  wbuffer[DIM0];
                                  
    complex *wdata = NULL, *rdata = NULL;

    int         ndims;
    hsize_t     i;

    /*
     * Initialize data.
     */
    wbuffer[0] = 1. + 2.*I;
    wbuffer[1] = 3. + 4.*I;
    wbuffer[2] = 5. + 6.*I;
    wbuffer[3] = 7. + 8.*I;

    wdata = (complex*)&(wbuffer[0]);

    /*
     * Create a new file using the default properties.
     */
    file = H5Fcreate (FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    /*
     * Create variable-length string datatype.
     */
    strtype = H5Tcopy (H5T_C_S1);
    status = H5Tset_size (strtype, H5T_VARIABLE);

    /*
     * Create the compound datatype for memory.
     */
    memtype = H5Tcreate (H5T_COMPOUND, sizeof ( complex ));

    status = H5Tinsert (memtype, "real part", HOFFSET ( complex, re ), H5T_NATIVE_DOUBLE );

    status = H5Tinsert (memtype, "imag part", HOFFSET ( complex, im ), H5T_NATIVE_DOUBLE );

    /*
     * Create the compound datatype for the file.  Because the standard
     * types we are using for the file may have different sizes than
     * the corresponding native types, we must manually calculate the
     * offset of each member.
     */
    filetype = H5Tcreate (H5T_COMPOUND, sizeof ( complex ) );
    status = H5Tinsert (filetype, "real part", 0, H5T_IEEE_F64BE );
    status = H5Tinsert (filetype, "imag part", sizeof(double), H5T_IEEE_F64BE );


    /*
     * Create dataspace.  Setting maximum size to NULL sets the maximum
     * size to be the current size.
     */
    space = H5Screate_simple (1, dims, NULL);

    /*
     * Create the dataset and write the compound data to it.
     */
    dset = H5Dcreate (file, DATASET, filetype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    status = H5Dwrite (dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, wdata);

    /*
     * Close and release resources.
     */
    status = H5Dclose (dset);
    status = H5Sclose (space);
    status = H5Tclose (filetype);
    status = H5Fclose (file);


    /*
     * Now we begin the read section of this example.  Here we assume
     * the dataset has the same name and rank, but can have any size.
     * Therefore we must allocate a new array to read in data using
     * malloc().  For simplicity, we do not rebuild memtype.
     */

    /*
     * Open file and dataset.
     */
    file = H5Fopen (FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen (file, DATASET, H5P_DEFAULT);

    /*
     * Get dataspace and allocate memory for read buffer.
     */
    space = H5Dget_space (dset);
    ndims = H5Sget_simple_extent_dims (space, dims, NULL);
    rdata = ( complex *) malloc (dims[0] * sizeof ( complex ));

    /*
     * Read the data.
     */
    status = H5Dread (dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);

    /*
     * Output the data to the screen.
     */
    for (i=0; i<dims[0]; i++) {
        printf ("%s[%llu]:\n", DATASET, i);
        printf ("real part       : %e\n", rdata[i].re );
        printf ("imag part       : %e\n", rdata[i].im );
    }

    /*
     * Close and release resources.  H5Dvlen_reclaim will automatically
     * traverse the structure and free any vlen data (strings in this
     * case).
     */
    status = H5Dvlen_reclaim (memtype, space, H5P_DEFAULT, rdata);
    free (rdata);
    status = H5Dclose (dset);
    status = H5Sclose (space);
    status = H5Tclose (memtype);
    status = H5Tclose (strtype);
    status = H5Fclose (file);

    return 0;
}
