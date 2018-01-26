#ifndef _MATRIX_INIT_H
#define _MATRIX_INIT_H
/****************************************************
 * matrix_init.h
 *
 * Mon Oct 17 08:10:58 CEST 2016
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

namespace cvc {

/************************************************************************************
 * (de-)allocate 1-level buffer (n1 vector )
 ************************************************************************************/
int init_1level_buffer (double**buffer, unsigned int n1 );

int fini_1level_buffer (double**buffer );

/************************************************************************************
 * (de-)allocate 2-level buffer (n1 x n2 matrix)
 ************************************************************************************/
int init_2level_buffer (double***buffer, unsigned int n1, unsigned int n2);

int fini_2level_buffer (double***buffer);

/************************************************************************************
 * (de-)allocate 3-level buffer (n1 x n2 x n3 matrix)
 ************************************************************************************/
int init_3level_buffer (double****buffer, unsigned int n1, unsigned int n2, unsigned int n3);

int fini_3level_buffer (double****buffer);

/************************************************************************************
 * (de-)allocate 4-level buffer (n1 x n2 x n3 x n4 matrix)
 ************************************************************************************/

int init_4level_buffer (double*****buffer, unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4);

int fini_4level_buffer (double*****buffer);

/************************************************************************************/
/************************************************************************************/

/************************************************************************************
 * (de-)allocate 1-level zbuffer n1 vector
 ************************************************************************************/
int init_1level_zbuffer (double _Complex**zbuffer, unsigned int n1 );

int fini_1level_zbuffer (double _Complex**zbuffer);


/************************************************************************************
 * (de-)allocate 2-level zbuffer (n1 x n2 matrix)
 ************************************************************************************/
int init_2level_zbuffer (double _Complex***zbuffer, unsigned int n1, unsigned int n2);

int fini_2level_zbuffer (double _Complex***zbuffer);

/************************************************************************************
 * (de-)allocate 2-level zbuffer ( asymmetric )
 ************************************************************************************/
int init_2level_zbuffer_asym (double _Complex***zbuffer, int n, int *dim);

int fini_2level_zbuffer_asym (double _Complex***zbuffer );


/************************************************************************************
 * (de-)allocate 3-level zbuffer (n1 x n2 x n3 matrix)
 ************************************************************************************/
int init_3level_zbuffer (double _Complex****zbuffer, unsigned int n1, unsigned int n2, unsigned int n3);

int fini_3level_zbuffer (double _Complex****zbuffer);

/************************************************************************************
 * (de-)allocate 4-level zbuffer (n1 x n2 x n3 x n4 matrix)
 ************************************************************************************/

int init_4level_zbuffer (double _Complex*****zbuffer, unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4);

int fini_4level_zbuffer (double _Complex*****zbuffer);

/************************************************************************************/
/************************************************************************************/

/************************************************************************************
 * integer matrices
 ************************************************************************************/

int init_1level_ibuffer ( int**buffer, unsigned int n1 );
int fini_1level_ibuffer ( int**buffer );


int init_2level_ibuffer ( int***buffer, unsigned int n1, unsigned int n2);
int fini_2level_ibuffer ( int***buffer );

int init_3level_ibuffer ( int****buffer, unsigned int n1, unsigned int n2, unsigned int n3);
int fini_3level_ibuffer ( int****buffer );

/************************************************************************************/
/************************************************************************************/

/************************************************************************************
 * character matrix
 ************************************************************************************/
int init_2level_cbuffer ( char***buffer, unsigned int n1, unsigned int n2);
int fini_2level_cbuffer ( char***buffer);

/************************************************************************************/
/************************************************************************************/

}  /* end of namespace cvc */

#endif
