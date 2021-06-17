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


/************************************************************************************
 * (de-)allocate 2-level zbuffer (n1 x n2 matrix)
 ************************************************************************************/
int init_2level_zbuffer (double _Complex***zbuffer, unsigned int n1, unsigned int n2);

int fini_2level_zbuffer (double _Complex***zbuffer);

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

/************************************************************************************
 * (de-)allocate 5-level zbuffer (n1 x n2 x n3 x n4 x n5 matrix)
 ************************************************************************************/
int init_5level_zbuffer (double _Complex******zbuffer, unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4, unsigned int n5);

int fini_5level_zbuffer (double _Complex******zbuffer );

/************************************************************************************
 * (de-)allocate 6-level zbuffer (n1 x n2 x n3 x n4 x n5 x n6 matrix)
 ************************************************************************************/
int init_6level_zbuffer (double _Complex*******zbuffer, unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4, unsigned int n5, unsigned int n6);

int fini_6level_zbuffer (double _Complex*******zbuffer );

/************************************************************************************
 * (de-)allocate 2-level ibuffer (n1 x n2 matrix)
 ************************************************************************************/
int init_2level_ibuffer (int***buffer, unsigned int n1, unsigned int n2);

int fini_2level_ibuffer (int***buffer );

int init_1level_zbuffer (double _Complex**zbuffer, unsigned int n1 );
int fini_1level_zbuffer (double _Complex**zbuffer );


}  /* end of namespace cvc */

#endif
