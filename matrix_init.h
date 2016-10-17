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

}  /* end of namespace cvc */

#endif
