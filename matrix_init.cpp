/****************************************************
 * matrix_init.cpp
 *
 * Mon Oct 17 08:10:58 CEST 2016
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef HAVE_MPI
#  include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#  include <omp.h>
#endif
#include "matrix_init.h"

namespace cvc {

/************************************************************************************
 * (de-)allocate 2-level buffer (n1 x n2 double matrix)
 ************************************************************************************/
int init_2level_buffer (double***buffer, unsigned int n1, unsigned int n2) {

  unsigned int i;
  size_t items = (size_t)n2;
  size_t bytes = items * n1 * sizeof(double);

  if(*buffer != NULL) {
    fprintf(stderr, "[init_2level_buffer] Error, buffer not NULL\n"); 
    return(1);
  }

  /* 1st, outer level */
  (*buffer) = (double**)malloc(n1 * sizeof(double*));
  if( *buffer == NULL ) {
    fprintf(stderr, "[init_2level_buffer] Error from malloc\n"); 
    return(2);
  }

  /* 2nd, inner level */
  (*buffer)[0] = (double*)malloc(bytes);
  if( (*buffer)[0] == NULL ) {
    fprintf(stderr, "[init_2level_buffer] Error from malloc\n"); 
    return(3);
  }
  for(i=1; i<n1; i++) (*buffer)[i] = (*buffer)[i-1] + items;
  memset( (*buffer)[0], 0, bytes);
  return(0);
}  /* end of init_2level_buffer */

int fini_2level_buffer (double***buffer) {
  if(*buffer != NULL) {
    if( (*buffer)[0] != NULL) { free( (*buffer)[0] ); }
    free( *buffer );
    *buffer = NULL;
  }
  return(0);
}  /* end of fini_2level_buffer */

/************************************************************************************
 * (de-)allocate 3-level buffer (n1 x n2 x n3 matrix)
 ************************************************************************************/
int init_3level_buffer (double****buffer, unsigned int n1, unsigned int n2, unsigned int n3) {

  unsigned int i, k, l;
  size_t items=0, bytes=0;

  if(*buffer != NULL) {
    fprintf(stderr, "[init_3level_buffer] Error, buffer not NULL\n"); 
    return(1);
  }

  /* 1st, outermost level */
  (*buffer) = (double***)malloc(n1 * sizeof(double**));
  if( *buffer == NULL ) {
    fprintf(stderr, "[init_3level_buffer] Error from malloc\n"); 
    return(2);
  }

  /* 2nd, middle level */
  items = (size_t)n2;
  bytes = items * n1 * sizeof(double*);
  (*buffer)[0] = (double**)malloc(bytes);
  if( (*buffer)[0] == NULL ) {
    fprintf(stderr, "[init_3level_buffer] Error from malloc\n"); 
    return(3);
  }
  for(i=1; i<n1; i++) (*buffer)[i] = (*buffer)[i-1] + items;

  /* 3rd, innermost level */
  items = (size_t)n3 * (size_t)n2 * (size_t)n1;
  bytes = items * sizeof(double);
  (*buffer)[0][0] = (double*)malloc(bytes);
  if( (*buffer)[0][0] == NULL ) {
    fprintf(stderr, "[init_3level_buffer] Error from malloc\n"); 
    return(4);
  }
  l=0;
  for(i=0; i<n1; i++) {
  for(k=0; k<n2; k++) {
    (*buffer)[i][k] = (*buffer)[0][0] + l*n3;
    l++;
  }}
  memset( (*buffer)[0][0], 0, bytes);
  return(0);
}  /* end of init_3level_buffer */

int fini_3level_buffer (double****buffer) {
  if(*buffer != NULL) {
    if( (*buffer)[0] != NULL) { 
      if( (*buffer)[0][0] != NULL) { free( (*buffer)[0][0] ); }
      free( (*buffer)[0] ); 
    }
    free( *buffer );
    *buffer = NULL;
  }
  return(0);
}  /* end of fini_3level_buffer */


/************************************************************************************
 * (de-)allocate 4-level buffer (n1 x n2 x n3 x n4 matrix)
 ************************************************************************************/
int init_4level_buffer (double*****buffer, unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4) {

  unsigned int i, k, l, m;
  size_t items=0, bytes=0;

  if(*buffer != NULL) {
    fprintf(stderr, "[init_4level_buffer] Error, buffer not NULL\n"); 
    return(1);
  }

  /* 1st, outermost level */
  (*buffer) = (double****)malloc(n1 * sizeof(double***));
  if( *buffer == NULL ) {
    fprintf(stderr, "[init_4level_buffer] Error from malloc\n"); 
    return(2);
  }

  /* 2nd level */
  items = (size_t)n2;
  bytes = items * n1 * sizeof(double**);
  (*buffer)[0] = (double***)malloc(bytes);
  if( (*buffer)[0] == NULL ) {
    fprintf(stderr, "[init_4level_buffer] Error from malloc\n"); 
    return(3);
  }
  for(i=1; i<n1; i++) (*buffer)[i] = (*buffer)[i-1] + items;

  /* 3rd, penultimate level */
  items = (size_t)n3 * (size_t)n2 * (size_t)n1;
  bytes = items * sizeof(double*);
  (*buffer)[0][0] = (double**)malloc(bytes);
  if( (*buffer)[0][0] == NULL ) {
    fprintf(stderr, "[init_4level_buffer] Error from malloc\n"); 
    return(4);
  }
  l=0;
  for(i=0; i<n1; i++) {
  for(k=0; k<n2; k++) {
    (*buffer)[i][k] = (*buffer)[0][0] + l*n3;
    l++;
  }}

  /* 4th, innermost level */
  items = (size_t)n3 * (size_t)n2 * (size_t)n1 * (size_t)n4;
  bytes = items * sizeof(double);
  (*buffer)[0][0][0] = (double*)malloc(bytes);
  if( (*buffer)[0][0][0] == NULL ) {
    fprintf(stderr, "[init_4level_buffer] Error from malloc\n"); 
    return(5);
  }
  m=0;
  for(i=0; i<n1; i++) {
  for(k=0; k<n2; k++) {
  for(l=0; l<n3; l++) {
    (*buffer)[i][k][l] = (*buffer)[0][0][0] + m*n4;
    m++;
  }}}

  memset( (*buffer)[0][0][0], 0, bytes);
  return(0);
}  /* end of init_4level_buffer */

int fini_4level_buffer (double*****buffer) {
  if(*buffer != NULL) {
    if( (*buffer)[0] != NULL) { 
      if( (*buffer)[0][0] != NULL) { 
        if( (*buffer)[0][0][0] != NULL) { free( (*buffer)[0][0][0] );} 
        free( (*buffer)[0][0] ); 
      }
      free( (*buffer)[0] ); 
    }
    free( *buffer );
    *buffer = NULL;
  }
  return(0);
}  /* end of fini_4level_buffer */

/************************************************************************************
 * (de-)allocate 1-level buffer (n1 x n2 double matrix)
 ************************************************************************************/
int init_1level_zbuffer (double _Complex**zbuffer, unsigned int n1 ) {
  if(*zbuffer != NULL) {
    fprintf(stderr, "[init_1level_zbuffer] Error, zbuffer not NULL\n");
    return(1);
  }

  (*zbuffer) = (double _Complex*)malloc(n1 * sizeof(double _Complex));
  if( *zbuffer == NULL ) {
    fprintf(stderr, "[init_1level_zbuffer] Error from malloc\n");
    return(2);
  }
  return(0);
}  /* end of init_1level_zbuffer */

int fini_1level_zbuffer (double _Complex**zbuffer ) {
  if( *zbuffer != NULL) { 
    free( *zbuffer );
    *zbuffer = NULL;
   }
}  /* end of fini_1level_zbuffer */


/************************************************************************************
 * (de-)allocate 2-level buffer (n1 x n2 double matrix)
 ************************************************************************************/
int init_2level_zbuffer (double _Complex***zbuffer, unsigned int n1, unsigned int n2) {

  unsigned int i;
  size_t items = (size_t)n2;
  size_t bytes = items * n1 * sizeof(double _Complex);

  if(*zbuffer != NULL) {
    fprintf(stderr, "[init_2level_zbuffer] Error, zbuffer not NULL\n"); 
    return(1);
  }

  /* 1st, outer level */
  (*zbuffer) = (double _Complex**)malloc(n1 * sizeof(double _Complex*));
  if( *zbuffer == NULL ) {
    fprintf(stderr, "[init_2level_zbuffer] Error from malloc\n"); 
    return(2);
  }

  /* 2nd, inner level */
  (*zbuffer)[0] = (double _Complex*)malloc(bytes);
  if( (*zbuffer)[0] == NULL ) {
    fprintf(stderr, "[init_2level_zbuffer] Error from malloc\n"); 
    return(3);
  }
  for(i=1; i<n1; i++) (*zbuffer)[i] = (*zbuffer)[i-1] + items;
  memset( (*zbuffer)[0], 0, bytes);
  return(0);
}  /* end of init_2level_zbuffer */

int fini_2level_zbuffer (double _Complex***zbuffer) {
  if(*zbuffer != NULL) {
    if( (*zbuffer)[0] != NULL) { free( (*zbuffer)[0] ); }
    free( *zbuffer );
    *zbuffer = NULL;
  }
  return(0);
}  /* end of fini_2level_zbuffer */

/************************************************************************************
 * (de-)allocate 3-level zbuffer (n1 x n2 x n3 matrix)
 ************************************************************************************/
int init_3level_zbuffer (double _Complex****zbuffer, unsigned int n1, unsigned int n2, unsigned int n3) {

  unsigned int i, k, l;
  size_t items=0, bytes=0;

  if(*zbuffer != NULL) {
    fprintf(stderr, "[init_3level_zbuffer] Error, zbuffer not NULL\n"); 
    return(1);
  }

  /* 1st, outermost level */
  (*zbuffer) = (double _Complex***)malloc(n1 * sizeof(double _Complex**));
  if( *zbuffer == NULL ) {
    fprintf(stderr, "[init_3level_zbuffer] Error from malloc\n"); 
    return(2);
  }

  /* 2nd, middle level */
  items = (size_t)n2;
  bytes = items * n1 * sizeof(double _Complex*);
  (*zbuffer)[0] = (double _Complex**)malloc(bytes);
  if( (*zbuffer)[0] == NULL ) {
    fprintf(stderr, "[init_3level_zbuffer] Error from malloc\n"); 
    return(3);
  }
  for(i=1; i<n1; i++) (*zbuffer)[i] = (*zbuffer)[i-1] + items;

  /* 3rd, innermost level */
  items = (size_t)n3 * (size_t)n2 * (size_t)n1;
  bytes = items * sizeof(double _Complex);
  (*zbuffer)[0][0] = (double _Complex*)malloc(bytes);
  if( (*zbuffer)[0][0] == NULL ) {
    fprintf(stderr, "[init_3level_zbuffer] Error from malloc\n"); 
    return(4);
  }
  l=0;
  for(i=0; i<n1; i++) {
  for(k=0; k<n2; k++) {
    (*zbuffer)[i][k] = (*zbuffer)[0][0] + l*n3;
    l++;
  }}
  memset( (*zbuffer)[0][0], 0, bytes);
  return(0);
}  /* end of init_3level_zbuffer */

int fini_3level_zbuffer (double _Complex****zbuffer) {
  if(*zbuffer != NULL) {
    if( (*zbuffer)[0] != NULL) { 
      if( (*zbuffer)[0][0] != NULL) { free( (*zbuffer)[0][0] ); }
      free( (*zbuffer)[0] ); 
    }
    free( *zbuffer );
    *zbuffer = NULL;
  }
  return(0);
}  /* end of fini_3level_zbuffer */


/************************************************************************************
 * (de-)allocate 4-level zbuffer (n1 x n2 x n3 x n4 matrix)
 ************************************************************************************/
int init_4level_zbuffer (double _Complex*****zbuffer, unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4) {

  unsigned int i, k, l, m;
  size_t items=0, bytes=0;

  if(*zbuffer != NULL) {
    fprintf(stderr, "[init_4level_zbuffer] Error, zbuffer not NULL\n"); 
    return(1);
  }

  /* 1st, outermost level */
  (*zbuffer) = (double _Complex****)malloc(n1 * sizeof(double _Complex***));
  if( *zbuffer == NULL ) {
    fprintf(stderr, "[init_4level_zbuffer] Error from malloc\n"); 
    return(2);
  }

  /* 2nd level */
  items = (size_t)n2;
  bytes = items * n1 * sizeof(double _Complex**);
  (*zbuffer)[0] = (double _Complex***)malloc(bytes);
  if( (*zbuffer)[0] == NULL ) {
    fprintf(stderr, "[init_4level_zbuffer] Error from malloc\n"); 
    return(3);
  }
  for(i=1; i<n1; i++) (*zbuffer)[i] = (*zbuffer)[i-1] + items;

  /* 3rd, penultimate level */
  items = (size_t)n3 * (size_t)n2 * (size_t)n1;
  bytes = items * sizeof(double _Complex*);
  (*zbuffer)[0][0] = (double _Complex**)malloc(bytes);
  if( (*zbuffer)[0][0] == NULL ) {
    fprintf(stderr, "[init_4level_zbuffer] Error from malloc\n"); 
    return(4);
  }
  l=0;
  for(i=0; i<n1; i++) {
  for(k=0; k<n2; k++) {
    (*zbuffer)[i][k] = (*zbuffer)[0][0] + l*n3;
    l++;
  }}

  /* 4th, innermost level */
  items = (size_t)n3 * (size_t)n2 * (size_t)n1 * (size_t)n4;
  bytes = items * sizeof(double _Complex);
  (*zbuffer)[0][0][0] = (double _Complex*)malloc(bytes);
  if( (*zbuffer)[0][0][0] == NULL ) {
    fprintf(stderr, "[init_4level_zbuffer] Error from malloc\n"); 
    return(5);
  }
  m=0;
  for(i=0; i<n1; i++) {
  for(k=0; k<n2; k++) {
  for(l=0; l<n3; l++) {
    (*zbuffer)[i][k][l] = (*zbuffer)[0][0][0] + m*n4;
    m++;
  }}}

  memset( (*zbuffer)[0][0][0], 0, bytes);
  return(0);
}  /* end of init_4level_zbuffer */

int fini_4level_zbuffer (double _Complex*****zbuffer) {
  if(*zbuffer != NULL) {
    if( (*zbuffer)[0] != NULL) { 
      if( (*zbuffer)[0][0] != NULL) { 
        if( (*zbuffer)[0][0][0] != NULL) { free( (*zbuffer)[0][0][0] );} 
        free( (*zbuffer)[0][0] ); 
      }
      free( (*zbuffer)[0] ); 
    }
    free( *zbuffer );
    *zbuffer = NULL;
  }
  return(0);
}  /* end of fini_4level_zbuffer */

/************************************************************************************
 * (de-)allocate 5-level zbuffer (n1 x n2 x n3 x n4 x n5 matrix)
 ************************************************************************************/
int init_5level_zbuffer (double _Complex ******zbuffer, unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4, unsigned int n5) {

  unsigned int i, k, l, m, n;
  size_t items=0, bytes=0;

  if(*zbuffer != NULL) {
    fprintf(stderr, "[init_5level_zbuffer] Error, zbuffer not NULL\n"); 
    return(1);
  }

  /* 1st, outermost level */
  (*zbuffer) = (double _Complex*****)malloc(n1 * sizeof(double _Complex****));
  if( *zbuffer == NULL ) {
    fprintf(stderr, "[init_5level_zbuffer] Error from malloc\n"); 
    return(2);
  }

  /* 2nd level */
  items = (size_t)n2;
  bytes = items * n1 * sizeof(double _Complex***);
  (*zbuffer)[0] = (double _Complex****)malloc(bytes);
  if( (*zbuffer)[0] == NULL ) {
    fprintf(stderr, "[init_5level_zbuffer] Error from malloc\n"); 
    return(3);
  }
  for(i=1; i<n1; i++) (*zbuffer)[i] = (*zbuffer)[i-1] + items;

  /* 3rd level */
  items = (size_t)n1 * (size_t)n2 * (size_t)n3;
  bytes = items * sizeof(double _Complex**);
  (*zbuffer)[0][0] = (double _Complex***)malloc(bytes);
  if( (*zbuffer)[0][0] == NULL ) {
    fprintf(stderr, "[init_5level_zbuffer] Error from malloc\n"); 
    return(4);
  }
  l=0;
  for(i=0; i<n1; i++) {
  for(k=0; k<n2; k++) {
    (*zbuffer)[i][k] = (*zbuffer)[0][0] + l*n3;
    l++;
  }}

  /* 4th, penultimate level */
  items = (size_t)n1 * (size_t)n2 * (size_t)n3 * (size_t)n4;
  bytes = items * sizeof(double _Complex*);
  (*zbuffer)[0][0][0] = (double _Complex**)malloc(bytes);
  if( (*zbuffer)[0][0][0] == NULL ) {
    fprintf(stderr, "[init_5level_zbuffer] Error from malloc\n"); 
    return(5);
  }
  m=0;
  for(i=0; i<n1; i++) {
  for(k=0; k<n2; k++) {
  for(l=0; l<n3; l++) {
    (*zbuffer)[i][k][l] = (*zbuffer)[0][0][0] + m*n4;
    m++;
  }}}

  /* 5th, innermost level */
  items = (size_t)n1 * (size_t)n2 * (size_t)n3 * (size_t)n4 * (size_t)n5;
  bytes = items * sizeof(double _Complex);
  (*zbuffer)[0][0][0][0] = (double _Complex*)malloc(bytes);
  if( (*zbuffer)[0][0][0][0] == NULL ) {
    fprintf(stderr, "[init_5level_zbuffer] Error from malloc\n"); 
    return(6);
  }
  n = 0;
  for(i=0; i<n1; i++) {
  for(k=0; k<n2; k++) {
  for(l=0; l<n3; l++) {
  for(m=0; m<n4; m++) {
    (*zbuffer)[i][k][l][m] = (*zbuffer)[0][0][0][0] + n * n5;
    n++;
  }}}}

  memset( (*zbuffer)[0][0][0][0], 0, bytes);
  return(0);
}  /* end of init_5level_zbuffer */

int fini_5level_zbuffer (double _Complex******zbuffer) {
  if(*zbuffer != NULL) {
    if( (*zbuffer)[0] != NULL) { 
      if( (*zbuffer)[0][0] != NULL) { 
        if( (*zbuffer)[0][0][0] != NULL) { 
          if( (*zbuffer)[0][0][0][0] != NULL) { 
            free( (*zbuffer)[0][0][0][0] );
          }
          free( (*zbuffer)[0][0][0] ); 
        } 
        free( (*zbuffer)[0][0] ); 
      }
      free( (*zbuffer)[0] ); 
    }
    free( *zbuffer );
    *zbuffer = NULL;
  }
  return(0);
}  /* end of fini_5level_zbuffer */

/************************************************************************************
 * (de-)allocate 6-level zbuffer (n1 x n2 x n3 x n4 x n5 x n6 matrix)
 ************************************************************************************/
int init_6level_zbuffer (double _Complex*******zbuffer, unsigned int n1, unsigned int n2, unsigned int n3, unsigned int n4, unsigned int n5, unsigned int n6) {

  unsigned int i, k, l, m, n, o;
  size_t items=0, bytes=0;

  if(*zbuffer != NULL) {
    fprintf(stderr, "[init_6level_zbuffer] Error, zbuffer not NULL\n"); 
    return(1);
  }

  /* 1st, outermost level */
  (*zbuffer) = (double _Complex******)malloc(n1 * sizeof(double _Complex*****));
  if( *zbuffer == NULL ) {
    fprintf(stderr, "[init_6level_zbuffer] Error from malloc\n"); 
    return(2);
  }

  /* 2nd level */
  items = (size_t)n2;
  bytes = items * n1 * sizeof(double _Complex****);
  (*zbuffer)[0] = (double _Complex*****)malloc(bytes);
  if( (*zbuffer)[0] == NULL ) {
    fprintf(stderr, "[init_6level_zbuffer] Error from malloc\n"); 
    return(3);
  }
  for(i=1; i<n1; i++) (*zbuffer)[i] = (*zbuffer)[i-1] + items;

  /* 3rd level */
  items = (size_t)n3 * (size_t)n2 * (size_t)n1;
  bytes = items * sizeof(double _Complex***);
  (*zbuffer)[0][0] = (double _Complex****)malloc(bytes);
  if( (*zbuffer)[0][0] == NULL ) {
    fprintf(stderr, "[init_6level_zbuffer] Error from malloc\n"); 
    return(4);
  }
  l=0;
  for(i=0; i<n1; i++) {
  for(k=0; k<n2; k++) {
    (*zbuffer)[i][k] = (*zbuffer)[0][0] + l*n3;
    l++;
  }}

  /* 4th level */
  items = (size_t)n3 * (size_t)n2 * (size_t)n1 * (size_t)n4;
  bytes = items * sizeof(double _Complex**);
  (*zbuffer)[0][0][0] = (double _Complex***)malloc(bytes);
  if( (*zbuffer)[0][0][0] == NULL ) {
    fprintf(stderr, "[init_6level_zbuffer] Error from malloc\n"); 
    return(5);
  }
  m=0;
  for(i=0; i<n1; i++) {
  for(k=0; k<n2; k++) {
  for(l=0; l<n3; l++) {
    (*zbuffer)[i][k][l] = (*zbuffer)[0][0][0] + m*n4;
    m++;
  }}}

  /* 5th, penultimate level */
  items = (size_t)n3 * (size_t)n2 * (size_t)n1 * (size_t)n4 * (size_t)n5;
  bytes = items * sizeof(double _Complex*);
  (*zbuffer)[0][0][0][0] = (double _Complex**)malloc(bytes);
  if( (*zbuffer)[0][0][0][0] == NULL ) {
    fprintf(stderr, "[init_6level_zbuffer] Error from malloc\n"); 
    return(6);
  }
  n = 0;
  for(i=0; i<n1; i++) {
  for(k=0; k<n2; k++) {
  for(l=0; l<n3; l++) {
  for(m=0; m<n4; m++) {
    (*zbuffer)[i][k][l][m] = (*zbuffer)[0][0][0][0] + n * n5;
    n++;
  }}}}

  /* 6th, innermost level */
  items = (size_t)n3 * (size_t)n2 * (size_t)n1 * (size_t)n4 * (size_t)n5 * (size_t)n6;
  bytes = items * sizeof(double _Complex);
  (*zbuffer)[0][0][0][0][0] = (double _Complex*)malloc(bytes);
  if( (*zbuffer)[0][0][0][0][0] == NULL ) {
    fprintf(stderr, "[init_6level_zbuffer] Error from malloc\n");
    return(6);
  }
  o = 0;
  for(i=0; i<n1; i++) {
  for(k=0; k<n2; k++) {
  for(l=0; l<n3; l++) {
  for(m=0; m<n4; m++) {
  for(n=0; n<n5; n++) {
    (*zbuffer)[i][k][l][m][n] = (*zbuffer)[0][0][0][0][0] + o * n6;
    o++;
  }}}}}

  memset( (*zbuffer)[0][0][0][0][0], 0, bytes);
  return(0);
}  /* end of init_6level_zbuffer */

int fini_6level_zbuffer (double _Complex*******zbuffer) {
  if(*zbuffer != NULL) {
    if( (*zbuffer)[0] != NULL) { 
      if( (*zbuffer)[0][0] != NULL) { 
        if( (*zbuffer)[0][0][0] != NULL) { 
          if( (*zbuffer)[0][0][0][0] != NULL) { 
            if( (*zbuffer)[0][0][0][0][0] != NULL) {
              free( (*zbuffer)[0][0][0][0][0] );
            }
            free( (*zbuffer)[0][0][0][0] );
          }
          free( (*zbuffer)[0][0][0] ); 
        } 
        free( (*zbuffer)[0][0] ); 
      }
      free( (*zbuffer)[0] ); 
    }
    free( *zbuffer );
    *zbuffer = NULL;
  }
  return(0);
}  /* end of fini_6level_zbuffer */

/************************************************************************************
 * (de-)allocate 2-level integer buffer (n1 x n2 int matrix)
 ************************************************************************************/
int init_2level_ibuffer (int***buffer, unsigned int n1, unsigned int n2) {

  unsigned int i;
  size_t items = (size_t)n2;
  size_t bytes = items * n1 * sizeof(int);

  if(*buffer != NULL) {
    fprintf(stderr, "[init_2level_ibuffer] Error, buffer not NULL\n"); 
    return(1);
  }

  /* 1st, outer level */
  (*buffer) = (int**)malloc(n1 * sizeof(int*));
  if( *buffer == NULL ) {
    fprintf(stderr, "[init_2level_ibuffer] Error from malloc\n"); 
    return(2);
  }

  /* 2nd, inner level */
  (*buffer)[0] = (int*)malloc(bytes);
  if( (*buffer)[0] == NULL ) {
    fprintf(stderr, "[init_2level_ibuffer] Error from malloc\n"); 
    return(3);
  }
  for(i=1; i<n1; i++) (*buffer)[i] = (*buffer)[i-1] + items;
  memset( (*buffer)[0], 0, bytes);
  return(0);
}  /* end of init_2level_ibuffer */

int fini_2level_ibuffer (int***buffer) {
  if(*buffer != NULL) {
    if( (*buffer)[0] != NULL) { free( (*buffer)[0] ); }
    free( *buffer );
    *buffer = NULL;
  }
  return(0);
}  /* end of fini_2level_ibuffer */

}  /* end of namespace cvc */
