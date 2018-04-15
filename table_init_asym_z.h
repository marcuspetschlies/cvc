#ifndef _TABLE_INIT_ASYM_H
#define _TABLE_INIT_ASYM_H

/****************************************************
 * table_init_asym_z.h
 *
 * Fr 13. Apr 13:32:51 CEST 2018
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/

namespace cvc {


/************************************************************************************/
/************************************************************************************/

/************************************************************************************
 * (de-)allocate 2-level asymmetric buffer (n1 x dim[i] double _Complex matrix)
 ************************************************************************************/
inline double _Complex ** init_2level_ztable_asym ( int const N0, int * const dim) {

  // 1st, outer level
  double _Complex ** s = (double _Complex**)malloc( N0 * sizeof(double _Complex*));
  if( s == NULL ) {
    fprintf(stderr, "[init_2level_zbuffer_asym] Error from malloc %s %d\n", __FILE__, __LINE__); 
    return( NULL );
  }

  // 2nd, inner level
  size_t items = 0;
  for ( int i = 0; i < N0; i++) items += (size_t)dim[i];

  s[0] = (double _Complex*)malloc( items * sizeof(double _Complex) );
  if( s[0] == NULL ) {
    fprintf(stderr, "[init_2level_zbuffer_asym] Error from malloc %s %d\n", __FILE__, __LINE__); 
    return( NULL );
  }
  for( int i = 1; i < N0; i++) s[i] = s[i-1] + dim[i-1];
  memset( s[0], 0, items * sizeof(double _Complex) );

  return( s );
}  // end of init_2level_ztable_asym


/************************************************************************************/
/************************************************************************************/

inline double _Complex *** init_3level_ztable_asym (int const N0, int const N1, int * const dim ) {

  double _Complex ** s__ = init_2level_ztable_asym ( N0*N1, dim );
  if ( s__ == NULL ) return( NULL );

  double _Complex *** s_ = ( double _Complex ***) malloc( N0 * sizeof( double _Complex **) );
  if ( s_ == NULL ) return ( NULL );

  for ( int i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_3level_ztable_asym

/************************************************************************************/
/************************************************************************************/

inline double _Complex **** init_4level_ztable_asym (int const N0, int const N1, int const N2, int * const dim ) {

  double _Complex *** s__ = init_3level_ztable_asym ( N0*N1, N2, dim );
  if ( s__ == NULL ) return( NULL );

  double _Complex **** s_ = ( double _Complex ****) malloc( N0 * sizeof( double _Complex ***) );
  if ( s_ == NULL ) return ( NULL );

  for ( int i = 0; i < N0; i++ ) s_[i] = s__ + i * N1;
  return( s_ );
}  // end of init_4level_ztable_asym

/************************************************************************************/
/************************************************************************************/

inline void fini_2level_ztable_asym ( double _Complex *** s ) {
  if( *s != NULL) {
    if( ( *s)[0] != NULL) { free( (*s)[0] ); }
    free( *s );
    *s = NULL;
  }
}  // end of fini_2level_zbuffer_asym

/************************************************************************************/
/************************************************************************************/

inline void fini_3level_ztable_asym ( double _Complex **** s  ) {
  if ( *s != NULL ) {
    fini_2level_ztable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_3level_ztable_asym
          
/************************************************************************************/
/************************************************************************************/

inline void fini_4level_ztable_asym ( double _Complex ***** s  ) {
  if ( *s != NULL ) {
    fini_3level_ztable ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_4level_ztable_asym
          
/************************************************************************************/
/************************************************************************************/

}  // end of namespace cvc

#endif
