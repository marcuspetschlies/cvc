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
inline double _Complex ** init_2level_ztable_asym ( int const N0, int * const dim, int const ndim ) {


  // 1st, outer level
  double _Complex ** s = (double _Complex**)malloc( N0 * ndim * sizeof(double _Complex*));
  if( s == NULL ) {
    fprintf(stderr, "[init_2level_zbuffer_asym] Error from malloc %s %d\n", __FILE__, __LINE__); 
    return( NULL );
  }

  // 2nd, inner level
  size_t items = 0;
  for ( int i = 0; i < ndim; i++) items += (size_t)dim[i];
  size_t const bytes = N0 * items * sizeof(double _Complex);

  s[0] = (double _Complex*)malloc( bytes );
  if( s[0] == NULL ) {
    fprintf(stderr, "[init_2level_zbuffer_asym] Error from malloc for items = %lu %s %d\n", items, __FILE__, __LINE__); 
    return( NULL );
  }

  for( int i = 0; i < N0; i++) {
    double _Complex ** const s_ = s + i * ndim;
    s_[0] = s[0] + i * items;

    for ( int k = 1; k < ndim; k++ ) {
      s_[k] = s_[k-1] + dim[k-1];
    }
  }
  memset( s[0], 0, bytes );

  return( s );
}  // end of init_2level_ztable_asym


/************************************************************************************/
/************************************************************************************/

inline double _Complex *** init_3level_ztable_asym (int const N0, int const N1, int * const dim ) {

  double _Complex ** s__ = init_2level_ztable_asym ( N0, dim, N1 );
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
    fini_2level_ztable_asym ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_3level_ztable_asym
          
/************************************************************************************/
/************************************************************************************/

inline void fini_4level_ztable_asym ( double _Complex ***** s  ) {
  if ( *s != NULL ) {
    fini_3level_ztable_asym ( *s );
    free ( *s );
    *s = NULL;
  }
}  // end of fini_4level_ztable_asym
          
/************************************************************************************/
/************************************************************************************/

}  // end of namespace cvc

#endif
