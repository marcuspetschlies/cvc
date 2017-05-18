#ifndef _ROTATIONS_H
#define _ROTATIONS_H

namespace cvc {

typedef struct {
  int n[3];
  double w;
} rotation_type; 

extern rotation_type cubic_group_double_cover_rotations[48];

void init_rotation_table (void);

long unsigned int factorial (int n);

void axis2polar ( double*theta, double*phi, int n[3] );

void rotation_matrix_spherical_basis ( double _Complex**R, int J2, int n[3], double w);

void spherical2cartesian_3x3 (double _Complex **C, double _Complex **S);

void rot_mat_adj (double _Complex **C, double _Complex **R, int N);

void rot_mat_ti_mat (double _Complex **C, double _Complex **A, double _Complex **B, int N);

void rot_mat_ti_mat_adj (double _Complex **C, double _Complex **A, double _Complex **B, int N);

void rot_mat_adj_ti_mat (double _Complex **C, double _Complex **A, double _Complex **B, int N);

void rot_printf_matrix (double _Complex **R, int N, char *A, FILE*ofs );

double rot_mat_norm2 (double _Complex **R, int N);

int rot_mat_check_is_sun (double _Complex **R, int N);

void rot_point ( int nrot[3], int n[3], double _Complex **R);

void rot_point_inv ( int nrot[3], int n[3], double _Complex **R);

int rot_mat_check_is_real_int (double _Complex **R, int N );

void rot_global_point_mod ( int nrot[3], int n[3], double _Complex **R);

void rot_center_global_point ( int nrot[3], int n[3], double _Complex **R);

void rot_center_global_point_inv ( int nrot[3], int n[3], double _Complex **R);

void rot_center_local_point ( int nrot[3], int n[3], double _Complex **R, int l[3]);

void rot_center_local_point_inv ( int nrot[3], int n[3], double _Complex **R, int l[3]);

int rot_gauge_field ( double*gf_rot, double *gf, double _Complex **R);

void rot_printf_rint_matrix (double _Complex **R, int N, char *A, FILE*ofs );


/***********************************************************
 * check boundary status of a point
 ***********************************************************/
inline int rot_check_point_bnd ( int n[3] ) {
  int bnd = 0;
  if ( n[0] == LX || n[0] == -1) bnd++;
  if ( n[1] == LY || n[1] == -1) bnd++;
  if ( n[2] == LZ || n[2] == -1) bnd++;
  if (bnd > 2 ) {
    if (g_cart_id == 0 ) {
      fprintf(stdout, "# [rot_check_point_bnd] *******************************************************\n"\
                      "# [rot_check_point_bnd] * WARNING bnd exceeds 2 for n = (%3d, %3d, %3d)\n"\
                      "# [rot_check_point_bnd] *******************************************************\n",
                      n[0], n[1], n[2]);
      fflush(stdout);
    }
  }
  return(bnd > 2);
}  /* end of rot_check_point_bnd */

/***********************************************************
 * reduce point coordinates according to MPI boundary
 ***********************************************************/
inline void rot_reduce_point_bnd ( int nred[3], int n[3] ) {
#ifndef HAVE_MPI
  nred[0] = ( n[0] + LX ) % LX;
  nred[1] = ( n[1] + LY ) % LY;
  nred[2] = ( n[2] + LZ ) % LZ;
#else
  nred[0] = n[0];
  nred[1] = n[1];
  nred[2] = n[2];
# if !( defined PARALLELTXYZ ) /* z-direction NOT parallelized */
  nred[2] = ( n[2] + LZ ) % LZ;
#  endif

#  if !( defined PARALLELTXY ) && !( defined PARALLELTXYZ )  /* y-direction NOT parallelized */
  nred[1] = ( n[1] + LY ) % LY;
#  endif

#  if !( defined PARALLELTX ) && !( defined PARALLELTXY ) && !( defined PARALLELTXYZ ) /* x-direction NOT parallelized */
  nred[0] = ( n[0] + LX ) % LX;
#  endif
#endif
  return;
}  /* end of rot_reduce_point_bnd */


}  /* end of namespace cvc */

#endif
