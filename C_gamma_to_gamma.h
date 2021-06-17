#ifndef C_GAMMA_TO_GAMMA_H
#define C_GAMMA_TO_GAMMA_H
/* [gamma_eq_C_ti_gamma_project] Thu Jul  6 09:54:35 2017 */

namespace cvc {

extern int C_gamma_to_gamma[16][2];

inline void init_c_gamma_to_gamma () {
  C_gamma_to_gamma[0][0] = 2;
  C_gamma_to_gamma[0][1] = -1;

  C_gamma_to_gamma[1][0] = 9;
  C_gamma_to_gamma[1][1] = 1;

  C_gamma_to_gamma[2][0] = 0;
  C_gamma_to_gamma[2][1] = 1;

  C_gamma_to_gamma[3][0] = 7;
  C_gamma_to_gamma[3][1] = -1;

  C_gamma_to_gamma[4][0] = 11;
  C_gamma_to_gamma[4][1] = 1;

  C_gamma_to_gamma[5][0] = 14;
  C_gamma_to_gamma[5][1] = 1;

  C_gamma_to_gamma[6][0] = 8;
  C_gamma_to_gamma[6][1] = -1;

  C_gamma_to_gamma[7][0] = 3;
  C_gamma_to_gamma[7][1] = 1;

  C_gamma_to_gamma[8][0] = 6;
  C_gamma_to_gamma[8][1] = 1;

  C_gamma_to_gamma[9][0] = 1;
  C_gamma_to_gamma[9][1] = -1;

  C_gamma_to_gamma[10][0] = 13;
  C_gamma_to_gamma[10][1] = 1;

  C_gamma_to_gamma[11][0] = 4;
  C_gamma_to_gamma[11][1] = -1;

  C_gamma_to_gamma[12][0] = 15;
  C_gamma_to_gamma[12][1] = -1;

  C_gamma_to_gamma[13][0] = 10;
  C_gamma_to_gamma[13][1] = -1;

  C_gamma_to_gamma[14][0] = 5;
  C_gamma_to_gamma[14][1] = -1;

  C_gamma_to_gamma[15][0] = 12;
  C_gamma_to_gamma[15][1] = 1;

}  /* end of init_c_gamma_to_gamma */

}  /* end of namespace cvc */
#endif
