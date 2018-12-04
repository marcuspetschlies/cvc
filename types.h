#ifndef _TYPES_H
#define _TYPES_H

#include <stdexcept>
#include <string>
#include <cstring>

namespace cvc {

typedef double * spinor_vector_type;
typedef double * fermion_vector_type;
typedef double ** fermion_propagator_type;
typedef double ** spinor_propagator_type;

typedef struct {
  int gi;
  int gf;
  int pi[3];
  int pf[3];
} m_m_2pt_type;

typedef struct {
  int gi1;
  int gi2;
  int gf1;
  int gf2;
  int pi1[3];
  int pi2[3];
  int pf1[3];
  int pf2[3];
} mxb_mxb_2pt_type;

typedef struct {
  double *****v;
  double *ptr;
  int np;
  int ng;
  int nt;
  int nv;
} gsp_type;

static inline std::string check_flav(const std::string flav_in)
{
  if( !(flav_in == "u" || flav_in == "d" || 
        flav_in == "sp" || flav_in == "sm" ||
        flav_in == "cp" || flav_in == "cm") ){
    throw std::invalid_argument("flavor has to be one of 'u', 'd', 'sp', 'sm', 'cp', 'cm'\n");
  }
  return flav_in;
}

static inline std::string check_mom_prop(const std::string prop_in)
{
  if( !(prop_in == "fwd" || prop_in == "bwd") ){
    throw std::invalid_argument("propagator with momentum has to be either 'fwd' or 'bwd'\n");
  }
  return prop_in;
}

typedef struct stoch_prop_meta_t
{
  int p[3];
  int gamma;
  std::string flav;

  stoch_prop_meta_t() {}

  stoch_prop_meta_t(const int p_in[3],
                    const int gamma_in,
                    const std::string flav_in)
  {
    p[0] = p_in[0];
    p[1] = p_in[1];
    p[2] = p_in[2];
    gamma = gamma_in;
    flav = check_flav(flav_in);
  }

  std::string make_key(void){
    return make_key(p, gamma, flav);
  }

  std::string make_key(const int p_in[3],
                       const int gamma_in,
                       const std::string flav_in)
  {
    char temp[100];
    snprintf(temp, 100, 
             "f%s_g%02dd_px%+dpy%+dpz%+d",
             flav_in.c_str(), gamma_in, p_in[0], p_in[1], p_in[2]);
    return std::string(temp);
  }

} stoch_prop_meta_t;


/**
 * @brief Meta-description of a meson two-point function
 *
 * Describes the connected part of the two-point function
 *
 * \sum_{x_i, x_f}  e^{-i p_i x_i} e^{i p_f x_f}
 * \bar{chi}_{f_1}(x_f) gf chi_{f_2}(x_f) \bar{chi}_{f_2}(x_i) gi f_1(x_i)
 *
 * in the twisted basis. This is computed as
 *
 * Tr[ g5 (S_{f_1})^dag g5 gf S_{f_2} gi ]
 * = [D^{-1}_{f_1^dag} gb eta]^dag g5 gf [D^{-1}_{f_2} g_i eta]  (1)
 *
 * where (f_1)^dag is bprop_flav (so if f_1 = d, bprop_flav = u)
 *        f_2      is fprop_flav
 *
 * The momentum at source can either be inlcuded in the backward propagator
 * (in which case the phase is complex conjugated) or the forward propagator,
 * such that one can specify either "fwd" or "bwd" for src_mom_prop
 *
 * Similarly, in the writing of (1), it can be that it is more convenient to change
 * the gamma structure in the backward propagator, so we can specify 'gb' freely with the caveat
 * that what happens under complex conjugation must be considered externally,
 * unlike for the momentum.
 *
 */

typedef struct twopt_oet_meta_t {
  twopt_oet_meta_t(const std::string fprop_flav_in,
                   const std::string bprop_flav_in,
                   const std::string src_mom_prop_in,
                   const int gi_in,
                   const int gf_in,
                   const int gb_in)
  {
    gi = gi_in;
    gf = gf_in;
    gb = gb_in;
    fprop_flav = check_flav(fprop_flav_in);
    bprop_flav = check_flav(bprop_flav_in);
    src_mom_prop = check_mom_prop(src_mom_prop_in);
  }

  std::string fprop_flav;
  std::string bprop_flav;
  std::string src_mom_prop;
  int gi;
  int gf;
  int gb;

} twopt_oet_meta_t;


typedef struct threept_meta_t {
  int pi[3];
  int pf[3];
  int pc[3];
  int gf;
  int gi;
  std::string fpropkey;
  std::string bpropkey;
  std::string spropkey;
} threept_oet_meta_t; 

}  /* end of namespace cvc */
#endif
