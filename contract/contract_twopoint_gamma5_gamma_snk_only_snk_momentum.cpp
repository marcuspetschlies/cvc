#include "global.h"
#include "loop_tools.h"
#include "cvc_complex.h"
#include "cvc_linalg.h"

#include <vector>
#include <cstring>

namespace cvc {

void contract_twopoint_gamma5_gamma_snk_only_snk_momentum(
    double * const contr, const int idsink, int const snk_mom[3],
    double const * chi, double const * psi, 
    double const factor)
{
  size_t const VOL3 = LX*LY*LZ;
  double const TWO_MPI = 2.0 * M_PI;
  double const px = TWO_MPI * (double)snk_mom[0] / (double)LX_global;
  double const py = TWO_MPI * (double)snk_mom[1] / (double)LY_global;
  double const pz = TWO_MPI * (double)snk_mom[2] / (double)LZ_global;

  double const phase_offset = (double)( g_proc_coords[1] * LX ) * px + 
                              (double)( g_proc_coords[2] * LY ) * py + 
                              (double)( g_proc_coords[3] * LZ ) * pz;
  
  // we pre-generate the vector of phases
  std::vector<cvc::complex> phases(VOL3);

#ifdef HAVE_OPENMP
#pragma omp parallel
#endif
  {
    std::vector<double> contr_tmp(2*T, 0.0);
    // pointer to make sure that accumulation into "contr" below
    // can be done atomically, although I'm not sure
    // it's even neessary
    double * contr_tmp_ptr = contr_tmp.data();

    double phase;
    unsigned int x, y, z;
    double spinor1[24], spinor2[24];

    size_t iix, ix3d;
    unsigned int x0;

    cvc::complex w1, w2;

    FOR_IN_PARALLEL(ix, 0, VOL3){
       x = g_lexic2coords[ix][1];
       y = g_lexic2coords[ix][2];
       z = g_lexic2coords[ix][3];

       phase = phase_offset + x*px + y*py + z*pz;
       phases[ix].re = factor*cos(phase);
       phases[ix].im = factor*sin(phase);
    }

    FOR_IN_PARALLEL(ix, 0, VOLUME)
    {
      iix = _GSI(ix);
      ix3d = ix % VOL3;
      x0 = g_lexic2coords[ix][0];

      _fv_eq_gamma_ti_fv(spinor1, idsink, psi+iix);
      _fv_eq_gamma_ti_fv(spinor2, 5, spinor1);
      _co_eq_fv_dag_ti_fv(&w1, chi+iix, spinor2);
      w2.re = w1.re * phases[ix3d].re - w1.im * phases[ix3d].im;
      w2.im = w1.re * phases[ix3d].im + w1.im * phases[ix3d].re;

      contr_tmp[2*x0  ] += w2.re;
      contr_tmp[2*x0+1] += w2.im;
    }

    // atomic write should be faster than having a lock
    for(int t = 0; t < 2*T; t++){
#ifdef HAVE_OPENMP
#pragma omp atomic
#endif
      contr[t] += contr_tmp_ptr[t];
    }
  } // end of parallel section if OpenMP in use

#ifdef HAVE_MPI
#if (defined PARALLELTX) || (defined PARALLELTXY) || (defined PARALLELTXYZ)
  std::vector<double> mpi_buffer(2*T);
  memcpy(mpi_buffer.data(), contr, 2*T*sizeof(double) );
  if( MPI_Reduce(mpi_buffer.data(), contr, 2*T, MPI_DOUBLE, MPI_SUM, 0, g_ts_comm ) != MPI_SUCCESS ) {
    fprintf( stderr, "[contract_twopoint_gamma5_gamma_snk_only_snk_momentum] Error from MPI_Reduce %s %d\n", __FILE__, __LINE__);
  }
#endif 
#endif

}

} // namespace(cvc)
