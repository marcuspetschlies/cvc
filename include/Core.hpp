#ifdef HAVE_MPI
#include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#ifdef HAVE_LHPC_AFF
#include "lhpc-aff.h"
#endif

extern "C"
{
#ifdef HAVE_TMLQCD_LIBWRAPPER
#include "tmLQCD.h"
#endif
}

#include "Stopwatch.hpp"
#include "read_input_parser.h"
#include "mpi_init.h"
#include "global.h"
#include "cvc_geometry.h"

namespace cvc {

class Core {
  public:
    Core(int argc, char ** argv){
#ifdef HAVE_TMLQCD_LIBWRAPPER
      tmLQCD_init_parallel_and_read_input(argc, argv, 1, "invert.input");
#else
#ifdef HAVE_MPI
      MPI_Init(&argc, &argv);
#endif
#endif
      // TODO cmd line parsing
      read_input_parser("cvc.input");

#ifdef HAVE_TMLQCD_LIBWRAPPER
      tmLQCD_invert_init(argc, argv, 1, 0);
      tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
      tmLQCD_get_lat_params(&g_tmLQCD_lat);
#endif
      mpi_init(argc, argv);
      sw = new Stopwatch(g_cart_grid);

      init_geometry();
      geometry();
    }

    Core() = delete;

    ~Core()
    {
      sw->elapsed_print("Core lifetime");
      delete(sw);
#ifdef HAVE_TMLQCD_LIBWRAPPER
      tmLQCD_finalise();
#endif
      free_geometry();
#ifdef HAVE_MPI
      mpi_fini_datatypes();
      MPI_Finalize();
#endif
    }


  private:
    Stopwatch* sw;
};

} // namespace(cvc)
