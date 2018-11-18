
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
#include "debug_printf.hpp"
#include "read_input_parser.h"
#include "mpi_init.h"
#include "global.h"
#include "cvc_geometry.h"

#include <stdexcept>
#include <string>
#include <boost/program_options.hpp>

namespace cvc {

  namespace po=boost::program_options;

  /**
   * @brief CVC Core class
   * The CVC core class manages execution of a CVC run. It processes command line
   * options and provides a representation of these to the outside world.
   * Command line arguments for different applications are supported by providing, 
   * at construction, a function pointer to a function which initialises 
   * any command line options beyond the defaults. 
   * It also reads the input file (after processing command line options),
   * initialises MPI and geomtry as well as, if compiled in, the tmLQCD wrapper.
   *
   * Note that the destructor calls MPI_Finalize, such that cvc::Core is
   * a singleton.
   *
   */
class Core {
  public:
    Core(int argc,
         char ** argv,
         std::string application_name_in = "cvc",
         void (*external_declare_cmd_options_in)(po::options_description &) = nullptr ) :
      sw(nullptr),
      world_rank(0),
      application_name(application_name_in),
      cmd_desc("cmd_options"),
      core_mpi_initialised(false),
      mpi_initialised(false),
      geom_initialised(false),
      tmLQCD_initialised(false),
      core_tmLQCD_initialised(false),
      initialised(false)
    {
      // we want to init MPI as soon as possible
      core_mpi_init(argc,argv);
      // and this allows us to have "world_rank" well-defined
      // as well as giving us access to MPI functions
      sw = new Stopwatch(MPI_COMM_WORLD);
      
      // now declare default cmdline arguments
      declare_default_cmd_options();
      // and external arguments, if the function has been provided
      if( external_declare_cmd_options_in ){
        external_declare_cmd_options = external_declare_cmd_options_in;
        (*external_declare_cmd_options)(cmd_desc);
      }
      // parse the cmdline arguments
      try
      {
        po::store(po::parse_command_line(argc, argv, cmd_desc),
                  cmd_options);
        notify(cmd_options);
      }
      catch( const std::exception &ex )
      {
        std::cerr << ex.what() << std::endl;
      }

      // if usage information is requested, we display it and finalise
      if( cmd_options.count("help") ){
        if(world_rank == 0){
          std::cout << cmd_desc << std::endl;
        }
        EXIT(0);
        return;
      // otherwise we're finally ready to initialise everything else
      } else {
        init(argc,argv);
      }
      initialised = true;
    }

    // we don't want a default constructor because initialisation is non-trivial
    Core() = delete;

    ~Core()
    {
      sw->elapsed_print("Core lifetime");
      delete(sw);

#ifdef HAVE_TMLQCD_LIBWRAPPER
      if( tmLQCD_initialised ) tmLQCD_finalise();
#endif
      if( geom_initialised ) free_geometry();
#ifdef HAVE_MPI
      if( mpi_initialised ) mpi_fini_datatypes();
      if( core_mpi_initialised ) MPI_Finalize();
#endif
    }

    const po::variables_map &
    get_cmd_options(void) const
    {
      return cmd_options;
    }

    const bool
    is_initialised(void) const
    {
      return initialised;
    }

  private:
    int world_rank;
    
    bool core_mpi_initialised,
         mpi_initialised,
         geom_initialised,
         core_tmLQCD_initialised,
         tmLQCD_initialised,
         initialised;
    
    cvc::Stopwatch* sw;
    std::string application_name;
    po::options_description cmd_desc;
    po::variables_map cmd_options;
    
    /**
     * @brief Pointer to external function to populate cmd_desc.
     * If command line arguments beyond the default are required,
     * the constructor can be passed a function pointer to a function
     * which defines additional cmdline options.
     *
     * @param reference to cmd_desc
     */
    void (*external_declare_cmd_options)(po::options_description &);

    /**
     * @brief Initialise core MPI functionality
     * Calls either tmLQCD_init_parallel_and_read_input or MPI_Init
     * to initialise MPI. Sets core_mpi_initialised and, if tmLQCD
     * is used, core_tmLQCD_initialised.
     *
     * @param argc
     * @param argv
     */
    void core_mpi_init(int argc, char **argv)
    {
      // need to check if MPI_Init has already been called somewhere else
      int check_core_mpi_initialised = 0;
#ifdef HAVE_MPI
      MPI_Initialized(&check_core_mpi_initialised);
#endif
      std::cout << "check_core_mpi_initialised=" << check_core_mpi_initialised << std::endl;
      if( !check_core_mpi_initialised ){
        int status;
        // Currently, tmLQCD_init_parallel_and_read_input must be called because
        // tmLQCD initialises either QMP or MPI and chooses the thread level
        // which is requested at initialisation
        #ifdef HAVE_TMLQCD_LIBWRAPPER
        status = tmLQCD_init_parallel_and_read_input(argc, argv, 1, "tmLQCD.input");
        if(status == 0){
          core_tmLQCD_initialised = true;
          core_mpi_initialised = true;
        } else {
          debug_printf(0,0,"[Core::core_mpi_init] tmLQCD_init_parallel_and_read_input failed!\n");
          return;
        } 
        #else
        #ifdef HAVE_MPI
        status = MPI_Init(&argc, &argv);
        if( status == MPI_SUCCESS ){
          core_mpi_initialised = true;
        } else {
          debug_print(0,0,"[Core::core_mpi_init] MPI_Init failed!\n");
          return;
        }
        #endif
        #endif
      } else {
        core_mpi_initialised = true;
      } // if(&check_core_mpi_initialised)

      #ifdef HAVE_MPI
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
      #endif
    }

    void init(int argc, char ** argv){
      int status;
      read_input_parser( cmd_options["input_filename"].as<std::string>().c_str() );
      
#ifdef HAVE_TMLQCD_LIBWRAPPER
      status = tmLQCD_invert_init(argc, argv, 1, 0);
      if( status == 0){
        tmLQCD_initialised = true;
      } else {
        debug_printf(0,0,"[Core::init] tmLQCD_invert_init failed!\n");
        return;
      }
      status = tmLQCD_get_mpi_params(&g_tmLQCD_mpi);
      if( status != 0 ){
        debug_printf(0,0,"[Core::init] tmLQCD_get_mpi_params failed\n");
        return;
      }
      status = tmLQCD_get_lat_params(&g_tmLQCD_lat);
      if( status != 0){
        debug_printf(0,0,"[Core::init] tmLQCD_get_lat_params failed\n");
        return;
      }
#endif
      mpi_init(argc, argv);
      mpi_initialised = true;

      status = init_geometry();
      if( status == 0 ){
        geom_initialised = true;
      } else {
        debug_printf(0,0,"[Core::init] init_geometry failed!\n");
        return;
      }
      geometry();
    }

    void declare_default_cmd_options(void){
      cmd_desc.add_options()
        ("help,h", "usage information")
        ("input_filename,f",
         po::value<std::string>()->default_value( std::string("cvc.input") ),
         "CVC input file name")
        ("verbose,v", "verbose core output");
    }
};

} // namespace(cvc)
