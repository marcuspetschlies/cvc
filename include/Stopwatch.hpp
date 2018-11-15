/***********************************************************************
 * Copyright (C) 2018 Bartosz Kostrzewa
 *
 * This file is part of cvc and is derived from the equivalent part of nyom.
 *
 * nyom is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * nyom is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with nyom.  If not, see <http://www.gnu.org/licenses/>.
 ***********************************************************************/

#pragma once

#include <chrono>
#include <mpi.h>
#include <iostream>

namespace cvc {

  typedef struct duration {
    double min;
    double max;
    double mean;
  } duration;

  /**
   * @brief Practical MPI parallel stopwatch. 
   */
class Stopwatch {
  public:
    Stopwatch() = delete;

#ifdef HAVE_MPI
    Stopwatch(MPI_Comm comm_in){
#else
    Stopwatch(int comm_in){
#endif
      comm = comm_in;
      init();
    }

    void init(void){
#ifdef HAVE_MPI
      MPI_Comm_rank(comm,
                    &rank);
      MPI_Comm_size(comm,
                    &Nranks);
#else
      rank = 0;
      Nranks = 0;
#endif
      reset();
    }

    void reset(void){
#ifdef HAVE_MPI
      MPI_Barrier(comm);
#endif
      time = std::chrono::steady_clock::now();
    }
   
    cvc::duration elapsed(void) {
      cvc::duration duration;
      double seconds;

      std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - time;
      seconds = elapsed_seconds.count();
#ifdef HAVE_MPI
      MPI_Allreduce(&seconds,
                    &(duration.mean),
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    comm);
      MPI_Allreduce(&seconds,
                    &(duration.min),
                    1,
                    MPI_DOUBLE,
                    MPI_MIN,
                    comm);
      MPI_Allreduce(&seconds,
                    &(duration.max),
                    1,
                    MPI_DOUBLE,
                    MPI_MAX,
                    comm);
      duration.mean = duration.mean / Nranks;
#else
      duration.mean = seconds;
      duration.min = seconds;
      duration.max = seconds;
#endif

      return(duration);
    }

    cvc::duration elapsed_print(const char* const name){
      cvc::duration duration = elapsed();
      if(rank==0){ 
        std::cout << name << " " << duration.mean 
          << " seconds" << std::endl
          << "min(" << duration.min << ") max(" 
          << duration.max << ")" << std::endl;
      }
      return(duration);
    }
    
    cvc::duration elapsed_print_and_reset(const char* const name){
      cvc::duration duration = elapsed_print(name);
      reset();
      return(duration);
    }

  private:
    std::chrono::time_point<std::chrono::steady_clock> time;
    int rank;
    int Nranks;
#ifdef HAVE_MPI
    MPI_Comm comm;
#else
    int comm;
#endif
};

} //namespace(cvc)

