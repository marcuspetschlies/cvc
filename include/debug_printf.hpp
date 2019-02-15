/***********************************************************************
 *  
 * Copyright (C) 2018 Bartosz Kostrzewa
 *
 ***********************************************************************/

#ifndef DEBUG_PRINTF_HPP
#define DEBUG_PRINTF_HPP

#include "global.h"

#include <cstdarg>
#include <cstdio>

/* Function along the lines of printf which produces output on a single
 * or all MPI tasks (unordered) when g_debug_level is at or
 * above the provided threshold 
 * to have output by all MPI tasks, simply pass g_proc_id for proc_id */

namespace cvc {

static inline void debug_printf(const int proc_id,
                                const int verbosity_level_threshold,
                                const char * format,
                                ...)
{
  if( g_proc_id == proc_id && g_verbose >= verbosity_level_threshold ){
    va_list arglist;
    va_start(arglist, format);
    vprintf(format, arglist);
    va_end(arglist);
    fflush(stdout);
  }
}

} // namespace(cvc)

#endif
