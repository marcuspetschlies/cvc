/*************************************************************
 * cvc_timer.h                                               *
 *************************************************************/
#ifndef _CVC_TIMER_H
#define _CVC_TIMER_H

#include <sys/time.h>

namespace cvc {

/***************************************************************************
 * calculate elapsed wall-time
 ***************************************************************************/
inline void show_time ( struct timeval * const ta, struct timeval * const tb, char * tag, char * timer, int const io ) {

  long int seconds =  tb->tv_sec  - ta->tv_sec;
  long int useconds = tb->tv_usec - ta->tv_usec;
  if ( useconds < 0 ) {
    useconds += 1000000;
    seconds--;
  }
  if ( io ) {
    /* fprintf ( stdout, "# [%s] time for %s %ld sec %ld usec\n", tag, timer, seconds, useconds ); */
    fprintf ( stdout, "# [%s] time for %s %e sec\n", tag, timer, (double)seconds + (double)useconds/1000000. );
  }

}  /* end of show_time */


}
#endif
