#ifndef INIT_G_GAUGE_FIELD_H
#define INIT_G_GAUGE_FIELD_H

/**
 * @brief Initialize g_gauge_field global
 * Either the field is read using lime or taken from the tmLQCD libwrapper
 * functions.
 *
 * @return 0 on success, negative on any kind of failure
 */

namespace cvc { 

int init_g_gauge_field(void);

}

#endif // INIT_G_GAUGE_FIELD_H
