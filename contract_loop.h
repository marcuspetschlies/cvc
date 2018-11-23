#ifndef _CONTRACT_LOOP_H
#define _CONTRACT_LOOP_H
/****************************************************
 * contract_loop.h
 *
 * PURPOSE:
 * DONE:
 * TODO:
 ****************************************************/
namespace cvc {

int contract_local_loop_stochastic ( double *** const loop, double * const source, double * const prop, int const momentum_number, int ( * const momentum_list)[3] );

}  /* end of namespace cvc */
