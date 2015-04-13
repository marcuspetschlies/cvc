/***********************************************************************
 * read_input_parser.h 
 ***********************************************************************
 * 
 * This is the function to parse the input file.
 * No default values for any paramter will be set
 *
 * read_inputg expects the filename of the input file
 * as an input parameter.
 *
 * read_input returns 2 if the input file did not exist 
 *
 ***********************************************************************/

#ifndef _PARSER_H
# define _PARSER_H

  extern int verbose;
  extern int myverbose;
  extern double ft_rmax[4];

  int read_input_parser(char *);
  int reread_input_parser(char *);

  
#endif
