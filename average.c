#include <stdio.h>
#include <math.h>
#define N 1000

/**************************************************************************/
/* Purpose: This program reads a file of real numbers and computes its    */
/*    mean, variance, and standard deviation                              */ 
/**************************************************************************/

FILE *datafile; /* pointer to data file */

main(int argc, char *argv[])
{
  char temp[60];
  int i;
  float data[N], sum, mean, variance, stdev;

  strcpy(temp, argv[1]);

  /* open data file */
  datafile = fopen(temp,"r");
  if (!datafile)
  {
    printf("*** Error opening %s ***\n",temp);
    exit(1);
  }

  /* read entries from data file */
  for (i=0; i<N; i++)
    fscanf(datafile,"%f\n",&data[i]);

  /* compute and print mean */
  sum = 0.0;
  for (i=0; i<N; i++)
    sum = sum + data[i];
  mean = sum/((float)N);
  printf("mean = %f\n",mean);

  /* compute and print variance */
  sum = 0.0;
  for (i=0; i<N; i++)
    sum = sum + pow(data[i]-mean,2);
  variance = sum/((float)N);
  printf("variance = %f\n", variance);

  /* compute and print standard deviation */
  stdev = sqrt(variance);
  printf("standard deviation = %f\n", stdev);
 
  /* close file and exit */
  fclose(datafile);
  exit(1);
}
