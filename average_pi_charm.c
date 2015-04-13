/*************************************************************
*average_pi_charm.c 
*
*September 28th, 2011
*************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "stat5.h"

FILE *datafile, *resultfile;

#define FILESTOTAL 2
#define LINESTOTAL 3984

int main(){
	char g[3], file[15], file_prefix[10],result[10];
	int i,j, gc;
	double *pi_q, *q2, *im;
	int *numberq;

strcpy(file_prefix,"pi_");
strcpy(result, "pi_all");

pi_q=(double *) calloc(LINESTOTAL,sizeof(double));
  if(pi_q==(double*)NULL) {
    printf("could not allocate memory for pi_q\n");
    return(105);
  }
q2=(double *) calloc(LINESTOTAL,sizeof(double));
  if(q2==(double*)NULL) {
    printf("could not allocate memory for q2\n");
    return(106);
  }
im=(double *) calloc(LINESTOTAL,sizeof(double));
  if(im==(double*)NULL) {
    printf("could not allocate memory for im\n");
    return(107);
  }
numberq=(int *) calloc(LINESTOTAL,sizeof(int));
  if(numberq==(int*)NULL) {
    printf("could not allocate memory for numberq\n");
    return(108);
  }

clear5(LINESTOTAL,100);

for(i=0;i<FILESTOTAL;i++){

  sprintf(file,"%s%.2d",file_prefix,i);

  datafile=fopen(file,"r");

   if (!datafile)
    {
     printf("*** Error opening %s ***\n",file);
     exit;
    }

   for(j=0; j<LINESTOTAL; j++){
    fscanf(datafile,"%lf %lf %lf %d ", q2+j , pi_q+j, im+j, numberq+j);
//    printf("j=%4d,pi_q= %25.16lf\n",j,pi_q[j]);
    accum5(j+1,*(pi_q+j));
   }

  fclose(datafile);
}

resultfile=fopen(result,"w");

  if (!resultfile)
    {
     printf("*** Error opening %s ***\n",result);
     exit;
    }

for(j=0; j<LINESTOTAL; j++){
   fprintf(resultfile,"%21.12e\t%25.16e\t%21.12e\n",q2[j], aver5(j+1), sigma5(j+1));
}

fclose(resultfile);

free(q2);
free(pi_q);
free(im);
free(numberq);

return 0;
}