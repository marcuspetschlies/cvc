#include<stdio.h>
#include<stdlib.h>
#include<string.h>
int main( int argc, char *argv[]){

  printf("T = 128\n");
  printf("LX = 64\n");
  printf("LY = 64\n");
  printf("LZ = 64\n");
  printf("filename_prefix = mod_Diagramm\n");
  printf("Nconf = 0700\n");

  printf("BeginTwopointFunctionGeneric\n");
  printf("  n       = 1\n");
  printf("  d       = 4\n");
  printf("  type    = b-b\n");
  printf("  tag     = D\n");
  printf("  reorder = 0\n");
  printf("  T       = 128\n");
  printf("EndTwopointFunction\n\n");

  FILE *in=fopen("source_positions.txt","r");
  for (int i=0;i<16;++i){
    int sx,sy,sz,st;
    fscanf(in,"%d,%d,%d,%d\n",&st,&sx,&sy,&sz);
    printf("source_coords = %d,%d,%d,%d\n",st,sx,sy,sz);
  }
  printf("\n");
  fclose(in);
 
  char ***irrep=(char ***)malloc(sizeof(char **) *4);

  char **group=(char **)malloc(sizeof(char *)*4);

  group[0]=(char*)malloc(sizeof(char)*100);
  group[1]=(char*)malloc(sizeof(char)*100);
  group[2]=(char*)malloc(sizeof(char)*100);
  group[3]=(char*)malloc(sizeof(char)*100); 

  snprintf(group[0],100,"2Oh");
  snprintf(group[1],100,"2C4v");
  snprintf(group[2],100,"2C2v");
  snprintf(group[3],100,"2C3v");


  int irrep_number[4]={4,2,1,3};

  irrep[0]=(char **)malloc(sizeof(char*)*4);
  irrep[1]=(char **)malloc(sizeof(char*)*2);
  irrep[2]=(char **)malloc(sizeof(char*)*1);
  irrep[3]=(char **)malloc(sizeof(char*)*3);
/* Irreps in the CM frame */

 
  irrep[0][0]=(char *)malloc(sizeof(char)*100);
  irrep[0][1]=(char *)malloc(sizeof(char)*100);
  irrep[0][2]=(char *)malloc(sizeof(char)*100);
  irrep[0][3]=(char *)malloc(sizeof(char)*100);
  irrep[1][0]=(char *)malloc(sizeof(char)*100);
  irrep[1][1]=(char *)malloc(sizeof(char)*100);
  irrep[2][0]=(char *)malloc(sizeof(char)*100);
  irrep[3][0]=(char *)malloc(sizeof(char)*100);
  irrep[3][1]=(char *)malloc(sizeof(char)*100);
  irrep[3][2]=(char *)malloc(sizeof(char)*100);

  snprintf(irrep[0][0],100,"G1g");
  snprintf(irrep[0][1],100,"G1u");
  snprintf(irrep[0][2],100,"Hg");
  snprintf(irrep[0][3],100,"Hu");


  snprintf(irrep[1][0],100,"G1");
  snprintf(irrep[1][1],100,"G1");


  snprintf(irrep[2][0],100,"G1");

  snprintf(irrep[3][0],100,"G1");
  snprintf(irrep[3][1],100,"K1");
  snprintf(irrep[3][2],100,"K2");

  int possible_mom_coord[3]={0,1,-1};

  for (int total_momentum_x_index=0; total_momentum_x_index<3; ++total_momentum_x_index){
   
    for (int total_momentum_y_index=0; total_momentum_y_index<3; ++total_momentum_y_index){

      for (int total_momentum_z_index=0; total_momentum_z_index<3; ++total_momentum_z_index){

//        printf("(%d,%d,%d)\n",possible_mom_coord[total_momentum_x_index],possible_mom_coord[total_momentum_y_index],possible_mom_coord[total_momentum_z_index]);
        int total_momentum_magnitude= possible_mom_coord[total_momentum_x_index]*possible_mom_coord[total_momentum_x_index]+possible_mom_coord[total_momentum_y_index]*possible_mom_coord[total_momentum_y_index]+possible_mom_coord[total_momentum_z_index]*possible_mom_coord[total_momentum_z_index];

        for (int irrep_index=0; irrep_index<irrep_number[total_momentum_magnitude]; ++irrep_index){
         
          printf("BeginTwopointFunctionInit\n");
          printf("  particlenamesource = D\n");
          printf("  particlenamesink   = D\n");
          printf("  irrep            = %s\n", irrep[total_momentum_magnitude][irrep_index]);
          printf("  listofmomentumf1 = %d %d %d\n", possible_mom_coord[total_momentum_x_index], possible_mom_coord[total_momentum_y_index],possible_mom_coord[total_momentum_z_index]);
          printf("  listofmomentumf2 = 0 0 0\n");
          printf("  sinkpionnucleontotalmomentum = 0 , 0\n");
          printf("  sourcepionnucleontotalmomentum = 0 , 0\n");
          printf("  sinkspin  = 3\n");
          printf("  sourcespin = 3\n");
          printf("  listofgammasf1 = 9,4 0,4 7,4 13,4 4,4 15,4 12,5 5,5 10,5\n");
          printf("  listofgammasi1 = 9,4 0,4 7,4 13,4 4,4 15,4 12,5 5,5 10,5\n");
          printf("  group         = %s\n", group[total_momentum_magnitude]);
          printf("EndTwopointFunction\n\n");
        
        }
        
      }

    }
 
  }
  free(irrep[3][0]);
  free(irrep[2][0]);
  free(irrep[1][0]);
  free(irrep[0][1]);
  free(irrep[0][0]);
  free(irrep[0]);
  free(irrep[1]);
  free(irrep[2]);
  free(irrep[3]);
  free(irrep);
  free(group[0]);
  free(group[1]);
  free(group[2]);
  free(group[3]);
  free(group);
}
