#include<stdio.h>
#include<stdlib.h>
#include<string.h>
int main( int argc, char *argv[]){

  printf("BeginTwopointFunctionGeneric\n");
  printf("  n       = 1\n");
  printf("  d       = 1\n");
  printf("  type    = b-b\n");
  printf("  tag     = piN\n");
  printf("  reorder = 0\n");
  printf("  T       = 128\n");
  printf("EndTwopointFunction\n\n");
 
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


  int irrep_number[4]={6,2,1,3};

  irrep[0]=(char **)malloc(sizeof(char*)*6);
  irrep[1]=(char **)malloc(sizeof(char*)*2);
  irrep[2]=(char **)malloc(sizeof(char*)*1);
  irrep[3]=(char **)malloc(sizeof(char*)*3);
/* Irreps in the CM frame */
 
  irrep[0][0]=(char *)malloc(sizeof(char)*100);
  irrep[0][1]=(char *)malloc(sizeof(char)*100);
  irrep[0][2]=(char *)malloc(sizeof(char)*100);
  irrep[0][3]=(char *)malloc(sizeof(char)*100);
  irrep[0][4]=(char *)malloc(sizeof(char)*100);
  irrep[0][5]=(char *)malloc(sizeof(char)*100);
  irrep[1][0]=(char *)malloc(sizeof(char)*100);
  irrep[1][1]=(char *)malloc(sizeof(char)*100);
  irrep[2][0]=(char *)malloc(sizeof(char)*100);
  irrep[3][0]=(char *)malloc(sizeof(char)*100);
  irrep[3][1]=(char *)malloc(sizeof(char)*100);
  irrep[3][2]=(char *)malloc(sizeof(char)*100);

  snprintf(irrep[0][0],100,"G1g");
  snprintf(irrep[0][1],100,"G1u");
  snprintf(irrep[0][2],100,"G2g");
  snprintf(irrep[0][3],100,"G2u");
  snprintf(irrep[0][4],100,"Hg");
  snprintf(irrep[0][5],100,"Hu");

  int *momlistf1=(int *)malloc(sizeof(int)*200);
  int *momlistf2=(int *)malloc(sizeof(int)*200);


  snprintf(irrep[1][0],100,"G1");
  snprintf(irrep[1][1],100,"G2");

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
         
          for (int baryon_momentum=0;baryon_momentum<4;++baryon_momentum) {
          
            for (int meson_momentum=0; meson_momentum<4; ++meson_momentum) {

              if ( ( meson_momentum + baryon_momentum ) == total_momentum_magnitude || ( (-meson_momentum+baryon_momentum)== total_momentum_magnitude) || ( ( meson_momentum- baryon_momentum)== total_momentum_magnitude) ) {

  //               printf("meson magnitude %d baryon magnitude %d\n",meson_momentum, baryon_momentum );

                 int momlist_index=0;

                 for ( int baryon_x_ind=0 ;  baryon_x_ind< 3; ++baryon_x_ind ) {

                   for ( int baryon_y_ind=0 ;  baryon_y_ind< 3; ++baryon_y_ind ) {

                     for ( int baryon_z_ind=0 ;  baryon_z_ind< 3; ++baryon_z_ind ) {

                       for ( int meson_x_ind=0 ;  meson_x_ind< 3; ++meson_x_ind ) {

                         for ( int meson_y_ind=0 ;  meson_y_ind< 3; ++meson_y_ind ) {

                           for ( int meson_z_ind=0 ;  meson_z_ind< 3; ++meson_z_ind ) {

                             if ( ( ( possible_mom_coord[ baryon_x_ind ]*possible_mom_coord[ baryon_x_ind ]+ possible_mom_coord[baryon_y_ind]*possible_mom_coord[ baryon_y_ind ] + possible_mom_coord[baryon_z_ind]*possible_mom_coord[ baryon_z_ind ] )== baryon_momentum ) &&
                                  ( ( possible_mom_coord[ meson_x_ind  ]*possible_mom_coord[ meson_x_ind  ]+ possible_mom_coord[meson_y_ind]*possible_mom_coord[ meson_y_ind  ] +  possible_mom_coord[meson_z_ind]*possible_mom_coord[ meson_z_ind  ] ) == meson_momentum ) && 
                                  ( ( possible_mom_coord[ baryon_x_ind]+possible_mom_coord[ meson_x_ind]  ) == possible_mom_coord[total_momentum_x_index] ) &&  
                                  ( ( possible_mom_coord[ baryon_y_ind]+possible_mom_coord[ meson_y_ind]  ) == possible_mom_coord[total_momentum_y_index] ) &&
                                  ( ( possible_mom_coord[ baryon_z_ind]+possible_mom_coord[ meson_z_ind]  ) == possible_mom_coord[total_momentum_z_index] ) ) {

//                             printf("f1(%d,%d,%d),f2(%d,%d,%d)\n", possible_mom_coord[ baryon_x_ind ], possible_mom_coord[ baryon_y_ind ], possible_mom_coord[ baryon_z_ind ], possible_mom_coord[ meson_x_ind ], possible_mom_coord[ meson_y_ind ], possible_mom_coord[ meson_z_ind ] );
                               momlistf1[momlist_index+0]= possible_mom_coord[ baryon_x_ind ];
                               momlistf1[momlist_index+1]= possible_mom_coord[ baryon_y_ind ];
                               momlistf1[momlist_index+2]= possible_mom_coord[ baryon_z_ind ];

                               momlistf2[momlist_index+0]= possible_mom_coord[ meson_x_ind ];
                               momlistf2[momlist_index+1]= possible_mom_coord[ meson_y_ind ];
                               momlistf2[momlist_index+2]= possible_mom_coord[ meson_z_ind ];

                               momlist_index+=3;



                             }

                           }
     
                         }
 
                       }

                     }

                   }
                 }
                 printf("BeginTwopointFunctionInit\n");
                 printf("  irrep            = %s\n", irrep[total_momentum_magnitude][irrep_index]);
                 printf("  listofmomentumf1 = ");
                 for (int i=0; i<momlist_index; ++i){
                   if (i==(momlist_index-1)){
                     printf("%d\n", momlistf1[i]);
                   }
                   else{
                     printf("%d, ",momlistf1[i]);
                   }      
                 }
                 printf("  listofmomentumf2 = ");
                 for (int i=0; i<momlist_index; ++i){
                   if (i==(momlist_index-1)){
                     printf("%d\n", momlistf2[i]);
                   }
                   else{
                     printf("%d, ",momlistf2[i]);
                   }      
                 }
                 printf("  listofgammasf1 = 14,4\n");
                 printf("  listofgammasf2 = 5\n");
                 printf("   group         = %s\n", group[total_momentum_magnitude]);
                 printf("EndTwopointFunction\n\n");


              }

            }
          }
        }
        
      }

    }
 
  }
  free(momlistf2);
  free(momlistf1);
  free(irrep[3][2]);
  free(irrep[3][1]);
  free(irrep[3][0]);
  free(irrep[2][0]);
  free(irrep[1][1]);
  free(irrep[1][0]);
  free(irrep[0][5]);
  free(irrep[0][4]);
  free(irrep[0][3]);  
  free(irrep[0][2]);
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
