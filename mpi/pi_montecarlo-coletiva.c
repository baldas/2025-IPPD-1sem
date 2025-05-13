#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

int main(int argc, char* argv[]) 
{
  int i, count, n;
  double x,y,z,pi;
  struct timeval start, stop;

  if (argc < 2) {
    printf("Necessário informar um número N.\n");
    exit(-1);
  }
  n = strtoll(argv[1], NULL, 10);

  int        comm_sz;               /* Number of processes    */
  int        my_rank;               /* My process rank        */

  MPI_Init(&argc, &argv);

  /* Get the number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  /* Get my rank among all the processes */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  srand(my_rank);

  count = 0;

  gettimeofday(&start, NULL);
  for (i=0; i < n/comm_sz; ++i) {

    x = (double)rand() / RAND_MAX;
    y = (double)rand() / RAND_MAX;

    z = x * x + y * y;

    if( z <= 1 ) count++;
  }

  // Ateh esse ponto, todos os ranks tem uma copia local do 'count' que
  // produziram
  int count_global = 0;
  
  MPI_Reduce(&count, &count_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);


  // SPMD
  if (my_rank == 0) { // se sou o nodo responsável pelo cálculo ...
  
    pi = ((double) count_global / n) * 4;
  
    gettimeofday(&stop, NULL);
  
    printf(" Nossa estimativa de pi = %.14f\n", pi);
    printf("                     pi = %.14f\n", 4.0*atan(1.0));
   
    double t = (((double)stop.tv_sec)*1000.0 + ((double)(stop.tv_usec)/1000.0)) - \
             (((double)start.tv_sec)*1000.0 + ((double)(start.tv_usec)/1000.0));

    fprintf(stdout, "Tempo gasto: %g ms\n", t);
  }



  MPI_Finalize();


  return(0);
}
