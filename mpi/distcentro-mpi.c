#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h> 

//#define QUERO_DEPURAR 1

int main(int argc, char **argv) 
{
  int i, j;
  struct timeval start, stop;
  
  if (argc < 2) {
    printf("Necessário informar o número de pontos.\n");
    exit(-1);
  }
  int npontos = atoi(argv[1]);
  
  int comm_sz;               /* Number of processes    */
  int my_rank;               /* My process rank        */

  MPI_Init(&argc, &argv);

  /* Get the number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  /* Get my rank among all the processes */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // aloca memória para os pontos (x,y) - i-ésimo ponto está na posição i*2 e
  // i*2+1 do vetor
  
  int *pvetor;
  double *dist;

  if (my_rank == 0) { // sou o rank raiz
    pvetor = (int *)malloc(npontos*2*sizeof(int));
  // aloca vetor de saída para as distâncias
    dist = (double *)malloc(npontos*sizeof(double));
  }
  else { // sou um rank nao-raiz
    pvetor = (int *)malloc(npontos*2*sizeof(int)/comm_sz);
  // aloca vetor de saída para as distâncias
    dist = (double *)malloc(npontos*sizeof(double)/comm_sz);
  }

  // gera os pontos de forma aleatória (somente rank 0 executa)
  if (my_rank == 0) {
    srand(0);
    for (i=0; i<npontos; i++)
    {
      pvetor[i*2] = rand() % 2048;
      pvetor[(i*2)+1] = rand() % 2048;
    }
#ifdef QUERO_DEPURAR      
    for (i=0; i<npontos; i++)
      printf("%d - (%d, %d)\n", i, pvetor[i*2], pvetor[i*2+1]);
    
#endif
  }
  
  gettimeofday(&start, NULL);

  /* Esquema da paralelizacao
   * 1. Scatter do pvetor para os ranks do comunicador
   * 2. Cada rank computa a soma local e centroide eh computada por meio de uma reducao (AllReduction)
   * 3. Barreira para sincronizar os ranks (opcional, jah que Reduction jah vai sincronizar a execucao)
   * 4. Cada rank calcula a distancia de seus pontos para a centroide, alterando o vetor 'dist'
   * 5. Uma operacao GATHER final eh feita para gerar o vetor da distancia final no rank raiz
   */

// 1. Scatter do pvetor para os ranks do comunicador
  MPI_Scatter(pvetor, /* vetor a ser distribuido */ 
              npontos*2/comm_sz, /* numero de elementos a distribuir */
              MPI_INT, 
              pvetor, /* vetor destino que armazena vetor distribuido */
              npontos*2/comm_sz,
              MPI_INT,
              0,  /* rank raiz */  
              MPI_COMM_WORLD);

#ifdef QUERO_DEPURAR  
  // codigo de depuracao
  for (i=0; i<npontos/comm_sz; i++)
    printf("rank %d - %d = (%d, %d)\n", my_rank, i, 
                                       pvetor[i*2], pvetor[i*2+1]);
#endif

  // calcula a centroide
  double x = 0, y = 0;
  for (i = 0; i < npontos/comm_sz; i++) {
    x += pvetor[i*2];
    y += pvetor[(i*2)+1];
  }
  
  double centroid_x;
  double centroid_y;
// 2. Cada rank computa a soma local e centroide eh computada por meio de uma reducao (AllReduction)
  MPI_Allreduce(&x, &centroid_x, 1, MPI_DOUBLE, MPI_SUM,
               MPI_COMM_WORLD);
  
  MPI_Allreduce(&y, &centroid_y, 1, MPI_DOUBLE, MPI_SUM,
               MPI_COMM_WORLD);

  centroid_x = centroid_x / npontos;
  centroid_y = centroid_y / npontos;

#ifdef QUERO_DEPURAR  
  printf("x %g - centroid x = %g\n", x, centroid_x);
  printf("y %g - centroid y = %g\n", y, centroid_y);
#endif

// 4. Cada rank calcula a distancia de seus pontos para a centroide, alterando o vetor 'dist'

  // calcula as distâncias euclidianas de cada ponto para a centróide
  for (i = 0; i < npontos/comm_sz; i++) {
    double a = pvetor[i*2], b = pvetor[(i*2)+1];
    dist[i] = sqrt((centroid_x - a) * (centroid_x - a)  +  (centroid_y - b) * (centroid_y - b));
  }
   
// 5. Uma operacao GATHER final eh feita para gerar o vetor da distancia final no rank raiz

  MPI_Gather(dist, npontos/comm_sz, MPI_DOUBLE,
             dist, npontos/comm_sz, MPI_DOUBLE,
             0, MPI_COMM_WORLD);
  
  gettimeofday(&stop, NULL); 
 
  if (my_rank == 0) {
    double t = (((double)(stop.tv_sec)*1000.0  + (double)(stop.tv_usec / 1000.0)) - \
                   ((double)(start.tv_sec)*1000.0 + (double)(start.tv_usec / 1000.0)));
    fprintf(stdout, "Tempo decorrido = %g ms\n", t);

    printf("Vetor com as distâncias para a centroide (%g,%g):\n", centroid_x, centroid_y);
  
  // imprime o vetor com as distâncias
    for (i=0; i < npontos; i++) {
      printf("%.3f \n", dist[i]);
    }
    printf("\n");
  }

  MPI_Finalize();
 
  return 0;
}
