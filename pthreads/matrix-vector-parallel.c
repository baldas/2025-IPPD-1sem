#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

#define SIZE 10000

long int M[SIZE][SIZE];
long int v1[SIZE], v2[SIZE];
long thread_count;

void *mvthread(void *id)
{
  long i, j;
  long tid = (long)id;

  long tbloco = SIZE/thread_count;
  
  long i_inicial = tid*tbloco;
  long i_final = i_inicial+tbloco;

  if (tid == thread_count-1)    // se eu sou a última thread, fico com o 
    i_final = SIZE;             // restante do trabalho

  for (i=i_inicial; i<i_final; i++) {
    v2[i] = 0;
    for(j=0; j<SIZE; j++) {
      v2[i] += M[i][j]*v1[j];
    }
  }

}

int main(int argc, char *argv[])
{
  struct timeval start, stop;
  pthread_t *thread_handles;

  thread_count = strtol(argv[1], NULL, 10);
  thread_handles = malloc(thread_count*sizeof(pthread_t));
  
  // inicializacao
  long i,j;
  for (i=0; i<SIZE; i++) {
    v1[i] = i;
    for (j=0; j<SIZE; j++)
      M[i][j] = i+j;
  }

  gettimeofday(&start, NULL);
  /* FORK Cria as threads informadas na linha de comando */
  for (i=0; i<thread_count; i++) {
    if (pthread_create(&thread_handles[i], NULL, mvthread, (void *)i) != 0) {
      fprintf(stderr, "Nao consegui criar a thread\n");
      exit(-1);
    }
  }

  /* JOIN */
  for (i=0; i<thread_count; i++)
  {
    pthread_join(thread_handles[i], NULL);
  }
  gettimeofday(&stop, NULL);

  free(thread_handles);

  double temp = \
    (((double)(stop.tv_sec)*1000.0 + (double)(stop.tv_usec/1000.0)) - \
    ((double)(start.tv_sec)*1000.0 + (double)(start.tv_usec/1000.0)));

  fprintf(stdout, "Time elapsed = %g ms\n", temp);

  for (i=0; i<SIZE; i++)
    fprintf(stdout, "v[%ld]=%ld\n", i, v2[i]);

	return 0;
}

