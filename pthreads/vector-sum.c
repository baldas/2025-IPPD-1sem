#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>


double *vetor;
long vsize;
double sum = 0.0;
long thread_count;
pthread_mutex_t mutex;

void *vsthread(void *id)
{
  long i, j;
  long tid = (long)id;

  long tbloco = vsize/thread_count;
  
  long i_inicial = tid*tbloco;
  long i_final = i_inicial+tbloco;

  if (tid == thread_count-1)    // se eu sou a última thread, fico com o 
    i_final = vsize;            // restante do trabalho

  double local_sum = 0.0;
  for (i=i_inicial; i<i_final; i++) {
    local_sum += vetor[i];
  }

  pthread_mutex_lock(&mutex);
  sum += local_sum;
  pthread_mutex_unlock(&mutex);

}

int main(int argc, char *argv[])
{
  struct timeval start, stop;
  pthread_t *thread_handles;
  
  vsize = strtol(argv[1], NULL, 10);
  thread_count = strtol(argv[2], NULL, 10);

  thread_handles = malloc(thread_count*sizeof(pthread_t));
  vetor = (double *)malloc(vsize*sizeof(double));

  pthread_mutex_init(&mutex, NULL);

  long i;
  for (i=0; i<vsize; i++) 
    vetor[i] = (double)i;


  gettimeofday(&start, NULL);
  /* FORK Cria as threads informadas na linha de comando */
  for (i=0; i<thread_count; i++) {
    if (pthread_create(&thread_handles[i], NULL, vsthread, (void *)i) != 0) {
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


  double temp = \
    (((double)(stop.tv_sec)*1000.0 + (double)(stop.tv_usec/1000.0)) - \
    ((double)(start.tv_sec)*1000.0 + (double)(start.tv_usec/1000.0)));

  fprintf(stdout, "Time elapsed = %g ms\n", temp);
  fprintf(stdout, "Sum = %g\n", sum);
  
  pthread_mutex_destroy(&mutex);
  free(thread_handles);
  free(vetor);

	return 0;
}

