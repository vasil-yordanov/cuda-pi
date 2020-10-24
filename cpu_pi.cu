#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h> 
#define TRIALS_PER_THREAD 1000
#define BLOCKS 256
#define THREADS 256
#define PI 3.14159265358979 // known value of pi 

long pi_mc(unsigned long long trials) {
  double x, y;
  long points_in_circle=0;
  for(long i = 0; i < trials; i++) {
    x = rand() / (double) RAND_MAX;
    y = rand() / (double) RAND_MAX;
    points_in_circle += (x*x + y*y <= 1.0f);
  }
  
  return points_in_circle;
}

int main (int argc, char *argv[]) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	  
	unsigned long long points_in_circle = 0;
	unsigned long long total_points = 0;
		
	cudaEventRecord(start);
	printf("   time (ms)  |  total points   |  points in 1/4 circle |       estimated pi        |          error        \n");
	printf("------------------------------------------------------------------------------------------------------------\n");

	for (int j=1; j < 10000; j++) 
	{ 
		points_in_circle += pi_mc((unsigned long long)BLOCKS * (unsigned long long)THREADS * (unsigned long long)TRIALS_PER_THREAD);

		total_points += (unsigned long long)BLOCKS * (unsigned long long)THREADS * (unsigned long long)TRIALS_PER_THREAD;
		
		long double pi = 4 * (long double) points_in_circle / (long double)total_points;
		
		long double error = pi - (long double) PI;
		
		cudaEventRecord(stop);
		
		cudaEventSynchronize(stop);
		
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		
		printf("%14.0f\t%16lld\t%16lld\t%20.14lf\t%20.14lf\n", milliseconds, total_points, points_in_circle, pi, error);
	}

  return 0;
}
