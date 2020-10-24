#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h> // CURAND lib header file
#define TRIALS_PER_THREAD 1000000
#define BLOCKS 256
#define THREADS 256
#define PI 3.14159265358979 // known value of pi 

__global__ void setup_kernel(curandState *states)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(tid, 0, 0, &states[tid]);
}

__global__ void pi_mc(unsigned long *estimate, curandState *states) 
{
	unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
	
	unsigned long points_in_circle = 0;
	double x, y;

	curandState localState = states[tid];
	
	for(int i = 0; i < TRIALS_PER_THREAD; i++) 
	{
		x = curand_uniform(&localState);
		y = curand_uniform(&localState);

		points_in_circle += (x*x + y*y <= 1.0f);
	}
	
	states[tid] = localState;
	estimate[tid] = points_in_circle;
} 

int main(int argc, char *argv[]) 
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	  
	unsigned long host[BLOCKS * THREADS];
	unsigned long *dev;
	curandState *devStates;
	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(unsigned long));
	cudaMalloc( (void **)&devStates, BLOCKS*THREADS*sizeof(curandState) );

	unsigned long long points_in_circle = 0;
	unsigned long long total_points = 0;
	
	setup_kernel<<<BLOCKS, THREADS>>>(devStates);
	
	cudaEventRecord(start);
	printf("   time (ms)  |  total points   |  points in 1/4 circle |       estimated pi        |          error        \n");
	printf("------------------------------------------------------------------------------------------------------------\n");

	for (int j=1; j < 10000; j++) 
	{ 
		pi_mc<<<BLOCKS, THREADS>>>(dev, devStates);
		cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(unsigned long), cudaMemcpyDeviceToHost);

		for(int i = 0; i < BLOCKS * THREADS; i++) 
		{
			points_in_circle += host[i];
		}

		total_points += (unsigned long long)BLOCKS * (unsigned long long)THREADS * (unsigned long long)TRIALS_PER_THREAD;
		
		long double pi = 4 * (long double) points_in_circle / (long double)total_points;
		
		long double error = pi - (long double) PI;
		
		cudaEventRecord(stop);
		
		cudaEventSynchronize(stop);
		
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		
		printf("%14.0f\t%16lld\t%16lld\t%20.14lf\t%20.14lf\n", milliseconds, total_points, points_in_circle, pi, error);
	}
	cudaFree(dev);
	cudaFree(devStates);
 return 0;
} 
