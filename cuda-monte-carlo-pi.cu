#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h> // CURAND lib header file
#define TRIALS_PER_THREAD 2048
#define BLOCKS 256
#define THREADS 256
#define PI 3.1415926535 // known value of pi 


__global__ void pi_mc(float *estimate, curandState *states) {
 unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
 int points_in_circle = 0;
 float x, y;
// Initialize CURAND
 curand_init(tid, 0, 0, &states[tid]);
 for(int i = 0; i < TRIALS_PER_THREAD; i++) {
 x = curand_uniform(&states[tid]);
 y = curand_uniform(&states[tid]);
// count if x & y is in the circule.
 points_in_circle += (x*x + y*y <= 1.0f);
 }
 estimate[tid] = 4.0f * points_in_circle / (float) TRIALS_PER_THREAD;
} 

int main(int argc, char *argv[]) {
	 float host[BLOCKS * THREADS];
	 float *dev;
	curandState *devStates;
	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(float));
	cudaMalloc( (void **)&devStates, BLOCKS*THREADS*sizeof(curandState) );
	pi_mc<<<BLOCKS, THREADS>>>(dev, devStates);
	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float),
	cudaMemcpyDeviceToHost);
	 float pi_gpu=0.0;
	 for(int i = 0; i < BLOCKS * THREADS; i++) pi_gpu += host[i];
	pi_gpu /= (BLOCKS * THREADS);
	printf("CUDA estimate of PI = %f [error of %f ]\n",
	pi_gpu, pi_gpu - PI);
	cudaFree(dev);
	cudaFree(devStates);
 return 0;
} 
