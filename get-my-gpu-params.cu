#include <stdio.h>
#include <stdlib.h>

#include <curand_kernel.h> 

// Device code
__global__ void MyKernel()
{
 //int idx = threadIdx.x + blockIdx.x * blockDim.x;
}
// Host code
int main()
{

	int blockSize; // The launch configurator returned block size
	int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)MyKernel, 0);
				
	printf("blockSize= %d\n", blockSize);
	printf("minGridSize= %d\n", minGridSize);

	int numBlocks; // Occupancy in terms of active blocks
	// These variables are used to convert occupancy to warps
	int device;
	cudaDeviceProp prop;
	int activeWarps;
	int maxWarps;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, MyKernel, blockSize, 0);
	
	activeWarps = numBlocks * blockSize / prop.warpSize;
	maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
	printf("numBlocks: %d\n", numBlocks);
	printf("warpSize: %d\n", prop.warpSize);
	printf("maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
	printf("activeWarps: %d\n", activeWarps);
	printf("maxWarps: %d\n", maxWarps);

	printf("Occupancy: %f %\n", (double)activeWarps / maxWarps * 100);

 return 0;
}
