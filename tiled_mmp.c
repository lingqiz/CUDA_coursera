// Matrix Multiplication kernel, Tiling Version 
// Tiling is a very general technique :
    /*
      If we have a metrix which could not fit into global memory ->
      Write Tiled kernel (outer loop (tiled control loop) in the host, l
      aunch several kernels, copy result back into memory, accumulate. 
    */

#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// Compute C = A * B
#define TILE_WIDTH 16

__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) 
{
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
	
	
	int tdx = threadIdx.x;
	int tdy = threadIdx.y;
	int row = tdy + blockIdx.y * blockDim.y;
	int col = tdx + blockIdx.x * blockDim.x;
	
	int n = numAColumns;
	float acc = 0;
	
	for(int phase = 0; phase < (n - 1) / TILE_WIDTH + 1; phase++)
	{
    // Loading data into shared memory 
		if(row < numARows && (phase * TILE_WIDTH + tdx) < n)  
			ds_A[tdy][tdx] = A[row * n + phase * TILE_WIDTH + tdx];
		else 
			ds_A[tdy][tdx] = 0.0;
		
		if((phase * TILE_WIDTH + tdy) < n && col < numBColumns)
			ds_B[tdy][tdx] = 
			B[(phase * TILE_WIDTH + tdy) * numBColumns + col];
		else 
			ds_B[tdy][tdx] = 0.0;
		
		__syncthreads();
		
    // Perform calculation on data 
		for(int i = 0; i < TILE_WIDTH; i++)
			acc = acc + ds_A[tdy][i] * ds_B[i][tdx];
		
		__syncthreads();
	}
	
	if(row < numCRows && col < numCColumns)
		C[row * numCColumns + col] = acc;

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)
	
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
	
  numCRows = numARows;
  numCColumns = numBColumns;
	
  int Asize = sizeof(float) * numARows * numAColumns;
  int Bsize = sizeof(float) * numBRows * numBColumns;
  int Csize = sizeof(float) * numCRows * numCColumns;
	
  //@@ Allocate the hostC matrix
  hostC = (float*) malloc(Csize);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**) &deviceA, Asize);
  cudaMalloc((void**) &deviceB, Bsize);
  cudaMalloc((void**) &deviceC, Csize);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, Asize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, Bsize, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 grid((numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
	
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<grid, block>>>(deviceA, deviceB, deviceC, numARows, numAColumns,
										numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, Csize, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
