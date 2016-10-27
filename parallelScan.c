// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>
#define BLOCK_SIZE 1024 //@@ You can change this
#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
    
void cudaCheck(cudaError_t err)
{
    if (err != cudaSuccess)                                           
        wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    
}

__global__ void parallelScan(float* input, float* output, int len, float* blockResult) 
{
    int idx = threadIdx.x;
    int globalIdx = blockIdx.x * (BLOCK_SIZE << 1) + idx;

    // Load input data into memory
    __shared__ float list[BLOCK_SIZE * 2];
    if(globalIdx < len)
        list[idx] = input[globalIdx];
    if(globalIdx + BLOCK_SIZE < len)
        list[idx + BLOCK_SIZE] = input[globalIdx + BLOCK_SIZE];

    // Reduction phase 
    for(int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
    {
        __syncthreads();
        int dataIdx = (idx + 1) * stride * 2 - 1;
        if(dataIdx < BLOCK_SIZE * 2)
            list[dataIdx] += list[dataIdx - stride];
    }

    // Reconstruction phase 
    for(int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        int dataIdx = (idx + 1) * stride * 2 - 1;
        if(dataIdx + stride < BLOCK_SIZE * 2)
            list[dataIdx + stride] += list[dataIdx];
    }

    __syncthreads();

    if(globalIdx < len)
        output[globalIdx] = list[idx];
    if(globalIdx + BLOCK_SIZE < len)
        output[globalIdx + BLOCK_SIZE] = list[idx + BLOCK_SIZE];
    if(idx == 0)
       blockResult[blockIdx.x] = list[BLOCK_SIZE * 2 - 1];
}

__global__ void addBack(float* input, float* reduced, int len)
{

    int idx  = threadIdx.x;
    int bIdx = blockIdx.x;
    int globalIdx = bIdx * BLOCK_SIZE * 2 + idx;

    if(bIdx > 0)
    {
        float sumed = reduced[bIdx - 1];

        if(globalIdx < len)
            input[globalIdx] += sumed;
        if(globalIdx + BLOCK_SIZE < len)
            input[globalIdx + BLOCK_SIZE] += sumed;
    }
    
}

float* gpuScan(float* hostInput, int length)
{
    // Allocating Memory Space 
    int    inputSize   = length * sizeof(float);
    float* hostOutput  = (float*) malloc(inputSize);
    float* deviceInput;
    float* deviceOutput;

    cudaMalloc((void**) &deviceInput,  inputSize);
    cudaMalloc((void**) &deviceOutput, inputSize);

    // Copy Input To Device Memory 
    cudaMemcpy(deviceInput, hostInput, inputSize, cudaMemcpyHostToDevice);

    // Set launch parameters
    int blockNum = (length - 1) / (BLOCK_SIZE << 1) + 1;
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid (blockNum, 1, 1);

    float* deviceBlockResult;
    cudaMalloc((void**) &deviceBlockResult, blockNum * sizeof(float));

    parallelScan<<< grid, block >>>(deviceInput, deviceOutput, length, deviceBlockResult);

    cudaCheck(cudaDeviceSynchronize());
    cudaFree(deviceInput);    

    if(blockNum == 1)
    {
        cudaMemcpy(hostOutput, deviceOutput, inputSize, cudaMemcpyDeviceToHost);
        cudaFree(deviceOutput);
        cudaFree(deviceBlockResult);

        return hostOutput;
    }

    else
    {
        float* hostBlockResult = (float*) malloc(blockNum * sizeof(float));
        cudaMemcpy(hostBlockResult, deviceBlockResult, blockNum * sizeof(float), cudaMemcpyDeviceToHost);
        float* reduced = gpuScan(hostBlockResult, blockNum);

        cudaMemcpy(deviceBlockResult, reduced, blockNum * sizeof(float), cudaMemcpyHostToDevice);
        free(hostBlockResult);
        free(reduced);

        addBack<<< grid, block >>>(deviceOutput, deviceBlockResult, length);
        cudaCheck(cudaDeviceSynchronize());

        cudaFree(deviceBlockResult);
        cudaMemcpy(hostOutput, deviceOutput, inputSize, cudaMemcpyDeviceToHost);
        cudaFree(deviceOutput);

        return hostOutput;

    }


}

int main(int argc, char ** argv) 
{
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    free(hostOutput);
    //@@ Initialize the grid and block dimensions here

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    hostOutput = gpuScan(hostInput, numElements);
    wbTime_stop(Compute, "Performing CUDA computation");

    // wbTime_start(Copy, "Copying output memory to the CPU");
    // wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    // wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

