#include	<wb.h>
#define segSize   512 
#define blockSize 128
#define streamCount 4

__global__ void vecAdd(float * in1, float * in2, float * out, int len) 
{
    int gloablIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if(gloablIdx < len)
    	out[gloablIdx] = in1[gloablIdx] + in2[gloablIdx];
}

void cudaErrorCheck(cudaError_t err)
{
	if(err != cudaSuccess)
	{
		wbLog(ERROR, "CUDA Check Fail");
		wbLog(ERROR, cudaGetErrorString(err));
	}
}

int main(int argc, char ** argv) 
{
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    
    float* deviceInput1;
    float* deviceInput2;
    float* deviceOutput;
    cudaStream_t streams[streamCount];

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "Data Size : ", inputLength);
    cudaErrorCheck(cudaMalloc((void**) &deviceInput1, sizeof(float) * segSize * streamCount));
    cudaErrorCheck(cudaMalloc((void**) &deviceInput2, sizeof(float) * segSize * streamCount));
    cudaErrorCheck(cudaMalloc((void**) &deviceOutput, sizeof(float) * segSize * streamCount));

    wbTime_start(Generic, "Computing process.");
    for(int i = 0; i < streamCount; i++)
    	{ cudaErrorCheck(cudaStreamCreate(&streams[i])); }

    int streamId = -1;
    int tempLength;
    for(int pos = 0; pos < inputLength; pos += segSize)
    {
    	streamId = (streamId + 1) % streamCount;
    	tempLength = inputLength - pos >= segSize? segSize : inputLength - pos;

    	float* in1 = deviceInput1 + streamId * segSize;
    	float* in2 = deviceInput2 + streamId * segSize;
    	float* out = deviceOutput + streamId * segSize;

    	cudaErrorCheck(cudaMemcpyAsync(in1, hostInput1 + pos, 
    		sizeof(float) * tempLength, cudaMemcpyHostToDevice, streams[streamId]));

    	cudaErrorCheck(cudaMemcpyAsync(in2, hostInput2 + pos, 
    		sizeof(float) * tempLength, cudaMemcpyHostToDevice, streams[streamId]));

    	dim3 grid ((tempLength - 1) / blockSize + 1, 1, 1);
    	dim3 block(blockSize, 1, 1);
    	vecAdd<<< grid, block, 0, streams[streamId] >>>(in1, in2, out, tempLength);

    	cudaErrorCheck(cudaMemcpyAsync(hostOutput + pos, out, 
    		sizeof(float) * tempLength, cudaMemcpyDeviceToHost, streams[streamId]));
	    
    }

    cudaErrorCheck(cudaDeviceSynchronize());
    wbTime_stop(Generic, "Computing process.");
    
    wbSolution(args, hostOutput, inputLength);

    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

