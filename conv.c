#include    <wb.h>
#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width / 2
#define O_TILE_WIDTH  12
#define I_TILE_WIDTH  16

//@@ INSERT CODE HERE

#define CHANNEL 3  //Assume always three channels 
__device__ int getIndex(int colIdx, int rowIdx, int cIdx, int width)
{
    return (rowIdx * width + colIdx) * 3 + cIdx;
}

__global__ void imageConv(float* input, float* output, 
                          const float* __restrict__ mask,
                          int height, int width)
{   
    __shared__ float buffer[I_TILE_WIDTH][I_TILE_WIDTH];

    // Calculate index for input / output tile
    int idx = threadIdx.x;
    int idy = threadIdx.y;
    int idc = blockIdx.z;

    int o_rowIdx = blockIdx.y * O_TILE_WIDTH + idy;
    int o_colIdx = blockIdx.x * O_TILE_WIDTH + idx;

    int i_rowIdx = o_rowIdx - Mask_radius;
    int i_colIdx = o_colIdx - Mask_radius;

    // Load data into shared memory 
    if(i_rowIdx >= 0 && i_rowIdx < height &&
        i_colIdx >= 0 && i_colIdx < width)
    {
        buffer[idy][idx] = input[getIndex(i_colIdx, i_rowIdx, idc, width)];
    }
    else
    {
        buffer[idy][idx] = 0.0f;
    }

    __syncthreads();

    // Perform calculation 
    if(idx < O_TILE_WIDTH && idy < O_TILE_WIDTH &&
        o_colIdx < width && o_rowIdx < height)
    {
        float result = 0.0;

        for(int i = 0; i < Mask_width; i++)
            for(int j = 0; j < Mask_width; j++)
                result += mask[i * Mask_width + j] * buffer[idy + i][idx + j];            

        
        output[getIndex(o_colIdx, o_rowIdx, idc, width)] = result;    
        
    }

}



int main(int argc, char* argv[]) 
{
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    dim3 block(I_TILE_WIDTH, I_TILE_WIDTH, 1);
    dim3 grid ((imageWidth - 1) / O_TILE_WIDTH + 1, (imageHeight - 1) / O_TILE_WIDTH + 1, 3);
    imageConv<<< grid, block >>>(deviceInputImageData, deviceOutputImageData, 
                                deviceMaskData, imageHeight, imageWidth);
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
