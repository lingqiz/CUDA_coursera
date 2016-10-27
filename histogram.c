// Histogram Equalization

#include    <wb.h>
#define HISTOGRAM_LENGTH 256
#define STRIDE 16

__global__ void equalization(unsigned char* input, float* output, 
                            const float* __restrict__ cdf, float cdfmin, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len)
    {
        output[idx] = cdf[input[idx]];
//        float inter = 255 * (cdf[input[idx]] - cdfmin) / (1 - cdfmin);
//        unsigned char res = min(max(inter, 0.0), 255.0);
//        output[idx] = (float) (res / 255.0);
    }   
}

__global__ void castImage(float* input, unsigned char* output, int len)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len)
        output[idx] = (unsigned char) (255 * input[idx]);
}

__global__ void computeHisto(unsigned char* input, int* histo, int width, int height)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdy = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = threadIdx.x;
    int idy = threadIdx.y;
    int start = blockIdx.y * STRIDE * 3;
    __shared__ unsigned char imageData[STRIDE][STRIDE * 3];
    __shared__ int blockHisto[HISTOGRAM_LENGTH];
    blockHisto[idx * STRIDE + idy] = 0;

    if(globalIdx < height && globalIdy < width)
    {
        int offset = globalIdx * width * 3;
        imageData[idx][idy] =              input[offset + start + idy];
        imageData[idx][idy + STRIDE] =     input[offset + start + idy + STRIDE];
        imageData[idx][idy + STRIDE * 2] = input[offset + start + idy + STRIDE * 2];
        __syncthreads();

        unsigned char r = imageData[idx][idy * 3];
        unsigned char g = imageData[idx][idy * 3 + 1];
        unsigned char b = imageData[idx][idy * 3 + 2];
        
        unsigned char gray = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
        atomicAdd(&(blockHisto[gray]), 1); 
    }
     __syncthreads();
     atomicAdd(&histo[idx * STRIDE + idy], blockHisto[idx * STRIDE + idy]);
}

void cudaCheckError(cudaError_t err)
{
    if(err != cudaSuccess)
        wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float* hostInputImageData;
    float* hostOutputImageData;
    const char* inputImageFile;

    //@@ Insert more code here

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage  = wbImport(inputImageFile);
    imageWidth  = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "Image Height : ", imageHeight);
    wbLog(TRACE, "Image Width : ", imageWidth);

    hostInputImageData  = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
    // Better to decleared in the pinned memory 

    wbTime_start(Generic, "Computing Process");
    int pixelCount = imageHeight * imageWidth * imageChannels;
    float*  deviceInputImageData;
    unsigned char* deviceUnsigned;
    int* histogram;

    cudaCheckError(cudaMalloc((void**) &deviceInputImageData,  pixelCount * sizeof(float)));
    cudaCheckError(cudaMalloc((void**) &deviceUnsigned, pixelCount * sizeof(char)));
    cudaCheckError(cudaMalloc((void**) &histogram, HISTOGRAM_LENGTH * sizeof(int)));
    cudaCheckError(cudaMemset(histogram, 0, HISTOGRAM_LENGTH * sizeof(int)));
    cudaCheckError(cudaMemcpy(deviceInputImageData, hostInputImageData, 
                    pixelCount * sizeof(float), cudaMemcpyHostToDevice));
    
    int blockLength = 512;
    dim3 convBlock(blockLength, 1, 1);
    dim3 convGrid ((pixelCount - 1) / blockLength + 1, 1, 1);
    castImage<<< convGrid, convBlock >>>(deviceInputImageData, deviceUnsigned, pixelCount);
    cudaCheckError(cudaDeviceSynchronize());

    int blockHight = 16;
    int blockWidth = 16;
    dim3 histoBlock(blockHight, blockWidth, 1);
    dim3 histoGrid ((imageHeight - 1) / blockHight + 1, (imageWidth - 1) / blockWidth + 1, 1);
    computeHisto<<< histoGrid, histoBlock >>>(deviceUnsigned, histogram, imageWidth, imageHeight);
    cudaCheckError(cudaDeviceSynchronize());

    int hostHisto[HISTOGRAM_LENGTH];
    cudaCheckError(cudaMemcpy(hostHisto, histogram, sizeof(int) * HISTOGRAM_LENGTH, cudaMemcpyDeviceToHost));

    float hostCDF[HISTOGRAM_LENGTH];
    float imageSize = imageHeight * imageWidth;
    float min = ((float) hostHisto[0]) / imageSize;
    hostCDF[0] = min;

    for(int i = 1; i < HISTOGRAM_LENGTH; i++)
    {
        float elem = hostCDF[i - 1] + ((float) hostHisto[i]) / imageSize;
        hostCDF[i] = elem;
        if(elem < min)
            min = elem;
    }

    float* deviceCDF;    
    cudaCheckError(cudaMalloc((void**) &deviceCDF, sizeof(float) * HISTOGRAM_LENGTH));
    cudaCheckError(cudaMemcpy(deviceCDF, hostCDF, sizeof(float) * HISTOGRAM_LENGTH, cudaMemcpyHostToDevice));
    equalization<<< convGrid, convBlock >>>(deviceUnsigned, deviceInputImageData, deviceCDF, min, pixelCount);
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaMemcpy(hostOutputImageData, deviceInputImageData, sizeof(float) * pixelCount, cudaMemcpyDeviceToHost));
    wbTime_stop(Generic, "Computing Process");
    wbSolution(args, outputImage);

    return 0;
}

