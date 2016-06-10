// start here: http://ch.mathworks.com/help/distcomp/run-mex-functions-containing-cuda-code.html

#include "mex.h"
#include "gpu/mxGPUArray.h"

#define RADIUS 2

// device code
void __global__ copy(float const * const I,
                     float * const O,
                     int const N)
{
    int const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        O[i] =  I[i];
    }
}

void __global__ createIntegralY(float const * const I,
                                float * const O,
                                int r,
                                int width,
                                int height)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < width) {
        O[index] = I[index];
        for(int i=1; i<height; i++) {
            O[i*width + index] = O[(i-1)*width + index] + I[i*width + index];
        }
    }
}

void __global__ createIntegralX(float const * const I,
                              float * const O,
                              int r,
                              int width,
                              int height)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < height) {
        O[index*width] = I[index*width];
        for(int i=1; i<width; i++) {
            O[index*width + i] = O[index*width + i -1] + I[index*width + i];
        }
    }
}

void __global__ createBox(float const * const I,
                              float * const O,
                              int r,
                              int width,
                              int height)
{
    int index = (blockDim.y*blockIdx.y + threadIdx.y)*width + blockDim.x*blockIdx.x + threadIdx.x;
    int index_x = index % width;
    int index_y = index / width;

    if (index < height*width) {
        //O[index] = I[min(index_y+RADIUS,height)*width + min(index_x+RADIUS, width)] - I[
    }
}

void __global__ createBoxY(float const * const I,
                              float * const O,
                              int r,
                              int width,
                              int height)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < width) {
        O[index] = I[index];
        for(int i=1; i<height; i++) {
            O[i*width + index] = O[(i-1)*width + index] + I[i*width + index];
        }
    }
}


// host code
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{

    mxGPUArray const *I;
    //mxGPUArray const *r;
    mxGPUArray *O;
    mwSize const *dim;
    float const *d_I;
    float *d_O;
    int input_width;
    int input_height;

    mxInitGPU();

    // check input type is already GPU array, then check it is using doubles
    if (nrhs!=1) 
        mexErrMsgIdAndTxt(
            "parallel:gpu:mexGPUExample:InvalidInput", 
           "Too many input arguments.");
    I = mxGPUCreateFromMxArray(prhs[0]);

    if (mxGPUGetClassID(I) != mxSINGLE_CLASS) // mxDOUBLE_CLASS
        mexErrMsgIdAndTxt(
            "parallel:gpu:mexGPUExample:InvalidInput", 
            "Invalid input to MEX file (underlying data type wrong (single vs. double?).");
    
    d_I = (float const *)(mxGPUGetDataReadOnly(I));

    
    //Define dimensions of input
    dim = mxGPUGetDimensions(I);
    input_width = (int)dim[0];
    input_height = (int)dim[1];

    printf("Input width = %i\n", input_width);
    printf("Input height = %i\n", input_height);
    
    // create GPUArray to hold the result
    O = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(I),
                            dim,
                            mxGPUGetClassID(I),
                            mxGPUGetComplexity(I),
                            MX_GPU_INITIALIZE_VALUES);
    d_O = (float *)(mxGPUGetData(O));

    // kernel call
    int const threadsPerBlock_x = 256;
    int blocksPerGrid_x = (input_width + threadsPerBlock_x - 1) / threadsPerBlock_x;
    //copy<<<blocksPerGrid_x, threadsPerBlock_x>>>(d_I, d_O, input_width*input_height);
    createIntegralY<<<blocksPerGrid_x, threadsPerBlock_x>>>(d_I, d_O, RADIUS, input_width, input_height);

    cudaDeviceSynchronize();

    //second kernel call
    blocksPerGrid_x = (input_height + threadsPerBlock_x - 1) / threadsPerBlock_x;
    createIntegralX<<<blocksPerGrid_x, threadsPerBlock_x>>>(d_O, d_O, RADIUS, input_width, input_height);

    // Wrap the result up as a MATLAB gpuArray for return.
    plhs[0] = mxGPUCreateMxArrayOnGPU(O);

    // destroy the mxGPUArray host-side objects
    mxGPUDestroyGPUArray(I);
    mxGPUDestroyGPUArray(O);
}
