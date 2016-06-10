// start here: http://ch.mathworks.com/help/distcomp/run-mex-functions-containing-cuda-code.html

#include "mex.h"
#include "gpu/mxGPUArray.h"

// device code
void __global__ TimesTwo(float const * const A,
                         float * const B,
                         int const N)
{
    int const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        B[i] = 2.0 * A[i];
    }
}


// host code
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{

    mxGPUArray const *A;
    mxGPUArray *B;
    float const *d_A;
    float *d_B;
    int N;
    mxInitGPU();

    // ccheck input type is already GPU array, then check it is using doubles
    if ((nrhs!=1) || !(mxIsGPUArray(prhs[0]))) 
        mexErrMsgIdAndTxt(
            "parallel:gpu:mexGPUExample:InvalidInput", 
            "Invalid input to MEX file (gpuArray?).");
    A = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetClassID(A) != mxSINGLE_CLASS) // mxDOUBLE_CLASS
        mexErrMsgIdAndTxt(
            "parallel:gpu:mexGPUExample:InvalidInput", 
            "Invalid input to MEX file (underlying data type wrong (single vs. double?).");
    d_A = (float const *)(mxGPUGetDataReadOnly(A));

    // create GPUArray to hold the result
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_B = (float *)(mxGPUGetData(B));

    // kernel call
    N = (int)(mxGPUGetNumberOfElements(A));
    int const threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    TimesTwo<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);

    // Wrap the result up as a MATLAB gpuArray for return.
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);

    // destroy the mxGPUArray host-side objects
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);
}
