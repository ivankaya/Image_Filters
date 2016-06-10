#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "cuda.h"

/*
 * Device code
 */
void __global__ some_kernel()
{
}

/*
 * Host code
 */
void f_mean(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /*Declare all variables*/
    mxArray const *I;
    mxArray const *r;
    mxGPUArray *Out;
    double *d_I;
    double *d_Out;
    int h_r;

    /*Initialize GPU*/
    mxInitGpu();

    /*Create GPU copy of I*/
    I = mxGPUCreateFromMxArray(prhs[0]);
    d_I = (double *)(mxGPUGetDataReadOnly(I));

    /*Create GPU array for output*/
    Out = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(I),
                            mxGPUGetDimensions(I),
                            mxGPUGetClassID(I),
                            mxGPUGetComplexity(I),
                            MX_GPU_INITIALIZE_VALUES);
    d_Out = (double *)(mxGPUGetData(Out));

    plhs[0] = mxGPUCreateMxArrayOnGPU(Out);

}
