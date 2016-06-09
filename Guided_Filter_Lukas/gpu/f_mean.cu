#include "mex.h"
#include "gpu/mxGPUArray.h"

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
    mxArray *I;
    mxArray *r;

    mxInitGpu();
}
