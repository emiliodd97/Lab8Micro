#include "kernel.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>
#include <complex>
#include "math.h"
#include <cuComplex.h>
//compute numer of iterations to diverge
__device__ int mandelbrotIterations(const cuDoubleComplex &z0, const int max){
    cuDoubleComplex z = z0;
    for (int t = 0; t < max; t++){

        if( (cuCreal(z)*cuCreal(z) + cuCimag(z)*cuCimag(z) ) > 4.0f){

           return t;

        }
        z = cuCadd(cuCmul(z,z), z0);

    }
    return max;
}
__device__ int mandelbrotSet(const cuDoubleComplex &z0, const int maxIter=500){
   //does it diverge?
    int iterations = mandelbrotIterations(z0, maxIter);
    //avoid division by zero
    if(maxIter - iterations == 0){
        return 0;
    }
    //rescale value to 8 bits (CV_U8)
    return lrint(sqrt(iterations / (float) maxIter) * 255);
}
__global__ void kernel(unsigned char *d_output, int rows, int cols,float x1, float y1, float scaleX, float scaleY){

  // get correspondig coordinates from grid indexes
  int c = blockIdx.x*blockDim.x + threadIdx.x;
  int r = blockIdx.y*blockDim.y + threadIdx.y;
  const int i = r*cols + c;

  // check image bounds
  if( (r>=rows) || (c>=cols) ){
    return;
  }

  //perform operation
  float x0= c/scaleX + x1;
  float y0= r/scaleY +y1;
  cuDoubleComplex z0 = make_cuDoubleComplex(x0, y0);
  uchar value = (uchar) mandelbrotSet(z0);
  d_output[i]= value;
}

void wrapper_gpu(Mat output){
  unsigned char *outputPtr = (unsigned char*) output.data;
  unsigned int cols = output.cols;
  unsigned int rows = output.rows;
  float x1 = -2.1f;
  float x2 =  0.6f;
  float y1 = -1.2f;
  float y2 =  1.2f;
  float scaleX = output.cols / (x2 - x1);
  float scaleY = output.rows / (y2 - y1);

  //block dimensions (threads)
  int Tx = 32;
  int Ty = 32;

  //grid size dimensions (blocks)
  int Bx = (Tx + rows -1)/Tx;
  int By = (Ty + cols -1)/Ty;

  // declare pointers to device memory
  unsigned char *d_in  = 0;
  unsigned char *d_out = 0;

  // allocate memory in device
  cudaMalloc(&d_in, cols*rows*sizeof(unsigned char));
  cudaMalloc(&d_out, cols*rows*sizeof(unsigned char));

  //prepare kernel lauch dimensions
  const dim3 blockSize = dim3(Tx, Ty);
  const dim3 gridSize= dim3(Bx, By);

  // launch kernel in GPU
  kernel<<<gridSize, blockSize>>>(d_out, rows, cols, x1,y1, scaleX, scaleY);

  // copy output from device to host
  cudaMemcpy(outputPtr, d_out, rows*cols*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  // free the memory allocated for device arrays
  cudaFree(d_in);
  cudaFree(d_out);

}