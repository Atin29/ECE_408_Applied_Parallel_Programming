#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 8
#define K 7
#include <mxnet/base.h>
#include <math.h>
namespace mxnet
{
namespace op
{

//__constant__ float CONSTANT_MEMORY_KERNEL[24*12*7*7];
__global__ void forward_kernel(float * y, const float *  __restrict__ x, const int B, const int M, const int C, const int H, const int W)
{
const int H_out = H - K + 1;
const int W_out = W - K + 1;
const int W_grid = ceil(W_out/(1.0*TILE_WIDTH));
extern __shared__ float shmem[];

float * X_shared = &shmem[0];
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define constantMem4d(i3, i2, i1, i0) CONSTANT_MEMORY_KERNEL[(i3)*(C*K*K)+(i2)*(K*K)+(i1)*(K)+i0]
//extern __shared__ float shmem[];
//float * X_shared = &shmem[0];
int h_base = (blockIdx.z/W_grid)*TILE_WIDTH;
int w_base = (blockIdx.z%W_grid)*TILE_WIDTH;
int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;
int h = h_base+ty;
int w = w_base+tx;
int b, c, i, j, p, q;
int X_tile_width = TILE_WIDTH+K-1;
int miniBIdx = blockIdx.x * B;
//#pragma unroll
float acc = 0.0;
for(b = 0; b < B; b++){
  acc = 0.0;
  //__syncthreads();
  for(c = 0; c < C; c++){
    for(i = h; i<h_base + X_tile_width; i+=TILE_WIDTH){
      for(j = w; j<w_base+X_tile_width;j+=TILE_WIDTH){
        if(i<H && j<W )
          X_shared[(i-h_base)*X_tile_width+(j-w_base)]=x4d(miniBIdx+b, c, i, j);
        else
          X_shared[(i-h_base)*X_tile_width+(j-w_base)] = 0.0;
      }
    }
    __syncthreads();
    for( p = 0; p < K; p++){
      for(q = 0; q < K; q++){
          if((h+p)<H && (w+q)<W)
              acc += X_shared[X_tile_width*(ty+p)+(tx+q)]*constantMem4d(by, c, p, q);
          }
      }
      __syncthreads();
  }
  if(h<H_out && w<W_out)
    y4d(miniBIdx+b, by, h, w) = acc;
}

#undef y4d
#undef x4d
//#undef k4d
#undef constantMem4d

}

//A is W, B is X, Y is C
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns,
                                     int B, int C, int H, int W) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float M[TILE_WIDTH][TILE_WIDTH];
  __shared__ float N[TILE_WIDTH][TILE_WIDTH];
  int miniBIdx = blockIdx.z * B;
  int H_out = H - K + 1;
  int W_out = W - K + 1;
  int W_unroll = H_out * W_out;
  int H_unroll = C*K*K;
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  int bIdxActual;
  for(b = 0; b < B; b++){
    bIdxActual = miniBIdx+b;
    float Cvalue = 0;
    for (int ph = 0; ph < (float)numAColumns/(float)TILE_WIDTH; ++ph) {
      if ((Row< numARows) && (ph*TILE_WIDTH+tx)< numAColumns)
        M[ty][tx] = A[Row*numAColumns + ph*TILE_WIDTH + tx];
      else
        M[ty][tx] = 0;
      if ((ph*TILE_WIDTH+ty)<numBRows && Col<numBColumns)
        N[ty][tx] = B[bIdxActual*H_unroll*W_unroll + (ph*TILE_WIDTH + ty)*numBColumns + Col];
      else
        N[ty][tx] = 0;
      __syncthreads();
      for (int k = 0; k < TILE_WIDTH; ++k) {
        Cvalue += M[ty][k] * N[k][tx];
      }
      __syncthreads();
    }
    if (Row<numCRows && Col<numCColumns)
      C[bIdxActual*numCColumns*numCRows+Row*numCColumns + Col] = Cvalue;
  }
}

// Change this
__global__ void unroll_Kernel(int C, int H, int W, int K, int B, const float* __restrict__ x, float* X_unroll)
{
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
int c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q, b;
int t = blockIdx.x * CUDA_MAX_NUM_THREADS + threadIdx.x;
int H_out = H - K + 1;
int W_out = W - K + 1;
int W_unroll = H_out * W_out;
int H_unroll = C*K*K;
int miniBIdx = blockIdx.y * B;

if (t < C * W_unroll){
  c = t / W_unroll;
  s = t % W_unroll;
  h_out = s / W_out;
  w_out = s % W_out;
  w_unroll = h_out * W_out + w_out;
  w_base = c * K * K;
  for(b = 0; b < B; b++){
    for(p = 0; p < K; p++){
        for(q = 0; q < K; q++){
          h_unroll = w_base + p *K + q;
          X_unroll[(miniBIdx+b)*W_unroll*H_unroll+h_unroll * W_unroll + w_unroll] = x4d(miniBIdx+b, c, h_out + p, w_out +q);
        }
      }
    }
}
#undef x4d
}

template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    //const int K = w.shape_[3];
    int H_Out = H-K+1;
    int W_Out = W-K+1;
    int W_grid = ceil(W_Out / (1.0*TILE_WIDTH));
    int H_grid = ceil(H_Out / (1.0*TILE_WIDTH));
    int Z = H_grid * W_grid;
    int MINI_BATCH_SIZE;
    // Set the kernel dimensions
    if(M == 12)
      MINI_BATCH_SIZE = 10;
    else
      MINI_BATCH_SIZE = 1000;
    int miniBatch = B/MINI_BATCH_SIZE;
    float *X_unrolled;
    int W_unroll = H_out * W_out;
    int H_unroll = C*K*K;
    cudaMalloc((void**) &X_unrolled, H_unroll * W_unroll * B * sizeof(float));

    dim3 unrollGridDim(ceil((1.0*C*H_out*W_out)/CUDA_MAX_NUM_THREADS), miniBatch, 1);
    dim3 unrollBlockDim(CUDA_MAX_NUM_THREADS,1,1);
    unroll_Kernel<<<unrollGridDim, unrollBlockDim>>>(C, H, W, K, miniBatch, x.dptr_, X_unrolled);
    dim3 dimGrid(ceil((float)W_unroll/(float)(TILE_WIDTH)), ceil((float)H_unroll/(float)(TILE_WIDTH)), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, MINI_BATCH_SIZE);
    matrixMultiplyShared<<<dimGrid, dimBlock>>>(w.dptr_, X_unrolled, y.dptr_, M,H_unroll, H_unroll, W_unroll,M,W_unroll, miniBatch, C, H, W);
/*
    size_t shmem_size = sizeof(float) * ((TILE_WIDTH+K-1)*(TILE_WIDTH+K-1));

    cudaMemcpyToSymbol(CONSTANT_MEMORY_KERNEL, w.dptr_, (sizeof(float) * M * C * K * K), (size_t) 0, cudaMemcpyHostToDevice);

    forward_kernel<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_, miniBatch,M,C,H,W);
*/
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
  assert(0);
}
}
}

#endif
