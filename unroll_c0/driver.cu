// #include <iostream>
// #include <cuda.h>
// #include <stdio.h>
//you can change the grid_size
//you can change the block_size
#define BLOCK_SIZE 128
__global__ void cnn(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
               float *d_input, float * d_weight, float * d_output){
    //@@ cnn kernel design

    int idx = threadIdx.x + blockIdx.x*BLOCK_SIZE;

    // long int Nout = N*K*P*Q;

    if( idx < N*K*P*Q ) {
        int q = idx%Q;
        int p = (idx/Q)%P;
        int k = (idx/Q/P)%K;
        int n = idx/Q/P/K;

        int ij = p*u;
        int ii = q*v;

        float sumval = 0;

        // for(unsigned int c=0; c<C; c++) { // input feature map
        for (unsigned int r = 0; r<R; r++) { // filter height
            for (unsigned int s = 0; s < S; s++) {// filter width
                for(unsigned int c=0; c<C; c++) {

            //output_seq[n][k][p][q] += input [n][c][ij+r][ii+s] * weight[k][c][r][s];
                    sumval += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+c*R*S+r*S+s];
                }
            }
        }

        d_output[idx] = sumval;
    }

}
