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

    if( idx < N*K*P*Q/4 ) {
        int q = idx%Q;
        int p = (idx/Q)%P;
        int k = ( ( 4*(idx/Q/P) )%K );
        int n = ( 4*(idx/Q/P) )/K;

        int ij = p*u;
        int ii = q*v;

        float sumval1 = 0;
        float sumval2 = 0;
        float sumval3 = 0;
        float sumval4 = 0;

         // input feature map
        for(unsigned int c=0; c<C; c++) {
            for (unsigned int r = 0; r<R; r++) { // filter height
                for (unsigned int s = 0; s < S; s++) {// filter width
                
                    //output_seq[n][k][p][q] += input [n][c][ij+r][ii+s] * weight[k][c][r][s];
                    sumval1 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+c*R*S+r*S+s];
                    sumval2 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+c*R*S+r*S+s];
                    sumval3 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 2)*C*R*S+c*R*S+r*S+s];
                    sumval4 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 3)*C*R*S+c*R*S+r*S+s];

                }
            }
        }

        d_output[n*K*P*Q + k*P*Q + p*Q + q] = sumval1;
        d_output[n*K*P*Q + (k + 1)*P*Q + p*Q + q] = sumval2;
        d_output[n*K*P*Q + (k + 2)*P*Q + p*Q + q] = sumval3;
        d_output[n*K*P*Q + (k + 3)*P*Q + p*Q + q] = sumval4;

    }

}
