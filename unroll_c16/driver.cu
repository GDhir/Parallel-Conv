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
        int remc{ C%16 };

        // for(unsigned int c=0; c<C; c++) { // input feature map
        for (unsigned int r = 0; r<R; r++) { // filter height
            for (unsigned int s = 0; s < S; s++) {// filter width

                for( unsigned int c = 0; c < remc; c++ ) {
                    sumval += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+c*R*S+r*S+s];
                }

                for(unsigned int c = remc; c < C; c += 16) {
            //output_seq[n][k][p][q] += input [n][c][ij+r][ii+s] * weight[k][c][r][s];
                    sumval += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+c*R*S+r*S+s];
                    sumval += d_input[n*C*H*W + (c + 1)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 1)*R*S + r*S + s];
                    sumval += d_input[n*C*H*W + (c + 2)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 2)*R*S+r*S+s];
                    sumval += d_input[n*C*H*W + (c + 3)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 3)*R*S + r*S + s];
                    sumval += d_input[n*C*H*W + (c + 4)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 4)*R*S+r*S+s];
                    sumval += d_input[n*C*H*W + (c + 5)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 5)*R*S + r*S + s];
                    sumval += d_input[n*C*H*W + (c + 6)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 6)*R*S+r*S+s];
                    sumval += d_input[n*C*H*W + (c + 7)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 7)*R*S + r*S + s];
                    sumval += d_input[n*C*H*W + (c + 8)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 8)*R*S + r*S + s];
                    sumval += d_input[n*C*H*W + (c + 9)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 9)*R*S+r*S+s];
                    sumval += d_input[n*C*H*W + (c + 10)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 10)*R*S + r*S + s];
                    sumval += d_input[n*C*H*W + (c + 11)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 11)*R*S+r*S+s];
                    sumval += d_input[n*C*H*W + (c + 12)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 12)*R*S + r*S + s];
                    sumval += d_input[n*C*H*W + (c + 13)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 13)*R*S+r*S+s];
                    sumval += d_input[n*C*H*W + (c + 14)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 14)*R*S + r*S + s];
                    sumval += d_input[n*C*H*W + (c + 15)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 15)*R*S + r*S + s];
                }
            }
        }

        // for (unsigned int r = 0; r<R; r++) { // filter height
        //     for (unsigned int s = 0; s < S; s++) {// filter width

        //         for( unsigned int cb = 0; cb < C; cb += BLOCK_SIZE ) {

        //             for(unsigned int c = cb; c < min( cb + BLOCK_SIZE, C ); c++) {
        //             //output_seq[n][k][p][q] += input [n][c][ij+r][ii+s] * weight[k][c][r][s];
        //                 sumval += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+c*R*S+r*S+s];
        //             }

        //         }
        //     }
        // }

        d_output[idx] = sumval;

        // # if __CUDA_ARCH__>=200
        //     // if( n == 63 )
        //     //     printf("%f \t %d \t %d \t %d \t %d \n", sumval, q, p, k, n);

        //     if( idx == 0 )
        //         printf( "%d \n", blockDim.x*gridDim.x );
        // #endif  
    }

}
