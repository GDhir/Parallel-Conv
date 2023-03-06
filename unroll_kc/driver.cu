// #include <iostream>
// #include <cuda.h>
// #include <stdio.h>
//you can change the grid_size
//you can change the block_size
#define BLOCK_SIZE 1024
__global__ void cnn(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
               float *d_input, float * d_weight, float * d_output){
    //@@ cnn kernel design

    int idx = threadIdx.x + blockIdx.x*BLOCK_SIZE;

    // long int Nout = N*K*P*Q;

    if( idx < N*K*P*Q/2 ) {
        int q = idx%Q;
        int p = (idx/Q)%P;
        int k = ( ( 2*(idx/Q/P) )%K );
        int n = ( 2*(idx/Q/P) )/K;

        int ij = p*u;
        int ii = q*v;

        float sumval1 = 0;
        float sumval2 = 0;
        int remc{ C%8 };

        // for(unsigned int c=0; c<C; c++) { // input feature map
        for (unsigned int r = 0; r<R; r++) { // filter height
            for (unsigned int s = 0; s < S; s++) {// filter width

                for( unsigned int c = 0; c < remc; c++ ) {
                    sumval1 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+c*R*S+r*S+s];
                    sumval2 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+c*R*S+r*S+s];
                }

                for(unsigned int c = remc; c < C; c += 8) {
            //output_seq[n][k][p][q] += input [n][c][ij+r][ii+s] * weight[k][c][r][s];
                    sumval1 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+c*R*S+r*S+s];
                    sumval1 += d_input[n*C*H*W + (c + 1)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 1)*R*S + r*S + s];
                    sumval1 += d_input[n*C*H*W + (c + 2)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 2)*R*S+r*S+s];
                    sumval1 += d_input[n*C*H*W + (c + 3)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 3)*R*S + r*S + s];
                    sumval1 += d_input[n*C*H*W + (c + 4)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 4)*R*S+r*S+s];
                    sumval1 += d_input[n*C*H*W + (c + 5)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 5)*R*S + r*S + s];
                    sumval1 += d_input[n*C*H*W + (c + 6)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 6)*R*S+r*S+s];
                    sumval1 += d_input[n*C*H*W + (c + 7)*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+ (c + 7)*R*S + r*S + s];

                    sumval2 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+c*R*S+r*S+s];
                    sumval2 += d_input[n*C*H*W + (c + 1)*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+ (c + 1)*R*S + r*S + s];
                    sumval2 += d_input[n*C*H*W + (c + 2)*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+ (c + 2)*R*S+r*S+s];
                    sumval2 += d_input[n*C*H*W + (c + 3)*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+ (c + 3)*R*S + r*S + s];
                    sumval2 += d_input[n*C*H*W + (c + 4)*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+ (c + 4)*R*S+r*S+s];
                    sumval2 += d_input[n*C*H*W + (c + 5)*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+ (c + 5)*R*S + r*S + s];
                    sumval2 += d_input[n*C*H*W + (c + 6)*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+ (c + 6)*R*S+r*S+s];
                    sumval2 += d_input[n*C*H*W + (c + 7)*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+ (c + 7)*R*S + r*S + s];
                }
            }
        }


        d_output[n*K*P*Q + k*P*Q + p*Q + q] = sumval1;
        d_output[n*K*P*Q + (k + 1)*P*Q + p*Q + q] = sumval2;

    }

}
