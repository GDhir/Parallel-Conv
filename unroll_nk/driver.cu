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

    if( idx < N*K*P*Q/16 ) {
        int q = idx%Q;
        int p = (idx/Q)%P;
        int k = ( ( 4*(idx/Q/P) )%K );
        int n = 4*( ( 4*(idx/Q/P) )/K );

        int ij = p*u;
        int ii = q*v;

        float sumval1 = 0;
        float sumval2 = 0;
        float sumval3 = 0;
        float sumval4 = 0;
        float sumval5 = 0;
        float sumval6 = 0;
        float sumval7 = 0;
        float sumval8 = 0;
        float sumval9 = 0;
        float sumval10 = 0;
        float sumval11 = 0;
        float sumval12 = 0;
        float sumval13 = 0;
        float sumval14 = 0;
        float sumval15 = 0;
        float sumval16 = 0;

            for (unsigned int r = 0; r<R; r++) { // filter height
                for (unsigned int s = 0; s < S; s++) {// filter width
                    for(unsigned int c=0; c<C; c++) { 
                    //output_seq[n][k][p][q] += input [n][c][ij+r][ii+s] * weight[k][c][r][s];
                    sumval1 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+c*R*S+r*S+s];
                    sumval2 += d_input[(n)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+c*R*S+r*S+s];
                    sumval3 += d_input[(n)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 2)*C*R*S+c*R*S+r*S+s];
                    sumval4 += d_input[(n)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 3)*C*R*S+c*R*S+r*S+s];
                    sumval5 += d_input[(n + 1)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k)*C*R*S+c*R*S+r*S+s];
                    sumval6 += d_input[(n + 1)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+c*R*S+r*S+s];
                    sumval7 += d_input[(n + 1)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 2)*C*R*S+c*R*S+r*S+s];
                    sumval8 += d_input[(n + 1)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 3)*C*R*S+c*R*S+r*S+s];
                    sumval9 += d_input[(n + 2)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k)*C*R*S+c*R*S+r*S+s];
                    sumval10 += d_input[(n + 2)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+c*R*S+r*S+s];
                    sumval11 += d_input[(n + 2)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 2)*C*R*S+c*R*S+r*S+s];
                    sumval12 += d_input[(n + 2)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 3)*C*R*S+c*R*S+r*S+s];
                    sumval13 += d_input[(n + 3)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k)*C*R*S+c*R*S+r*S+s];
                    sumval14 += d_input[(n + 3)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+c*R*S+r*S+s];
                    sumval15 += d_input[(n + 3)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 2)*C*R*S+c*R*S+r*S+s];
                    sumval16 += d_input[(n + 3)*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 3)*C*R*S+c*R*S+r*S+s];

                }
            }
        }

        d_output[n*K*P*Q + k*P*Q + p*Q + q] = sumval1;
        d_output[(n)*K*P*Q + (k + 1)*P*Q + p*Q + q] = sumval2;
        d_output[(n)*K*P*Q + (k + 2)*P*Q + p*Q + q] = sumval3;
        d_output[(n)*K*P*Q + (k + 3)*P*Q + p*Q + q] = sumval4;
        d_output[(n + 1)*K*P*Q + (k)*P*Q + p*Q + q] = sumval5;
        d_output[(n + 1)*K*P*Q + (k + 1)*P*Q + p*Q + q] = sumval6;
        d_output[(n + 1)*K*P*Q + (k + 2)*P*Q + p*Q + q] = sumval7;
        d_output[(n + 1)*K*P*Q + (k + 3)*P*Q + p*Q + q] = sumval8;
        d_output[(n + 2)*K*P*Q + (k)*P*Q + p*Q + q] = sumval9;
        d_output[(n + 2)*K*P*Q + (k + 1)*P*Q + p*Q + q] = sumval10;
        d_output[(n + 2)*K*P*Q + (k + 2)*P*Q + p*Q + q] = sumval11;
        d_output[(n + 2)*K*P*Q + (k + 3)*P*Q + p*Q + q] = sumval12;
        d_output[(n + 3)*K*P*Q + (k)*P*Q + p*Q + q] = sumval13;
        d_output[(n + 3)*K*P*Q + (k + 1)*P*Q + p*Q + q] = sumval14;
        d_output[(n + 3)*K*P*Q + (k + 2)*P*Q + p*Q + q] = sumval15;
        d_output[(n + 3)*K*P*Q + (k + 3)*P*Q + p*Q + q] = sumval16;

    }

}
