#include <iostream>
// #include <cuda.h>
// #include <stdio.h>
//you can change the grid_size
//you can change the block_size
#define BLOCK_SIZE 128
__global__ void cnn(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
               float *d_input, float * d_weight, float * d_output){
    //@@ cnn kernel design

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if( N > 4 ) {

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

    else if( C < 256 ) { 

        if( idx < N*K*P*Q/8 ) {

            int q = idx%Q;
            int p = (idx/Q)%P;
            int k = ( ( 8*(idx/Q/P) )%K );
            int n = ( 8*(idx/Q/P) )/K;

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

            // input feature map
            for (unsigned int r = 0; r<R; r++) { // filter height
                for (unsigned int s = 0; s < S; s++) {// filter width
                    for(unsigned int c=0; c<C; c++) {
                        //output_seq[n][k][p][q] += input [n][c][ij+r][ii+s] * weight[k][c][r][s];
                        sumval1 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[k*C*R*S+c*R*S+r*S+s];
                        sumval2 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 1)*C*R*S+c*R*S+r*S+s];
                        sumval3 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 2)*C*R*S+c*R*S+r*S+s];
                        sumval4 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 3)*C*R*S+c*R*S+r*S+s];
                        sumval5 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 4)*C*R*S+c*R*S+r*S+s];
                        sumval6 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 5)*C*R*S+c*R*S+r*S+s];
                        sumval7 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 6)*C*R*S+c*R*S+r*S+s];
                        sumval8 += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * d_weight[(k + 7)*C*R*S+c*R*S+r*S+s];
                    }
                }
            }

            d_output[n*K*P*Q + k*P*Q + p*Q + q] = sumval1;
            d_output[n*K*P*Q + (k + 1)*P*Q + p*Q + q] = sumval2;
            d_output[n*K*P*Q + (k + 2)*P*Q + p*Q + q] = sumval3;
            d_output[n*K*P*Q + (k + 3)*P*Q + p*Q + q] = sumval4;
            d_output[n*K*P*Q + (k + 4)*P*Q + p*Q + q] = sumval5;
            d_output[n*K*P*Q + (k + 5)*P*Q + p*Q + q] = sumval6;
            d_output[n*K*P*Q + (k + 6)*P*Q + p*Q + q] = sumval7;
            d_output[n*K*P*Q + (k + 7)*P*Q + p*Q + q] = sumval8;

        }
    }
    else {

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

            d_output[idx] = sumval;

        }

    }

}
