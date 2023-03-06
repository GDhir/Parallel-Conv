#include <cuda_runtime.h>
#include <iostream>
#include "driver.cu"
#include <iomanip>

__global__ void cnn(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
     float *d_input, float * d_weight, float * d_output);


inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cerr << "ERROR!!!:" << cudaGetErrorString(code) <<std::endl;
        exit(-1);
    }
}


int main(int argc, char *argv[]) {
    // READ PROBLEM SIZES
    if (argc != 10) exit(1);
    int N = atoi(argv[1]);
    int C = atoi(argv[2]);
    int K = atoi(argv[3]);
    int H = atoi(argv[4]);
    int W = atoi(argv[5]);
    int R = atoi(argv[6]);
    int S = atoi(argv[7]);
    int u = atoi(argv[8]);
    int v = atoi(argv[9]);
    int P = (H-R)/u + 1;
    int Q = (W-S)/v + 1;

    float *output_seq = new float[N*K*P*Q];
    memset(output_seq,0, N * K * P * Q*sizeof(float));
    float *output_par = new float[N*K*P*Q];
    memset(output_par,0, N * K * P * Q*sizeof(float));
    float *input = new float[N*C*H*W];
    float *weight = new float[K*C*R*S];
    // ASSIGN INITIAL VALUES FOR INPUT AND WEIGHT

    for(unsigned int n=0; n<N; ++n){
        for(unsigned int c=0; c<C; ++c){
            for(unsigned int h=0; h<H; ++h){
                for(unsigned int w=0; w<W; ++w){
                    input[n*C*H*W + c*H*W + h*W + w] =  ((float)(n+c+h+w));
                }
            }
        }
    }
    for (unsigned int k=0; k<K; k++) {
        for (unsigned int c=0; c<C; c++) {
            for (unsigned int r =0; r<R; r++) {
                for (unsigned int s =0; s<S; s++) {
                    //weight[k][c][r][s] = ((float) (k+c+r+s));
                    weight[k*C*R*S + c*R*S + r*S + s] = ((float) (k+c+r+s));
                }
            }
        }
    }
    // TIME SEQUENTIAL CALCULATION
    cudaEvent_t seq_start,seq_stop;
    float seq_time;
    cudaEventCreate(&seq_start);
    cudaEventCreate(&seq_stop);
    cudaEventRecord(seq_start);

    for(unsigned int n=0; n<N; n++) { // minibatch size
        for(unsigned int k=0; k<K; k++) { // output feature map
            for(unsigned int c=0; c<C; c++) { // input feature map
                for(unsigned int p=0; p<P; p++) { // output height
                    unsigned int ij = p * u; // input height
                    for (unsigned int q = 0; q<Q; q ++) { // output width
                        unsigned int ii = q * v; // input width
                        for (unsigned int r = 0; r<R; r ++) { // filter height
                            for (unsigned int s = 0; s < S; s ++) {// filter width
                                //output_seq[n][k][p][q] += input [n][c][ij+r][ii+s] * weight[k][c][r][s];
                                output_seq[n*K*P*Q + k*P*Q + p*Q + q] += input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * weight[k*C*R*S+c*R*S+r*S+s];
                            }
                        }
                    }
                }
            }
        }
    }

    cudaEventRecord(seq_stop);
    cudaEventSynchronize(seq_stop);
    cudaEventElapsedTime(&seq_time,seq_start, seq_stop);
    //@@ Copy input, weight and output data, input as example

    float * d_input, *d_weight, * d_output;
    chkerr(cudaMalloc((void **) &d_input,  sizeof(float) * N * C * H * W));
    chkerr(cudaMalloc((void **) &d_weight,  sizeof(float) * K * C* R * S));
    chkerr(cudaMalloc((void **) &d_output,  sizeof(float) * N * K * P * Q));

    chkerr(cudaMemcpy(d_input, input, sizeof(float) * N * C * H * W, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(d_weight, weight, sizeof(float) * K * C * R * S, cudaMemcpyHostToDevice));
    // chkerr(cudaMemcpy(d_output, output_seq, sizeof(float) * N * K * P * Q, cudaMemcpyHostToDevice));

    // INITIALIZE PARALLEL TIMER
    cudaEvent_t par_start,par_stop;
    float par_time;

    //@@ Launch the GPU Kernel here, you may want multiple implementations to compare

    // int Nout = N*K*P*Q;
    // std::cout << Nout << "\n";
    std::cout << N << "\t" << C << "\t" << K << "\t" << H << "\t" << W << "\t" << R << "\t" << S << "\t" << u << "\t" << v <<  P << "\t" << Q << "\n";

    int gsize;
    int bsize;

    dim3 Block_Spec;
    
    gsize = ceil( N*K*P*Q/ (double)(BLOCK_SIZE) );

    dim3 Grid_Spec( gsize, 1);

    cudaEventCreate(&par_start);
    cudaEventCreate(&par_stop);
    cudaEventRecord(par_start);

    cnn<<<Grid_Spec, Block_Spec>>>(N,C,K,H,W,R,S,u,v,P,Q,d_input,d_weight,d_output);

    cudaEventRecord(par_stop);
    cudaEventSynchronize(par_stop);
    cudaEventElapsedTime(&par_time,par_start, par_stop);


    cudaError_t err = cudaGetLastError();
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }

    //@@ Copy the GPU memory back to the CPU here
    chkerr(cudaMemcpy(output_par, d_output, sizeof(float) * N * K * P * Q, cudaMemcpyDeviceToHost));

    //@@ Free the GPU memory here
    chkerr( cudaFree( d_input ) );
    chkerr( cudaFree( d_output ) );
    chkerr( cudaFree( d_weight ) );

    double errorval = 0;

    // VERIFY CORRECTNESS BY COMPARING OUTPUTS
    for (unsigned int n=0; n<N; n++) { // minibatch size
        for (unsigned int k=0; k<K; k ++) { // output feature map
            for (unsigned int p=0; p<P; p ++) { // output height
                for (unsigned int q =0; q<Q; q ++) { // output width
                    errorval = abs(output_seq[n*K*P*Q+k*P*Q+p*Q+q]-output_par[n*K*P*Q+k*P*Q+p*Q+q]);
                    if( errorval > .001) {
                        printf("Outputs do not match!!!\n");
                        std::cout << std::setprecision(20) << errorval << "\t" << n << "\t" << k << "\t" << p << "\t" << q << "\t" << output_seq[n*K*P*Q+k*P*Q+p*Q+q] << "\t" << output_par[n*K*P*Q+k*P*Q+p*Q+q] << "\n";
                        exit(2);
                    }
                }
            }
        }
    }

    // PRINT OUT SPEEDUP
    printf ("Sequential time = %f, Parallel time = %f, Speedup = %f\n",seq_time, par_time, seq_time/par_time);
}

