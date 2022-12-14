// Based upon following papers: http://worldcomp-proceedings.com/proc/p2011/CSC8087.pdf
// https://www.nvidia.com/content/PDF/isc-2011/Brandvik.pdf


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// I3D to index into a linear memory space from a 3D array index
#define I3D(i, j, k) ((i) + (N)*(j) + (N)*(N)*(k)) //newcode

#define N 16


double CLOCK() {
        struct timespec t;
        clock_gettime(CLOCK_MONOTONIC,  &t);
        return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}


// Following the HW structure, b_d is the data matrix and a_d will be the result matrix
__global__ void stencil(float *b, float *a) {
	int i, j, k, i000, im100, ip100, i0m10, i0p10, i00m1, i00p1;
	 // find i and j indices of this thread	 
	i = blockIdx.x * blockDim.x + threadIdx.x; 
	j = blockIdx.y * blockDim.y + threadIdx.y;
	k = blockIdx.z * blockDim.z + threadIdx.z;

	// find indices into linear memory for central point and neighbours
	
	 // i,j,k
	 i000 = I3D(i, j,k);
	 
	 // i-1, j, k
	 // i 'minus' 1, 0, 0
	 im100 = I3D(i-1, j,k);

	 // i+1, j,k
	 // i 'plus' 1, 0, 0 
	 ip100 = I3D(i+1, j,k);

	 // i, j-1, k
	 // 0, j 'minus' 1, k
	 i0m10 = I3D(i, j-1,k);

	 // i, j +1, k
	 i0p10 = I3D(i, j+1,k);
	  
	 // i, j, k-1
	 i00m1 = I3D(i, j,k-1);

	 // i, j, k+1
	 i00p1 = I3D(i, j,k+1);

	// checks all the variables aren't exceeding bounds (0 or outside tile)
	if (i > 0 && i < N-1 && j > 0 && j < N-1 && k> 0 && k < N-1) {
	
	// update temperatures
	a[i000] =  0.8*(b[im100] + b[ip100] 
					+ b[i0m10] + b[i0p10]
					+ b[i00m1] + b[i00p1]);
	}
}


int main(){

dim3 threadsPerBlock(N,N,N);

    double start, finish, total;
 
    float *h_a;
    float *h_b;
 
    float *d_a;
    float *d_b;
  
 
	// Size, in bytes, of each vector
    size_t bytes = N*sizeof(float);
 
	 // Allocate memory for each vector on host
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

	 int i,j,k;
	 float a[N][N][N], b[N][N][N];

    for( i = 0; i < N; i++ ) {
        h_b[i] = 1;
    }

	for( i = 0; i < N; i++){
		for( j = 0; j < N; j++){
			for( k = 0 ; k < N; k++){
				b[i][j][k] = 1;
			}
		}
	}

	//CPU calculation to check for accuracy
for (i=1; i<N-1; i++)  
   for (j=1; j<N-1; j++)  
           for (k=1; k<N-1; k++) {  
 a[i][j][k]=0.8*(b[i-1][j][k]+b[i+1][j][k]+b[i][j-1][k] 
 + b[i][j+1][k]+b[i][j][k-1]+b[i][j][k+1]); 
     }  

	// Copy host vectors to device
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);

	start = CLOCK();

	stencil<<<N, threadsPerBlock>>>(d_b, d_a);

	// Copy array back to host
    cudaMemcpy( h_a, d_a, bytes, cudaMemcpyDeviceToHost );

	finish = CLOCK();
	total = finish -start;
	printf("Time for the CUDA execution = %4.2f milliseconds\n", total);

	for (i=1; i<N-1; i++)  
   for (j=1; j<N-1; j++)  
     for (k=1; k<N-1; k++) {
			int ind = I3D(i,j,k);
			float op1 = a[i][j][k];
			float op2 = h_a[ind];
			if( op1 == op2){
			continue;
			}
			else{
				printf("Answer incorrect, %f != %f", a[i][j][k], h_a[ind]);
				return -1;
			}
     }  
	 printf("Wow all the numbers were the same, here's proof %f == %f", a[1][2][3], h_a[292]);

	// Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
 
    // Release host memory
    free(h_a);
    free(h_b);
	free(a);
	free(b);
    return 0;
}


