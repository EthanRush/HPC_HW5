#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double CLOCK() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}


int main(int argc, char* argv[])
{
    // Size of vectors
    int n = 10000000;

    double start, finish, total;

    // Host input vectors
    double* h_a;
    double* h_b;
    //Host output vector
    double* h_c;

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);

    int i;
    // Initialize vectors on host
    for (i = 0; i < n; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    start = CLOCK();
 
    #pragma acc kernels
    for (i = 0; i < n; i++) {
        h_c[i] = h_a[i] + h_b[i];
    }


    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for (i = 0; i < n; i++)
        sum += h_c[i];

    finish = CLOCK();

    printf("final result: %f\n", sum / (float)n);
    total = finish - start;
    printf("Time for the loop = %4.2f milliseconds\n", total);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
