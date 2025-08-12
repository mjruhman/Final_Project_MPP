#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "support.h"
#include "kernel.cu" 

// Verify that array is sorted ascending
void verifySorted(float *h, unsigned n)
{
    for (unsigned i = 1; i < n; ++i) {
        if (h[i-1] > h[i]) {
            printf("    Verification FAILED at index %u: %f > %f\n", i-1, h[i-1], h[i]);
            return;
        }
    }
    printf("    Verification PASSED (array sorted ascending)\n");
}

int main(int argc, char* argv[])
{
    Timer timer;

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *in_h, *out_h;
    float *in_d, *out_d;
    unsigned num_elements;
    int algo = 0;    // 0 = Bitonic, 1 = Odd-Even, 2 = Bucket
    cudaError_t cuda_ret;

    if (argc == 1) {
        num_elements = 1000000;
    } else if (argc == 2) {
        num_elements = atoi(argv[1]);
    } else if (argc == 4) {
        num_elements = atoi(argv[1]);
        algo = atoi(argv[2]);
    } else {
        printf("\nInvalid input parameters!"
               "\nUsage: ./sort <n> <algo>"
               "\n   <n>       : number of elements (default 1,000,000)"
               "\n   <algo>    : 0=Bitonic, 1=Odd-Even, 2=Bucket");
        exit(0);
    }

    initVector(&in_h, num_elements); // your support.h must fill with floats

    out_h = (float*)malloc(num_elements * sizeof(float));
    if (out_h == NULL) FATAL("Unable to allocate host");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u, algo=%d", num_elements, algo);

    // Allocate device memory
    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**)&in_d, num_elements * sizeof(float));
    if (cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for in_d");

    // For bucket sort we'll also use out_d; for consistency allocate it always
    cuda_ret = cudaMalloc((void**)&out_d, num_elements * sizeof(float));
    if (cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for out_d");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host -> device
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(in_d, in_h, num_elements * sizeof(float),
                          cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch chosen sort
    printf("Launching sort kernel(s)..."); fflush(stdout);
    startTime(&timer);

    // dispatcher in kernel.cu:
    // sortDispatch(in_d, out_d, num_elements, algo)
    sortDispatch(in_d, out_d, num_elements, algo);

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device -> host
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    // Most sorts place result in in_d (in-place) or out_d (bucket scatter)
    // We copy from out_d if algo==2 (bucket), else from in_d
    if (algo == 2) {
        cuda_ret = cudaMemcpy(out_h, out_d, num_elements * sizeof(float),
                              cudaMemcpyDeviceToHost);
    } else {
        cuda_ret = cudaMemcpy(out_h, in_d, num_elements * sizeof(float),
                              cudaMemcpyDeviceToHost);
    }
    if (cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness
    printf("Verifying results..."); fflush(stdout);
    verifySorted(out_h, num_elements);

    // Free memory
    cudaFree(in_d);
    cudaFree(out_d);
    free(in_h);
    free(out_h);

    return 0;
}
