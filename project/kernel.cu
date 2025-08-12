#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#define BLOCK_SIZE 512

// Forward declaration of what is going to be used by my main
void sortDispatch(float *d_in, float *d_out, unsigned n, int algo);

// ----------------------------- BITONIC SORT -----------------------------

// Used for debugging. It is a safety wrapper around Cuda runtime API calls
// Executes x to check for errors and will print the errors with the file, line number.
#define CUDA_CALL(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)


// Kernel to set values (used to fill padding with FLT_MAX)
// This makes the padding values act as if they are the largest so they will naturally
// end up at the end of the array. 
__global__ void setValueKernel(float* a, unsigned cnt, float v) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cnt) a[i] = v;
}


// Compare kernel
__global__ void bitonicCompareGlobal(float *data, unsigned sizePow2, unsigned j, unsigned k)
{
    // Global thread ID
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // This is a check to see if the thread index is out of range, if it is stop
    if (i >= sizePow2) return;

    unsigned int ixj = i ^ j;

    // only do compare and swap for each pair once
    if (ixj > i && ixj < sizePow2) {

        // need to determine if it is ascending or descending
        bool up = ((i & k) == 0);
        float a = data[i];
        float b = data[ixj];
        
        // Checking to see if the values are out of order, if they are then swap
        if ( (up && a > b) || (!up && a < b) ) {
            // swap
            data[i] = b;
            data[ixj] = a;
        }
    }
}

// Naive bitonic sort for size n
void bitonicNaive(float *d_data_orig, unsigned n)
{
    // compute next power of two
    unsigned sizePow2 = 1;
    while (sizePow2 < n) sizePow2 <<= 1;

    // if already power-of-two equals n, just run in-place
    // no padding needed
    if (sizePow2 == n) {
        unsigned threads = 256;
        unsigned blocks = (sizePow2 + threads - 1) / threads;

        // K is the size of the subsequence being merged
        for (unsigned k = 2; k <= sizePow2; k <<= 1) {

            // J is distance of comparison partners
            for (unsigned j = k >> 1; j > 0; j >>= 1) {
                // Perform the compare and swap across array one time
                bitonicCompareGlobal<<<blocks, threads>>>(d_data_orig, sizePow2, j, k);

                // Error check
                CUDA_CALL(cudaGetLastError());
            }
        }
        return;
    }


    // If it is not a power of two we will need padding

    // allocate padded buffer for our new size
    float *d_padded = nullptr;
    CUDA_CALL(cudaMalloc((void**)&d_padded, sizePow2 * sizeof(float)));

    // Copying existing data into start of the padded buffer
    CUDA_CALL(cudaMemcpy(d_padded, d_data_orig, n * sizeof(float), cudaMemcpyDeviceToDevice));

    // fill remainder [n, sizePow2) with FLT_MAX so it acts like infinity
    unsigned tail = sizePow2 - n;
    if (tail > 0) {
        unsigned threads = 256;
        unsigned blocks = (tail + threads - 1) / threads;

        // Filling with large values so they will go to the end
        setValueKernel<<<blocks, threads>>>(d_padded + n, tail, FLT_MAX);
        CUDA_CALL(cudaGetLastError());
    }

    // run bitonic sort with the new padding, uses same principles as the one without padding now
    // CAUTION PLEASE REMEMBER TO USE CORRECT VARIABLES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    {
        unsigned threads = 256;
        unsigned blocks = (sizePow2 + threads - 1) / threads;
        for (unsigned k = 2; k <= sizePow2; k <<= 1) {
            for (unsigned j = k >> 1; j > 0; j >>= 1) {
                bitonicCompareGlobal<<<blocks, threads>>>(d_padded, sizePow2, j, k);
                CUDA_CALL(cudaGetLastError());
            }
        }
    }

    // Copy back only first n sorted elements so now we have our sorted array with no padding
    CUDA_CALL(cudaMemcpy(d_data_orig, d_padded, n * sizeof(float), cudaMemcpyDeviceToDevice));

    // MAKE SURE TO FREE!!!!!!!!!!!!!!!!!!!!!!!!!!
    CUDA_CALL(cudaFree(d_padded));
}

// ----------------------------- ODD-EVEN SORT -----------------------------

__global__ void oddEvenCompareGlobal(float *data, unsigned size, int phase)
{
    // Global thread ID
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // calc first index of the first element in the pair
    unsigned int i = 2 * gid + (phase & 1);

    // Condition
    if (i + 1 < size) {
        float a = data[i];
        float b = data[i+1];

        // Swap if out of order
        if (a > b) {
            data[i] = b;
            data[i+1] = a;
        }
    }
}


void oddEvenNaive(float *d_data, unsigned n)
{
    unsigned threads = 256;

    // Number of blocks needed to cover n / 2 pairs that we will be measuring
    unsigned blocks = ( (n/2) + threads - 1 ) / threads;

    // Running n phases because that would be the worst case senerio
    for (int phase = 0; phase < (int)n; ++phase) {

        // launch one kernel per phase to do compare and swap
        oddEvenCompareGlobal<<<blocks, threads>>>(d_data, n, phase);
    }
}

// ----------------------------- BUCKET SORT -----------------------------
__global__ void bucketCountKernelNaive(const float *data, unsigned size, int numBuckets, unsigned *d_counts)
{
    // Global thread ID
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < size) {
        // Compute bucket index for element
        int b = (int)(data[gid] * numBuckets);

        // Clamp bucket to valid range
        if (b >= numBuckets) b = numBuckets - 1; 

        // Atomic add to increment count of the bucket
        atomicAdd(&d_counts[b], 1u);
    }
}

// Scatter elements
__global__ void bucketScatterKernel(const float *data, unsigned size, int numBuckets, unsigned *d_positions, float *d_output)
{
    // Global thread ID
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {

        int b = (int)(data[gid] * numBuckets);

        if (b >= numBuckets) b = numBuckets - 1;

        // Get the position to write in this bucket and increment 
        unsigned pos = atomicAdd(&d_positions[b], 1u);

        // Place element in array at computed position
        d_output[pos] = data[gid];
    }
}



void bucketNaive(float *d_in, float *d_out, unsigned n, int numBuckets)
{
    unsigned threads = 256;
    unsigned blocks = (n + threads - 1) / threads;

    // Allocate device memory to hold counts for each bucket and set to 0
    unsigned *d_counts;
    cudaMalloc(&d_counts, numBuckets * sizeof(unsigned));
    cudaMemset(d_counts, 0, numBuckets * sizeof(unsigned));

    // Count elements per bucket
    bucketCountKernelNaive<<<blocks, threads>>>(d_in, n, numBuckets, d_counts);
    cudaDeviceSynchronize();

    // Copy counts to host for prefix sum
    unsigned *h_counts = (unsigned*)malloc(numBuckets * sizeof(unsigned));
    cudaMemcpy(h_counts, d_counts, numBuckets * sizeof(unsigned), cudaMemcpyDeviceToHost);

    // Exclusive Prefix Scan
    unsigned total = 0;
    for (int i = 0; i < numBuckets; ++i) {
        unsigned c = h_counts[i];
        h_counts[i] = total;
        total += c;
    }

    // Copy offsets back to device
    unsigned *d_positions;
    cudaMalloc(&d_positions, numBuckets * sizeof(unsigned));
    cudaMemcpy(d_positions, h_counts, numBuckets * sizeof(unsigned), cudaMemcpyHostToDevice);

    // Scatter elements to their bucket positions
    bucketScatterKernel<<<blocks, threads>>>(d_in, n, numBuckets, d_positions, d_out);
    cudaDeviceSynchronize();


    cudaFree(d_counts);
    cudaFree(d_positions);
    free(h_counts);
}


// ----------------------------- Dispatcher -----------------------------

// Made this to not have to have 6 different mains and kernels
void sortDispatch(float *d_in, float *d_out, unsigned n, int algo)
{
    // algo: 0=Bitonic, 1=Odd-Even, 2=Bucket
    if (algo == 0) {
        
        bitonicNaive(d_in, n);

    } else if (algo == 1) {
        oddEvenNaive(d_in, n);

    } else if (algo == 2) {
        // choose numBuckets heuristic
        int numBuckets = 1024;
        if (n < 4096) numBuckets = 64;
        bucketNaive(d_in, d_out, n, numBuckets);
    } else {
        printf("sortDispatch: unknown algorithm %d\n", algo);
        return;
    }
}