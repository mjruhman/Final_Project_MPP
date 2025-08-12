#include <cuda.h>
#include <stdio.h>
#include <float.h>
#include <algorithm> 
#include <cuda_runtime.h>

// Must be power of two and fit in shared memory
#define BITONIC_BLOCK_SIZE 1024

#define BLOCK_SIZE 512
#define NUM_BUCKETS 256

// Forward declaration of what is going to be used by my main
void sortDispatch(float *d_in, float *d_out, unsigned n, int algo);

// ------------------Bitonic Sort Op-----------------------------------------
__device__ void bitonicSortShared(float* sharedData)
{
    unsigned int tid = threadIdx.x;
    const unsigned int N = BITONIC_BLOCK_SIZE;

    // Outer loop will control size and inner will control stride
    for (unsigned int size = 2; size <= N; size <<= 1) {
        for (unsigned int stride = size >> 1; stride > 0; stride >>= 1) {

            // Make sure to sync threads
            __syncthreads();
            unsigned int pos = 2 * tid - (tid & (stride - 1));
            if (pos + stride < N) {

                //Determine if going up or down
                bool ascending = ((pos & size) == 0);
                float val1 = sharedData[pos];
                float val2 = sharedData[pos + stride];

                // Swap
                if ((val1 > val2) == ascending) {
                    sharedData[pos] = val2;
                    sharedData[pos + stride] = val1;
                }
            }
        }
    }
}

__global__ void bitonicSortBlocksKernel(float* d_data, unsigned size)
{
    __shared__ float s_data[BITONIC_BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int start = blockIdx.x * BITONIC_BLOCK_SIZE;

    // Load data or padding
    if (start + tid < size) {
        s_data[tid] = d_data[start + tid];
    } else {
        s_data[tid] = FLT_MAX;
    }
    __syncthreads();

    // Sort chunk in shared memory
    bitonicSortShared(s_data);

    __syncthreads();

    // Write back to global memory 
    if (start + tid < size) {
        d_data[start + tid] = s_data[tid];
    }
}

// Include the global for fallback if needed!!!!!!!!!!!!!!!!!
__global__ void bitonicCompareGlobal(float *data, unsigned size, unsigned j, unsigned k)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ixj = i ^ j;

    if (i < size && ixj < size && ixj > i) {
        bool up = ((i & k) == 0);
        float a = data[i];
        float b = data[ixj];
        if ( (up && a > b) || (!up && a < b) ) {
            // swap
            data[i] = b;
            data[ixj] = a;
        }
    }
}

// Include the global for fallback if needed!!!!!!!!!!!!!!!!!
void bitonicNaive(float *d_data, unsigned n)
{
    unsigned sizePow2 = 1;
    while (sizePow2 < n) sizePow2 <<= 1;
    unsigned threads = 256;
    unsigned blocks = (sizePow2 + threads - 1) / threads;

    for (unsigned k = 2; k <= sizePow2; k <<= 1) {
        for (unsigned j = k >> 1; j > 0; j >>= 1) {
            bitonicCompareGlobal<<<blocks, threads>>>(d_data, sizePow2, j, k);
            cudaDeviceSynchronize();
        }
    }
}

// Partition arrays when merging
__device__ unsigned int binarySearch(const float* A, unsigned int sizeA,
                                     const float* B, unsigned int sizeB,
                                     unsigned int k)
{
    unsigned int low = (k > sizeB) ? k - sizeB : 0;
    unsigned int high = (k < sizeA) ? k : sizeA;

    // Binary search for the correct partition 
    while (low < high) {
        unsigned int mid = (low + high) / 2;
        unsigned int j = k - mid - 1;

        if (mid < sizeA && j < sizeB && A[mid] < B[j])
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}


// Kernel merges two sorted subarrays A and B into C
__global__ void mergeKernel(
    const float* A, unsigned int sizeA,
    const float* B, unsigned int sizeB,
    float* C)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int totalSize = sizeA + sizeB;
    unsigned int numThreads = gridDim.x * blockDim.x;

    // Divide output array into chunks per thread 
    unsigned int chunkSize = (totalSize + numThreads - 1) / numThreads;

    unsigned int start = tid * chunkSize;
    unsigned int end = start + chunkSize;
    if (end > totalSize) end = totalSize;

    if (start >= end) return;

    // Find partition points for subarrays A and B
    unsigned int aStart = binarySearch(A, sizeA, B, sizeB, start);
    unsigned int bStart = start - aStart;

    unsigned int aEnd = binarySearch(A, sizeA, B, sizeB, end);
    unsigned int bEnd = end - aEnd;

    unsigned int i = aStart;
    unsigned int j = bStart;
    unsigned int outIdx = start;

    // Merge
    while (i < aEnd && j < bEnd) {
        if (A[i] <= B[j]) {
            C[outIdx++] = A[i++];
        } else {
            C[outIdx++] = B[j++];
        }
    }

    // Copy eany remaining elements if needed
    while (i < aEnd) {
        C[outIdx++] = A[i++];
    }
    while (j < bEnd) {
        C[outIdx++] = B[j++];
    }
}

// Merge chunks repeatedly until entire array sorted
void gpuMultiPassMerge(float* d_data, float* d_buffer, unsigned size, unsigned chunkSize)
{
    float* in = d_data;
    float* out = d_buffer;

    unsigned currentChunkSize = chunkSize;
    unsigned numChunks = (size + currentChunkSize - 1) / currentChunkSize;

    const unsigned threadsPerBlock = 256;

    while (numChunks > 1) {
        unsigned numPairs = numChunks / 2;

        // Launch the kernel for all pairs of chunks

        for (unsigned pair = 0; pair < numPairs; ++pair) {
            unsigned startA = pair * 2 * currentChunkSize;
            unsigned sizeA = currentChunkSize;
            if (startA + sizeA > size)
                sizeA = size - startA;

            unsigned startB = startA + sizeA;
            unsigned sizeB = currentChunkSize;
            if (startB + sizeB > size)
                sizeB = size - startB;

            unsigned totalSize = sizeA + sizeB;

            unsigned blocks = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

            mergeKernel<<<blocks, threadsPerBlock>>>(
                in + startA, sizeA,
                in + startB, sizeB,
                out + startA);
        }

        cudaDeviceSynchronize();

        // Copy leftover chunk if odd number of chunks
        if (numChunks % 2 == 1) {
            unsigned lastStart = (numChunks - 1) * currentChunkSize;
            unsigned lastSize = size - lastStart;
            cudaMemcpy(out + lastStart, in + lastStart, lastSize * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        // Swap buffers for next merge

        std::swap(in, out);

        currentChunkSize *= 2;
        numChunks = (numChunks + 1) / 2;
    }

    // If final data is in buffer, copy back
    if (in != d_data) {
        cudaMemcpy(d_data, in, size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

// ------------------------------Odd-Even-Op-----------------------------------
// Global kernel: one phase (odd or even) of compare-swap over the entire array
// Exact same as in the naive
__global__ void oddEvenCompareGlobal(float *data, unsigned size, int phase)
{
    // Global thread ID
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // calc first index of the first element in the pair
    unsigned int i = 2 * gid + (phase & 1);

    // Boundary condition
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

// Shared-memory local odd-even sort per block: fully sorts local chunk
__global__ void oddEvenLocalSort(float *data, unsigned size)
{
    __shared__ float s[BLOCK_SIZE];
    unsigned int base = blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;

    // Load data into shared mem with padding
    if (base + tid < size) {
        s[tid] = data[base + tid];
    } else {
        s[tid] = FLT_MAX;
    }
    __syncthreads();

    // Perform full odd-even sort inside shared mem (BLOCK_SIZE phases)
    for (int phase = 0; phase < BLOCK_SIZE; ++phase) {
        int start = phase & 1; // Either odd or even (0 or 1)
        // Odd-even compare-swap
        if ((tid & 1) == start && tid + 1 < BLOCK_SIZE) {
            if (s[tid] > s[tid + 1]) {
                float tmp = s[tid];
                s[tid] = s[tid + 1];
                s[tid + 1] = tmp;
            }
        }
        __syncthreads();
    }

    // Write back sorted data to global mem
    if (base + tid < size) {
        data[base + tid] = s[tid];
    }
}

// Host function that runs n phases of global + local odd-even
void oddEvenShared(float *d_data, unsigned n)
{
    unsigned threads = BLOCK_SIZE;
    unsigned blocks = (n + threads - 1) / threads;

    unsigned globalPairs = (n + 1) / 2;
    unsigned globalBlocks = (globalPairs + threads - 1) / threads;

    // Loop through all phases
    for (int phase = 0; phase < (int)n; ++phase) {
        // 1) Run global odd-even phase (one pass over entire array)
        oddEvenCompareGlobal<<<globalBlocks, threads>>>(d_data, n, phase);
        cudaDeviceSynchronize();

        // 2) Run local odd-even full sort per block (improves local order)
        oddEvenLocalSort<<<blocks, threads>>>(d_data, n);
        cudaDeviceSynchronize();
    }
}

//---------------------------------Bucket Op------------------------

// Build histogram of bucket counts for each block
__global__ void histogramKernel(
    const float* d_input,
    unsigned int* d_blockHistograms,
    unsigned int numElements)
{
    __shared__ unsigned int localHist[NUM_BUCKETS];
    unsigned int tid = threadIdx.x;
    unsigned int blockOffset = blockIdx.x * blockDim.x;

    // Initialize shared histogram to 0
    for (int i = tid; i < NUM_BUCKETS; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Have each thread process multiple elements strided by the grid size
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = blockOffset + tid; i < numElements; i += stride) {
        float val = d_input[i];
        int bucket = (int)(val * NUM_BUCKETS);
        if (bucket >= NUM_BUCKETS) bucket = NUM_BUCKETS - 1;
        if (bucket < 0) bucket = 0;
        atomicAdd(&localHist[bucket], 1);
    }
    __syncthreads();

    // Write block local to global
    for (int i = tid; i < NUM_BUCKETS; i += blockDim.x) {
        d_blockHistograms[blockIdx.x * NUM_BUCKETS + i] = localHist[i];
    }
}

// Reduce histograms kernel: sum all block histograms into global histogram
__global__ void reduceHistograms(
    const unsigned int* d_blockHistograms,
    unsigned int* d_globalHistogram,
    unsigned int numBlocks)
{
    unsigned int tid = threadIdx.x;
    unsigned int bucket = blockIdx.x;

    unsigned int sum = 0;

    // Each thread sums partial counts for one bucket over multiple blocks
    for (unsigned int i = tid; i < numBlocks; i += blockDim.x) {
        sum += d_blockHistograms[i * NUM_BUCKETS + bucket];
    }

    __shared__ unsigned int sdata[BLOCK_SIZE];
    sdata[tid] = sum;
    __syncthreads();

    // Parallel Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write
    if (tid == 0) {
        d_globalHistogram[bucket] = sdata[0];
    }
}

// Scatter kernel to sort
__global__ void scatterKernel(
    const float* d_input,
    float* d_output,
    const unsigned int* d_prefixSum,
    unsigned int* d_bucketOffsets_global,  // global atomic counters for buckets
    unsigned int numElements)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numElements) return;

    float val = d_input[tid];
    int bucket = (int)(val * NUM_BUCKETS);
    if (bucket >= NUM_BUCKETS) bucket = NUM_BUCKETS - 1;
    if (bucket < 0) bucket = 0;

    // Automatically get position within bucket to write
    unsigned int pos = atomicAdd(&d_bucketOffsets_global[bucket], 1);

    // Calc final output position using prefix sum + offset
    unsigned int finalPos = d_prefixSum[bucket] + pos;
    d_output[finalPos] = val;
}

// prefix sum of histogram counts for bucket offsets
void prefixSumCPU(unsigned int* hist, unsigned int* prefixHist, int size)
{
    prefixHist[0] = 0;
    for (int i = 1; i < size; i++) {
        prefixHist[i] = prefixHist[i-1] + hist[i-1];
    }
}

// Bucket sort in multiple stages using what we have above
void bucketSort(float* d_in, float* d_out, unsigned numElements)
{
    int threads = BLOCK_SIZE;
    int blocks = (numElements + threads - 1) / threads;

    // Allocate histograms
    unsigned int* d_blockHistograms;
    unsigned int* d_globalHistogram;
    unsigned int* d_prefixSum;
    unsigned int* d_bucketOffsets_global;

    cudaMalloc(&d_blockHistograms, blocks * NUM_BUCKETS * sizeof(unsigned int));
    cudaMalloc(&d_globalHistogram, NUM_BUCKETS * sizeof(unsigned int));
    cudaMalloc(&d_prefixSum, NUM_BUCKETS * sizeof(unsigned int));
    cudaMalloc(&d_bucketOffsets_global, NUM_BUCKETS * sizeof(unsigned int));
    cudaMemset(d_bucketOffsets_global, 0, NUM_BUCKETS * sizeof(unsigned int));  // clear counters

    // Build block histograms
    histogramKernel<<<blocks, threads>>>(d_in, d_blockHistograms, numElements);
    cudaDeviceSynchronize();

    // Reduce histograms to global histogram
    reduceHistograms<<<NUM_BUCKETS, BLOCK_SIZE>>>(d_blockHistograms, d_globalHistogram, blocks);
    cudaDeviceSynchronize();

    // Copy global histogram to host and prefix sum
    unsigned int h_globalHistogram[NUM_BUCKETS];
    unsigned int h_prefixSum[NUM_BUCKETS];
    cudaMemcpy(h_globalHistogram, d_globalHistogram, NUM_BUCKETS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    prefixSumCPU(h_globalHistogram, h_prefixSum, NUM_BUCKETS);

    cudaMemcpy(d_prefixSum, h_prefixSum, NUM_BUCKETS * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Scatter elements into output array using global atomic counters
    scatterKernel<<<blocks, threads>>>(d_in, d_out, d_prefixSum, d_bucketOffsets_global, numElements);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_blockHistograms);
    cudaFree(d_globalHistogram);
    cudaFree(d_prefixSum);
    cudaFree(d_bucketOffsets_global);
}

// this is the dispatch i chose to use for the shared
void sortDispatch(float *d_in, float *d_out, unsigned n, int algo)
{
    switch(algo) {
        case 0: {
            unsigned numBlocks = (n + BITONIC_BLOCK_SIZE - 1) / BITONIC_BLOCK_SIZE;

            // Bitonic sort each chunk
            bitonicSortBlocksKernel<<<numBlocks, BITONIC_BLOCK_SIZE>>>(d_in, n);
            cudaDeviceSynchronize();

            // Merge sorted chunks until fully merged
            gpuMultiPassMerge(d_in, d_out, n, BITONIC_BLOCK_SIZE);
            break;
        }
        case 1:
            oddEvenShared(d_in, n);
            break;
        case 2:
            bucketSort(d_in, d_out, n);
            break;
        default:
            printf("Unknown algorithm.\n");
    }
}

