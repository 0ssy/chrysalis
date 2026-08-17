#include "csr_dag.h"
#include <cstdio>
#include <cassert>

// This is a KERNEL - a function that runs on the GPU, once per thread.
// `__global__` marks it as callable from the CPU but running on the GPU.
//
// Each thread computes the degree (dependency count) of ONE node.
// With N nodes and N threads, this whole computation happens in parallel
// instead of one node at a time like on the CPU.
__global__ void compute_degrees(const int* offsets, int num_nodes, int* out_degrees) {
    // Figure out which node THIS thread is responsible for.
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    // We might launch MORE threads than we have nodes, so extra
    // threads need to do nothing instead of reading out of bounds.
    if (node >= num_nodes) return;

    out_degrees[node] = offsets[node + 1] - offsets[node];
}

int main() {
    // --- Step 1: build the same tiny DAG as before, on the CPU (host) ---
    std::vector<std::pair<int,int>> raw_edges = {
        {5, 2}, {5, 7}, {6, 1}
    };
    int num_nodes = 8;
    CSRGraph graph = build_csr(num_nodes, raw_edges);

    // --- Step 2: allocate memory ON THE GPU (device) ---
    int* d_offsets;
    int* d_degrees;
    cudaMalloc(&d_offsets, graph.offsets.size() * sizeof(int));
    cudaMalloc(&d_degrees, num_nodes * sizeof(int));

    // --- Step 3: copy the offsets array from host memory to device memory ---
    cudaMemcpy(d_offsets, graph.offsets.data(),
               graph.offsets.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    // --- Step 4: launch the kernel ---
    int threads_per_block = 256;
    int blocks = (num_nodes + threads_per_block - 1) / threads_per_block;
    compute_degrees<<<blocks, threads_per_block>>>(d_offsets, num_nodes, d_degrees);

    cudaDeviceSynchronize();

    // --- Step 5: copy the result back from device to host ---
    std::vector<int> degrees(num_nodes);
    cudaMemcpy(degrees.data(), d_degrees, num_nodes * sizeof(int),
               cudaMemcpyDeviceToHost);

    // --- Step 6: verify against the CPU version we already trust ---
    printf("Degrees computed on GPU:\n");
    for (int i = 0; i < num_nodes; i++) {
        printf("  node %d: degree %d\n", i, degrees[i]);
        assert(degrees[i] == graph.degree(i));
    }
    printf("All GPU results match CPU results.\n");

    // --- Step 7: free GPU memory ---
    cudaFree(d_offsets);
    cudaFree(d_degrees);

    return 0;
}