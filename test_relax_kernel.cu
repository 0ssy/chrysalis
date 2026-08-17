#include "csr_dag.h"
#include <cstdio>
#include <cassert>
#include <algorithm>

// GROUND TRUTH: compute each node's depth the "obviously correct" way -
// recursively, following dependencies first. We use this afterward to
// check our parallel answer is actually right, not just fast.
int cpu_depth(const CSRGraph& g, int node, std::vector<int>& memo) {
    if (memo[node] != -1) return memo[node];
    int best = 0;
    for (int i = 0; i < g.degree(node); i++) {
        int dep = g.neighbors(node)[i];
        best = std::max(best, cpu_depth(g, dep, memo) + 1);
    }
    memo[node] = best;
    return best;
}

// THE KERNEL: one round of parallel relaxation.
// Every thread independently recomputes ITS node's depth based on
// whatever its dependencies' depths currently are - even if those
// dependencies haven't finished updating yet this round.
__global__ void relax_step(const int* offsets, const int* edges,
                            const int* old_depth, int* new_depth,
                            int num_nodes, int* changed_flag) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    int best = 0;
    int start = offsets[node];
    int end = offsets[node + 1];
    for (int i = start; i < end; i++) {
        int dep = edges[i];
        int candidate = old_depth[dep] + 1;
        if (candidate > best) best = candidate;
    }

    if (best > old_depth[node]) {
        new_depth[node] = best;
        *changed_flag = 1;
    } else {
        new_depth[node] = old_depth[node];
    }
}

int main() {
    std::vector<std::pair<int,int>> raw_edges = {
        {5, 2}, {5, 7},
        {6, 1},
        {2, 0},
        {7, 1},
    };
    int num_nodes = 8;
    CSRGraph graph = build_csr(num_nodes, raw_edges);

    std::vector<int> memo(num_nodes, -1);
    std::vector<int> cpu_result(num_nodes);
    for (int i = 0; i < num_nodes; i++) cpu_result[i] = cpu_depth(graph, i, memo);

    int* d_offsets;
    int* d_edges;
    int* d_depth_a;
    int* d_depth_b;
    int* d_changed;

    cudaMalloc(&d_offsets, graph.offsets.size() * sizeof(int));
    cudaMalloc(&d_edges, graph.edges.size() * sizeof(int));
    cudaMalloc(&d_depth_a, num_nodes * sizeof(int));
    cudaMalloc(&d_depth_b, num_nodes * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));

    cudaMemcpy(d_offsets, graph.offsets.data(),
               graph.offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, graph.edges.data(),
               graph.edges.size() * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> zeros(num_nodes, 0);
    cudaMemcpy(d_depth_a, zeros.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

    int* current = d_depth_a;
    int* next = d_depth_b;
    int round = 0;
    int host_changed;

    do {
        host_changed = 0;
        cudaMemcpy(d_changed, &host_changed, sizeof(int), cudaMemcpyHostToDevice);

        relax_step<<<blocks, threads_per_block>>>(
            d_offsets, d_edges, current, next, num_nodes, d_changed);
        cudaDeviceSynchronize();

        cudaMemcpy(&host_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);

        std::swap(current, next);
        round++;
    } while (host_changed);

    printf("Converged after %d rounds.\n", round);

    std::vector<int> gpu_result(num_nodes);
    cudaMemcpy(gpu_result.data(), current, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_nodes; i++) {
        printf("node %d: gpu=%d cpu=%d\n", i, gpu_result[i], cpu_result[i]);
        assert(gpu_result[i] == cpu_result[i]);
    }
    printf("GPU relaxation matches CPU ground truth.\n");

    cudaFree(d_offsets);
    cudaFree(d_edges);
    cudaFree(d_depth_a);
    cudaFree(d_depth_b);
    cudaFree(d_changed);

    return 0;
}