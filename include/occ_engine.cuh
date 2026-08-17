#pragma once
#include <vector>
#include <cassert>

// A "Domain" is a struct that plugs a specific problem (blockchain,
// build graphs, git merges, etc.) into this shared engine. Every
// domain must provide:
//   - Item        : the type being processed (a transaction, a build
//                   target, a merge hunk...)
//   - State       : the shared thing items read/write (account
//                   balances, file contents, whatever "the resource"
//                   means in this domain)
//   - MAX_TOUCHES : the most resources any single item can touch
//   - touches()   : which resource IDs does this item read/write?
//   - apply()     : actually perform the item's effect on State
//
// This is the exact three-function shape from the spec's §1.4 -
// parse (handled outside, building the Item list), conflict_check
// (derived automatically from touches(), see below), apply (apply()).

// PRECOMPUTE DEPENDENCIES: for each item, track one dependency per
// touched resource. This generalizes dep_from/dep_to to up to
// Domain::MAX_TOUCHES resources.
template <typename Domain>
void compute_dependencies(const std::vector<typename Domain::Item>& items,
                           int num_resources,
                           std::vector<int>& dep) {
    int n = items.size();
    dep.assign(n * Domain::MAX_TOUCHES, -1); // unused slots remain -1

    std::vector<int> last_toucher(num_resources, -1);

    for (int i = 0; i < n; i++) {
        int touched[Domain::MAX_TOUCHES];
        int count = Domain::touches(items[i], touched);
        assert(count <= Domain::MAX_TOUCHES);

        // One dependency per touched resource, not a merged dependency.
        for (int t = 0; t < count; t++) {
            dep[i * Domain::MAX_TOUCHES + t] = last_toucher[touched[t]];
        }

        for (int t = 0; t < count; t++) {
            last_toucher[touched[t]] = i;
        }
    }
}

template <typename Domain>
__global__ void check_ready(const int* dep, int n,
                             const int* committed, int* ready) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (committed[i]) { ready[i] = 0; return; }

    bool safe = true;
    for (int t = 0; t < Domain::MAX_TOUCHES; t++) {
        int d = dep[i * Domain::MAX_TOUCHES + t];
        if (d != -1 && !committed[d]) { safe = false; break; }
    }
    ready[i] = safe ? 1 : 0;
}

template <typename Domain>
__global__ void commit_ready(const typename Domain::Item* items, int n,
                              const int* ready, typename Domain::State state,
                              int* committed, int* total_committed_counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!ready[i]) return;

    Domain::apply(items[i], state);
    committed[i] = 1;
    atomicAdd(total_committed_counter, 1);
}

// THE ENGINE ENTRY POINT: runs the full OCC loop for ANY domain that
// satisfies the Domain interface. This is the payoff - this function
// never changes no matter which of the 7 adapters we're running.
template <typename Domain>
int run_occ_engine(const std::vector<typename Domain::Item>& items,
                    int num_resources,
                    typename Domain::State d_state) {
    int n = items.size();

    std::vector<int> dep;
    compute_dependencies<Domain>(items, num_resources, dep);

    typename Domain::Item* d_items;
    int* d_dep;
    int* d_committed;
    int* d_ready;
    int* d_total_committed;

    cudaMalloc(&d_items, n * sizeof(typename Domain::Item));
    cudaMalloc(&d_dep, n * Domain::MAX_TOUCHES * sizeof(int));
    cudaMalloc(&d_committed, n * sizeof(int));
    cudaMalloc(&d_ready, n * sizeof(int));
    cudaMalloc(&d_total_committed, sizeof(int));

    cudaMemcpy(d_items, items.data(), n * sizeof(typename Domain::Item), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dep, dep.data(), n * Domain::MAX_TOUCHES * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_committed, 0, n * sizeof(int));
    cudaMemset(d_total_committed, 0, sizeof(int));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    int round = 0;
    int total_committed = 0;
    do {
        check_ready<Domain><<<blocks, threads>>>(d_dep, n, d_committed, d_ready);
        commit_ready<Domain><<<blocks, threads>>>(d_items, n, d_ready, d_state, d_committed, d_total_committed);
        cudaDeviceSynchronize();

        cudaMemcpy(&total_committed, d_total_committed, sizeof(int), cudaMemcpyDeviceToHost);
        round++;
    } while (total_committed < n);

    cudaFree(d_items);
    cudaFree(d_dep);
    cudaFree(d_committed);
    cudaFree(d_ready);
    cudaFree(d_total_committed);

    return round;
}