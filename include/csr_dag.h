#pragma once
#include <vector>
#include <cstdint>

// CSRGraph stores a DAG using Compressed Sparse Row format.
//
// Instead of each node owning its own little array of dependencies
// (bad for GPUs - scattered memory), we store ALL dependencies in one
// flat array, and use `offsets` to say where each node's slice begins.
//
// Example: if node 5 depends on nodes [2, 7, 9], and node 6 depends on [1],
// then somewhere in `edges` you'd find "...2, 7, 9, 1..." and
// offsets[5] would point to where the "2" is, offsets[6] to where "1" is.
struct CSRGraph {
    // Flat array of every node's dependencies, back to back.
    std::vector<int> edges;

    // offsets[i] = index into `edges` where node i's dependencies start.
    // offsets[i+1] - offsets[i] = how many dependencies node i has.
    // Size is always (num_nodes + 1) - that extra last slot marks the
    // end of the final node's slice, so we don't need a special case.
    std::vector<int> offsets;

    // How many nodes are in this graph. Stored separately so we don't
    // have to keep computing offsets.size() - 1 everywhere.
    int num_nodes = 0;

    // Returns how many dependencies node `i` has.
    int degree(int i) const {
        return offsets[i + 1] - offsets[i];
    }

    // Returns a pointer to the start of node i's dependency list.
    // Combined with degree(i), this lets you loop over node i's deps:
    //   const int* deps = graph.neighbors(i);
    //   for (int j = 0; j < graph.degree(i); j++) { deps[j] ... }
    const int* neighbors(int i) const {
        return &edges[offsets[i]];
    }
};

// Converts a plain list of (node, dependency) pairs into CSR format.
//
// Example input: num_nodes=7, raw_edges = {{5,2}, {5,7}, {6,1}}
// means "node 5 depends on node 2", "node 5 depends on node 7",
// "node 6 depends on node 1". This is the natural way you'd describe
// a DAG before converting it to the GPU-friendly layout.
inline CSRGraph build_csr(int num_nodes,
                           const std::vector<std::pair<int,int>>& raw_edges) {
    CSRGraph graph;
    graph.num_nodes = num_nodes;
    graph.offsets.assign(num_nodes + 1, 0);

    // Step 1: count how many dependencies each node has.
    // We temporarily store the count in offsets[node + 1] - this trick
    // sets us up for step 2.
    for (const auto& e : raw_edges) {
        int node = e.first;
        graph.offsets[node + 1]++;
    }

    // Step 2: turn per-node counts into cumulative offsets.
    // After this, offsets[i] is the START index for node i in `edges`.
    // This is called a "prefix sum" - you'll see this exact pattern
    // constantly once we move to GPU kernels, since prefix sum is one
    // of the fundamental parallel primitives.
    for (int i = 0; i < num_nodes; i++) {
        graph.offsets[i + 1] += graph.offsets[i];
    }

    // Step 3: place each dependency into its correct slot in `edges`.
    // We use a scratch copy of offsets as a "next free slot" cursor
    // per node, so we don't clobber the real offsets while filling in.
    graph.edges.resize(raw_edges.size());
    std::vector<int> cursor = graph.offsets;
    for (const auto& e : raw_edges) {
        int node = e.first;
        int dep = e.second;
        graph.edges[cursor[node]] = dep;
        cursor[node]++;
    }

    return graph;
}