#if __has_include("csr_dag.h")
#include "csr_dag.h"
#elif __has_include("include/csr_dag.h")
#include "include/csr_dag.h"
#elif __has_include("src/csr_dag.h")
#include "src/csr_dag.h"
#else
#error "Cannot find csr_dag.h. Add its folder to includePath or adjust the include path."
#endif
#include <cstdio>
#include <cassert>

int main() {
    // Build a tiny DAG:
    //   node 5 depends on node 2
    //   node 5 depends on node 7
    //   node 6 depends on node 1
    std::vector<std::pair<int,int>> raw_edges = {
        {5, 2}, {5, 7}, {6, 1}
    };
    int num_nodes = 8; // nodes 0..7

    CSRGraph graph = build_csr(num_nodes, raw_edges);

    // Node 5 should have 2 dependencies: 2 and 7 (order may vary
    // depending on insertion order, but count must be right).
    printf("Node 5 has %d dependencies: ", graph.degree(5));
    const int* deps5 = graph.neighbors(5);
    for (int i = 0; i < graph.degree(5); i++) {
        printf("%d ", deps5[i]);
    }
    printf("\n");
    assert(graph.degree(5) == 2);

    // Node 6 should have exactly 1 dependency: node 1.
    printf("Node 6 has %d dependencies: ", graph.degree(6));
    const int* deps6 = graph.neighbors(6);
    for (int i = 0; i < graph.degree(6); i++) {
        printf("%d ", deps6[i]);
    }
    printf("\n");
    assert(graph.degree(6) == 1);
    assert(deps6[0] == 1);

    // Node 0 should have zero dependencies - never appeared as a
    // "depends on" source in raw_edges.
    printf("Node 0 has %d dependencies\n", graph.degree(0));
    assert(graph.degree(0) == 0);

    printf("All checks passed.\n");
    return 0;
}