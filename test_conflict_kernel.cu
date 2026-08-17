#include <cstdio>
#include <cassert>
#include <vector>

// A transaction is a transfer: move `amount` from account `from` to
// account `to`. This is our stand-in for "a blockchain transaction" -
// the exact domain from Adapter 1 in the spec.
struct Transaction {
    int from;
    int to;
    int amount;
};

// Two transactions CONFLICT if they touch any of the same accounts.
// This is the core concept: conflicts are about overlapping READ/WRITE
// sets, not about the transactions being "similar" in any other way.
__device__ __host__ bool conflicts(const Transaction& a, const Transaction& b) {
    return a.from == b.from || a.from == b.to ||
           a.to   == b.from || a.to   == b.to;
}

// KERNEL 1: for each not-yet-committed transaction, check if it's safe
// to apply THIS round. "Safe" means: no transaction with a LOWER index
// (i.e. one that must logically happen first) is both (a) still
// uncommitted and (b) touches the same accounts we do.
//
// This is the conflict check. It only READS `committed[]` - never
// writes it - so there's no race between threads doing this check.
__global__ void check_ready(const Transaction* txns, int n,
                             const int* committed, int* ready) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (committed[i]) {
        ready[i] = 0; // already done, nothing to do
        return;
    }

    bool safe = true;
    for (int j = 0; j < i; j++) {
        if (!committed[j] && conflicts(txns[i], txns[j])) {
            safe = false; // an earlier, unresolved conflict exists - wait
            break;
        }
    }
    ready[i] = safe ? 1 : 0;
}

// KERNEL 2: for every transaction marked ready this round, actually
// apply it - read the current balances, compute the transfer, write
// the result, mark it committed.
//
// This is safe to run in parallel because of a fact we can PROVE from
// how check_ready works: any two transactions that are BOTH ready in
// the same round cannot conflict with each other. If they did, the
// higher-indexed one would have seen the lower one as an unresolved
// conflict and marked itself not-ready. So every write here touches
// accounts nothing else in this round is touching.
__global__ void commit_ready(const Transaction* txns, int n,
                              const int* ready, int* balance, int* committed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!ready[i]) return;

    balance[txns[i].from] -= txns[i].amount;
    balance[txns[i].to]   += txns[i].amount;
    committed[i] = 1;
}

int main() {
    int num_accounts = 5;
    std::vector<int> initial_balance = {100, 100, 100, 100, 100};

    // A mix of conflicting and non-conflicting transactions.
    // 0 and 1 conflict (both touch account 0). 2 is independent.
    // 3 conflicts with 1 (both touch account 2). 4 is independent.
    std::vector<Transaction> txns = {
        {0, 1, 10}, // 0: account0 -> account1
        {2, 0, 5},  // 1: account2 -> account0  (conflicts with 0 via account 0)
        {3, 4, 20}, // 2: account3 -> account4  (independent)
        {1, 2, 15}, // 3: account1 -> account2  (conflicts with 1 via account 2)
        {4, 3, 8},  // 4: account4 -> account3  (conflicts with 2 via account 3/4)
    };
    int n = txns.size();

    // --- CPU ground truth: apply strictly in order 0..n-1 ---
    std::vector<int> cpu_balance = initial_balance;
    for (const auto& t : txns) {
        cpu_balance[t.from] -= t.amount;
        cpu_balance[t.to]   += t.amount;
    }

    // --- GPU setup ---
    Transaction* d_txns;
    int* d_balance;
    int* d_committed;
    int* d_ready;

    cudaMalloc(&d_txns, n * sizeof(Transaction));
    cudaMalloc(&d_balance, num_accounts * sizeof(int));
    cudaMalloc(&d_committed, n * sizeof(int));
    cudaMalloc(&d_ready, n * sizeof(int));

    cudaMemcpy(d_txns, txns.data(), n * sizeof(Transaction), cudaMemcpyHostToDevice);
    cudaMemcpy(d_balance, initial_balance.data(), num_accounts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_committed, 0, n * sizeof(int));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // --- The OCC loop: alternate readiness-check and commit until done ---
    int round = 0;
    std::vector<int> committed_host(n);
    int total_committed = 0;

    do {
        check_ready<<<blocks, threads>>>(d_txns, n, d_committed, d_ready);
        cudaDeviceSynchronize();

        commit_ready<<<blocks, threads>>>(d_txns, n, d_ready, d_balance, d_committed);
        cudaDeviceSynchronize();

        cudaMemcpy(committed_host.data(), d_committed, n * sizeof(int), cudaMemcpyDeviceToHost);
        total_committed = 0;
        for (int c : committed_host) total_committed += c;

        round++;
        printf("Round %d: %d/%d transactions committed\n", round, total_committed, n);
    } while (total_committed < n);

    // --- Compare final balances against CPU ground truth ---
    std::vector<int> gpu_balance(num_accounts);
    cudaMemcpy(gpu_balance.data(), d_balance, num_accounts * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nFinal balances:\n");
    for (int i = 0; i < num_accounts; i++) {
        printf("  account %d: gpu=%d cpu=%d\n", i, gpu_balance[i], cpu_balance[i]);
        assert(gpu_balance[i] == cpu_balance[i]);
    }
    printf("GPU optimistic execution matches strict sequential CPU execution.\n");

    cudaFree(d_txns);
    cudaFree(d_balance);
    cudaFree(d_committed);
    cudaFree(d_ready);

    return 0;
}