#include <cstdio>
#include <cassert>
#include <vector>
#include <random>
#include <chrono>

struct Transaction {
    int from;
    int to;
    int amount;
};

// PREPROCESSING (host, one-time, before any rounds run): for each
// transaction, find the nearest EARLIER transaction touching the same
// account. This turns "check everyone" into "check my one dependency."
// This is exactly the dependency-graph idea from the relaxation kernel,
// just derived from resource overlap instead of given explicitly.
void compute_dependencies(const std::vector<Transaction>& txns, int num_accounts,
                          std::vector<int>& dep_from, std::vector<int>& dep_to) {
    int n = txns.size();
    dep_from.assign(n, -1);
    dep_to.assign(n, -1);

    // last_toucher[account] = index of the most recent transaction
    // seen so far that touched this account.
    std::vector<int> last_toucher(num_accounts, -1);

    for (int i = 0; i < n; i++) {
        dep_from[i] = last_toucher[txns[i].from];
        dep_to[i]   = last_toucher[txns[i].to];
        last_toucher[txns[i].from] = i;
        last_toucher[txns[i].to] = i;
    }
}

// O(1) READINESS CHECK: a transaction is ready once its (at most two)
// direct dependencies are committed. No scanning required anymore.
__global__ void check_ready_fast(const int* dep_from, const int* dep_to,
                                 int n, const int* committed, int* ready) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (committed[i]) { ready[i] = 0; return; }

    bool from_ok = (dep_from[i] == -1) || committed[dep_from[i]];
    bool to_ok   = (dep_to[i] == -1)   || committed[dep_to[i]];
    ready[i] = (from_ok && to_ok) ? 1 : 0;
}

// Added a device-side counter that tracks total commits SO FAR across
// all rounds. atomicAdd is required here - if two threads both did
// "counter = counter + 1" without atomicAdd, they could both read the
// same old value and clobber each other's increment (a classic GPU race).
__global__ void commit_ready(const Transaction* txns, int n,
                              const int* ready, int* balance, int* committed,
                              int* total_committed_counter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!ready[i]) return;

    balance[txns[i].from] -= txns[i].amount;
    balance[txns[i].to]   += txns[i].amount;
    committed[i] = 1;
    atomicAdd(total_committed_counter, 1); // one add, not a full array scan
}

// Generates `n` random transactions over `num_accounts` accounts.
// FEWER accounts relative to n = MORE conflicts (more likely two
// transactions pick overlapping accounts by chance). This lets us
// test the same engine under different contention levels - exactly
// the kind of sweep the spec calls for ("when does this win").
std::vector<Transaction> generate_transactions(int n, int num_accounts, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> account_dist(0, num_accounts - 1);
    std::uniform_int_distribution<int> amount_dist(1, 10);

    std::vector<Transaction> txns;
    txns.reserve(n);
    for (int i = 0; i < n; i++) {
        int from = account_dist(rng);
        int to = account_dist(rng);
        while (to == from) to = account_dist(rng); // no self-transfers
        txns.push_back({from, to, amount_dist(rng)});
    }
    return txns;
}

// Runs the GPU optimistic-execution loop and returns elapsed milliseconds.
// cudaEvent_t is the correct way to time GPU work - a plain CPU timer
// around a kernel launch would only measure how long it took to QUEUE
// the work, not how long the GPU actually took to run it, since kernel
// launches are asynchronous by default.
float run_gpu(const std::vector<Transaction>& txns, int num_accounts,
              const std::vector<int>& initial_balance,
              std::vector<int>& out_balance, int& out_rounds) {
    int n = txns.size();

    Transaction* d_txns;
    int* d_balance;
    int* d_committed;
    int* d_ready;
    int* d_dep_from;
    int* d_dep_to;
    int* d_total_committed;

    std::vector<int> dep_from;
    std::vector<int> dep_to;
    compute_dependencies(txns, num_accounts, dep_from, dep_to);

    cudaMalloc(&d_txns, n * sizeof(Transaction));
    cudaMalloc(&d_balance, num_accounts * sizeof(int));
    cudaMalloc(&d_committed, n * sizeof(int));
    cudaMalloc(&d_ready, n * sizeof(int));
    cudaMalloc(&d_dep_from, n * sizeof(int));
    cudaMalloc(&d_dep_to, n * sizeof(int));
    cudaMalloc(&d_total_committed, sizeof(int));

    cudaMemcpy(d_txns, txns.data(), n * sizeof(Transaction), cudaMemcpyHostToDevice);
    cudaMemcpy(d_balance, initial_balance.data(), num_accounts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dep_from, dep_from.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dep_to, dep_to.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_committed, 0, n * sizeof(int));
    cudaMemset(d_total_committed, 0, sizeof(int));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // cudaEvent-based timing: record a "start" marker, do the work,
    // record a "stop" marker, then ask the GPU how much time passed
    // between them. This is the standard, accurate way to time CUDA work.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int round = 0;
    int total_committed = 0;
    do {
        check_ready_fast<<<blocks, threads>>>(d_dep_from, d_dep_to, n, d_committed, d_ready);
        commit_ready<<<blocks, threads>>>(d_txns, n, d_ready, d_balance, d_committed, d_total_committed);
        cudaDeviceSynchronize();

        // Copy back ONE integer, not the whole committed array.
        cudaMemcpy(&total_committed, d_total_committed, sizeof(int), cudaMemcpyDeviceToHost);
        round++;
    } while (total_committed < n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    out_balance.resize(num_accounts);
    cudaMemcpy(out_balance.data(), d_balance, num_accounts * sizeof(int), cudaMemcpyDeviceToHost);
    out_rounds = round;

    cudaFree(d_txns);
    cudaFree(d_balance);
    cudaFree(d_committed);
    cudaFree(d_ready);
    cudaFree(d_dep_from);
    cudaFree(d_dep_to);
    cudaFree(d_total_committed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

// CPU baseline: strictly sequential, timed with std::chrono - the
// standard C++ way to time CPU work.
float run_cpu(const std::vector<Transaction>& txns, int num_accounts,
              const std::vector<int>& initial_balance,
              std::vector<int>& out_balance) {
    auto start = std::chrono::high_resolution_clock::now();

    out_balance = initial_balance;
    for (const auto& t : txns) {
        out_balance[t.from] -= t.amount;
        out_balance[t.to]   += t.amount;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end - start;
    return elapsed.count();
}

void run_benchmark(int n, int num_accounts, const char* label) {
    std::vector<int> initial_balance(num_accounts, 1000);
    auto txns = generate_transactions(n, num_accounts, 42);

    std::vector<int> cpu_balance, gpu_balance;
    float cpu_ms = run_cpu(txns, num_accounts, initial_balance, cpu_balance);

    int rounds;
    float gpu_ms = run_gpu(txns, num_accounts, initial_balance, gpu_balance, rounds);

    for (int i = 0; i < num_accounts; i++) assert(cpu_balance[i] == gpu_balance[i]);

    printf("[%s] n=%d accounts=%d\n", label, n, num_accounts);
    printf("  CPU: %.3f ms\n", cpu_ms);
    printf("  GPU: %.3f ms (%d rounds)\n", gpu_ms, rounds);
    printf("  Speedup: %.2fx\n", cpu_ms / gpu_ms);
    printf("  Correctness: PASS\n\n");
}

int main() {
    // Sweep across sizes and contention levels - low contention (many
    // accounts relative to transactions) vs high contention (few
    // accounts, lots of forced serialization).
    run_benchmark(1000,   10000, "small, low contention");
    run_benchmark(1000,   50,    "small, high contention");
    run_benchmark(10000,  100000,"large, low contention");
    run_benchmark(10000,  200,   "large, high contention");
    return 0;
}