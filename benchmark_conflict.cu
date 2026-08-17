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

__device__ __host__ bool conflicts(const Transaction& a, const Transaction& b) {
    return a.from == b.from || a.from == b.to ||
           a.to   == b.from || a.to   == b.to;
}

__global__ void check_ready(const Transaction* txns, int n,
                             const int* committed, int* ready) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (committed[i]) { ready[i] = 0; return; }

    bool safe = true;
    for (int j = 0; j < i; j++) {
        if (!committed[j] && conflicts(txns[i], txns[j])) {
            safe = false;
            break;
        }
    }
    ready[i] = safe ? 1 : 0;
}

__global__ void commit_ready(const Transaction* txns, int n,
                              const int* ready, int* balance, int* committed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!ready[i]) return;

    balance[txns[i].from] -= txns[i].amount;
    balance[txns[i].to]   += txns[i].amount;
    committed[i] = 1;
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

    cudaMalloc(&d_txns, n * sizeof(Transaction));
    cudaMalloc(&d_balance, num_accounts * sizeof(int));
    cudaMalloc(&d_committed, n * sizeof(int));
    cudaMalloc(&d_ready, n * sizeof(int));

    cudaMemcpy(d_txns, txns.data(), n * sizeof(Transaction), cudaMemcpyHostToDevice);
    cudaMemcpy(d_balance, initial_balance.data(), num_accounts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_committed, 0, n * sizeof(int));

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
    std::vector<int> committed_host(n);
    int total_committed = 0;
    do {
        check_ready<<<blocks, threads>>>(d_txns, n, d_committed, d_ready);
        commit_ready<<<blocks, threads>>>(d_txns, n, d_ready, d_balance, d_committed);
        cudaDeviceSynchronize();

        cudaMemcpy(committed_host.data(), d_committed, n * sizeof(int), cudaMemcpyDeviceToHost);
        total_committed = 0;
        for (int c : committed_host) total_committed += c;
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