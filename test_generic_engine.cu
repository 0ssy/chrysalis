#include "occ_engine.cuh"
#include "domains/blockchain_domain.cuh"
#include <cstdio>
#include <cassert>
#include <random>

std::vector<BlockchainDomain::Item> generate_transactions(int n, int num_accounts, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> account_dist(0, num_accounts - 1);
    std::uniform_int_distribution<int> amount_dist(1, 10);

    std::vector<BlockchainDomain::Item> txns;
    for (int i = 0; i < n; i++) {
        int from = account_dist(rng);
        int to = account_dist(rng);
        while (to == from) to = account_dist(rng);
        txns.push_back({from, to, amount_dist(rng)});
    }
    return txns;
}

int main() {
    int num_accounts = 100000;
    int n = 10000;
    std::vector<int> initial_balance(num_accounts, 1000);

    auto txns = generate_transactions(n, num_accounts, 42);

    // CPU ground truth - strictly sequential.
    std::vector<int> cpu_balance = initial_balance;
    for (const auto& t : txns) {
        cpu_balance[t.from] -= t.amount;
        cpu_balance[t.to]   += t.amount;
    }

    // GPU via the GENERIC engine - notice this line doesn't mention
    // conflicts, resources, or rounds at all. All of that lives inside
    // BlockchainDomain and occ_engine.cuh now.
    int* d_balance;
    cudaMalloc(&d_balance, num_accounts * sizeof(int));
    cudaMemcpy(d_balance, initial_balance.data(), num_accounts * sizeof(int), cudaMemcpyHostToDevice);

    int rounds = run_occ_engine<BlockchainDomain>(txns, num_accounts, d_balance);

    std::vector<int> gpu_balance(num_accounts);
    cudaMemcpy(gpu_balance.data(), d_balance, num_accounts * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_accounts; i++) assert(cpu_balance[i] == gpu_balance[i]);
    printf("Generic engine matches CPU ground truth. Rounds: %d\n", rounds);

    cudaFree(d_balance);
    return 0;
}s