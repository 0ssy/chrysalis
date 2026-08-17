#pragma once

// This struct IS Adapter 1 from the spec. Everything domain-specific
// about "blockchain transactions" lives here and nowhere else. The
// engine above knows nothing about accounts or transfers - it just
// knows "Items touch Resources and can be applied to State."
struct BlockchainDomain {
    struct Item {
        int from;
        int to;
        int amount;
    };

    // State = what apply() mutates. Here it's a raw balance array.
    using State = int*;

    static const int MAX_TOUCHES = 2; // a transfer touches exactly 2 accounts

    // Which resources (account IDs) does this item touch?
    // Writes them into `out`, returns how many.
    __device__ __host__ static int touches(const Item& item, int* out) {
        out[0] = item.from;
        out[1] = item.to;
        return 2;
    }

    // The actual effect of applying this item to shared state.
    __device__ static void apply(const Item& item, State balance) {
        balance[item.from] -= item.amount;
        balance[item.to]   += item.amount;
    }
};