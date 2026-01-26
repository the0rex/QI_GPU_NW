#include "syncmer_strobemer.hpp"
#include <limits>

// ==============================
// Constants & helpers
// ==============================

static constexpr uint64_t MASK64 = (uint64_t)-1;

// ntHash-style seed table (4-bit alphabet)
static constexpr uint64_t SEED[16] = {
    0x9ae16a3b2f90404fULL ^ 0x0,
    0x9ae16a3b2f90404fULL ^ 0x1,
    0x9ae16a3b2f90404fULL ^ 0x2,
    0x9ae16a3b2f90404fULL ^ 0x3,
    0x9ae16a3b2f90404fULL ^ 0x4,
    0x9ae16a3b2f90404fULL ^ 0x5,
    0x9ae16a3b2f90404fULL ^ 0x6,
    0x9ae16a3b2f90404fULL ^ 0x7,
    0x9ae16a3b2f90404fULL ^ 0x8,
    0x9ae16a3b2f90404fULL ^ 0x9,
    0x9ae16a3b2f90404fULL ^ 0xA,
    0x9ae16a3b2f90404fULL ^ 0xB,
    0x9ae16a3b2f90404fULL ^ 0xC,
    0x9ae16a3b2f90404fULL ^ 0xD,
    0x9ae16a3b2f90404fULL ^ 0xE,
    0x9ae16a3b2f90404fULL ^ 0xF
};

inline uint64_t rol(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

inline uint8_t base4(const uint8_t* buf, uint32_t i) {
    uint8_t b = buf[i >> 1];
    return (i & 1) ? (b & 0xF) : ((b >> 4) & 0xF);
}

inline uint64_t roll_hash(uint64_t prev, uint8_t outb, uint8_t inb) {
    return rol(prev, 1) ^ SEED[outb] ^ rol(SEED[inb], 1);
}

// ==============================
// Main pipeline
// ==============================

std::vector<Strobemer>
strobes_from_4bit_buffer(
    const uint8_t* buf,
    uint32_t L,
    uint32_t k,
    uint32_t s,
    uint32_t sync_pos,
    uint32_t w_min,
    uint32_t w_max
) {
    std::vector<Strobemer> out;
    if (L < k || s > k) return out;

    // ------------------------------
    // Syncmer extraction (O(L))
    // ------------------------------

    std::deque<std::pair<uint64_t, uint32_t>> minq;
    std::deque<uint64_t> s_hashes;
    std::vector<Syncmer> syncmers;

    uint64_t h = 0;

    // init first s-mer
    for (uint32_t i = 0; i < s; ++i)
        h = roll_hash(h, 0, base4(buf, i));

    s_hashes.push_back(h);
    minq.emplace_back(h, 0);

    // build first k-mer
    for (uint32_t i = 1; i <= k - s; ++i) {
        h = roll_hash(h,
                      base4(buf, i - 1),
                      base4(buf, i + s - 1));
        s_hashes.push_back(h);
        while (!minq.empty() && minq.back().first >= h)
            minq.pop_back();
        minq.emplace_back(h, i);
    }

    if (!minq.empty() && minq.front().second == sync_pos)
        syncmers.push_back({0, minq.front().first});

    // slide k-mer window
    for (uint32_t i = 1; i <= L - k; ++i) {
        while (!minq.empty() && minq.front().second < i)
            minq.pop_front();

        h = roll_hash(
            s_hashes.back(),
            base4(buf, i + k - s - 1),
            base4(buf, i + k - 1)
        );
        s_hashes.push_back(h);

        uint32_t idx = i + k - s;
        while (!minq.empty() && minq.back().first >= h)
            minq.pop_back();
        minq.emplace_back(h, idx);

        if (!minq.empty() && minq.front().second == i + sync_pos)
            syncmers.push_back({i, minq.front().first});
    }

    // ------------------------------
    // Randstrobe linking (O(N))
    // ------------------------------

    std::deque<Syncmer> win;

    for (const auto& s1 : syncmers) {
        while (!win.empty() && win.front().pos < s1.pos - w_max)
            win.pop_front();

        const Syncmer* best = nullptr;
        uint64_t best_val = std::numeric_limits<uint64_t>::max();

        for (const auto& s2 : win) {
            if (s2.pos >= s1.pos + w_min && s2.pos <= s1.pos + w_max) {
                uint64_t v = s1.hash ^ s2.hash;
                if (v < best_val) {
                    best_val = v;
                    best = &s2;
                }
            }
        }

        // density control (critical for identical sequences)
        if (best && ((best_val & 0xFF) == 0)) {
            out.push_back({
                s1.pos,
                best->pos,
                best_val,
                best->pos - s1.pos,
                2 * k
            });
        }

        win.push_back(s1);
    }

    return out;
}