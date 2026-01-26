#pragma once
#include <cstdint>
#include <vector>
#include <deque>

// ==============================
// Data structures (same as Python)
// ==============================

struct Syncmer {
    uint32_t pos;
    uint64_t hash;
};

struct Strobemer {
    uint32_t pos1;
    uint32_t pos2;
    uint64_t hash;
    uint32_t span;
    uint32_t length;
};

// ==============================
// API
// ==============================

std::vector<Strobemer>
strobes_from_4bit_buffer(
    const uint8_t* buf,
    uint32_t L,
    uint32_t k = 21,
    uint32_t s = 5,
    uint32_t sync_pos = 2,
    uint32_t w_min = 20,
    uint32_t w_max = 70
);
