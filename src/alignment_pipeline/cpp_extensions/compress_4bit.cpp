#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>

namespace py = pybind11;

// ASCII → 4-bit table (256 entries)
static uint8_t MAP[256];

void init_map() {
    // Initialize all to 0xF (unknown -> N)
    for (int i = 0; i < 256; ++i) MAP[i] = 0xF;
    
    // Standard bases
    MAP['A'] = MAP['a'] = 0x1;
    MAP['C'] = MAP['c'] = 0x2;
    MAP['G'] = MAP['g'] = 0x4;
    MAP['T'] = MAP['t'] = 0x8;
    //MAP['U'] = MAP['u'] = 0x8;
    
    // IUPAC ambiguous bases
    MAP['N'] = MAP['n'] = 0xF;
    MAP['R'] = MAP['r'] = 0x5;  // A or G
    MAP['Y'] = MAP['y'] = 0xA;  // C or T
    MAP['S'] = MAP['s'] = 0x6;  // G or C
    MAP['W'] = MAP['w'] = 0x9;  // A or T
    MAP['K'] = MAP['k'] = 0xC;  // G or T
    MAP['M'] = MAP['m'] = 0x3;  // A or C
    
    // ADD THESE MISSING IUPAC CODES:
    MAP['B'] = MAP['b'] = 0xE;  // C, G, or T (not A)
    MAP['D'] = MAP['d'] = 0xD;  // A, G, or T (not C)
    MAP['H'] = MAP['h'] = 0xB;  // A, C, or T (not G)
    MAP['V'] = MAP['v'] = 0x7;  // A, C, or G (not T)
    
    // Gap character
    MAP['-'] = 0x0;
}

py::bytes compress_4bit_cpp(const std::string& seq) {
    size_t L = seq.size();
    std::vector<uint8_t> out((L + 1) / 2, 0);

    for (size_t i = 0; i < L; i += 2) {
        uint8_t hi = MAP[(uint8_t)seq[i]];
        uint8_t lo = (i + 1 < L) ? MAP[(uint8_t)seq[i + 1]] : 0xF;
        out[i >> 1] = (hi << 4) | lo;
    }

    return py::bytes(reinterpret_cast<char*>(out.data()), out.size());
}

PYBIND11_MODULE(_compress_cpp, m) {
    init_map();
    m.def("compress_4bit", &compress_4bit_cpp);
}