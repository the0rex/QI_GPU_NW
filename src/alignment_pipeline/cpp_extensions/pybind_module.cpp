#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "syncmer_strobemer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_syncmer_cpp, m) {
    py::class_<Strobemer>(m, "Strobemer")
        .def_readonly("pos1", &Strobemer::pos1)
        .def_readonly("pos2", &Strobemer::pos2)
        .def_readonly("hash", &Strobemer::hash)
        .def_readonly("span", &Strobemer::span)
        .def_readonly("length", &Strobemer::length);

    m.def(
        "strobes_from_4bit_buffer",
        [](py::bytes buf,
           uint32_t L,
           uint32_t k,
           uint32_t s,
           uint32_t sync_pos,
           uint32_t w_min,
           uint32_t w_max)
        {
            std::string data = buf;  // zero-copy view
            const uint8_t* raw = reinterpret_cast<const uint8_t*>(data.data());

            return strobes_from_4bit_buffer(
                raw, L, k, s, sync_pos, w_min, w_max
            );
        },
        py::arg("buf"),
        py::arg("L"),
        py::arg("k") = 21,
        py::arg("s") = 5,
        py::arg("sync_pos") = 2,
        py::arg("w_min") = 20,
        py::arg("w_max") = 70
    );
}
