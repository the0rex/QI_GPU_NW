from setuptools import setup, find_packages, Extension
import os
import platform

# -------------------------------------------------
# OpenMP flags
# -------------------------------------------------

def get_openmp_flags():
    system = platform.system().lower()

    if system == "darwin":  # macOS (clang)
        return {
            "compile": ["-std=c++11", "-O3", "-Xpreprocessor", "-fopenmp"],
            "link": ["-lomp"],
        }
    elif system == "windows":
        return {
            "compile": ["/std:c++17", "/O2", "/openmp"],
            "link": [],
        }
    else:  # Linux
        return {
            "compile": ["-std=c++11", "-O3", "-fopenmp"],
            "link": ["-fopenmp"],
        }

omp_flags = get_openmp_flags()

# Avoid -march=native for portable builds
if os.environ.get("ALIGNMENT_PIPELINE_PORTABLE", "0") != "1":
    if platform.system().lower() != "windows":
        omp_flags["compile"].append("-march=native")

# -------------------------------------------------
# pybind11 include handling
# -------------------------------------------------

def get_pybind_include():
    try:
        import pybind11
        return pybind11.get_include()
    except Exception:
        return "src/alignment_pipeline/cpp_extensions"

include_dirs = [
    "src/alignment_pipeline/cpp_extensions",
    get_pybind_include(),
]

# -------------------------------------------------
# C++ Extensions
# -------------------------------------------------

extensions = [
    Extension(
        name="alignment_pipeline.cpp_extensions._compress_cpp",
        sources=["src/alignment_pipeline/cpp_extensions/compress_4bit.cpp"],
        include_dirs=include_dirs,
        extra_compile_args=omp_flags["compile"],
        extra_link_args=omp_flags["link"],
        language="c++",
    ),
    Extension(
        name="alignment_pipeline.cpp_extensions._syncmer_cpp",
        sources=[
            "src/alignment_pipeline/cpp_extensions/syncmer_strobemer.cpp",
            "src/alignment_pipeline/cpp_extensions/pybind_module.cpp",
        ],
        include_dirs=include_dirs,
        extra_compile_args=omp_flags["compile"],
        extra_link_args=omp_flags["link"],
        language="c++",
    ),
]

# -------------------------------------------------
# Setup (metadata comes from pyproject.toml)
# -------------------------------------------------

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=extensions,
    include_package_data=True,
    zip_safe=False,
)
