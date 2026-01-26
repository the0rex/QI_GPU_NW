from .run_pipeline import main as run_pipeline_main
from .visualize_results import main as visualize_main
from .diagnostics import main as diagnostics_main
from .gpu_test import main as gpu_test_main
from .benchmark import main as benchmark_main
from .config_generator import main as config_generator_main

__version__ = "2.0.0"
__author__ = "Rowel Facunla"
__description__ = "Alignment Pipeline CLI Tools with GPU Acceleration"

__all__ = [
    'run_pipeline_main',
    'visualize_main',
    'diagnostics_main',
    'gpu_test_main',
    'benchmark_main',
    'config_generator_main',
]