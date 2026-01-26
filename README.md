Sequence Alignment Pipeline
============================

A high-performance, quantum-inspired sequence alignment pipeline with GPU acceleration and chunked processing.

Author: Rowel Facunla
Version: 2.0.0
Status: Production Ready with GPU Acceleration

TABLE OF CONTENTS
=================
1. Key Features
2. Quick Start
3. Installation
4. Usage Examples
5. Configuration
6. System Requirements
7. Testing
8. Docker Deployment
9. Development
10. Visualization
11. Real-World Examples
12. Citing This Work
13. Support
14. Changelog
15. License

1. KEY FEATURES
===============

1.1 Performance Acceleration
----------------------------
- GPU/CUDA Acceleration: Optional PyTorch-based GPU acceleration
- C++ Extensions: High-performance C++ extensions for compression and seeding
- Parallel Processing: Multi-core CPU and multi-GPU support
- Quantum-inspired Algorithms: Ready for future quantum computing integration

1.2 Advanced Alignment Capabilities
-----------------------------------
- Global & Local Alignment: Needleman-Wunsch with affine gap penalties
- Chromosome-to-Chromosome Alignment: Align specific chromosomes across genomes
- Combined Chromosome Alignment: Align multiple chromosomes (e.g., chr1+chr2 vs chr1+chr2+chr3)
- Chunked Processing: Handle ultra-long sequences with intelligent chunking
- Anchor-based Chaining: Minimap2-style anchor chaining for accurate alignment
- IUPAC Support: Full support for ambiguous nucleotide codes

1.3 Technical Features
----------------------
- 4-bit Compression: Efficient memory usage for large sequences
- Strobemer Seeding: Syncmer-based seeding for fast anchor detection
- Beam Search: Configurable beam width for efficient alignment
- Window Prediction: Neural network-based optimal window size prediction
- Chromosome Discovery: List and filter chromosomes in FASTA files

1.4 Output Formats
------------------
- CIGAR Strings: Standard alignment format
- PAF Format: Pairwise mApping Format
- MAF Format: Multiple Alignment Format
- Visualizations: Interactive alignment visualizations
- JSON Reports: Comprehensive alignment statistics

2. QUICK START
==============

2.1 Basic Installation (CPU-only)
---------------------------------
# Clone repository
git clone https://github.com/the0rex/alignment-pipeline.git
cd alignment-pipeline

# Install with core features
pip install -e .

# Or install with all features
pip install -e .[full]

2.2 GPU-accelerated Installation
--------------------------------
# Install with CUDA support (requires NVIDIA GPU)
pip install -e .[cuda]

# Or install everything including GPU
pip install -e .[full,cuda]

# Check GPU availability
align-gpu-test

2.3 Using Docker
----------------
# CPU-only container
docker build -t alignment-pipeline:cpu .
docker run --rm alignment-pipeline:cpu --help

# GPU-accelerated container
docker build --build-arg USE_CUDA=true -t alignment-pipeline:gpu .
docker run --rm --gpus all alignment-pipeline:gpu --help

# Using docker-compose
docker-compose up -d alignment-gpu

3. INSTALLATION DETAILS
=======================

3.1 Dependencies
----------------
Core dependencies (installed by default):
- numpy>=1.21.0
- biopython>=1.79
- pandas>=1.3.0
- scikit-learn>=1.0.0
- matplotlib>=3.5.0
- pybind11>=2.10.0
- tqdm>=4.64.0
- pyyaml>=6.0
- colorama>=0.4.6
- psutil>=5.9.0
- keras>=3.9.2
- tensorflow==2.16.1
- seaborn>=0.11.0
- plotly>=5.5.0

Optional GPU dependencies:
- torch>=2.0.0
- torchvision>=0.15.0
- torchaudio>=2.0.0

3.2 Building from Source
------------------------
# Install build dependencies
pip install setuptools wheel pybind11

# Build C++ extensions
python setup.py build_ext --inplace

# Install the package
pip install -e .

4. USAGE EXAMPLES
=================

4.1 Basic Alignment
-------------------
# Run with default configuration
align-pipeline --fasta1 examples/test_fasta_1.fa --fasta2 examples/test_fasta_2.fa

# Run with custom parameters
align-pipeline --fasta1 seq1.fa --fasta2 seq2.fa \
    --gap-open -30 --gap-extend -0.5 \
    --chunk-size 5000 --workers 4 \
    --output-dir my_results

4.2 Chromosome Alignment Features
---------------------------------

4.2.1 List Chromosomes in FASTA Files
-------------------------------------
# List all chromosomes in a FASTA file
align-pipeline --list-chromosomes GRCh38.fasta

# List chromosomes in a specific range (inclusive)
align-pipeline --list-chromosomes GRCh38.fasta --chromosome-range "1:10"
align-pipeline --list-chromosomes GRCh38.fasta --chromosome-range "chr1:chr10"
align-pipeline --list-chromosomes GRCh38.fasta --chromosome-range "X:Y"

# List with limit
align-pipeline --list-chromosomes Pan_tro_3.0.fasta --list-limit 20

4.2.2 Align Specific Chromosomes
--------------------------------
# Align single chromosomes
align-pipeline --fasta1 GRCh38.fasta --chrom1 chr1 --fasta2 Pan_tro_3.0.fasta --chrom2 chr2

# Align with numeric chromosome names
align-pipeline --fasta1 GRCh38.fasta --chrom1 1 --fasta2 Pan_tro_3.0.fasta --chrom2 2

# Align sex chromosomes
align-pipeline --fasta1 GRCh38.fasta --chrom1 X --fasta2 Pan_tro_3.0.fasta --chrom2 X

4.2.3 Align Combined Chromosomes
--------------------------------
# Align chromosome 1+2 from human vs chromosome 1+2+3 from chimpanzee
align-pipeline --fasta1 GRCh38.fasta --chrom1 "chr1+chr2" --fasta2 Pan_tro_3.0.fasta --chrom2 "chr1+chr2+chr3"

# Align multiple sex chromosomes
align-pipeline --fasta1 GRCh38.fasta --chrom1 "X+Y" --fasta2 Pan_tro_3.0.fasta --chrom2 "X"

# Complex combinations
align-pipeline --fasta1 GRCh38.fasta --chrom1 "chr1+chr2+X" --fasta2 Pan_tro_3.0.fasta --chrom2 "chr1+Y"

4.3 GPU-accelerated Alignment
-----------------------------
# Run with GPU acceleration
align-pipeline --config config/pipeline_config_gpu.yaml \
    --fasta1 large_seq1.fa --fasta2 large_seq2.fa

# Force CPU-only mode
align-pipeline --config config/pipeline_config.yaml \
    --fasta1 seq1.fa --fasta2 seq2.fa

4.4 Advanced Features
---------------------
# Generate visualizations
align-viz --input results/alignment.json --output-dir visualizations/

# Run diagnostics
align-diag --check-all

# Run benchmarks
align-benchmark --test-data examples/ --iterations 10

5. CONFIGURATION
================

5.1 Configuration Files
-----------------------
The pipeline supports YAML configuration files. Example configuration:

# config/pipeline_config_gpu.yaml
pipeline:
  name: "GPU Alignment Pipeline"
  version: "2.0.0"

io:
  fasta_dir: "fasta_sequences/"
  default_fasta1: "GRCh38.fasta"
  default_fasta2: "Pan_tro_3.0.fasta"
  temp_dir: "Data/temp_chromosomes/"
  chromosome_separator: ""

alignment:
  gap_open: -30
  gap_extend: -0.5
  window_size_predictor: "gpu"
  use_gpu_acceleration: true
  beam_width: 100

performance:
  use_gpu: true
  gpu_device_id: 0
  gpu_batch_size: 32
  num_workers: "auto"

chunking:
  default_chunk_size: 5000
  use_gpu_anchoring: true
  use_gpu_pruning: true

5.2 Environment Variables
-------------------------
# Enable GPU acceleration
export USE_CUDA=true
export CUDA_VISIBLE_DEVICES=0

# Performance tuning
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Memory management
export TF_GPU_ALLOCATOR=cuda_malloc_async
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Chromosome alignment
export ALIGNMENT_TEMP_DIR="./temp_chromosomes"

6. SYSTEM REQUIREMENTS
======================

6.1 GPU Requirements
--------------------
- NVIDIA GPU: Compute Capability 3.5 or higher
- CUDA Toolkit: 11.8 or higher
- Driver: NVIDIA driver 450.80.02 or higher
- Memory: 4GB+ VRAM recommended for chromosome alignment
- Storage: 20GB+ free space for large genome alignments

6.2 CPU Requirements
--------------------
- Processor: Multi-core x86-64 CPU (Intel/AMD)
- Memory: 8GB+ RAM (16GB+ recommended for whole genomes)
- Storage: SSD recommended for fast I/O during chunk processing

6.3 Software Requirements
-------------------------
- Python: 3.8 or higher
- Operating System: Linux, macOS, or Windows (WSL2 recommended for Windows)

7. TESTING
==========

7.1 Running Tests
-----------------
# Run all tests
pytest tests/ -v

# Run CPU-specific tests
pytest tests/test_alignment.py tests/test_chunking.py -v

# Run GPU tests (requires GPU)
pytest tests/test_cuda.py -v

# Run chromosome alignment tests
pytest tests/test_chromosome_alignment.py -v

# Run with coverage
pytest --cov=alignment_pipeline tests/ --cov-report=html

7.2 Test Categories
-------------------
- Unit Tests: Individual component testing
- Integration Tests: Pipeline integration testing
- GPU Tests: CUDA/GPU functionality testing
- Chromosome Tests: Chromosome alignment and listing
- Performance Tests: Benchmarking and profiling
- Edge Cases: Boundary condition testing

8. DOCKER DEPLOYMENT
====================

8.1 Building Images
-------------------
# Build CPU image
docker build -t alignment-pipeline:cpu .

# Build GPU image
docker build --build-arg USE_CUDA=true -t alignment-pipeline:gpu .

# Build development image
docker build -f Dockerfile.dev -t alignment-pipeline:dev .

8.2 Running Containers
----------------------
# CPU container
docker run --rm -v $(pwd)/data:/data alignment-pipeline:cpu \
    --fasta1 /data/seq1.fa --fasta2 /data/seq2.fa

# GPU container with chromosome alignment
docker run --rm --gpus all -v $(pwd)/data:/data alignment-pipeline:gpu \
    --fasta1 /data/GRCh38.fasta --chrom1 "1:10" \
    --fasta2 /data/Pan_tro_3.0.fasta --chrom2 "1:10"

# Development container
docker run --rm -it -v $(pwd)/src:/app/src alignment-pipeline:dev bash

8.3 Docker Compose
------------------
# docker-compose.yml
version: '3.8'
services:
  alignment-gpu:
    build:
      context: .
      args:
        USE_CUDA: "true"
    runtime: nvidia
    volumes:
      - ./data:/data:ro
      - ./results:/results
      - ./temp_chromosomes:/app/Data/temp_chromosomes
    environment:
      - USE_CUDA=true
      - CUDA_VISIBLE_DEVICES=0

9. DEVELOPMENT
==============

9.1 Setting Up Development Environment
--------------------------------------
# Clone and install development dependencies
git clone https://github.com/the0rex/alignment-pipeline.git
cd alignment-pipeline
pip install -e .[dev,full,cuda]

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

9.2 Building C++ Extensions
---------------------------
# Build extensions
python setup.py build_ext --inplace

# Clean build
python setup.py clean --all

# Build with specific flags
CFLAGS="-O3 -march=native" python setup.py build_ext --inplace

9.3 Code Quality Tools
----------------------
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Run all checks
pre-commit run --all-files

10. VISUALIZATION
=================

10.1 Generating Visualizations
------------------------------
# Python API example
from alignment_pipeline.visualization import visualize_alignment
from alignment_pipeline.io.fasta_reader import list_chromosomes

# List chromosomes in a FASTA file
chromosomes = list_chromosomes("GRCh38.fasta", range_spec="1:10")
print(f"Chromosomes 1-10: {chromosomes}")

# Create alignment visualization
visualize_alignment(
    seq1="ACGTACGT",
    seq2="ACGTTACGT",
    output_file="alignment.png",
    title="Example Alignment"
)

10.2 Available Visualizations
-----------------------------
1. Alignment Matrix: Dot plot of sequence similarity
2. Score Distribution: Histogram of alignment scores
3. Gap Distribution: Visualization of gap patterns
4. Memory Usage: Runtime memory consumption
5. GPU Utilization: GPU memory and compute usage
6. Chromosome Coverage: Alignment coverage across chromosomes

11. REAL-WORLD EXAMPLES
=======================

11.1 Whole Genome Alignment
---------------------------
# Align human vs chimpanzee chromosomes 1-22
for i in {1..22}; do
    align-pipeline --fasta1 GRCh38.fasta --chrom1 "chr$i" \
                   --fasta2 Pan_tro_3.0.fasta --chrom2 "chr$i" \
                   --output-dir "results/chr$i" --workers 8
done

# Combine results
align-pipeline --combine-results results/ --output whole_genome_alignment.json

11.2 Comparative Genomics Study
-------------------------------
# Align specific gene regions
align-pipeline --fasta1 human_genome.fa --chrom1 "chr1:1000000-2000000" \
               --fasta2 mouse_genome.fa --chrom2 "chr1:1500000-2500000" \
               --output-dir gene_comparison

# Generate comparative report
align-viz --input gene_comparison/alignment.json \
          --output-dir gene_comparison/visualizations \
          --title "Human-Mouse Gene Comparison"

11.3 Quality Control Pipeline
-----------------------------
# Run diagnostics
align-diag --check-all --fasta1 sample1.fasta --fasta2 sample2.fasta

# Generate quality report
align-pipeline --fasta1 sample1.fasta --fasta2 sample2.fasta \
               --config config/qc_config.yaml \
               --output-dir qc_results

12. CITING THIS WORK
====================
If you use this pipeline in your research, please cite:

@software{alignment_pipeline_2024,
  author = {Facunla, Rowel},
  title = {Alignment Pipeline: A High-Performance Quantum-Inspired Sequence Alignment Tool with GPU Acceleration},
  version = {2.0.0},
  year = {2024},
  url = {https://github.com/the0rex/alignment-pipeline},
  note = {Includes advanced chromosome alignment features}
}

13. SUPPORT
===========

13.1 Issue Tracking
-------------------
- GitHub Issues: https://github.com/the0rex/alignment-pipeline/issues
- Documentation: https://alignment-pipeline.readthedocs.io/

13.2 Community
--------------
- Discussions: GitHub Discussions
- Email: rowel.facunla@tip.edu.ph
- Twitter: @the0rex

14. CHANGELOG
=============

Version 2.0.0
-------------
- Added: Chromosome-to-chromosome alignment support
- Added: Combined chromosome alignment (e.g., chr1+chr2)
- Added: Chromosome listing and range filtering
- Enhanced: GPU acceleration for chromosome alignment
- Improved: Error handling and validation
- Updated: Comprehensive documentation

Version 1.5.0
-------------
- Added: Quantum-inspired scoring algorithms
- Enhanced: GPU acceleration with PyTorch
- Added: Neural network window prediction
- Improved: Memory management for large genomes

Version 1.0.0
-------------
- Initial release: Basic alignment pipeline
- Features: Global/local alignment, chunked processing
- Formats: CIGAR, PAF, MAF output support

15. LICENSE
===========
This project is licensed under the MIT License - see the LICENSE file for details.

===============================================================================
END OF DOCUMENT
===============================================================================