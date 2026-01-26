"""
Input validation and system checks.
Author: Rowel Facunla
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Tuple, Dict, List

def validate_inputs(
    fasta1: str,
    fasta2: str,
    config: Dict
) -> Tuple[bool, List[str]]:
    """
    Validate all pipeline inputs.
    
    Args:
        fasta1: Path to first FASTA file
        fasta2: Path to second FASTA file
        config: Pipeline configuration
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check FASTA files
    from ..io.fasta_reader import validate_fasta_file
    
    valid1, msg1 = validate_fasta_file(fasta1)
    if not valid1:
        errors.append(f"FASTA file 1: {msg1}")
    
    valid2, msg2 = validate_fasta_file(fasta2)
    if not valid2:
        errors.append(f"FASTA file 2: {msg2}")
    
    # Check output directory
    output_dir = config.get('io', {}).get('output_dir', 'Results')
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create output directory {output_dir}: {e}")
    
    # Check disk space
    from ..io.file_handler import check_disk_space
    required_gb = config.get('validation', {}).get('required_disk_gb', 10.0)
    has_space, available_gb, required_gb = check_disk_space('.', required_gb)
    if not has_space:
        errors.append(f"Insufficient disk space: {available_gb:.1f} GB available, "
                     f"{required_gb:.1f} GB required")
    
    # Check memory requirements
    max_seq_length = config.get('validation', {}).get('max_sequence_length', 10000000)
    for fasta in [fasta1, fasta2]:
        from Bio import SeqIO
        try:
            records = list(SeqIO.parse(fasta, "fasta"))
            for record in records:
                if len(record.seq) > max_seq_length:
                    errors.append(f"Sequence too long: {record.id} has {len(record.seq)} bp, "
                                f"maximum allowed is {max_seq_length} bp")
        except Exception:
            pass  # Already caught by FASTA validation
    
    return len(errors) == 0, errors

def check_system_requirements(config: Dict) -> Tuple[bool, List[str]]:
    """
    Check system requirements.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Tuple of (meets_requirements, warnings)
    """
    warnings = []
    
    # Check Python version
    min_python = (3, 8)
    current_python = sys.version_info[:2]
    if current_python < min_python:
        warnings.append(f"Python {current_python[0]}.{current_python[1]} detected. "
                       f"Minimum required is {min_python[0]}.{min_python[1]}")
    
    # Check CPU cores
    import multiprocessing
    min_cores = config.get('validation', {}).get('min_cpu_cores', 2)
    available_cores = multiprocessing.cpu_count()
    if available_cores < min_cores:
        warnings.append(f"Only {available_cores} CPU cores available. "
                       f"Recommended minimum is {min_cores}")
    
    # Check memory
    import psutil
    min_memory_gb = config.get('validation', {}).get('min_memory_gb', 4.0)
    available_memory_gb = psutil.virtual_memory().total / (1024**3)
    if available_memory_gb < min_memory_gb:
        warnings.append(f"Only {available_memory_gb:.1f} GB RAM available. "
                       f"Recommended minimum is {min_memory_gb:.1f} GB")
    
    # Check for C++ compiler
    try:
        # Try to compile a simple C++ program
        test_cpp = """
        #include <iostream>
        int main() { std::cout << "test"; return 0; }
        """
        result = subprocess.run(
            ['c++', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            warnings.append("C++ compiler not found or not working")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        warnings.append("C++ compiler not found")
    
    return len(warnings) == 0, warnings

def validate_configuration(config: Dict) -> Tuple[bool, List[str]]:
    """
    Validate pipeline configuration.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required sections
    required_sections = ['io', 'alignment', 'chunking']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing configuration section: {section}")
    
    # Check alignment parameters
    if 'alignment' in config:
        align_config = config['alignment']
        
        # Gap penalties should be negative
        if align_config.get('gap_open', 0) > 0:
            errors.append("gap_open should be negative")
        if align_config.get('gap_extend', 0) > 0:
            errors.append("gap_extend should be negative")
        
        # Beam width should be positive
        if align_config.get('beam_width', 0) <= 0:
            errors.append("beam_width should be positive")
    
    # Check chunking parameters
    if 'chunking' in config:
        chunk_config = config['chunking']
        
        # Chunk size should be reasonable
        chunk_size = chunk_config.get('default_chunk_size', 5000)
        if chunk_size < 100 or chunk_size > 100000:
            errors.append(f"default_chunk_size {chunk_size} is outside reasonable range (100-100000)")
        
        # Overlap should be less than chunk size
        overlap = chunk_config.get('overlap', 250)
        if overlap >= chunk_size:
            errors.append(f"overlap {overlap} should be less than chunk_size {chunk_size}")
    
    # Check seeding parameters
    if 'seeding' in config:
        seed_config = config['seeding']
        
        # k should be positive
        if seed_config.get('kmer_size', 0) <= 0:
            errors.append("kmer_size should be positive")
        
        # s should be less than k
        k = seed_config.get('kmer_size', 21)
        s = seed_config.get('syncmer_size', 5)
        if s >= k:
            errors.append(f"syncmer_size {s} should be less than kmer_size {k}")
    
    return len(errors) == 0, errors

def check_sequence_compatibility(seq1_path: str, seq2_path: str) -> Tuple[bool, str]:
    """
    Check if sequences are compatible for alignment.
    
    Args:
        seq1_path: Path to first sequence
        seq2_path: Path to second sequence
        
    Returns:
        Tuple of (is_compatible, message)
    """
    from Bio import SeqIO
    
    try:
        records1 = list(SeqIO.parse(seq1_path, "fasta"))
        records2 = list(SeqIO.parse(seq2_path, "fasta"))
        
        if len(records1) != 1 or len(records2) != 1:
            return False, "Each FASTA file should contain exactly one sequence"
        
        seq1 = str(records1[0].seq)
        seq2 = str(records2[0].seq)
        
        # Check lengths
        len1 = len(seq1)
        len2 = len(seq2)
        
        if len1 == 0 or len2 == 0:
            return False, "Sequences cannot be empty"
        
        # Check for reasonable length ratio
        length_ratio = max(len1, len2) / min(len1, len2)
        if length_ratio > 100:
            return False, f"Sequence length ratio too large ({length_ratio:.1f}x)"
        
        # Check GC content difference
        from ..algorithms.predict_window import compute_gc_content
        gc1 = compute_gc_content(seq1)
        gc2 = compute_gc_content(seq2)
        gc_diff = abs(gc1 - gc2)
        
        if gc_diff > 50:
            return False, f"GC content difference too large ({gc_diff:.1f}%)"
        
        return True, f"Sequences compatible: {len1:,} bp vs {len2:,} bp, GC: {gc1:.1f}% vs {gc2:.1f}%"
    
    except Exception as e:
        return False, f"Error checking sequence compatibility: {str(e)}"

__all__ = [
    'validate_inputs',
    'check_system_requirements',
    'validate_configuration',
    'check_sequence_compatibility',
]