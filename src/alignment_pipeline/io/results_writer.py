"""
Results writing utilities for the alignment pipeline.
Author: Rowel Facunla
"""

import os
import gzip
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from ..core.utilities import (
    compute_alignment_stats,
    build_cigar,
    write_cigar_gz,
    write_paf,
    write_maf,
    write_summary
)

def base_set(ch):
    """Get the set of bases represented by an IUPAC character."""
    IUPAC_MAP = {
        'A': {'A'}, 'C': {'C'}, 'G': {'G'}, 'T': {'T'}, 'U': {'T'},
        'R': {'A', 'G'}, 'Y': {'C', 'T'}, 'S': {'G', 'C'}, 'W': {'A', 'T'},
        'K': {'G', 'T'}, 'M': {'A', 'C'}, 'B': {'C', 'G', 'T'}, 'D': {'A', 'G', 'T'},
        'H': {'A', 'C', 'T'}, 'V': {'A', 'C', 'G'}, 'N': {'A', 'C', 'G', 'T'}, '-': {'-'}
    }
    if ch is None:
        return set()
    return IUPAC_MAP.get(ch.upper(), IUPAC_MAP["N"])

def iupac_is_match(a, b):
    """Exact-match only. Ambiguous bases (N, R, Y, etc.) never count as a match."""
    if a is None or b is None:
        return False
    if a == '-' or b == '-':
        return False
    return a.upper() == b.upper()

def iupac_partial_overlap(a, b):
    """True when bases overlap but are not identical."""
    s1, s2 = base_set(a), base_set(b)
    if not s1 or not s2:
        return False
    inter = s1 & s2
    return bool(inter) and s1 != s2

def save_all_results(
    a1: str,
    a2: str,
    total_score: float,
    fasta1: str,
    fasta2: str,
    output_dir: str = "Results",
    base_name: Optional[str] = None,
    chrom1: Optional[str] = None,
    chrom2: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Save all alignment results in various formats.
    
    Args:
        a1: Aligned sequence 1
        a2: Aligned sequence 2
        total_score: Total alignment score
        fasta1: Path to first FASTA file
        fasta2: Path to second FASTA file
        output_dir: Base output directory
        base_name: Optional base name for output files
        chrom1: Optional chromosome name for sequence 1
        chrom2: Optional chromosome name for sequence 2
        config: Optional configuration dictionary
    
    Returns:
        Dictionary of saved file paths
    """
    # Create output directory structure
    output_path = Path(output_dir)
    
    # If base_name is provided, create subdirectory
    if base_name:
        output_path = output_path / base_name
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract base names from file paths
    name1 = Path(fasta1).stem
    name2 = Path(fasta2).stem
    
    # Clean up names
    for ext in ['.fa', '.fasta', '.fn', '.fna', '.fas']:
        name1 = name1.replace(ext, '')
        name2 = name2.replace(ext, '')
    
    # Use chromosome names if provided
    display_name1 = chrom1 if chrom1 else name1
    display_name2 = chrom2 if chrom2 else name2
    
    # Compute alignment statistics
    stats = compute_alignment_stats(a1, a2)
    cigar = build_cigar(a1, a2)
    
    # Generate file names
    if base_name:
        file_prefix = base_name
    else:
        file_prefix = f"{display_name1}_vs_{display_name2}"
    
    # Dictionary to store all saved file paths
    saved_files = {}
    
    # Save individual result files
    cigar_path = output_path / f"{file_prefix}.cigar.gz"
    paf_path = output_path / f"{file_prefix}.paf"
    maf_path = output_path / f"{file_prefix}.maf"
    summary_path = output_path / f"{file_prefix}_summary.txt"
    json_path = output_path / f"{file_prefix}_results.json"
    csv_path = output_path / f"{file_prefix}_stats.csv"
    
    # Save compressed CIGAR
    write_cigar_gz(str(cigar_path), cigar)
    saved_files['cigar'] = str(cigar_path)
    
    # Save PAF
    write_paf(str(paf_path), a1, a2, cigar, stats, display_name1, display_name2)
    saved_files['paf'] = str(paf_path)
    
    # Save MAF
    write_maf(str(maf_path), a1, a2, display_name1, display_name2)
    saved_files['maf'] = str(maf_path)
    
    # Save summary
    write_summary(str(summary_path), stats, cigar, total_score)
    saved_files['summary'] = str(summary_path)
    
    # Save JSON with all results
    json_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'fasta1': fasta1,
            'fasta2': fasta2,
            'sequence1_name': display_name1,
            'sequence2_name': display_name2,
            'chromosome1': chrom1,
            'chromosome2': chrom2,
            'total_score': total_score,
            'base_name': base_name,
        },
        'statistics': stats,
        'alignment': {
            'sequence1': a1,
            'sequence2': a2,
            'cigar': cigar,
            'length': len(a1),
        }
    }
    
    # Add configuration if provided
    if config:
        json_results['configuration'] = {
            k: v for k, v in config.items() 
            if k not in ['_source'] and not k.startswith('__')
        }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    saved_files['json'] = str(json_path)
    
    # Save CSV statistics
    save_stats_csv(str(csv_path), stats, total_score, display_name1, display_name2, chrom1, chrom2)
    saved_files['csv'] = str(csv_path)
    
    # Save sequences separately
    seq1_path = output_path / f"{display_name1}_aligned.fa"
    seq2_path = output_path / f"{display_name2}_aligned.fa"
    
    with open(seq1_path, 'w') as f:
        f.write(f">{display_name1}\n{a1.replace('-', '')}\n")
    
    with open(seq2_path, 'w') as f:
        f.write(f">{display_name2}\n{a2.replace('-', '')}\n")
    
    saved_files['seq1'] = str(seq1_path)
    saved_files['seq2'] = str(seq2_path)
    
    # Save aligned sequences (with gaps)
    aligned1_path = output_path / f"{display_name1}_with_gaps.fa"
    aligned2_path = output_path / f"{display_name2}_with_gaps.fa"
    
    with open(aligned1_path, 'w') as f:
        f.write(f">{display_name1} (aligned with gaps)\n{a1}\n")
    
    with open(aligned2_path, 'w') as f:
        f.write(f">{display_name2} (aligned with gaps)\n{a2}\n")
    
    saved_files['aligned_seq1'] = str(aligned1_path)
    saved_files['aligned_seq2'] = str(aligned2_path)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SAVED:")
    print("="*60)
    for key, path in saved_files.items():
        print(f" → {Path(path).name}")
    print("="*60)
    
    # Print detailed statistics - using .get() to handle missing keys
    print(f"\nAlignment Statistics:")
    print(f"  Sequences: {display_name1} vs {display_name2}")
    if chrom1 or chrom2:
        print(f"  Chromosomes: {chrom1 if chrom1 else 'N/A'} vs {chrom2 if chrom2 else 'N/A'}")
    print(f"  Total score: {total_score:.2f}")
    print(f"  Alignment length: {stats.get('total_aligned', 0):,} bp")
    print(f"  Identity: {stats.get('identity', 0)*100:.2f}%")
    print(f"  Matches: {stats.get('matches', 0):,}")
    print(f"  Mismatches: {stats.get('mismatches', 0):,}")
    print(f"  Insertions: {stats.get('insertions', 0):,}")
    print(f"  Deletions: {stats.get('deletions', 0):,}")
    
    # Handle gap statistics safely
    gap_openings = stats.get('gap_openings')
    if gap_openings is not None:
        print(f"  Gap openings: {gap_openings:,}")
    
    total_gaps = stats.get('total_gaps')
    if total_gaps is not None:
        print(f"  Total gaps: {total_gaps:,}")
    
    gap_percentage = stats.get('gap_percentage')
    if gap_percentage is not None:
        print(f"  Gap percentage: {gap_percentage:.2f}%")
    
    # Save list of files
    files_list_path = output_path / "file_manifest.txt"
    with open(files_list_path, 'w') as f:
        f.write("Alignment Pipeline Results Manifest\n")
        f.write("="*40 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Alignment: {display_name1} vs {display_name2}\n")
        if chrom1 or chrom2:
            f.write(f"Chromosomes: {chrom1 if chrom1 else 'N/A'} vs {chrom2 if chrom2 else 'N/A'}\n")
        f.write("="*40 + "\n\n")
        for key, path in saved_files.items():
            f.write(f"{key}: {Path(path).name}\n")
    
    saved_files['manifest'] = str(files_list_path)
    
    return saved_files

def save_stats_csv(
    csv_path: str,
    stats: Dict[str, Any],
    total_score: float,
    name1: str,
    name2: str,
    chrom1: Optional[str] = None,
    chrom2: Optional[str] = None
):
    """Save alignment statistics to CSV file."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['Metric', 'Value', 'Description'])
        writer.writerow([''] * 3)  # Empty row
        
        # Write metadata
        writer.writerow(['Sequence 1', name1, ''])
        writer.writerow(['Sequence 2', name2, ''])
        if chrom1:
            writer.writerow(['Chromosome 1', chrom1, ''])
        if chrom2:
            writer.writerow(['Chromosome 2', chrom2, ''])
        writer.writerow(['Total Score', f"{total_score:.2f}", 'Overall alignment score'])
        writer.writerow([''] * 3)  # Empty row
        
        # Write statistics - using .get() with defaults for missing keys
        writer.writerow(['Alignment Length', stats.get('total_aligned', 0), 'Total aligned positions'])
        writer.writerow(['Identity', f"{stats.get('identity', 0)*100:.2f}%", 'Percentage of identical bases'])
        writer.writerow(['Matches', stats.get('matches', 0), 'Number of matching bases'])
        writer.writerow(['Mismatches', stats.get('mismatches', 0), 'Number of mismatching bases'])
        writer.writerow(['Insertions', stats.get('insertions', 0), 'Insertions in sequence 1'])
        writer.writerow(['Deletions', stats.get('deletions', 0), 'Deletions in sequence 1'])
        
        # Only write gap statistics if they exist
        if 'gap_openings' in stats:
            writer.writerow(['Gap Openings', stats.get('gap_openings', 0), 'Number of gap openings'])
        if 'total_gaps' in stats:
            writer.writerow(['Total Gaps', stats.get('total_gaps', 0), 'Total gap characters'])
        if 'gap_percentage' in stats:
            writer.writerow(['Gap Percentage', f"{stats.get('gap_percentage', 0):.2f}%", 'Percentage of gaps in alignment'])

def save_chunk_results_metadata(
    chunk_files: list, 
    output_dir: str = "Results",
    base_name: Optional[str] = None
) -> str:
    """
    Save metadata about chunk processing results.
    """
    output_path = Path(output_dir)
    if base_name:
        output_path = output_path / base_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'num_chunks': len(chunk_files),
        'chunk_files': chunk_files,
        'chunk_count': len(chunk_files),
        'timestamp': datetime.now().isoformat(),
    }
    
    metadata_path = output_path / "chunk_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return str(metadata_path)

def save_performance_report(
    report: dict, 
    output_dir: str = "Results",
    base_name: Optional[str] = None
) -> str:
    """
    Save performance report.
    """
    output_path = Path(output_dir)
    if base_name:
        output_path = output_path / base_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_path = output_path / "performance_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also save as text summary
    summary_path = output_path / "performance_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Performance Report\n")
        f.write("="*40 + "\n")
        f.write(f"Total time: {report.get('total_time_seconds', 0):.2f} seconds\n")
        f.write(f"Peak memory: {report.get('peak_memory_mb', 0):.1f} MB\n")
        f.write(f"CPU usage: {report.get('avg_cpu_percent', 0):.1f}%\n")
        
        if 'gpu_info' in report:
            gpu_info = report['gpu_info']
            f.write("\nGPU Information:\n")
            f.write(f"  Device: {gpu_info.get('device_name', 'Unknown')}\n")
            if 'max_memory_used_gb' in gpu_info:
                f.write(f"  Memory used: {gpu_info['max_memory_used_gb']:.2f} GB\n")
            if 'memory_utilization' in gpu_info:
                f.write(f"  Memory utilization: {gpu_info['memory_utilization']:.1f}%\n")
    
    return str(report_path)

def save_configuration(
    config: dict, 
    output_dir: str = "Results",
    base_name: Optional[str] = None
) -> str:
    """
    Save pipeline configuration.
    """
    output_path = Path(output_dir)
    if base_name:
        output_path = output_path / base_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Remove large or non-serializable objects
    config_copy = {}
    for key, value in config.items():
        if key not in ['_source'] and not key.startswith('__'):
            # Try to serialize, skip if not serializable
            try:
                json.dumps(value)
                config_copy[key] = value
            except:
                config_copy[key] = str(type(value))
    
    config_path = output_path / "pipeline_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_copy, f, indent=2)
    
    return str(config_path)

def create_results_archive(
    output_dir: str = "Results",
    base_name: Optional[str] = None
) -> str:
    """
    Create a compressed archive of all results.
    
    Returns:
        Path to archive file
    """
    import tarfile
    
    output_path = Path(output_dir)
    if base_name:
        output_path = output_path / base_name
    
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = base_name if base_name else "alignment_results"
    archive_path = output_path.parent / f"{archive_name}_{timestamp}.tar.gz"
    
    with tarfile.open(archive_path, "w:gz") as tar:
        for file in output_path.iterdir():
            if file.is_file() and file.suffix not in ['.tar', '.gz']:
                tar.add(file, arcname=file.name)
    
    print(f"Created archive: {archive_path}")
    return str(archive_path)

def save_alignment_snippet(
    a1: str,
    a2: str,
    comp: str,
    output_dir: str = "Results",
    base_name: Optional[str] = None,
    snippet_length: int = 100
) -> str:
    """
    Save a snippet of the alignment for quick inspection.
    """
    output_path = Path(output_dir)
    if base_name:
        output_path = output_path / base_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    snippet_path = output_path / "alignment_snippet.txt"
    
    with open(snippet_path, 'w') as f:
        f.write("Alignment Snippet (first {} columns)\n".format(snippet_length))
        f.write("="*60 + "\n\n")
        
        # Write sequences in blocks of 50
        for i in range(0, min(len(a1), snippet_length), 50):
            end = min(i + 50, snippet_length)
            
            f.write(f"Seq1 [{i+1:4d}-{end:4d}]: {a1[i:end]}\n")
            f.write(f"Comp [{i+1:4d}-{end:4d}]: {comp[i:end]}\n")
            f.write(f"Seq2 [{i+1:4d}-{end:4d}]: {a2[i:end]}\n")
            f.write("\n")
    
    return str(snippet_path)

def save_visualization_data(
    a1: str,
    a2: str,
    stats: Dict[str, Any],
    output_dir: str = "Results",
    base_name: Optional[str] = None
) -> str:
    """
    Save data for visualization tools.
    """
    output_path = Path(output_dir)
    if base_name:
        output_path = output_path / base_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    viz_data_path = output_path / "visualization_data.json"
    
    # Create data suitable for visualization
    viz_data = {
        'sequences': {
            'seq1': a1,
            'seq2': a2,
            'length': len(a1)
        },
        'statistics': stats,
        'matches': [],
        'mismatches': [],
        'gaps': []
    }
    
    # Find match/mismatch/gap positions
    for i in range(len(a1)):
        if a1[i] == a2[i] and a1[i] != '-':
            viz_data['matches'].append(i)
        elif a1[i] != '-' and a2[i] != '-':
            viz_data['mismatches'].append(i)
        elif a1[i] == '-' or a2[i] == '-':
            viz_data['gaps'].append(i)
    
    with open(viz_data_path, 'w') as f:
        json.dump(viz_data, f, indent=2)
    
    return str(viz_data_path)

__all__ = [
    'base_set',
    'iupac_is_match',
    'iupac_partial_overlap',
    'save_all_results',
    'save_stats_csv',
    'save_chunk_results_metadata',
    'save_performance_report',
    'save_configuration',
    'create_results_archive',
    'save_alignment_snippet',
    'save_visualization_data',
]