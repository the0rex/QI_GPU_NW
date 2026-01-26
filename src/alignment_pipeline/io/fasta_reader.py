"""
Author: Rowel Facunla
"""

import os
from Bio import SeqIO
from pathlib import Path
from typing import Tuple, Optional, Dict, List

def validate_fasta_file(filepath: str) -> Tuple[bool, str]:
    """
    Validate if a file is a valid FASTA file.
    
    Args:
        filepath: Path to the FASTA file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(filepath):
        return False, f"File does not exist: {filepath}"
    
    if not os.path.isfile(filepath):
        return False, f"Not a file: {filepath}"
    
    if os.path.getsize(filepath) == 0:
        return False, f"File is empty: {filepath}"
    
    # Check if it looks like a FASTA file
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            if not first_line.startswith('>'):
                return False, f"File does not start with '>' character: {filepath}"
    except UnicodeDecodeError:
        return False, f"File is not a valid text file: {filepath}"
    
    # Try to parse with BioPython
    try:
        records = list(SeqIO.parse(filepath, "fasta"))
        if len(records) == 0:
            return False, f"No sequences found in file: {filepath}"
        
        # Check sequence lengths
        for i, record in enumerate(records):
            if len(record.seq) == 0:
                return False, f"Sequence {i+1} is empty in file: {filepath}"
            
            # Check for invalid characters
            seq_str = str(record.seq).upper()
            valid_chars = set('ACGTURYSWKMBDHVN-')
            invalid_chars = set(seq_str) - valid_chars
            if invalid_chars:
                return False, f"Sequence {i+1} contains invalid characters: {invalid_chars}"
        
        return True, f"Valid FASTA file with {len(records)} sequence(s)"
        
    except Exception as e:
        return False, f"Error parsing FASTA file: {str(e)}"

def read_fasta_sequence(filepath: str, sequence_index: int = 0) -> Tuple[str, str]:
    """
    Read a specific sequence from a FASTA file.
    
    Args:
        filepath: Path to FASTA file
        sequence_index: Index of sequence to read (0-based)
        
    Returns:
        Tuple of (sequence_id, sequence)
    """
    records = list(SeqIO.parse(filepath, "fasta"))
    
    if sequence_index >= len(records):
        raise IndexError(f"Sequence index {sequence_index} out of bounds. "
                        f"File contains {len(records)} sequences.")
    
    record = records[sequence_index]
    return record.id, str(record.seq)

def read_fasta_file(filepath: str) -> Dict[str, str]:
    """
    Read all sequences from a FASTA file.
    
    Args:
        filepath: Path to FASTA file
        
    Returns:
        Dictionary with sequence IDs as keys and sequences as values
    """
    try:
        records = list(SeqIO.parse(filepath, "fasta"))
        if not records:
            raise ValueError(f"No sequences found in file: {filepath}")
        
        sequences = {}
        for record in records:
            sequences[record.id] = str(record.seq)
        
        return sequences
        
    except Exception as e:
        raise ValueError(f"Error reading FASTA file {filepath}: {str(e)}")

def list_fasta_sequences(filepath: str, max_sequences: int = 50) -> List[str]:
    """
    List all sequence IDs in a FASTA file.
    
    Args:
        filepath: Path to FASTA file
        max_sequences: Maximum number of sequences to list (0 for all)
        
    Returns:
        List of sequence IDs
    """
    try:
        records = list(SeqIO.parse(filepath, "fasta"))
        sequence_ids = [record.id for record in records]
        
        # Sort naturally (chr1, chr2, ..., chr10, etc.)
        def natural_sort_key(name):
            import re
            parts = re.split(r'(\d+)', name)
            return [int(part) if part.isdigit() else part.lower() for part in parts]
        
        sequence_ids.sort(key=natural_sort_key)
        
        if max_sequences > 0 and len(sequence_ids) > max_sequences:
            return sequence_ids[:max_sequences]
        return sequence_ids
        
    except Exception as e:
        raise ValueError(f"Error listing sequences in FASTA file {filepath}: {str(e)}")

def get_fasta_stats(filepath: str) -> dict:
    """
    Get statistics about a FASTA file.
    
    Args:
        filepath: Path to FASTA file
        
    Returns:
        Dictionary with file statistics
    """
    stats = {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'size_bytes': os.path.getsize(filepath),
        'num_sequences': 0,
        'total_length': 0,
        'min_length': float('inf'),
        'max_length': 0,
        'avg_length': 0,
        'gc_content': 0,
        'sequence_ids': [],
    }
    
    try:
        records = list(SeqIO.parse(filepath, "fasta"))
        stats['num_sequences'] = len(records)
        stats['sequence_ids'] = [record.id for record in records]
        
        if records:
            total_gc = 0
            total_bases = 0
            
            for record in records:
                seq = str(record.seq).upper()
                length = len(seq)
                
                stats['total_length'] += length
                stats['min_length'] = min(stats['min_length'], length)
                stats['max_length'] = max(stats['max_length'], length)
                
                # Calculate GC content for this sequence
                gc = seq.count('G') + seq.count('C')
                total_gc += gc
                total_bases += length
            
            stats['avg_length'] = stats['total_length'] / len(records)
            stats['gc_content'] = (total_gc / total_bases * 100) if total_bases > 0 else 0
            
    except Exception as e:
        stats['error'] = str(e)
    
    # Format for readability
    stats['size_mb'] = stats['size_bytes'] / (1024 * 1024)
    stats['size_gb'] = stats['size_bytes'] / (1024 * 1024 * 1024)
    
    return stats

def print_fasta_stats(filepath: str):
    """
    Print statistics about a FASTA file.
    """
    stats = get_fasta_stats(filepath)
    
    print(f"\nFASTA File Statistics:")
    print(f"  File: {stats['filename']}")
    print(f"  Size: {stats['size_mb']:.2f} MB ({stats['size_gb']:.2f} GB)")
    print(f"  Sequences: {stats['num_sequences']}")
    print(f"  Total length: {stats['total_length']:,} bp")
    
    if stats['num_sequences'] > 0:
        print(f"  Min length: {stats['min_length']:,} bp")
        print(f"  Max length: {stats['max_length']:,} bp")
        print(f"  Avg length: {stats['avg_length']:,.0f} bp")
        print(f"  GC content: {stats['gc_content']:.1f}%")
        
        # Show first few sequence IDs
        if stats['sequence_ids']:
            print(f"\n  Sequence IDs (first 10):")
            for i, seq_id in enumerate(stats['sequence_ids'][:10]):
                print(f"    {i+1}. {seq_id}")
            if len(stats['sequence_ids']) > 10:
                print(f"    ... and {len(stats['sequence_ids']) - 10} more")
    
    if 'error' in stats:
        print(f"  Error: {stats['error']}")

def write_fasta(filepath: str, sequences: Dict[str, str]):
    """
    Write sequences to a FASTA file.
    
    Args:
        filepath: Path to output FASTA file
        sequences: Dictionary with sequence IDs as keys and sequences as values
    """
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    
    try:
        records = []
        for seq_id, seq in sequences.items():
            record = SeqRecord(Seq(seq), id=seq_id, description="")
            records.append(record)
        
        with open(filepath, 'w') as output_handle:
            SeqIO.write(records, output_handle, "fasta")
            
    except Exception as e:
        raise ValueError(f"Error writing FASTA file {filepath}: {str(e)}")

__all__ = [
    'validate_fasta_file',
    'read_fasta_sequence',
    'read_fasta_file',
    'list_fasta_sequences',
    'get_fasta_stats',
    'print_fasta_stats',
    'write_fasta',
]