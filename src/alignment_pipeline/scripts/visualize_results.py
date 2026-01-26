#!/usr/bin/env python3
"""
Script to visualize alignment results.
Author: Rowel Facunla
"""

import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Visualize alignment results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='Results',
        help='Directory containing results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for visualizations'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Create interactive HTML visualizations'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip creating plot images'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return 1
    
    # Try to import visualization module
    try:
        from alignment_pipeline.visualization.visualizer import (
            visualize_pipeline_results,
            create_comprehensive_visualization
        )
    except ImportError as e:
        print(f"ERROR: Could not import visualization module: {e}")
        print("Make sure matplotlib, seaborn, and plotly are installed.")
        return 1
    
    # Load results
    results_json = results_dir / "results.json"
    if not results_json.exists():
        print(f"ERROR: Results file not found: {results_json}")
        return 1
    
    try:
        with open(results_json, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load results: {e}")
        return 1
    
    # Extract data
    metadata = results.get('metadata', {})
    stats = results.get('statistics', {})
    alignment = results.get('alignment', {})
    
    fasta1 = metadata.get('fasta1', 'sequence1')
    fasta2 = metadata.get('fasta2', 'sequence2')
    total_score = metadata.get('total_score', 0)
    seq1_name = metadata.get('sequence1_name', 'Sequence 1')
    seq2_name = metadata.get('sequence2_name', 'Sequence 2')
    
    a1 = alignment.get('sequence1', '')
    a2 = alignment.get('sequence2', '')
    
    if not a1 or not a2:
        print("ERROR: No alignment data found in results")
        return 1
    
    # Create comparison string
    from alignment_pipeline.core.utilities import iupac_is_match
    comp = ''.join('|' if iupac_is_match(a, b) else ' ' for a, b in zip(a1, a2))
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "visualizations"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating visualizations in: {output_dir}")
    
    # Create visualizations
    try:
        if not args.no_plots:
            print("Creating static visualizations...")
            create_comprehensive_visualization(
                a1, a2, comp,
                chunks_info=None,
                seq1_len=len(a1.replace('-', '')),
                seq2_len=len(a2.replace('-', '')),
                output_dir=str(output_dir),
                seq1_name=seq1_name,
                seq2_name=seq2_name
            )
        
        if args.interactive:
            print("Creating interactive visualization...")
            from alignment_pipeline.visualization.visualizer import create_interactive_alignment_report
            html_path = output_dir / "interactive_report.html"
            create_interactive_alignment_report(
                a1, a2, comp, stats,
                chunks=None,
                seq1_name=seq1_name,
                seq2_name=seq2_name,
                output_path=str(html_path)
            )
        
        print(f"\nVisualizations saved to: {output_dir}")
        
    except Exception as e:
        print(f"ERROR: Could not create visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())