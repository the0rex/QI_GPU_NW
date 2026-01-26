"""
Author: Rowel Facunla
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import Counter
import json
from typing import List, Tuple, Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from Bio import SeqIO
import warnings
warnings.filterwarnings('ignore')

# Color schemes for visualization
COLORS = {
    'match': '#2ecc71',      # Green
    'mismatch': '#e74c3c',   # Red
    'gap': '#3498db',        # Blue
    'insertion': '#9b59b6',  # Purple
    'deletion': '#e67e22',   # Orange
    'anchor': '#f1c40f',     # Yellow
    'chunk': '#1abc9c',      # Teal
    'background': '#ecf0f1'  # Light gray
}

# --------------------------
# Alignment Visualization
# --------------------------

def visualize_alignment_snippet(
    a1: str,
    a2: str,
    comp: str,
    start_idx: int = 0,
    snippet_length: int = 100,
    title: str = "Alignment Snippet",
    save_path: str = None
):
    """
    Visualize a snippet of the alignment with colored bases.
    
    Args:
        a1: First sequence with gaps
        a2: Second sequence with gaps
        comp: Comparison string ('|' for matches, ' ' for mismatches/gaps)
        start_idx: Starting index in alignment
        snippet_length: Number of bases to display
        title: Plot title
        save_path: Path to save figure
    """
    end_idx = min(start_idx + snippet_length, len(a1))
    
    # Extract snippet
    a1_snippet = a1[start_idx:end_idx]
    a2_snippet = a2[start_idx:end_idx]
    comp_snippet = comp[start_idx:end_idx]
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 3), 
                           gridspec_kw={'height_ratios': [1, 1, 0.5]})
    
    # Plot first sequence
    ax1 = axes[0]
    for i, base in enumerate(a1_snippet):
        x_pos = i
        if base == '-':
            color = COLORS['gap']
            alpha = 0.7
        else:
            color = get_base_color(base)
            alpha = 1.0
        ax1.add_patch(Rectangle((x_pos, 0), 0.8, 0.8, 
                               color=color, alpha=alpha))
        if base != '-':
            ax1.text(x_pos + 0.4, 0.4, base, 
                    ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
    
    ax1.set_xlim(-0.5, len(a1_snippet) + 0.5)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_title(f"Sequence 1 (Positions {start_idx}-{end_idx})")
    ax1.axis('off')
    
    # Plot comparison
    ax2 = axes[1]
    for i, (base1, base2, symbol) in enumerate(zip(a1_snippet, a2_snippet, comp_snippet)):
        x_pos = i
        if base2 == '-':
            color = COLORS['gap']
            alpha = 0.7
        else:
            color = get_base_color(base2)
            alpha = 1.0
        ax2.add_patch(Rectangle((x_pos, 0), 0.8, 0.8, 
                               color=color, alpha=alpha))
        if base2 != '-':
            ax2.text(x_pos + 0.4, 0.4, base2, 
                    ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='white')
        
        # Add match indicator
        if symbol == '|':
            ax2.plot([x_pos + 0.2, x_pos + 0.6], [0.9, 0.9], 
                    color=COLORS['match'], linewidth=3)
    
    ax2.set_xlim(-0.5, len(a2_snippet) + 0.5)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_title("Sequence 2 (Matches shown as green lines)")
    ax2.axis('off')
    
    # Plot match/mismatch/gap indicators
    ax3 = axes[2]
    for i, (base1, base2) in enumerate(zip(a1_snippet, a2_snippet)):
        x_pos = i
        if base1 == '-' or base2 == '-':
            color = COLORS['gap']
        elif iupac_is_match(base1, base2):
            color = COLORS['match']
        else:
            color = COLORS['mismatch']
        
        ax3.add_patch(Rectangle((x_pos, 0), 0.8, 0.4, color=color))
    
    ax3.set_xlim(-0.5, len(a1_snippet) + 0.5)
    ax3.set_ylim(-0.1, 0.6)
    ax3.set_title("Alignment Quality (Green=Match, Red=Mismatch, Blue=Gap)")
    ax3.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved alignment snippet to {save_path}")
    
    plt.show()

def get_base_color(base: str) -> str:
    """Get color for DNA base."""
    base_colors = {
        'A': '#3498db',  # Blue
        'C': '#2ecc71',  # Green
        'G': '#f39c12',  # Orange
        'T': '#e74c3c',  # Red
        'U': '#e74c3c',  # Red
        'N': '#95a5a6',  # Gray
        'R': '#9b59b6',  # Purple (A or G)
        'Y': '#16a085',  # Dark teal (C or T)
        'S': '#27ae60',  # Medium green (G or C)
        'W': '#8e44ad',  # Dark purple (A or T)
        'K': '#d35400',  # Dark orange (G or T)
        'M': '#2980b9',  # Medium blue (A or C)
        'B': '#c0392b',  # Dark red (C, G, or T)
        'D': '#e67e22',  # Light orange (A, G, or T)
        'H': '#7f8c8d',  # Gray (A, C, or T)
        'V': '#2c3e50',  # Very dark (A, C, or G)
        '-': '#bdc3c7'   # Light gray
    }
    return base_colors.get(base.upper(), '#95a5a6')

def visualize_alignment_heatmap(
    a1: str,
    a2: str,
    window_size: int = 100,
    title: str = "Alignment Quality Heatmap",
    save_path: str = None
):
    """
    Create a heatmap showing alignment quality over windows.
    
    Args:
        a1: First aligned sequence
        a2: Second aligned sequence
        window_size: Size of sliding window
        title: Plot title
        save_path: Path to save figure
    """
    n_windows = len(a1) // window_size
    if n_windows == 0:
        print("Sequence too short for heatmap")
        return
    
    # Calculate metrics for each window
    matches_per_window = []
    mismatches_per_window = []
    gaps_per_window = []
    identities = []
    
    for i in range(0, len(a1) - window_size + 1, window_size):
        window_a1 = a1[i:i+window_size]
        window_a2 = a2[i:i+window_size]
        
        matches = sum(1 for b1, b2 in zip(window_a1, window_a2) 
                     if iupac_is_match(b1, b2))
        mismatches = sum(1 for b1, b2 in zip(window_a1, window_a2) 
                        if b1 != '-' and b2 != '-' and not iupac_is_match(b1, b2))
        gaps = sum(1 for b1, b2 in zip(window_a1, window_a2) 
                  if b1 == '-' or b2 == '-')
        
        matches_per_window.append(matches)
        mismatches_per_window.append(mismatches)
        gaps_per_window.append(gaps)
        
        total_non_gap = matches + mismatches
        identity = matches / total_non_gap if total_non_gap > 0 else 0
        identities.append(identity)
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    
    # Plot match distribution
    axes[0].bar(range(len(matches_per_window)), matches_per_window, 
                color=COLORS['match'], alpha=0.7)
    axes[0].set_ylabel('Matches')
    axes[0].set_title('Matches per Window')
    axes[0].grid(True, alpha=0.3)
    
    # Plot mismatch distribution
    axes[1].bar(range(len(mismatches_per_window)), mismatches_per_window, 
                color=COLORS['mismatch'], alpha=0.7)
    axes[1].set_ylabel('Mismatches')
    axes[1].set_title('Mismatches per Window')
    axes[1].grid(True, alpha=0.3)
    
    # Plot gap distribution
    axes[2].bar(range(len(gaps_per_window)), gaps_per_window, 
                color=COLORS['gap'], alpha=0.7)
    axes[2].set_ylabel('Gaps')
    axes[2].set_title('Gaps per Window')
    axes[2].grid(True, alpha=0.3)
    
    #Normalize plot range
    y_max = max(
        max(matches_per_window, default=0),
        max(mismatches_per_window, default=0),
        max(gaps_per_window, default=0)
    )
    y_max = int(y_max * 1.05) + 1
    
    for ax in axes[:3]:
        ax.set_ylim(0, y_max)
    
    for ax in axes[:3]:
        ax.set_ylim(0, y_max)
    
    # Plot identity
    axes[3].plot(range(len(identities)), identities, 
                color=COLORS['match'], linewidth=2, marker='o')
    axes[3].axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% identity')
    axes[3].axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80% identity')
    axes[3].set_ylabel('Identity')
    axes[3].set_xlabel('Window Index')
    axes[3].set_title('Sequence Identity per Window')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(0, 1.1)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    
    plt.show()

# --------------------------
# Statistical Visualizations
# --------------------------

def visualize_alignment_statistics(
    a1: str,
    a2: str,
    save_path: str = None
):
    """
    Create comprehensive statistical visualizations of alignment.
    
    Args:
        a1: First aligned sequence
        a2: Second aligned sequence
        save_path: Path to save figure
    """
def visualize_alignment_statistics(
    a1: str,
    a2: str,
    save_path: str = None
):
    """
    Create comprehensive statistical visualizations of alignment.
    
    Args:
        a1: First aligned sequence
        a2: Second aligned sequence
        save_path: Path to save figure
    """
    # Calculate statistics
    matches = sum(1 for x, y in zip(a1, a2) if iupac_is_match(x, y))
    mismatches = sum(1 for x, y in zip(a1, a2) 
                    if x != '-' and y != '-' and not iupac_is_match(x, y))
    insertions = sum(1 for x, y in zip(a1, a2) if x == '-' and y != '-')
    deletions = sum(1 for x, y in zip(a1, a2) if x != '-' and y == '-')
    total_gaps = insertions + deletions
    
    total_aligned = len(a1)
    identity = matches / (matches + mismatches) if (matches + mismatches) > 0 else 0
    gap_percentage = total_gaps / total_aligned
    
    # Create figure with tight layout control
    fig = plt.figure(figsize=(15, 10))
    
    # Pie chart of alignment composition - FIXED VERSION
    ax1 = plt.subplot(2, 3, 1)
    sizes = [matches, mismatches, insertions, deletions]
    labels = ['Matches', 'Mismatches', 'Insertions', 'Deletions']
    colors = [COLORS['match'], COLORS['mismatch'], 
              COLORS['insertion'], COLORS['deletion']]
    
    # Calculate percentages
    total = sum(sizes)
    percentages = [f'{size/total*100:.2f}%' if size/total*100 < 1 else f'{size/total*100:.1f}%' 
                   for size in sizes]
    
    # Create pie chart
    wedges, texts = ax1.pie(
        sizes, 
        colors=colors,
        startangle=90,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )
    
    # Set title FIRST (before adding any other elements)
    ax1.set_title('Alignment Composition', fontweight='bold', pad=20)
    
    # Check if we have very small values
    has_very_small_values = any(size/total < 0.01 for size in sizes[1:])
    
    if has_very_small_values:
        # Group small values together in a single "Other" category
        small_threshold = 0.01
        small_indices = [i for i, size in enumerate(sizes) if size/total < small_threshold and i > 0]
        
        if len(small_indices) >= 2:
            # Group small categories
            small_total = sum(sizes[i] for i in small_indices)
            small_labels = [labels[i] for i in small_indices]
            
            # Create new sizes and labels
            main_sizes = [sizes[0]] + [sizes[i] for i in range(1, 4) if i not in small_indices]
            main_labels = [labels[0]] + [labels[i] for i in range(1, 4) if i not in small_indices]
            main_colors = [colors[0]] + [colors[i] for i in range(1, 4) if i not in small_indices]
            
            # Add "Other" category
            if small_total > 0:
                main_sizes.append(small_total)
                # Shorten the "Other" label to prevent overlap
                if len(small_labels) > 2:
                    other_label = f'Other\n({small_labels[0]}, {small_labels[1]}, ...)'
                else:
                    other_label = f'Other\n({", ".join(small_labels)})'
                main_labels.append(other_label)
                main_colors.append('#95a5a6')
            
            # Recreate pie with grouped categories
            ax1.clear()
            wedges, texts = ax1.pie(
                main_sizes, 
                colors=main_colors,
                startangle=90,
                wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
            )
            
            # Set title again after clearing
            ax1.set_title('Alignment Composition', fontweight='bold', pad=40)
            
            # Add labels for grouped version
            for i, (wedge, label) in enumerate(zip(wedges, main_labels)):
                ang = (wedge.theta2 + wedge.theta1) / 2.0
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                
                # Calculate wedge size
                wedge_size = main_sizes[i] / sum(main_sizes)
                
                if wedge_size >= 0.05:  # Large enough for internal label
                    # Position label inside with offset based on angle
                    if abs(y) > 0.7:  # Top or bottom wedge
                        vertical_offset = 0.2 if y > 0 else -0.2
                    else:
                        vertical_offset = 0
                    
                    label_x = 0.4 * x
                    label_y = 0.4 * y + vertical_offset
                    
                    ax1.text(
                        label_x, 
                        label_y, 
                        label, 
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=8,
                        fontweight='bold',
                        color='white'
                    )
                else:
                    # Small wedge - place label outside
                    # Position label further out and adjust based on quadrant
                    distance = 1.4
                    
                    # Adjust position to avoid title area (top of graph)
                    if y > 0.7:  # If label would be near the top
                        # Move it slightly downward and further out
                        label_x = distance * x
                        label_y = distance * y * 0.9  # Move down slightly
                        fontsize = 7
                    else:
                        label_x = distance * x
                        label_y = distance * y
                        fontsize = 8
                    
                    # Determine text alignment based on position
                    if x > 0.5:
                        halign = 'left'
                    elif x < -0.5:
                        halign = 'right'
                    else:
                        halign = 'center'
                    
                    bbox_props = dict(
                        boxstyle="round,pad=0.2", 
                        facecolor="white", 
                        alpha=0.9, 
                        edgecolor="gray"
                    )
                    
                    # Add connecting line for clarity
                    connectionstyle = f"angle,angleA=0,angleB={ang}"
                    ax1.annotate(
                        "", 
                        xy=(x, y), 
                        xytext=(label_x - 0.1 if halign == 'left' else 
                               label_x + 0.1 if halign == 'right' else label_x, 
                               label_y),
                        arrowprops=dict(
                            arrowstyle="-", 
                            color="gray", 
                            linewidth=0.5,
                            connectionstyle=connectionstyle
                        )
                    )
                    
                    ax1.text(
                        label_x, 
                        label_y, 
                        label, 
                        horizontalalignment=halign,
                        verticalalignment="center",
                        fontsize=fontsize,
                        bbox=bbox_props
                    )
        else:
            # Use donut chart for visibility
            ax1.clear()
            
            # Create donut chart
            wedges, texts = ax1.pie(
                sizes, 
                colors=colors,
                startangle=90,
                wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                radius=0.75  # Smaller radius
            )
            
            # Add center circle
            centre_circle = plt.Circle((0, 0), 0.5, fc='white', edgecolor='gray', linewidth=1)
            ax1.add_artist(centre_circle)
            
            # Add identity in center
            ax1.text(0, 0, f'Identity:\n{identity*100:.1f}%', 
                    ha='center', va='center', fontsize=9, fontweight='bold')
            
            # Set title
            ax1.set_title('Alignment Composition', fontweight='bold', pad=20)
            
            # Place all labels outside
            for i, (wedge, label) in enumerate(zip(wedges, labels)):
                ang = (wedge.theta2 + wedge.theta1) / 2.0
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                
                # Position label - avoid top area
                distance = 1.5
                if y > 0.6:  # Top area - move further out and adjust
                    label_x = distance * x
                    label_y = distance * 0.85  # Keep below title
                    fontsize = 7
                else:
                    label_x = distance * x
                    label_y = distance * y
                    fontsize = 8
                
                # Text alignment
                if x > 0.3:
                    halign = 'left'
                elif x < -0.3:
                    halign = 'right'
                else:
                    halign = 'center'
                
                # Create label text
                label_text = f'{label}\n({percentages[i]})'
                
                bbox_props = dict(
                    boxstyle="round,pad=0.2", 
                    facecolor="white", 
                    alpha=0.9, 
                    edgecolor="gray"
                )
                
                # Add connecting line
                connectionstyle = f"angle,angleA=0,angleB={ang}"
                ax1.annotate(
                    "", 
                    xy=(0.75*x, 0.75*y),  # From outer edge
                    xytext=(label_x - 0.1 if halign == 'left' else 
                           label_x + 0.1 if halign == 'right' else label_x, 
                           label_y),
                    arrowprops=dict(
                        arrowstyle="-", 
                        color="gray", 
                        linewidth=0.5,
                        connectionstyle=connectionstyle
                    )
                )
                
                ax1.text(
                    label_x, 
                    label_y, 
                    label_text, 
                    horizontalalignment=halign,
                    verticalalignment="center",
                    fontsize=fontsize,
                    bbox=bbox_props
                )
    else:
        # Normal case - no very small values
        for i, (wedge, label) in enumerate(zip(wedges, labels)):
            ang = (wedge.theta2 + wedge.theta1) / 2.0
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            
            wedge_size = sizes[i] / total
            
            if wedge_size >= 0.05:
                # Internal label
                label_x = 0.5 * x
                label_y = 0.5 * y
                
                ax1.text(
                    label_x, 
                    label_y, 
                    f'{label}\n({percentages[i]})', 
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=9,
                    fontweight='bold',
                    color='white'
                )
            else:
                # External label - position carefully
                distance = 1.3
                if y > 0.5:  # Top half - be careful
                    label_x = distance * x
                    label_y = distance * 0.7  # Keep away from top
                    fontsize = 8
                else:
                    label_x = distance * x
                    label_y = distance * y
                    fontsize = 9
                
                # Alignment
                if x > 0.3:
                    halign = 'left'
                elif x < -0.3:
                    halign = 'right'
                else:
                    halign = 'center'
                
                bbox_props = dict(
                    boxstyle="round,pad=0.2", 
                    facecolor="white", 
                    alpha=0.9, 
                    edgecolor="gray"
                )
                
                ax1.text(
                    label_x, 
                    label_y, 
                    f'{label}\n({percentages[i]})', 
                    horizontalalignment=halign,
                    verticalalignment="center",
                    fontsize=fontsize,
                    bbox=bbox_props
                )
    
    ax1.axis('equal')
    # Ensure title is visible (already set)
    
    # Bar chart of base frequencies
    ax2 = plt.subplot(2, 3, 2)
    bases = ['A', 'C', 'G', 'T', 'N', 'Other']
    
    # Count bases in each sequence (excluding gaps)
    a1_bases = [b for b in a1 if b != '-']
    a2_bases = [b for b in a2 if b != '-']
    
    a1_counts = Counter(a1_bases)
    a2_counts = Counter(a2_bases)
    
    a1_freq = [a1_counts.get(b, 0) / len(a1_bases) * 100 
               for b in bases[:4]]
    a2_freq = [a2_counts.get(b, 0) / len(a2_bases) * 100 
               for b in bases[:4]]
    
    x = np.arange(len(bases[:4]))
    width = 0.35
    
    ax2.bar(x - width/2, a1_freq, width, label='Sequence 1', 
            color=COLORS['match'], alpha=0.7)
    ax2.bar(x + width/2, a2_freq, width, label='Sequence 2', 
            color=COLORS['mismatch'], alpha=0.7)
    
    ax2.set_xlabel('Base')
    ax2.set_ylabel('Frequency (%)')
    ax2.set_title('Base Frequency Distribution', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bases[:4])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gap length distribution
    ax3 = plt.subplot(2, 3, 3)
    
    # Find gap runs
    gap_lengths = []
    current_gap = 0
    
    for b1, b2 in zip(a1, a2):
        if b1 == '-' or b2 == '-':
            current_gap += 1
        elif current_gap > 0:
            gap_lengths.append(current_gap)
            current_gap = 0
    
    if current_gap > 0:
        gap_lengths.append(current_gap)
    
    if gap_lengths:
        ax3.hist(gap_lengths, bins=min(20, max(gap_lengths)), 
                color=COLORS['gap'], alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Gap Length')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Gap Length Distribution')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No gaps found', 
                ha='center', va='center', fontsize=12)
        ax3.set_title('Gap Length Distribution')
    
    # Identity score over position
    ax4 = plt.subplot(2, 3, 4)
    
    window_size = max(100, len(a1) // 100)
    identities = []
    positions = []
    
    for i in range(0, len(a1), window_size):
        end = min(i + window_size, len(a1))
        window_a1 = a1[i:end]
        window_a2 = a2[i:end]
        
        window_matches = sum(1 for b1, b2 in zip(window_a1, window_a2) 
                           if iupac_is_match(b1, b2))
        window_total = sum(1 for b1, b2 in zip(window_a1, window_a2) 
                          if b1 != '-' and b2 != '-')
        
        if window_total > 0:
            identity = window_matches / window_total
            identities.append(identity)
            positions.append(i)
    
    ax4.plot(positions, identities, color=COLORS['match'], linewidth=2)
    ax4.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90%')
    ax4.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80%')
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Identity')
    ax4.set_title('Identity vs Position')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
    # Summary statistics
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    stats_text = f"""
    Alignment Statistics
    ===================
    Total length: {total_aligned:,} bp
    Matches: {matches:,} ({identity*100:.1f}%)
    Mismatches: {mismatches:,}
    Insertions: {insertions:,}
    Deletions: {deletions:,}
    Total gaps: {total_gaps:,} ({gap_percentage*100:.1f}%)
    
    Sequence 1 length: {len(a1.replace('-', '')):,} bp
    Sequence 2 length: {len(a2.replace('-', '')):,} bp
    
    Overall identity: {identity*100:.2f}%
    Gap percentage: {gap_percentage*100:.2f}%
    """
    
    ax5.text(0.1, 0.5, stats_text, fontfamily='monospace', 
             fontsize=10, verticalalignment='center')
    
    # Match/mismatch pattern
    ax6 = plt.subplot(2, 3, 6)
    
    # Take a sample of positions for visualization
    sample_size = min(1000, len(a1))
    step = len(a1) // sample_size
    
    sample_indices = range(0, len(a1), step)
    sample_a1 = [a1[i] for i in sample_indices]
    sample_a2 = [a2[i] for i in sample_indices]
    
    match_pattern = []
    for b1, b2 in zip(sample_a1, sample_a2):
        if b1 == '-' or b2 == '-':
            match_pattern.append(0)  # Gap
        elif iupac_is_match(b1, b2):
            match_pattern.append(1)  # Match
        else:
            match_pattern.append(-1)  # Mismatch
    
    ax6.scatter(range(len(match_pattern)), match_pattern, 
               c=match_pattern, cmap='RdYlGn', alpha=0.6, s=10)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax6.set_xlabel('Sample Position')
    ax6.set_ylabel('Alignment State')
    ax6.set_title('Match Pattern (Green=Match, Red=Mismatch, Gray=Gap)')
    ax6.set_yticks([-1, 0, 1])
    ax6.set_yticklabels(['Mismatch', 'Gap', 'Match'])
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Alignment Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved statistics visualization to {save_path}")
    
    plt.show()
    
    return {
        'matches': matches,
        'mismatches': mismatches,
        'insertions': insertions,
        'deletions': deletions,
        'total_aligned': total_aligned,
        'identity': identity,
        'gap_percentage': gap_percentage
    }

# --------------------------
# Chunk Visualization
# --------------------------

def visualize_chunk_alignment(
    chunks: List[Dict[str, Any]],
    seq1_len: int,
    seq2_len: int,
    title: str = "Chunk Alignment Overview",
    save_path: str = None
):
    """
    Visualize how chunks are distributed across sequences.
    
    Args:
        chunks: List of chunk dictionaries with q_start, q_end, t_start, t_end
        seq1_len: Length of first sequence
        seq2_len: Length of second sequence
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Plot chunks on sequence 1
    ax1 = axes[0]
    for i, chunk in enumerate(chunks):
        q_start = chunk.get('q_start', chunk.get('s1_start', 0))
        q_end = chunk.get('q_end', chunk.get('s1_end', 0))
        t_start = chunk.get('t_start', chunk.get('s2_start', 0))
        t_end = chunk.get('t_end', chunk.get('s2_end', 0))
        
        # Rectangle for chunk
        rect = Rectangle((q_start, 0), q_end - q_start, 1, 
                        facecolor=COLORS['chunk'], alpha=0.7,
                        edgecolor='black', linewidth=1)
        ax1.add_patch(rect)
        
        # Add chunk number
        ax1.text(q_start + (q_end - q_start)/2, 0.5, str(i),
                ha='center', va='center', fontweight='bold')
    
    ax1.set_xlim(0, seq1_len)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_xlabel('Sequence 1 Position')
    ax1.set_title('Chunks on Sequence 1')
    ax1.set_yticks([])
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot chunks on sequence 2
    ax2 = axes[1]
    for i, chunk in enumerate(chunks):
        q_start = chunk.get('q_start', chunk.get('s1_start', 0))
        q_end = chunk.get('q_end', chunk.get('s1_end', 0))
        t_start = chunk.get('t_start', chunk.get('s2_start', 0))
        t_end = chunk.get('t_end', chunk.get('s2_end', 0))
        
        # Rectangle for chunk
        rect = Rectangle((t_start, 0), t_end - t_start, 1, 
                        facecolor=COLORS['chunk'], alpha=0.7,
                        edgecolor='black', linewidth=1)
        ax2.add_patch(rect)
        
        # Add chunk number
        ax2.text(t_start + (t_end - t_start)/2, 0.5, str(i),
                ha='center', va='center', fontweight='bold')
    
    ax2.set_xlim(0, seq2_len)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_xlabel('Sequence 2 Position')
    ax2.set_title('Chunks on Sequence 2')
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved chunk visualization to {save_path}")
    
    plt.show()

# --------------------------
# Interactive HTML Report
# --------------------------

def create_interactive_alignment_report(
    a1: str,
    a2: str,
    comp: str,
    stats: Dict[str, Any],
    chunks: List[Dict[str, Any]] = None,
    seq1_name: str = "Sequence 1",
    seq2_name: str = "Sequence 2",
    output_path: str = "alignment_report.html"
):
    """
    Create an interactive HTML report with Plotly visualizations.
    
    Args:
        a1: First aligned sequence
        a2: Second aligned sequence
        comp: Comparison string
        stats: Alignment statistics
        chunks: List of chunk information
        seq1_name: Name of first sequence
        seq2_name: Name of second sequence
        output_path: Path to save HTML report
    """
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Alignment Overview', 'Base Composition',
                       'Identity Distribution', 'Gap Analysis',
                       'Alignment Statistics', 'Chunk Overview'),
        specs=[[{'type': 'heatmap'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'histogram'}],
               [{'type': 'table'}, {'type': 'scatter'}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.15
    )
    
    # 1. Alignment overview heatmap
    sample_size = min(1000, len(a1))
    step = len(a1) // sample_size
    
    positions = list(range(0, len(a1), step))
    alignment_states = []
    
    for i in positions:
        b1 = a1[i]
        b2 = a2[i]
        if b1 == '-' or b2 == '-':
            alignment_states.append(0)  # Gap
        elif iupac_is_match(b1, b2):
            alignment_states.append(1)  # Match
        else:
            alignment_states.append(-1)  # Mismatch
    
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=alignment_states,
            mode='markers',
            marker=dict(
                size=5,
                color=alignment_states,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Alignment State")
            ),
            name="Alignment State",
            hovertemplate="Position: %{x}<br>State: %{y}<br>"
        ),
        row=1, col=1
    )
    
    # 2. Base composition bar chart
    bases = ['A', 'C', 'G', 'T', 'N', 'Other']
    a1_bases = [b for b in a1 if b != '-']
    a2_bases = [b for b in a2 if b != '-']
    
    a1_counts = Counter(a1_bases)
    a2_counts = Counter(a2_bases)
    
    a1_freq = [a1_counts.get(b, 0) for b in bases[:4]]
    a2_freq = [a2_counts.get(b, 0) for b in bases[:4]]
    
    fig.add_trace(
        go.Bar(
            name=seq1_name,
            x=bases[:4],
            y=a1_freq,
            marker_color=get_base_color('A')
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            name=seq2_name,
            x=bases[:4],
            y=a2_freq,
            marker_color=get_base_color('C')
        ),
        row=1, col=2
    )
    
    # 3. Identity distribution
    window_size = max(100, len(a1) // 100)
    identities = []
    window_positions = []
    
    for i in range(0, len(a1), window_size):
        end = min(i + window_size, len(a1))
        window_a1 = a1[i:end]
        window_a2 = a2[i:end]
        
        matches = sum(1 for b1, b2 in zip(window_a1, window_a2) 
                     if iupac_is_match(b1, b2))
        total = sum(1 for b1, b2 in zip(window_a1, window_a2) 
                   if b1 != '-' and b2 != '-')
        
        if total > 0:
            identity = matches / total
            identities.append(identity)
            window_positions.append(i)
    
    fig.add_trace(
        go.Scatter(
            x=window_positions,
            y=identities,
            mode='lines+markers',
            name='Identity',
            line=dict(color=COLORS['match'], width=2),
            marker=dict(size=4)
        ),
        row=2, col=1
    )
    
    # Add reference lines
    fig.add_hline(y=0.9, line_dash="dash", line_color="red", 
                 opacity=0.5, row=2, col=1)
    fig.add_hline(y=0.8, line_dash="dash", line_color="orange", 
                 opacity=0.5, row=2, col=1)
    
    # 4. Gap length distribution
    gap_lengths = []
    current_gap = 0
    
    for b1, b2 in zip(a1, a2):
        if b1 == '-' or b2 == '-':
            current_gap += 1
        elif current_gap > 0:
            gap_lengths.append(current_gap)
            current_gap = 0
    
    if current_gap > 0:
        gap_lengths.append(current_gap)
    
    if gap_lengths:
        fig.add_trace(
            go.Histogram(
                x=gap_lengths,
                nbinsx=min(20, max(gap_lengths)),
                name='Gap Lengths',
                marker_color=COLORS['gap'],
                opacity=0.7
            ),
            row=2, col=2
        )
    
    # 5. Statistics table
    stats_data = [
        ["Metric", "Value"],
        ["Total Alignment Length", f"{stats.get('total_aligned', 0):,} bp"],
        ["Matches", f"{stats.get('matches', 0):,} ({stats.get('identity', 0)*100:.1f}%)"],
        ["Mismatches", f"{stats.get('mismatches', 0):,}"],
        ["Insertions", f"{stats.get('insertions', 0):,}"],
        ["Deletions", f"{stats.get('deletions', 0):,}"],
        ["Total Gaps", f"{stats.get('total_gaps', 0):,} ({stats.get('gap_percentage', 0)*100:.1f}%)"],
        ["Sequence 1 Length", f"{len(a1.replace('-', '')):,} bp"],
        ["Sequence 2 Length", f"{len(a2.replace('-', '')):,} bp"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=list(zip(*stats_data)),
                fill_color='lavender',
                align='left'
            )
        ),
        row=3, col=1
    )
    
    # 6. Chunk overview (if chunks provided)
    if chunks:
        chunk_starts = []
        chunk_ends = []
        chunk_scores = []
        
        for chunk in chunks:
            chunk_starts.append(chunk.get('q_start', 0))
            chunk_ends.append(chunk.get('q_end', 0))
            chunk_scores.append(chunk.get('score', 0))
        
        fig.add_trace(
            go.Scatter(
                x=chunk_starts,
                y=chunk_scores,
                mode='markers',
                name='Chunk Scores',
                marker=dict(
                    size=10,
                    color=chunk_scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Chunk Score")
                ),
                text=[f"Chunk {i}: Score {s}" for i, s in enumerate(chunk_scores)],
                hovertemplate="Chunk %{text}<br>Start: %{x}<br>Score: %{y}"
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text=f"Alignment Report: {seq1_name} vs {seq2_name}",
        title_font_size=20,
        showlegend=True,
        height=1200,
        template="plotly_white"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Position", row=1, col=1)
    fig.update_yaxes(title_text="Alignment State", row=1, col=1)
    fig.update_xaxes(title_text="Base", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Position", row=2, col=1)
    fig.update_yaxes(title_text="Identity", row=2, col=1)
    fig.update_xaxes(title_text="Gap Length", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    # Save HTML report
    fig.write_html(output_path)
    print(f"Created interactive HTML report: {output_path}")
    
    return fig

# --------------------------
# Main Visualization Function
# --------------------------

def create_comprehensive_visualization(
    a1: str,
    a2: str,
    comp: str,
    chunks_info: List[Dict[str, Any]] = None,
    seq1_len: int = None,
    seq2_len: int = None,
    output_dir: str = "visualizations",
    seq1_name: str = "Sequence 1",
    seq2_name: str = "Sequence 2"
):
    """
    Create all visualizations for the alignment.
    
    Args:
        a1: First aligned sequence
        a2: Second aligned sequence
        comp: Comparison string
        chunks_info: List of chunk information
        seq1_len: Original length of sequence 1
        seq2_len: Original length of sequence 2
        output_dir: Directory to save visualizations
        seq1_name: Name of first sequence
        seq2_name: Name of second sequence
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Creating Comprehensive Visualizations")
    print("="*60)
    
    # Calculate statistics
    print("\n1. Calculating alignment statistics...")
    stats = visualize_alignment_statistics(
        a1, a2,
        save_path=os.path.join(output_dir, "alignment_statistics.png")
    )
    
    # Visualize alignment snippet
    print("\n2. Creating alignment snippet visualization...")
    visualize_alignment_snippet(
        a1, a2, comp,
        start_idx=0,
        snippet_length=100,
        title=f"Alignment Snippet: {seq1_name} vs {seq2_name}",
        save_path=os.path.join(output_dir, "alignment_snippet.png")
    )
    
    # Visualize middle snippet
    mid_point = max(0, len(a1) // 2 - 50)
    visualize_alignment_snippet(
        a1, a2, comp,
        start_idx=mid_point,
        snippet_length=100,
        title=f"Alignment Snippet (Middle): {seq1_name} vs {seq2_name}",
        save_path=os.path.join(output_dir, "alignment_snippet_middle.png")
    )
    
    # Create heatmap
    print("\n3. Creating alignment heatmap...")
    visualize_alignment_heatmap(
        a1, a2,
        window_size=max(100, len(a1) // 50),
        title=f"Alignment Quality Heatmap: {seq1_name} vs {seq2_name}",
        save_path=os.path.join(output_dir, "alignment_heatmap.png")
    )
    
    # Visualize chunks if information available
    if chunks_info and seq1_len and seq2_len:
        print("\n4. Visualizing chunk distribution...")
        visualize_chunk_alignment(
            chunks_info,
            seq1_len,
            seq2_len,
            title=f"Chunk Distribution: {seq1_name} vs {seq2_name}",
            save_path=os.path.join(output_dir, "chunk_distribution.png")
        )
    
    # Create interactive HTML report
    print("\n5. Creating interactive HTML report...")
    create_interactive_alignment_report(
        a1, a2, comp, stats,
        chunks=chunks_info,
        seq1_name=seq1_name,
        seq2_name=seq2_name,
        output_path=os.path.join(output_dir, "alignment_report.html")
    )
    
    # Save raw data for further analysis
    print("\n6. Saving alignment data...")
    save_alignment_data(a1, a2, comp, stats, output_dir)
    
    print("\n" + "="*60)
    print(f"All visualizations saved to: {output_dir}")
    print("="*60)
    
    return stats

def save_alignment_data(
    a1: str,
    a2: str,
    comp: str,
    stats: Dict[str, Any],
    output_dir: str
):
    """Save alignment data in various formats."""
    # Save alignment as text
    with open(os.path.join(output_dir, "alignment.txt"), "w") as f:
        f.write(f"Sequence 1: {a1}\n")
        f.write(f"Comparison: {comp}\n")
        f.write(f"Sequence 2: {a2}\n")
    
    # Save statistics as JSON
    with open(os.path.join(output_dir, "statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    # Save summary report
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("="*60 + "\n")
        f.write("ALIGNMENT SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total Alignment Length: {stats.get('total_aligned', 0):,} bp\n")
        f.write(f"Matches: {stats.get('matches', 0):,} ({stats.get('identity', 0)*100:.2f}%)\n")
        f.write(f"Mismatches: {stats.get('mismatches', 0):,}\n")
        f.write(f"Insertions: {stats.get('insertions', 0):,}\n")
        f.write(f"Deletions: {stats.get('deletions', 0):,}\n")
        f.write(f"Total Gaps: {stats.get('total_gaps', 0):,} ({stats.get('gap_percentage', 0)*100:.2f}%)\n\n")
        
        f.write(f"Sequence 1 (no gaps): {len(a1.replace('-', '')):,} bp\n")
        f.write(f"Sequence 2 (no gaps): {len(a2.replace('-', '')):,} bp\n\n")
        
        if stats.get('identity', 0) >= 0.9:
            f.write("Alignment Quality: EXCELLENT (≥90% identity)\n")
        elif stats.get('identity', 0) >= 0.8:
            f.write("Alignment Quality: GOOD (80-90% identity)\n")
        elif stats.get('identity', 0) >= 0.7:
            f.write("Alignment Quality: FAIR (70-80% identity)\n")
        else:
            f.write("Alignment Quality: POOR (<70% identity)\n")

# --------------------------
# Integration with Main Pipeline
# --------------------------

def visualize_pipeline_results(
    final_a1: str,
    final_a2: str,
    total_score: float,
    fasta1: str,
    fasta2: str,
    chunk_files: List[str] = None,
    output_dir: str = "Results/visualizations",
    title: str = None
):
    """
    Main function to visualize pipeline results.
    Call this after your pipeline completes.
    
    Args:
        final_a1: Final aligned sequence 1
        final_a2: Final aligned sequence 2
        total_score: Total alignment score
        fasta1: Path to first FASTA file
        fasta2: Path to second FASTA file
        chunk_files: List of chunk NPZ files (optional)
        output_dir: Output directory for visualizations
    """
    # Create comparison string
    comp = ''.join('|' if iupac_is_match(a, b) else ' ' 
                   for a, b in zip(final_a1, final_a2))
    
    # Extract sequence names
    seq1_name = os.path.basename(fasta1).replace('.fa', '').replace('.fasta', '')
    seq2_name = os.path.basename(fasta2).replace('.fa', '').replace('.fasta', '')
    
    # Load chunk information if available
    chunks_info = []
    if chunk_files:
        print(f"Loading information from {len(chunk_files)} chunk files...")
        for chunk_file in chunk_files[:100]:  # Limit to first 100 chunks
            try:
                data = np.load(chunk_file, allow_pickle=True)
                meta = json.loads(str(data['meta'].tolist()))
                chunks_info.append(meta)
            except:
                continue
    
    # Create comprehensive visualizations
    stats = create_comprehensive_visualization(
        final_a1, final_a2, comp,
        chunks_info=chunks_info,
        seq1_len=len(final_a1.replace('-', '')),
        seq2_len=len(final_a2.replace('-', '')),
        output_dir=output_dir,
        seq1_name=seq1_name if not title else title.split(" vs ")[0],
        seq2_name=seq2_name if not title else title.split(" vs ")[1] if " vs " in title else seq2_name
    )
    
    # Add total score to stats
    stats['total_score'] = total_score
    
    # Print summary
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Sequences: {seq1_name} vs {seq2_name}")
    print(f"Total alignment score: {total_score:.2f}")
    print(f"Identity: {stats['identity']*100:.2f}%")
    print(f"Total aligned length: {stats['total_aligned']:,} bp")
    print(f"Matches: {stats['matches']:,}")
    print(f"Mismatches: {stats['mismatches']:,}")
    print(f"Gaps: {stats['insertions'] + stats['deletions']:,}")
    print(f"Visualizations saved to: {output_dir}")
    print("="*60)
    
    return stats

# --------------------------
# Helper functions for main code
# --------------------------

def iupac_is_match(a, b):
    """Exact-match only. Ambiguous bases never count as a match."""
    if a is None or b is None:
        return False
    if a == '-' or b == '-':
        return False
    
    a_up = a.upper()
    b_up = b.upper()
    
    # Check if either base is ambiguous (not A, C, G, T, U)
    valid_bases = {'A', 'C', 'G', 'T', 'U'}
    if a_up not in valid_bases or b_up not in valid_bases:
        return False
    
    return a_up == b_up

def load_chunk_metadata(chunk_dir: str) -> List[Dict[str, Any]]:
    """
    Load metadata from all chunk files in a directory.
    
    Args:
        chunk_dir: Directory containing chunk NPZ files
        
    Returns:
        List of chunk metadata dictionaries
    """
    import glob
    
    chunks = []
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.npz")))
    
    for chunk_file in chunk_files:
        try:
            data = np.load(chunk_file, allow_pickle=True)
            meta = json.loads(str(data['meta'].tolist()))
            chunks.append(meta)
        except Exception as e:
            print(f"Warning: Could not load {chunk_file}: {e}")
    
    return chunks

__all__ = [
    'visualize_pipeline_results',
    'create_comprehensive_visualization',
    'visualize_alignment_snippet',
    'visualize_alignment_statistics',
    'visualize_alignment_heatmap',
    'visualize_chunk_alignment',
    'create_interactive_alignment_report',
    'iupac_is_match',
]