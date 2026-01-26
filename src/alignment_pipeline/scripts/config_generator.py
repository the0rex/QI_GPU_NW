#!/usr/bin/env python3
"""
Configuration generator script for alignment pipeline.
Author: Rowel Facunla
"""

import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any

def generate_basic_config(output_file: str = "pipeline_config.yaml") -> Dict[str, Any]:
    """Generate basic configuration for CPU-only pipeline."""
    
    config = {
        'pipeline': {
            'name': 'Alignment Pipeline',
            'version': '2.0.0',
            'description': 'Basic configuration for CPU-only alignment'
        },
        'io': {
            'fasta_dir': 'examples',
            'default_fasta1': 'test_fasta_1.fa',
            'default_fasta2': 'test_fasta_2.fa',
            'output_dir': 'Results',
            'chunk_dir': 'chunks_tmp',
            'visualization_dir': 'Results/visualizations',
            'logs_dir': 'Results/logs'
        },
        'alignment': {
            'gap_open': -30,
            'gap_extend': -0.5,
            'match_score': 5,
            'mismatch_score': -50,
            'window_size_predictor': 'simple',
            'beam_width': 30,
            'carry_gap_state': True,
            'use_gpu_acceleration': False
        },
        'chunking': {
            'default_chunk_size': 5000,
            'overlap': 250,
            'min_chunk_size': 100,
            'max_chunk_size': 10000,
            'chunk_tolerance': 0.05,
            'use_gpu_anchoring': False,
            'use_gpu_pruning': False
        },
        'seeding': {
            'kmer_size': 21,
            'syncmer_size': 5,
            'syncmer_position': 2,
            'window_min': 20,
            'window_max': 70,
            'max_occurrences': 200,
            'gpu_strobe_generation': False,
            'gpu_index_building': False
        },
        'models': {
            'window_predictor': 'model_Adam.h5',
            'torch_window_predictor': 'model_torch.pt',
            'x_scaler': 'x_scaler.pkl',
            'y_scaler': 'y_scaler.pkl',
            'gpu_inference': False
        },
        'performance': {
            'use_multiprocessing': True,
            'num_workers': 'auto',
            'chunk_buffer_size': 10,
            'use_gpu': False,
            'gpu_device_id': 0,
            'gpu_memory_fraction': 0.8,
            'gpu_batch_size': 32,
            'cache_enabled': True,
            'cache_maxsize': 4
        },
        'validation': {
            'validate_inputs': True,
            'check_memory': True,
            'max_sequence_length': 10000000,
            'min_sequence_length': 100,
            'check_gpu_memory': False
        },
        'debug': {
            'verbose': True,
            'log_level': 'INFO',
            'profile_performance': False,
            'profile_gpu': False,
            'save_intermediate': False
        }
    }
    
    return config

def generate_gpu_config(output_file: str = "pipeline_config_gpu.yaml") -> Dict[str, Any]:
    """Generate GPU-optimized configuration."""
    
    config = generate_basic_config(output_file)
    
    # Update for GPU
    config['pipeline']['description'] = 'GPU-accelerated alignment pipeline'
    config['alignment'].update({
        'window_size_predictor': 'gpu',
        'use_gpu_acceleration': True,
        'gpu_beam_pruning': True,
        'gpu_score_calculation': True
    })
    
    config['chunking'].update({
        'use_gpu_anchoring': True,
        'use_gpu_pruning': True,
        'gpu_batch_size': 64,
        'gpu_memory_limit_gb': 8.0
    })
    
    config['seeding'].update({
        'gpu_strobe_generation': True,
        'gpu_index_building': True
    })
    
    config['models'].update({
        'gpu_inference': True,
        'mixed_precision': True
    })
    
    config['performance'].update({
        'use_gpu': True,
        'gpu_device_id': 0,
        'gpu_memory_fraction': 0.8,
        'gpu_batch_size': 32,
        'cache_gpu_tensors': True
    })
    
    config['validation'].update({
        'check_gpu_memory': True,
        'min_gpu_memory_gb': 4.0,
        'validate_gpu_operations': True
    })
    
    config['debug'].update({
        'profile_performance': True,
        'profile_gpu': True,
        'plot_gpu_utilization': True
    })
    
    # Add GPU optimization section
    config['gpu_optimization'] = {
        'use_tensor_cores': True,
        'use_cudnn': True,
        'memory_pinning': True,
        'async_data_transfer': True,
        'dynamic_batching': True,
        'auto_tune_batch_size': True,
        'compute_precision': 'float32',
        'accumulate_precision': 'float32'
    }
    
    config['monitoring'] = {
        'monitor_gpu_usage': True,
        'monitor_memory': True,
        'log_interval_seconds': 5,
        'gpu_memory_threshold': 0.9,
        'gpu_utilization_threshold': 0.8
    }
    
    return config

def generate_custom_config(**kwargs) -> Dict[str, Any]:
    """Generate custom configuration based on parameters."""
    
    # Start with basic config
    if kwargs.get('use_gpu', False):
        config = generate_gpu_config()
    else:
        config = generate_basic_config()
    
    # Apply customizations
    if 'sequence_length' in kwargs:
        length = kwargs['sequence_length']
        if length > 1000000:
            config['chunking']['default_chunk_size'] = 10000
            config['performance']['gpu_batch_size'] = 16
        elif length < 10000:
            config['chunking']['default_chunk_size'] = 1000
            config['performance']['gpu_batch_size'] = 64
    
    if 'memory_limit' in kwargs:
        memory_gb = kwargs['memory_limit']
        if memory_gb < 4:
            config['performance']['gpu_memory_fraction'] = 0.5
            config['performance']['gpu_batch_size'] = 16
        elif memory_gb >= 16:
            config['performance']['gpu_memory_fraction'] = 0.9
            config['performance']['gpu_batch_size'] = 128
    
    if 'accuracy_mode' in kwargs:
        if kwargs['accuracy_mode'] == 'high':
            config['alignment']['beam_width'] = 200
            config['alignment']['gap_open'] = -40
            config['alignment']['gap_extend'] = -0.2
        elif kwargs['accuracy_mode'] == 'fast':
            config['alignment']['beam_width'] = 50
            config['alignment']['gap_open'] = -20
            config['alignment']['gap_extend'] = -1.0
    
    return config

def save_config(config: Dict[str, Any], output_file: str):
    """Save configuration to file."""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Configuration saved to: {output_path}")
    
    # Also save as JSON for reference
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration."""
    
    errors = []
    
    # Check required sections
    required_sections = ['alignment', 'chunking', 'performance', 'io']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Check alignment parameters
    if 'alignment' in config:
        align = config['alignment']
        if align.get('gap_open', 0) >= 0:
            errors.append("gap_open should be negative")
        if align.get('gap_extend', 0) >= 0:
            errors.append("gap_extend should be negative")
        if align.get('beam_width', 0) <= 0:
            errors.append("beam_width should be positive")
    
    # Check GPU settings if enabled
    if config.get('performance', {}).get('use_gpu', False):
        if not config.get('alignment', {}).get('use_gpu_acceleration', False):
            errors.append("GPU enabled in performance but not in alignment")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  ❌ {error}")
        return False
    
    print("✅ Configuration validation passed")
    return True

def main():
    """Main configuration generator function."""
    
    parser = argparse.ArgumentParser(
        description="Configuration Generator for Alignment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --basic                        # Generate basic CPU config
  %(prog)s --gpu                          # Generate GPU-optimized config
  %(prog)s --custom --seq-length 1000000  # Generate custom config
  %(prog)s --validate my_config.yaml      # Validate existing config
        """
    )
    
    parser.add_argument(
        '--basic',
        action='store_true',
        help='Generate basic CPU-only configuration'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Generate GPU-accelerated configuration'
    )
    
    parser.add_argument(
        '--custom',
        action='store_true',
        help='Generate custom configuration'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='config/pipeline_config.yaml',
        help='Output configuration file path'
    )
    
    parser.add_argument(
        '--seq-length',
        type=int,
        help='Expected sequence length for tuning'
    )
    
    parser.add_argument(
        '--memory-limit',
        type=float,
        help='Memory limit in GB for tuning'
    )
    
    parser.add_argument(
        '--accuracy-mode',
        choices=['high', 'balanced', 'fast'],
        default='balanced',
        help='Accuracy vs speed trade-off'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Enable GPU acceleration in custom config'
    )
    
    parser.add_argument(
        '--validate',
        type=str,
        help='Validate existing configuration file'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show generated configuration without saving'
    )
    
    args = parser.parse_args()
    
    # Validate existing config if requested
    if args.validate:
        config_path = Path(args.validate)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return 1
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if validate_config(config):
            print(f"✅ Configuration is valid: {config_path}")
            return 0
        else:
            return 1
    
    # Generate new configuration
    if args.custom:
        print("Generating custom configuration...")
        config = generate_custom_config(
            sequence_length=args.seq_length,
            memory_limit=args.memory_limit,
            accuracy_mode=args.accuracy_mode,
            use_gpu=args.use_gpu
        )
    elif args.gpu:
        print("Generating GPU-optimized configuration...")
        config = generate_gpu_config(args.output)
    else:
        print("Generating basic CPU configuration...")
        config = generate_basic_config(args.output)
    
    # Validate generated config
    if not validate_config(config):
        return 1
    
    # Show or save configuration
    if args.show:
        print("\n" + "=" * 70)
        print("GENERATED CONFIGURATION")
        print("=" * 70)
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))
        print("=" * 70)
    else:
        save_config(config, args.output)
        
        # Print usage instructions
        print("\n" + "=" * 70)
        print("CONFIGURATION READY")
        print("=" * 70)
        print(f"To use this configuration:")
        print(f"  align-pipeline --config {args.output}")
        print("\nTo run with GPU acceleration (if configured):")
        print(f"  align-pipeline --config {args.output} --use-gpu")
        print("\nTo generate a GPU-specific config:")
        print("  align-config --gpu --output config/pipeline_config_gpu.yaml")
        print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())