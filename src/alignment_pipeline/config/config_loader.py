"""
Configuration loader for the alignment pipeline.
Author: Rowel Facunla
"""

import os
import yaml
from typing import Dict, Any, Optional

class ConfigLoader:
    """Loads and manages configuration from YAML files."""
    
    def __init__(self, config_path: str = "default_config.yaml"):
        self.config_path = config_path
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            self.config = self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error loading config file: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if YAML file is not found."""
        return {
            'alignment': {
                'gap_open': -30,
                'gap_extend': -0.5,
                'match_score': 5,
                'mismatch_score': -40,
                'beam_width': 30,
                'carry_gap_state': True,
                'nw_affine': {
                    'tau': 3.0,
                    'energy_match': -5.0,
                    'energy_mismatch': 40.0,
                    'energy_gap_open': 30.0,
                    'energy_gap_extend': 0.5,
                    'carry_gap_penalty': True
                }
            },
            'chunking': {
                'default_chunk_size': 10000,
                'overlap': 500,
                'min_chunk_size': 500,
                'max_chunk_size': 50000,
                'use_gpu_anchoring': True,
                'use_gpu_pruning': True
            }
        }
    
    def get_alignment_params(self) -> Dict[str, Any]:
        """Get alignment parameters."""
        return self.config.get('alignment', {})
    
    def get_chunking_params(self) -> Dict[str, Any]:
        """Get chunking parameters."""
        return self.config.get('chunking', {})
    
    def get_nw_affine_params(self) -> Dict[str, Any]:
        """Get NW affine specific parameters."""
        alignment = self.config.get('alignment', {})
        return alignment.get('nw_affine', {})
    
    def get_gpu_params(self) -> Dict[str, Any]:
        """Get GPU parameters."""
        return self.config.get('gpu', {})
    
    def get_performance_params(self) -> Dict[str, Any]:
        """Get performance parameters."""
        return self.config.get('performance', {})

# Global config instance
config_loader = ConfigLoader()

def get_config() -> ConfigLoader:
    """Get the global config loader instance."""
    return config_loader

def reload_config(config_path: Optional[str] = None):
    """Reload configuration from file."""
    global config_loader
    if config_path:
        config_loader = ConfigLoader(config_path)
    else:
        config_loader.load_config()