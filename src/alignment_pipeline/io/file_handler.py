"""
File handling utilities for the alignment pipeline.
Author: Rowel Facunla
"""

import os
import shutil
import psutil
from pathlib import Path
from typing import Optional, Tuple

def check_disk_space(path: str, required_gb: float = 10.0) -> Tuple[bool, float, float]:
    """
    Check if there is enough disk space at the given path.
    
    Args:
        path: Path to check disk space for
        required_gb: Required space in GB
        
    Returns:
        Tuple of (has_space, available_gb, required_gb)
    """
    try:
        # Get disk usage for the partition containing the path
        usage = psutil.disk_usage(path)
        available_gb = usage.free / (1024**3)  # Convert to GB
        
        return available_gb >= required_gb, available_gb, required_gb
    except Exception as e:
        # If we can't check, assume there's enough space
        print(f"WARNING: Could not check disk space: {e}")
        return True, float('inf'), required_gb

def ensure_directory(path: str, clean: bool = False) -> bool:
    """
    Ensure a directory exists, optionally cleaning it.
    
    Args:
        path: Directory path
        clean: If True, remove existing directory and recreate
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path_obj = Path(path)
        
        if clean and path_obj.exists():
            shutil.rmtree(path)
        
        path_obj.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"ERROR: Could not create directory {path}: {e}")
        return False

def get_file_size(filepath: str, human_readable: bool = True) -> str:
    """
    Get file size in human-readable format.
    
    Args:
        filepath: Path to file
        human_readable: If True, return human-readable string
        
    Returns:
        File size string
    """
    try:
        size_bytes = os.path.getsize(filepath)
        
        if not human_readable:
            return str(size_bytes)
        
        # Convert to human readable
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.1f} PB"
    except Exception:
        return "Unknown"

def find_latest_file(directory: str, pattern: str = "*") -> Optional[str]:
    """
    Find the latest file in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        
    Returns:
        Path to latest file, or None if no files found
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            return None
        
        files = list(directory.glob(pattern))
        if not files:
            return None
        
        # Sort by modification time
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(files[0])
    except Exception:
        return None

def safe_remove(filepath: str) -> bool:
    """
    Safely remove a file if it exists.
    
    Args:
        filepath: Path to file
        
    Returns:
        True if file was removed or didn't exist, False on error
    """
    try:
        path = Path(filepath)
        if path.exists():
            path.unlink()
        return True
    except Exception as e:
        print(f"WARNING: Could not remove file {filepath}: {e}")
        return False

def safe_remove_directory(dirpath: str) -> bool:
    """
    Safely remove a directory if it exists.
    
    Args:
        dirpath: Path to directory
        
    Returns:
        True if directory was removed or didn't exist, False on error
    """
    try:
        path = Path(dirpath)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
        return True
    except Exception as e:
        print(f"WARNING: Could not remove directory {dirpath}: {e}")
        return False

def backup_file(filepath: str, backup_dir: str = "backups") -> Optional[str]:
    """
    Create a backup of a file.
    
    Args:
        filepath: Path to file to backup
        backup_dir: Backup directory
        
    Returns:
        Path to backup file, or None if failed
    """
    try:
        source = Path(filepath)
        if not source.exists():
            return None
        
        # Create backup directory
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Create backup filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"{source.stem}_{timestamp}{source.suffix}"
        
        # Copy file
        shutil.copy2(source, backup_file)
        return str(backup_file)
    except Exception as e:
        print(f"WARNING: Could not backup file {filepath}: {e}")
        return None

def get_memory_usage() -> dict:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory usage information
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024),
            'total_mb': psutil.virtual_memory().total / (1024 * 1024),
        }
    except Exception:
        return {'error': 'Could not get memory info'}

__all__ = [
    'check_disk_space',
    'ensure_directory',
    'get_file_size',
    'find_latest_file',
    'safe_remove',
    'safe_remove_directory',
    'backup_file',
    'get_memory_usage',
]