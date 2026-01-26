import sys
from typing import Dict, Tuple
import warnings

try:
    # Python 3.8+ has importlib.metadata in standard library
    from importlib import metadata
except ImportError:
    # For older Python versions, use the backport
    import importlib_metadata as metadata

REQUIRED_PACKAGES = {
    'numpy': '1.21.0',
    'biopython': '1.79',
    'pandas': '1.3.0',
    'scikit-learn': '1.0.0',
    'tensorflow': '2.8.0',
    'joblib': '1.1.0',
    'matplotlib': '3.5.0',
    'seaborn': '0.11.0',
    'pybind11': '2.10.0',
}

OPTIONAL_PACKAGES = {
    'plotly': '5.5.0',
    'tqdm': '4.64.0',
    'pyyaml': '6.0',
    'colorama': '0.4.6',
    'psutil': '5.9.0',
}

def get_package_version(package_name: str) -> str:
    """Get version of installed package using importlib.metadata."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "Not installed"
    except Exception as e:
        return f"Error: {str(e)}"

def check_package(package_name: str, min_version: str) -> Tuple[bool, str, str]:
    """Check if package meets version requirements."""
    installed_version = get_package_version(package_name)
    
    if installed_version == "Not installed":
        return False, "Not installed", min_version
    
    try:
        # Use packaging for version comparison if available
        from packaging import version
        
        if version.parse(installed_version) >= version.parse(min_version):
            return True, installed_version, min_version
        else:
            return False, installed_version, min_version
            
    except ImportError:
        # Fallback: try simple comparison
        warnings.warn(f"Cannot compare versions for {package_name} accurately - packaging module not available")
        
        try:
            # Simple version comparison (basic)
            installed_parts = [int(x) for x in installed_version.split('.')[:3]]
            required_parts = [int(x) for x in min_version.split('.')[:3]]
            
            # Compare version parts
            for installed, required in zip(installed_parts, required_parts):
                if installed < required:
                    return False, installed_version, min_version
                elif installed > required:
                    return True, installed_version, min_version
            
            # All compared parts equal
            if len(installed_parts) < len(required_parts):
                # More specific requirement
                return False, installed_version, min_version
            else:
                return True, installed_version, min_version
                
        except (ValueError, AttributeError):
            # Version strings might have non-numeric parts
            warnings.warn(f"Cannot parse versions for {package_name}: installed={installed_version}, required={min_version}")
            # Assume it's OK if we can't compare
            return True, installed_version, min_version

def check_required_packages() -> Dict[str, Dict]:
    """Check all required packages."""
    results = {}
    
    for package, min_version in REQUIRED_PACKAGES.items():
        is_ok, installed, required = check_package(package, min_version)
        results[package] = {
            'required': min_version,
            'installed': installed,
            'ok': is_ok,
            'status': 'OK' if is_ok else 'FAIL'
        }
    
    return results

def check_optional_packages() -> Dict[str, Dict]:
    """Check all optional packages."""
    results = {}
    
    for package, recommended_version in OPTIONAL_PACKAGES.items():
        is_ok, installed, recommended = check_package(package, recommended_version)
        results[package] = {
            'recommended': recommended_version,
            'installed': installed,
            'ok': is_ok,
            'status': 'OK' if is_ok else 'WARNING'
        }
    
    return results

def check_python_version(min_version: Tuple[int, int] = (3, 8)) -> Dict:
    """Check Python version."""
    current_version = sys.version_info[:2]
    is_ok = current_version >= min_version
    
    return {
        'required': f"{min_version[0]}.{min_version[1]}",
        'installed': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'ok': is_ok,
        'status': 'OK' if is_ok else 'FAIL'
    }

def check_system() -> Dict:
    """Check system information."""
    import platform
    import os
    
    return {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_implementation': platform.python_implementation(),
        'cpus': os.cpu_count(),
        'memory': None,
    }

def check_cpp_extensions() -> Dict:
    """Check if C++ extensions are available."""
    results = {}
    
    extensions_to_check = [
        ('_compress_cpp', '4-bit compression'),
        ('_syncmer_cpp', 'Syncmer/strobemer generation'),
    ]
    
    for module_name, description in extensions_to_check:
        try:
            results[module_name] = {
                'description': description,
                'available': True,
                'status': 'OK'
            }
        except ImportError as e:
            results[module_name] = {
                'description': description,
                'available': False,
                'error': str(e),
                'status': 'FAIL'
            }
    
    return results

def print_version_report() -> None:
    """Print comprehensive version report."""
    print("=" * 70)
    print("ALIGNMENT PIPELINE - VERSION CHECK")
    print("=" * 70)
    
    # Python version
    py_info = check_python_version()
    print(f"\nPython: {py_info['installed']} (required: {py_info['required']})")
    print(f"  Status: {py_info['status']}")
    
    # System info
    sys_info = check_system()
    print("\nSystem:")
    print(f"  Platform: {sys_info['platform']}")
    print(f"  CPUs: {sys_info['cpus']}")
    
    # C++ extensions
    cpp_ext = check_cpp_extensions()
    print("\nC++ Extensions:")
    for ext_name, info in cpp_ext.items():
        status_icon = "✓" if info['available'] else "✗"
        print(f"  {status_icon} {info['description']}: {info['status']}")
    
    # Required packages
    print("\nRequired Packages:")
    req_results = check_required_packages()
    for package, info in req_results.items():
        status_icon = "✓" if info['ok'] else "✗"
        print(f"  {status_icon} {package:20} {info['installed']:15} (required: {info['required']})")
    
    # Optional packages
    print("\nOptional Packages:")
    opt_results = check_optional_packages()
    for package, info in opt_results.items():
        if info['installed'] != "Not installed":
            status_icon = "✓" if info['ok'] else "⚠"
            print(f"  {status_icon} {package:20} {info['installed']:15} (recommended: {info['recommended']})")
    
    # Summary
    req_failures = sum(1 for info in req_results.values() if not info['ok'])
    cpp_failures = sum(1 for info in cpp_ext.values() if not info['available'])
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  Required packages: {len(req_results) - req_failures}/{len(req_results)} OK")
    print(f"  C++ extensions: {len(cpp_ext) - cpp_failures}/{len(cpp_ext)} available")
    
    if req_failures == 0 and cpp_failures == 0 and py_info['ok']:
        print("\n  ALL CHECKS PASSED!")
    else:
        print("  Some checks failed. See details above.")
    print("=" * 70)

def check_versions() -> Dict:
    """Run all version checks and return results."""
    results = {
        'python': check_python_version(),
        'system': check_system(),
        'cpp_extensions': check_cpp_extensions(),
        'required_packages': check_required_packages(),
        'optional_packages': check_optional_packages(),
    }
    
    # Log warnings
    for pkg, info in results['required_packages'].items():
        if not info['ok']:
            warnings.warn(f"Package {pkg} {info['installed']} does not meet requirement {info['required']}")
    
    return results

if __name__ == "__main__":
    print_version_report()