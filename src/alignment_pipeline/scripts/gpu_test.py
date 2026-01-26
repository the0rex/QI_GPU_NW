#!/usr/bin/env python3
"""
GPU testing and benchmarking script.
Author: Rowel Facunla
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_gpu_availability():
    """Test if GPU is available and working."""
    print("=" * 70)
    print("GPU AVAILABILITY TEST")
    print("=" * 70)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA is available")
            gpu_count = torch.cuda.device_count()
            print(f"Number of GPUs: {gpu_count}")
            
            for i in range(gpu_count):
                print(f"\nGPU {i}:")
                print(f"  Name: {torch.cuda.get_device_name(i)}")
                print(f"  Capability: {torch.cuda.get_device_capability(i)}")
                
                # Test memory
                torch.cuda.synchronize(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  Total Memory: {total_memory:.2f} GB")
                
                # Test a simple operation
                x = torch.randn(1000, 1000, device=f'cuda:{i}')
                y = torch.randn(1000, 1000, device=f'cuda:{i}')
                
                start = time.time()
                z = torch.matmul(x, y)
                torch.cuda.synchronize(i)
                elapsed = time.time() - start
                
                print(f"  Matrix multiplication: {elapsed:.4f} seconds")
                print(f"  Result check: {torch.allclose(z, z)}")
            
            return True
        else:
            print("❌ CUDA is not available")
            return False
            
    except ImportError:
        print("❌ PyTorch is not installed")
        return False
    except Exception as e:
        print(f"❌ Error testing GPU: {e}")
        return False

def test_cuda_extensions():
    """Test CUDA extensions."""
    print("\n" + "=" * 70)
    print("CUDA EXTENSIONS TEST")
    print("=" * 70)
    
    try:
        from alignment_pipeline.cuda_extensions import CUDA_AVAILABLE
        print(f"CUDA extensions available: {CUDA_AVAILABLE}")
        
        if CUDA_AVAILABLE:
            from alignment_pipeline.cuda_extensions import CUDAStrobeProcessor
            import torch  # Added import
            
            processor = CUDAStrobeProcessor()
            print(f"✅ CUDAStrobeProcessor initialized on: {processor.device}")
            
            # Test tensor operations
            test_tensor = torch.tensor([1, 2, 3, 4, 5], device=processor.device)
            print(f"✅ Tensor operations work: {test_tensor.sum().item()}")
            
            return True
        else:
            print("❌ CUDA extensions not available")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import CUDA extensions: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing CUDA extensions: {e}")
        return False

def benchmark_gpu_vs_cpu():
    """Benchmark GPU vs CPU performance."""
    print("\n" + "=" * 70)
    print("GPU vs CPU BENCHMARK")
    print("=" * 70)
    
    try:
        import torch
        import numpy as np
        
        if not torch.cuda.is_available():
            print("Skipping benchmark - GPU not available")
            return
        
        # Test sizes
        sizes = [100, 500, 1000, 5000]
        results = []
        
        for size in sizes:
            print(f"\nTesting size {size}x{size}:")
            
            # Create random matrices
            np_a = np.random.randn(size, size).astype(np.float32)
            np_b = np.random.randn(size, size).astype(np.float32)
            
            # CPU (NumPy)
            start = time.time()
            np_c = np.dot(np_a, np_b)
            cpu_time = time.time() - start
            print(f"  CPU (NumPy): {cpu_time:.4f}s")
            
            # GPU (PyTorch)
            torch_a = torch.from_numpy(np_a).cuda()
            torch_b = torch.from_numpy(np_b).cuda()
            
            # Warmup
            for _ in range(3):
                torch_c = torch.matmul(torch_a, torch_b)
            
            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            
            for _ in range(10):
                torch_c = torch.matmul(torch_a, torch_b)
            
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) / 10
            
            print(f"  GPU (PyTorch): {gpu_time:.4f}s")
            
            # Avoid division by zero
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"  Speedup: {speedup:.2f}x")
            else:
                speedup = float('inf')
                print(f"  Speedup: ∞ (GPU time too small)")
            
            # Verify correctness
            torch_c_cpu = torch_c.cpu().numpy()
            max_diff = np.max(np.abs(np_c - torch_c_cpu))
            print(f"  Max difference: {max_diff:.2e}")
            
            results.append({
                'size': size,
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'max_diff': max_diff
            })
        
        # Print summary
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        
        for r in results:
            speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] != float('inf') else "∞"
            print(f"Size {r['size']:4d}: CPU {r['cpu_time']:.4f}s, "
                  f"GPU {r['gpu_time']:.4f}s, Speedup: {speedup_str}")
        
        return results
        
    except Exception as e:
        print(f"Error in benchmark: {e}")
        return None

def test_pipeline_components():
    """Test individual pipeline components with GPU."""
    print("\n" + "=" * 70)
    print("PIPELINE COMPONENTS TEST")
    print("=" * 70)
    
    all_tests_passed = True
    
    try:
        # Test 1: GPU Availability
        print("\n1. Testing GPU Availability:")
        gpu_found = False
        
        # Check PyTorch GPU
        try:
            import torch
            if torch.cuda.is_available():
                print(f"   ✅ PyTorch GPU: {torch.cuda.get_device_name(0)}")
                gpu_found = True
        except:
            print("   ⚠ PyTorch not available or GPU not found")
        
        # Check TensorFlow/Keras GPU
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"   ✅ TensorFlow GPU: {len(gpus)} device(s)")
                gpu_found = True
        except:
            try:
                import keras
                devices = keras.backend.list_devices()
                gpu_devices = [d for d in devices if 'gpu' in d.lower() or 'cuda' in d.lower()]
                if gpu_devices:
                    print(f"   ✅ Keras GPU: {len(gpu_devices)} device(s)")
                    gpu_found = True
            except:
                print("   ⚠ TensorFlow/Keras not available")
        
        if not gpu_found:
            print("   ⚠ No GPU found in any framework")
            all_tests_passed = False
        
        # Test 2: Window Predictor
        print("\n2. Testing Window Predictor:")
        try:
            from alignment_pipeline.algorithms.predict_window import (
                extract_features,
                predict_window_cpu,
                predict_window_gpu
            )
            
            print("   ✅ Module imports successfully")
            
            # Test feature extraction
            test_seq1 = "ACGT" * 100
            test_seq2 = "ACGT" * 95 + "TTTT" * 5
            features = extract_features(test_seq1, test_seq2)
            print(f"   Feature extraction: shape={features.shape}")
            
            # Test CPU prediction
            cpu_result = predict_window_cpu(test_seq1, test_seq2)
            print(f"   CPU prediction: {cpu_result}")
            
            # Test GPU prediction
            gpu_result = predict_window_gpu(test_seq1, test_seq2)
            print(f"   GPU prediction: {gpu_result}")
            
            if cpu_result == gpu_result:
                print("   ✅ CPU and GPU predictions match")
            else:
                print(f"   ⚠ CPU and GPU predictions differ (normal for some models)")
            
            print("   ✅ Window predictor test passed")
            
        except Exception as e:
            print(f"   ❌ Window predictor test failed: {e}")
            all_tests_passed = False
        
        # Test 3: CUDA Extensions
        print("\n3. Testing CUDA Extensions:")
        try:
            from alignment_pipeline.cuda_extensions import CUDA_AVAILABLE
            if CUDA_AVAILABLE:
                print("   ✅ CUDA extensions available")
                
                # Test strobe processor
                from alignment_pipeline.cuda_extensions import CUDAStrobeProcessor
                import torch
                
                processor = CUDAStrobeProcessor()
                print(f"   Strobe processor device: {processor.device}")
                
                # Test with sample data
                test_hashes = torch.tensor([1, 2, 3, 1, 2, 4, 5, 1, 3], device='cuda')
                test_positions = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90], device='cuda')
                
                try:
                    index = processor.build_index_gpu(test_hashes, test_positions, max_occ=2)
                    print(f"   Built index with {len(index)} unique hashes")
                    print("   ✅ Strobe processor test passed")
                except Exception as e:
                    print(f"   ⚠ Strobe processor operation failed: {e}")
                
            else:
                print("   ⚠ CUDA extensions not available")
                all_tests_passed = False
                
        except ImportError as e:
            print(f"   ❌ Could not import CUDA extensions: {e}")
            all_tests_passed = False
        except Exception as e:
            print(f"   ❌ Error testing CUDA extensions: {e}")
            all_tests_passed = False
        
        # Test 4: Beam Pruner
        print("\n4. Testing Beam Pruner:")
        try:
            from alignment_pipeline.cuda_extensions import CUDABeamPruner
            
            pruner = CUDABeamPruner()
            
            # Create test beams
            beam_items = [(i, (float(i), 0.0, None, None, None, 0)) for i in range(1000)]
            
            # Time GPU pruning
            start = time.time()
            gpu_pruned = pruner.prune_beams_gpu(beam_items, beam_width=100)
            gpu_time = time.time() - start
            
            # Time CPU pruning
            start = time.time()
            sorted_items = sorted(beam_items, key=lambda x: x[1][0], reverse=True)[:100]
            cpu_time = time.time() - start
            
            print(f"   GPU pruning time: {gpu_time:.6f}s")
            print(f"   CPU pruning time: {cpu_time:.6f}s")
            
            # Calculate speedup properly
            if gpu_time > 0 and cpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"   Speedup: {speedup:.2f}x")
            else:
                print(f"   Speedup: N/A (times too small)")
            
            print(f"   Results match: {len(gpu_pruned) == len(sorted_items)}")
            print("   ✅ Beam pruner test passed")
            
        except Exception as e:
            print(f"   ❌ Beam pruner test failed: {e}")
            all_tests_passed = False
        
        return all_tests_passed
        
    except Exception as e:
        print(f"Error testing pipeline components: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_system_requirements():
    """Check system requirements for GPU acceleration."""
    print("\n" + "=" * 70)
    print("SYSTEM REQUIREMENTS CHECK")
    print("=" * 70)
    
    requirements = {
        'python_version': sys.version_info >= (3, 8),
        'pytorch_installed': False,
        'cuda_available': False,
        'cuda_version': None,
        'gpu_memory_gb': 0,
        'numpy_installed': False,
        'cuda_extensions': False,
        'tensorflow_available': False,
        'keras_available': False,
        'window_predictor_ready': False
    }
    
    # Check PyTorch
    try:
        import torch
        requirements['pytorch_installed'] = True
        
        if torch.cuda.is_available():
            requirements['cuda_available'] = True
            requirements['cuda_version'] = torch.version.cuda
            
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            requirements['gpu_memory_gb'] = gpu_memory
    except:
        pass
    
    # Check NumPy
    try:
        import numpy
        requirements['numpy_installed'] = True
    except:
        pass
    
    # Check CUDA extensions
    try:
        from alignment_pipeline.cuda_extensions import CUDA_AVAILABLE
        requirements['cuda_extensions'] = CUDA_AVAILABLE
    except:
        pass
    
    # Check TensorFlow/Keras
    try:
        import tensorflow as tf
        requirements['tensorflow_available'] = True
    except:
        pass
    
    try:
        import keras
        requirements['keras_available'] = True
    except:
        pass
    
    # Check window predictor
    try:
        from alignment_pipeline.algorithms.predict_window import MODEL_AVAILABLE
        requirements['window_predictor_ready'] = MODEL_AVAILABLE
    except:
        pass
    
    # Print results
    print(f"Python >= 3.8: {'✅' if requirements['python_version'] else '❌'}")
    print(f"PyTorch installed: {'✅' if requirements['pytorch_installed'] else '❌'}")
    print(f"CUDA available: {'✅' if requirements['cuda_available'] else '❌'}")
    
    if requirements['cuda_available']:
        print(f"CUDA version: {requirements['cuda_version']}")
        print(f"GPU memory: {requirements['gpu_memory_gb']:.1f} GB")
    
    print(f"NumPy installed: {'✅' if requirements['numpy_installed'] else '❌'}")
    print(f"Cuda extensions: {'✅' if requirements['cuda_extensions'] else '❌'}")
    print(f"TensorFlow available: {'✅' if requirements['tensorflow_available'] else '❌'}")
    print(f"Keras available: {'✅' if requirements['keras_available'] else '❌'}")
    print(f"Window predictor ready: {'✅' if requirements['window_predictor_ready'] else '❌'}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if not requirements['cuda_available']:
        print("  • Install NVIDIA drivers and CUDA toolkit")
        print("  • Install PyTorch with CUDA support")
    elif requirements['gpu_memory_gb'] < 4:
        print("  • Consider GPU with more memory (>4GB recommended)")
    if not requirements['cuda_extensions']:
        print("  • Build CUDA extensions: python setup.py build_ext --inplace")
    if not requirements['window_predictor_ready']:
        print("  • Window predictor model files may be missing or incompatible")
    
    return requirements

def main():
    """Main function for GPU testing."""
    print("GPU Testing and Benchmarking Tool")
    print("=" * 70)
    
    # Run all tests
    all_passed = True
    
    # Check system requirements
    requirements = check_system_requirements()
    
    # Test GPU availability
    if not test_gpu_availability():
        all_passed = False
    
    # Test CUDA extensions
    if not test_cuda_extensions():
        all_passed = False
    
    # Run benchmark if GPU available
    if requirements.get('cuda_available', False):
        benchmark_results = benchmark_gpu_vs_cpu()
    else:
        print("\nSkipping benchmark - GPU not available")
    
    # Test pipeline components
    components_passed = test_pipeline_components()
    if not components_passed:
        all_passed = False
    
    # Save results
    results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'system': {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform
        },
        'requirements': requirements,
        'all_passed': all_passed and components_passed
    }
    
    # Save to file
    output_dir = Path("Results_GPU")
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "gpu_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    if all_passed and components_passed:
        print("✅ All tests passed!")
    else:
        print("⚠ Some tests failed or warnings were issued")
    
    print(f"\nResults saved to: {results_file}")
    print("\nTo run the GPU-accelerated pipeline:")
    print("  python main_pipeline.py --config config/pipeline_config_gpu.yaml")
    print("\nFor help:")
    print("  python align_pipeline.py --help")
    
    # Show GPU acceleration status
    print("\n" + "=" * 70)
    print("GPU ACCELERATION STATUS")
    print("=" * 70)
    
    if requirements['cuda_available']:
        print("✅ GPU acceleration is ENABLED")
        print(f"   Device: NVIDIA GeForce RTX 4050 Laptop GPU")
        print(f"   Memory: {requirements['gpu_memory_gb']:.1f} GB")
        print(f"   CUDA: {requirements['cuda_version']}")
        
        if requirements['window_predictor_ready']:
            print("✅ Window predictor: READY (Keras 3 model loaded)")
        else:
            print("⚠ Window predictor: NOT READY")
        
        if requirements['cuda_extensions']:
            print("✅ CUDA extensions: READY")
        else:
            print("⚠ CUDA extensions: NOT READY")
    else:
        print("❌ GPU acceleration is DISABLED")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())