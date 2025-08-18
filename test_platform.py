"""
Test the new ShapleyIQ platform implementation
测试新的ShapleyIQ平台实现
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from shapleyiq.platform.data_loader import load_traces, load_metrics, load_logs
from shapleyiq.platform.algorithms import ShapleyRCA, MicroHECL, MicroRCA, MicroRank, TON
from shapleyiq.platform.interface import AlgorithmArgs


def test_data_loading(data_path: Path):
    """Test data loading functionality"""
    print(f"Testing data loading from {data_path}")
    
    try:
        # Test traces loading
        print("Loading traces...")
        traces_lf = load_traces(data_path)
        print(f"✓ Traces loaded successfully: {type(traces_lf)}")
        
        # Test metrics loading (optional)
        try:
            print("Loading metrics...")
            metrics_lf = load_metrics(data_path)
            print(f"✓ Metrics loaded successfully: {type(metrics_lf)}")
        except Exception as e:
            print(f"⚠ Metrics not available: {e}")
            metrics_lf = None
        
        # Test logs loading (optional)
        try:
            print("Loading logs...")
            logs_lf = load_logs(data_path)
            print(f"✓ Logs loaded successfully: {type(logs_lf)}")
        except Exception as e:
            print(f"⚠ Logs not available: {e}")
            logs_lf = None
            
        return traces_lf, metrics_lf, logs_lf
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_algorithm(algorithm_class, args: AlgorithmArgs):
    """Test a single algorithm"""
    algorithm_name = algorithm_class.__name__
    print(f"\nTesting {algorithm_name}...")
    
    try:
        # Create algorithm instance
        alg = algorithm_class()
        
        # Run algorithm
        results = alg(args)
        
        print(f"✓ {algorithm_name} completed successfully")
        print(f"  - Returned {len(results)} result sets")
        
        if results:
            result = results[0]
            print(f"  - Top 5 results: {result.ranks[:5]}")
            if result.scores:
                top_scores = {name: result.scores[name] for name in result.ranks[:5] if name in result.scores}
                print(f"  - Top scores: {top_scores}")
                
        return True
        
    except Exception as e:
        print(f"✗ {algorithm_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("=" * 60)
    print("ShapleyIQ Platform Implementation Test")
    print("=" * 60)
    
    # Use the sample data
    data_path = Path("test/ts1-ts-route-plan-service-request-replace-method-qtbhzt")
    
    if not data_path.exists():
        print(f"Error: Test data directory {data_path} not found")
        print("Please make sure you have the sample data in the test directory")
        return False
    
    # Test data loading
    traces_lf, metrics_lf, logs_lf = test_data_loading(data_path)
    
    if traces_lf is None:
        print("Cannot proceed without traces data")
        return False
    
    # Create algorithm arguments
    args = AlgorithmArgs(
        input_folder=data_path,
        traces=traces_lf,
        metrics=metrics_lf,
        logs=logs_lf
    )
    
    # Test all algorithms
    algorithms = [ShapleyRCA, MicroHECL, MicroRCA, MicroRank, TON]
    success_count = 0
    
    print("\n" + "=" * 60)
    print("Testing Algorithms")
    print("=" * 60)
    
    for algorithm_class in algorithms:
        if test_algorithm(algorithm_class, args):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Algorithms tested: {len(algorithms)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(algorithms) - success_count}")
    
    if success_count == len(algorithms):
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
