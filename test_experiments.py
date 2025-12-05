"""
Quick test script to verify the experiments.py works correctly
"""

import subprocess
import sys

def run_test(description, command):
    """Run a test command"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(command, check=True, timeout=120)
        print(f"\n✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED (exit code {e.returncode})")
        return False
    except subprocess.TimeoutExpired:
        print(f"\n⚠️  {description} - TIMEOUT (taking too long)")
        return False
    except Exception as e:
        print(f"\n❌ {description} - ERROR: {str(e)}")
        return False

def main():
    print("\n" + "="*60)
    print("EXPERIMENTS.PY TEST SUITE")
    print("="*60)
    print("\nThis will run quick tests to verify experiments.py works.")
    print("Each test uses minimal epochs and batch size for speed.\n")
    
    tests = [
        (
            "Single experiment (2 epochs)",
            [sys.executable, "experiments.py", "--mode", "single", 
             "--epochs", "2", "--batch-size", "64", "--no-wandb"]
        ),
        (
            "Activation comparison (1 run, 2 epochs)",
            [sys.executable, "experiments.py", "--mode", "compare", 
             "--compare-type", "activation", "--runs", "1", 
             "--epochs", "2", "--batch-size", "64", "--no-wandb"]
        ),
    ]
    
    results = []
    for description, command in tests:
        results.append(run_test(description, command))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (description, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"{status} - {description}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! experiments.py is ready to use.")
        print("\nNext steps:")
        print("1. Run: wandb login")
        print("2. See EXPERIMENTS_QUICKSTART.md for common commands")
        print("3. See EXPERIMENTS_GUIDE.md for detailed documentation")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
