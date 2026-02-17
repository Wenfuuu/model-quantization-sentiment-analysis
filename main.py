import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).parent


def main():
    print("\n" + "=" * 60)
    print("  MODEL QUANTIZATION & SENTIMENT ANALYSIS")
    print("=" * 60)

    print("\n  What would you like to run?")
    print("  [1] PTQ (Post-Training Quantization)")
    print("  [2] QAT (Quantization-Aware Training)")
    print("  [3] XAI (Explainability Analysis)")

    choice = input("\n  Enter choice (1/2/3): ").strip()

    if choice == "1":
        subprocess.run([sys.executable, str(ROOT_DIR / "scripts" / "run_ptq.py")])
    elif choice == "2":
        subprocess.run([sys.executable, str(ROOT_DIR / "scripts" / "run_qat.py")])
    elif choice == "3":
        subprocess.run([sys.executable, str(ROOT_DIR / "scripts" / "run_xai.py")])
    else:
        print("\n  Invalid choice.")


if __name__ == "__main__":
    main()
