"""
Master Execution Script for QM640 Experiments.

Allows running individual experiments or the full suite.
Usage:
    python scripts/run_experiments.py --rq1
    python scripts/run_experiments.py --all
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Define experiment scripts
EXPERIMENTS = {
    'rq1': 'scripts/run_domain_shift.py',
    'rq2': 'scripts/run_fine_tuning.py',
    'rq3': 'scripts/run_robustness.py',
    'rq4': 'scripts/run_label_efficiency.py',
}

def run_script(script_path):
    """Run a python script as a subprocess."""
    print(f"\n{'='*60}")
    print(f"Running: {script_path}")
    print(f"{'='*60}\n")
    
    try:
        # Use the same python interpreter
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        return False
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run QM640 Capstone Experiments")
    
    parser.add_argument('--all', action='store_true', help='Run all experiments (RQ1-RQ4)')
    parser.add_argument('--rq1', action='store_true', help='Run RQ1: Domain Shift Analysis')
    parser.add_argument('--rq2', action='store_true', help='Run RQ2: Fine-Tuning Analysis')
    parser.add_argument('--rq3', action='store_true', help='Run RQ3: Robustness Analysis')
    parser.add_argument('--rq4', action='store_true', help='Run RQ4: Label Efficiency Analysis')
    parser.add_argument('--data', action='store_true', help='Download and preprocess data only')
    
    args = parser.parse_args()
    
    # Base directory
    base_dir = Path(__file__).parent.parent
    
    # If no args provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # 1. Data Setup (always run if explicitly requested or ensuring dependencies)
    if args.data:
        run_script('scripts/download_data.py')
        return

    # Run experiments
    scripts_to_run = []
    
    if args.all:
        scripts_to_run = [EXPERIMENTS[key] for key in sorted(EXPERIMENTS.keys())]
    else:
        if args.rq1: scripts_to_run.append(EXPERIMENTS['rq1'])
        if args.rq2: scripts_to_run.append(EXPERIMENTS['rq2'])
        if args.rq3: scripts_to_run.append(EXPERIMENTS['rq3'])
        if args.rq4: scripts_to_run.append(EXPERIMENTS['rq4'])
    
    # Execute
    success_count = 0
    for script in scripts_to_run:
        full_path = str(base_dir / script)
        if run_script(full_path):
            success_count += 1
        else:
            print(f"Stopping execution due to error in {script}")
            sys.exit(1)
            
    print(f"\nCompleted {success_count}/{len(scripts_to_run)} experiments successfully.")

if __name__ == "__main__":
    main()
