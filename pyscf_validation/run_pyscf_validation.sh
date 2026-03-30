#!/bin/bash
# Optimized PySCF calculation runner - PARALLELIZED VERSION
# - Intelligently parallelizes calculations across multiple molecules
# - Dynamically adjusts thread allocation based on workload
# - Sets CPU governor to performance for optimal frequency scaling
# - Monitors calculation progress
# - Handles persistent HDF5 storage

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "PARALLELIZED PySCF Calculation Runner"
echo "AMD Ryzen 7 PRO 7840U Edition"
echo "Intelligent Multi-Core Parallelization"
echo "=========================================="
echo ""

# Check if running as root for CPU frequency control
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}Warning: Not running as root. CPU frequency optimization will be skipped.${NC}"
    echo "To enable CPU optimization, run with sudo."
    echo ""
    SKIP_CPU_OPT=true
else
    SKIP_CPU_OPT=false
fi

# ============================================================================
# CPU FREQUENCY OPTIMIZATION (AMD P-State EPP)
# ============================================================================

optimize_cpu_frequency() {
    echo -e "${BLUE}Optimizing CPU frequency for AMD Ryzen...${NC}"
    
    # Detect CPU driver
    DRIVER=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null || echo "unknown")
    echo "Detected CPU driver: $DRIVER"
    
    # Get number of CPUs
    NUM_CPUS=$(nproc)
    echo "Detected $NUM_CPUS CPUs (8 cores, 16 threads)"
    
    # Check current governor
    CURRENT_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    echo "Current governor: $CURRENT_GOV"
    
    # Set governor to performance for all CPUs
    echo "Setting governor to 'performance' for all CPUs..."
    for ((i=0; i<$NUM_CPUS; i++)); do
        echo "performance" > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor 2>/dev/null || {
            echo -e "${YELLOW}Warning: Could not set governor for CPU $i${NC}"
        }
    done
    
    echo -e "${GREEN}✓ CPU governor set to performance${NC}"
    
    # Display current frequencies
    echo ""
    echo "Current CPU frequencies:"
    grep "MHz" /proc/cpuinfo | head -8 | awk '{printf "  CPU %d: %.0f MHz\n", NR-1, $4}'
    
    # Show max frequency
    MAX_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo "unknown")
    if [ "$MAX_FREQ" != "unknown" ]; then
        MAX_FREQ_MHZ=$((MAX_FREQ / 1000))
        echo ""
        echo "Maximum CPU frequency: ${MAX_FREQ_MHZ} MHz"
    fi
    
    echo ""
    echo -e "${GREEN}✓ CPU optimization complete${NC}"
    echo "  CPUs will now boost to maximum frequency under load"
    echo ""
}

restore_cpu_frequency() {
    echo ""
    echo -e "${BLUE}Restoring CPU frequency settings...${NC}"
    
    NUM_CPUS=$(nproc)
    
    # Set governor back to powersave
    for ((i=0; i<$NUM_CPUS; i++)); do
        echo "powersave" > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor 2>/dev/null || true
    done
    
    echo -e "${GREEN}✓ CPU governor restored to powersave${NC}"
}

# Trap to restore CPU settings on exit
trap restore_cpu_frequency EXIT

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

setup_environment() {
    echo -e "${BLUE}Setting up environment...${NC}"
    
    # Persistent storage directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Detect and activate virtual environment
    VENV_ACTIVATED=false
    
    # Check for virtual environment in common locations
    for VENV_PATH in "/home/tchapet/VirtualEnv" "$SCRIPT_DIR/venv" "$SCRIPT_DIR/../venv" "$SCRIPT_DIR/../../venv" "$SCRIPT_DIR/.venv" "$SCRIPT_DIR/../.venv" "$HOME/VirtualEnv"; do
        if [ -f "$VENV_PATH/bin/activate" ]; then
            echo "  Found virtual environment: $VENV_PATH"
            source "$VENV_PATH/bin/activate"
            VENV_ACTIVATED=true
            PYTHON_CMD="python"
            echo -e "  ${GREEN}✓ Virtual environment activated${NC}"
            break
        fi
    done
    
    if [ "$VENV_ACTIVATED" = false ]; then
        echo -e "  ${YELLOW}⚠ No virtual environment found${NC}"
        echo "  Using system Python (not recommended)"
        echo ""
        echo "  To create a virtual environment:"
        echo "    cd $SCRIPT_DIR"
        echo "    python3 -m venv venv"
        echo "    source venv/bin/activate"
        echo "    pip install pyscf pandas numpy h5py joblib"
        echo ""
        PYTHON_CMD="python3"
    fi
    
    # Check if joblib is installed (required for parallelization)
    echo ""
    echo "  Checking for joblib (required for parallelization)..."
    JOBLIB_INSTALLED=$($PYTHON_CMD -c "import joblib; print('OK')" 2>/dev/null || echo "MISSING")
    
    if [ "$JOBLIB_INSTALLED" = "OK" ]; then
        echo -e "  ${GREEN}✓ joblib is installed${NC}"
    else
        echo -e "  ${RED}✗ joblib is NOT installed${NC}"
        echo ""
        echo "  Parallelization requires joblib. Install with:"
        echo "    pip install joblib"
        echo ""
        echo "  Without joblib, calculations will run sequentially."
        echo ""
        read -p "  Continue anyway? (y/N): " continue_without_joblib
        if [[ ! $continue_without_joblib =~ ^[Yy]$ ]]; then
            echo "Exiting. Please install joblib first."
            exit 1
        fi
    fi
    
    # Fix PySCF environment to use optimized OpenBLAS
    OPENBLAS_DIR="${HOME}/openblas_zen4_optimized"
    
    # Test if optimized OpenBLAS works
    if [ -d "$OPENBLAS_DIR/lib" ]; then
        # Test library
        TEST_RESULT=$($PYTHON_CMD -c "
import sys
sys.path.insert(0, '')
import os
os.environ['LD_LIBRARY_PATH'] = '$OPENBLAS_DIR/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
try:
    import numpy as np
    print('OK')
except:
    print('FAIL')
" 2>/dev/null)
        
        if [ "$TEST_RESULT" = "OK" ]; then
            export LD_LIBRARY_PATH="$OPENBLAS_DIR/lib:$LD_LIBRARY_PATH"
            echo -e "  ${GREEN}✓ Using optimized OpenBLAS (Zen4)${NC}"
        else
            echo -e "  ${YELLOW}⚠ Optimized OpenBLAS broken, using system BLAS${NC}"
        fi
    else
        echo -e "  ${YELLOW}⚠ Optimized OpenBLAS not found, using system BLAS${NC}"
    fi
    
    # Note: Threading is now dynamically controlled by the Python script
    # We don't set OMP_NUM_THREADS here anymore
    export OPENBLAS_MAIN_FREE=1
    export OMP_PROC_BIND=true
    export OMP_PLACES=cores
    
    export PYSCF_TMPDIR="$SCRIPT_DIR/pyscf_persistent_data"
    
    # Create directories
    mkdir -p "$PYSCF_TMPDIR/optimized_geometries" 2>/dev/null || true
    mkdir -p "$PYSCF_TMPDIR/energy_results" 2>/dev/null || true
    mkdir -p "$SCRIPT_DIR/pyscf_results" 2>/dev/null || true
    
    # Fix permissions if running as root (so user can access files later)
    if [ "$EUID" -eq 0 ] && [ -n "$SUDO_USER" ]; then
        echo "  Fixing directory permissions for user $SUDO_USER..."
        chown -R $SUDO_USER:$SUDO_USER "$PYSCF_TMPDIR" 2>/dev/null || true
        chown -R $SUDO_USER:$SUDO_USER "$SCRIPT_DIR/pyscf_results" 2>/dev/null || true
    fi
    
    echo "  Python: $PYTHON_CMD"
    echo "  Persistent storage: $PYSCF_TMPDIR"
    echo "  Threading: Dynamic (controlled by Python script)"
    
    # Verify OpenBLAS is actually loaded
    echo ""
    echo "  Verifying optimized libraries..."
    $PYTHON_CMD -c "
import numpy as np
config = np.__config__.show()
" 2>&1 | grep -i "blas\|lapack" | head -3 || echo "  (numpy config not available)"
    
    echo -e "${GREEN}✓ Environment configured${NC}"
    echo ""
}

# ============================================================================
# CALCULATION PARAMETERS
# ============================================================================

show_menu() {
    echo ""
    echo "=========================================="
    echo "Calculation Options (PARALLELIZED)"
    echo "=========================================="
    echo ""
    echo "1) Run all default molecules (6 molecules, auto-optimized parallelism)"
    echo "2) Run specific molecules (you choose)"
    echo "3) Run with custom basis set"
    echo "4) Run with manual parallelism control"
    echo "5) Force recalculation (ignore saved results)"
    echo "6) Show calculation status"
    echo "7) Exit"
    echo ""
}

get_molecules() {
    echo ""
    echo "Available molecules:"
    echo "  DMAC-DPS, DMAC-TRZ, 4CzIPN, PXZ-NAI, TPA-APy, BMZ-TZ"
    echo ""
    read -p "Enter molecules (space-separated): " MOLECULES
}

get_basis() {
    echo ""
    echo "Basis set options:"
    echo "  1) def2-svp (recommended: fast and accurate)"
    echo "  2) def2-tzvp (higher accuracy, slower)"
    echo "  3) 3-21g (very fast, lower accuracy)"
    echo ""
    read -p "Select basis set (1-3): " basis_choice
    
    case $basis_choice in
        1) BASIS="def2-svp" ;;
        2) BASIS="def2-tzvp" ;;
        3) BASIS="3-21g" ;;
        *) BASIS="def2-svp" ;;
    esac
}

get_parallelism() {
    echo ""
    echo "Parallelism options:"
    echo "  1) Auto-optimize (recommended - adjusts based on molecule count)"
    echo "  2) Manual control (specify number of parallel tasks)"
    echo ""
    read -p "Select option (1-2): " parallel_choice
    
    if [ "$parallel_choice" = "2" ]; then
        echo ""
        echo "Number of parallel tasks (molecule/solvent combinations):"
        echo "  Note: Each task will use (14 CPUs / n_jobs) threads"
        echo "  Recommended: 2-7 for optimal performance"
        echo ""
        read -p "Enter number of parallel tasks (2-7): " N_JOBS
        N_JOBS_ARG="--n-jobs $N_JOBS"
    else
        N_JOBS_ARG=""
    fi
}

# ============================================================================
# MONITORING FUNCTIONS
# ============================================================================

monitor_calculation() {
    echo ""
    echo -e "${BLUE}Monitoring calculation progress...${NC}"
    echo "Press Ctrl+C to stop monitoring (calculation will continue)"
    echo ""
    
    ENERGY_FILE="$SCRIPT_DIR/pyscf_persistent_data/energy_results/energy_results.h5"
    
    while true; do
        if [ -f "$ENERGY_FILE" ]; then
            # Count completed calculations
            COMPLETED=$($PYTHON_CMD -c "
import h5py
try:
    with h5py.File('$ENERGY_FILE', 'r') as f:
        print(len([k for k in f.keys() if f[k].attrs.get('completed', False)]))
except:
    print(0)
" 2>/dev/null || echo "0")
            
            echo -ne "\rCompleted calculations: $COMPLETED"
        fi
        sleep 5
    done
}

show_status() {
    echo ""
    echo "=========================================="
    echo "Calculation Status"
    echo "=========================================="
    
    ENERGY_FILE="$SCRIPT_DIR/pyscf_persistent_data/energy_results/energy_results.h5"
    
    if [ ! -f "$ENERGY_FILE" ]; then
        echo "No calculations found."
        return
    fi
    
    $PYTHON_CMD << EOF
import h5py
from datetime import datetime

try:
    with h5py.File('$ENERGY_FILE', 'r') as f:
        print(f"\nTotal calculations: {len(f.keys())}")
        print("\nCompleted calculations:")
        print("-" * 80)
        
        for key in sorted(f.keys()):
            grp = f[key]
            if grp.attrs.get('completed', False):
                molecule = grp.attrs['molecule']
                solvent = grp.attrs['solvent']
                timestamp = grp.attrs.get('timestamp', 'unknown')
                delta_est = grp.attrs.get('delta_est_opt_eV', 0.0)
                omega = grp.attrs.get('omega_bohr-1', 0.0)
                
                print(f"{molecule:15s} {solvent:8s}  ΔE_ST={delta_est:.4f} eV  ω={omega:.4f}  [{timestamp}]")
        
        print("-" * 80)
except Exception as e:
    print(f"Error reading results: {e}")
EOF
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    # Setup environment
    setup_environment
    
    # Optimize CPU if running as root
    if [ "$SKIP_CPU_OPT" = false ]; then
        optimize_cpu_frequency
    fi
    
    # Interactive menu
    while true; do
        show_menu
        read -p "Select option (1-7): " choice
        
        case $choice in
            1)
                # Default molecules with auto-optimization
                MOLECULES="DMAC-DPS DMAC-TRZ 4CzIPN PXZ-NAI TPA-APy BMZ-TZ"
                BASIS="def2-svp"
                FORCE=""
                N_JOBS_ARG=""
                break
                ;;
            2)
                # Custom molecules
                get_molecules
                BASIS="def2-svp"
                FORCE=""
                N_JOBS_ARG=""
                break
                ;;
            3)
                # Custom basis
                MOLECULES="DMAC-DPS DMAC-TRZ 4CzIPN"
                get_basis
                FORCE=""
                N_JOBS_ARG=""
                break
                ;;
            4)
                # Manual parallelism
                MOLECULES="DMAC-DPS DMAC-TRZ 4CzIPN"
                BASIS="def2-svp"
                FORCE=""
                get_parallelism
                break
                ;;
            5)
                # Force recalculation
                MOLECULES="DMAC-DPS DMAC-TRZ"
                BASIS="def2-svp"
                FORCE="--force"
                N_JOBS_ARG=""
                break
                ;;
            6)
                # Show status
                show_status
                continue
                ;;
            7)
                # Exit
                echo "Exiting..."
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option${NC}"
                continue
                ;;
        esac
    done
    
    # Build command
    CMD="$PYTHON_CMD optimized_pyscf_fixed_omega_parallel.py --molecules $MOLECULES --basis $BASIS $FORCE $N_JOBS_ARG"
    
    # Calculate expected parallelism
    N_MOLECULES=$(echo $MOLECULES | wc -w)
    N_TASKS=$((N_MOLECULES * 2))  # Each molecule has gas + toluene
    
    if [ -z "$N_JOBS_ARG" ]; then
        # Auto-optimization
        if [ $N_TASKS -le 2 ]; then
            PARALLEL_INFO="2 tasks in parallel, 7 threads each"
        elif [ $N_TASKS -le 7 ]; then
            PARALLEL_INFO="$N_TASKS tasks in parallel, 2 threads each"
        else
            PARALLEL_INFO="7 tasks in parallel, 2 threads each (batched)"
        fi
    else
        N_JOBS_NUM=$(echo $N_JOBS_ARG | grep -o '[0-9]*')
        THREADS_PER=$((14 / N_JOBS_NUM))
        PARALLEL_INFO="$N_JOBS_NUM tasks in parallel, $THREADS_PER threads each"
    fi
    
    # Show summary
    echo ""
    echo "=========================================="
    echo "CALCULATION SUMMARY (PARALLELIZED)"
    echo "=========================================="
    echo ""
    echo "Molecules:     $MOLECULES ($N_MOLECULES molecules)"
    echo "Total tasks:   $N_TASKS (molecule × solvent)"
    echo "Parallelism:   $PARALLEL_INFO"
    echo "Basis set:     $BASIS"
    echo "CPU governor:  performance (max frequency)"
    echo "Phases:        GAS + TOLUENE"
    echo "Force recalc:  $([ -n "$FORCE" ] && echo "Yes" || echo "No")"
    echo ""
    echo "Command:"
    echo "  $CMD"
    echo ""
    
    # Estimate time savings
    if [ $N_TASKS -gt 2 ]; then
        SPEEDUP=$(awk "BEGIN {printf \"%.1f\", $N_TASKS / 2.0}")
        echo -e "${GREEN}Expected speedup: ~${SPEEDUP}x faster than sequential!${NC}"
        echo ""
    fi
    
    read -p "Proceed with calculation? (Y/n): " confirm
    if [[ $confirm =~ ^[Nn]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    
    # Run calculation
    echo ""
    echo "=========================================="
    echo "STARTING PARALLELIZED CALCULATION"
    echo "=========================================="
    echo ""
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Run in background with monitoring
    $CMD 2>&1 | tee "pyscf_results/calculation.log" &
    CALC_PID=$!
    
    # Give process time to start
    sleep 2
    
    # Check if process is still running
    if ! kill -0 $CALC_PID 2>/dev/null; then
        echo -e "${RED}✗ Calculation failed to start${NC}"
        echo "Check log file: $SCRIPT_DIR/pyscf_results/calculation.log"
        echo ""
        echo "Last 30 lines of log:"
        tail -30 "pyscf_results/calculation.log" 2>/dev/null || echo "(log file empty or not created)"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Calculation started successfully (PID: $CALC_PID)${NC}"
    echo ""
    echo "Monitor CPU usage in another terminal with: htop"
    echo "You should see 14 CPUs at ~100% utilization"
    echo ""
    
    # Monitor progress
    sleep 1
    monitor_calculation &
    MONITOR_PID=$!
    
    # Wait for calculation to complete
    if wait $CALC_PID 2>/dev/null; then
        CALC_EXIT=0
    else
        CALC_EXIT=$?
    fi
    
    # Stop monitoring
    kill $MONITOR_PID 2>/dev/null || true
    
    echo ""
    echo ""
    echo "=========================================="
    echo "CALCULATION COMPLETE"
    echo "=========================================="
    
    if [ $CALC_EXIT -eq 0 ]; then
        echo -e "${GREEN}✓ Calculation completed successfully${NC}"
        echo ""
        echo "Results saved to:"
        echo "  $SCRIPT_DIR/pyscf_results/optimized_pyscf_delta_roks_results_corrected.csv"
        echo ""
        echo "Persistent data:"
        echo "  Geometries: $SCRIPT_DIR/pyscf_persistent_data/optimized_geometries/"
        echo "  Energies:   $SCRIPT_DIR/pyscf_persistent_data/energy_results/"
    else
        echo -e "${RED}✗ Calculation failed with exit code $CALC_EXIT${NC}"
        echo "Check log file: $SCRIPT_DIR/pyscf_results/calculation.log"
        echo ""
        echo "Last 30 lines of log:"
        tail -30 "pyscf_results/calculation.log" 2>/dev/null || echo "(log file empty or not created)"
    fi
    
    echo "=========================================="
}

# Run main function
main
