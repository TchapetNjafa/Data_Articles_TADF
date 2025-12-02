#!/bin/bash
#
# High-Level Theory (HLT) Calculations for Article3
# OT-LC-ωPBE validation of sTDA/sTD-DFT-xTB protocol
#
# Usage: ./run_hlt_calculations.sh [molecule]
#   If molecule is specified, only that molecule is calculated
#   Otherwise, all 3 benchmark molecules are calculated
#
# Example:
#   ./run_hlt_calculations.sh              # Run all
#   ./run_hlt_calculations.sh DMAC-DPS     # Run only DMAC-DPS
#

# Configuration
ORCA_PATH="/home/tchapet/orca-6-1-0/bin/orca"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/home/tchapet/Documents/GitHub/TADF/smiEmpirical-TADF/Public_Results/Result_article1_TADF_xTB/Data_calculation_747Mol/gas"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Molecules to process
declare -A MOLECULES
MOLECULES["BACN"]="${DATA_DIR}/BACN/BACN_gas_S0_finalOpt.xtbopt.xyz"
MOLECULES["DMAC-TRZ"]="${DATA_DIR}/DMAC-TRZ/DMAC-TRZ_gas_S0_finalOpt.xtbopt.xyz"
MOLECULES["4CzIPN"]="${DATA_DIR}/4CzIPN/4CzIPN_gas_S0_finalOpt.xtbopt.xyz"

# Log file
LOG_FILE="${RESULTS_DIR}/hlt_calculations.log"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

# Function to run omega tuning for a molecule
run_omega_tuning() {
    local mol_name=$1
    local xyz_file=$2

    log "Starting ω tuning for ${mol_name}..."
    log "XYZ file: ${xyz_file}"

    if [ ! -f "${xyz_file}" ]; then
        log "ERROR: XYZ file not found: ${xyz_file}"
        return 1
    fi

    cd "${SCRIPT_DIR}"

    # Run omega tuning
    python3 omega_tuning.py "${mol_name}" "${xyz_file}" 2>&1 | tee -a "${LOG_FILE}"

    if [ $? -eq 0 ]; then
        log "ω tuning completed for ${mol_name}"
    else
        log "ERROR: ω tuning failed for ${mol_name}"
        return 1
    fi

    return 0
}

# Function to run TD-DFT calculation
run_tddft() {
    local mol_name=$1
    local mol_dir="${RESULTS_DIR}/${mol_name}"
    local input_file="${mol_dir}/${mol_name}_TDDFT.inp"

    log "Starting TD-DFT calculation for ${mol_name}..."

    if [ ! -f "${input_file}" ]; then
        log "ERROR: TD-DFT input file not found: ${input_file}"
        return 1
    fi

    cd "${mol_dir}"

    # Run ORCA
    ${ORCA_PATH} "${input_file}" > "${mol_name}_TDDFT.out" 2>&1

    if [ $? -eq 0 ]; then
        log "TD-DFT completed for ${mol_name}"
    else
        log "ERROR: TD-DFT failed for ${mol_name}"
        return 1
    fi

    return 0
}

# Function to extract results
extract_results() {
    local mol_name=$1
    local mol_dir="${RESULTS_DIR}/${mol_name}"
    local tddft_out="${mol_dir}/${mol_name}_TDDFT.out"
    local summary_file="${mol_dir}/${mol_name}_summary.txt"

    log "Extracting results for ${mol_name}..."

    if [ ! -f "${tddft_out}" ]; then
        log "ERROR: TD-DFT output not found: ${tddft_out}"
        return 1
    fi

    # Extract S1 and T1 energies
    echo "=== Results for ${mol_name} ===" > "${summary_file}"
    echo "" >> "${summary_file}"

    # Get optimal omega
    if [ -f "${mol_dir}/optimal_omega.txt" ]; then
        cat "${mol_dir}/optimal_omega.txt" >> "${summary_file}"
        echo "" >> "${summary_file}"
    fi

    # Extract excited state energies from TD-DFT output
    echo "Excited States:" >> "${summary_file}"
    grep -A 2 "STATE\s*1:" "${tddft_out}" | head -5 >> "${summary_file}"

    # Try to get S1 and T1 energies
    # ORCA format: STATE  1:  E=   X.XXXXXX au   Y.YYYY eV
    S1_eV=$(grep "STATE\s*1:" "${tddft_out}" | head -1 | grep -oP '\d+\.\d+\s*eV' | head -1 | sed 's/eV//')

    echo "" >> "${summary_file}"
    echo "S1 energy: ${S1_eV} eV" >> "${summary_file}"

    log "Results extracted to ${summary_file}"

    return 0
}

# Main execution
log "=============================================="
log "HLT Calculations for Article3"
log "OT-LC-ωPBE Validation"
log "=============================================="

# Check if specific molecule requested
if [ -n "$1" ]; then
    MOLECULES_TO_RUN=("$1")
else
    MOLECULES_TO_RUN=("BACN" "DMAC-TRZ" "4CzIPN")
fi

# Process each molecule
for mol in "${MOLECULES_TO_RUN[@]}"; do
    xyz_file="${MOLECULES[$mol]}"

    if [ -z "${xyz_file}" ]; then
        log "ERROR: Unknown molecule: ${mol}"
        continue
    fi

    log ""
    log "Processing ${mol}..."
    log "=============================================="

    # Step 1: Omega tuning
    run_omega_tuning "${mol}" "${xyz_file}"

    if [ $? -ne 0 ]; then
        log "Skipping ${mol} due to omega tuning failure"
        continue
    fi

    # Step 2: TD-DFT with optimal omega
    run_tddft "${mol}"

    if [ $? -ne 0 ]; then
        log "Skipping result extraction for ${mol} due to TD-DFT failure"
        continue
    fi

    # Step 3: Extract results
    extract_results "${mol}"

    log "${mol} completed successfully!"
done

log ""
log "=============================================="
log "All calculations completed!"
log "Results are in: ${RESULTS_DIR}"
log "=============================================="

# Generate summary CSV
SUMMARY_CSV="${RESULTS_DIR}/hlt_summary.csv"
echo "Molecule,Optimal_Omega,S1_eV,T1_eV,Delta_EST_eV" > "${SUMMARY_CSV}"

for mol in "${MOLECULES_TO_RUN[@]}"; do
    mol_dir="${RESULTS_DIR}/${mol}"
    if [ -d "${mol_dir}" ]; then
        omega=$(grep "Optimal omega:" "${mol_dir}/optimal_omega.txt" 2>/dev/null | awk '{print $3}')
        echo "${mol},${omega},,,," >> "${SUMMARY_CSV}"
    fi
done

log "Summary saved to: ${SUMMARY_CSV}"
