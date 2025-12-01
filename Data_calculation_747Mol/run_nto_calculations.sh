#!/bin/bash
#
# NTO Calculation Script for 747 TADF Molecules
# Generates NTOs using xtb4stda and stda for both gas and toluene environments
#
# Usage: ./run_nto_calculations.sh [start_index] [end_index]
#   - start_index: 1-based index of first molecule to process (default: 1)
#   - end_index: 1-based index of last molecule to process (default: all)
#

set -e

# Tool paths
XTB4STDA="/home/tchapet/xtb4stda/xtb4stda"
STDA="/home/tchapet/stda_v1-6/stda"

# Base directory
BASE_DIR="/home/tchapet/Post-Doc/ARTICLES DOSSIERS DES ARTICLES EN REDACTION/NOUVELS AXES DE RECHERCHE A REGARDER URGEMMENT/ARTICLES EN REDACTIONS/ARTICLE1/redaction/Article3_ML/Result_article1_TADF_xTB/Data_calculation_747Mol"

# Log file
LOG_FILE="$BASE_DIR/nto_calculation.log"
PROGRESS_FILE="$BASE_DIR/nto_progress.txt"

# Number of NTOs to generate
NUM_NTOS=5

# Parse arguments
START_INDEX=${1:-1}
END_INDEX=${2:-0}

echo "================================================================================"
echo "NTO CALCULATION FOR 747 TADF MOLECULES"
echo "================================================================================"
echo "Start time: $(date)"
echo "XTB4STDA: $XTB4STDA"
echo "STDA: $STDA"
echo "NTOs to generate: $NUM_NTOS"
echo ""

# Get list of molecules from gas folder
cd "$BASE_DIR/gas"
MOLECULES=()
while IFS= read -r -d '' dir; do
    MOLECULES+=("$(basename "$dir")")
done < <(find . -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
cd "$BASE_DIR"
TOTAL_MOLS=${#MOLECULES[@]}

echo "Total molecules found: $TOTAL_MOLS"

# Adjust end index if not specified
if [ "$END_INDEX" -eq 0 ] || [ "$END_INDEX" -gt "$TOTAL_MOLS" ]; then
    END_INDEX=$TOTAL_MOLS
fi

echo "Processing molecules $START_INDEX to $END_INDEX"
echo ""

# Initialize counters
PROCESSED=0
SUCCESS=0
FAILED=0

# Function to process a single molecule in one environment
process_molecule() {
    local MOL=$1
    local ENV=$2
    local MOL_DIR="$BASE_DIR/$ENV/$MOL"

    # Find the xyz file
    local XYZ_FILE=$(find "$MOL_DIR" -name "*_S0_finalOpt.xtbopt.xyz" -type f 2>/dev/null | head -1)

    if [ -z "$XYZ_FILE" ]; then
        echo "  ✗ No xyz file found for $MOL ($ENV)" | tee -a "$LOG_FILE"
        return 1
    fi

    local XYZ_BASENAME=$(basename "$XYZ_FILE")

    # Define transitions and methods
    local TRANSITIONS=("S0S1" "S0T1")
    local METHODS=("stda" "stddft")

    for TRANS in "${TRANSITIONS[@]}"; do
        for METHOD in "${METHODS[@]}"; do
            local NTO_DIR="$MOL_DIR/${MOL}_${ENV}_${TRANS}_NTO_${METHOD}"
            local MARKER_FILE="$NTO_DIR/.nto_complete"

            # Skip if already completed
            if [ -f "$MARKER_FILE" ]; then
                echo "    [SKIP] $MOL | $ENV | $TRANS | $METHOD (already done)"
                continue
            fi

            echo "    [PROC] $MOL | $ENV | $TRANS | $METHOD"

            # Create NTO directory
            mkdir -p "$NTO_DIR"

            # Copy xyz file
            cp "$XYZ_FILE" "$NTO_DIR/"

            # Change to NTO directory
            cd "$NTO_DIR"

            # Run xtb4stda
            if [ "$ENV" == "gas" ]; then
                $XTB4STDA "$XYZ_BASENAME" > xtb4stda.log 2>&1
            else
                $XTB4STDA "$XYZ_BASENAME" gbsa toluene > xtb4stda.log 2>&1
            fi

            if [ ! -f "wfn.xtb" ]; then
                echo "      ✗ xtb4stda failed" | tee -a "$LOG_FILE"
                cd "$BASE_DIR"
                continue
            fi

            # Run stda
            local STDA_OPTS="-xtb -e 10 -nto $NUM_NTOS"

            # Add triplet flag for T1
            if [ "$TRANS" == "S0T1" ]; then
                STDA_OPTS="$STDA_OPTS -t"
            fi

            # Add RPA flag for stddft
            if [ "$METHOD" == "stddft" ]; then
                STDA_OPTS="$STDA_OPTS -rpa"
            fi

            $STDA $STDA_OPTS > "${MOL}_${ENV}_${TRANS}_NTO_${METHOD}.log" 2>&1

            # Check if NTO files were created
            if [ -f "nto001.molden" ]; then
                echo "      ✓ NTOs generated" | tee -a "$LOG_FILE"
                touch "$MARKER_FILE"
            else
                echo "      ✗ stda failed to generate NTOs" | tee -a "$LOG_FILE"
            fi

            cd "$BASE_DIR"
        done
    done

    return 0
}

# Main processing loop
for ((i=START_INDEX-1; i<END_INDEX; i++)); do
    MOL="${MOLECULES[$i]}"
    PROCESSED=$((PROCESSED + 1))

    echo ""
    echo "[$PROCESSED/$((END_INDEX-START_INDEX+1))] Processing: $MOL"
    echo "$(date) - Processing $MOL" >> "$LOG_FILE"

    # Process gas environment
    echo "  Environment: gas"
    if process_molecule "$MOL" "gas"; then
        SUCCESS=$((SUCCESS + 1))
    else
        FAILED=$((FAILED + 1))
    fi

    # Process toluene environment
    echo "  Environment: toluene"
    if process_molecule "$MOL" "toluene"; then
        SUCCESS=$((SUCCESS + 1))
    else
        FAILED=$((FAILED + 1))
    fi

    # Update progress file
    echo "$MOL" >> "$PROGRESS_FILE"
done

echo ""
echo "================================================================================"
echo "NTO CALCULATION COMPLETE"
echo "================================================================================"
echo "End time: $(date)"
echo "Molecules processed: $PROCESSED"
echo "Results logged to: $LOG_FILE"
echo ""
