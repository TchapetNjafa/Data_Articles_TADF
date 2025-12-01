#!/bin/bash
#
# Multiwfn Analysis Script for 747 TADF Molecules
# Calculates NTO orbital overlaps (hole-electron) using Multiwfn
#
# IMPORTANT: Moves molecule folders to /home/tchapet/ temporarily to avoid
# path issues with Multiwfn, then moves them back after analysis.
#
# Usage: ./run_multiwfn_analysis.sh [start_index] [end_index]
#

set -e

# Tool paths
MULTIWFN="/home/tchapet/Multiwfn/Multiwfn_3.8_dev_bin_Linux/Multiwfn"

# Base directory (original location)
BASE_DIR="/home/tchapet/Post-Doc/ARTICLES DOSSIERS DES ARTICLES EN REDACTION/NOUVELS AXES DE RECHERCHE A REGARDER URGEMMENT/ARTICLES EN REDACTIONS/ARTICLE1/redaction/Article3_ML/Result_article1_TADF_xTB/Data_calculation_747Mol"

# Temporary working directory in home
WORK_DIR="/home/tchapet/multiwfn_temp"

# Output CSV file
OUTPUT_CSV="$BASE_DIR/nto_orbital_overlap_747mol.csv"

# Log file
LOG_FILE="$BASE_DIR/multiwfn_analysis.log"

# Parse arguments
START_INDEX=${1:-1}
END_INDEX=${2:-0}

echo "================================================================================"
echo "MULTIWFN NTO ORBITAL OVERLAP ANALYSIS - 747 MOLECULES"
echo "================================================================================"
echo "Start time: $(date)"
echo "Multiwfn: $MULTIWFN"
echo ""

# Create work directory
mkdir -p "$WORK_DIR"

# Get list of molecules
cd "$BASE_DIR/gas"
MOLECULES=()
while IFS= read -r -d '' dir; do
    MOLECULES+=("$(basename "$dir")")
done < <(find . -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
cd "$BASE_DIR"

TOTAL_MOLS=${#MOLECULES[@]}

# Adjust end index
if [ "$END_INDEX" -eq 0 ] || [ "$END_INDEX" -gt "$TOTAL_MOLS" ]; then
    END_INDEX=$TOTAL_MOLS
fi

echo "Total molecules: $TOTAL_MOLS"
echo "Processing molecules $START_INDEX to $END_INDEX"
echo ""

# Create CSV header if file doesn't exist or starting from 1
if [ "$START_INDEX" -eq 1 ] || [ ! -f "$OUTPUT_CSV" ]; then
    echo "Molecule,Environment,Transition,Method,Overlap_1_2,Status" > "$OUTPUT_CSV"
fi

# Define transitions and methods
TRANSITIONS=("S0S1" "S0T1")
METHODS=("stda" "stddft")

# Counters
PROCESSED=0
SUCCESS=0
FAILED=0

# Function to process one NTO folder
process_nto_folder() {
    local MOL=$1
    local ENV=$2
    local TRANS=$3
    local METHOD=$4
    local NTO_DIR_NAME="${MOL}_${ENV}_${TRANS}_NTO_${METHOD}"
    local ORIGINAL_PATH="$BASE_DIR/$ENV/$MOL/$NTO_DIR_NAME"
    local TEMP_PATH="$WORK_DIR/$NTO_DIR_NAME"

    # Check if NTO folder exists
    if [ ! -d "$ORIGINAL_PATH" ]; then
        echo "      ✗ NTO folder not found" | tee -a "$LOG_FILE"
        echo "$MOL,$ENV,$TRANS,$METHOD,,Folder not found" >> "$OUTPUT_CSV"
        return 1
    fi

    # Check if nto001.molden exists
    if [ ! -f "$ORIGINAL_PATH/nto001.molden" ]; then
        echo "      ✗ nto001.molden not found" | tee -a "$LOG_FILE"
        echo "$MOL,$ENV,$TRANS,$METHOD,,NTO file not found" >> "$OUTPUT_CSV"
        return 1
    fi

    # Check if already processed
    if [ -f "$ORIGINAL_PATH/orbint.txt" ]; then
        OVERLAP=$(head -1 "$ORIGINAL_PATH/orbint.txt" | awk '{print $NF}')
        if [ -n "$OVERLAP" ]; then
            echo "      [SKIP] Already processed: $OVERLAP"
            echo "$MOL,$ENV,$TRANS,$METHOD,$OVERLAP,Success (cached)" >> "$OUTPUT_CSV"
            return 0
        fi
    fi

    # Move folder to temp location
    mv "$ORIGINAL_PATH" "$TEMP_PATH"

    cd "$TEMP_PATH"

    # Create Multiwfn input (200=Other functions, 10=Orbital integrals, 5=Overlap, 6=two specific orbitals, 1,2=orbital pair)
    # Using option 6 for speed - parses result from stdout instead of orbint.txt
    # Empty line after molden file to acknowledge ECP/charge warning (Press ENTER to continue)
    cat > mwfn_overlap_input.txt << EOF
nto001.molden

200
10
5
6
1,2
0,0
0
q
EOF

    # Run Multiwfn and parse Result from stdout
    if $MULTIWFN < mwfn_overlap_input.txt > mwfn_overlap_output.txt 2>&1; then
        # Extract overlap value from "Result:      0.0000000402" line
        OVERLAP=$(grep "Result:" mwfn_overlap_output.txt | head -1 | awk '{print $2}')
        if [ -n "$OVERLAP" ]; then
            echo "      ✓ Overlap: $OVERLAP" | tee -a "$LOG_FILE"
            echo "$MOL,$ENV,$TRANS,$METHOD,$OVERLAP,Success" >> "$OUTPUT_CSV"
            cd "$BASE_DIR"
            mv "$TEMP_PATH" "$ORIGINAL_PATH"
            return 0
        fi
        echo "      ⚠ Overlap not found in output" | tee -a "$LOG_FILE"
        echo "$MOL,$ENV,$TRANS,$METHOD,,Parse failed" >> "$OUTPUT_CSV"
    else
        echo "      ✗ Multiwfn execution failed" | tee -a "$LOG_FILE"
        echo "$MOL,$ENV,$TRANS,$METHOD,,Execution failed" >> "$OUTPUT_CSV"
    fi

    # Move folder back even if failed
    cd "$BASE_DIR"
    mv "$TEMP_PATH" "$ORIGINAL_PATH"
    return 1
}

# Main processing loop
for ((i=START_INDEX-1; i<END_INDEX; i++)); do
    MOL="${MOLECULES[$i]}"
    PROCESSED=$((PROCESSED + 1))

    echo ""
    echo "[$PROCESSED/$((END_INDEX-START_INDEX+1))] Processing: $MOL"
    echo "$(date) - Processing $MOL" >> "$LOG_FILE"

    for ENV in "gas" "toluene"; do
        echo "  Environment: $ENV"
        for TRANS in "${TRANSITIONS[@]}"; do
            for METHOD in "${METHODS[@]}"; do
                echo "    $TRANS | $METHOD"
                if process_nto_folder "$MOL" "$ENV" "$TRANS" "$METHOD"; then
                    SUCCESS=$((SUCCESS + 1))
                else
                    FAILED=$((FAILED + 1))
                fi
            done
        done
    done
done

# Cleanup
rmdir "$WORK_DIR" 2>/dev/null || true

echo ""
echo "================================================================================"
echo "MULTIWFN ANALYSIS COMPLETE"
echo "================================================================================"
echo "End time: $(date)"
echo "Molecules processed: $PROCESSED"
echo "Successful analyses: $SUCCESS"
echo "Failed analyses: $FAILED"
echo "Results: $OUTPUT_CSV"
echo ""
