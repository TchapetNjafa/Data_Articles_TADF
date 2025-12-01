#!/bin/bash
#
# Multiwfn Hole-Electron Analysis Script
# Calculates hole-electron descriptors (Sr, D, t, Δσ, H) for TADF molecules
#
# Usage: ./run_hole_electron_analysis.sh [base_dir] [output_csv]
#

set -e

# Tool paths
MULTIWFN="/home/tchapet/Multiwfn/Multiwfn_3.8_dev_bin_Linux/Multiwfn"

# Default directories
BASE_DIR="${1:-/home/tchapet/Post-Doc/ARTICLES DOSSIERS DES ARTICLES EN REDACTION/NOUVELS AXES DE RECHERCHE A REGARDER URGEMMENT/ARTICLES EN REDACTIONS/ARTICLE1/redaction/Article3_ML/Result_article1_TADF_xTB/Data_calculation_747Mol}"
OUTPUT_CSV="${2:-$BASE_DIR/hole_electron_descriptors.csv}"

# Temporary working directory
WORK_DIR="/home/tchapet/multiwfn_he_temp"

# Log file
LOG_FILE="$BASE_DIR/hole_electron_analysis.log"

echo "================================================================================"
echo "MULTIWFN HOLE-ELECTRON ANALYSIS"
echo "================================================================================"
echo "Start time: $(date)"
echo "Base directory: $BASE_DIR"
echo "Output: $OUTPUT_CSV"
echo ""

# Create work directory
mkdir -p "$WORK_DIR"

# Create CSV header
echo "Molecule,Environment,Transition,Method,Sr,D_index,t_index,Delta_sigma,H_index,Sm,Status" > "$OUTPUT_CSV"

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
    local NTO_FILE="$ORIGINAL_PATH/nto001.molden"

    # Check if NTO file exists
    if [ ! -f "$NTO_FILE" ]; then
        echo "      ✗ NTO file not found: $NTO_FILE" | tee -a "$LOG_FILE"
        echo "$MOL,$ENV,$TRANS,$METHOD,,,,,,Folder not found" >> "$OUTPUT_CSV"
        return 1
    fi

    # Check if already processed (cache file exists)
    local CACHE_FILE="$ORIGINAL_PATH/hole_electron_results.txt"
    if [ -f "$CACHE_FILE" ]; then
        # Read cached results
        local CACHED=$(cat "$CACHE_FILE")
        echo "      [SKIP] Using cached: $CACHED"
        echo "$MOL,$ENV,$TRANS,$METHOD,$CACHED,Success (cached)" >> "$OUTPUT_CSV"
        return 0
    fi

    # Create temporary directory for this calculation
    local TEMP_DIR="$WORK_DIR/${MOL}_${ENV}_${TRANS}_${METHOD}"
    mkdir -p "$TEMP_DIR"

    # Copy NTO file to temp directory (use short path)
    cp "$NTO_FILE" "$TEMP_DIR/nto.molden"

    cd "$TEMP_DIR"

    # Create Multiwfn input for hole-electron analysis
    # Option 18: Electronic excitation analysis
    # Sub-option 1: Analyze hole & electron distribution
    # Using default grid (medium quality for speed)
    cat > mwfn_he_input.txt << 'EOF'
nto.molden

18
1
1
3
0
q
EOF

    # Run Multiwfn
    local OUTPUT_FILE="mwfn_he_output.txt"
    if timeout 120 $MULTIWFN < mwfn_he_input.txt > "$OUTPUT_FILE" 2>&1; then
        # Parse results from output
        # Look for lines like:
        # Sr index (integral of rho^2(r)):    0.1234
        # D index (distance between centroid):   1.234 Angstrom
        # t index:    1.234 Angstrom
        # Delta_sigma:    0.123 Angstrom
        # H index:    1.234 Angstrom
        # Sm index:    0.1234

        local Sr=$(grep "Sr index" "$OUTPUT_FILE" | head -1 | awk '{print $NF}')
        local D=$(grep "D index" "$OUTPUT_FILE" | head -1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9.-]+$/) print $i}' | head -1)
        local t=$(grep "t index" "$OUTPUT_FILE" | head -1 | awk '{print $(NF-1)}')
        local Delta_sigma=$(grep -i "delta.*sigma\|Δσ" "$OUTPUT_FILE" | head -1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9.-]+$/) print $i}' | head -1)
        local H=$(grep "H index" "$OUTPUT_FILE" | head -1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9.-]+$/) print $i}' | head -1)
        local Sm=$(grep "Sm index" "$OUTPUT_FILE" | head -1 | awk '{print $NF}')

        # Check if we got valid results
        if [ -n "$Sr" ] && [ "$Sr" != "0" ]; then
            echo "      ✓ Sr=$Sr, D=$D, t=$t" | tee -a "$LOG_FILE"

            # Cache results
            echo "$Sr,$D,$t,$Delta_sigma,$H,$Sm" > "$CACHE_FILE"

            echo "$MOL,$ENV,$TRANS,$METHOD,$Sr,$D,$t,$Delta_sigma,$H,$Sm,Success" >> "$OUTPUT_CSV"
            cd "$BASE_DIR"
            rm -rf "$TEMP_DIR"
            return 0
        else
            echo "      ⚠ Could not parse results" | tee -a "$LOG_FILE"
            # Save output for debugging
            cp "$OUTPUT_FILE" "$ORIGINAL_PATH/mwfn_he_debug.txt" 2>/dev/null || true
        fi
    else
        echo "      ✗ Multiwfn execution failed or timed out" | tee -a "$LOG_FILE"
    fi

    echo "$MOL,$ENV,$TRANS,$METHOD,,,,,,Execution failed" >> "$OUTPUT_CSV"
    cd "$BASE_DIR"
    rm -rf "$TEMP_DIR"
    return 1
}

# Get list of molecules
cd "$BASE_DIR/gas"
MOLECULES=($(ls -d */ 2>/dev/null | sed 's/\///g' | head -10))  # Start with first 10 for testing
cd "$BASE_DIR"

TOTAL_MOLS=${#MOLECULES[@]}
echo "Processing $TOTAL_MOLS molecules (test run)"
echo ""

# Define transitions and methods
TRANSITIONS=("S0S1" "S0T1")
METHODS=("stda" "stddft")

# Main processing loop
for MOL in "${MOLECULES[@]}"; do
    PROCESSED=$((PROCESSED + 1))
    echo ""
    echo "[$PROCESSED/$TOTAL_MOLS] Processing: $MOL"
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
rm -rf "$WORK_DIR"

echo ""
echo "================================================================================"
echo "HOLE-ELECTRON ANALYSIS COMPLETE"
echo "================================================================================"
echo "End time: $(date)"
echo "Molecules processed: $PROCESSED"
echo "Successful analyses: $SUCCESS"
echo "Failed analyses: $FAILED"
echo "Results: $OUTPUT_CSV"
echo ""
