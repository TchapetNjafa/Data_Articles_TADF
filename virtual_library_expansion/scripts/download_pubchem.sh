#!/bin/bash
# Download PubChem CID-SMILES mapping file
# This file contains ~110 million compounds with SMILES

set -e  # Exit on error

echo "=========================================="
echo "PubChem CID-SMILES Download"
echo "=========================================="
echo ""

# Create data directory
mkdir -p ../data/pubchem
cd ../data/pubchem

# Check if file already exists
if [ -f "CID-SMILES" ]; then
    echo "✅ CID-SMILES already exists ($(du -h CID-SMILES | cut -f1))"
    echo "   Skipping download."
    exit 0
fi

if [ -f "CID-SMILES.gz" ]; then
    echo "✅ CID-SMILES.gz already exists ($(du -h CID-SMILES.gz | cut -f1))"
    echo "   Skipping download, will uncompress..."
else
    echo "📥 Downloading PubChem CID-SMILES.gz (~2 GB)..."
    echo "   This may take 10-30 minutes depending on connection speed"
    echo ""
    
    # Download with progress bar
    wget --progress=bar:force \
         https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz \
         -O CID-SMILES.gz
    
    echo ""
    echo "✅ Download complete: $(du -h CID-SMILES.gz | cut -f1)"
fi

echo ""
echo "📦 Uncompressing CID-SMILES.gz..."
echo "   This may take 5-10 minutes"
echo ""

gunzip -v CID-SMILES.gz

echo ""
echo "✅ Uncompressed: $(du -h CID-SMILES | cut -f1)"
echo ""
echo "📊 File statistics:"
wc -l CID-SMILES
echo ""
echo "=========================================="
echo "✅ DOWNLOAD COMPLETE"
echo "=========================================="
echo ""
echo "Next step: Run 02_filter_pubchem_tadf.py to extract TADF compounds"
