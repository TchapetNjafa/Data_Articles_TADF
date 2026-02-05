#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main pipeline for TADF molecule calculations.

This script orchestrates the complete workflow:
1. Generate initial structures from SMILES (RDKit)
2. Optimize geometries and search conformers (xTB + CREST)
3. Calculate excitation energies (sTDA/sTD-DFT)
4. Extract and save results to CSV

The workflow follows the methodology described in https://arxiv.org/abs/2502.20410
"""

from pathlib import Path
import subprocess as sp
from itertools import product
import pandas as pd
import ast

from rdkitGen_Mol import rdkitGen_Mol
from geo_Opt import geo_Opt
from excitationEner_Calc import excitationEner_Calc
from data_Extract import data_Extract


def diction_smiles(sourceData):
    """
    Extract SMILES dictionary from source data.
    
    Parameters
    ----------
    sourceData : pd.DataFrame
        DataFrame containing compound names and SMILES
    
    Returns
    -------
    dict
        Dictionary mapping molecule names to SMILES strings
    """
    smiles_dict = {}
    
    for index, row in sourceData.iterrows():
        compound_names = row['compound.names']
        compound_smiles = row['compound.SMILES']
        
        try:
            if isinstance(compound_names, str):
                try:
                    names_vector = ast.literal_eval(compound_names)
                except (ValueError, SyntaxError):
                    if ',' in compound_names:
                        names_vector = [x.strip() for x in compound_names.split(',')]
                    elif ';' in compound_names:
                        names_vector = [x.strip() for x in compound_names.split(';')]
                    else:
                        names_vector = [compound_names.strip()]
            else:
                names_vector = compound_names
            
            if names_vector and len(names_vector) > 0:
                key = min(names_vector, key=len)
                smiles_dict[key] = f'{compound_smiles}'
                
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
    
    return smiles_dict


def semiEmpi_tadf(input_csv='unique_subsidiary_database.csv', output_dir='./calculation_results', output_csv='tadf_results'):
    """
    Run the complete TADF calculation pipeline.
    
    Parameters
    ----------
    input_csv : str
        Path to input CSV file with SMILES data
    output_dir : str
        Directory for calculation results
    output_csv : str
        Name for output CSV file (without extension)
    """
    phase = ['gas', 'toluene']
    
    # Read input data
    sourceData = pd.read_csv(input_csv)
    smiles_dict = diction_smiles(sourceData)
    
    print(f"Processing {len(smiles_dict)} molecules...")
    
    # Setup directories
    data_dir = Path(output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate initial structures with RDKit
    print("\n=== Step 1: Generating initial structures with RDKit ===")
    rdkit_dir = Path(f'{data_dir}/RDKit')
    rdkit_dir.mkdir(parents=True, exist_ok=True)
    
    for smi_key, smiles in smiles_dict.items():
        print(f"  Processing {smi_key}...")
        rdkitGen_Mol(smi_key=smi_key, smiles=smiles, working_dir=rdkit_dir)
    
    # Copy .xyz files to phase directories
    for smi_key, ph in product(smiles_dict.keys(), phase):
        file_xyz = Path(f'{rdkit_dir}/{smi_key}.xyz')
        if file_xyz.exists():
            cp_dir = Path(f'{data_dir}/{ph}/{smi_key}')
            cp_dir.mkdir(parents=True, exist_ok=True)
            sp.run(['cp', f'{rdkit_dir}/{smi_key}.xyz', f'{cp_dir}'])
    
    # Step 2: Geometry optimization and conformer search
    print("\n=== Step 2: Geometry optimization and conformer search ===")
    for smi_key, ph in product(smiles_dict.keys(), phase):
        file_xyz = Path(f'{rdkit_dir}/{smi_key}.xyz')
        if file_xyz.exists():
            print(f"  {ph}/{smi_key}: Running xTB and CREST...")
            exe_dir = Path(f'{data_dir}/{ph}/{smi_key}')
            exe_dir.mkdir(parents=True, exist_ok=True)
            
            solvatation = "--gbsa toluene" if ph == 'toluene' else ""
            geo_Opt(smi_key=smi_key, working_dir=exe_dir, phase=ph, solvatation=solvatation)
    
    # Step 3: Calculate excitation energies
    print("\n=== Step 3: Calculating excitation energies with sTDA ===")
    for smi_key, ph in product(smiles_dict.keys(), phase):
        file_xyz = Path(f'{rdkit_dir}/{smi_key}.xyz')
        if file_xyz.exists():
            print(f"  {ph}/{smi_key}: Running sTDA/sTD-DFT...")
            exe_dir = Path(f'{data_dir}/{ph}/{smi_key}')
            solvatation = "--gbsa toluene" if ph == 'toluene' else ""
            excitationEner_Calc(smi_key=smi_key, working_dir=exe_dir, phase=ph, solvatation=solvatation)
    
    # Step 4: Extract results
    print("\n=== Step 4: Extracting results ===")
    data_Extract(phase=phase, smiles_keys=smiles_dict.keys(), working_dir=data_dir, csv_file=output_csv)
    
    print(f"\n=== Pipeline complete! Results saved to {output_csv}.csv ===")


if __name__ == "__main__":
    semiEmpi_tadf()
