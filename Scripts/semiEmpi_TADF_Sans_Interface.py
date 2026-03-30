# Date: 25/03/22
# Réécriture du code (exécution en local)
# This code calculate various electronic properties concerning TADF OLED
# * ES0, ET1, ES1
# * Homo-Lumo gap
# * Fluorescence Energy and Singlet-Triplet energy gap
# * Oscillator strength and lifetime
# Data files of each molecule are stored in a sub-directrory
#
# ********* Solvation is taking into account through Toluene solvent **********

#  Common packages
from pathlib import Path
import subprocess as sp

# pour créer une à partir de deux autres, ressemble à zip()
from itertools import product

# Classes
from rdkitGen_Mol import rdkitGen_Mol
from geo_Opt import geo_Opt
from excitationEner_Calc import excitationEner_Calc
from data_Extract import data_Extract

import pandas as pd
import ast

# Main self.script
def CalculInterface():


    def diction_smiles(sourceData):
        # Initialize the result dictionary
        smiles_dict = {}
        
        # Iterate through each row
        for index, row in sourceData.iterrows():
            compound_names = row['compound.names']
            compound_smiles = row['compound.SMILES']
            
            try:
                # Parse the vector string to get actual list/vector
                # This handles cases where the vector is stored as a string representation
                if isinstance(compound_names, str):
                    # Try to evaluate as a Python literal (list, tuple, etc.)
                    try:
                        names_vector = ast.literal_eval(compound_names)
                    except (ValueError, SyntaxError):
                        # If it fails, try splitting by common delimiters
                        if ',' in compound_names:
                            names_vector = [x.strip() for x in compound_names.split(',')]
                        elif ';' in compound_names:
                            names_vector = [x.strip() for x in compound_names.split(';')]
                        else:
                            # If no delimiter found, treat as single element
                            names_vector = [compound_names.strip()]
                else:
                    # If it's already a list or other iterable
                    names_vector = compound_names
                
                # Get the last element of the vector as the key
                if names_vector and len(names_vector) > 0:
                    key = min(names_vector, key=len)
                    
                    # Transform the SMILES value to string with quotes
                    value = f'{compound_smiles}'
                    
                    # Add to dictionary
                    smiles_dict[key] = value
                    
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue
        return smiles_dict
    
    
    def semiEmpi_tadf():
        """
        This function calculates various electronic structures concerning TADF OLED.
        - ES0, ET1, ES1
        - Fluorescence Energy and Singlet-Triplet energy gap
        - Oscillator strength and lifetime
        Data files of each molecule are stored in a sub-directory.
    
        """
        
        # Les phases
        phase = ['gas', 'toluene']
        
        #smiles_dict = { 'butan': 'CC'}
        # Read the CSV file
        sourceData = pd.read_csv("unique_subsidiary_database.csv")
        
        smiles_dict = diction_smiles(sourceData)
        
        print(smiles_dict)
        
        # Recupération des noms du répertoire et du fichier .csv
        # Repertoire des résultats
        # NOTE: Update this path to match your desired output directory
        # Default: './Data_calculation_747Mol' (matches repository structure)
        data_dir = Path('./Data_calculation_747Mol')
        data_dir.mkdir(parents=True, exist_ok=True)
        # Fichier .csv de résultats
        # NOTE: Output CSV files will be named 'Data_AllGas_results.csv' and 'Data_AllTol_results.csv'
        csv_file = 'Data_All'
        
        # Lancement du calcul dans un thread séparé
        def run_calculation():
            #try:
            # Conversion des SMILES en fichier .xyz et optimisation
            rdkit_dir = Path(f'{data_dir}/RDKit')
            rdkit_dir.mkdir(parents=True, exist_ok=True)
            for smi_key, smiles in smiles_dict.items():
                
                rdkitGen_Mol(smi_key=smi_key, smiles=smiles, working_dir=rdkit_dir)
            
            # Copie des fichiers .xyz dans les répertoires correspondants aux deux phases
            for smi_key,  ph in product(smiles_dict.keys(), phase):
                file_xyz = Path(f'{rdkit_dir}/{smi_key}.xyz') # On vérie qu'il existe
                if file_xyz.exists():
                    cp_dir = Path(f'{data_dir}/{ph}/{smi_key}')
                    cp_dir.mkdir(parents=True, exist_ok=True)
                    sp.run(['cp', f'{rdkit_dir}/{smi_key}.xyz', f'{cp_dir}'])
            
            for smi_key,  ph in product(smiles_dict.keys(), phase):
                file_xyz = Path(f'{rdkit_dir}/{smi_key}.xyz') # On vérie qu'il existe
                if file_xyz.exists():
                    exe_dir = Path(f'{data_dir}/{ph}/{smi_key}')
                    exe_dir.mkdir(parents=True, exist_ok=True)
                    
                    Solvatation = "--gbsa toluene" if ph=='toluene' else ""
                    
                    # Optimisation de géométrie et recherche de conformère
                    #self.progress_labelMol.config(text=f"Phase {ph}: {smi_key} -> xTB et CREST")
                    geo_Opt(smi_key=smi_key, working_dir=exe_dir, phase=ph, solvatation=Solvatation)
                
                    # Calcul des énergies d'excitation
                    #self.progress_labelMol.config(text=f"Phase {ph}: {smi_key} -> sTDA")
                    excitationEner_Calc(smi_key=smi_key, working_dir=exe_dir, phase=ph, solvatation=Solvatation)
            
            
            # Extraction des résultats
            data_Extract(phase=phase, smiles_keys=smiles_dict.keys(), working_dir=data_dir, csv_file=csv_file)
        
        #Lancement du calcul dans un thread
        run_calculation()
        
    semiEmpi_tadf()

# Point d'entrée du script
if __name__ == "__main__":
    CalculInterface()
