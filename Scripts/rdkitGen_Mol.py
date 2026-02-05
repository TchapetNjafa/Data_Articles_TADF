#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate initial molecular structures from SMILES using RDKit.

This module creates .xyz files optimized with MMFF94s force field from SMILES strings.
"""

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem


class rdkitGen_Mol:
    """
    Generate optimized .xyz files from SMILES using RDKit's MMFF94s force field.
    
    Parameters
    ----------
    smi_key : str
        Unique identifier for the molecule
    smiles : str
        SMILES string representation of the molecule
    working_dir : Path
        Directory where the .xyz file will be saved
    
    Returns
    -------
    None
        Creates an optimized .xyz file in the working directory
    """
    
    def __init__(self, smi_key: str, smiles: str, working_dir: str) -> None:
        self.smi_key = smi_key
        self.smiles = smiles
        self.working_dir = working_dir
        
        # Execution
        self.build_mol()
    
    def build_mol(self):
        """Generate MMFF94s-optimized molecules and save as .xyz files."""
        file_xyz = Path(self.working_dir / f'{self.smi_key}.xyz')
        
        if not self.smiles:
            raise ValueError("Invalid SMILES string")
        
        if not file_xyz.exists():
            try:
                mol_rdkit = Chem.MolFromSmiles(self.smiles)
                mol_rdkit = Chem.AddHs(mol=mol_rdkit, addCoords=True)
                AllChem.EmbedMolecule(mol=mol_rdkit, randomSeed=0)
                
                if mol_rdkit.GetNumConformers() == 0:
                    raise ValueError(f"No conformer generated for {self.smi_key}")
                
                AllChem.MMFFOptimizeMolecule(mol=mol_rdkit, mmffVariant="MMFF94s", maxIters=300)
                Chem.rdMolTransforms.CanonicalizeMol(mol=mol_rdkit, normalizeCovar=True, ignoreHs=False)
                self.mol_xyz = Chem.MolToXYZFile(mol=mol_rdkit, filename=file_xyz)
                
            except Exception as e:
                print(f"MMFF optimization failed: {e}")
                try:
                    print("Trying UFF optimization as fallback...")
                    AllChem.UFFOptimizeMolecule(mol=mol_rdkit, maxIters=300)
                    Chem.rdMolTransforms.CanonicalizeMol(mol=mol_rdkit, normalizeCovar=True, ignoreHs=False)
                    self.mol_xyz = Chem.MolToXYZFile(mol=mol_rdkit, filename=file_xyz)
                    print("UFF optimization successful")
                except Exception as e2:
                    print(f"UFF optimization also failed: {e2}")
        else:
            print(f"File {file_xyz} already exists, skipping")
