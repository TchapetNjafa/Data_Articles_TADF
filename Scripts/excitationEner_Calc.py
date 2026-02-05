#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate vertical excitation energies using sTDA and sTD-DFT methods.

This module computes excitation energies following the workflow described in
https://arxiv.org/abs/2502.20410
"""

import os
from pathlib import Path


class excitationEner_Calc:
    """
    Calculate vertical excitation energies using sTDA and sTD-DFT.
    
    Parameters
    ----------
    smi_key : str
        Unique identifier for the molecule
    working_dir : Path
        Directory where calculation files will be saved
    phase : str
        Calculation phase ('gas' or 'toluene')
    solvatation : str
        Solvation model option (e.g., '--gbsa toluene' or '')
    
    Returns
    -------
    None
        Creates .dat and .log files with excitation energies
    """
    
    def __init__(self, smi_key: str, working_dir: str, phase: str, solvatation: str) -> None:
        self.smi_key = smi_key
        self.working_dir = working_dir
        self.phase = phase
        self.solvatation = solvatation
        
        s0 = 'S0'
        t1 = 'T1'
        
        # Optimized .xyz files (input)
        self.optS0_xyz = Path(self.working_dir/f'{self.smi_key}_{self.phase}_{s0}_finalOpt.xtbopt.xyz')
        self.optT1_xyz = Path(self.working_dir/f'{self.smi_key}_{self.phase}_{t1}_finalOpt.xtbopt.xyz')
        
        # Log file for xtb4stda
        self.logS0 = Path(self.working_dir/f'{self.smi_key}_{self.phase}_{s0}_xtb4stda.log')
        
        # Calculate S0->S1 and S0->T1 excitations
        if (not self.logS0.exists()) and (self.optS0_xyz.exists()):
            # S0->S1 calculations
            self.call_stda(xyz_file=self.optS0_xyz, stateB=s0, stateE='S1')
            self.call_stda(xyz_file=self.optS0_xyz, stateB=s0, stateE='S1', method='-rpa', needsxtb4stda=False)
            
            # S0->T1 calculations
            self.call_stda(xyz_file=self.optS0_xyz, stateB=s0, stateE=t1, singletORtriplet='-t', needsxtb4stda=False)
            self.call_stda(xyz_file=self.optS0_xyz, stateB=s0, stateE=t1, singletORtriplet='-t', method='-rpa', needsxtb4stda=False, clean=True)
    
    def call_xtb4stda(self, xyz_file, state='S0', uhf=''):
        """
        Calculate ground state with xtb4stda to generate wfn.xtb file for sTDA.
        
        Parameters
        ----------
        xyz_file : Path
            Input .xyz file
        state : str, optional
            Electronic state ('S0' or 'T1'), default 'S0'
        uhf : str, optional
            UHF option for triplet states (e.g., '--uhf 2'), default ''
        """
        log_file = self.working_dir/f'{self.smi_key}_{self.phase}_{state}_xtb4stda.log'
        os.system(f'xtb4stda {xyz_file} {self.solvatation} {uhf} > {log_file}')
    
    def call_stda(self, xyz_file, uhf='', singletORtriplet='', method='', stateB='S0', stateE='S1', needsxtb4stda=True, clean=False):
        """
        Calculate excitation energies with sTDA or sTD-DFT.
        
        Parameters
        ----------
        xyz_file : Path
            Input .xyz file
        uhf : str, optional
            UHF option for xtb4stda, default ''
        singletORtriplet : str, optional
            '' for singlet-singlet or '-t' for singlet-triplet, default ''
        method : str, optional
            '' for sTDA or '-rpa' for sTD-DFT, default ''
        stateB : str, optional
            Initial state, default 'S0'
        stateE : str, optional
            Final state, default 'S1'
        needsxtb4stda : bool, optional
            Whether to call xtb4stda first, default True
        clean : bool, optional
            Whether to clean up xtb4stda files, default False
        """
        method_name = "stda" if method == "" else "stddft"
        tda_file = Path(self.working_dir/f'{self.smi_key}_{self.phase}_{stateB}{stateE}_{method_name}.dat')
        log_file = Path(self.working_dir/f'{self.smi_key}_{self.phase}_{stateB}{stateE}_{method_name}.log')
        
        if needsxtb4stda:
            self.call_xtb4stda(xyz_file=xyz_file, state=stateB, uhf=uhf)
        
        os.system(f'stda -xtb {singletORtriplet} -e 10 {method} > {log_file} && mv tda.dat {tda_file}')
        
        if clean:
            os.system('rm -f charge* wbo wfn.xtb')
        
        if not tda_file.exists():
            os.system('rm -f apbmat')
