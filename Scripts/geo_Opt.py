#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry optimization and conformer search for TADF molecules.

This module performs geometry optimization using xTB and conformer search using CREST
following the workflow described in https://arxiv.org/abs/2502.20410
"""

import os
import psutil
from pathlib import Path


class geo_Opt:
    """
    Perform geometry optimization and conformer search using xTB and CREST.
    
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
        Creates optimized .xyz files and intermediate calculation files
    """
    
    def __init__(self, smi_key: str, working_dir: str, phase: str, solvatation: str) -> None:
        self.smi_key = smi_key
        self.working_dir = working_dir
        self.phase = phase
        self.solvatation = solvatation
        
        self.nbrCpu = psutil.cpu_count() - 8
        s0 = 'S0'
        t1 = 'T1'
        
        # Define file paths for S0 optimization
        self.rdkit_xyz = Path(self.working_dir/f'{self.smi_key}.xyz')
        self.preOpt_s0_xyz = Path(self.working_dir/f'{self.smi_key}_{self.phase}_{s0}_preOpt.xtbopt.xyz')
        self.optS0_xyz = Path(self.working_dir/f'{self.smi_key}_{self.phase}_{s0}_finalOpt.xtbopt.xyz')
        self.crest_s0_xyz = Path(self.working_dir/f'{self.smi_key}_{self.phase}_{s0}_crest.xyz')
        
        # Define file paths for T1 optimization
        self.preOpt_t1_xyz = Path(self.working_dir/f'{self.smi_key}_{self.phase}_{t1}_preOpt.xtbopt.xyz')
        self.optT1_xyz = Path(self.working_dir/f'{self.smi_key}_{self.phase}_{t1}_finalOpt.xtbopt.xyz')
        self.crest_t1_xyz = Path(self.working_dir/f'{self.smi_key}_{self.phase}_{t1}_crest.xyz')
        
        # Clean up temporary files
        os.system('rm -f crest* coo* gfn* wbo .UHF && rm -rf TRIALMD')
        
        # S0 optimization workflow
        if self.rdkit_xyz.exists():
            if not self.preOpt_s0_xyz.exists():
                self.opt_xtb(xyz_file=self.rdkit_xyz, etapOpt='preOpt', state=s0)
        
            if self.preOpt_s0_xyz.exists():
                self.conformer_search(xyz_file=self.preOpt_s0_xyz, state=s0)
        
            if self.crest_s0_xyz.exists():
                self.opt_xtb(xyz_file=self.crest_s0_xyz, etapOpt='finalOpt', state=s0)
        
        # T1 optimization workflow
        if self.optS0_xyz.exists():
            tbl_uhf = '--uhf 2' if self.phase != 'gas' else '--spinpol --tblite --uhf 2'
            
            if not self.preOpt_t1_xyz.exists():
                self.opt_xtb(xyz_file=self.optS0_xyz, etapOpt='preOpt', state=t1, uhf=tbl_uhf)
        
            if self.preOpt_t1_xyz.exists():
                self.conformer_search(xyz_file=self.preOpt_t1_xyz, state=t1, uhf='--uhf 2')
                if not self.crest_t1_xyz.exists():
                    os.system('rm -f crest* coo* gfn* wbo .UHF && rm -rf TRIALMD')
                    self.conformer_search(xyz_file=self.preOpt_t1_xyz, state=t1)
        
            if self.crest_t1_xyz.exists():
                self.opt_xtb(xyz_file=self.crest_t1_xyz, etapOpt='finalOpt', state=t1, uhf=tbl_uhf)
        
    def opt_xtb(self, xyz_file, etapOpt: str, state='S0', uhf=''):
        """
        Perform geometry optimization with xTB.
        
        Parameters
        ----------
        xyz_file : Path
            Input .xyz file
        etapOpt : str
            Optimization step ('preOpt' or 'finalOpt')
        state : str, optional
            Electronic state ('S0' or 'T1'), default 'S0'
        uhf : str, optional
            UHF option for triplet states (e.g., '--uhf 2'), default ''
        """
        self.log_file = Path(f'{self.working_dir}/{self.smi_key}_{self.phase}_{state}_{etapOpt}.log')
        self.hessian_file = Path(f'{self.working_dir}/{self.smi_key}_{self.phase}_{state}_{etapOpt}.hessian')
        
        if not self.hessian_file.exists():
            namespace = f'{self.working_dir}/{self.smi_key}_{self.phase}_{state}_{etapOpt}'
            cmd = f'xtb {xyz_file} --gfn 2 {uhf} {self.solvatation} --ohess vtight --parallel {self.nbrCpu} --molden --ceasefiles --namespace {namespace} > {self.log_file}'
            cmd += f' && mv {namespace}.molden.input {namespace}.molden'
            os.system(cmd)
        
    def conformer_search(self, xyz_file, state='S0', uhf=''):
        """
        Search for the best conformer using CREST.
        
        Parameters
        ----------
        xyz_file : Path
            Pre-optimized .xyz file
        state : str, optional
            Electronic state ('S0' or 'T1'), default 'S0'
        uhf : str, optional
            UHF option for triplet states (e.g., '--uhf 2'), default ''
        """
        crest_dir = Path(f'{self.working_dir}/crest_{self.smi_key}_{state}')
        crest_dir.mkdir(parents=True, exist_ok=True)
        crest_bestXYZ = Path(f'{crest_dir}/crest_best.xyz')
        
        if not crest_bestXYZ.exists():
            log_file = f'{self.working_dir}/{self.smi_key}_{self.phase}_{state}_crest.log'
            output_xyz = f'{self.working_dir}/{self.smi_key}_{self.phase}_{state}_crest.xyz'
            cmd = f'crest {xyz_file} --gfn2 --mquick --prop hess --noreftopo {uhf} {self.solvatation} --T {self.nbrCpu} > {log_file}'
            cmd += f' && cp crest_best.xyz {output_xyz} && mv cre* gfn* ensemble* coo* wbo {crest_dir}'
            os.system(cmd)

