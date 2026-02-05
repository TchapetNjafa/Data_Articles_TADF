#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 08:45:44 2025

@author: tchapet

Permet d'extraire les données sous forme de fichier .csv
 et faire les tracés (voir comment appeler aussi directement MultiWfn)
"""

# Common package
import sys
import subprocess as sp
from pathlib import Path
import numpy as np
from datetime import datetime

# pour créer une à partir de deux autres, ressemble à zip()
from itertools import product

## convertion Hartree à électronvol
from pyscf.data.nist import HARTREE2EV as au2ev

## convertion eV pour nm (hc/\lambda)
import scipy.constants as const

# Conversion of Hartree to kcal per mol
au2kcal = 627.509474
# Conversion eV to Joule
ev2jl = 1.6021766339999E-19


# Pandas
try:
    import pandas as pd
except ImportError:
    sp.check_call([sys.executable, '-m', 'pip', 'install', 'pandas', '-U'])
    import pandas as pd
    

class data_Extract:
    """
        Extraction du maximum de données calculées.
        
        Parameters
        ----------
        phase : array
            DESCRIPTION. The array containing the phases where the calculations were made.
        smiles_keys : array
            Array of keys of SMILES extracted from the dictionnary containing keys and SMILES
        working_dir: str
            DESCRIPTION. The working directory where data files were saved
        csv_file: str
            DESCRIPTION. The name of the .csv file where results will be saved.
            Don't add ".cvs" at the end of the name.

        Returns
        -------
        The .csv file where results will be saved. The 8 first lines are resutls in
        gas phase, while the last 8 lines are for the toluene phase.

        """

    def __init__(self, phase, smiles_keys, working_dir, csv_file) -> None:
        self.phase = phase
        self.smi_key = smiles_keys
        self.working_dir = working_dir
        self.csv_file = csv_file
        
        
        list_Data =[
            "HOMO-LUMO gap",
            r"$\Delta E_r(S_0\to T_1)$",
            r"$\underset{sTDA}{\Delta E_v}(S_0\to T_1)$", r"$\underset{sTD-DFT}{\Delta E_v}(S_0\to T_1)$",
            r"$\underset{sTDA}{\Delta E_v}(S_0\to S_1)$", r"$\underset{sTD-DFT}{\Delta E_v}(S_0\to S_1)$",
            r"$\underset{sTDA}{\Delta E_r}(S_0\to S_1)$", r"$\underset{sTD-DFT}{\Delta E_r}(S_0\to S_1)$",
            r"$\underset{sTDA}{\Delta E_{ST}}$", r"$\underset{sTD-DFT}{\Delta E_{ST}}$",
            r"$\underset{sTDA}{S_{shift}(T_1)}$", r"$\underset{sTD-DFT}{S_{shift}(T_1)}$",
            r"$\underset{sTDA}{S_{shift}(S_1)}$", r"$\underset{sTD-DFT}{S_{shift}(S_1)}$",
            r"$\underset{sTDA}{\lambda_{abs}}$", r"$\underset{sTD-DFT}{\lambda_{abs}}$",
            r"$\underset{sTDA}{\lambda_{PL}}$", r"$\underset{sTD-DFT}{\lambda_{PL}}$",
            r"$\underset{sTDA}{f_{12}}(S_0\to S_1)$", r"$\underset{sTD-DFT}{f_{12}}(S_0\to S_1)$",
            r"$\tau_{sTDA}$", r"$\tau_{sTD-DFT}$",
            r"$\underset{sTDA}{MOF}$", r"$\underset{sTD-DFT}{MOF}$",
            r"$\underset{sTDA}{\lambda_{shift}}$", r"$\underset{sTD-DFT}{\lambda_{shift}}$",
            r"$\underset{sTDA}{t_v}(S_0\to S_1)$", r"$\underset{sTD-DFT}{t_v}(S_0\to S_1)$",
            r"$t_r(S_0\to S_1)$",
            r"$\underset{sTDA}{t_v}(S_0\to T_1)$", r"$\underset{sTD-DFT}{t_v}(S_0\to T_1)$",
            r"$\underset{sTDA}{k_{nr}}$", r"$\underset{sTD-DFT}{k_{nr}}$",
            r"$\underset{sTDA}{k_{rISC}}$", r"$\underset{sTD-DFT}{k_{rISC}}$",
            r"$\underset{sTDA}{k_{ISC}}$", r"$\underset{sTD-DFT}{k_{ISC}}$",
            r"$\underset{sTDA}{\eta_{TADF}}$", r"$\underset{sTD-DFT}{\eta_{TADF}}$"
            ]
        
        All_Data = []
        
        for ph, smi in product(self.phase, self.smi_key):
            All_Data.append(self.extrac_from_log(phase=ph, smi_key=smi))
        
        long_smi = list(self.smi_key) + list(self.smi_key)
        frameData = pd.DataFrame(data=All_Data, index=long_smi, columns=list_Data)
        frameData.index.name = 'Molecule'
        
        # Sauvegarde des résultats dans le fichier .csv
        frameData.to_csv(f'{self.csv_file}.csv')
        
    def ev_to_nm (self, energy: float) -> float:
        """
        Pour convertir l'electronvolt en nanomètre'

        Parameters
        ----------
        energy : float
            DESCRIPTION. L'énergie en électronvolt.

        Returns
        -------
        float
            DESCRIPTION. La conversion en nanomètre de l'énergie.

        """
        # hc
        hc = ((const.h * const.c) / const.e) * 1e9
        longueur_onde = hc / energy
        
        return longueur_onde
        
    def extrac_from_log(self, phase, smi_key) -> np.array:
        """
        Permet d'extraire les données à partir des fichiers .log et .dat

        Parameters
        ----------
        smi_key : str
            DESCRIPTION. Key of the SMILES
        phase : TYPE
            DESCRIPTION. Phase in which the computation was made

        Returns
        -------
        np.array
            DESCRIPTION. Vecteur contenant toutes les informations sur la molécule

        """
        
        # Vecteur de données
        Data = []
        
        # log xTB preoptimisation
        preOpt_S0 = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_S0_preOpt.log')
        prelOpt_T1 = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_T1_preOpt.log')
        
        # log CREST rechèche du meilleur conformère        
        crest_S0 = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_S0_crest.log')
        crest_T1 = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_T1_crest.log')
        
        # log xTB optimisation finale
        finalOpt_S0 = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_S0_finalOpt.log')
        finalOpt_T1 = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_T1_finalOpt.log')
        
        # log sTDA
        stda_S0S1log = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_S0S1_stda.log')
        stda_S0T1log = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_S0T1_stda.log')
        
        # log sTD-DFT
        stddft_S0S1log = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_S0S1_stddft.log')
        stddft_S0T1log = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_S0T1_stddft.log')
        
        # fichiers .dat sTDA
        stda_S0S1dat = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_S0S1_stda.dat')
        stda_S0T1dat = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_S0T1_stda.dat')
        stda_T1T2dat = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_T1T2_stda.dat')
        
        # fichiers .dat sTD-DFT
        stddft_S0S1dat = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_S0S1_stddft.dat')
        stddft_S0T1dat = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_S0T1_stddft.dat')
        stddft_T1T2dat = Path(f'{self.working_dir}/{phase}/{smi_key}/{smi_key}_{phase}_T1T2_stddft.dat')
        
        # Extraction gap HOMO-LUMO
        try:
            HL_line = [li for li in finalOpt_S0.read_text().splitlines() if 'HOMO-LUMO GAP' in li]
            HL_line = HL_line[0].split()
            HL_val = float(HL_line[3])
        except:
            HL_val = str('NA')
        
        # Énergie de relaxation S0->T1
        try:
            # Pour le fondamental S0
            ES0_line = [li for li in finalOpt_S0.read_text().splitlines() if 'TOTAL ENERGY' in li]
            ES0_line = ES0_line[0].split()
            ES0_val = float(ES0_line[3])
            # Pour T1
            ET1_line = [li for li in finalOpt_T1.read_text().splitlines() if 'TOTAL ENERGY' in li]
            ET1_line = ET1_line[0].split()
            ET1_val = float(ET1_line[3])
            
            # Valeur de l'énergie en electronvolt T1
            Er_S0T1 = float((ET1_val-ES0_val)*au2ev)
        except:
            Er_S0T1 = str('NA')
        
        # Les énergies verticales (à partir des fichiers .dat) en eV et forces d'oscillateur
        # 1. S0->S1 et S0->T1
        try:
            counter_line = 0
            for li in stda_S0S1dat.read_text().splitlines():
                counter_line += 1
                if 'DATXY' in li:
                    break
            S1_stda_line = stda_S0S1dat.read_text().splitlines()[counter_line].split()
            S1_stddft_line = stddft_S0S1dat.read_text().splitlines()[counter_line].split()
            
            T1_stda_line = stda_S0T1dat.read_text().splitlines()[counter_line].split()
            T1_stddft_line = stddft_S0T1dat.read_text().splitlines()[counter_line].split()
            
            Ev_S0S1_stda = float(S1_stda_line[1])
            Ev_S0S1_stddft = float(S1_stddft_line[1])
            osc_S0S1_stda = float(S1_stda_line[2])
            osc_S0S1_stddft = float(S1_stddft_line[2])
            
            Ev_S0T1_stda = float(T1_stda_line[1])
            Ev_S0T1_stddft = float(T1_stddft_line[1])
        except: # risqué car tout est censé fonctionner
            Ev_S0S1_stda = str('NA')
            Ev_S0S1_stddft = str('NA')
            osc_S0S1_stda = str('NA')
            osc_S0S1_stddft = str('NA')
            
            Ev_S0T1_stda = str('NA')
            Ev_S0T1_stddft = str('NA')
        
        #  S_shiftT1
        # 1. Pour stda
        try:
            S_shift_stdaT1 = Ev_S0T1_stda - Er_S0T1
        except:
            S_shift_stdaT1 = str('NA')
        # 2. stddft
        try:
            S_shift_stddftT1 = Ev_S0T1_stddft - Er_S0T1
        except:
            S_shift_stddftT1 = str('NA')
        
        #  S_shiftS1 evalué à partir de ce qui précède
        # 1. Pour stda
        try:
            S_shift_stdaS1 = Ev_S0S1_stda * (S_shift_stdaT1/Ev_S0T1_stda)
        except:
            S_shift_stdaS1 = str('NA')
        # 2. stddft
        try:
            S_shift_stddftS1 = Ev_S0S1_stddft * (S_shift_stddftT1/Ev_S0T1_stddft)
        except:
            S_shift_stddftS1 = str('NA')
        
        # Energies S1->S0 relaxées
        # FIXME : permet d'obtenir les longueurs d'onde de fluorescence \lambda_{PL}
        # 1. stda
        if isinstance(S_shift_stdaS1, float):
            Er_S0S1_stda = Ev_S0S1_stda - S_shift_stdaS1
        elif isinstance(S_shift_stddftS1, float):
            Er_S0S1_stda = Ev_S0S1_stda - S_shift_stddftS1
        else:
            Er_S0S1_stda = str('NA')
        # 2. stddft
        if isinstance(S_shift_stddftS1, float):
            Er_S0S1_stddft = Ev_S0S1_stddft - S_shift_stddftS1
        elif isinstance(S_shift_stdaS1, float):
            Er_S0S1_stddft = Er_S0S1_stddft - S_shift_stdaS1
        else:
            Er_S0S1_stddft = str('NA')

        # \Delta E_{ST} = \Delta Er(S1) - \Delta Er(T1) (Er_S0T1)
        # 1. stda
        try:
            delta_E_ST_stda = Er_S0S1_stda - Er_S0T1
        except:
            delta_E_ST_stda = str('NA')
        #2. stddft
        try:
            delta_E_ST_stddft = Er_S0S1_stddft - Er_S0T1
        except:
            delta_E_ST_stddft = str('NA')
        
        # Les longueurs d'onde d'absorption verticale \lambda_{abs} S0->S1
        # 1. stda
        try:
            lambda_abs_stda = self.ev_to_nm(Ev_S0S1_stda) 
        except:
            lambda_abs_stda = str('NA')
        # 2. stddft
        try:
            lambda_abs_stddft = self.ev_to_nm(Ev_S0S1_stddft) 
        except:
            lambda_abs_stddft = str('NA')
        
        # Les longueurs d'onde de fluorescence \lambda_{PL}  S1->S0
        # 1. stda
        try:
            lambda_PL_stda = self.ev_to_nm(Er_S0S1_stda)
        except:
            lambda_PL_stda = str('NA')
        
        # 2. stddft
        try:
            lambda_PL_stddft = self.ev_to_nm(Er_S0S1_stddft)
        except:
            lambda_PL_stddft = str('NA')
        
        # Shift longueurs d'onde
        # 1. stda
        try:
            shift_lambda_stda = float(lambda_PL_stda-lambda_abs_stda)
        except:
            shift_lambda_stda = str('NA')
        # 2. stddft
        try:
            shift_lambda_stddft = float(lambda_PL_stddft-lambda_abs_stddft)
        except:
            shift_lambda_stddft = str('NA')

        # Fonction multi-objective
        # 1. pour stda
        if isinstance(delta_E_ST_stda, float) and isinstance(Er_S0T1, float):
            MOF_stda = osc_S0S1_stda  - delta_E_ST_stda - abs(Er_S0T1-3.2)
        else:
            MOF_stda = str('NA')
        # 2. pour stddft
        if isinstance(delta_E_ST_stddft, float) and isinstance(Er_S0T1, float):
            MOF_stddft = osc_S0S1_stddft  - delta_E_ST_stddft - abs(Er_S0T1-3.2)
        else:
            MOF_stddft = str('NA')
        
        # Lifetime \tau (avec énergie relaxée Er_S0S1) en ns
        # 1. stda
        try:
            tau_stda = float(23.046 / ((Er_S0S1_stda)**2 * osc_S0S1_stda))
        except:
            tau_stda = str('NA')
        # 2. stddft
        try:
            tau_stddft = float(23.046 / ((Er_S0S1_stddft)**2 * osc_S0S1_stddft))
        except:
            tau_stddft = str('NA')
        
        # non-radiative decay rate (k_{nr})
        # 1. stda
        try:
            ndcay_stda = float((1-osc_S0S1_stda)/(tau_stda*1e-9))
        except:
            ndcay_stda = str('NA')
        # 2. stddft
        try:
            ndcay_stddft = float((1-osc_S0S1_stddft)/(tau_stddft*1e-9))
        except:
            ndcay_stddft = str('NA')
        
        # rate constant for reverse intersystem crossing (k_{rISC})
        # 1. stda
        try:
            rISC_stda = np.float128(1e7*np.exp(-delta_E_ST_stda/(300*8.61734355e-05)))
        except:
            rISC_stda = str('NA')
         # 2. stddft
        try:
             rISC_stddft = np.float128(1e7*np.exp(-delta_E_ST_stddft/(300*8.61734355e-05)))
        except:
             rISC_stddft = str('NA')
            
        # rate constant for intersystem crossing (kISC) 
        # 1. stda
        try:
            ISC_stda = np.float128(10*rISC_stda)
        except:
            ISC_stda = str('NA')
        # 2. stddft
        try:
            ISC_stddft = np.float128(10*rISC_stddft)
        except:
            ISC_stddft = str('NA')
        
        # TADF efficiency (\eta_{TADF})
        # 1. stda
        try:
            eff_stda = np.float128(rISC_stda/(rISC_stda+ISC_stda+ndcay_stda))
        except:
            eff_stda = str('NA')
        # 2. stddft
        try:
            eff_stddft = np.float128(rISC_stddft/(rISC_stddft+ISC_stddft+ndcay_stddft))
        except:
            eff_stddft = str('NA')
        
        # Temps S0->S1 vertical
        # 1. stda
        try:
            # conversion en seconde
            t_debut = stda_S0S1log.read_text().splitlines()[0]
            t_debut = t_debut.replace(". ", ".") # retirer les espaces genants
            t_debut = t_debut.replace(": ", ":")
            t_debut = t_debut.replace(" :", ":")
            t_debut = datetime.strptime(t_debut, '%Y-%m-%d %H:%M:%S.%f')
            t_fin = stda_S0S1log.read_text().splitlines()[-1]
            t_fin = t_fin.replace(". ", ".")
            t_fin = t_fin.replace(": ", ":")
            t_fin = t_fin.replace(" :", ":")
            t_fin = datetime.strptime(stda_S0S1log.read_text().splitlines()[-1], '%Y-%m-%d %H:%M:%S.%f')
            tv_S0S1_stda = float(t_fin.timestamp() - t_debut.timestamp())
        except:
            tv_S0S1_stda = str('NA')
        # 2. stddft
        try:
            # conversion en seconde
            t_debut = stddft_S0S1log.read_text().splitlines()[0]
            t_debut = t_debut.replace(". ", ".") # retirer les espaces genants
            t_debut = t_debut.replace(": ", ":")
            t_debut = t_debut.replace(" :", ":")
            t_debut = datetime.strptime(t_debut, '%Y-%m-%d %H:%M:%S.%f')
            t_fin = stddft_S0S1log.read_text().splitlines()[-1]
            t_fin = t_fin.replace(". ", ".")
            t_fin = t_fin.replace(": ", ":")
            t_fin = t_fin.replace(" :", ":")
            t_fin = datetime.strptime(t_fin, '%Y-%m-%d %H:%M:%S.%f')
            tv_S0S1_stddft = float(t_fin.timestamp() - t_debut.timestamp())
        except:
            tv_S0S1_stddft = str('NA')
        
        
        # Temps S0->T1 vertical
        # 1. stda
        try:
            # conversion en seconde
            t_debut = stda_S0T1log.read_text().splitlines()[0]
            t_debut = t_debut.replace(". ", ".") # retirer les espaces genants
            t_debut = t_debut.replace(": ", ":")
            t_debut = t_debut.replace(" :", ":")
            t_debut = datetime.strptime(t_debut, '%Y-%m-%d %H:%M:%S.%f')
            t_fin = stda_S0T1log.read_text().splitlines()[-1]
            t_fin = t_fin.replace(". ", ".")
            t_fin = t_fin.replace(": ", ":")
            t_fin = t_fin.replace(" :", ":")
            t_fin = datetime.strptime(t_fin, '%Y-%m-%d %H:%M:%S.%f')
            tv_S0T1_stda = float(t_fin.timestamp() - t_debut.timestamp())
        except:
            tv_S0T1_stda = str('NA')
        # 2. stddft
        try:
            # conversion en seconde
            t_debut = stddft_S0T1log.read_text().splitlines()[0]
            t_debut = t_debut.replace(". ", ".") # retirer les espaces genants
            t_debut = t_debut.replace(": ", ":")
            t_debut = t_debut.replace(" :", ":")
            t_debut = datetime.strptime(t_debut, '%Y-%m-%d %H:%M:%S.%f')
            t_fin = stddft_S0T1log.read_text().splitlines()[-1]
            t_fin = t_fin.replace(". ", ".")
            t_fin = t_fin.replace(": ", ":")
            t_fin = t_fin.replace(" :", ":")
            t_fin = datetime.strptime(t_fin, '%Y-%m-%d %H:%M:%S.%f')
            tv_S0T1_stddft = float(t_fin.timestamp() - t_debut.timestamp())
        except:
            tv_S0T1_stddft = str('NA')
        
        # Temps S0->T1 relaxé
        try:
            #j + h + min + s
            t_pre_line = prelOpt_T1.read_text().splitlines()[-16].split()
            t_pre = float(float(t_pre_line[2])*86400+float(t_pre_line[4])*3600+float(t_pre_line[6])*60+float(t_pre_line[8]))
            
            t_crest_line = crest_T1.read_text().splitlines()[-7].split()
            t_crest = float(float(t_crest_line[2])*86400+float(t_crest_line[4])*3600+float(t_crest_line[6])*60+float(t_crest_line[8]))
            
            t_fin_line = finalOpt_T1.read_text().splitlines()[-16].split()
            t_final = float(float(t_fin_line[2])*86400+float(t_fin_line[4])*3600+float(t_fin_line[6])*60+float(t_fin_line[8]))
            
            tr_S0T1 = float(t_pre+t_crest+t_final)
        except:
            tr_S0T1 = str('NA')
        
        # Rassemblement des données
        Data.append(HL_val)
        
        Data.append(Er_S0T1)
        
        Data.append(Ev_S0T1_stda)
        Data.append(Ev_S0T1_stddft)
        
        Data.append(Ev_S0S1_stda)
        Data.append(Ev_S0S1_stddft)
        
        Data.append(Er_S0S1_stda)
        Data.append(Er_S0S1_stddft)
        
        Data.append(delta_E_ST_stda)
        Data.append(delta_E_ST_stddft)
        
        Data.append(S_shift_stdaT1)
        Data.append(S_shift_stddftT1)
        
        Data.append(S_shift_stdaS1)
        Data.append(S_shift_stddftS1)
        
        Data.append(lambda_abs_stda)
        Data.append(lambda_abs_stddft)
        
        Data.append(lambda_PL_stda)
        Data.append(lambda_PL_stddft)
        
        Data.append(osc_S0S1_stda)
        Data.append(osc_S0S1_stddft)
        
        Data.append(tau_stda)
        Data.append(tau_stddft)
        
        Data.append(MOF_stda)
        Data.append(MOF_stddft)
        
        Data.append(shift_lambda_stda)
        Data.append(shift_lambda_stddft)
        
        Data.append(tv_S0S1_stda)
        Data.append(tv_S0S1_stddft)
        
        Data.append(tr_S0T1)
        
        Data.append(tv_S0T1_stda)
        Data.append(tv_S0T1_stddft)
        
        Data.append(ndcay_stda)
        Data.append(ndcay_stddft)
        
        Data.append(rISC_stda)
        Data.append(rISC_stddft)
        
        Data.append(ISC_stda)
        Data.append(ISC_stddft)
        
        Data.append(eff_stda)
        Data.append(eff_stddft)

        
        return Data
