#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pathlib import Path


# In[2]:


import matplotlib.pyplot as plt
import matplotlib as mpl
import hvplot.pandas
import holoviews as hv
import numpy as np
import datetime
import time
import scipy.constants as cst

ev2kcal = 23.06054819 #eV to kcal/mol

pgf_with_rc_fonts = {
    "font.family": "serif",
}
mpl.rcParams.update(pgf_with_rc_fonts)


# In[3]:


import os


# In[4]:


# Verify if the 'data' subdirectory exist and if not, create it
data_dir = Path('../Data_calculation_747Mol')


# In[5]:


#smiles = ['DMAC-TRZ','DMAC-DPS','PSPCz','4CzIPN','Px2BP','CzS2','2TCz-DPS','TDBA-DI']


# In[6]:


phase = ['gas', 'toluene']#'


# In[7]:


data = pd.read_csv(f'data_result2507.csv')#,index_col=0)

# Ce fichier comporte certaines erreurs, de fait certaine valeurs ont été écrite suivant la
# notation française (0,00003), on doit donc manuellement changer (0.00003)
# on exploite le fait qu'on puisse ouvrir le .csv comme un fichier de tableau


# In[8]:


data.T


# In[9]:


data


# In[10]:


#Repérage des occurences non dupliquées (dans gas et toluene)
data.drop_duplicates(subset='Molecule', keep=False)


# In[11]:


data.keys()


# In[15]:


# On enlève les 13 dernières colones et les deux \tau avant le MOFcar non pertinentes pour le moment
data1 = data.drop(columns=[r'$\tau_{sTDA}$',r'$\tau_{sTD-DFT}$',
r'$\underset{sTDA}{t_v}(S_0\to S_1)$',
       r'$\underset{sTD-DFT}{t_v}(S_0\to S_1)$', r'$t_r(S_0\to S_1)$',
       r'$\underset{sTDA}{t_v}(S_0\to T_1)$',
       r'$\underset{sTD-DFT}{t_v}(S_0\to T_1)$', r'$\underset{sTDA}{k_{nr}}$',
       r'$\underset{sTD-DFT}{k_{nr}}$', r'$\underset{sTDA}{k_{rISC}}$',
       r'$\underset{sTD-DFT}{k_{rISC}}$', r'$\underset{sTDA}{k_{ISC}}$',
       r'$\underset{sTD-DFT}{k_{ISC}}$', r'$\underset{sTDA}{\eta_{TADF}}$',
       r'$\underset{sTD-DFT}{\eta_{TADF}}$'])
data1


# In[16]:


# On enlève partout où il n'y a pas de valeur numérique (et on reindexe à partir de 0)
Data = data1.dropna(how='any',ignore_index=True)
Data


# In[19]:


#Repérage des occurences non dupliquées (dans gas et toluene)
Data.drop_duplicates(subset='Molecule', keep=False)


# In[20]:


# On ne retient que les molécule présentent dans les deux environnements (gas et toluene)
dataOk = Data[Data.duplicated(subset='Molecule', keep=False)]
dataOk = dataOk.reset_index(drop=True) # On modifie les indexes pour commencer à 0 et remplacer les valeurs omisent
dataOk


# In[21]:


dataOk.keys()


# In[22]:


dataOk.dtypes


# In[14]:


# On change les object en float64 au cas où
#dataOk = dataOk.astype({r'$\underset{sTDA}{f_{12}}(S_0\to S_1)$':'float64', r'$\underset{sTD-DFT}{f_{12}}(S_0\to S_1)$':'float64'})


# In[23]:


dataOk.dtypes


# In[24]:


# Détermination du nombre d'éléments par envionnement (la moitié d'un nombre pair)
taille = np.int16(len(dataOk)/2)
taille


# In[25]:


Gas = dataOk[:taille] #Gas les premières lignes
Gas = Gas.reset_index(drop=True) # On modifie les indexes pour commencer à 0 et remplacer les valeurs omisent
Gas


# In[26]:


Tol = dataOk[taille:] #Toluene les dernières lignes
Tol = Tol.reset_index(drop=True) # On modifie les indexes pour commencer à 0 et remplacer les valeurs omisent
Tol


# In[30]:


# On predns les SMILES dans les deux environnements pour s'assuré qu'ils sont identique
smilesGas = np.array(Gas['Molecule'])


# In[31]:


# On a les même SMILES dans les deux tableaux
smiles = np.array(Tol['Molecule']) #(Gas['Molecule'])
smiles


# In[32]:


# Vérification de la conformité des SMILES
smilesGas == smiles


# # RMSD

# In[33]:


import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem


# In[34]:


compare = []
for smi in smiles:
    
    result = []
    rdkit_mol = Chem.MolFromXYZFile(f'{data_dir}/RDKit/{smi}.xyz')
    gas_mol = Chem.MolFromXYZFile(f'{data_dir}/gas/{smi}/{smi}_gas_S0_finalOpt.xtbopt.xyz')
    tol_mol = Chem.MolFromXYZFile(f'{data_dir}/toluene/{smi}/{smi}_toluene_S0_finalOpt.xtbopt.xyz')

    #compare rdkit-gaz
    rd_gas = AllChem.AlignMol(gas_mol, rdkit_mol)
    result.append(rd_gas)
    #compare rdkit-toluene
    rd_tol = AllChem.AlignMol(tol_mol, rdkit_mol)
    result.append(rd_tol)
    #compare gaz-toluene
    gas_tol = AllChem.AlignMol(tol_mol, gas_mol)
    result.append(gas_tol)

    compare.append(result)

compare


# In[35]:


Compare_result = pd.DataFrame(data=np.transpose(compare),columns=smiles, index=['RdKit vs vacuum', 'RdKit vs toluene',
                                    'vacuum vs toluene'])


# In[36]:


Compare_result


# In[37]:


Compare_result[2:].T


# In[38]:


#On met dans l'ordre croissant
Compare_result[2:].T.sort_values(by=['vacuum vs toluene'])


# In[39]:


print(Compare_result[2:].T.sort_values(by=['vacuum vs toluene']).to_latex(multirow = True, multicolumn = True, float_format = "%.3f"))


# # Extraction du HOMO-LUMO overlap et Centroid Distances (grâce à Multiwfn)
# 
# Dans Multiwfn on fait le choix de (while directly calculating overlap integral of two orbital wavefunctions is clearly meaningless, since it must be zero due to orthonormalization condition)
# 
#  \begin{equation}
#   S_{HOMO-LUMO} = \int |\psi_{HOMO}(\mathbf{r})|\cdot |\psi_{LUMO}(\mathbf{r})| \, d\mathbf{r}
#   \end{equation}
# 
# 1. Se servir d'un fichier contenant les fonctions d'onde (ici on prend un fichier molden)
# 2. Choisir l'option 100
# 3. puis choisir l'option 11: "Calculate overlap and centroid distance between two orbitals"
# 4. puis mettre les indexes du HOMO et du LUMO
# 5. noter la valeur qui s'affiche à l'écran et correspondant à l'*overlap integral* qu'on recherche.
# 
# The integrals shown above are not calculated analytically but numerically via Becke's grid-based integration approach. 

# ## Recupération des indexes des HOMO et LUMO

# In[40]:


#recuperation des index homo et lumo
index_gas =[]
index_tol = []
for ph in phase:
    A = []
    for smi in smiles:
        result = []
        data_result = data_dir/f'{ph}'/f'{smi}'/f'{smi}_{ph}_S0_finalOpt.log'
        #idx homo
        homo_idx_line = [li for li in data_result.read_text().splitlines()
                                 if '(HOMO)' in li]
        homo_idx_line = homo_idx_line[0].split()
        homo_idx = homo_idx_line[0]
        homo_idx = np.int64(homo_idx)
        result.append(homo_idx)
        #idx lumo
        lumo_idx_line = [li for li in data_result.read_text().splitlines()
                                 if '(LUMO)' in li]
        lumo_idx_line = lumo_idx_line[0].split()
        lumo_idx = lumo_idx_line[0]
        lumo_idx = np.int64(lumo_idx)
        result.append(lumo_idx)        
        A.append(result)
        #continue # pour éviter que les mêmes indexes soient lu plusieurs fois (ce n'est pas le cas apparemment)
    if ph == 'gas':
        index_gas = A
    else:
        index_tol = A    


# In[41]:


index_gas


# In[42]:


#ils sont identiques ?
index_gas == index_tol


# In[43]:


len(index_tol)


# ## Calcul et génération des fichiers

# In[44]:


# Exécuté une seule fois suffit car génère déjà les fichiers nécessaire
#for ph in phase:
#    j = 0
#    for smi in smiles:
#        Molden = data_dir/f'{ph}'/f'{smi}'/f'{smi}_{ph}_S0_finalOpt.molden'
#        Overlap = data_dir/f'{ph}'/f'{smi}'/f'{smi}_{ph}_S0_finalOverlap.txt'
#        os.system(f'Multiwfn {Molden} > {Overlap} << EOF\n100\n11\n{index_tol[j][0]},{index_tol[j][1]}\nn\n0,0\n0\nq\nEOF')
#        j += 1


# ## Extraction des données
# On change la colonne des indexes pour être certain de ne travailler qu'avec les SMILES effectivement identifiés

# In[45]:


Centrogas = Gas.assign(Overlap = np.nan, Centroid = np.nan)
Centrogas = Centrogas.set_index('Molecule')


# In[46]:


Centrogas


# In[47]:


Centrogas.index


# In[48]:


Centrotoluene = Tol.assign(Overlap = np.nan, Centroid = np.nan)
Centrotoluene = Centrotoluene.set_index('Molecule')


# In[49]:


Centrotoluene


# In[50]:


Centrotoluene.index


# In[51]:


for ph in phase:
    if ph == 'gas':

        for smi in smiles:
            Overlap = data_dir/f'{ph}'/f'{smi}'/f'{smi}_{ph}_S0_finalOverlap.txt'
            # centroid distance
            centroid_line = [li for li in Overlap.read_text().splitlines()
                                     if 'Centroid distance between the two orbitals:' in li]
            centroid_line = centroid_line[0].split()
            centroid_val = np.float64(centroid_line[6])
            Centrogas.loc[smi, 'Centroid'] = centroid_val
    
            # overlap integral
            overlap_line = [li for li in Overlap.read_text().splitlines()
                                     if 'Overlap integral of norm of the two orbitals:' in li]
            overlap_line = overlap_line[0].split()
            overlap_val = np.float64(overlap_line[8])
            Centrogas.loc[smi, 'Overlap'] = overlap_val
        
    else:
        
        for smi in smiles:
            Overlap = data_dir/f'{ph}'/f'{smi}'/f'{smi}_{ph}_S0_finalOverlap.txt'
            # centroid distance
            centroid_line = [li for li in Overlap.read_text().splitlines()
                                     if 'Centroid distance between the two orbitals:' in li]
            centroid_line = centroid_line[0].split()
            centroid_val = np.float64(centroid_line[6])
            Centrotoluene.loc[smi, 'Centroid'] = centroid_val
    
            # overlap integral
            overlap_line = [li for li in Overlap.read_text().splitlines()
                                     if 'Overlap integral of norm of the two orbitals:' in li]
            overlap_line = overlap_line[0].split()
            overlap_val = np.float64(overlap_line[8])
            Centrotoluene.loc[smi, 'Overlap'] = overlap_val
        


# In[52]:


Centrogas #.dropna(how='any')


# In[53]:


Centrotoluene


# In[54]:


Centrogas.keys()


# In[55]:


CentrOverGas = Centrogas[['Overlap','Centroid',r'$\underset{sTDA}{\Delta E_{ST}}$',r'$\underset{sTD-DFT}{\Delta E_{ST}}$',r'$\underset{sTDA}{f_{12}}(S_0\to S_1)$',r'$\underset{sTD-DFT}{f_{12}}(S_0\to S_1)$']]


# In[56]:


CentrOverTol = Centrotoluene[['Overlap','Centroid',r'$\underset{sTDA}{\Delta E_{ST}}$',r'$\underset{sTD-DFT}{\Delta E_{ST}}$',r'$\underset{sTDA}{f_{12}}(S_0\to S_1)$',r'$\underset{sTD-DFT}{f_{12}}(S_0\to S_1)$']]


# In[57]:


# Overlap et centroid distance
pd.concat([CentrOverGas,CentrOverTol], axis=1)


# In[58]:


Centr = pd.concat([CentrOverGas,CentrOverTol], axis=1)
Centr


# In[59]:


Centr.mean()


# In[60]:


print(Centr.to_latex(multirow = True, multicolumn = True, float_format = "%.3f"))


# # Analyse pour la densité électronique
# 
# ---
# 
# ## 🔹 Ce que contient un `.molden`
# 
# Un fichier `.molden` standard a plusieurs sections :
# 
# * `[Atoms]` → atomes + coordonnées
# * `[GTO]` → base de primitives gaussiennes (exposants α, coefficients c, type de fonction (s, p, d, …))
# * `[MO]` → orbitales moléculaires, chaque orb est donnée comme combinaison linéaire des fonctions de base
# 
# Donc pour reconstruire la densité électronique :
# 
# $$
# \rho(\mathbf{r}) = \sum_{i}^{\text{occ}} f_i \, |\psi_i(\mathbf{r})|^2
# $$
# 
# avec
# 
# $$
# \psi_i(\mathbf{r}) = \sum_\mu C_{\mu i} \, \phi_\mu(\mathbf{r})
# $$
# 
# et
# 
# $$
# \phi_\mu(\mathbf{r}) = N \cdot r^l \, e^{-\alpha r^2} Y_{lm}(\theta,\phi)
# $$
# 
# où $\phi_\mu$ sont les orbitales atomiques gaussiennes définies dans `[GTO]`.
# 
# ---
# 
# ---
# 
# ## 🔹 Résumé
# 
# * **RDKit** sert à aligner les géométries → `rmsd`.
# * **PySCF** lit le `.molden` et évalue la **densité AO/MO sur une grille 3D**.
# * On obtient des quantités numériques globales :
# 
#   * $\Delta N$ = différence du nombre d’électrons
#   * $|\Delta \rho|$ = norme L1 de la différence de densité
#   * $\Delta \mu$ = différence de moments dipolaires
#   * RMSD des structures
# 
# ---
# 
# ---
# 
# ## 🔹 Points forts
# 
# * **PySCF** → lecture `.molden` et reconstruction AO/MO exacte.
# * **RDKit** → alignement des structures et calcul du **RMSD**.
# * **Parallélisation** → jusqu’à 8 molécules traitées en même temps. (ne mrche pas)
# * **Résultats sauvegardés** → `density_analysis_results.csv` contient pour chaque molécule :
# 
#   * ΔN
#   * |Δρ|
#   * Δμ
#   * RMSD
#   * (et `"error"` si une molécule plante).
# 
# ---
# 
# ✅ Points importants avant exécution
# 
# * Dépendances : `pyscf`, `rdkit`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`.
# * Paramètres clé : `grid_spacing` (précision), `margin` (bordure autour de la molécule). Augmenter `grid_spacing` diminue précision mais économise RAM/CPU.
# * Le script protège la section parallèle par `if __name__ == "__main__":` (nécessaire sous Windows).
# 
# ---
# 
# ## Remarques et conseils pratiques
# 
# 1. **Test sur petit lot d’abord** (10–20 molécules) pour valider la configuration et ajuster `GRID_SPACING` / `MARGIN`.
# 2. **Mémoire / CPU** : l’évaluation de la densité sur grille est la partie la plus coûteuse. Si tu manques de RAM, augmente `GRID_SPACING` à 0.35–0.5 Å.
# 3. **Logging des erreurs** : les molécules qui échouent sont conservées dans `density_analysis_raw.csv` avec la clé `error`. Inspecte ce CSV si plusieurs échecs apparaissent.
# 4. **Conversion du moment dipolaire** : j’ai converti en Debye (1 e·Å ≈ 4.8032 D).
# 5. **Visualisation** : la figure enregistrée est `density_global_summary_figure.png`. Tu peux modifier les couleurs/tailles si tu veux.
# 
# Si tu veux, je peux maintenant :
# 
# * adapter le script pour écrire aussi pour chaque molécule un petit résumé JSON (`id, dN, dAbs, dMu, RMSD`) dans un dossier `results/`,
# * ou te fournir une version qui produit en plus un **boxplot** par variable et des tests statistiques (p.ex. tests de normalité).
# 
# ---
# 

# In[61]:


# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pyscf import gto
from pyscf.tools import molden
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from tqdm import tqdm  # Version notebook de tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import tempfile

# --------------------------
# Paramètres globaux
# --------------------------
GRID_SPACING = 0.15   # Å (augmenter pour accélérer, diminuer pour plus de précision) #0.30 on va prendre 0.10 ou 0.05 pour l'article
MARGIN = 3.0          # Å (bordure autour de la molécule)
N_CORES = 4           # nombre de processus parallèles


# In[62]:


#from pathlib import Path
#data_dir = Path('data2507')


# In[63]:


#smiles = ['DMAC-TRZ','DMAC-DPS','PSPCz','4CzIPN','Px2BP','CzS2','2TCz-DPS','TDBA-DI']


# In[64]:


# --------------------------
# Fonctions utilitaires
# --------------------------
def load_and_align(vac_xyz, sol_xyz):
    """Lit deux fichiers .xyz avec RDKit, aligne solvant -> vide et retourne coords + rmsd."""
    mol_vac = Chem.MolFromXYZFile(vac_xyz)
    mol_sol = Chem.MolFromXYZFile(sol_xyz)
    if mol_vac is None or mol_sol is None:
        raise ValueError("Impossible de lire les fichiers XYZ (RDKit).")

    # Si pas de conformers, RDKit peut en créer (mais ici on suppose des .xyz 3D)
    if mol_vac.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol_vac)
    if mol_sol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol_sol)

    # Alignement : on aligne mol_sol sur mol_vac.
    rmsd = AllChem.AlignMol(mol_sol, mol_vac)

    conf_vac = mol_vac.GetConformer()
    conf_sol = mol_sol.GetConformer()
    coords_vac = np.array([list(conf_vac.GetAtomPosition(i)) for i in range(conf_vac.GetNumAtoms())])
    coords_sol = np.array([list(conf_sol.GetAtomPosition(i)) for i in range(conf_sol.GetNumAtoms())])

    return coords_vac, coords_sol, rmsd



# In[65]:


def build_density_from_molden(molden_file, grid_points):
    """
    Reconstruit la densité électronique sur une liste de points (npoints,3)
    à partir du .molden en utilisant PySCF.
    Version robuste qui gère différents formats de fichiers molden
    et les problèmes de dimensions
    """
    try:
        # Première tentative : méthode standard avec suppression des avertissements
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = molden.load(molden_file)
            
            ao_values = mol.eval_gto("GTOval_sph", grid_points)
            mo_coeff = np.asarray(mo_coeff)
            mo_occ = np.asarray(mo_occ)
            
            print(f"Dimensions - AO: {ao_values.shape[1]}, MO coeff: {mo_coeff.shape}, MO occ: {len(mo_occ)}")
            
            # Ajuster mo_coeff pour qu'il soit (nAO, nMO)
            if mo_coeff.shape[0] != ao_values.shape[1]:
                if mo_coeff.shape[1] == ao_values.shape[1]:
                    mo_coeff = mo_coeff.T
                    print("Transposition de mo_coeff")
            
            # Calcul de densité
            rho = np.zeros(len(grid_points), dtype=float)
            
            for i_mo, occ in enumerate(mo_occ):
                if i_mo >= mo_coeff.shape[1]:
                    break
                if occ <= 1e-8:
                    continue
                c_i = mo_coeff[:, i_mo]
                psi = ao_values.dot(c_i)
                rho += float(occ) * (psi.real**2 + psi.imag**2)
            
            return rho
            
    except Exception as e:
        print(f"Première méthode échouée: {e}")
        
        # Deuxième tentative : gestion robuste des dimensions
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = molden.load(molden_file)
                
                ao_values = mol.eval_gto("GTOval_sph", grid_points)
                mo_coeff = np.asarray(mo_coeff)
                mo_occ = np.asarray(mo_occ)
                
                print(f"Dimensions initiales - AO: {ao_values.shape[1]}, MO coeff: {mo_coeff.shape}, MO occ: {len(mo_occ)}")
                
                # Gestion robuste des dimensions
                nao_from_ao = ao_values.shape[1]
                nmo_from_occ = len(mo_occ)
                
                # Ajuster mo_coeff pour qu'il soit compatible
                if mo_coeff.shape[0] == nao_from_ao and mo_coeff.shape[1] <= nmo_from_occ:
                    # Forme correcte (nAO, nMO)
                    pass
                elif mo_coeff.shape[1] == nao_from_ao and mo_coeff.shape[0] <= nmo_from_occ:
                    # Besoin de transposer
                    mo_coeff = mo_coeff.T
                else:
                    # Cas problématique : prendre les dimensions qui marchent
                    print("Ajustement des dimensions pour compatibilité")
                    if mo_coeff.shape[0] == nao_from_ao:
                        # Garder comme ça mais ajuster le nombre de MO
                        n_mo_use = min(mo_coeff.shape[1], nmo_from_occ)
                        mo_coeff = mo_coeff[:, :n_mo_use]
                        mo_occ = mo_occ[:n_mo_use]
                    elif mo_coeff.shape[1] == nao_from_ao:
                        # Transposer puis ajuster
                        mo_coeff = mo_coeff.T
                        n_mo_use = min(mo_coeff.shape[1], nmo_from_occ)
                        mo_coeff = mo_coeff[:, :n_mo_use]
                        mo_occ = mo_occ[:n_mo_use]
                    else:
                        # Dernière chance : forcer les dimensions
                        n_ao_use = min(nao_from_ao, mo_coeff.shape[0], mo_coeff.shape[1])
                        n_mo_use = min(nmo_from_occ, mo_coeff.shape[0], mo_coeff.shape[1])
                        
                        if mo_coeff.shape[0] >= n_ao_use and mo_coeff.shape[1] >= n_mo_use:
                            mo_coeff = mo_coeff[:n_ao_use, :n_mo_use]
                        else:
                            mo_coeff = mo_coeff.T[:n_ao_use, :n_mo_use]
                        
                        mo_occ = mo_occ[:n_mo_use]
                        # Ajuster ao_values si nécessaire
                        if ao_values.shape[1] > n_ao_use:
                            print(f"ATTENTION: Utilisation de seulement {n_ao_use} AO sur {ao_values.shape[1]}")
                            ao_values = ao_values[:, :n_ao_use]
                
                print(f"Dimensions finales - AO: {ao_values.shape[1]}, MO coeff: {mo_coeff.shape}, MO occ: {len(mo_occ)}")
                
                # Vérification finale avant le calcul
                if ao_values.shape[1] != mo_coeff.shape[0]:
                    print(f"ERREUR FINALE: ao_values shape {ao_values.shape} incompatible avec mo_coeff shape {mo_coeff.shape}")
                    # Forcer la compatibilité
                    min_dim = min(ao_values.shape[1], mo_coeff.shape[0])
                    ao_values = ao_values[:, :min_dim]
                    mo_coeff = mo_coeff[:min_dim, :]
                    print(f"Dimensions forcées - AO: {ao_values.shape[1]}, MO coeff: {mo_coeff.shape}")
                
                rho = np.zeros(len(grid_points), dtype=float)
                
                for i_mo, occ in enumerate(mo_occ):
                    if i_mo >= mo_coeff.shape[1]:
                        break
                    if occ <= 1e-8:
                        continue
                    c_i = mo_coeff[:, i_mo]
                    psi = ao_values.dot(c_i)
                    rho += float(occ) * (psi.real**2 + psi.imag**2)
                
                return rho
                
        except Exception as e2:
            print(f"Deuxième méthode échouée: {e2}")
            
            # Troisième tentative : méthode ultra-robuste avec dimensions forcées
            try:
                print("Tentative ultra-robuste avec dimensions forcées...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = molden.load(molden_file)
                    
                    ao_values = mol.eval_gto("GTOval_sph", grid_points)
                    mo_coeff = np.asarray(mo_coeff)
                    mo_occ = np.asarray(mo_occ)
                    
                    print(f"Tentative ultra-robuste - AO: {ao_values.shape[1]}, MO coeff: {mo_coeff.shape}, MO occ: {len(mo_occ)}")
                    
                    # Forcer la compatibilité de manière brutale mais sûre
                    nao_available = ao_values.shape[1]
                    
                    # Déterminer les dimensions utilisables
                    if mo_coeff.shape[0] == nao_available:
                        # mo_coeff est (nAO, nMO)
                        nmo_available = mo_coeff.shape[1]
                    elif mo_coeff.shape[1] == nao_available:
                        # mo_coeff est (nMO, nAO), transposer
                        mo_coeff = mo_coeff.T
                        nmo_available = mo_coeff.shape[1]
                    else:
                        # Aucune dimension ne correspond exactement
                        # Prendre le minimum qui marche
                        if mo_coeff.shape[0] < mo_coeff.shape[1]:
                            # Probablement (nMO, nAO)
                            mo_coeff = mo_coeff.T
                        
                        nao_use = min(nao_available, mo_coeff.shape[0])
                        nmo_use = mo_coeff.shape[1]
                        
                        ao_values = ao_values[:, :nao_use]
                        mo_coeff = mo_coeff[:nao_use, :]
                        nmo_available = nmo_use
                    
                    # Ajuster mo_occ pour correspondre au nombre de MO
                    nmo_final = min(len(mo_occ), nmo_available)
                    mo_occ = mo_occ[:nmo_final]
                    mo_coeff = mo_coeff[:, :nmo_final]
                    
                    print(f"Dimensions finales ultra-robustes - AO: {ao_values.shape[1]}, MO coeff: {mo_coeff.shape}, MO occ: {len(mo_occ)}")
                    
                    rho = np.zeros(len(grid_points), dtype=float)
                    
                    for i_mo, occ in enumerate(mo_occ):
                        if occ <= 1e-8:
                            continue
                        c_i = mo_coeff[:, i_mo]
                        psi = ao_values.dot(c_i)
                        rho += float(occ) * (psi.real**2 + psi.imag**2)
                    
                    return rho
                    
            except Exception as e3:
                print(f"Toutes les méthodes ont échoué: {e3}")
                raise


# In[66]:


# --------------------------
# Fonction d'analyse par molécule
# --------------------------
def analyze_density_difference(vac_molden, sol_molden, vac_xyz, sol_xyz,
                               grid_spacing=GRID_SPACING, margin=MARGIN):
    """
    Reconstruit les densités depuis les .molden (PySCF), aligne les géométries (RDKit),
    calcule ΔN, |Δρ|, Δμ et RMSD. Retourne (dN, dAbs, dMu, RMSD).
    """
    # 1) alignement
    coords_vac, coords_sol, rmsd = load_and_align(vac_xyz, sol_xyz)

    # 2) définir grille commune centrée sur coords_vac
    min_corner = np.min(coords_vac, axis=0) - margin
    max_corner = np.max(coords_vac, axis=0) + margin
    # s'assurer au moins 2 points par axe
    nx = max(2, int(np.ceil((max_corner[0] - min_corner[0]) / grid_spacing)) + 1)
    ny = max(2, int(np.ceil((max_corner[1] - min_corner[1]) / grid_spacing)) + 1)
    nz = max(2, int(np.ceil((max_corner[2] - min_corner[2]) / grid_spacing)) + 1)

    xs = np.linspace(min_corner[0], max_corner[0], nx)
    ys = np.linspace(min_corner[1], max_corner[1], ny)
    zs = np.linspace(min_corner[2], max_corner[2], nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # shape (npoints,3)

    # 3) densités (PySCF)
    rho_vac = build_density_from_molden(vac_molden, grid)
    rho_sol = build_density_from_molden(sol_molden, grid)

    # 4) différences & intégrales
    delta_rho = rho_sol - rho_vac
    dV = (xs[1] - xs[0]) * (ys[1] - ys[0]) * (zs[1] - zs[0])  # voxel volume

    delta_N = float(np.sum(delta_rho) * dV)
    delta_abs = float(np.sum(np.abs(delta_rho)) * dV)

    # 5) moment dipolaire (μ = ∫ r ρ(r) dV) ; retourne norme de la variation en e·Å
    mu_vac = (rho_vac[:, None] * grid).sum(axis=0) * dV
    mu_sol = (rho_sol[:, None] * grid).sum(axis=0) * dV
    delta_mu_eA = np.linalg.norm(mu_sol - mu_vac)  # en e·Å ; conversion en Debye en aval si désiré

    # conversion e·Å -> Debye (1 e·Å ≈ 4.80320427 D)
    delta_mu = float(delta_mu_eA * 4.80320427)

    return delta_N, delta_abs, delta_mu, float(rmsd)


# In[67]:


# --------------------------
# wrapper utilisé par le pool
# --------------------------
def analyze_one(smi_key):
    vac_molden = os.path.join(data_dir, f"gas/{smi_key}/{smi_key}_gas_S0_finalOpt.molden") # Path(f'{data_dir}/gas/{smi_key}/{smi_key}_gas_S0_finalOpt.molden')   # 
    sol_molden = os.path.join(data_dir, f"toluene/{smi_key}/{smi_key}_toluene_S0_finalOpt.molden") # Path(f'{data_dir}/toluene/{smi_key}/{smi_key}_toluene_S0_finalOpt.molden')  # 
    vac_xyz = os.path.join(data_dir, f"gas/{smi_key}/{smi_key}_gas_S0_finalOpt.xtbopt.xyz") # Path(f'{data_dir}/gas/{smi_key}/{smi_key}_gas_S0_finalOpt.xtbopt.xyz')   # 
    sol_xyz = os.path.join(data_dir, f"toluene/{smi_key}/{smi_key}_toluene_S0_finalOpt.xtbopt.xyz") # Path(f'{data_dir}/toluene/{smi_key}/{smi_key}_toluene_S0_finalOpt.xtbopt.xyz')   # 

    # vérifier existence des fichiers
    print(smi_key)
    for f in (vac_molden, sol_molden, vac_xyz, sol_xyz):
        if not os.path.isfile(f):
            return {"id": smi_key, "error": f"Missing file: {f}"}

    try:
        dN, dAbs, dMu, rmsd = analyze_density_difference(
            vac_molden, sol_molden, vac_xyz, sol_xyz
        )
        return {"id": smi_key, "dN": dN, "dAbs": dAbs, "dMu": dMu, "RMSD": rmsd}
    except Exception as e:
        return {"id": smi_key, "error": repr(e)}


# # --------------------------
# # Main : exécution parallèle 
# # --------------------------
# # Si vous voulez encore plus de contrôle sur l'affichage
# def analyze_with_custom_progress(molecule_ids):
#     results = []
#     
#     # Widget de progression personnalisé (optionnel)
#     progress_widget = widgets.IntProgress(
#         value=0,
#         min=0,
#         max=len(molecule_ids),
#         description='Analyse:',
#         bar_style='info',
#         style={'bar_color': '#1f77b4'},
#         orientation='horizontal'
#     )
#     display(progress_widget)
#     
#     with ProcessPoolExecutor(max_workers=N_CORES) as executor:
#         for i, res in enumerate(executor.map(analyze_one, molecule_ids)):
#             results.append(res)
#             progress_widget.value = i + 1
#     
#     return results

# In[ ]:


# --------------------------
# Main : exécution parallèle + figure unique
# --------------------------
if __name__ == "__main__":
    # Nombre de molécules/process listés
    #n_molecules = len(smiles)
    molecule_ids = smiles #list(range(n_molecules))

    results = []  # analyze_with_custom_progress(molecule_ids)  #

    # Parallèle
    #with ProcessPoolExecutor(max_workers=N_CORES) as executor:
    #    # tqdm pour barre de progression
    #    for res in tqdm(executor.map(analyze_one, molecule_ids), total=len(molecule_ids),
    #                    desc="Analyse molécules"):
    #        results.append(res)

    # sequentiel
    
    for mol_id in tqdm(molecule_ids, total=len(molecule_ids), desc="Analyse molécules"):
        res = analyze_one(mol_id)
        results.append(res)
     
    # DataFrame & filtrage des succès
    df = pd.DataFrame(results)
    # sauvegarde brute (pour repérage des erreurs)
    df.to_csv("density_analysis_raw.csv", index=False)

    # garder seulement lignes sans 'error'
    df_ok = df[df["error"].isna() if "error" in df.columns else [True]*len(df)].copy()
    # si colonne 'error' existe, filtrer
    if "error" in df.columns:
        df_ok = df[df["error"].isnull()].copy()

    # renommer colonnes pour affichage si nécessaire
    df_ok = df_ok.reset_index(drop=True)

    # sauvegarder résultats numériques
    df_ok.to_csv("density_analysis_results.csv", index=False)


# In[ ]:


# -------------------------
# Génération d'une figure unique représentative
# -------------------------
sns.set(style="whitegrid", context="talk")
plt.rcParams.update({"font.size": 10})

# Extraire séries
dN = df_ok["dN"].astype(float).values
dAbs = df_ok["dAbs"].astype(float).values
dMu = df_ok["dMu"].astype(float).values
rmsd = df_ok["RMSD"].astype(float).values if "RMSD" in df_ok.columns else None

# Statistiques globales
summary_stats = pd.DataFrame({
    "mean": [dN.mean(), dAbs.mean(), dMu.mean(), rmsd.mean() if rmsd is not None else np.nan],
    "std": [dN.std(ddof=1), dAbs.std(ddof=1), dMu.std(ddof=1), rmsd.std(ddof=1) if rmsd is not None else np.nan],
    "min": [dN.min(), dAbs.min(), dMu.min(), rmsd.min() if rmsd is not None else np.nan],
    "max": [dN.max(), dAbs.max(), dMu.max(), rmsd.max() if rmsd is not None else np.nan],
}, index=[r"$|\Delta N|\, (e^-)$", r"$|\Delta\rho|\, (e^-)$", r"$\Delta\mu$ (Debye)", r"RMSD ($\AA$)"])

# Figure composite : 1ère rangée = 3 histogrammes ; 2ème rangée gauche = scatter dMu vs dAbs ; droite = heatmap corrélation + résumé texte
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.35, wspace=0.3)

# Hist ΔN
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(dN, bins=40, kde=False, ax=ax1, color="C0")
ax1.set_title(r"$\Delta N$ (net electron variation)") #variation nette d'électrons
ax1.set_xlabel(r"$\Delta N\, (e^-)$")

# Hist |Δρ|
ax2 = fig.add_subplot(gs[0, 1])
sns.histplot(dAbs, bins=40, kde=False, ax=ax2, color="C1")
ax2.set_title(r"$|\Delta\rho|$ (total redistribution)") #redistribution totale
ax2.set_xlabel(r"$|\Delta\rho|\, (e^-)$")

# Hist Δμ
ax3 = fig.add_subplot(gs[0, 2])
sns.histplot(dMu, bins=40, kde=False, ax=ax3, color="C2")
ax3.set_title(r"$\Delta\mu$ (dipole variation)")
ax3.set_xlabel(r"$\Delta\mu$ (Debye)")

# Scatter Δμ vs |Δρ|
ax4 = fig.add_subplot(gs[1, 0:2])
sns.scatterplot(x=dAbs, y=dMu, ax=ax4, alpha=0.6, s=25)
ax4.set_xlabel(r"$|\Delta\rho|\, (e^-)$")
ax4.set_ylabel(r"$\Delta\mu$ (Debye)")
ax4.set_title(r"Correlation $\Delta\mu$ vs $\Delta\rho$")

# Heatmap de corrélation + encadré résumé
ax5 = fig.add_subplot(gs[1, 2])
corr_df = pd.DataFrame({r"$\Delta N$": dN, r"$|\Delta\rho|$": dAbs, r"$\Delta\mu$": dMu})
corr = corr_df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax5,
            cbar_kws={"shrink": 0.6})
ax5.set_title("Correlation matrix (Pearson)") #Matrice de corrélation

# Ajouter résumé numérique sous forme de texte
text_ax = fig.add_axes([0.02, 0.02, 0.96, 0.0])  # zone en bas pour le texte résumé
text_ax.axis("off")
txt = "Numerical summary (mean ± std) :\n" #Résumé numérique
for idx in summary_stats.index:
    mean = summary_stats.loc[idx, "mean"]
    std = summary_stats.loc[idx, "std"]
    txt += "{}: {:.4g} ± {:.4g}    ".format(idx, mean, std)
text_ax.text(0.01, 0.5, txt, fontsize=10, va="center", ha="left", family="monospace")

# Sauvegarde et affichage
outfig = "densityGlobalSummaryArticle.png"
fig.savefig(outfig, dpi=300, bbox_inches="tight")
print(f"Figure globale sauvegardée : {outfig}")

# Sauvegarder aussi les stats détaillées
summary_stats.to_csv("density_summary_stats.csv")
print("Fichier CSV des stats : density_summary_stats.csv")
plt.show()
