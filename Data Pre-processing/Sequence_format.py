#Install
!pip install --pre deepchem
!pip install rdkit-pypi
!pip install pytoda
!pip install pubchempy

#Import
import pandas as pd
import numpy as np
from pytoda.proteins import aas_to_smiles
from rdkit.Chem import AllChem
from rdkit import Chem

#Reading txt files containing the sequences
facp = open("acp.txt",'r')
fnacp = open("nacp.txt", 'r')

acp = facp.readlines()
nacp = fnacp.readlines()

facp.close()
fnacp.close()

#transforming sequences into formats (MOLS, Smiles and FASTA)
smiles = []
mols = []
y = [] #classes, 1 for anticancer peptides and 0 for non-anticancer peptides
for i in range(len(acp)):

    mol = Chem.rdmolfiles.MolFromFASTA(nacp[i][:-1], sanitize=True)#mols
    mols.append(Chem.Mol(mol))#mols
    mols.append(mol)#mols


    smiles.append(aas_to_smiles(nacp[i][:-1]))#Smiles

    y.append(1)

for i in range(len(nacp)):

    mol = Chem.rdmolfiles.MolFromFASTA(nacp[i][:-1], sanitize=True)#mols
    mols.append(Chem.Mol(mol))#mols
    mols.append(mol)#mols


    smiles.append(aas_to_smiles(nacp[i][:-1]))#Smiles

    y.append(0)
