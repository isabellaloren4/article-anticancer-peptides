#Install
!pip install --pre deepchem
!pip install rdkit-pypi
!pip install pytoda
!pip install pubchempy

#Import
import deepchem as dc
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

#-----1D-MACCSKeysFingerprint

featurizer = dc.feat.MACCSKeysFingerprint()
d0 = featurizer.featurize(smiles)
pd.DataFrame(d0).to_csv('maccskeys.csv', index=False, header=False)

#-----1D-CircularFingerprint

featurizer = dc.feat.CircularFingerprint(size=2048, radius=4)
d1 = featurizer.featurize(smiles)
pd.DataFrame(d1).to_csv('circular.csv', index=False, header=False)

#-----1D-Mol2VecFingerprint

featurizer = dc.feat.Mol2VecFingerprint()
d2 = featurizer.featurize(smiles)
pd.DataFrame(d2).to_csv('mol2vec.csv', index=False, header=False)

#-----1D-SmilesToSeq

symbols = ['<unk>','<pad>','#', ')', '(', '+', '-', '/', '1', '3', '2', '5', '4', '7', '6', '8', '=', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'S', '[', ']', '\\', 'c', 'l', 'o', 'n', 'p', 's', 'r']
numbers = list(range(len(symbols)))
d = zip(symbols,numbers)
dic = dict(d)

featurizer = dc.feat.SmilesToSeq(dic,  max_len = 1011)
d9 = featurizer.featurize(smiles)
pd.DataFrame(d9).to_csv('features/smilestoseq.csv', index=False, header=False)

#-----1D-FastaToSeq

symbols = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W','Y']
numbers = list(range(len(symbols)))
d = zip(symbols,numbers)
dic = dict(d)

seqs = []

for s in fasta:

    seq = [-1] * 50
    for i in range(len(s)):
        seq[i] = dic[s[i]]

    seqs.append(seq)

pd.DataFrame(np.array(seqs)).to_csv('features/fastatoseq.csv', index=False, header=False)
