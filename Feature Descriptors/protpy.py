#install
!pip install protpy --upgrade
!pip install numpy

#import
import pandas as pd
import numpy as np
import protpy as protpy

#Amino Acid Composition Coding (AAC)
AAC = []

for seq in fasta:
  AAC.append(np.array(protpy.amino_acid_composition(seq))[0])

pd.DataFrame(AAC).to_csv('AAC.csv', index=False, header=False)

#Tripeptide Composition (TPC)
TPC = []

for seq in fasta:
  TPC.append(np.array(protpy.tripeptide_composition(seq))[0])

pd.DataFrame(TPC).to_csv('TPC.csv', index=False, header=False)

#Dipeptide Composition (DPC)
DPC = []

for seq in fasta:
  DPC.append(np.array(protpy.dipeptide_composition(seq))[0])

pd.DataFrame(DPC).to_csv('DPC.csv', index=False, header=False)

#Composition/Transition/Distribution (CTD)
CTD = []

for seq in fasta:
  CTD.append(np.array(protpy.ctd_(seq))[0])

pd.DataFrame(CTD).to_csv('CTD.csv', index=False, header=False)

#Conjugated Triad (Ctriad)
CTriad = []

for seq in fasta:
  CTriad.append(np.array(protpy.conjoint_triad(seq))[0])

pd.DataFrame(CTriad).to_csv('CTriad.csv', index=False, header=False)

#Pseudo-Amino Acid Composition (PAAC)
PAAC = []

for seq in fasta:
  #lamda = 1 pelo existência de um dipeptideo na base
  PAAC.append(np.array(protpy.pseudo_amino_acid_composition(seq, lamda=1))[0])

pd.DataFrame(PAAC).to_csv('PAAC.csv', index=False, header=False)

#Amphiphilic Pseudo-Amino Acid Composition (APAAC)
APAAC = []

for seq in fasta:
  APAAC.append(np.array(protpy.amphiphilic_pseudo_amino_acid_composition(seq, lamda=1))[0])

pd.DataFrame(APAAC).to_csv('APAAC.csv', index=False, header=False)
