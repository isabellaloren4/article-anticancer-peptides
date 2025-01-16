#Install
!pip install modlamp
!pip install numpy

#Import
import pandas as pd
import numpy as np
from modlamp.descriptors import PeptideDescriptor

#Peptide descriptor

descr_names = ['AASI', 'ABHPRK', 'argos', 'bulkiness', 'charge_phys', 'charge_acid', 'cougar',
               'eisenberg', 'Ez', 'flexibility', 'grantham', 'gravy', 'hopp-woods', 'ISAECI',
               'janin', 'kytedoolittle', 'levitt_alpha', 'MSS', 'MSW', 'pepArc', 'pepcats',
               'polarity', 'PPCALI', 'refractivity', 't_scale', 'TM_tend', 'z3', 'z5']

D = []

for descr in descr_names:

    #calculation of descriptors
    pepdesc = PeptideDescriptor(fasta, scalename=descr)
    pepdesc.calculate_global()
    dc = pepdesc.descriptor

    #converter para numpy array corretamente
    dc_numpy = np.zeros((len(dc), len(dc[0])))
    for i in range (len(dc)):
        dc_numpy[i] = np.array(dc[i])

    D.append(dc_numpy)

#Concatenation of all descriptors
M = np.concatenate((D[0], D[1]), axis=1)

for i in range(2,len(D)):
    M = np.concatenate((M, D[i]), axis=1)

#save to csv file
pd.DataFrame(M).to_csv('features/modlamp.csv', index=False, header=False)
