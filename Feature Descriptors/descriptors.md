# Feature Descriptors


---



The feature descriptors examined in this study were implemented using software packages in the Python 3 programming language, including ModlAMP, DeepChem, and Protpy.

***ModlAMP***

The ModlAMP package provides a range of tools for conducting in silico research to aid in the discovery and design of new synthetic AMPs. The library offers functions for computing various physicochemical properties of molecules, as well as peptide descriptors based on amino acid residues. Specifically, in this study, we utilized descriptors from the PeptideDescriptor class, which are detailed in Table 2\. A combination of all descriptors was utilized in this research.

**Table 2** \- Physicochemical descriptors derived from the modlAMP package  

| Descriptors | Descriptor overview |
| :---: | :---: |
| **AASI** | Amino Acid Selectivity Index |
| **ABHPRK** | Scale of internal physicochemical characteristics of modlabs |
| **Argos** | Argos hydrophobicity amino acid scale  |
| **Bulkiness** | Amino acid side chain volume scale  |
| **Charge\_phys** | Amino acid charge at ph 7.0 – Histidine \+0.1. |
| **Charge\_acid** | Amino acid charge at acidic ph – Histidine charge \+1.0 |
| **Cougar** | Internal selection of global peptide descriptors by modlabs |
| **Eisenberg** | Eisenberg consensus scale |
| **Ez** | Lipid bilayer insertion energy |
| **Flexibility** | Lateral chain flexibility |
| **Grantham** | Composition of amino acid side chains, polarity, and molecular volume  |
| **Gravy** | Amino acid hydrophobicity scale GRAVY |
| **Hopp-woods** | Hopp-Woods amino acid hydrophobicity scale |
| **ISAECI** | Isotropic surface area – electron charge index |
| **Janin** | Hydrophobicity  |
| **Kytedoolittle** | Hydrophobicity |
| **Levitt\_alpha** | α-helical propensity |
| **MSS** | Topological shape and size of the side chain |
| **MSW** | Key components of steric and 3D residue properties |
| **PepArc** | Pharmacophoric features of modlabs: hydrophobicity, polarity, positive charge, negative charge, proline |
| **Pepcats** | Binary pharmacophoric characteristics |
| **Polarity** | Amino acid polarity |
| **PPCALI** | Main components of selected side chain properties |
| **Refractivity** | Relative refractive index values |

The author (2023)

***DeepChem***

Developed by researchers at Stanford University, the DeepChem package provides a variety of functionalities for developing new drugs and conducting molecular analyses in general (Altae-Tran et al., 2017). The package offers various descriptors that enable a detailed analysis of the molecule's structure. Additionally, it includes different physicochemical feature descriptors and descriptors describing the molecule in 2D (Ramsundar, 2022). The descriptors utilized in this study from DeepChem can be found in Table 3\.

**Table 3** \- Descriptors derived from DeepChem

| Descriptors | Descriptor overview |
| :---: | ----- |
| **Circular *Fingerprint* (Circular)** | This strategy, also known as Extended-connectivity fingerprints (ECFPs), consists of a molecular representation method in which molecules are divided into unique and separate circular fragments. |
| **Fasta2Seq** | This descriptor converts a peptide sequence in FASTA format into a numerical vector. |
| **MACCSKeys Fingerprint**  | This descriptor is based on the identification of key substructures, with 166 predefined keys. These keys indicate the presence of functional groups. |
| **Mol2Vec Fingerprint (mol2vec)** | It refers to a strategy based on natural language processing (NLP) techniques. This method considers predefined compound substructures derived from the Morgan algorithm as 'words' and compounds as 'sentences', transforming the compound into a numerical vector. |
| **Smiles2Seq** | Similar to Fasta2Seq, this performs the transformation of a sequence in Smiles format into a numerical vector. |

The author (2023)

***Protpy***

The Protpy package, which is available for Python 3, provides physicochemical and structural descriptors that allow for the representation of amino acid composition and organization in peptide sequences. The descriptors utilized in this study can be seen in Table 4\.

**Table 4** \- Physicochemical descriptors derived from Protpy

| Descriptors | Descriptor overview |
| ----- | ----- |
| **Amino Acid Composition Coding (AAC)** | Calculates the frequency of each of the 20 amino acids in the sequence. |
| **Tripeptide Composition (TPC)** | Calculates the frequency of tripeptides in the sequence. |
| **Dipeptide Composition (DPC)** | Calculates the frequency of dipeptides in the sequence |
| **Composition/Transition/Distribution (CTD)** | Calculates the distribution frequency of amino acids based on structural properties and physicochemical characteristics |
| **Conjugated Triad (Ctriad)** | Calculates the physicochemical property of an amino acid triad. |
| **Pseudo-Amino Acid Composition (PAAC)** | Calculates the correlation between amino acids considering physicochemical properties such as hydrophobicity value |
| **Amphiphilic Pseudo-Amino Acid Composition (APAAC)** | Calculate the correlation between amino acids considering amphiphilic amino acids. |

The author (2023)
