
# coding: utf-8

# In[25]:


"""
test_yank_afep.py

"""
import os
import sys
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle

sys.path.append("..")
# /Users/yuanqingwang/Documents/GitHub/afep/afep/data/abl-imatinib/explicit/experiments/complex.nc

from yank_afep_analyzer import YankAtomEnergyAnalyzer
print("import successful")
analyzer = YankAtomEnergyAnalyzer(nc_path=os.path.abspath('../data/abl-imatinib/explicit/experiments/complex.nc'),
        nc_checkpoint_path=os.path.abspath('../data/abl-imatinib/explicit/experiments/complex_checkpoint.nc'))
print("initialize successful")


# In[ ]:


analyzer._analyze_atom_energies_with_all_checkpoint_positions()


# In[ ]:


lambda_shedule = analyzer.checkpoint_lambda_electrostatics_schedule
print(len(lambda_shedule))


# In[ ]:


ligand_atoms = topography.solute_atoms
print(ligand_atoms)
topology = topography.topology
atoms = list(topology.atoms)
tags = []
for atom in ligand_atoms:
    print(atom, atoms[atom])
    tags.append(str(atoms[atom]))


# In[ ]:


analyzer._average_free_energy_weights_by_state(1,18)


# In[ ]:


energies_diff_matrix = []
for idx in range(len(lambda_schedule)):
    energies_diff = analyzer._average_free_energy_weights_by_state(idx + 1, idx + 2)
    energies_diff_matrix.append(energies_diff)
energies_diff_matrix = np.array(energies_diff_matrix)
pickle.dump(energies_diff_matrix, open('m.p', 'wb'))
print(energies_diff_matrix)
print(np.sum(energies_diff_matrix, axis=1))


# In[ ]:


plt.clf()
for idx in range(len(ligand_atoms)):
    tag = tags[idx]
    x = np.array(range(17))
    y = energies_diff_matrix[:, idx]
    plt.plot(x, y, label=tag)
plt.legend()
plt.show()
