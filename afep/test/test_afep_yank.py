"""
test_yank_afep.py

"""
import os
import sys
sys.path.append("..")


from yank_afep_analyzer import YankAtomEnergyAnalyzer
print("import successful")
analyzer = YankAtomEnergyAnalyzer(nc_path=os.path.abspath('../data/phenol/solvent1.nc'),
        nc_checkpoint_path=os.path.abspath('../data/phenol/solvent1_checkpoint.nc'))
print("initialize successful")

print(analyzer.read_positions(0))
