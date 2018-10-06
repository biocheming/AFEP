#!/usr/bin/python

"""
yank_afep_reporter.py

MIT License

Copyright (c) 2018

Weill Cornell Medicine, Memorial Sloan Kettering Cancer Center, and Authors

Authors:
Yuanqing Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from openmmtools import testsystems
import mdtraj as md

import netCDF4
from netCDF4 import Dataset
import warnings
import time
import copy

from yank import analyze


#=========================================================================
# herlper functions to calculate distance, angles,
# and dihedral angels from positions of atoms

# NOTE: we use these functions to calculate all the bonded energies
#=========================================================================



def dist(self, atom0, atom1):
    """
    calculate the distance between two atoms
    require that self.pos is defined

    parameters
    ----------
    atom0 : the idx of the first atom
    atom1 : the idx of the second atom

    returns
    -------
    dist : a float representing the distance between the two atoms
    """

    pos0 = self.pos[atom0]
    pos1 = self.pos[atom1]

    dist = np.linalg.norm(pos0 - pos1)

    return dist

def angle(self, center_atom, atom0, atom1):
    """
    calculate the angle between bond:

    center_atom -- atom0
    and
    center_atom -- atom1

    $ cos(<v0, v1>) = (v0 \dot v1) / |v0||v1| $

    parameters
    ----------
    center_atom : the idx of the center atom
    atom0 : the idx of the first atom involved in the angle
    atom1 : the idx of the second atom involved in the angle

    returns
    -------
    angle : the value of the angle in rads
    """

    # get all the positions
    pos_center = self.pos[center_atom]
    pos0 = self.pos[atom0]
    pos1 = self.pos[atom1]

    # express the distance in vectors
    v0 = np.array(pos0) - np.array(pos_center)
    v1 = np.array(pos1) - np.array(pos_center)

    # to calculate:
    # $ cos(<v0, v1>) = (v0 \dot v1) / |v0||v1| $
    v0_dot_v1 = np.dot(v0, v1)
    v0_norm = np.linalg.norm(v0)
    v1_norm = np.linalg.norm(v1)

    angle = np.arccos(np.true_divide(v0_dot_v1, v0_norm * v1_norm))

    return Quantity(angle, radian)


def dihedral(self, atom0, atom1, atom2, atom3):
    """
    calculate the dihedral between the plane formed by:

    atom0, atom1, and atom2
    and that by
    atom1, atom2, and atom3

    $$
    n_A = q_0 \cross q_1 \\
    n_B = q_1 \cross q_2 \\

    \Phi = |n_A \dot n_B| / |n_A||n_B|
    $$

    parameters
    ----------
    atom0 : the idx of the first atom involved in the torsion
    atom1 : the idx of the second atom involved in the torsion
    atom2 : the idx of the thrid atom involved in the torsion
    atom3 : the idx of the fourth atom involved in the torsion

    returns
    -------
    angle : the value of the dihedral angle in rads
    """

    # get the positions of the atoms
    pos0 = self.pos[atom0]
    pos1 = self.pos[atom1]
    pos2 = self.pos[atom2]
    pos3 = self.pos[atom3]

    # calculate the vectors between these atoms
    q1 = pos1 - pos0
    q2 = pos2 - pos1
    q3 = pos3 - pos2

    # calculate the normal vectors
    na = np.cross(q1, q2)
    nb = np.cross(q2, q3)

    # calculate the dihedral angel
    na_dot_nb = np.dot(na, nb)
    na_norm = np.linalg.norm(na)
    nb_norm = np.linalg.norm(nb)
    angle = np.arccos(np.true_divide(np.absolute(na_dot_nb), na_norm * nb_norm))

    return Quantity(angle, radian)


#=========================================================================
# helper functions to analyze bonded forces
#=========================================================================



class YankAtomEnergyAnalyzer:
    """
    AtomEnergyReporter outputs information about every atom in a simulation,
    including the energy breakdown, to a file.

    to use it, create an AtomEnergyReporter, then add it to the list of reporters
    of the list of the Simulation. by default the data is written in CSV format.

    this module is written in order to implement the algorithm developed by
    Benedict W.J.Irwin and David J. Huggins in University of Cambridge.

    this calculates the Eq.11 in the paper:

    $$

    u_{X_a} = \\

    \frac12(u_{electrostaic} + u_{Lennard-Jones} + u_{bonded} + u_{Urey-Bradley}) \\
    + \frac13 u_{angle} \\
    + \frac14 u_{dihedral} + u_{improper}

    $$

    further data analysis is needed

    ref:
    https://pubs.acs.org/doi/abs/10.1021/acs.jctc.8b00027

    """

    def __init__(self, nc_path):
        self.ds = Dataset(nc_path, 'r')
        self.reference_system = None
        self.puppet_system = None

    def get_reference_system(self):
        """
        get the OpenMM System object for a Yank run

        this enable us to extract information regarding the small molecule, and
        parameters thereof

        """

        reporter = multistate.MultiStateReporter(self.nc_path, open_mode='r')
        metadata = reporter.read_dict('metadata')
        reference_system = mmtools.utils.deserialize(metadata['reference_state']).system
        self.reference_system = reference_system
        return reference_system

    def get_puppet_system(self):
        """
        generate a puppet system to play with

        """
        if self.puppet_system != None:
            return self.puppet_system

        if self.reference_system == None:
            self.get_reference_system()

        puppet_system = copy.deepcopy(self.reference_system)
        self.puppet_system = puppte_system
        return puppet_system


    #=========================================================================
    # helper functions to analyze forces
    #=========================================================================

    def analyze_nonbonded_force(self, force=None):
        # NOTE: we still need a force=None to work with the get_energy function
        # but it's redundant
        """
        analyze the nonbonded force in a simulation, despite of its kind.

        note that this method is different from others because it,
        instead of directly calculate the energy analytically,
        make a copy of the simulation and turn off the params of an atom
        to calculate the total nonbonded energy with the involvement of thereof.

        so the argument force is actually ignored.

        parameters
        ----------
        force : an OpenMM force object, has to be NonbondedForce

        returns
        -------
        energy_dict : a dictionary mapping atom idxs to the energies
        """


        energy_dict = dict() # initialize the dict
        current_energy = self.state.getPotentialEnergy() # get current energy

        # NOTE: currently this loop is VERY slow
        # TODO: try to write this without a loop
        for idx in self.idxs: # loop through the interested atoms
            for force in self.simulation.system.getForces():
                if force.__class__.__name__ == 'NonbondedForce':
                    charge, sigma, epsilon = force.getParticleParameters(idx)
                    force.setParticleParameters(idx, charge = 0.0, sigma = sigma, epsilon = 0.0)
                    force.updateParametersInContext(self.simulation.context)
                    new_energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
                    energy_diff = new_energy - current_energy
                    energy_dict[idx] = energy_diff

                    force.setParticleParameters(idx, charge = charge, sigma = sigma, epsilon = epsilon)
                    force.updateParametersInContext(self.simulation.context) # set the parameters back to original

        return energy_dict


    def analyze_amoeba_angle_force(force):

        pass

    def analyze_amoeba_bond_force(force):
        pass

    def analyze_amoeba_generalized_kirkwood_force(force):
        pass

    def analyze_amoeba_in_plane_angle_force(force):
        pass

    def analyze_amoeba_multipole_force(force):
        pass

    def analyze_amoeba_out_of_plane_bend_force(force):
        pass

    def analyze_amoeba_pi_torsion_force(force):
        pass

    def analyze_amoeba_stretch_bend_force(force):
        pass

    def analyze_amoeba_torsion_torsion_force(force):
        pass

    def analyze_amoeba_vdw_force(force):
        pass

    def analyze_amoeba_wca_dispersion_force(force):
        pass

    def analyze_andersen_thermostat(force):
        pass

    def analyze_cmap_torsion_force(force):
        pass

    def analyze_cmm_motion_remover(self, force):
        """
        analyze the cmm motion remover

        returns
        -------
        energy_dict : an dictionary containing the sum of harmonic angle forces
                      of each atom
        """

        # there is no energy assiociated with this force
        pass

    def analyze_custom_angle_force(force):
        pass

    def analyze_custom_bond_force(force):
        pass

    def analyze_custom_cv_force(force):
        pass

    def analyze_centroid_bond_force(force):
        pass

    def analyze_custom_compound_bond_force(force):
        pass

    def analyze_custom_external_force(force):
        pass

    def analyze_gb_force(force):
        pass

    def analyze_hbond_force(force):
        pass

    def analyze_custom_many_particle_force(force):
        pass

    def analyze_custom_nonbonded_force(force):
        pass

    def analyze_custom_torsion_force(force):
        pass

    def analyze_drude_force(force):
        pass

    def analyze_gbsaobc_force(force):
        pass

    def analyze_gay_berne_force(force):
        pass

    def analyze_harmonic_angle_force(self, force):
        """
        analyze the harmonic anlge force

        returns
        -------
        energy_dict : an dictionary containing the sum of harmonic angle forces
                      of each atom
        """

        # initialize a dict to put energy in
        energy_dict = dict()
        for idx in self.idxs:
            energy_dict[idx] = Quantity(0.0, kilojoule/mole)

        n_angles = force.getNumAngles()

        for idx in range(n_angles):
            atom0, center_atom, atom1, eq_angle, k = force.getAngleParameters(idx)

            if (center_atom in self.idxs) and (atom0 in self.idxs) and (atom1 in self.idxs):
                angle = self.angle(center_atom, atom0, atom1)
                energy = 0.5 * k * (angle - eq_angle) * (angle - eq_angle)
                # energy = Quantity(energy, kilojoule/mole)
                energy_dict[center_atom] += Quantity(np.true_divide(1, 3)) * energy
                energy_dict[atom0] += Quantity(np.true_divide(1, 3)) * energy
                energy_dict[atom1] += Quantity(np.true_divide(1, 3)) * energy

        return energy_dict


    def analyze_harmonic_bond_force(self, force):
        """
        analyze the harmonic bond force

        returns
        -------
        energy_dict : an dictionary containing the sum of harmonic bond forces
                      of each atom
        """

        # initialize a dict to put energy in
        energy_dict = dict()
        for idx in self.idxs:
            energy_dict[idx] = Quantity(0.0, kilojoule/mole)

        n_bonds = force.getNumBonds()

        for idx in range(n_bonds):
            atom0, atom1, eq_l, k = force.getBondParameters(idx)
            if (atom0 in self.idxs) and (atom1 in self.idxs):
                dist = self.dist(atom0, atom1)
                energy = 0.5 * k * np.power((dist - eq_l), 2)

                # since we consider the bond energy is contributed by two atoms,
                # we divide the enrgy by two

                energy_dict[atom0] += 0.5 * energy
                energy_dict[atom1] += 0.5 * energy

        return energy_dict

    def analyze_monte_carlo_anisotropic_barostat(force):
        pass

    def analyze_monte_carlo_barostat(force):
        pass

    def analyze_monte_carlo_membrane_barostat(force):
        pass


    def analyze_periodic_torsion_force(self, force):
        """
        analyze the periodic torsion force

        returns
        -------
        energy_dict : an dictionary containing the sum of periodic torsion forces
                      of each atom
        """

        energy_dict = dict()
        for idx in self.idxs:
            energy_dict[idx] = Quantity(0.0, kilojoule/mole)

        n_torsions = force.getNumTorsions()

        for idx in range(n_torsions):
            atom0, atom1, atom2, atom3, periodicity, angle_eq, k = force.getTorsionParameters(idx)
            if (atom0 in self.idxs) and (atom1 in self.idxs) and (atom2 in self.idxs) and (atom3 in self.idxs):
                angle = self.dihedral(atom0, atom1, atom2, atom3)
                energy = k * (1 + np.cos(periodicity * angle - angle_eq))

                # put energy into the dictionary
                # note that it is believed that the angle term is shared by four atoms
                energy_dict[atom0] += 0.25 * energy
                energy_dict[atom1] += 0.25 * energy
                energy_dict[atom2] += 0.25 * energy
                energy_dict[atom3] += 0.25 * energy

        return energy_dict

    def analyze_rb_torsion_force(force):
        pass

    def analyze_rpmd_monte_carlo_barostat(force):
        pass
