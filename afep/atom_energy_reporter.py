#!/usr/bin/python

"""
atom_energy_reporter.py

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

# NOTE:
    # - currently only the most common energy were implemented

# TODO:
    # - implement AMOEBA forces

class AtomEnergyReporter(object):
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

    def __init__(self, file_path, reportInterval, idxs = None):
        """
        create a AtomEnergyReporter

        parameters
        ----------
        file : a string
            the file to write to
        reportInterval : int
            the interval at which to write
        """

        self._reportInterval = reportInterval
        self.idxs = idxs

        self.force_map = {
            'AmoebaAngleForce' : self.analyze_amoeba_angle_force,
            'AmoebaBondForce' : self.analyze_amoeba_bond_force,
            'AmoebaGeneralizedKirkwoodForce' : self.analyze_amoeba_generalized_kirkwood_force,
            'AmoebaInPlaneAngleForce' : self.analyze_amoeba_in_plane_angle_force,
            'AmoebaMultipoleForce' : self.analyze_amoeba_multipole_force,
            'AmoebaOutOfPlaneBendForce' : self.analyze_amoeba_out_of_plane_bend_force,
            'AmoebaPiTorsionForce' : self.analyze_amoeba_pi_torsion_force,
            'AmoebaStretchBendForce' : self.analyze_amoeba_stretch_bend_force,
            'AmoebaTorsionTorsionForce' : self.analyze_amoeba_torsion_torsion_force,
            'AmoebaVdwForce' : self.analyze_amoeba_vdw_force,
            'AmoebaWcaDispersionForce' : self.analyze_amoeba_wca_dispersion_force,
            'AndersenThermostat' : self.analyze_andersen_thermostat,
            'CMAPTorsionForce' : self.analyze_cmap_torsion_force,
            'CMMotionRemover' : self.analyze_cmm_motion_remover,
            'CustomAngleForce' : self.analyze_custom_angle_force,
            'CustomBondForce' : self.analyze_custom_bond_force,
            'CustomCVForce' : self.analyze_custom_cv_force,
            'CustomCentroidBondForce' : self.analyze_centroid_bond_force,
            'CustomCompoundBondForce' : self.analyze_custom_compound_bond_force,
            'CustomExternalForce' : self.analyze_custom_external_force,
            'CustomGBForce' : self.analyze_gb_force,
            'CustomHbondForce' : self.analyze_hbond_force,
            'CustomManyParticleForce' : self.analyze_custom_many_particle_force,
            'CustomNonbondedForce' : self.analyze_custom_nonbonded_force,
            'CustomTorsionForce' : self.analyze_custom_torsion_force,
            'DrudeForce' : self.analyze_drude_force,
            'GBSAOBCForce' : self.analyze_gbsaobc_force,
            'GayBerneForce' : self.analyze_gay_berne_force,
            'HarmonicAngleForce' : self.analyze_harmonic_angle_force,
            'HarmonicBondForce' : self.analyze_harmonic_bond_force,
            'MonteCarloAnisotropicBarostat' : self.analyze_monte_carlo_anisotropic_barostat,
            'MonteCarloBarostat' : self.analyze_monte_carlo_barostat,
            'MonteCarloMembraneBarostat' : self.analyze_monte_carlo_membrane_barostat,
            'NonbondedForce' : self.analyze_nonbonded_force,
            'PeriodicTorsionForce' : self.analyze_periodic_torsion_force,
            'RBTorsionForce' : self.analyze_rb_torsion_force,
            'RPMDMonteCarloBarostat' : self.analyze_rpmd_monte_carlo_barostat
            }

        # create a netCDF4 Dataset to record the energy
        self._out = Dataset(file_path ,'w')

        self._out.createDimension("time", None)
        times = self._out.createVariable("time", "i8", ("time",))
        times.unit = str(self._reportInterval)
        self.time = 0
        # let the analyzer register for once
        self.registered = False

    def describeNextReport(self, simulation):
        """
        adopted from:

        openmm/wrappers/python/simtk/openmm/app/statedatareporter.py

        Get information about the next report this object will generate.
        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        Returns
        -------
        tuple
            A five element tuple. The first element is the number of steps
            until the next report. The remaining elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.
        """

        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, True, True)

    def report(self, simulation, state):
        """
        generate a report

        parameters
        ----------
        simulation : an OpenMM simulation object
        state : an OpenMM state object
        """

        # find the small molecule to analyze
        if self.registered == False: # if the system is not registered, register the system
            if self.idxs == None:
                self.find_small_mol(simulation, state)

            # set the attributes in Dataset
            self._out.description = 'record of an OpenMM run'
            self._out.history = 'created ' + time.ctime(time. time())

            # initialize the Dataset
            self._out.createDimension("atom", len(self.idxs))
            self._out.createVariable("atom", "i8", ("atom", ))
            atoms_name = ["idx = %s; mass = %s" % (idx, simulation.system.getParticleMass(idx)) for idx in self.idxs]
            self._out.setncattr('atoms_name', atoms_name)

            # get the forces
            self.forces = simulation.system.getForces()
            self.force_idx_mapping = [force for force in self.forces]
            forces_name = [force.__class__.__name__ for force in self.forces]
            self._out.setncattr('forces_name', forces_name)

            # create a force dimension, using idxs
            # and initialize the forces
            self._out.createDimension("force", len(self.forces))
            self._out.createVariable("force", "i8", ("force", ))

            # initialize the energy variable
            # that stands on the dimensions of: time, atom, and force
            self.energy_var = self._out.createVariable("energy", "f4", ("time", "atom", "force"))
            self.energy_var.units = 'kJ/mol'

            # keep a copy of all the positions
            self._out.createDimension("xyz", 3)
            self.pos_var = self._out.createVariable("pos", "f4", ("time", "atom", "xyz"))

            # keep a copy of the parameters of atoms
            param_array = np.zeros((len(self.idxs), 3))
            for force in self.forces:
                if force.__class__.__name__ == "NonbondedForce":
                    for idx in self.idxs:
                        charge, sigma, epsilon = force.getParticleParameters(idx)
                        param_array[idx, 0], param_array[idx, 1], param_array[idx, 2] = charge._value, sigma._value, epsilon._value
                        # note that the units here are: elementary charge, nanometer, kilojoule/mole

            self._out.setncattr('param_array', param_array)

            # set the registered flag to True,
            # since you only need to do this once
            self.registered = True


        # point these objects to the class, and update them
        self.simulation = simulation
        self.state = state

        # get the positions of the small molecules
        self.pos = tuple([state.getPositions()[idx] for idx in self.idxs])
        pos_matrix = np.array([state.getPositions(asNumpy=True)[idx]._value for idx in self.idxs])
        self.pos_var[self.time, :, :] = pos_matrix

        # analyze each force in the system
        for force_idx, force in enumerate(self.force_idx_mapping):
            energy_dict = self.get_energy(force)

            if energy_dict == None:
                warnings.warn("no force information could be extracted from %s" % force.__class__.__name__)
                continue

            for atom_idx, energy in energy_dict.items():
                self.energy_var[self.time, atom_idx, force_idx] = energy._value
                # note that the unit here is kilojoule/mole

        # increase the time dimension by one
        self.time += 1

    def find_small_mol(self, simulation, state):
        """
        find the atoms of the smallest molecule, which is most likely to be
        the region of greates interest for a simulation

        parameters
        ----------
        simulation : an OpenMM Simulation object
        state : an OpenMM State object

        returns
        -------
        atoms : a tuple of indicies of atoms that belongs to the small molecule
        """

        context = simulation.context
        mols = context.getMolecules()
        small_mol = sorted([mol for mol in mols if len(mol) > 4],
            key = lambda mol : len(mol), reverse = False)[0]

        # register the atoms and idxs in the class
        self.idxs = small_mol
        return small_mol

    def get_energy(self, force):
        """
        anlyzes force and return the energy,
        to be more specific, match the force with a certain type of analysis function
        """

        name = str(force.__class__.__name__) # get the name of the force
        energy_dict = self.force_map[name](force) # map the force to its specific analyze function and get the energy

        return energy_dict

    #################################################
    # herlper functions to calculate distance, angles,
    # and dihedral angels from positions of atoms
    #################################################

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

    #################################################
    # force to energy functions
    #################################################

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
