#!/usr/bin/python

"""
yank_afep_analyzer.py

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
import openmmtools as mmtools
import mdtraj as md

import netCDF4
from netCDF4 import Dataset
import warnings
import time
import copy

from yank import *




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

    \Delta F_{AB}(a) = \frac{<\frac{\Delta u_k}{\Delta U}(e^(-\beta \Delta U) - 1)>}
                            {<e^(\beta \Delta U) - 1>}

    u_{X_a} = \\

    \frac12(u_{electrostaic} + u_{Lennard-Jones} + u_{bonded} + u_{Urey-Bradley}) \\
    + \frac13 u_{angle} \\
    + \frac14 u_{dihedral} + u_{improper}

    $$

    further data analysis might needed, and is enabled through .nc output

    ref:
    https://pubs.acs.org/doi/abs/10.1021/acs.jctc.8b00027

    """

    def __init__(self, nc_path, nc_checkpoint_path, output_path = 'output.nc'):

        self.nc_path = nc_path
        self.nc_checkpoint_path = nc_checkpoint_path
        self.reference_system = None
        self.puppet_system = None

        # initialize the yank reporter
        reporter = multistate.MultiStateReporter(self.nc_path, open_mode='r', checkpoint_storage = self.nc_checkpoint_path)
        self.reporter = reporter

        assert reporter.storage_exists(), "there is no checkpoint file!"
        print(reporter.n_replicas)
        assert (reporter.n_replicas == 1), "currently we don't support replica exchange analysis"

        # get the reference system
        metadata = self.reporter.read_dict('metadata')
        reference_system = mmtools.utils.deserialize(metadata['reference_state']).system

        # get the topography
        topography = mmtools.utils.deserialize(metadata['topography'])
        topology = topography.topology.to_openmm()
        ligand_atoms = topography.ligand_atoms
        if len(ligand_atoms) == 0:
            ligand_atoms = topography.solute_atoms
        self.ligand_atoms = ligand_atoms


        # create puppet system and puppet simulation
        puppet_system = copy.deepcopy(reference_system)
        integrator = mmtools.integrators.DummyIntegrator()
        simulation = Simulation(topology, puppet_system, integrator)
        self.simulation = simulation

        # get the lambda schedules
        lambda_electrostatics_schedule = []
        lambda_sterics_schedule = []
        beta_schedule = []

        states = self.reporter.read_thermodynamic_states()[0]
        for state in states:
            lambda_electrostatics_schedule.append(state.lambda_electrostatics)
            lambda_sterics_schedule.append(state.lambda_sterics)
            beta_schedule.append(np.power(state.kT, -1))

        self.lambda_electrostatics_schedule = lambda_electrostatics_schedule
        self.lambda_sterics_schedule = lambda_sterics_schedule
        self.beta_schedule = beta_schedule

        thermodynamic_states = self.reporter.read_replica_thermodynamic_states().flatten()
        self._thermodynamic_states = thermodynamic_states
        # read checkpoint schedules
        checkpoint_interval = self.reporter._checkpoint_interval
        n_thermodynamic_states = thermodynamic_states.shape[0]
        checkpoint_schedule = [idx * checkpoint_interval for idx in range(int(n_thermodynamic_states / checkpoint_interval))]
        self._checkpoint_schedule = checkpoint_schedule
        checkpoint_lambda_electrostatics_schedule = [lambda_electrostatics_schedule[thermodynamic_states[idx]] for idx in checkpoint_schedule]
        checkpoint_lambda_sterics_schedule = [lambda_sterics_schedule[thermodynamic_states[idx]] for idx in checkpoint_schedule]
        self._checkpoint_lambda_electrostatics_schedule = checkpoint_lambda_electrostatics_schedule
        self._checkpoint_lambda_sterics_schedule = checkpoint_lambda_sterics_schedule

        # create thermodynamic_states to checkpoints map
        states_checkpoints_mapping = dict()
        for state in self._thermodynamic_states:
            states_checkpoints_mapping[state] = []
        checkpoint_interval = self.reporter._checkpoint_interval
        for idx, state in enumerate(self._thermodynamic_states):
            if idx % checkpoint_interval == 0:
                states_checkpoints_mapping[state].append(idx // checkpoint_interval)

        self.states_checkpoints_mapping = states_checkpoints_mapping

    @property
    def checkpoint_schedule(self):
        return copy.deepcopy(self._checkpoint_schedule)

    @property
    def checkpoint_lambda_electrostatics_schedule(self):
        return copy.deepcopy(self._checkpoint_lambda_electrostatics_schedule)

    @property
    def checkpoint_lambda_sterics_schedule(self):
        return copy.deepcopy(self._checkpoint_lambda_sterics_schedule)

    @property
    def thermodynamic_states(self):
        return copy.deepcopy(self._thermodynamic_states)


    def read_positions(self, checkpoint_idx):
        """
        read the positions in checkpoint so that we could apply them

        Returns
        -------
        positions : positions object
        """

        positions = self.reporter.read_sampler_states(self.checkpoint_schedule[checkpoint_idx], analysis_particles_only=False)[0].positions
        return positions

    #=========================================================================
    # helper functions to analyze forces
    #=========================================================================

    def _analyze_nonbonded_force(self, force):
        # NOTE: we still need a force=None to work with the get_energy function
        # but it's redundant
        """
        analyze the nonbonded force in a simulation, despite of its kind.

        note that this method is different from others because it,
        instead of directly calculate the energy analytically,
        make a copy of the simulation and turn off the params of an atom
        to calculate the total nonbonded energy with the involvement of thereof.

        so the argument force is actually ignored.

        Parameters
        ----------
        force : an OpenMM force object, has to be NonbondedForce

        Returns
        -------
        energy_dict : a dictionary mapping atom idxs to the energies
        """


        energy_dict = dict()
        current_energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy() # get current energy

        # NOTE: currently this loop is VERY slow
        # TODO: try to write this without a loop
        for idx in self.ligand_atoms: # loop through the interested atoms
            charge, sigma, epsilon = force.getParticleParameters(idx)
            force.setParticleParameters(idx, charge = 0.0, sigma = sigma, epsilon = 0.0)
            force.updateParametersInContext(self.simulation.context)
            new_energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()
            energy_diff = new_energy - current_energy
            energy_dict[idx] = energy_diff

            # set the properties back
            force.setParticleParameters(idx, charge = charge, sigma = sigma, epsilon = epsilon)
            force.updateParametersInContext(self.simulation.context) # set the Parameters back to original

        return energy_dict

    def _analyze_harmonic_angle_force(self, force):
        """
        analyze the harmonic anlge force

        Returns
        -------
        energy_dict : an dictionary containing the sum of harmonic angle forces
                      of each atom
        """

        # initialize a dict to put energy in
        energy_dict = dict()
        for idx in self.ligand_atoms:
            energy_dict[idx] = Quantity(0.0, kilojoule/mole)

        n_angles = force.getNumAngles()

        for idx in range(n_angles):
            atom0, center_atom, atom1, eq_angle, k = force.getAngleParameters(idx)

            if (center_atom in self.ligand_atoms) and (atom0 in self.ligand_atoms) and (atom1 in self.ligand_atoms):
                angle = self.angle(center_atom, atom0, atom1)
                energy = 0.5 * k * (angle - eq_angle) * (angle - eq_angle)
                # energy = Quantity(energy, kilojoule/mole)
                energy_dict[center_atom] += Quantity(np.true_divide(1, 3)) * energy
                energy_dict[atom0] += Quantity(np.true_divide(1, 3)) * energy
                energy_dict[atom1] += Quantity(np.true_divide(1, 3)) * energy

        return energy_dict


    def _analyze_harmonic_bond_force(self, force):
        """
        analyze the harmonic bond force

        Returns
        -------
        energy_dict : an dictionary containing the sum of harmonic bond forces
                      of each atom
        """

        # initialize a dict to put energy in
        energy_dict = dict()
        for idx in self.ligand_atoms:
            energy_dict[idx] = Quantity(0.0, kilojoule/mole)

        n_bonds = force.getNumBonds()

        for idx in range(n_bonds):
            atom0, atom1, eq_l, k = force.getBondParameters(idx)
            if (atom0 in self.ligand_atoms) and (atom1 in self.ligand_atoms):
                dist = self.dist(atom0, atom1)
                energy = 0.5 * k * (dist - eq_l) * (dist - eq_l)

                # since we consider the bond energy is contributed by two atoms,
                # we divide the enrgy by two

                energy_dict[atom0] += 0.5 * energy
                energy_dict[atom1] += 0.5 * energy

        return energy_dict

    def _analyze_periodic_torsion_force(self, force):
        """
        analyze the periodic torsion force

        Returns
        -------
        energy_dict : an dictionary containing the sum of periodic torsion forces
                      of each atom
        """

        energy_dict = dict()
        for idx in self.ligand_atoms:
            energy_dict[idx] = Quantity(0.0, kilojoule/mole)

        n_torsions = force.getNumTorsions()

        for idx in range(n_torsions):
            atom0, atom1, atom2, atom3, periodicity, angle_eq, k = force.getTorsionParameters(idx)
            if (atom0 in self.ligand_atoms) and (atom1 in self.ligand_atoms) and (atom2 in self.ligand_atoms) and (atom3 in self.ligand_atoms):
                angle = self.dihedral(atom0, atom1, atom2, atom3)
                energy = k * (1 + np.cos(periodicity * angle - angle_eq))

                # put energy into the dictionary
                # note that it is believed that the angle term is shared by four atoms
                energy_dict[atom0] += 0.25 * energy
                energy_dict[atom1] += 0.25 * energy
                energy_dict[atom2] += 0.25 * energy
                energy_dict[atom3] += 0.25 * energy

        return energy_dict


    #=========================================================================
    # herlper functions to calculate distance, angles,
    # and dihedral angels from positions of atoms

    # NOTE: we use these functions to calculate all the bonded energies
    #=========================================================================

    def dist(self, atom0, atom1):
        """
        calculate the distance between two atoms
        require that self.pos is defined

        Parameters
        ----------
        atom0 : the idx of the first atom
        atom1 : the idx of the second atom

        Returns
        -------
        dist : a float representing the distance between the two atoms
        """

        pos0 = self.pos[atom0]
        pos1 = self.pos[atom1]

        dist = np.linalg.norm(pos0 - pos1)

        return Quantity(dist, nanometer)

    def angle(self, center_atom, atom0, atom1):
        """
        calculate the angle between bond:

        center_atom -- atom0
        and
        center_atom -- atom1

        $ cos(<v0, v1>) = (v0 \dot v1) / |v0||v1| $

        Parameters
        ----------
        center_atom : the idx of the center atom
        atom0 : the idx of the first atom involved in the angle
        atom1 : the idx of the second atom involved in the angle

        Returns
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

        Parameters
        ----------
        atom0 : the idx of the first atom involved in the torsion
        atom1 : the idx of the second atom involved in the torsion
        atom2 : the idx of the thrid atom involved in the torsion
        atom3 : the idx of the fourth atom involved in the torsion

        Returns
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
    # functions to calculate the energies associated with checkpoints
    #=========================================================================

    def _analyze_atom_energies_in_simulation_with_positions(self, positions, as_numpy = True):
        """
        to get the atom energies in a system (usually puppet) with certain position
        note that this method will unapologetically modify the system

        params
        ------
        positions : xyz of each atoms, including that of solvents

        Returns
        -------
            as_numpy == True:
                res_matrix : a matrix with shape (n_atoms, 4), recording the energy contribution of each part

            as_numpy == False:
                res : a dictionary version of the matrix, with keys being Quantity objects
        """

        # set the simulation to match the positions
        self.simulation.context.setPositions(positions)
        self.pos = positions
        self.simulation.step(1) # make one step with dummy integrator to get information on states

        # initialize the result matrix
        n_atoms = len(self.ligand_atoms)
        res = dict()

        # analyze each force
        forces = self.simulation.system.getForces()
        for force in forces:
            name = force.__class__.__name__
            if name == "NonbondedForce":
                res[0] = self._analyze_nonbonded_force(force)
            elif name == "HarmonicBondForce":
                res[1] = self._analyze_harmonic_bond_force(force)
            elif name == "HarmonicAngleForce":
                res[2] = self._analyze_harmonic_angle_force(force)
            elif name == "PeriodicTorsionForce":
                res[3] = self._analyze_periodic_torsion_force(force)

        if as_numpy == False:
            return res

        if as_numpy == True:
            # the returning matrix will be in the shape of (n_atoms * 4)
            # the columns represent, respectively:
                # nonbonded force
                # bond force
                # angle force
                # torsion force
            # the element in i'th row and j'th column of the matrix is
            # the j'th atom's i'th type of force

            # TODO: set values without using for loop
            res_matrix = np.zeros((n_atoms, 4))
            for idx_force in range(4):
                for idx_atom in range(n_atoms):
                    res_matrix[idx_atom, idx_force] = res[idx_force][self.ligand_atoms[idx_atom]]._value
            return res_matrix

    def _analyze_atom_energies_with_all_checkpoint_positions(self):
        """
        analyze the atom emergies with all the checkpoints
        """
        atom_energies_with_all_checkpoint_positions_ = []
        for checkpoint_frame in range(len(self._checkpoint_schedule)):
            ref_positions = self.read_positions(checkpoint_frame)
            atom_energies_at_ref_positions = self._analyze_atom_energies_in_simulation_with_positions(ref_positions) # shape: n_atoms * 4
            atom_energies_with_all_checkpoint_positions_.append(atom_energies_at_ref_positions)
        atom_energies_with_all_checkpoint_positions_ = np.array(atom_energies_with_all_checkpoint_positions_)
        self.atom_energies_with_all_checkpoint_positions = atom_energies_with_all_checkpoint_positions_
        return atom_energies_with_all_checkpoint_positions_

    def _free_energy_weights(self, ref_checkpoint_idx, ptb_checkpoint_idx = None,
                                   ptb_state_idx = None, all_checkpoints = True):
        """
        calculates $\Delta U$ and $\Delta u_a$ this function:

        $$
        \Delta F_{AB}(a) = \frac{<\frac{\Delta u_k}{\Delta U}(e^(-\beta \Delta U) - 1)>}
                                {<e^(\beta \Delta U) - 1>}
        $$


        params
        ------
        reference_checkpoint_idx : the checkpoint idx with the positions information (state A)
        perturbation_state_idx : the perturbation state idx (state B)
        perturbation_checkpoint_idx : the perturbation state checkpoint idx

        Returns
        -------
        delta_u_a : n_atoms * 1 np array
        delta_u : float
        """

        # get the state indicies
        ref_state_idx = self._thermodynamic_states[self._checkpoint_schedule[ref_checkpoint_idx]]
        if ptb_state_idx == None:
            try:
                ptb_state_idx = self._thermodynamic_states[self._checkpoint_schedule[ptb_checkpoint_idx]]
            except:
                raise ValueError('no info was provided with perturbation state')

        # get the positions
        if all_checkpoints == True:
            atom_energies_at_ref_positions = self.atom_energies_with_all_checkpoint_positions[ref_checkpoint_idx]

        else:
            ref_positions = self.analyzer.read_positions(self._checkpoint_schedule[ref_checkpoint_idx])
            atom_energies_at_ref_positions = self._analyze_atom_energies_in_simulation_with_positions(ref_positions) # shape: n_atoms * 4

        # get the configuration at reference state
        nonbonded_atom_energies_at_ref_positions = atom_energies_at_ref_positions[:, 0]

        # get the total energies at both state
        energies_ref_positions = self.reporter.read_energies(self._checkpoint_schedule[ref_checkpoint_idx])[0].flatten()
        energy_ref_position_ref_state = energies_ref_positions[ref_state_idx]
        energy_ref_position_ptb_state = energies_ref_positions[ptb_state_idx]

        # get the lambda values
        ref_lambda_electrostatics = self.lambda_electrostatics_schedule[ref_state_idx]
        ptb_lambda_electrostatics = self.lambda_electrostatics_schedule[ptb_state_idx]
        diff_lambda_electrostatics = ref_lambda_electrostatics - ptb_lambda_electrostatics

        # compute the atomic free energy contribution weights
        # TODO: it is not a good idea to assume the linearity here
        delta_u_a = diff_lambda_electrostatics * nonbonded_atom_energies_at_ref_positions
        delta_u = energy_ref_position_ref_state - energy_ref_position_ptb_state

        return delta_u_a, delta_u


    def _average_free_energy_weights_by_state(self, ref_state_idx, ptb_state_idx):
        """
        averaging the free energy weights by states, implementing

        $$
        \Delta F_{AB}(a) = \frac{<\frac{\Delta u_k}{\Delta U}(e^(-\beta \Delta U) - 1)>}
                                {<e^(\beta \Delta U) - 1>}
        $$

        params
        ------
        reference_state_idx
        perturbation_state_idx

        Returns
        -------
        weights : the free energy weights for each atom
        """

        # initialize numerator and demoninator in the averaging eq
        numerators = []
        denominators = []

        # $\beta = (kT) ^ (-1)$
        beta = self.beta_schedule[ref_state_idx]

        ref_checkpoint_idxs = self.states_checkpoints_mapping[ref_state_idx]
        for ref_checkpoint_idx in ref_checkpoint_idxs:
            delta_u_a, delta_u = self._free_energy_weights(ref_checkpoint_idx = ref_checkpoint_idx,
                        ptb_state_idx = ptb_state_idx,
                        all_checkpoints = True)
            temp_weight = np.exp(-beta * delta_u) - 1
            numerator = np.true_divide(temp_weight, delta_u) * delta_u_a
            denominator = temp_weight

            numerators.append(numerator)
            denominators.append(denominator)

        numerators = np.array(numerators)
        denominators = np.array(denominators)

        avg_numerator = np.average(numerators, axis = 0)
        avg_denominator = np.average(denominators)

        weights = np.true_divide(avg_numerator, avg_denominator)
        return(weights)
