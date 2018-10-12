"""
test_afep.py
"""

# imports
## import the reporter
import sys
sys.path.append("..")
import atom_energy_reporter
from atom_energy_reporter import AtomEnergyReporter

## import netCDF4
import netCDF4
from netCDF4 import Dataset

## import openmm
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from openmmtools import testsystems
import mdtraj as md

## import infrastructure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def time_on_toy(n_iter):

    ##################################
    # start a simulation with reporter
    ##################################
    prmtop = AmberPrmtopFile('../data/toy_solvation/input.prmtop')
    inpcrd = AmberInpcrdFile('../data/toy_solvation/input.inpcrd')

    time0 = time.time()
    system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer,
            constraints=HBonds)
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(prmtop.topology, system, integrator)
    simulation.context.setPositions(inpcrd.positions)

    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    simulation.reporters.append(AtomEnergyReporter('test.nc', 10))

    top = simulation.topology
    simulation.minimizeEnergy(maxIterations = 100)
    simulation.step(n_iter)
    time1 = time.time()

    ##################################
    # start a simulation without reporter
    ##################################

    time2 = time.time()
    system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer,
            constraints=HBonds)
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(prmtop.topology, system, integrator)
    simulation.context.setPositions(inpcrd.positions)

    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    # simulation.reporters.append(AtomEnergyReporter('test.nc', 1))

    top = simulation.topology
    simulation.minimizeEnergy(maxIterations = 100)
    simulation.step(n_iter)
    time3 = time.time()


    print("the time with reporter is: %s" % (time1 - time0))
    print("the time without reporter is: %s" % (time3 - time2))

time_on_toy(300)


def plot_atom_energy_over_time(file_path):
    sim_data = Dataset(file_path, 'r')
    energy = sim_data.variables["energy"]
    forces_name = getattr(sim_data, 'forces_name')
    atoms_name = getattr(sim_data, 'atoms_name')
    for atom_idx in range(energy.shape[1]):
        plt.clf()
        atom_name = atoms_name[atom_idx]
        for force_idx in range(energy.shape[2]):
            energy_array = energy[:, atom_idx, force_idx]
            force_name = forces_name[force_idx]
            plt.plot(energy_array, label=force_name)
            plt.title(atom_name)
        plt.legend()
        plt.savefig("%s.png" % atom_idx)
