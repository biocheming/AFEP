import sys
sys.path.append("..")
import atom_energy_reporter

prmtop = AmberPrmtopFile('../data/toy_solvation/input.prmtop')
inpcrd = AmberInpcrdFile('../data/toy_solvation/input.inpcrd')

system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1*nanometer,
        constraints=HBonds)
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
simulation = Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)

if inpcrd.boxVectors is not None:
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)


simulation.reporters.append(AtomEnergyReporter('test.txt', 1))

top = simulation.topology
simulation.minimizeEnergy(maxIterations = 5)
simulation.step(5)
