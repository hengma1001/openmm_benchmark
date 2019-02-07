#!/usr/bin/env python 

# OpenMM imports
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
#from simtk.openmm.app.gromacstopfile import _defaultGromacsIncludeDir

# ParmEd imports
from parmed import load_file

from sys import stdout, argv

GPU_index = 0 
print 'Loading GMX files.'
top = load_file('exab.top', xyz='exab.gro')


# Create system with 1.2 nm LJ and Coul, 1.0 nm LJ switch. 
print 'Creating OpenMM system.'
system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.2*nanometer,
 	switchDistance=1.0*nanometer,
	constraints=HBonds, verbose=0)

# Adding pressure coupling, NPT system
system.addForce(MonteCarloAnisotropicBarostat((1, 1, 1)*bar, 300*kelvin, False, False, True))

# Langevin integrator at 300 K, 1 ps^{-1} collision frequency, time step at 2 fs. 
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds) 
# using CUDA platform, fastest of all 
try:
    platform = Platform_getPlatformByName("CUDA")
    properties = {'DeviceIndex': str(GPU_index), 'CudaPrecision': 'mixed'}
except Exception:
    platform = Platform_getPlatformByName("OpenCL")
    properties = {'DeviceIndex': str(GPU_index)}

simulation = Simulation(top.topology, system, integrator, platform, properties)

# initial position of simulation at t=0
simulation.context.setPositions(top.positions)

# energy minimization relaxing the configuration
print 'Minimizing energy.'
simulation.minimizeEnergy()

# simulation outputs pdb and log every 50000 steps and checkpoint every 50000 steps
simulation.reporters.append(DCDReporter('exab.dcd', 50000))
simulation.reporters.append(StateDataReporter('exab.log', 
	50000, step=True, time=True, speed=True, 
	potentialEnergy=True, temperature=True, totalEnergy=True))
simulation.reporters.append(StateDataReporter(stdout,
        50000, step=True, time=True, speed=True,
        potentialEnergy=True, temperature=True, totalEnergy=True))
simulation.reporters.append(CheckpointReporter('checkpnt.chk', 50000))

# necessary to load checkpoint for continuing an interupted simulation
#simulation.loadCheckpoint('state.chk')

# run the simulation for 500,000,000 steps
print 'Starting the simulation!'
simulation.step(500000000)
