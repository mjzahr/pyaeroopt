import sys, os

# PYAEROOPT
from pyaeroopt.interface.aerof import AerofInputBlock

# Problem Instance
from pyaeroopt.test.naca0012.pyao import frg

## Problem
prob=AerofInputBlock('Problem',
               ['Type', None],
               ['Mode', 'Dimensional'])

## Input
inpu=AerofInputBlock('Input',
               ['Prefix', ""],
               ['GeometryPrefix', frg.geom_pre],
               ['InitialWallDisplacement', None],
               ['ShapeDerivative', None],
               ['MultipleSolutions',None],
               ['OptimalPressure', None],
               #['OptimalPressureDimensionality', 'Dimensional'],
               ['StateSnapshotData', None],
               ['SensitivitySnapshotData', None],
               ['StateSnapshotReferenceSolution', None])

## Output
post=AerofInputBlock('Postpro',
               ['Prefix', ''],
               ['LiftandDrag', None],
               ['Force', None],
               ['MatchPressure', None],
               ['LiftandDragSensitivity', None],
               ['MatchPressureSensitivity', None],
               ['PressureSensitivity', None],
               ['StateVectorSensitivity', None],
               ['FluxPartialSensitivity', None],
               ['Pressure', None],
               ['Displacement', None],
               ['Frequency', 0])

nonlRom=AerofInputBlock('NonlinearROM',
               ['Prefix', ''],
               ['StateVector', None],
               ['StateVectorOutputFrequencyTime', 200],
               ['StateVectorOutputFrequencyNewton', 0],
               ['FluxNormSensitivity', None])

rest=AerofInputBlock('Restart',
               ['Prefix', ''],
               ['Solution', None],
               ['RestartData', None],
               ['Frequency', 0])

outp=AerofInputBlock('Output',
               ['Postpro', post],
               ['NonlinearROM', nonlRom],
               ['Restart', rest])

## NonlinearRomFileSystem
nonlRomDire=AerofInputBlock('Directories',
               ['Prefix', ''],
               ['TopLevelDirectoryName', None],
               ['ClusterDirectoryName', None],
               ['SensitivityClusterDirectoryName', None])

nonlRomFile=AerofInputBlock('Files',
               ['StatePrefix', None],
               ['StateBasisPrefix', None],
               ['SensitivityPrefix', None],
               ['SensitivityBasisPrefix', None],
               ['ResidualPrefix', None],
               ['GNATPrefix', None])

nonlRomFileSyst=AerofInputBlock('NonlinearRomFileSystem',
               ['NumClusters', 1],
               ['Directories', nonlRomDire],
               ['Files', nonlRomFile])

## NonlinearRomOffline
# Clustering
clus=AerofInputBlock('Clustering',
               ['ClusteringAlgorithm', 'KMeansWithBounds'],
               ['PercentOverlap', 1.0],
               ['KMeansMaxIterations', 1],
               ['KMeansMaxAggressiveIterations', 1],
               ['KMeansRandomSeed',10101],
               ['MinClusterSize', 1],
               ['UseExistingClusters', False])

# State snapshots
stSnap=AerofInputBlock('Snapshots',
               ['NormalizeSnaps', None],
               ['IncrementalSnaps', False],
               ['SubtractNearestSnapToCenter', False],
               ['SubtractClusterCenters', None],
               ['SubtractReferenceState', None])

stDataComp=AerofInputBlock('DataCompression',
               ['ComputePOD', True],
               ['PODMethod', 'ScalapackSVD'],
               ['SingularValueTolerance', 1e-16],
               ['MaxBasisSize', 50000],
               ['MinBasisSize', 10],
               ['MaxEnergyRetained', 1.0])

stRob=AerofInputBlock('StateROB',
               ['Snapshots', stSnap],
               ['DataCompression', stDataComp])


# Sensitivity snapshots
sensSnap=AerofInputBlock('Snapshots',
               ['NormalizeSnaps', False])

sensDataComp=AerofInputBlock('DataCompression',
               ['ComputePOD', True],
               ['PODMethod', 'ScalapackSVD'],
               ['SingularValueTolerance', 1e-16],
               ['MaxBasisSize', 50000],
               ['MinBasisSize', 10],
               ['MaxEnergyRetained', 1.0])

sensRob=AerofInputBlock('SensitivityROB',
               ['Snapshots', sensSnap],
               ['DataCompression', sensDataComp])


# Residual snapshots
resSnap=AerofInputBlock('Snapshots',
               ['NormalizeSnaps', None])

resDataComp=AerofInputBlock('DataCompression',
               ['ComputePOD', True],
               ['PODMethod', 'ScalapackSVD'],
               ['SingularValueTolerance', 1e-16],
               ['MaxBasisSize', 10000],
               ['MinBasisSize', 10],
               ['MaxEnergyRetained', 1.0])

resRob=AerofInputBlock('ResidualROB',
               ['Snapshots', resSnap],
               ['DataCompression', resDataComp])

onliBasiUpda=AerofInputBlock('OnlineBasisUpdates',
               ['PreprocessForNoUpdates', 'On'],
               ['PreprocessForFastExactUpdates', 'On'])

# Construct ROB
consROB=AerofInputBlock('ConstructROB',
               ['Clustering', clus],
               ['StateROB', stRob],
               ['SensitivityROB', sensRob],
               ['ResidualROB', resRob],
               ['OnlineBasisUpdates', onliBasiUpda])

# Construct GNAT
consGNAT=AerofInputBlock('ConstructGNAT',
               ['UseUnionOfSampledNodes', None],
               ['MaxDimensionStateROB', None],
               ['MinDimensionStateROB', None],
               ['EnergyStateROB', None],
               ['MaxDimensionResidualROB', None],
               ['MinDimensionResidualROB', None],
               ['EnergyResidualROB', None],
               ['ROBGreedy', 'Residual'],
               ['MaxDimensionROBGreedy', None],
               ['MinDimensionROBGreedy', None],
               ['ROBGreedyFactor', None],
               ['MaxSampledNodes', None],
               ['MinSampledNodes', None],
               ['SampledNodesFactor',None],
               ['NumPseudoInvNodesAtATime', 1000000],
               ['OutputReducedBases', 'True'])

nonlRomOffl=AerofInputBlock('NonlinearRomOffline',
               ['ConstructROB', consROB],
               ['ConstructGNAT', consGNAT])

## NonlinearRomOnline
sensOnli=AerofInputBlock('Sensitivities',
               ['Include', 'On'],
               ['GramSchmidt', 'On'],
               ['MaximumDimension', 50000],
               ['MinimumDimension', 1],
               ['MaximumEnergy', None])

# NonlinearRomOnline
nonlRomOnli=AerofInputBlock('NonlinearRomOnline',
               ['Projection', 'PetrovGalerkin'],
               ['PerformLineSearch', None], # 'Backtracking'],
               ['MinimumDimension', 5],
               ['MaximumDimension', 50000],
               ['MaximumEnergy', None],
               ['FastDistanceComparisons', 'Off'],
               ['BasisUpdates', None],
               ['ProjectSwitchStateOntoAffineSubspace', None],
               ['StoreAllClustersInMemory', False],
               ['Sensitivities', sensOnli])

## SensitivityAnalysis
precSa=AerofInputBlock('Preconditioner',
               ['Type', 'Ras'],
               ['Fill', 0])

lineSolvSa=AerofInputBlock('LinearSolver',
               ['Type', 'Gmres'],
               ['MaxIts', 2000],
               ['KrylovVectors', 2000],
               ['Eps', 1e-10],
               ['Output', '"stdout"'],
               ['Preconditioner', precSa])

sensAnal=AerofInputBlock('SensitivityAnalysis',
               ['SensitivityComputation', 'Analytical'],
               ['MatrixVectorProduct', 'Exact'],
               ['SensitivityMesh', 'On'],
               ['LinearSolver', lineSolvSa])

## MeshMotion
precMm=AerofInputBlock('Preconditioner',
               ['Type', 'Jacobi'])

lineSolvMm=AerofInputBlock('LinearSolver',
               ['Type', 'Cg'],
               ['MaxIts', 5000],
               ['KrylovVectors', 100],
               ['Eps', 1e-12],
               ['Preconditioner', precMm])

newtMm=AerofInputBlock('Newton',
               ['MaxIts', 1],
               ['Eps', 0.01],
               ['LinearSolver', lineSolvMm])

meshMoti=AerofInputBlock('MeshMotion',
               ['Type', 'Basic'],
               ['Element', 'BallVertexSprings'],
               ['Mode', 'NonRecursive'],
               ['NumIncrements', 1],
               ['Newton', newtMm])

## Surfaces
surfData=AerofInputBlock('SurfaceData[2]',
               ['Nx', 0.0],
               ['Ny', 0.0],
               ['Nz', 1.0])

surf=AerofInputBlock('Surfaces',
               ['SurfaceData[2]', surfData])

## Equations
equa=AerofInputBlock('Equations',
               ['Type', 'Euler'])

## BoundaryConditions
inle=AerofInputBlock('Inlet',
               ['Mach', 0.5],
               ['Alpha', 0.0],
               ['Beta', 0.0],
               ['Pressure', 30397.5],
               ['Density', 0.45])

bounCond=AerofInputBlock('BoundaryConditions',
               ['Inlet', inle])

## Space
naviStokSp=AerofInputBlock('NavierStokes',
               ['Flux', 'Roe'],
               ['Reconstruction', 'Linear'],
               ['AdvectiveOperator', 'FiniteVolume'],
               ['Limiter', 'VanAlbada'],
               ['Gradient', 'LeastSquares'],
               ['Dissipation', 'SecondOrder'],
               ['Beta', 0.333333333333],
               ['Gamma', 1.0])

spac=AerofInputBlock('Space',
               ['NavierStokes', naviStokSp])

## Time
precTi=AerofInputBlock('Preconditioner',
               ['Type', 'Ras'],
               ['Fill', 0])

naviStokTi=AerofInputBlock('NavierStokes',
               ['Type', 'Gmres'],
               ['EpsFormula', 'Eisenstadt'],
               ['MaxIts', 80],
               ['KrylovVectors', 80],
               ['Eps', 0.001],
               ['Preconditioner', precTi])

lineSolvTi=AerofInputBlock('LinearSolver',
               ['NavierStokes', naviStokTi])

newtTi=AerofInputBlock('Newton',
               ['MaxIts', 1],
               ['Eps', 1e-05],
               ['EpsAbs', None],
               ['LinearSolver', lineSolvTi])

impl=AerofInputBlock('Implicit',
               ['Type', None],
               ['MatrixVectorProduct', 'Approximate'],
               ['Newton', newtTi])

time=AerofInputBlock('Time',
               ['Type', 'Implicit'],
               ['MaxIts', 10000],
               ['Eps', 1e-16],
               ['EpsAbs', 1e-06],
               ['Cfl0', 1.0],
               ['Cfl1', 1.0],
               ['Cfl2', 5.0],
               ['CflMax', 10000.0],
               ['Ser', 0.7],
               ['CheckSolution', 'Off'],
               ['CheckVelocity', 'Off'],
               ['CheckPressure', 'Off'],
               ['CheckDensity', 'Off'],
               ['Implicit', impl])
