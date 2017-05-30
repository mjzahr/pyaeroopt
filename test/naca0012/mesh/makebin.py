
from pyaeroopt.test.naca0012.pyao.factory import frg, hpc

def make_binaries():

    # Decomposition
    max_nodes = 5
    ndec   = hpc.ppn * max_nodes
    nclust = hpc.ppn * max_nodes
    cpus   = [1]+[2*(x+1) for x in range(hpc.ppn/2)]+[hpc.ppn*(x+1)
                                                   for x in range(1, max_nodes)]

    # Make binaries
    frg.part_mesh(ndec)               # partnmesh
    frg.sower_fluid_top(cpus, nclust) # Sower top file

if __name__ == '__main__':
    make_binaries()
