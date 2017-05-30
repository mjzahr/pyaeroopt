from pyaeroopt.interface import Frg, Hpc
from pyaeroopt.test.naca0012 import src_dir

# FRG utility class (default binary locations, $SOWER, $XP2EXO, etc)
frg = Frg(top=src_dir+'top', surf_top=src_dir+'surf.top',
          geom_pre=src_dir+'binary/top')

# HPC utility class
hpc = Hpc(machine='independence', mpi='mpiexec', # use modules on independence
          batch=False, bg=False)                 # so mpiexec is sufficient to
                                                 # use appropriate version of
                                                 # MPI
